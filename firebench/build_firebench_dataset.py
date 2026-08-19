import argparse
import itertools
import json
import os
import time
import warnings

import gcsfs
import h5py
import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt


# ============================================================
# CONFIGURATION
# ============================================================

GCS_ROOT = "gs://firebench/v2024.04"

OUTPUT_ROOT = (
    "/media/phantom/Volume/"
    "FireBench Data/windseer_6m_dataset"
)

WINDS = [
    "u6",
    "u8",
    "u10",
    "u12",
    "u14",
    "u16",
    "u18",
    "u20",
    "u22",
]

SLOPES = [
    "ramp0",
    "ramp2.5",
    "ramp5",
    "ramp7.5",
    "ramp10",
    "ramp12.5",
    "ramp15",
    "ramp17.5",
    "ramp20",
    "ramp22.5",
    "ramp25",
    "ramp27.5",
    "ramp30",
]

# We scout more densely early in the simulation because the
# high-wind / steep-slope fires can advance extremely quickly.
#
# These are only candidate times. We DO NOT simply take these
# same five times for every simulation.
SCOUT_TIMES_S = [
    260,
    275,
    290,
    305,
    320,
    335,
    350,
    375,
    400,
    450,
    500,
    550,
    600,
    650,
    725,
    800,
]

# We eventually select five stages representing approximately
# 10%, 30%, 50%, 70%, and 90% of the observed fire-front
# progression within each simulation.
STAGE_FRACTIONS = [
    0.10,
    0.30,
    0.50,
    0.70,
    0.90,
]

T_AMBIENT_K = 290.0

# Used ONLY to distinguish an active fire while selecting
# progression stages.
#
# This does NOT threshold or alter the fire channel supplied
# to model C.
ACTIVE_FIRE_EXCESS_K = 20.0

# If the detected front barely moves, progression quantiles
# are not meaningful. We then fall back to spreading five
# samples over the available active times.
MIN_FRONT_SPAN_M = 50.0

# Native crop:
#
# x: 192 × 1 m
# y: 192 × 1 m
# z: 384 × 0.5 m
#
# Physical dimensions = 192 m × 192 m × 192 m.
NX = 192
NY = 192
NZ = 384

# Final model grid.
N = 32

# Native samples grouped into each 6 m cell.
BX = 6
BY = 6
BZ = 12

# Keep approximately one coarse terrain layer below the
# lowest ground surface while maximizing atmospheric space.
BELOW_LOWEST_TERRAIN_M = 3.0

# Warn if the steepest uphill part leaves very little
# atmosphere in the cube.
MIN_HEADROOM_M = 48.0


warnings.filterwarnings(
    "ignore",
    message="Object at .*\\$folder\\$ is not recognized"
)


# ============================================================
# BASIC HELPERS
# ============================================================

def slope_degrees(slope_name):
    """
    Convert a name such as 'ramp22.5' into 22.5 degrees.
    """
    return float(
        slope_name.replace(
            "ramp",
            ""
        )
    )


def time_in_seconds(t):
    """
    Convert FireBench's stored integer millisecond times
    into physical seconds.
    """
    units = t.attrs.get(
        "units",
        ""
    )

    if units != "milliseconds":
        raise RuntimeError(
            "Unexpected time units: "
            f"{units!r}"
        )

    return (
        t.values.astype(np.float64)
        / 1000.0
    )


def split_for_case(wind_index, slope_index):
    """
    Split entire SIMULATIONS, never individual snapshots.

    This deterministic pattern distributes every wind regime
    and every slope across train/validation/test while keeping
    each wind+slope simulation wholly inside one split.

    Across 117 simulations this gives:

        train: 84 simulations
        val:   16 simulations
        test:  17 simulations
    """
    code = (
        3 * wind_index
        +
        2 * slope_index
    ) % 7

    if code == 0:
        return "test"

    if code == 1:
        return "val"

    return "train"


def case_name(wind, slope):
    return f"{wind}_{slope}"


def case_url(wind, slope):
    return (
        f"{GCS_ROOT}/"
        f"{wind}/"
        f"{slope}/"
        "fire.zarr"
    )


# ============================================================
# FIRE-FRONT SCOUT
# ============================================================

def scout_front(ds, requested_time_s):
    """
    Estimate the mean x-location of the leading fire front.

    The inexpensive T_s_y_sum field is used instead of
    downloading the complete 3-D solid-temperature field.

    Returns the detected front, thermal activity level,
    actual FireBench time, and array index.
    """
    start = time.perf_counter()

    times_s = time_in_seconds(
        ds["t"]
    )

    it = int(
        np.argmin(
            np.abs(
                times_s
                - requested_time_s
            )
        )
    )

    actual_time_s = float(
        times_s[it]
    )

    scout = (
        ds["T_s_y_sum"]
        .isel(t=it)
        .transpose(
            "x",
            "z"
        )
        .load()
        .values
    )

    # T_s_y_sum is summed across y.
    # Convert it to an approximate y-mean.
    scout_mean = (
        scout
        / ds.sizes["y"]
    )

    # For every x coordinate, retain the hottest vertical
    # position in this y-averaged thermal field.
    heat_x = scout_mean.max(
        axis=1
    )

    heat_excess_x = np.maximum(
        heat_x - T_AMBIENT_K,
        0.0
    )

    peak_excess = float(
        heat_excess_x.max()
    )

    # Smooth the x-profile so one isolated hot point cannot
    # determine the crop.
    kernel = (
        np.ones(
            11,
            dtype=np.float64
        )
        / 11.0
    )

    heat_smooth = np.convolve(
        heat_excess_x,
        kernel,
        mode="same"
    )

    x_all = (
        ds["x"]
        .values
    )

    gradient = np.gradient(
        heat_smooth,
        x_all
    )

    # Ignore far-domain boundaries.
    valid = (
        (x_all > 150.0)
        &
        (x_all < 1300.0)
    )

    valid_indices = np.where(
        valid
    )[0]

    # Moving in +x, the leading edge is approximately the
    # strongest hot -> cool transition.
    front_index = int(
        valid_indices[
            np.argmin(
                gradient[valid]
            )
        ]
    )

    front_x = float(
        x_all[
            front_index
        ]
    )

    front_strength = float(
        -gradient[
            front_index
        ]
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    return {
        "requested_time_s":
            float(requested_time_s),

        "time_s":
            actual_time_s,

        "time_index":
            it,

        "front_index":
            front_index,

        "front_x_m":
            front_x,

        "peak_excess_K":
            peak_excess,

        "front_strength":
            front_strength,

        "scout_seconds":
            elapsed,
    }


def select_progression_stages(
    ds,
    wind,
    slope,
    scout_cache_path,
):
    """
    Scout several inexpensive time slices and choose five
    distinct times that span the fire's observed progression.

    Results are cached to JSON so an interrupted overnight
    run does not have to scout the simulation again.
    """
    if os.path.exists(
        scout_cache_path
    ):
        with open(
            scout_cache_path,
            "r"
        ) as f:
            cached = json.load(f)

        return cached[
            "selected_stages"
        ]

    print(
        "  scouting progression..."
    )

    scouts = []

    times_s = time_in_seconds(ds["t"])

    available_min_s = float(times_s[0])
    available_max_s = float(times_s[-1])

    valid_scout_times = [
        requested_time
        for requested_time in SCOUT_TIMES_S
        if (
            available_min_s
            <= requested_time
            <= available_max_s
        )
    ]

    for requested_time in valid_scout_times:

        result = scout_front(
            ds,
            requested_time
        )

        scouts.append(
            result
        )

        print(
            "    "
            f"t={result['time_s']:.3f}s  "
            f"front={result['front_x_m']:.1f}m  "
            f"peak excess={result['peak_excess_K']:.1f}K  "
            f"({result['scout_seconds']:.1f}s)"
        )

    # Use clearly active-fire candidate times.
    active = [
        result
        for result in scouts
        if result[
            "peak_excess_K"
        ] >= ACTIVE_FIRE_EXCESS_K
    ]

    # If the conservative activity detector leaves fewer than
    # five times, take the five thermally strongest candidates.
    #
    # This is only a stage-selection fallback; it does not
    # modify the actual thermal input.
    if len(active) < 5:

        print(
            "  WARNING: fewer than five "
            "active-fire anchors; using "
            "thermally strongest anchors."
        )

        active = sorted(
            scouts,
            key=lambda r:
                r["peak_excess_K"],
            reverse=True
        )[:5]

        active = sorted(
            active,
            key=lambda r:
                r["time_s"]
        )

    # Fire spread is predominantly toward +x.
    #
    # Small detector fluctuations can make the measured front
    # move backward slightly, so construct a monotonic
    # progression coordinate for stage selection.
    raw_fronts = np.array(
        [
            r["front_x_m"]
            for r in active
        ],
        dtype=np.float64
    )

    # The FireBench fire propagates mainly in +x.
    #
    # Once our detector reaches its furthest-downstream front,
    # later large backward jumps usually mean the actual leading
    # edge has left the useful detection region and the detector
    # has latched onto another hot/cold boundary.
    furthest_index = int(
        np.argmax(
            raw_fronts
        )
    )

    if furthest_index >= 4:
        active = active[
            :furthest_index + 1
        ]

        raw_fronts = raw_fronts[
            :furthest_index + 1
        ]

    progression_fronts = (
        np.maximum.accumulate(
            raw_fronts
        )
    )

    progression_fronts = (
        np.maximum.accumulate(
            raw_fronts
        )
    )

    front_min = float(
        progression_fronts[0]
    )

    front_max = float(
        progression_fronts[-1]
    )

    front_span = (
        front_max
        - front_min
    )

    selected_indices = None

    if (
        len(active) >= 5
        and
        front_span >= MIN_FRONT_SPAN_M
    ):
        targets = np.array(
            [
                front_min
                +
                fraction
                * front_span

                for fraction
                in STAGE_FRACTIONS
            ],
            dtype=np.float64
        )

        # There are only ~16 anchors, so we can cheaply test
        # every ordered combination of five distinct anchors
        # and select the combination closest to the desired
        # progression fractions.
        best_score = None
        best_combo = None

        for combo in itertools.combinations(
            range(
                len(active)
            ),
            5
        ):
            chosen_fronts = np.array(
                [
                    progression_fronts[
                        index
                    ]
                    for index
                    in combo
                ]
            )

            score = float(
                np.mean(
                    (
                        chosen_fronts
                        - targets
                    ) ** 2
                )
            )

            if (
                best_score is None
                or
                score < best_score
            ):
                best_score = score
                best_combo = combo

        selected_indices = list(
            best_combo
        )

    else:
        print(
            "  WARNING: observed front span "
            f"is only {front_span:.1f} m; "
            "falling back to evenly spaced "
            "active times."
        )

        # Pick five ordered positions through the active
        # candidate list.
        selected_indices = np.linspace(
            0,
            len(active) - 1,
            5
        ).round().astype(int).tolist()

        # Guarantee distinct indices if rounding duplicated one.
        selected_indices = sorted(
            set(
                selected_indices
            )
        )

        if len(selected_indices) < 5:
            for index in range(
                len(active)
            ):
                if index not in selected_indices:
                    selected_indices.append(
                        index
                    )

                if len(
                    selected_indices
                ) == 5:
                    break

            selected_indices = sorted(
                selected_indices
            )

    selected = []

    for stage_number, active_index in enumerate(
        selected_indices
    ):
        source = dict(
            active[
                active_index
            ]
        )

        source[
            "stage_number"
        ] = stage_number + 1

        source[
            "stage_fraction"
        ] = STAGE_FRACTIONS[
            stage_number
        ]

        source[
            "progression_front_x_m"
        ] = float(
            progression_fronts[
                active_index
            ]
        )

        selected.append(
            source
        )

    print(
        "  selected stages:"
    )

    for stage in selected:
        print(
            "    "
            f"{int(stage['stage_fraction'] * 100):02d}%: "
            f"t={stage['time_s']:.3f}s  "
            f"front={stage['front_x_m']:.1f}m"
        )

    cache = {
        "wind":
            wind,

        "slope":
            slope,

        "all_scouts":
            scouts,

        "selected_stages":
            selected,
    }

    temp_cache = (
        scout_cache_path
        + ".tmp"
    )

    with open(
        temp_cache,
        "w"
    ) as f:
        json.dump(
            cache,
            f,
            indent=2
        )

    os.replace(
        temp_cache,
        scout_cache_path
    )

    return selected


# ============================================================
# AUTOMATIC CROP
# ============================================================

def choose_crop(
    ds,
    front_index,
    slope_deg,
):
    """
    Center a 192 m x-crop on the detected fire front,
    center the y-crop, and move the vertical crop with terrain.
    """
    x_start = (
        front_index
        - NX // 2
    )

    # Align to exact 6 m downsampling blocks.
    x_start = (
        x_start // BX
    ) * BX

    x_start = max(
        0,
        min(
            x_start,
            ds.sizes["x"] - NX
        )
    )

    y_start = (
        ds.sizes["y"]
        - NY
    ) // 2

    y_start = (
        y_start // BY
    ) * BY

    x_native = (
        ds["x"]
        .values[
            x_start:
            x_start + NX
        ]
    )

    y_native = (
        ds["y"]
        .values[
            y_start:
            y_start + NY
        ]
    )

    angle = np.deg2rad(
        slope_deg
    )

    ground_native = np.maximum(
        (
            x_native
            - 100.0
        )
        * np.tan(
            angle
        ),
        0.0
    )

    ground_min = float(
        ground_native.min()
    )

    ground_max = float(
        ground_native.max()
    )

    z_all = (
        ds["z"]
        .values
    )

    # Previous prototype used 12 m here.
    #
    # 3 m retains terrain context near the lowest surface
    # while giving steep slopes more atmospheric headroom.
    desired_z_start = max(
        float(
            z_all[0]
        ),
        ground_min
        - BELOW_LOWEST_TERRAIN_M
    )

    z_start = int(
        np.searchsorted(
            z_all,
            desired_z_start
        )
    )

    # 12 native z cells × 0.5 m = 6 m.
    z_start = (
        z_start // BZ
    ) * BZ

    z_start = max(
        0,
        min(
            z_start,
            ds.sizes["z"] - NZ
        )
    )

    z_native = (
        z_all[
            z_start:
            z_start + NZ
        ]
    )

    headroom = float(
        z_native[-1]
        - ground_max
    )

    return {
        "x_start":
            x_start,

        "y_start":
            y_start,

        "z_start":
            z_start,

        "x_native":
            x_native,

        "y_native":
            y_native,

        "z_native":
            z_native,

        "ground_native":
            ground_native,

        "ground_min":
            ground_min,

        "ground_max":
            ground_max,

        "headroom":
            headroom,
    }


# ============================================================
# SAMPLE GENERATION
# ============================================================

def generate_sample(
    ds,
    wind,
    slope,
    split,
    stage,
    output_path,
):
    """
    Download one native FireBench crop and convert it into
    our 32^3 WindSeer-compatible representation.
    """
    slope_deg = slope_degrees(
        slope
    )

    crop = choose_crop(
        ds,
        int(
            stage[
                "front_index"
            ]
        ),
        slope_deg,
    )

    x_start = crop[
        "x_start"
    ]

    y_start = crop[
        "y_start"
    ]

    z_start = crop[
        "z_start"
    ]

    x_native = crop[
        "x_native"
    ]

    y_native = crop[
        "y_native"
    ]

    z_native = crop[
        "z_native"
    ]

    ground_native = crop[
        "ground_native"
    ]

    print(
        "      crop "
        f"x={x_native[0]:.1f}..{x_native[-1]:.1f}  "
        f"z={z_native[0]:.1f}..{z_native[-1]:.1f}  "
        f"headroom={crop['headroom']:.1f}m"
    )

    if crop[
        "headroom"
    ] < MIN_HEADROOM_M:
        print(
            "      WARNING: low atmospheric "
            "headroom on this crop."
        )

    # --------------------------------------------------------
    # Expensive cloud read
    # --------------------------------------------------------

    cloud_start = time.perf_counter()

    block = (
        ds[
            [
                "u",
                "v",
                "w",
                "T_s",
            ]
        ]
        .isel(
            t=int(
                stage[
                    "time_index"
                ]
            ),
            x=slice(
                x_start,
                x_start + NX
            ),
            y=slice(
                y_start,
                y_start + NY
            ),
            z=slice(
                z_start,
                z_start + NZ
            ),
        )
        .load()
    )

    cloud_seconds = (
        time.perf_counter()
        - cloud_start
    )

    print(
        "      cloud read: "
        f"{cloud_seconds:.1f}s"
    )

    u_native = (
        block["u"]
        .transpose(
            "x",
            "y",
            "z"
        )
        .values
    )

    v_native = (
        block["v"]
        .transpose(
            "x",
            "y",
            "z"
        )
        .values
    )

    w_native = (
        block["w"]
        .transpose(
            "x",
            "y",
            "z"
        )
        .values
    )

    Ts_native = (
        block["T_s"]
        .transpose(
            "x",
            "y",
            "z"
        )
        .values
    )

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    preprocess_start = time.perf_counter()

    angle = np.deg2rad(
        slope_deg
    )

    Z = (
        z_native[
            None,
            None,
            :
        ]
    )

    GROUND = (
        ground_native[
            :,
            None,
            None
        ]
    )

    native_air = (
        Z > GROUND
    )

    native_air = np.broadcast_to(
        native_air,
        (
            NX,
            NY,
            NZ,
        )
    )

    x_coarse = (
        x_native
        .reshape(
            N,
            BX
        )
        .mean(
            axis=1
        )
    )

    y_coarse = (
        y_native
        .reshape(
            N,
            BY
        )
        .mean(
            axis=1
        )
    )

    z_coarse = (
        z_native
        .reshape(
            N,
            BZ
        )
        .mean(
            axis=1
        )
    )

    ground_coarse = np.maximum(
        (
            x_coarse
            - 100.0
        )
        * np.tan(
            angle
        ),
        0.0
    )

    coarse_air_xz = (
        z_coarse[
            None,
            :
        ]
        >
        ground_coarse[
            :,
            None
        ]
    )

    coarse_air_xyz = (
        np.broadcast_to(
            coarse_air_xz[
                :,
                None,
                :
            ],
            (
                N,
                N,
                N,
            )
        )
        .copy()
    )

    mask_blocks = (
        native_air.reshape(
            N, BX,
            N, BY,
            N, BZ
        )
    )

    air_count = (
        mask_blocks.sum(
            axis=(
                1,
                3,
                5
            )
        )
    )

    def downsample_wind(native):
        """
        Average atmospheric native cells only.

        Coarse cells whose center lies in terrain are
        explicitly zeroed.
        """
        native_blocks = (
            native.reshape(
                N, BX,
                N, BY,
                N, BZ
            )
        )

        air_sum = np.where(
            mask_blocks,
            native_blocks,
            0.0
        ).sum(
            axis=(
                1,
                3,
                5
            )
        )

        coarse = np.divide(
            air_sum,
            air_count,
            out=np.zeros_like(
                air_sum,
                dtype=np.float32
            ),
            where=(
                air_count > 0
            )
        )

        coarse[
            ~coarse_air_xyz
        ] = 0.0

        # FireBench x,y,z -> WindSeer z,y,x.
        return (
            coarse
            .transpose(
                2,
                1,
                0
            )
            .astype(
                np.float32
            )
        )

    u = downsample_wind(
        u_native
    )

    v = downsample_wind(
        v_native
    )

    w = downsample_wind(
        w_native
    )

    # --------------------------------------------------------
    # Terrain distance field
    # --------------------------------------------------------

    PAD = 32

    x_extended = (
        x_coarse[0]
        +
        np.arange(
            -PAD,
            N + PAD
        )
        * 6.0
    )

    z_extended = (
        z_coarse[0]
        +
        np.arange(
            -PAD,
            N + PAD
        )
        * 6.0
    )

    ground_extended = np.maximum(
        (
            x_extended
            - 100.0
        )
        * np.tan(
            angle
        ),
        0.0
    )

    air_extended = (
        z_extended[
            :,
            None
        ]
        >
        ground_extended[
            None,
            :
        ]
    )

    terrain_extended = (
        distance_transform_edt(
            air_extended
        )
    )

    terrain_xz = (
        terrain_extended[
            PAD:
            PAD + N,

            PAD:
            PAD + N
        ]
    )

    terrain = (
        np.broadcast_to(
            terrain_xz[
                :,
                None,
                :
            ],
            (
                N,
                N,
                N,
            )
        )
        .astype(
            np.float32
        )
        .copy()
    )

    # --------------------------------------------------------
    # 2-D synthetic thermal map
    # --------------------------------------------------------

    surface_temperature = (
        Ts_native.max(
            axis=2
        )
    )

    fire_excess_native = np.maximum(
        surface_temperature
        - T_AMBIENT_K,
        0.0
    )

    fire_xy = (
        fire_excess_native
        .reshape(
            N, BX,
            N, BY
        )
        .mean(
            axis=(
                1,
                3
            )
        )
    )

    # x,y -> y,x
    fire_2d = (
        fire_xy.T
        .astype(
            np.float32
        )
    )

    # --------------------------------------------------------
    # Fifth 3-D channel:
    # thermal value at first air voxel above terrain
    # --------------------------------------------------------

    coarse_air_zyx = (
        coarse_air_xyz
        .transpose(
            2,
            1,
            0
        )
    )

    fire = np.zeros(
        (
            N,
            N,
            N,
        ),
        dtype=np.float32
    )

    for iy in range(N):
        for ix in range(N):

            air_z = np.where(
                coarse_air_zyx[
                    :,
                    iy,
                    ix
                ]
            )[0]

            if len(
                air_z
            ) == 0:
                continue

            iz = int(
                air_z[0]
            )

            fire[
                iz,
                iy,
                ix
            ] = fire_2d[
                iy,
                ix
            ]

    preprocess_seconds = (
        time.perf_counter()
        - preprocess_start
    )

    bad_fire = (
        (fire > 0)
        &
        ~(terrain > 0)
    )

    if int(
        bad_fire.sum()
    ) != 0:
        raise RuntimeError(
            "Thermal values were placed "
            "inside terrain."
        )

    # --------------------------------------------------------
    # Atomic HDF5 save
    # --------------------------------------------------------

    temp_path = (
        output_path
        + ".tmp"
    )

    if os.path.exists(
        temp_path
    ):
        os.remove(
            temp_path
        )

    with h5py.File(
        temp_path,
        "w"
    ) as f:

        f.create_dataset(
            "u",
            data=u
        )

        f.create_dataset(
            "v",
            data=v
        )

        f.create_dataset(
            "w",
            data=w
        )

        f.create_dataset(
            "terrain",
            data=terrain
        )

        f.create_dataset(
            "fire",
            data=fire
        )

        # Preserve the unredundant 2-D thermal map as well.
        f.create_dataset(
            "fire_excess",
            data=fire_2d
        )

        f.create_dataset(
            "ds",
            data=np.array(
                [
                    6.0,
                    6.0,
                    6.0,
                ],
                dtype=np.float32
            )
        )

        f.create_dataset(
            "x",
            data=x_coarse
        )

        f.create_dataset(
            "y",
            data=y_coarse
        )

        f.create_dataset(
            "z",
            data=z_coarse
        )

        f.attrs[
            "split"
        ] = split

        f.attrs[
            "wind_case"
        ] = wind

        f.attrs[
            "slope_case"
        ] = slope

        f.attrs[
            "slope_deg"
        ] = slope_deg

        f.attrs[
            "time_s"
        ] = stage[
            "time_s"
        ]

        f.attrs[
            "stage_fraction"
        ] = stage[
            "stage_fraction"
        ]

        f.attrs[
            "fire_front_x_m"
        ] = stage[
            "front_x_m"
        ]

        f.attrs[
            "source_url"
        ] = case_url(
            wind,
            slope
        )

        f.attrs[
            "thermal_ambient_K"
        ] = T_AMBIENT_K

        f.attrs[
            "fire_representation"
        ] = (
            "surface_temperature_excess_"
            "at_first_air_voxel"
        )

        f.attrs[
            "wind_downsampling"
        ] = (
            "native_air_only_mean"
        )

        f.attrs[
            "terrain_distance_units"
        ] = (
            "grid_cells"
        )

        f.attrs[
            "grid_spacing_m"
        ] = 6.0

        f.attrs[
            "native_cloud_read_seconds"
        ] = cloud_seconds

        f.attrs[
            "preprocess_seconds"
        ] = preprocess_seconds

        f.attrs[
            "terrain_headroom_m"
        ] = crop[
            "headroom"
        ]

    os.replace(
        temp_path,
        output_path
    )

    file_size_mb = (
        os.path.getsize(
            output_path
        )
        / 1024**2
    )

    return {
        "cloud_seconds":
            cloud_seconds,

        "preprocess_seconds":
            preprocess_seconds,

        "file_size_mb":
            file_size_mb,

        "headroom_m":
            crop[
                "headroom"
            ],
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scout-only",
        action="store_true",
        help=(
            "Select progression stages but "
            "do not download native 3-D crops."
        ),
    )

    parser.add_argument(
        "--only-cases",
        nargs="*",
        default=None,
        help=(
            "Optional cases such as "
            "'u6/ramp0 u22/ramp30'."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Regenerate processed HDF5 files "
            "that already exist."
        ),
    )

    args = parser.parse_args()

    scout_dir = os.path.join(
        OUTPUT_ROOT,
        "scouts"
    )

    os.makedirs(
        scout_dir,
        exist_ok=True
    )

    for split in [
        "train",
        "val",
        "test",
    ]:
        os.makedirs(
            os.path.join(
                OUTPUT_ROOT,
                split
            ),
            exist_ok=True
        )

    failure_path = os.path.join(
        OUTPUT_ROOT,
        "failures.log"
    )

    fs = gcsfs.GCSFileSystem(
        token="anon"
    )

    all_cases = []

    for wind_index, wind in enumerate(
        WINDS
    ):
        for slope_index, slope in enumerate(
            SLOPES
        ):
            label = (
                f"{wind}/{slope}"
            )

            if (
                args.only_cases
                and
                label not in args.only_cases
            ):
                continue

            split = split_for_case(
                wind_index,
                slope_index
            )

            all_cases.append(
                (
                    wind_index,
                    slope_index,
                    wind,
                    slope,
                    split,
                )
            )

    print(
        f"Cases selected: {len(all_cases)}"
    )

    print(
        f"Samples expected: "
        f"{len(all_cases) * 5}"
    )

    print()

    global_start = time.perf_counter()

    completed = 0
    skipped = 0
    failures = 0

    for case_number, (
        wind_index,
        slope_index,
        wind,
        slope,
        split,
    ) in enumerate(
        all_cases,
        start=1
    ):

        label = case_name(
            wind,
            slope
        )

        print(
            "=" * 72
        )

        print(
            f"CASE {case_number}/"
            f"{len(all_cases)}: "
            f"{wind}/{slope}  "
            f"[{split}]"
        )

        print(
            "=" * 72
        )

        case_start = (
            time.perf_counter()
        )

        try:
            mapper = fs.get_mapper(
                case_url(
                    wind,
                    slope
                )
            )

            ds = xr.open_zarr(
                mapper,
                consolidated=False
            )

            scout_cache_path = os.path.join(
                scout_dir,
                f"{label}.json"
            )

            stages = (
                select_progression_stages(
                    ds,
                    wind,
                    slope,
                    scout_cache_path,
                )
            )

            if args.scout_only:
                ds.close()
                print()
                continue

            for stage in stages:

                percent = int(
                    round(
                        stage[
                            "stage_fraction"
                        ]
                        * 100
                    )
                )

                time_label = int(
                    round(
                        stage[
                            "time_s"
                        ]
                    )
                )

                filename = (
                    f"{wind}_"
                    f"{slope}_"
                    f"p{percent:02d}_"
                    f"t{time_label}.h5"
                )

                output_path = os.path.join(
                    OUTPUT_ROOT,
                    split,
                    filename
                )

                if (
                    os.path.exists(
                        output_path
                    )
                    and
                    not args.overwrite
                ):
                    print(
                        f"  p{percent:02d}: "
                        "already exists; skipping"
                    )

                    skipped += 1
                    continue

                print(
                    f"  p{percent:02d}: "
                    f"t={stage['time_s']:.3f}s  "
                    f"front={stage['front_x_m']:.1f}m"
                )

                result = generate_sample(
                    ds,
                    wind,
                    slope,
                    split,
                    stage,
                    output_path,
                )

                print(
                    "      saved "
                    f"{result['file_size_mb']:.2f} MiB; "
                    f"preprocess "
                    f"{result['preprocess_seconds']:.2f}s"
                )

                completed += 1

            ds.close()

            case_seconds = (
                time.perf_counter()
                - case_start
            )

            print(
                f"  CASE TIME: "
                f"{case_seconds / 60.0:.1f} min"
            )

            print()

        except Exception as exc:

            failures += 1

            print(
                "  ERROR:",
                repr(
                    exc
                )
            )

            with open(
                failure_path,
                "a"
            ) as f:
                f.write(
                    f"{wind}/{slope}: "
                    f"{repr(exc)}\n"
                )

            try:
                ds.close()
            except Exception:
                pass

            print(
                "  Continuing to next case."
            )

            print()

    total_seconds = (
        time.perf_counter()
        - global_start
    )

    print(
        "=" * 72
    )

    print(
        "RUN COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        "Generated:",
        completed
    )

    print(
        "Skipped:",
        skipped
    )

    print(
        "Failed cases:",
        failures
    )

    print(
        "Elapsed:",
        f"{total_seconds / 3600.0:.2f} h"
    )

    print(
        "Output:",
        OUTPUT_ROOT
    )


if __name__ == "__main__":
    main()
