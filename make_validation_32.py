"""
Create a deterministic 32^3 version of the official WindSeer
validation set.

Each of the 6708 official 64^3 validation samples contributes one
fixed 32^3 crop. HDF5 virtual datasets point back to the original
arrays, so the data are not duplicated on disk.
"""

from pathlib import Path
import random

import h5py
import numpy as np


SOURCE = Path(
    "/media/phantom/Volume/WindSeer Data/download (2)/658323/"
    "validation_resampled.hdf5"
)

OUTPUT = Path(
    "/media/phantom/Volume/WindSeer Data/download (2)/658323/"
    "validation_resampled_32.hdf5"
)

CROP_SIZE = 32
SEED = 42

# Avoid completely useless crops containing almost nothing but terrain.
MIN_AIR_FRACTION = 0.01
MAX_ATTEMPTS = 16


def choose_crop(terrain, rng):
    """
    Choose a deterministic random 32^3 crop.

    If a candidate contains almost no non-terrain cells, try another
    location. If none of the attempts pass the threshold, keep the
    candidate containing the most non-terrain cells.
    """
    z_size, y_size, x_size = terrain.shape

    max_z = z_size - CROP_SIZE
    max_y = y_size - CROP_SIZE
    max_x = x_size - CROP_SIZE

    best = None
    best_air_fraction = -1.0

    for _ in range(MAX_ATTEMPTS):
        z0 = rng.randint(0, max_z)
        y0 = rng.randint(0, max_y)
        x0 = rng.randint(0, max_x)

        crop = terrain[
            z0:z0 + CROP_SIZE,
            y0:y0 + CROP_SIZE,
            x0:x0 + CROP_SIZE,
        ]

        air_fraction = float(np.mean(crop > 0))

        if air_fraction > best_air_fraction:
            best = (z0, y0, x0)
            best_air_fraction = air_fraction

        if air_fraction >= MIN_AIR_FRACTION:
            return (z0, y0, x0), air_fraction, False

    return best, best_air_fraction, True


if OUTPUT.exists():
    OUTPUT.unlink()

rng = random.Random(SEED)

air_fractions = []
fallbacks = 0

with h5py.File(SOURCE, "r") as src, h5py.File(
    OUTPUT,
    "w",
    libver="latest",
) as dst:

    for name, value in src.attrs.items():
        dst.attrs[name] = value

    keys = sorted(src.keys())

    print(f"Official validation samples: {len(keys)}")
    print("Creating fixed 32^3 virtual crops...")

    for i, key in enumerate(keys, start=1):
        src_group = src[key]
        dst_group = dst.create_group(key)

        for name, value in src_group.attrs.items():
            dst_group.attrs[name] = value

        crop_origin, air_fraction, used_fallback = choose_crop(
            src_group["terrain"],
            rng,
        )

        z0, y0, x0 = crop_origin

        air_fractions.append(air_fraction)

        if used_fallback:
            fallbacks += 1

        # Save crop coordinates for reproducibility.
        dst_group.attrs["crop_z0"] = z0
        dst_group.attrs["crop_y0"] = y0
        dst_group.attrs["crop_x0"] = x0

        for name, src_dataset in src_group.items():

            if src_dataset.shape == (64, 64, 64):
                source = h5py.VirtualSource(
                    str(SOURCE),
                    f"/{key}/{name}",
                    shape=src_dataset.shape,
                )

                layout = h5py.VirtualLayout(
                    shape=(CROP_SIZE, CROP_SIZE, CROP_SIZE),
                    dtype=src_dataset.dtype,
                )

                layout[:, :, :] = source[
                    z0:z0 + CROP_SIZE,
                    y0:y0 + CROP_SIZE,
                    x0:x0 + CROP_SIZE,
                ]

                virtual_dataset = dst_group.create_virtual_dataset(
                    name,
                    layout,
                )

                for attr_name, attr_value in src_dataset.attrs.items():
                    virtual_dataset.attrs[attr_name] = attr_value

            else:
                # Small metadata arrays such as ds are cheap to copy.
                copied = dst_group.create_dataset(
                    name,
                    data=src_dataset[...],
                )

                for attr_name, attr_value in src_dataset.attrs.items():
                    copied.attrs[attr_name] = attr_value

        if i % 500 == 0 or i == len(keys):
            print(f"  {i}/{len(keys)}")


print("\nVerifying output...")

with h5py.File(OUTPUT, "r") as f:
    keys = list(f.keys())
    first = keys[0]

    print(f"Samples: {len(keys)}")
    print(f"First sample: {first}")
    print(f"Terrain shape: {f[first]['terrain'].shape}")
    print(f"Ux shape: {f[first]['ux'].shape}")
    print(f"Uz shape: {f[first]['uz'].shape}")
    print(f"Virtual terrain: {f[first]['terrain'].is_virtual}")


print(f"Minimum air fraction: {min(air_fractions):.4f}")
print(f"Mean air fraction:    {np.mean(air_fractions):.4f}")
print(f"Fallback crops:       {fallbacks}")

print("\nSUCCESS")
print(OUTPUT)
