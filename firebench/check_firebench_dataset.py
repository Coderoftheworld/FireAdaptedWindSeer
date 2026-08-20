from pathlib import Path
import h5py
import numpy as np

ROOT = Path("firebench 6m dataset")

EXPECTED_COUNTS = {
    "train": 420,
    "val": 80,
    "test": 85,
}

EXPECTED_3D = {
    "terrain",
    "u",
    "v",
    "w",
    "fire",
}

EXPECTED_KEYS = EXPECTED_3D | {
    "fire_excess",
    "ds",
    "x",
    "y",
    "z",
}

global_ranges = {
    key: [np.inf, -np.inf]
    for key in EXPECTED_3D
}

train_fire_positive = []
train_fire_nonzero = 0
train_fire_total = 0

problems = []
total_files = 0

for split, expected_count in EXPECTED_COUNTS.items():
    folder = ROOT / split
    files = sorted(folder.glob("*.h5"))

    print(f"{split}: {len(files)} files")

    if len(files) != expected_count:
        problems.append(
            f"{split}: expected {expected_count}, found {len(files)}"
        )

    for path in files:
        total_files += 1

        try:
            with h5py.File(path, "r") as f:
                missing = EXPECTED_KEYS - set(f.keys())

                if missing:
                    problems.append(
                        f"{path}: missing {sorted(missing)}"
                    )
                    continue

                for key in EXPECTED_3D:
                    arr = f[key][:]

                    if arr.shape != (32, 32, 32):
                        problems.append(
                            f"{path}: {key} shape {arr.shape}"
                        )

                    if not np.isfinite(arr).all():
                        problems.append(
                            f"{path}: {key} contains NaN/Inf"
                        )

                    global_ranges[key][0] = min(
                        global_ranges[key][0],
                        float(arr.min()),
                    )
                    global_ranges[key][1] = max(
                        global_ranges[key][1],
                        float(arr.max()),
                    )

                fire_excess = f["fire_excess"][:]

                if fire_excess.shape != (32, 32):
                    problems.append(
                        f"{path}: fire_excess shape "
                        f"{fire_excess.shape}"
                    )

                if not np.isfinite(fire_excess).all():
                    problems.append(
                        f"{path}: fire_excess contains NaN/Inf"
                    )

                ds = f["ds"][:]

                if ds.shape != (3,):
                    problems.append(
                        f"{path}: ds shape {ds.shape}"
                    )
                elif not np.allclose(ds, [6.0, 6.0, 6.0]):
                    problems.append(
                        f"{path}: unexpected ds {ds}"
                    )

                for key in ("x", "y", "z"):
                    coord = f[key][:]

                    if coord.shape != (32,):
                        problems.append(
                            f"{path}: {key} shape {coord.shape}"
                        )

                    if not np.isfinite(coord).all():
                        problems.append(
                            f"{path}: {key} contains NaN/Inf"
                        )

                attr_split = f.attrs.get("split")

                if attr_split != split:
                    problems.append(
                        f"{path}: split attribute is "
                        f"{attr_split!r}, expected {split!r}"
                    )

                if np.any(f["terrain"][:] < 0):
                    problems.append(
                        f"{path}: negative terrain distance"
                    )

                if np.any(f["fire"][:] < 0):
                    problems.append(
                        f"{path}: negative fire value"
                    )

                if split == "train":
                    fire = f["fire"][:]

                    positive = fire[fire > 0]

                    if positive.size:
                        train_fire_positive.append(positive)

                    train_fire_nonzero += np.count_nonzero(fire)
                    train_fire_total += fire.size

        except Exception as exc:
            problems.append(
                f"{path}: failed to read: {exc}"
            )

print("\nGLOBAL RANGES")

for key in sorted(global_ranges):
    low, high = global_ranges[key]
    print(f"{key:10s}: {low:12.5f} to {high:12.5f}")

print("\nTRAINING FIRE STATISTICS")

if train_fire_positive:
    values = np.concatenate(train_fire_positive)

    for percentile in (50, 90, 95, 99, 99.5, 99.9):
        value = np.percentile(values, percentile)
        print(f"positive p{percentile:<4}: {value:10.4f} K")

    print(f"positive max  : {values.max():10.4f} K")

    fraction = train_fire_nonzero / train_fire_total
    print(f"nonzero fraction: {fraction:.6%}")
else:
    print("No positive fire values found in training data.")

print("\nSUMMARY")
print(f"Files checked: {total_files}")
print(f"Problems:      {len(problems)}")

if problems:
    print("\nPROBLEMS")
    for problem in problems[:100]:
        print("-", problem)

    if len(problems) > 100:
        print(
            f"... plus {len(problems) - 100} more"
        )

    raise SystemExit(1)

print("\nDataset sanity check PASSED.")
