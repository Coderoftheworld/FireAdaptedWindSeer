"""
Build lightweight WindSeer-compatible HDF5 index files for the
processed FireBench dataset.

The original FireBench samples remain unchanged. The generated
HDF5 files contain external links to those samples, so the wind
fields are not copied or duplicated.
"""

from pathlib import Path
import os

import h5py


ROOT = Path("firebench 6m dataset").resolve()

EXPECTED_COUNTS = {
    "train": 420,
    "val": 80,
    "test": 85,
}

# WindSeer channel name -> name in our FireBench sample.
CHANNEL_LINKS = {
    "terrain": "terrain",
    "ux": "u",
    "uy": "v",
    "uz": "w",
    "fire": "fire",
    "ds": "ds",
}


def build_split(split, expected_count):
    sample_dir = ROOT / split
    files = sorted(sample_dir.glob("*.h5"))

    if len(files) != expected_count:
        raise RuntimeError(
            f"{split}: expected {expected_count} samples, "
            f"found {len(files)}"
        )

    output_path = ROOT / f"firebench_{split}.hdf5"
    temporary_path = Path(str(output_path) + ".tmp")

    if temporary_path.exists():
        temporary_path.unlink()

    with h5py.File(
        temporary_path,
        "w",
        libver="latest",
    ) as output:
        output.attrs["dataset"] = "FireBench"
        output.attrs["split"] = split
        output.attrs["grid_spacing_m"] = 6.0
        output.attrs["sample_shape"] = [32, 32, 32]
        output.attrs[
            "description"
        ] = "WindSeer-compatible external-link index"

        for index, source_path in enumerate(files):
            relative_source = Path(
                os.path.relpath(
                    source_path,
                    output_path.parent,
                )
            ).as_posix()

            group_name = (
                f"{index:04d}_{source_path.stem}"
            )

            group = output.create_group(group_name)

            group.attrs["source_file"] = relative_source

            for windseer_name, source_name in CHANNEL_LINKS.items():
                group[windseer_name] = h5py.ExternalLink(
                    relative_source,
                    f"/{source_name}",
                )

    temporary_path.replace(output_path)

    # Immediately verify that the external links work.
    with h5py.File(output_path, "r") as check:
        names = list(check.keys())

        if len(names) != expected_count:
            raise RuntimeError(
                f"{output_path}: expected {expected_count} groups, "
                f"found {len(names)}"
            )

        first = check[names[0]]

        expected_shapes = {
            "terrain": (32, 32, 32),
            "ux": (32, 32, 32),
            "uy": (32, 32, 32),
            "uz": (32, 32, 32),
            "fire": (32, 32, 32),
            "ds": (3,),
        }

        for name, expected_shape in expected_shapes.items():
            shape = first[name].shape

            if shape != expected_shape:
                raise RuntimeError(
                    f"{output_path}: {name} has shape {shape}, "
                    f"expected {expected_shape}"
                )

    print(
        f"{split:5s}: {expected_count:3d} samples -> "
        f"{output_path}"
    )


def main():
    if not ROOT.exists():
        raise RuntimeError(
            f"Dataset directory does not exist: {ROOT}"
        )

    for split, expected_count in EXPECTED_COUNTS.items():
        build_split(
            split,
            expected_count,
        )

    print("\nAll WindSeer FireBench indexes created successfully.")


if __name__ == "__main__":
    main()
