from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from toporecon.cli import main
from toporecon.data.cfl import read_cfl, write_cfl
from toporecon.postprocessing import stitch


def split_range(total: int, group: int, groups: int) -> tuple[int, int]:
    return total * group // groups, total * (group + 1) // groups


def write_run(
    directory: Path,
    *,
    stem: str = "reconstruction",
    motion_bins: int = 5,
    echoes: int = 7,
    motion_groups: int = 2,
    echo_groups: int = 3,
    omit_last: bool = False,
) -> np.ndarray:
    shape = (motion_bins, 1, 1, echoes, 1, 1, 2, 3, 4)
    expected = np.empty(shape, dtype=np.complex64)
    for motion in range(motion_bins):
        for echo in range(echoes):
            expected[motion, :, :, echo, ...] = 100 * motion + echo

    shards = []
    for motion_group in range(motion_groups):
        m0, m1 = split_range(motion_bins, motion_group, motion_groups)
        for echo_group in range(echo_groups):
            e0, e1 = split_range(echoes, echo_group, echo_groups)
            shards.append((m0, m1, e0, e1))
    if omit_last:
        shards.pop()
    for m0, m1, e0, e1 in shards:
        write_cfl(
            directory / f"{stem}_e{e0}-{e1 - 1}_m{m0}-{m1 - 1}",
            expected[m0:m1, :, :, e0:e1, ...],
        )

    manifest = {
        "format_version": 1,
        "output_stem": stem,
        "parameters": {"num_bins": motion_bins},
        "distributed": {
            "motion_groups": motion_groups,
            "echo_groups": echo_groups,
        },
    }
    (directory / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return expected


class PostprocessingTests(unittest.TestCase):
    def test_stitches_uneven_grid_with_custom_prefix_and_output_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            expected = write_run(directory, stem="custom_prefix")

            output = stitch(directory, output_name="final_image")

            self.assertEqual(output, (directory / "final_image").resolve())
            np.testing.assert_array_equal(read_cfl(output), expected)

    def test_cli_uses_manifest_prefix_and_default_output_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            expected = write_run(
                directory,
                motion_bins=2,
                echoes=2,
                motion_groups=1,
                echo_groups=1,
            )

            status = main(["stitch", str(directory)])

            self.assertEqual(status, 0)
            np.testing.assert_array_equal(read_cfl(directory / "imout"), expected)

    def test_missing_grid_tile_is_rejected_without_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_run(directory, omit_last=True)

            with self.assertRaisesRegex(ValueError, "Expected 6 shards"):
                stitch(directory)

            self.assertFalse((directory / "imout.hdr").exists())
            self.assertFalse((directory / "imout.cfl").exists())

    def test_output_cannot_replace_an_input_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_run(
                directory,
                motion_bins=2,
                echoes=2,
                motion_groups=1,
                echo_groups=1,
            )

            with self.assertRaisesRegex(ValueError, "cannot overwrite"):
                stitch(directory, output_name="reconstruction_e0-1_m0-1")


if __name__ == "__main__":
    unittest.main()
