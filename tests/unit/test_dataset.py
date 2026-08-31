from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from toporecon.data.cfl import write_cfl
from toporecon.data.dataset import RawDataset, RawDatasetPaths, canonicalize_kspace
from toporecon.data.validation import validate_raw_dataset


def write_raw_dataset(root: Path) -> tuple[int, int, int, int]:
    echoes, coils, trajectories, readout = 2, 3, 5, 7
    kspace_shape = (1, 1, 1, echoes, 1, coils, trajectories, readout, 1)
    trajectory_shape = (1, 1, 1, 1, 1, 1, trajectories, readout, 3)
    density_shape = (1, 1, 1, 1, 1, 1, trajectories, readout, 1)

    write_cfl(root / "ksp", np.ones(kspace_shape, dtype=np.complex64))
    write_cfl(root / "ktraj", np.ones(trajectory_shape, dtype=np.complex64))
    write_cfl(root / "dens", np.ones(density_shape, dtype=np.complex64))
    (root / "imageDim.txt").write_text("32:24:16\n", encoding="utf-8")
    (root / "voxelSize.txt").write_text("0.2:0.3:0.4\n", encoding="utf-8")
    (root / "tr.txt").write_text("0.05\n", encoding="utf-8")
    return echoes, coils, trajectories, readout


class DatasetTests(unittest.TestCase):
    def test_public_raw_dataset_contract_contains_the_nine_zenodo_files(self) -> None:
        names = {
            path.name
            for path in RawDatasetPaths.from_directory("dataset").required_files()
        }

        self.assertEqual(
            names,
            {
                "ksp.hdr",
                "ksp.cfl",
                "ktraj.hdr",
                "ktraj.cfl",
                "dens.hdr",
                "dens.cfl",
                "imageDim.txt",
                "voxelSize.txt",
                "tr.txt",
            },
        )

    def test_valid_dataset_reports_canonical_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = write_raw_dataset(root)

            report = validate_raw_dataset(root)
            dataset = RawDataset.open(root)

            self.assertTrue(report.is_valid, report.render())
            self.assertEqual(dataset.kspace_layout.canonical_shape, expected)
            self.assertEqual(dataset.image_shape_zyx, (16, 24, 32))
            self.assertEqual(dataset.voxel_size_zyx, (0.4, 0.3, 0.2))
            self.assertEqual(dataset.read_kspace().shape, expected)

    def test_missing_files_are_reported_without_loading_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = validate_raw_dataset(temporary)

            self.assertFalse(report.is_valid)
            self.assertTrue(any("ksp.hdr" in error for error in report.errors))

    def test_preparation_contract_reports_missing_repetition_time(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_raw_dataset(root)
            (root / "tr.txt").unlink()

            report = validate_raw_dataset(root)

            self.assertFalse(report.is_valid)
            self.assertIn("Missing required file: tr.txt", report.errors)

    def test_non_singleton_unsupported_kspace_dimension_is_rejected(self) -> None:
        array = np.ones((2, 1, 1, 2, 1, 3, 5, 7, 1), dtype=np.complex64)

        with self.assertRaisesRegex(ValueError, "Unsupported non-singleton"):
            canonicalize_kspace(array)


if __name__ == "__main__":
    unittest.main()
