from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from toporecon.data.cfl import (
    expected_cfl_bytes,
    read_cfl,
    read_cfl_header,
    read_kspace_shard,
    read_mps_coils,
    validate_cfl_storage,
    write_cfl,
)


class CflTests(unittest.TestCase):
    def test_round_trip_preserves_shape_and_complex64_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "array"
            source = (
                np.arange(24, dtype=np.float32).reshape(2, 3, 4)
                + 1j * np.float32(2)
            )

            write_cfl(base, source)
            result = read_cfl(base)

            self.assertEqual(read_cfl_header(base), source.shape)
            self.assertEqual(result.dtype, np.complex64)
            np.testing.assert_array_equal(result, source.astype(np.complex64))
            self.assertEqual(
                base.with_suffix(".cfl").stat().st_size,
                expected_cfl_bytes(source.shape),
            )

    def test_payload_size_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "broken"
            write_cfl(base, np.ones((2, 2), dtype=np.complex64))
            base.with_suffix(".cfl").write_bytes(b"\x00")

            with self.assertRaisesRegex(ValueError, "payload size mismatch"):
                validate_cfl_storage(base)

    def test_kspace_shard_preserves_requested_index_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "ksp"
            canonical = np.arange(
                2 * 3 * 5 * 7, dtype=np.float32
            ).reshape(2, 3, 5, 7)
            stored = canonical.reshape((1, 1, 1, 2, 1, 3, 5, 7, 1))
            write_cfl(base, stored)

            result = read_kspace_shard(
                base,
                echo_start=1,
                echo_end=2,
                coil_indices=[2, 0],
                trajectory_indices=[4, 1],
                readout=3,
            )

            expected = canonical[1:2][:, [2, 0]][:, :, [4, 1], :3]
            np.testing.assert_array_equal(result, expected)

    def test_mps_reader_returns_canonical_coil_first_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "mps"
            canonical = np.arange(
                3 * 2 * 4 * 5, dtype=np.float32
            ).reshape(3, 2, 4, 5)
            stored = canonical.reshape((1, 1, 1, 1, 1, 3, 2, 4, 5))
            write_cfl(base, stored)

            result = read_mps_coils(base, coil_indices=[2, 0])

            np.testing.assert_array_equal(result, canonical[[2, 0]])


if __name__ == "__main__":
    unittest.main()
