from __future__ import annotations

import unittest

import numpy as np

from toporecon.data.trajectory import prepare_trajectory


class TrajectoryTests(unittest.TestCase):
    def test_physical_scaling_reverses_xyz_once_and_applies_fov(self) -> None:
        raw_xyz = np.zeros((2, 4, 3), dtype=np.float32)
        raw_xyz[..., 0] = 1.0
        raw_xyz[..., 1] = 2.0
        raw_xyz[..., 2] = 3.0
        density = np.ones((2, 4), dtype=np.float32)

        result = prepare_trajectory(
            raw_xyz,
            density,
            image_shape_zyx=(10, 20, 30),
            voxel_size_zyx=(0.5, 0.25, 0.1),
            readout_fraction=0.75,
            fov_scale_zyx=(2.0, 1.0, 1.0),
            scaling="physical",
            unstable_coordinate_threshold=None,
        )

        self.assertEqual(result.readout, 3)
        self.assertEqual(result.image_shape_zyx, (20, 20, 30))
        # Raw [x,y,z] = [1,2,3] becomes [z,y,x], then voxel*shape*FOV.
        np.testing.assert_allclose(result.coordinates_zyx[0, 0], (30.0, 10.0, 3.0))

    def test_density_shape_must_match_trajectory(self) -> None:
        with self.assertRaisesRegex(ValueError, "density shape"):
            prepare_trajectory(
                np.zeros((2, 4, 3), dtype=np.float32),
                np.zeros((2, 3), dtype=np.float32),
                image_shape_zyx=(4, 4, 4),
                voxel_size_zyx=(1, 1, 1),
                readout_fraction=1.0,
            )


if __name__ == "__main__":
    unittest.main()
