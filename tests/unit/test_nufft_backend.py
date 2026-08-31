from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import numpy as np

from toporecon.nufft_backend import SigPyNufftBackend


class FakeDevice:
    def __init__(self, identifier=0) -> None:
        self.identifier = identifier
        self.xp = np

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


class SigPyBackendTests(unittest.TestCase):
    def test_forward_and_adjoint_delegate_with_bin_coordinates(self) -> None:
        calls: list[tuple] = []
        fake_sigpy = types.ModuleType("sigpy")
        fake_sigpy.Device = FakeDevice
        fake_sigpy.to_device = lambda array, device: np.asarray(array)

        def fake_forward(image, coordinate):
            calls.append(("forward", coordinate.copy()))
            return np.asarray(image)

        def fake_adjoint(kspace, coordinate, oshape):
            calls.append(("adjoint", coordinate.copy(), tuple(oshape)))
            return np.asarray(kspace).reshape(oshape)

        fake_sigpy.nufft = fake_forward
        fake_sigpy.nufft_adjoint = fake_adjoint

        coordinate_0 = np.zeros((2, 2, 3), dtype=np.float32)
        coordinate_1 = np.ones((2, 2, 3), dtype=np.float32)
        with mock.patch.dict(sys.modules, {"sigpy": fake_sigpy}):
            backend = SigPyNufftBackend(
                img_shape=(2, 2, 1),
                coordinates=[coordinate_0, coordinate_1],
                device=FakeDevice(0),
            )
            image = np.arange(4, dtype=np.complex64).reshape(2, 2, 1)
            backend.forward(image, 1)
            backend.adjoint(image.reshape(2, 2), 0)

        np.testing.assert_array_equal(calls[0][1], coordinate_1)
        np.testing.assert_array_equal(calls[1][1], coordinate_0)
        self.assertEqual(calls[1][2], (2, 2, 1))

    def test_invalid_bin_is_rejected_before_dispatch(self) -> None:
        fake_sigpy = types.ModuleType("sigpy")
        fake_sigpy.Device = FakeDevice
        fake_sigpy.to_device = lambda array, device: np.asarray(array)
        fake_sigpy.nufft = lambda image, coordinate: image
        fake_sigpy.nufft_adjoint = lambda kspace, coordinate, oshape: kspace

        with mock.patch.dict(sys.modules, {"sigpy": fake_sigpy}):
            backend = SigPyNufftBackend(
                img_shape=(2, 2, 1),
                coordinates=[np.zeros((2, 2, 3), dtype=np.float32)],
                device=FakeDevice(0),
            )
            with self.assertRaises(IndexError):
                backend.forward(np.ones((2, 2, 1)), 1)


if __name__ == "__main__":
    unittest.main()
