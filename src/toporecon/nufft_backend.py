"""Backend-neutral NUFFT interface and the MICCAI v1 SigPy implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Sequence

import numpy as np


class NufftBackend(ABC):
    """Minimal contract required by the reconstruction algorithms.

    Coordinates are stored per local motion bin. They must already be in the
    documented internal ``[z,y,x]`` grid-coordinate convention before the
    backend is constructed.
    """

    def __init__(self, img_shape: Sequence[int]) -> None:
        self.img_shape = tuple(int(value) for value in img_shape)
        if len(self.img_shape) != 3 or any(value < 1 for value in self.img_shape):
            raise ValueError(f"img_shape must be three positive values: {img_shape}.")

    @property
    @abstractmethod
    def xp(self) -> Any:
        """Array module used by this backend."""

    @property
    @abstractmethod
    def num_bins(self) -> int:
        """Number of local trajectory bins."""

    @abstractmethod
    def to_device(self, array: Any) -> Any:
        """Move or convert an array to the backend device."""

    @abstractmethod
    def forward(self, image: Any, bin_index: int) -> Any:
        """Apply the forward NUFFT for one image and one motion bin."""

    @abstractmethod
    def adjoint(self, kspace: Any, bin_index: int) -> Any:
        """Apply the adjoint NUFFT for one k-space array and one motion bin."""

    def _validate_bin(self, bin_index: int) -> int:
        resolved = int(bin_index)
        if not 0 <= resolved < self.num_bins:
            raise IndexError(
                f"bin_index {resolved} is outside [0, {self.num_bins})."
            )
        return resolved


class SigPyNufftBackend(NufftBackend):
    """SigPy implementation used as the MICCAI v1 numerical reference."""

    def __init__(
        self,
        img_shape: Sequence[int],
        coordinates: Iterable[np.ndarray],
        device: Any,
    ) -> None:
        super().__init__(img_shape)
        try:
            import sigpy as sp
        except ImportError as exc:  # pragma: no cover - depends on release environment.
            raise RuntimeError(
                "SigPy is required for SigPyNufftBackend. "
                "Install the toporecon package dependencies first."
            ) from exc

        self._sp = sp
        self.device = device if isinstance(device, sp.Device) else sp.Device(device)
        prepared = []
        for index, coordinate in enumerate(coordinates):
            # Coordinates may already be CuPy arrays after motion binning.
            # Avoid forcing a device-to-host conversion merely to inspect shape.
            array = coordinate if hasattr(coordinate, "shape") else np.asarray(coordinate)
            if array.ndim < 2 or array.shape[-1] != len(self.img_shape):
                raise ValueError(
                    f"coordinates[{index}] must end in dimension "
                    f"{len(self.img_shape)}, got {array.shape}."
                )
            prepared.append(sp.to_device(array, self.device))
        if not prepared:
            raise ValueError("At least one trajectory bin is required.")
        self.coordinates = tuple(prepared)

    @property
    def xp(self) -> Any:
        return self.device.xp

    @property
    def num_bins(self) -> int:
        return len(self.coordinates)

    def to_device(self, array: Any) -> Any:
        return self._sp.to_device(array, self.device)

    def forward(self, image: Any, bin_index: int) -> Any:
        resolved = self._validate_bin(bin_index)
        with self.device:
            return self._sp.nufft(
                self.to_device(image),
                self.coordinates[resolved],
            )

    def adjoint(self, kspace: Any, bin_index: int) -> Any:
        resolved = self._validate_bin(bin_index)
        with self.device:
            return self._sp.nufft_adjoint(
                self.to_device(kspace),
                self.coordinates[resolved],
                oshape=self.img_shape,
            )


def create_nufft_backend(
    name: str,
    *,
    img_shape: Sequence[int],
    coordinates: Iterable[np.ndarray],
    device: Any,
) -> NufftBackend:
    """Construct a registered backend without conditionals in algorithms."""

    normalized = name.strip().lower()
    if normalized == "sigpy":
        return SigPyNufftBackend(
            img_shape=img_shape,
            coordinates=coordinates,
            device=device,
        )
    raise ValueError(f"Unknown NUFFT backend {name!r}. Available backends: sigpy.")
