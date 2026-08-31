"""Dataset paths and conversion from on-disk CFL layouts to canonical arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .cfl import read_cfl, read_cfl_header


@dataclass(frozen=True)
class RawDatasetPaths:
    """Paths belonging to one raw MICCAI release dataset."""

    root: Path
    kspace: Path
    trajectory: Path
    density: Path
    image_dimensions: Path
    voxel_size: Path
    repetition_time: Path

    @classmethod
    def from_directory(cls, directory: str | Path) -> "RawDatasetPaths":
        root = Path(directory).expanduser().resolve()
        return cls(
            root=root,
            kspace=root / "ksp",
            trajectory=root / "ktraj",
            density=root / "dens",
            image_dimensions=root / "imageDim.txt",
            voxel_size=root / "voxelSize.txt",
            repetition_time=root / "tr.txt",
        )

    def required_files(self) -> tuple[Path, ...]:
        return (
            self.kspace.with_suffix(".hdr"),
            self.kspace.with_suffix(".cfl"),
            self.trajectory.with_suffix(".hdr"),
            self.trajectory.with_suffix(".cfl"),
            self.density.with_suffix(".hdr"),
            self.density.with_suffix(".cfl"),
            self.image_dimensions,
            self.voxel_size,
            self.repetition_time,
        )


@dataclass(frozen=True)
class KSpaceLayout:
    """Canonical dimensions embedded in the historical nine-dimensional CFL."""

    echoes: int
    coils: int
    trajectories: int
    readout: int

    @property
    def canonical_shape(self) -> tuple[int, int, int, int]:
        return self.echoes, self.coils, self.trajectories, self.readout

    @classmethod
    def from_cfl_shape(cls, shape: Iterable[int]) -> "KSpaceLayout":
        dimensions = tuple(int(value) for value in shape)
        if len(dimensions) < 8:
            raise ValueError(
                "k-space CFL must contain at least eight dimensions; "
                f"found {dimensions}."
            )

        layout = cls(
            echoes=dimensions[-6],
            coils=dimensions[-4],
            trajectories=dimensions[-3],
            readout=dimensions[-2],
        )
        canonical_elements = int(np.prod(layout.canonical_shape, dtype=np.int64))
        stored_elements = int(np.prod(dimensions, dtype=np.int64))
        if canonical_elements != stored_elements:
            raise ValueError(
                "Unsupported non-singleton dimensions in k-space CFL shape "
                f"{dimensions}; expected only echo, coil, trajectory, and readout."
            )
        return layout


def squeezed_shape(shape: Iterable[int]) -> tuple[int, ...]:
    """Return non-singleton dimensions without loading an array."""

    return tuple(int(value) for value in shape if int(value) != 1)


def canonicalize_kspace(
    array: np.ndarray,
    layout: KSpaceLayout | None = None,
) -> np.ndarray:
    """Return k-space with shape ``[echo, coil, trajectory, readout]``."""

    resolved = layout or KSpaceLayout.from_cfl_shape(array.shape)
    expected = int(np.prod(resolved.canonical_shape, dtype=np.int64))
    if array.size != expected:
        raise ValueError(
            f"k-space contains {array.size} elements; expected {expected} "
            f"for shape {resolved.canonical_shape}."
        )
    return np.asarray(array).reshape(resolved.canonical_shape)


def canonicalize_mps(array: np.ndarray) -> np.ndarray:
    """Return sensitivity maps with shape ``[coil, z, y, x]``."""

    if array.ndim < 4:
        raise ValueError(f"MPS CFL must have at least four dimensions, got {array.shape}.")
    coils = int(array.shape[-4])
    spatial_shape = tuple(int(value) for value in array.shape[-3:])
    expected = coils * int(np.prod(spatial_shape, dtype=np.int64))
    if array.size != expected:
        raise ValueError(
            "Unsupported non-singleton dimensions in MPS CFL shape "
            f"{array.shape}."
        )
    return np.asarray(array).reshape((coils,) + spatial_shape)


def read_colon_vector(
    path: str | Path,
    *,
    dtype: type[int] | type[float],
    reverse: bool = True,
) -> tuple[int, int, int] | tuple[float, float, float]:
    """Read a three-value colon-delimited spatial metadata file."""

    text = Path(path).read_text(encoding="utf-8").strip()
    values = [dtype(value) for value in text.split(":")]
    if len(values) != 3:
        raise ValueError(f"Expected three colon-separated values in {path}, got {text!r}.")
    if reverse:
        values.reverse()
    return tuple(values)  # type: ignore[return-value]


@dataclass(frozen=True)
class RawDataset:
    """Validated access to one raw dataset."""

    paths: RawDatasetPaths
    kspace_layout: KSpaceLayout
    image_shape_zyx: tuple[int, int, int]
    voxel_size_zyx: tuple[float, float, float]
    tr_seconds: float

    @classmethod
    def open(cls, directory: str | Path) -> "RawDataset":
        paths = RawDatasetPaths.from_directory(directory)
        missing = [path for path in paths.required_files() if not path.is_file()]
        if missing:
            formatted = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"Dataset is missing required files: {formatted}")

        layout = KSpaceLayout.from_cfl_shape(read_cfl_header(paths.kspace))
        image_shape = read_colon_vector(
            paths.image_dimensions, dtype=int, reverse=True
        )
        voxel_size = read_colon_vector(paths.voxel_size, dtype=float, reverse=True)
        tr_seconds = float(paths.repetition_time.read_text(encoding="utf-8").strip())

        return cls(
            paths=paths,
            kspace_layout=layout,
            image_shape_zyx=image_shape,
            voxel_size_zyx=voxel_size,
            tr_seconds=tr_seconds,
        )

    def read_kspace(self, *, mmap: bool = False) -> np.ndarray:
        return canonicalize_kspace(
            read_cfl(self.paths.kspace, mmap=mmap),
            self.kspace_layout,
        )

    def read_trajectory_xyz(self) -> np.ndarray:
        return np.squeeze(read_cfl(self.paths.trajectory)).real

    def read_density(self) -> np.ndarray:
        return np.squeeze(read_cfl(self.paths.density)).real
