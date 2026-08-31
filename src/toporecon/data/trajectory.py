"""Trajectory axis conversion, readout truncation, and coordinate scaling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


CoordinateScaling = Literal["physical", "legacy_max"]


@dataclass(frozen=True)
class PreparedTrajectory:
    """Trajectory and DCF prepared for a NUFFT backend."""

    coordinates_zyx: np.ndarray
    density: np.ndarray
    image_shape_zyx: tuple[int, int, int]
    original_image_shape_zyx: tuple[int, int, int]
    readout: int
    scaling: CoordinateScaling


def _triple(values: Sequence[float] | Sequence[int], name: str) -> tuple:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values.")
    return tuple(values)


def prepare_trajectory(
    trajectory_xyz: np.ndarray,
    density: np.ndarray,
    *,
    image_shape_zyx: Sequence[int],
    voxel_size_zyx: Sequence[float],
    readout_fraction: float,
    fov_scale_zyx: Sequence[float] = (1.0, 1.0, 1.0),
    scaling: CoordinateScaling = "physical",
    unstable_coordinate_threshold: float | None = 1000.0,
) -> PreparedTrajectory:
    """Prepare raw trajectories while preserving the research-code conventions.

    ``physical`` implements ``coord *= voxel_size * image_shape``.
    ``legacy_max`` implements the historical JSENSE per-axis maximum scaling.
    The raw CFL coordinate axis is assumed to be ``[x,y,z]`` and is reversed
    once here.
    """

    coordinates = np.asarray(trajectory_xyz, dtype=np.float32)
    weights = np.asarray(density, dtype=np.float32)
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError(
            "trajectory must have shape [trajectory, readout, 3], "
            f"got {coordinates.shape}."
        )
    if weights.shape != coordinates.shape[:2]:
        raise ValueError(
            f"density shape {weights.shape} does not match trajectory "
            f"{coordinates.shape[:2]}."
        )
    if not 0 < readout_fraction <= 1:
        raise ValueError("readout_fraction must be in (0, 1].")

    original_shape = tuple(int(value) for value in _triple(image_shape_zyx, "image_shape"))
    voxel_size = tuple(float(value) for value in _triple(voxel_size_zyx, "voxel_size"))
    fov_scale = tuple(float(value) for value in _triple(fov_scale_zyx, "fov_scale"))
    if any(value < 1 for value in fov_scale):
        raise ValueError("fov_scale values must be >= 1.")

    readout = max(1, int(readout_fraction * coordinates.shape[-2]))
    coordinates = coordinates[:, :readout, ::-1].copy()
    weights = weights[:, :readout].copy()

    if unstable_coordinate_threshold is not None:
        while readout > 1 and float(coordinates.max()) > unstable_coordinate_threshold:
            readout -= 1
            coordinates = coordinates[:, :readout]
            weights = weights[:, :readout]

    if scaling == "physical":
        for axis in range(3):
            coordinates[..., axis] *= voxel_size[axis] * original_shape[axis]
    elif scaling == "legacy_max":
        for axis in range(3):
            axis_max = float(np.max(coordinates[..., axis]))
            if axis_max == 0:
                raise ValueError(f"trajectory axis {axis} has zero maximum.")
            coordinates[..., axis] *= original_shape[axis] / (2.0 * axis_max)
    else:
        raise ValueError(f"Unknown coordinate scaling mode: {scaling!r}")

    reconstruction_shape = tuple(
        int(original_shape[axis] * fov_scale[axis]) for axis in range(3)
    )
    for axis in range(3):
        coordinates[..., axis] *= fov_scale[axis]

    return PreparedTrajectory(
        coordinates_zyx=np.ascontiguousarray(coordinates),
        density=np.ascontiguousarray(weights),
        image_shape_zyx=reconstruction_shape,
        original_image_shape_zyx=original_shape,
        readout=readout,
        scaling=scaling,
    )
