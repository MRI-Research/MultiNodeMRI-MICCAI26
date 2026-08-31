"""JSENSE coil sensitivity estimation extracted from the historical CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from toporecon.data.cfl import write_cfl
from toporecon.data.dataset import RawDataset
from toporecon.data.trajectory import CoordinateScaling, prepare_trajectory


@dataclass(frozen=True)
class JsenseConfig:
    """JSENSE settings.

    Optional numerical parameters default to ``None`` so MICCAI v1 preserves
    the SigPy application defaults used by the historical script.
    """

    readout_fraction: float = 0.95
    coordinate_scaling: CoordinateScaling = "legacy_max"
    mps_kernel_width: int | None = None
    kspace_calibration_width: int | None = None
    regularization: float | None = None
    max_iter: int | None = None
    max_inner_iter: int | None = None


def estimate_sensitivity_maps(
    kspace: np.ndarray,
    coordinates_zyx: np.ndarray,
    density: np.ndarray,
    *,
    image_shape_zyx: Sequence[int],
    device: Any = 0,
    show_progress: bool = False,
    config: JsenseConfig | None = None,
) -> np.ndarray:
    """Estimate canonical ``[coil,z,y,x]`` sensitivity maps with SigPy."""

    try:
        import sigpy as sp
        import sigpy.mri as mr
    except ImportError as exc:  # pragma: no cover - depends on release environment.
        raise RuntimeError("SigPy is required for JSENSE estimation.") from exc

    data = np.asarray(kspace)
    if data.ndim != 4:
        raise ValueError(
            "k-space must have shape [echo,coil,trajectory,readout], "
            f"got {data.shape}."
        )
    coordinates = np.asarray(coordinates_zyx)
    weights = np.asarray(density)
    if coordinates.shape[:-1] != data.shape[-2:]:
        raise ValueError(
            f"trajectory shape {coordinates.shape} does not match k-space "
            f"{data.shape[-2:]}."
        )
    if weights.shape != data.shape[-2:]:
        raise ValueError(
            f"density shape {weights.shape} does not match k-space {data.shape[-2:]}."
        )

    resolved = config or JsenseConfig()
    target_shape = tuple(int(value) for value in image_shape_zyx)
    sigpy_device = device if isinstance(device, sp.Device) else sp.Device(device)

    kwargs: dict[str, Any] = {
        "coord": coordinates,
        "weights": weights,
        "device": sigpy_device,
        "show_pbar": show_progress,
    }
    optional_parameters = {
        "mps_ker_width": resolved.mps_kernel_width,
        "ksp_calib_width": resolved.kspace_calibration_width,
        "lamda": resolved.regularization,
        "max_iter": resolved.max_iter,
        "max_inner_iter": resolved.max_inner_iter,
    }
    kwargs.update(
        {name: value for name, value in optional_parameters.items() if value is not None}
    )

    # The research implementation uses only the first echo for JSENSE.
    maps = mr.app.JsenseRecon(data[0], **kwargs).run()
    maps = sp.to_device(maps, sp.cpu_device)
    output = np.empty((data.shape[1],) + target_shape, dtype=np.complex64)
    for coil in range(data.shape[1]):
        resized = sp.ifft(sp.resize(sp.fft(maps[coil]), target_shape))
        output[coil] = sp.to_device(resized, sp.cpu_device)
    return output


def estimate_sensitivity_from_dataset(
    input_directory: str | Path,
    output_directory: str | Path,
    *,
    device: Any = 0,
    fov_scale_zyx: Sequence[float] = (1.0, 1.0, 1.0),
    show_progress: bool = False,
    config: JsenseConfig | None = None,
) -> Path:
    """Estimate and write ``mps.hdr/.cfl`` without modifying raw data."""

    resolved = config or JsenseConfig()
    dataset = RawDataset.open(input_directory)
    trajectory = prepare_trajectory(
        dataset.read_trajectory_xyz(),
        dataset.read_density(),
        image_shape_zyx=dataset.image_shape_zyx,
        voxel_size_zyx=dataset.voxel_size_zyx,
        readout_fraction=resolved.readout_fraction,
        fov_scale_zyx=fov_scale_zyx,
        scaling=resolved.coordinate_scaling,
    )
    kspace = dataset.read_kspace()[..., : trajectory.readout]
    maps = estimate_sensitivity_maps(
        kspace,
        trajectory.coordinates_zyx,
        trajectory.density,
        image_shape_zyx=trajectory.image_shape_zyx,
        device=device,
        show_progress=show_progress,
        config=resolved,
    )

    cfl_layout = np.empty(
        (1, 1, 1, 1, 1, maps.shape[0]) + maps.shape[1:],
        dtype=np.complex64,
    )
    cfl_layout[0, 0, 0, 0, 0] = maps
    output_base = Path(output_directory) / "mps"
    write_cfl(output_base, cfl_layout)
    return output_base
