"""Read-only validation for the public raw dataset contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .cfl import read_cfl_header, validate_cfl_storage
from .dataset import (
    KSpaceLayout,
    RawDatasetPaths,
    read_colon_vector,
    squeezed_shape,
)


@dataclass
class DatasetReport:
    """Structured validation result used by both the CLI and tests."""

    root: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, str] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def render(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        lines = [f"Dataset: {self.root}", f"Status: {status}"]
        for key, value in self.details.items():
            lines.append(f"{key}: {value}")
        for warning in self.warnings:
            lines.append(f"Warning: {warning}")
        for error in self.errors:
            lines.append(f"Error: {error}")
        return "\n".join(lines)


def validate_raw_dataset(directory: str | Path) -> DatasetReport:
    """Validate filenames, payload sizes, shapes, and scalar metadata."""

    paths = RawDatasetPaths.from_directory(directory)
    report = DatasetReport(root=paths.root)

    missing = [path for path in paths.required_files() if not path.is_file()]
    if missing:
        report.errors.extend(f"Missing required file: {path.name}" for path in missing)
        return report

    try:
        kspace_shape = read_cfl_header(paths.kspace)
        trajectory_shape = read_cfl_header(paths.trajectory)
        density_shape = read_cfl_header(paths.density)
        for cfl_path, shape in (
            (paths.kspace, kspace_shape),
            (paths.trajectory, trajectory_shape),
            (paths.density, density_shape),
        ):
            validate_cfl_storage(cfl_path, shape)

        layout = KSpaceLayout.from_cfl_shape(kspace_shape)
        trajectory_canonical = squeezed_shape(trajectory_shape)
        density_canonical = squeezed_shape(density_shape)

        expected_trajectory = (layout.trajectories, layout.readout, 3)
        expected_density = (layout.trajectories, layout.readout)
        if trajectory_canonical != expected_trajectory:
            report.errors.append(
                f"ktraj shape {trajectory_canonical} does not match "
                f"{expected_trajectory}."
            )
        if density_canonical != expected_density:
            report.errors.append(
                f"dens shape {density_canonical} does not match {expected_density}."
            )

        image_shape = read_colon_vector(paths.image_dimensions, dtype=int, reverse=True)
        voxel_size = read_colon_vector(paths.voxel_size, dtype=float, reverse=True)
        tr_seconds = float(paths.repetition_time.read_text(encoding="utf-8").strip())

        if any(value <= 0 for value in image_shape):
            report.errors.append(f"image dimensions must be positive: {image_shape}.")
        if any(value <= 0 for value in voxel_size):
            report.errors.append(f"voxel sizes must be positive: {voxel_size}.")
        if tr_seconds <= 0:
            report.errors.append(f"TR must be positive, found {tr_seconds}.")

        report.details.update(
            {
                "k-space": str(layout.canonical_shape),
                "trajectory": str(trajectory_canonical),
                "density": str(density_canonical),
                "image shape [z,y,x]": str(image_shape),
                "voxel size [z,y,x]": str(voxel_size),
                "TR [s]": str(tr_seconds),
            }
        )
    except (OSError, TypeError, ValueError) as exc:
        report.errors.append(str(exc))

    return report
