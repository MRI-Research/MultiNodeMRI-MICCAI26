"""High-level preparation workflow for raw MICCAI datasets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from toporecon import __version__
from toporecon.data.dataset import RawDataset
from toporecon.data.validation import validate_raw_dataset
from toporecon.preprocessing.jsense import (
    JsenseConfig,
    estimate_sensitivity_from_dataset,
)
from toporecon.preprocessing.respiratory import (
    RespiratoryConfig,
    estimate_respiratory_from_dataset,
)


@dataclass(frozen=True)
class PreparedArtifacts:
    """Locations written by the preparation stage."""

    directory: Path
    respiratory: Path
    sensitivity_maps: Path
    manifest: Path


def prepare_dataset(
    input_directory: str | Path,
    work_directory: str | Path,
    *,
    device: int = 0,
    fov_scale_zyx: Sequence[float] = (1.0, 1.0, 1.0),
    show_progress: bool = False,
    respiratory_config: RespiratoryConfig | None = None,
    jsense_config: JsenseConfig | None = None,
) -> PreparedArtifacts:
    """Run RESP and single-GPU JSENSE into a separate work directory."""

    report = validate_raw_dataset(input_directory)
    if not report.is_valid:
        raise ValueError(report.render())

    raw = RawDataset.open(input_directory)
    destination = Path(work_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    resolved_resp = respiratory_config or RespiratoryConfig()
    resolved_jsense = jsense_config or JsenseConfig()

    respiratory = estimate_respiratory_from_dataset(
        raw.paths.root,
        destination,
        config=resolved_resp,
    )
    sensitivity_maps = estimate_sensitivity_from_dataset(
        raw.paths.root,
        destination,
        device=device,
        fov_scale_zyx=fov_scale_zyx,
        show_progress=show_progress,
        config=resolved_jsense,
    )

    manifest_path = destination / "manifest.json"
    manifest = {
        "format_version": 1,
        "toporecon_version": __version__,
        "input_directory": str(raw.paths.root),
        "canonical_shapes": {
            "kspace": list(raw.kspace_layout.canonical_shape),
            "image_zyx": list(raw.image_shape_zyx),
        },
        "voxel_size_zyx": list(raw.voxel_size_zyx),
        "tr_seconds": raw.tr_seconds,
        "fov_scale_zyx": [float(value) for value in fov_scale_zyx],
        "respiratory": asdict(resolved_resp),
        "jsense": asdict(resolved_jsense),
        "artifacts": {
            "respiratory": respiratory.name,
            "sensitivity_maps": sensitivity_maps.name,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return PreparedArtifacts(
        directory=destination,
        respiratory=respiratory,
        sensitivity_maps=sensitivity_maps,
        manifest=manifest_path,
    )
