"""Stitch motion/echo reconstruction shards into one CFL/HDR image."""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from toporecon.data import cfl


@dataclass(frozen=True)
class Shard:
    base: Path
    echo_start: int
    echo_stop: int
    motion_start: int
    motion_stop: int
    shape: tuple[int, ...]


def _load_manifest(directory: Path) -> dict:
    path = directory / "run_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing reconstruction manifest: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid reconstruction manifest: {path}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid reconstruction manifest: {path}")
    return manifest


def _resolve_shard_location(directory: Path, stem: str) -> tuple[Path, str]:
    stem_path = Path(stem)
    if stem_path.is_absolute() or not stem_path.name:
        raise ValueError("The input prefix must be relative to INPUT_DIR.")
    shard_directory = (directory / stem_path.parent).resolve()
    try:
        shard_directory.relative_to(directory)
    except ValueError as error:
        raise ValueError("The input prefix must stay inside INPUT_DIR.") from error
    if not shard_directory.is_dir():
        raise FileNotFoundError(f"Shard directory does not exist: {shard_directory}")
    return shard_directory, stem_path.name


def _discover_shards(directory: Path, stem: str) -> list[Shard]:
    shard_directory, stem_name = _resolve_shard_location(directory, stem)
    pattern = re.compile(
        rf"{re.escape(stem_name)}_e(?P<e0>\d+)-(?P<e1>\d+)"
        rf"_m(?P<m0>\d+)-(?P<m1>\d+)\.hdr"
    )
    shards = []
    for header in sorted(shard_directory.iterdir()):
        match = pattern.fullmatch(header.name)
        if match is None:
            continue
        base = header.with_suffix("")
        if not base.with_suffix(".cfl").is_file():
            raise FileNotFoundError(f"Missing CFL payload for shard: {base}")
        shape = cfl.read_cfl_header(base)
        cfl.validate_cfl_storage(base, shape)
        shards.append(
            Shard(
                base=base,
                echo_start=int(match.group("e0")),
                echo_stop=int(match.group("e1")) + 1,
                motion_start=int(match.group("m0")),
                motion_stop=int(match.group("m1")) + 1,
                shape=shape,
            )
        )
    if not shards:
        raise FileNotFoundError(f"No shards found for input prefix {stem!r}.")
    return shards


def _check_partition(ranges: set[tuple[int, int]], total: int, name: str) -> None:
    cursor = 0
    for start, stop in sorted(ranges):
        if start != cursor or stop <= start:
            raise ValueError(f"The {name} shard ranges have a gap or overlap.")
        cursor = stop
    if cursor != total:
        raise ValueError(f"The {name} shard ranges do not cover 0:{total}.")


def _validate_shards(shards: list[Shard], manifest: dict) -> tuple[int, ...]:
    try:
        total_motion = int(manifest["parameters"]["num_bins"])
        motion_groups = int(manifest["distributed"]["motion_groups"])
        echo_groups = int(manifest["distributed"]["echo_groups"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("run_manifest.json is missing its grid configuration.") from error
    if total_motion < 1 or motion_groups < 1 or echo_groups < 1:
        raise ValueError("The grid configuration must contain positive integers.")
    if len(shards) != motion_groups * echo_groups:
        raise ValueError(
            f"Expected {motion_groups * echo_groups} shards, found {len(shards)}."
        )

    inferred_echoes = max(shard.echo_stop for shard in shards)
    recorded_echoes = manifest["parameters"].get("num_echoes")
    try:
        total_echoes = (
            inferred_echoes if recorded_echoes is None else int(recorded_echoes)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("run_manifest.json contains an invalid echo count.") from error
    if total_echoes < 1:
        raise ValueError("run_manifest.json contains an invalid echo count.")
    motion_ranges = {(shard.motion_start, shard.motion_stop) for shard in shards}
    echo_ranges = {(shard.echo_start, shard.echo_stop) for shard in shards}
    if len(motion_ranges) != motion_groups or len(echo_ranges) != echo_groups:
        raise ValueError("Shard ranges do not match the manifest grid configuration.")
    expected_tiles = {
        (motion_range, echo_range)
        for motion_range in motion_ranges
        for echo_range in echo_ranges
    }
    actual_tiles = {
        (
            (shard.motion_start, shard.motion_stop),
            (shard.echo_start, shard.echo_stop),
        )
        for shard in shards
    }
    if actual_tiles != expected_tiles:
        raise ValueError("The motion/echo grid contains a missing or duplicate tile.")
    _check_partition(motion_ranges, total_motion, "motion")
    _check_partition(echo_ranges, total_echoes, "echo")

    reference = list(shards[0].shape)
    if len(reference) != 9:
        raise ValueError(f"Expected a 9D reconstruction shard, got {reference}.")
    for axis in (1, 2, 4, 5):
        if reference[axis] != 1:
            raise ValueError("Reconstruction shard singleton axes are invalid.")
    for shard in shards:
        if len(shard.shape) != 9:
            raise ValueError(f"Unexpected shard shape: {shard.base}")
        if shard.shape[0] != shard.motion_stop - shard.motion_start:
            raise ValueError(f"Motion range does not match {shard.base.name}.")
        if shard.shape[3] != shard.echo_stop - shard.echo_start:
            raise ValueError(f"Echo range does not match {shard.base.name}.")
        for axis, (actual, expected) in enumerate(zip(shard.shape, reference)):
            if axis not in (0, 3) and actual != expected:
                raise ValueError(f"Incompatible shard shape: {shard.base.name}.")

    reference[0] = total_motion
    reference[3] = total_echoes
    return tuple(reference)


def _write_header(base: Path, shape: tuple[int, ...]) -> None:
    base.with_suffix(".hdr").write_text(
        "# Dimensions\n" + " ".join(str(value) for value in shape[::-1]) + "\n",
        encoding="utf-8",
    )


def _resolve_output(
    directory: Path,
    output_name: str,
    output: Path | None,
) -> Path:
    if output is None:
        name = Path(output_name)
        if (
            name.is_absolute()
            or len(name.parts) != 1
            or name.name != output_name
            or output_name in {".", ".."}
        ):
            raise ValueError("--output-name must be a filename, not a path.")
        requested = directory / name
    else:
        requested = output.expanduser()
    if requested.suffix not in {"", ".hdr", ".cfl"}:
        raise ValueError("The output must be a base path or end in .hdr/.cfl.")
    return cfl.base_path(requested).resolve()


def stitch(
    directory: str | Path,
    *,
    stem: str | None = None,
    output_name: str = "imout",
    output: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Assemble all shards recorded by one reconstruction run."""

    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Reconstruction directory does not exist: {directory}")
    manifest = _load_manifest(directory)
    stem = stem or manifest.get("output_stem")
    if not isinstance(stem, str) or not stem:
        raise ValueError("run_manifest.json is missing output_stem.")

    shards = _discover_shards(directory, stem)
    output_shape = _validate_shards(shards, manifest)
    output_base = _resolve_output(directory, output_name, output)
    output_header = output_base.with_suffix(".hdr")
    output_payload = output_base.with_suffix(".cfl")
    input_files = {
        path.resolve()
        for shard in shards
        for path in (shard.base.with_suffix(".hdr"), shard.base.with_suffix(".cfl"))
    }
    if output_header in input_files or output_payload in input_files:
        raise ValueError("The assembled output cannot overwrite an input shard.")
    if not overwrite and (output_header.exists() or output_payload.exists()):
        raise FileExistsError(f"Output already exists: {output_base}")

    output_base.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_base.parent / f"_toporecon_stitch_{uuid.uuid4().hex}"
    try:
        _write_header(temporary, output_shape)
        destination = np.memmap(
            temporary.with_suffix(".cfl"),
            mode="w+",
            dtype=cfl.CFL_DTYPE,
            shape=output_shape,
        )
        for shard in shards:
            source = cfl.read_cfl(shard.base, mmap=True)
            destination[
                shard.motion_start : shard.motion_stop,
                :,
                :,
                shard.echo_start : shard.echo_stop,
                :,
                :,
                :,
                :,
                :,
            ] = source
            del source
        destination.flush()
        del destination
        cfl.validate_cfl_storage(temporary, output_shape)
        os.replace(temporary.with_suffix(".cfl"), output_payload)
        os.replace(temporary.with_suffix(".hdr"), output_header)
    finally:
        temporary.with_suffix(".cfl").unlink(missing_ok=True)
        temporary.with_suffix(".hdr").unlink(missing_ok=True)
    return output_base
