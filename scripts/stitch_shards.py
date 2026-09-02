#!/usr/bin/env python3
"""Assemble motion/echo reconstruction shards into one CFL/HDR image."""

from __future__ import annotations

import argparse
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
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid reconstruction manifest: {path}")
    return manifest


def _discover_shards(directory: Path, stem: str) -> list[Shard]:
    stem_path = Path(stem)
    shard_directory = directory / stem_path.parent
    pattern = re.compile(
        rf"{re.escape(stem_path.name)}_e(?P<e0>\d+)-(?P<e1>\d+)"
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
        raise FileNotFoundError(f"No shards found for output stem {stem!r}.")
    return shards


def _check_partition(ranges: set[tuple[int, int]], total: int, name: str) -> None:
    cursor = 0
    for start, stop in sorted(ranges):
        if start != cursor or stop <= start:
            raise ValueError(f"The {name} shard ranges have a gap or overlap.")
        cursor = stop
    if cursor != total:
        raise ValueError(f"The {name} shard ranges do not cover 0:{total}.")


def _validate_shards(
    shards: list[Shard], manifest: dict
) -> tuple[tuple[int, ...], list[Shard]]:
    try:
        total_motion = int(manifest["parameters"]["num_bins"])
        motion_groups = int(manifest["distributed"]["motion_groups"])
        echo_groups = int(manifest["distributed"]["echo_groups"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("run_manifest.json is missing its grid configuration.") from error

    if len(shards) != motion_groups * echo_groups:
        raise ValueError(
            f"Expected {motion_groups * echo_groups} shards, found {len(shards)}."
        )

    total_echoes = max(shard.echo_stop for shard in shards)
    motion_ranges = {
        (shard.motion_start, shard.motion_stop) for shard in shards
    }
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
    return tuple(reference), shards


def _write_header(base: Path, shape: tuple[int, ...]) -> None:
    base.with_suffix(".hdr").write_text(
        "# Dimensions\n" + " ".join(str(value) for value in shape[::-1]) + "\n",
        encoding="utf-8",
    )


def stitch(
    directory: Path,
    *,
    stem: str | None = None,
    output_name: str = "imout",
    output: Path | None = None,
    overwrite: bool = False,
) -> Path:
    directory = directory.expanduser().resolve()
    manifest = _load_manifest(directory)
    stem = stem or manifest.get("output_stem")
    if not isinstance(stem, str) or not stem:
        raise ValueError("run_manifest.json is missing output_stem.")

    output_shape, shards = _validate_shards(
        _discover_shards(directory, stem), manifest
    )
    requested_output = output or directory / output_name
    if requested_output.suffix not in {"", ".hdr", ".cfl"}:
        raise ValueError("--output must be a base path or end in .hdr/.cfl.")
    output_base = cfl.base_path(requested_output).expanduser().resolve()
    output_header = output_base.with_suffix(".hdr")
    output_payload = output_base.with_suffix(".cfl")
    if output_base in {shard.base.resolve() for shard in shards}:
        raise ValueError("The assembled output cannot overwrite an input shard.")
    if not overwrite and (output_header.exists() or output_payload.exists()):
        raise FileExistsError(f"Output already exists: {output_base}")

    output_base.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_base.parent / f"_stitch_{uuid.uuid4().hex}"
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
        os.replace(temporary.with_suffix(".cfl"), output_payload)
        os.replace(temporary.with_suffix(".hdr"), output_header)
    finally:
        temporary.with_suffix(".cfl").unlink(missing_ok=True)
        temporary.with_suffix(".hdr").unlink(missing_ok=True)
    return output_base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument(
        "--input-prefix",
        "--stem",
        dest="stem",
        help="Input shard prefix; default: output_stem from run_manifest.json.",
    )
    parser.add_argument(
        "--output-name",
        default="imout",
        help="Final filename stem inside INPUT_DIR; default: imout.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional full output path; overrides --output-name.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = stitch(
        args.input_dir,
        stem=args.stem,
        output_name=args.output_name,
        output=args.output,
        overwrite=args.overwrite,
    )
    print(f"Wrote {output}.hdr and {output}.cfl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
