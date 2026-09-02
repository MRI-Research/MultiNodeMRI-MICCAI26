"""Command-line interface for preparation, reconstruction, and output assembly."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from toporecon import __version__
from toporecon.data.validation import validate_raw_dataset
from toporecon.pipeline import prepare_dataset
from toporecon.preprocessing.jsense import (
    JsenseConfig,
    estimate_sensitivity_from_dataset,
)
from toporecon.preprocessing.respiratory import estimate_respiratory_from_dataset


ALGORITHMS = ("tvm", "tvme", "tvmw")


def _add_jsense_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device index; use -1 for CPU (very slow for JSENSE).",
    )
    parser.add_argument(
        "--readout-fraction",
        type=float,
        default=0.95,
        help="Fraction of readout samples used by JSENSE.",
    )
    parser.add_argument(
        "--coordinate-scaling",
        choices=("legacy_max", "physical"),
        default="legacy_max",
        help=(
            "Coordinate scaling for JSENSE. legacy_max preserves the historical "
            "script until the MICCAI numerical baseline is frozen."
        ),
    )
    parser.add_argument(
        "--fov-scale",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("Z", "Y", "X"),
        help="Reconstruction FOV scale in z, y, x order.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show the SigPy JSENSE progress bar.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="toporecon",
        description="Topology-aware motion-resolved MRI reconstruction.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Validate a raw MICCAI dataset without loading k-space.",
    )
    inspect_parser.add_argument("input", type=Path)

    resp_parser = subparsers.add_parser(
        "resp",
        help="Estimate resp.hdr/.cfl in a separate work directory.",
    )
    resp_parser.add_argument("input", type=Path)
    resp_parser.add_argument("--work-dir", type=Path, required=True)

    jsense_parser = subparsers.add_parser(
        "jsense",
        help="Estimate mps.hdr/.cfl in a separate work directory.",
    )
    jsense_parser.add_argument("input", type=Path)
    jsense_parser.add_argument("--work-dir", type=Path, required=True)
    _add_jsense_arguments(jsense_parser)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Run RESP and JSENSE and write a preparation manifest.",
    )
    prepare_parser.add_argument("input", type=Path)
    prepare_parser.add_argument("--work-dir", type=Path, required=True)
    _add_jsense_arguments(prepare_parser)

    reconstruct_parser = subparsers.add_parser(
        "reconstruct",
        help="Run a topology-aware reconstruction algorithm.",
    )
    reconstruct_parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    reconstruct_parser.add_argument(
        "algorithm_arguments",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected algorithm.",
    )

    stitch_parser = subparsers.add_parser(
        "stitch",
        help="Assemble reconstruction shards into one CFL/HDR image.",
    )
    stitch_parser.add_argument("input_dir", type=Path)
    stitch_parser.add_argument(
        "--input-prefix",
        "--stem",
        dest="stem",
        help="Shard prefix; default: output_stem from run_manifest.json.",
    )
    output_group = stitch_parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output-name",
        default="imout",
        help="Final filename inside INPUT_DIR; default: imout.",
    )
    output_group.add_argument(
        "--output",
        type=Path,
        help="Full path for the final image.",
    )
    stitch_parser.add_argument("--overwrite", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.command == "inspect":
        report = validate_raw_dataset(args.input)
        print(report.render())
        return 0 if report.is_valid else 1

    if args.command == "reconstruct":
        forwarded = list(args.algorithm_arguments)
        if forwarded[:1] == ["--"]:
            forwarded = forwarded[1:]
        if args.algorithm == "tvm":
            from toporecon.algorithms.tvm import main as tvm_main

            return tvm_main(forwarded)
        if args.algorithm == "tvme":
            from toporecon.algorithms.tvme import main as tvme_main

            return tvme_main(forwarded)
        if args.algorithm == "tvmw":
            from toporecon.algorithms.tvmw import main as tvmw_main

            return tvmw_main(forwarded)

    if args.command == "stitch":
        from toporecon.postprocessing import stitch

        output = stitch(
            args.input_dir,
            stem=args.stem,
            output_name=args.output_name,
            output=args.output,
            overwrite=args.overwrite,
        )
        print(f"Wrote {output}.hdr and {output}.cfl")
        return 0

    args.work_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "resp":
        output = estimate_respiratory_from_dataset(args.input, args.work_dir)
        print(f"Wrote {output}.hdr and {output}.cfl")
        return 0

    jsense_config = JsenseConfig(
        readout_fraction=args.readout_fraction,
        coordinate_scaling=args.coordinate_scaling,
    )
    if args.command == "jsense":
        output = estimate_sensitivity_from_dataset(
            args.input,
            args.work_dir,
            device=args.device,
            fov_scale_zyx=args.fov_scale,
            show_progress=args.show_progress,
            config=jsense_config,
        )
        print(f"Wrote {output}.hdr and {output}.cfl")
        return 0

    if args.command == "prepare":
        artifacts = prepare_dataset(
            args.input,
            args.work_dir,
            device=args.device,
            fov_scale_zyx=args.fov_scale,
            show_progress=args.show_progress,
            jsense_config=jsense_config,
        )
        print(f"Prepared data in {artifacts.directory}")
        print(f"Manifest: {artifacts.manifest}")
        return 0

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
