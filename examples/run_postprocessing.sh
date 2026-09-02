#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:?Usage: run_postprocessing.sh OUTPUT_DIR [STITCH_OPTIONS...]}"
shift

exec toporecon stitch "$OUTPUT_DIR" "$@"
