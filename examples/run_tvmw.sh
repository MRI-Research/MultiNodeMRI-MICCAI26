#!/usr/bin/env bash
set -euo pipefail

# Replace these paths and the MPI process count for the target system.
RAW_DATA_DIR="${1:?usage: run_tvmw.sh RAW_DATA_DIR PREPARED_DIR OUTPUT_DIR}"
PREPARED_DIR="${2:?usage: run_tvmw.sh RAW_DATA_DIR PREPARED_DIR OUTPUT_DIR}"
OUTPUT_DIR="${3:?usage: run_tvmw.sh RAW_DATA_DIR PREPARED_DIR OUTPUT_DIR}"

mpiexec -n 1 toporecon reconstruct --algorithm tvmw -- \
  --prepared-dir "$PREPARED_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --nufft-backend sigpy \
  --num-bins 6 \
  --motion-groups 1 \
  --echo-groups 1 \
  "$RAW_DATA_DIR" \
  reconstruction
