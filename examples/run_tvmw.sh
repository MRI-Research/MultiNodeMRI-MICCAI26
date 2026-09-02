#!/usr/bin/env bash
set -euo pipefail

# Override any value below through the environment when needed.
RAW_DATA_DIR="${1:?Usage: run_tvmw.sh RAW_DATA_DIR PREPARED_DIR OUTPUT_DIR}"
PREPARED_DIR="${2:?Usage: run_tvmw.sh RAW_DATA_DIR PREPARED_DIR OUTPUT_DIR}"
OUTPUT_DIR="${3:?Usage: run_tvmw.sh RAW_DATA_DIR PREPARED_DIR OUTPUT_DIR}"
MPI_RANKS="${MPI_RANKS:-1}"
OUTPUT_STEM="${OUTPUT_STEM:-imout}"

mpiexec -n "$MPI_RANKS" toporecon reconstruct --algorithm tvmw -- \
  --prepared-dir "$PREPARED_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --device "${DEVICE:-0}" \
  --nufft-backend "${NUFFT_BACKEND:-sigpy}" \
  --readout-fraction "${READOUT_FRACTION:-0.98}" \
  --num-bins "${NUM_BINS:-6}" \
  --motion-groups "${MOTION_GROUPS:-1}" \
  --echo-groups "${ECHO_GROUPS:-1}" \
  --fov-scale "${FOV_SCALE_Z:-1.0}" "${FOV_SCALE_Y:-1.0}" "${FOV_SCALE_X:-1.0}" \
  --lambda-motion "${LAMBDA_MOTION:-1e-5}" \
  --lambda-echo-wavelet "${LAMBDA_ECHO_WAVELET:-1e-5}" \
  --lambda-spatial-wavelet "${LAMBDA_SPATIAL_WAVELET:-1e-5}" \
  --tol "${TOL:-1e-3}" \
  --max-iter "${MAX_ITER:-300}" \
  --l2-coupling \
  --show-progress \
  "$RAW_DATA_DIR" "$OUTPUT_STEM"

# When a node launches more than one MPI rank, add --multi-gpu after the `--`.
