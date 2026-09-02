#!/usr/bin/env bash
#SBATCH -p ghx4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J toporecon-prepare-1g
#SBATCH -o %x.%j.log
#SBATCH -e %x.%j.log

set -euo pipefail

# Run RESP followed by single-GPU JSENSE. Submit from the release root with:
#   sbatch --export=ALL,INPUT_DIR=/path/to/raw,PREPARED_DIR=/path/to/prepared \
#     examples/deltaai_prepare.slurm

# === Modules / environment ===
module reset
module unload cuda cudatoolkit 2>/dev/null || true
module load cudatoolkit/25.5_12.9
module load cudnn/9.3.0.75

export CUDA_PATH="${CUDA_HOME}"
export CUDAToolkit_ROOT="${CUDA_PATH}"
export CPATH="${CUDA_PATH}/include${CPATH:+:${CPATH}}"
export LD_LIBRARY_PATH="${CUDA_PATH}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# === Paths and preparation parameters ===
TOPORECON_DIR="${TOPORECON_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PY="${PY:-python}"
INPUT_DIR="${INPUT_DIR:?Set INPUT_DIR to the raw dataset directory.}"
PREPARED_DIR="${PREPARED_DIR:?Set PREPARED_DIR to a writable preparation directory.}"

DEVICE="${DEVICE:-0}"
READOUT_FRACTION="${READOUT_FRACTION:-0.95}"
COORDINATE_SCALING="${COORDINATE_SCALING:-legacy_max}"
FOV_SCALE_Z="${FOV_SCALE_Z:-1.0}"
FOV_SCALE_Y="${FOV_SCALE_Y:-1.0}"
FOV_SCALE_X="${FOV_SCALE_X:-1.0}"
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"
OVERWRITE_PREPARED="${OVERWRITE_PREPARED:-0}"

export PYTHONPATH="${TOPORECON_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

command -v "${PY}" >/dev/null 2>&1 || fail "Python is not executable: ${PY}"
[[ -f "${TOPORECON_DIR}/pyproject.toml" ]] || fail \
  "TOPORECON_DIR is not a release checkout: ${TOPORECON_DIR}"
[[ -f "${TOPORECON_DIR}/src/toporecon/cli.py" ]] || fail \
  "toporecon CLI is missing below TOPORECON_DIR."
"${PY}" -c "import cupy, scipy, sigpy, toporecon" || fail \
  "The selected Python environment cannot import the preparation dependencies."

for base in ksp ktraj dens; do
  [[ -f "${INPUT_DIR}/${base}.hdr" ]] || fail "Missing ${INPUT_DIR}/${base}.hdr"
  [[ -f "${INPUT_DIR}/${base}.cfl" ]] || fail "Missing ${INPUT_DIR}/${base}.cfl"
done
for name in imageDim.txt voxelSize.txt tr.txt; do
  [[ -f "${INPUT_DIR}/${name}" ]] || fail "Missing ${INPUT_DIR}/${name}"
done

[[ "${SHOW_PROGRESS}" == "0" || "${SHOW_PROGRESS}" == "1" ]] || fail \
  "SHOW_PROGRESS must be 0 or 1."
[[ "${OVERWRITE_PREPARED}" == "0" || "${OVERWRITE_PREPARED}" == "1" ]] || fail \
  "OVERWRITE_PREPARED must be 0 or 1."
[[ "${DEVICE}" == "0" ]] || fail \
  "DEVICE must be 0 because this job exposes one logical GPU."

mkdir -p "${PREPARED_DIR}"
[[ -w "${PREPARED_DIR}" ]] || fail \
  "PREPARED_DIR is not writable: ${PREPARED_DIR}"

if [[ "${OVERWRITE_PREPARED}" != "1" ]]; then
  for name in resp.hdr resp.cfl mps.hdr mps.cfl manifest.json; do
    [[ ! -e "${PREPARED_DIR}/${name}" ]] || fail \
      "${PREPARED_DIR}/${name} already exists; reuse it or set OVERWRITE_PREPARED=1."
  done
fi

LOG="${PREPARED_DIR}/prepare.${SLURM_JOB_ID}.log"

CMD=(
  srun --kill-on-bad-exit=1 --wait=0
  -N 1 -n 1 --ntasks-per-node=1 --gpus-per-node=1
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}"
  --gpu-bind=closest
  --mem-bind=local
  --chdir="${PREPARED_DIR}"
  "${PY}" -m toporecon --verbose prepare "${INPUT_DIR}"
  --work-dir "${PREPARED_DIR}"
  --device "${DEVICE}"
  --readout-fraction "${READOUT_FRACTION}"
  --coordinate-scaling "${COORDINATE_SCALING}"
  --fov-scale "${FOV_SCALE_Z}" "${FOV_SCALE_Y}" "${FOV_SCALE_X}"
)

if [[ "${SHOW_PROGRESS}" == "1" ]]; then
  CMD+=(--show-progress)
fi

{
  echo "=== Preparation job ==="
  echo "Job ID = ${SLURM_JOB_ID}"
  echo "Node = ${SLURM_NODELIST:-unknown}"
  echo "Start time = $(date)"
  echo "Release directory = ${TOPORECON_DIR}"
  echo "Release commit = $(git -C "${TOPORECON_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "Python = ${PY}"
  echo "Input directory = ${INPUT_DIR}"
  echo "Prepared directory = ${PREPARED_DIR}"
  echo "Device = ${DEVICE}"
  echo "Readout fraction = ${READOUT_FRACTION}"
  echo "Coordinate scaling = ${COORDINATE_SCALING}"
  echo "FOV scale (z,y,x) = ${FOV_SCALE_Z} ${FOV_SCALE_Y} ${FOV_SCALE_X}"
  module list
  echo "Command: ${CMD[*]}"
} 2>&1 | tee -a "${LOG}"

set +e
"${CMD[@]}" 2>&1 | tee -a "${LOG}"
rc=${PIPESTATUS[0]}
set -e

if (( rc == 0 )); then
  for name in resp.hdr resp.cfl mps.hdr mps.cfl manifest.json; do
    [[ -s "${PREPARED_DIR}/${name}" ]] || {
      echo "ERROR: preparation did not produce ${PREPARED_DIR}/${name}" | tee -a "${LOG}"
      rc=2
      break
    }
  done
fi

echo "End time: $(date)" | tee -a "${LOG}"
echo "Exit code: ${rc}" | tee -a "${LOG}"
exit "${rc}"
