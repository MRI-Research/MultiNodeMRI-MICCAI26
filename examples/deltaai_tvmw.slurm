#!/usr/bin/env bash
#SBATCH -p ghx4
#SBATCH -N 4
#SBATCH -n 16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -J toporecon-tvmw-16g-2e2b
#SBATCH -o %x.%j.log
#SBATCH -e %x.%j.log

set -euo pipefail

# TVMW reconstruction reference for four DeltaAI nodes. Set a tested
# walltime with `sbatch --time=HH:MM:SS`; every value below can be overridden
# through `--export=ALL,NAME=value`.

# === Modules / environment ===
module reset
module unload cuda cudatoolkit 2>/dev/null || true
module load cudatoolkit/25.5_12.9
module load cudnn/9.3.0.75
module load cray-mpich

export CUDA_PATH="${CUDA_HOME}"
export CUDAToolkit_ROOT="${CUDA_PATH}"
export CPATH="${CUDA_PATH}/include${CPATH:+:${CPATH}}"
export LD_LIBRARY_PATH="${CUDA_PATH}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# NCCL is used only for the four GPUs within each node. Cross-node
# communication is handled by CUDA-aware Cray MPI/CXI.
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
unset NCCL_SOCKET_IFNAME NCCL_NET_PLUGIN AWS_OFI_NCCL_NET_IFACE FI_PROVIDER

export MPICH_GPU_SUPPORT_ENABLED=1
export LD_PRELOAD="/opt/cray/pe/lib64/libmpi_gtl_cuda.so.0${LD_PRELOAD:+:${LD_PRELOAD}}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export MPICH_OFI_NIC_VERBOSE=0
export MPICH_ENV_DISPLAY=0

# === User paths ===
# Pass the allocation with `sbatch -A ACCOUNT`. INPUT_DIR, PREPARED_DIR, and
# OUT_DIR are required so this public template never embeds private data paths.
TOPORECON_DIR="${TOPORECON_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PY="${PY:-python}"
INPUT_DIR="${INPUT_DIR:?Set INPUT_DIR to the extracted Zenodo dataset directory.}"
PREPARED_DIR="${PREPARED_DIR:?Set PREPARED_DIR to the directory containing mps and resp.}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR to a writable reconstruction output directory.}"
OUTPUT_STEM="${OUTPUT_STEM:-imout}"

export PYTHONPATH="${TOPORECON_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

# === TVMW parameters ===
NUM_BINS="${NUM_BINS:-6}"
READOUT_FRACTION="${READOUT_FRACTION:-0.98}"
LAMBDA_MOTION="${LAMBDA_MOTION:-1e-5}"
LAMBDA_ECHO_WAVELET="${LAMBDA_ECHO_WAVELET:-1e-5}"
LAMBDA_SPATIAL_WAVELET="${LAMBDA_SPATIAL_WAVELET:-1e-5}"
TOL="${TOL:-1e-3}"
MAX_ITER="${MAX_ITER:-300}"
MOTION_GROUPS="${MOTION_GROUPS:-2}"
ECHO_GROUPS="${ECHO_GROUPS:-2}"

FOV_SCALE_Z="${FOV_SCALE_Z:-1.0}"
FOV_SCALE_Y="${FOV_SCALE_Y:-1.0}"
FOV_SCALE_X="${FOV_SCALE_X:-1.0}"
CROP_TO_ORIGINAL="${CROP_TO_ORIGINAL:-0}"
USE_FP16_COMM="${USE_FP16_COMM:-0}"

# === Validate the launch before starting sixteen MPI ranks ===
fail() {
  echo "ERROR: $*" >&2
  exit 2
}

command -v "${PY}" >/dev/null 2>&1 || fail "Python is not executable: ${PY}"
[[ -f "${TOPORECON_DIR}/pyproject.toml" ]] || fail \
  "TOPORECON_DIR is not a release checkout: ${TOPORECON_DIR}"
[[ -f "${TOPORECON_DIR}/src/toporecon/algorithms/tvmw.py" ]] || fail \
  "TVMW module is missing below TOPORECON_DIR."
"${PY}" -c \
  "import cupy, mpi4py, sigpy, toporecon; from toporecon.algorithms import tvmw" \
  || fail "The selected Python environment cannot import the release GPU/MPI dependencies."

for base in ksp ktraj dens; do
  [[ -f "${INPUT_DIR}/${base}.hdr" ]] || fail "Missing ${INPUT_DIR}/${base}.hdr"
  [[ -f "${INPUT_DIR}/${base}.cfl" ]] || fail "Missing ${INPUT_DIR}/${base}.cfl"
done
# tr.txt is required while preparing resp, but reconstruction consumes the
# prepared resp file and intentionally does not require tr.txt at this stage.
for name in imageDim.txt voxelSize.txt; do
  [[ -f "${INPUT_DIR}/${name}" ]] || fail "Missing ${INPUT_DIR}/${name}"
done
for base in mps resp; do
  [[ -f "${PREPARED_DIR}/${base}.hdr" ]] || fail \
    "Missing ${PREPARED_DIR}/${base}.hdr; run the preparation stage first."
  [[ -f "${PREPARED_DIR}/${base}.cfl" ]] || fail \
    "Missing ${PREPARED_DIR}/${base}.cfl; run the preparation stage first."
done

if (( MOTION_GROUPS * ECHO_GROUPS != SLURM_JOB_NUM_NODES )); then
  fail "MOTION_GROUPS * ECHO_GROUPS must equal SLURM_JOB_NUM_NODES."
fi

mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/run.${SLURM_JOB_ID}.log"

# === Job header ===
{
  echo "=== Job info ==="
  echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
  echo "Job name = ${SLURM_JOB_NAME}"
  echo "Nodes = ${SLURM_NODELIST:-unknown}"
  echo "Tasks = ${SLURM_NTASKS:-16}"
  echo "Tasks per node = ${SLURM_NTASKS_PER_NODE:-4}"
  echo "GPUs per node = ${SLURM_GPUS_PER_NODE:-4}"
  echo "Start time = $(date)"
  echo "Release directory = ${TOPORECON_DIR}"
  echo "Release commit = $(git -C "${TOPORECON_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "Python = ${PY}"
  echo "Input directory = ${INPUT_DIR}"
  echo "Prepared directory = ${PREPARED_DIR}"
  echo "Output directory = ${OUT_DIR}"
  echo "================"
  module list
  echo "================"
  echo "MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED}"
  echo "LD_PRELOAD=${LD_PRELOAD}"
} 2>&1 | tee -a "${LOG}"

# === Run release TVMW ===
CMD=(
  srun --kill-on-bad-exit=1 --wait=0
  -N 4 -n 16 --ntasks-per-node=4 --gpus-per-node=4
  --chdir="${OUT_DIR}"
  "${PY}" -m toporecon reconstruct --algorithm tvmw --
  --multi-gpu
  --nufft-backend sigpy
  --num-bins "${NUM_BINS}"
  --readout-fraction "${READOUT_FRACTION}"
  --lambda-motion "${LAMBDA_MOTION}"
  --lambda-echo-wavelet "${LAMBDA_ECHO_WAVELET}"
  --lambda-spatial-wavelet "${LAMBDA_SPATIAL_WAVELET}"
  --tol "${TOL}"
  --max-iter "${MAX_ITER}"
  --echo-groups "${ECHO_GROUPS}"
  --motion-groups "${MOTION_GROUPS}"
  --l2-coupling
  --show-progress
  --verbose
  --fov-scale "${FOV_SCALE_Z}" "${FOV_SCALE_Y}" "${FOV_SCALE_X}"
  --prepared-dir "${PREPARED_DIR}"
  --output-dir "${OUT_DIR}"
)

if [[ "${CROP_TO_ORIGINAL}" == "1" ]]; then
  CMD+=(--crop-to-original)
fi
if [[ "${USE_FP16_COMM}" == "1" ]]; then
  CMD+=(--fp16-communication)
fi

CMD+=("${INPUT_DIR}" "${OUTPUT_STEM}")

echo "Command: ${CMD[*]}" | tee -a "${LOG}"
set +e
"${CMD[@]}" 2>&1 | tee -a "${LOG}"
rc=${PIPESTATUS[0]}
set -e

echo "End time: $(date)" | tee -a "${LOG}"
echo "Exit code: ${rc}" | tee -a "${LOG}"
exit "${rc}"
