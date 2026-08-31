# Multi-Node/Multi-GPU-Accelerated Respiratory Motion-Resolved Reconstruction of 3D non-Cartesian mGRE MRI

This repository contains the reconstruction code and dataset associated with
the following paper:

> Zhang et al. Large-Scale Distributed GPU-Accelerated Respiratory Motion-Resolved Reconstruction of 3D non-Cartesian mGRE MRI. <em>International Conference on Medical Image Computing and Computer Assisted Interventions (MICCAI) 2026</em>.

Contact: Chao Zhang (<Chao.Zhang.1@stonybrook.edu>)

## Dataset

**3D Multi-Echo Cones Liver MRI:** [![DOI: 10.5281/zenodo.21414404](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21414404-blue.svg)](https://doi.org/10.5281/zenodo.21414404)

Download and extract the dataset before running the preparation and
reconstruction commands below. Keep the downloaded raw data, prepared data,
and reconstruction outputs in separate directories.

## Reconstruction methods

The repository includes three topology-aware PDHG reconstruction methods:

- **TVM** applies total-variation regularization along the respiratory-motion
  dimension (https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.29779).
- **TVME** couples motion and echo dimensions with total-variation
  regularization (https://papers.miccai.org/miccai-2025/paper/3082_paper.pdf).
- **TVMW** combines motion regularization with a db1 echo wavelet and a db6
  spatial wavelet (https://doi.org/10.1016/j.media.2025.103532).

The multi-node implementation uses a two-dimensional motion/echo node grid.
Within each node, coils are distributed across the local GPUs. NCCL handles
intra-node reductions, while CUDA-aware MPI handles communication between
nodes.

## Environment

Create the supplied Conda environment and install the package from the
repository root:

```bash
conda env create -f environment.yml
conda activate toporecon-miccai
python -m pip install -e .
```

CUDA and CuPy must be compatible with the GPUs and drivers on the target
system. TVMW additionally requires PyTorch and PTWT; both are included in
`environment.yml`.

## Data preparation

Set paths for the extracted dataset and a persistent preparation directory:

```bash
export RAW_DATA_DIR=/path/to/raw-data
export PREPARED_DIR=/path/to/prepared-data
```

Validate the raw data and run RESP and JSENSE once:

```bash
toporecon inspect "$RAW_DATA_DIR"

toporecon prepare "$RAW_DATA_DIR" \
  --work-dir "$PREPARED_DIR" \
  --device 0 \
  --fov-scale 1.0 1.0 1.0 \
  --show-progress
```

The preparation directory will contain:

```text
mps.hdr
mps.cfl
resp.hdr
resp.cfl
manifest.json
```

RESP and JSENSE can also be run separately:

```bash
toporecon resp "$RAW_DATA_DIR" --work-dir "$PREPARED_DIR"

toporecon jsense "$RAW_DATA_DIR" \
  --work-dir "$PREPARED_DIR" \
  --device 0 \
  --fov-scale 1.0 1.0 1.0 \
  --show-progress
```

Prepared files must come from the same raw dataset used for reconstruction.
The `--fov-scale Z Y X` values used by JSENSE must exactly match those used by
TVM, TVME, or TVMW.

### JSENSE reproducibility

GPU JSENSE gridding performs a dense set of `atomicAdd` operations. Because
the parallel accumulation order is not deterministic, repeated gridding runs
can differ by approximately `1e-6`; after multiple JSENSE iterations, these
differences may grow to approximately `1e-3` in the sensitivity maps.

For reproducible comparisons, run JSENSE once and reuse the same
`mps.hdr/.cfl` for every reconstruction. CPU JSENSE avoids the GPU
`atomicAdd` path but is very slow:

```bash
toporecon jsense "$RAW_DATA_DIR" \
  --work-dir "$PREPARED_DIR" \
  --device -1 \
  --fov-scale 1.0 1.0 1.0
```

## Single-process reconstruction

The wrappers in `examples/` provide short single-process commands:

```bash
examples/run_tvm.sh  "$RAW_DATA_DIR" "$PREPARED_DIR" /path/to/output/tvm
examples/run_tvme.sh "$RAW_DATA_DIR" "$PREPARED_DIR" /path/to/output/tvme
examples/run_tvmw.sh "$RAW_DATA_DIR" "$PREPARED_DIR" /path/to/output/tvmw
```

The equivalent TVME command is:

```bash
mpiexec -n 1 toporecon reconstruct --algorithm tvme -- \
  --prepared-dir "$PREPARED_DIR" \
  --output-dir /path/to/output/tvme \
  --nufft-backend sigpy \
  --num-bins 6 \
  --motion-groups 1 \
  --echo-groups 1 \
  --fov-scale 1.0 1.0 1.0 \
  "$RAW_DATA_DIR" reconstruction
```

Replace `tvme` with `tvm` or `tvmw` as needed. Arguments after `--` are passed
to the selected algorithm.

## Multi-node reconstruction on DeltaAI

The supplied Slurm templates use four nodes, sixteen MPI ranks, one rank per
GPU, and a `2 x 2` motion/echo node grid:

- `examples/deltaai_tvm.slurm`
- `examples/deltaai_tvme.slurm`
- `examples/deltaai_tvmw.slurm`

The templates default to ten iterations and a three-minute walltime for smoke
testing. A full reconstruction should explicitly set a longer walltime and the
intended iteration count.

From the repository root, define the shared paths and allocation:

```bash
export TOPORECON_DIR="$PWD"
export PY="$(command -v python)"
export INPUT_DIR="$RAW_DATA_DIR"
export PREPARED_DIR=/path/to/prepared-data
export RUN_ROOT=/path/to/reconstruction-runs
export ACCOUNT=your_allocation
export WALLTIME=HH:MM:SS

mkdir -p "$RUN_ROOT"
```

Submit full TVM, TVME, and TVMW reconstructions as separate jobs:

```bash
sbatch -A "$ACCOUNT" -t "$WALLTIME" \
  --export=ALL,OUT_DIR="$RUN_ROOT/tvm",MAX_ITER=300 \
  examples/deltaai_tvm.slurm

sbatch -A "$ACCOUNT" -t "$WALLTIME" \
  --export=ALL,OUT_DIR="$RUN_ROOT/tvme",MAX_ITER=300 \
  examples/deltaai_tvme.slurm

sbatch -A "$ACCOUNT" -t "$WALLTIME" \
  --export=ALL,OUT_DIR="$RUN_ROOT/tvmw",MAX_ITER=300 \
  examples/deltaai_tvmw.slurm
```

Choose `WALLTIME` from the measured smoke-test runtime for the dataset and
allocation. Keep `MOTION_GROUPS * ECHO_GROUPS` equal to the number of MPI
nodes. The templates expose regularization and runtime settings through
environment variables:

- **TVM:** `LAMBDA_MOTION`, `ACCELERATION`, and `REDUCE_PER_BIN`.
- **TVME:** `LAMBDA_MOTION`, `LAMBDA_ECHO`, and `ACCELERATION`.
- **TVMW:** `LAMBDA_MOTION`, `LAMBDA_ECHO_WAVELET`, and
  `LAMBDA_SPATIAL_WAVELET`.
- **Common:** `NUM_BINS`, `READOUT_FRACTION`, `MAX_ITER`, `FOV_SCALE_Z`,
  `FOV_SCALE_Y`, `FOV_SCALE_X`, `CROP_TO_ORIGINAL`, and `USE_FP16_COMM`.

Reconstruction outputs, `run_manifest.json`, and per-rank GPU/CPU monitoring
files are written below each `OUT_DIR`.

## Tests

Run the CPU unit and migration checks with:

```bash
PYTHONPATH=src python -m unittest discover -s tests/unit -p 'test_*.py'
```
