# Multi-Node/Multi-GPU-Accelerated Respiratory Motion-Resolved Reconstruction of 3D non-Cartesian mGRE MRI

This repository contains the reconstruction code and dataset associated with
the following paper:

> Zhang et al. Large-Scale Distributed GPU-Accelerated Respiratory Motion-Resolved Reconstruction of 3D non-Cartesian mGRE MRI. <em>International Conference on Medical Image Computing and Computer Assisted Interventions (MICCAI) 2026</em>.

Contact: Chao Zhang (<Chao.Zhang.1@stonybrook.edu>)

## Dataset

**3D Multi-Echo Cones Liver MRI:** [![DOI: 10.5281/zenodo.21414404](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21414404-blue.svg)](https://doi.org/10.5281/zenodo.21414404)

Download and extract the dataset before running the preparation and
reconstruction commands below. Keep the downloaded raw data, prepared data,
and reconstruction outputs in separate directories. The raw data directory
must contain:

```text
ksp.hdr        ksp.cfl
ktraj.hdr      ktraj.cfl
dens.hdr       dens.cfl
imageDim.txt   voxelSize.txt   tr.txt
```

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

The main dependencies are Python 3.12, NumPy 1.26, SciPy, CuPy 13.2,
SigPy 0.1.27, MPI/`mpi4py`, and tqdm. TVMW additionally requires PyTorch and
PTWT. CUDA, CuPy, MPI, and the GPU driver must be compatible with the target
system; multi-node runs require a CUDA-aware MPI installation.

## Data preprocessing

Set paths for the extracted dataset and a persistent preparation directory,
then run respiratory self-gating and JSENSE once:

```bash
export RAW_DATA_DIR=/path/to/raw-data
export PREPARED_DIR=/path/to/prepared-data

toporecon inspect "$RAW_DATA_DIR"
toporecon prepare "$RAW_DATA_DIR" \
  --work-dir "$PREPARED_DIR" \
  --device 0 \
  --fov-scale 1.0 1.0 1.0 \
  --show-progress
```

This writes `resp.hdr/.cfl`, `mps.hdr/.cfl`, and `manifest.json` to
`PREPARED_DIR`.

GPU JSENSE can show small run-to-run floating-point differences because its
gridding uses atomic additions. For reproducible comparisons, reconstruct the
sensitivity maps once and keep reusing the same prepared directory. Run
JSENSE again only when the raw data, FOV, or another preprocessing setting
changes. The preprocessing and reconstruction values of
`--fov-scale Z Y X` must match.

## Reconstruction

Single- and multi-node reconstruction use the same command. The example below
runs TVME; replace `tvme` and the lambda options to use TVM or TVMW. Use one
MPI rank per GPU and replace `mpiexec` with the site's corresponding launcher
when needed.

```bash
export OUTPUT_DIR=/path/to/output
export MPI_RANKS=1

mpiexec -n "$MPI_RANKS" toporecon reconstruct --algorithm tvme -- \
  --multi-gpu \
  --prepared-dir "$PREPARED_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --nufft-backend sigpy \
  --readout-fraction 0.98 \
  --num-bins 6 \
  --motion-groups 1 \
  --echo-groups 1 \
  --fov-scale 1.0 1.0 1.0 \
  --lambda-motion 1e-4 \
  --lambda-echo 1e-5 \
  --l2-coupling \
  --max-iter 300 \
  --show-progress \
  "$RAW_DATA_DIR" imout
```

`motion-groups * echo-groups` must equal the number of physical compute
nodes, not the number of MPI ranks. A single-node run uses a `1 x 1` grid; a
four-node run can use `2 x 2`. Within each node, MPI ranks divide the receiver
coils across the local GPUs.

### Main parameters

| Parameter | Meaning |
| --- | --- |
| `RAW_DATA_DIR` | Raw k-space, trajectory, density, and scan metadata directory. |
| `--prepared-dir` | Directory containing the reusable `mps` and `resp` files. |
| `--output-dir` | Directory for reconstruction shards and `run_manifest.json`. |
| `imout` | Output filename stem. |
| `--fov-scale Z Y X` | Per-axis FOV scale; each value must be at least 1 and must match JSENSE. |
| `--num-bins` | Number of respiratory motion phases; default `6`. |
| `--motion-groups` | Number of node-grid rows used to divide motion bins. |
| `--echo-groups` | Number of node-grid columns used to divide echoes. |
| Lambda options | Regularization strengths; larger values apply stronger regularization. |
| `tol` | Relative L2-change stopping threshold. The current reconstruction entry points use the fixed value `1e-3`. |
| `--max-iter` | Maximum number of PDHG iterations; default `300`. |
| `--readout-fraction` | Fraction of readout samples used for reconstruction; default `0.98`. |
| `--multi-gpu` | Assign each node-local MPI rank to the GPU with the same local rank. |
| `--l2-coupling` | Couple motion regularization across echoes. |

The algorithm-specific lambda options and defaults are:

| Algorithm | Lambda options |
| --- | --- |
| TVM | `--lambda-motion 1e-6` |
| TVME | `--lambda-motion 1e-4`, `--lambda-echo 1e-5` |
| TVMW | `--lambda-motion 1e-6`, `--lambda-echo-wavelet 1e-6`, `--lambda-spatial-wavelet 1e-6` |

Each physical node writes one shard named
`imout_e<ECHO_START>-<ECHO_END>_m<MOTION_START>-<MOTION_END>.hdr/.cfl`;
the end indices are inclusive.

## Post-processing

After the MPI job finishes, assemble all grid shards into one image:

```bash
python scripts/stitch_shards.py "$OUTPUT_DIR"
```

The script reads `run_manifest.json`, places each shard according to the
motion and echo ranges in its filename, and writes `imout.hdr` and
`imout.cfl`. The same command works for single- and multi-node output. The
assembled CFL layout is `[motion, 1, 1, echo, 1, 1, z, y, x]`.
