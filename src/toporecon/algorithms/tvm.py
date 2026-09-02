#!/usr/bin/env python3
"""Motion total-variation reconstruction on a distributed node grid."""

import argparse
import json
import logging
import os
import time

import numpy as np
import sigpy as sp
from mpi4py import MPI
from tqdm.auto import tqdm

from toporecon.data import cfl
from toporecon.data.dataset import KSpaceLayout
from toporecon.distributed.runtime import (
    GpuComm,
    IntraNodeGroup,
    NodeGrid2D,
    make_comm_tagger,
)
from toporecon.nufft_backend import create_nufft_backend


try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


def split_coils(C_total, n):
    return np.array_split(np.arange(C_total), n)


def max_abs_cplx_chunked(x, chunk_elems=8_000_000):
    x = x.reshape(-1)
    maximum = 0.0
    for start in range(0, x.size, chunk_elems):
        chunk_maximum = np.max(np.abs(x[start : start + chunk_elems]))
        if chunk_maximum > maximum:
            maximum = float(chunk_maximum)
    return maximum


def center_crop_zyx(arr, out_shape_zyx):
    Nz0, Ny0, Nx0 = out_shape_zyx
    Nz, Ny, Nx = arr.shape[-3:]
    if Nz0 > Nz or Ny0 > Ny or Nx0 > Nx:
        raise ValueError(
            f"Requested crop {out_shape_zyx} larger than input {(Nz, Ny, Nx)}"
        )

    z0 = (Nz - Nz0) // 2
    y0 = (Ny - Ny0) // 2
    x0 = (Nx - Nx0) // 2
    return arr[..., z0 : z0 + Nz0, y0 : y0 + Ny0, x0 : x0 + Nx0]


class TvmReconstructor:
    """Motion-TV PDHG reconstruction over local motion/echo/coil shards."""

    def __init__(
        self,
        ksp,
        coord,
        dcf,
        mps,
        resp,
        dual_q,
        B_local,
        l2_coupling=False,
        lamda=1e-5,
        sigma=0.01,
        tau=0.01,
        max_iter=300,
        tol=1e-3,
        margin=10,
        device=sp.cpu_device,
        B_total=None,
        b0=None,
        b1=None,
        motion_left_peer=None,
        motion_right_peer=None,
        bin_edges=None,
        world_comm=None,
        group_comm=None,
        leader_comm=None,
        nccl_group=None,
        dist_grid=None,
        tagger=None,
        show_pbar=True,
        reduce_per_bin=False,
        nufft_backend="sigpy",
    ):
        self.mps = mps
        self.B = int(B_local)
        self.C = len(mps)
        self.E = ksp.shape[0]

        self.B_total = int(B_total) if B_total is not None else self.B
        self.b0 = int(b0) if b0 is not None else 0
        self.b1 = int(b1) if b1 is not None else self.b0 + self.B
        self.is_motion_first = self.b0 == 0
        self.is_motion_last = self.b1 == self.B_total
        self.motion_left_peer = motion_left_peer
        self.motion_right_peer = motion_right_peer

        self.lamda = lamda
        self.sigma = sigma
        self.tau = tau
        self.l2_coupling = l2_coupling
        self.max_iter = max_iter
        self.tol = tol
        self.dist_grid = dist_grid
        self.tagger = tagger
        self.reduce_per_bin = reduce_per_bin

        self.device = sp.Device(device)
        self.xp = self.device.xp
        self.world_comm = world_comm
        self.group_comm = group_comm
        self.leader_comm = leader_comm
        self.nccl_group = nccl_group

        self.need_coil_reduce = (
            self.group_comm is not None and self.group_comm.Get_size() > 1
        )
        self.use_nccl_reduce = (
            self.need_coil_reduce
            and CUPY_AVAILABLE
            and self.nccl_group is not None
            and getattr(self.nccl_group, "stream", None) is not None
        )
        self.group_rank = (
            self.group_comm.Get_rank() if self.group_comm is not None else 0
        )
        self.show_pbar = show_pbar
        if self.world_comm is not None:
            self.show_pbar = show_pbar and self.world_comm.rank == 0

        self.img_shape = list(mps.shape[-3:])

        if bin_edges is None:
            bins = np.percentile(
                resp,
                np.linspace(0 + margin, 100 - margin, self.B_total + 1),
            )
        else:
            bins = np.asarray(bin_edges)
            assert bins.ndim == 1 and bins.size == self.B_total + 1, (
                f"bin_edges must have shape ({self.B_total + 1},), "
                f"got {bins.shape}"
            )

        self.bksp = []
        self.bcoord = []
        self.bdcf = []
        self.dual_q = []
        for global_bin in range(self.b0, self.b1):
            idx = (resp >= bins[global_bin]) & (resp < bins[global_bin + 1])
            self.bksp.append(sp.to_device(ksp[:, :, idx], self.device))
            self.bcoord.append(sp.to_device(coord[idx], self.device))
            self.bdcf.append(sp.to_device(dcf[idx], self.device))
            self.dual_q.append(sp.to_device(dual_q[:, :, idx], self.device))

        self.nufft = create_nufft_backend(
            nufft_backend,
            img_shape=self.img_shape,
            coordinates=self.bcoord,
            device=self.device,
        )

        self.mps_dev = sp.to_device(self.mps, self.device)
        self.mps_conj = self.xp.conj(self.mps_dev)

    def pdhg_init(self):
        u_bar = self.xp.zeros(
            [self.B, self.E] + self.img_shape,
            dtype=self.mps.dtype,
        )
        p_m = self.xp.zeros_like(u_bar)
        return u_bar, p_m, self.dual_q

    def _coil_allreduce_in_place(self, array):
        if not self.need_coil_reduce:
            return

        if self.use_nccl_reduce and isinstance(array, cp.ndarray):
            computation_stream = cp.cuda.get_current_stream()
            communication_stream = self.nccl_group.stream
            ready = cp.cuda.Event()
            ready.record(computation_stream)
            communication_stream.wait_event(ready)
            with communication_stream:
                self.nccl_group.alreduce_sum_(array)
            complete = cp.cuda.Event()
            complete.record(communication_stream)
            computation_stream.wait_event(complete)
            return

        host_buffer = (
            cp.asnumpy(array)
            if CUPY_AVAILABLE and isinstance(array, cp.ndarray)
            else np.asarray(array)
        )
        self.group_comm.Allreduce(MPI.IN_PLACE, host_buffer, op=MPI.SUM)
        if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
            array.set(host_buffer)
        else:
            array[...] = host_buffer

    def _accumulate_data_adjoint(self, result, q_data, bin_index):
        for coil in range(self.C):
            for echo in range(self.E):
                result[echo] += (
                    self.nufft.adjoint(
                        self.bdcf[bin_index] * q_data[bin_index][echo][coil],
                        bin_index,
                    )
                    * self.mps_conj[coil]
                )

    def pdhg(self, u_bar, u_now, p_m, q_data, iteration):
        u_old = self.xp.copy(u_now)

        ghost_right_first = self.xp.empty_like(u_bar[0])
        ghost_right_first = self.xp.ascontiguousarray(ghost_right_first)
        sendbuf = self.xp.ascontiguousarray(u_bar[0])
        motion_delta_tag = self.tagger(
            iteration,
            axis="motion",
            kind="delta",
        )
        motion_delta_requests = self.dist_grid.halo_exchange(
            sendbuf,
            ghost_right_first,
            send_peer=self.motion_left_peer,
            recv_peer=self.motion_right_peer,
            tag=motion_delta_tag,
        )

        for bin_index in range(self.B):
            residual = self.xp.zeros_like(self.dual_q[bin_index])
            for coil in range(self.C):
                mps_coil = self.mps_dev[coil]
                for echo in range(self.E):
                    residual[echo, coil] = (
                        self.nufft.forward(
                            u_bar[bin_index, echo] * mps_coil,
                            bin_index,
                        )
                        - self.bksp[bin_index][echo][coil]
                    )
            q_data[bin_index] = (
                q_data[bin_index] + self.sigma * residual
            ) / (1 + self.sigma)

        motion_difference = self.xp.zeros_like(u_bar)
        if self.B > 1:
            motion_difference[:-1] = u_bar[1:] - u_bar[:-1]
        MPI.Request.Waitall(motion_delta_requests)
        if not self.is_motion_last:
            motion_difference[-1] = ghost_right_first - u_bar[-1]
        else:
            motion_difference[-1] = 0

        p_m = p_m + self.sigma * motion_difference

        # Match the verified TVME L2 communication structure: sum the local
        # echoes once across echo-axis node leaders, then broadcast that one
        # result to the coil ranks on each node.
        if self.l2_coupling:
            magnitude_sq = self.xp.sum(
                self.xp.abs(p_m) ** 2,
                axis=1,
                keepdims=True,
            )
            magnitude_sq = self.xp.ascontiguousarray(magnitude_sq)

            request = None
            if self.leader_comm is not None:
                if CUPY_AVAILABLE and isinstance(magnitude_sq, cp.ndarray):
                    cp.cuda.get_current_stream().synchronize()
                request = self.leader_comm.Iallreduce(
                    MPI.IN_PLACE,
                    magnitude_sq,
                    op=MPI.SUM,
                )
            if request is not None:
                request.Wait()

            if self.group_comm is not None:
                if CUPY_AVAILABLE and isinstance(magnitude_sq, cp.ndarray):
                    cp.cuda.get_current_stream().synchronize()
                self.group_comm.Bcast(magnitude_sq, root=0)

            magnitude = self.xp.sqrt(magnitude_sq)
            denominator = self.xp.maximum(
                self.xp.asarray(1.0, dtype=magnitude.dtype),
                magnitude / self.lamda,
            )
            p_m = p_m / denominator
        else:
            magnitude = self.xp.abs(p_m)
            p_m = p_m / self.xp.maximum(1, magnitude / self.lamda)

        ghost_left_last = self.xp.empty_like(p_m[0])
        ghost_left_last = self.xp.ascontiguousarray(ghost_left_last)
        sendbuf = self.xp.ascontiguousarray(p_m[-1])
        motion_divergence_tag = self.tagger(
            iteration,
            axis="motion",
            kind="div",
        )
        motion_divergence_requests = self.dist_grid.halo_exchange(
            sendbuf,
            ghost_left_last,
            send_peer=self.motion_right_peer,
            recv_peer=self.motion_left_peer,
            tag=motion_divergence_tag,
        )

        # Preserve the two legacy communication modes.  The default builds
        # the full gradient and performs one reduction while the motion halo
        # is in flight.  The memory-saving mode waits for the halo first and
        # then reduces one bin at a time below.
        data_gradient = None
        if not self.reduce_per_bin:
            data_gradient = self.xp.zeros_like(u_now)
            for bin_index in range(self.B):
                self._accumulate_data_adjoint(
                    data_gradient[bin_index],
                    q_data,
                    bin_index,
                )
            self._coil_allreduce_in_place(data_gradient)

        divergence = self.xp.zeros_like(u_now)
        if self.B > 1:
            divergence[1:] = p_m[1:] - p_m[:-1]
        MPI.Request.Waitall(motion_divergence_requests)
        if self.is_motion_first:
            divergence[0] = p_m[0]
        else:
            divergence[0] = p_m[0] - ghost_left_last

        if self.is_motion_last:
            if self.B > 1:
                divergence[-1] = -p_m[-2]
            else:
                divergence[0] = (
                    0 if self.is_motion_first else -ghost_left_last
                )

        if self.reduce_per_bin:
            for bin_index in range(self.B):
                data_gradient = self.xp.zeros(
                    [self.E] + self.img_shape,
                    dtype=self.mps.dtype,
                )
                self._accumulate_data_adjoint(
                    data_gradient,
                    q_data,
                    bin_index,
                )
                self._coil_allreduce_in_place(data_gradient)
                sp.axpy(
                    u_now[bin_index],
                    -self.tau,
                    data_gradient - divergence[bin_index],
                )
        else:
            sp.axpy(u_now, -self.tau, data_gradient - divergence)

        u_bar[...] = 2 * u_now - u_old
        return u_bar, u_now, p_m, q_data

    def _sync_all(self):
        if self.world_comm is not None:
            self.world_comm.Barrier()
        if CUPY_AVAILABLE and self.xp is cp:
            cp.cuda.Device().synchronize()
        if self.world_comm is not None:
            self.world_comm.Barrier()

    def run(self):
        done = False
        while not done:
            try:
                with tqdm(
                    total=self.max_iter,
                    desc="ReconTVM",
                    disable=not self.show_pbar,
                ) as progress:
                    with self.device:
                        mrimg, p_m, q_data = self.pdhg_init()
                        primal_u = self.xp.zeros_like(mrimg)

                        self._sync_all()
                        start = time.perf_counter()

                        for iteration in range(self.max_iter):
                            primal_u_old = self.xp.copy(primal_u)
                            mrimg, primal_u, p_m, q_data = self.pdhg(
                                mrimg,
                                primal_u,
                                p_m,
                                q_data,
                                iteration,
                            )

                            difference = primal_u - primal_u_old
                            numerator_local = self.xp.vdot(
                                difference.ravel(),
                                difference.ravel(),
                            ).real
                            denominator_local = self.xp.vdot(
                                primal_u.ravel(),
                                primal_u.ravel(),
                            ).real

                            buf = self.xp.empty((2,), dtype=self.xp.float32)
                            buf[0] = numerator_local.astype(
                                buf.dtype,
                                copy=False,
                            )
                            buf[1] = denominator_local.astype(
                                buf.dtype,
                                copy=False,
                            )

                            if self.group_comm is not None and self.group_rank != 0:
                                buf[0] = self.xp.float32(0.0)
                                buf[1] = self.xp.float32(0.0)

                            # Host staging is intentional.  A device-buffer MPI
                            # reduction allowed different ranks to observe different
                            # stopping values and leave the iteration loop unevenly.
                            if CUPY_AVAILABLE and self.xp is cp:
                                buf_host = cp.asnumpy(buf)
                            else:
                                buf_host = np.asarray(buf)

                            if self.world_comm is not None:
                                self.world_comm.Allreduce(MPI.IN_PLACE, buf_host, op=MPI.SUM)

                            numerator = float(buf_host[0])
                            denominator = float(buf_host[1])
                            if (
                                np.isfinite(numerator)
                                and np.isfinite(denominator)
                                and denominator > 0.0
                            ):
                                iteration_tolerance = float(
                                    np.sqrt(max(numerator, 0.0))
                                    / np.sqrt(denominator)
                                )
                            else:
                                iteration_tolerance = float("inf")

                            progress.set_postfix(tol=iteration_tolerance)
                            if iteration_tolerance < self.tol:
                                logging.info(
                                    "Converged in %s iterations.",
                                    iteration,
                                )
                                break
                            progress.update()

                        self._sync_all()
                        elapsed_local = time.perf_counter() - start
                        elapsed_global = (
                            self.world_comm.allreduce(elapsed_local, op=MPI.MAX)
                            if self.world_comm is not None
                            else elapsed_local
                        )
                        if self.show_pbar:
                            logging.info(
                                "[PDHG iterations only] %.2f s (max over ranks)",
                                elapsed_global,
                            )
                        done = True
            except OverflowError:
                self.sigma *= 0.9
                self.tau *= 0.9

        return mrimg


def main(argv=None) -> int:
    mainstart = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="TVM topology-aware PDHG reconstruction."
    )
    parser.add_argument(
        "--readout-fraction",
        "--frac",
        dest="frac",
        type=float,
        default=0.98,
        help="Readout fraction.",
    )
    parser.add_argument(
        "--num-bins",
        "--num_bins",
        dest="num_bins",
        type=int,
        default=6,
        help="Number of motion phases.",
    )
    parser.add_argument(
        "--lambda-motion",
        "--lamda",
        dest="lamda",
        type=float,
        default=1e-5,
        help="Regularization for motion.",
    )
    parser.add_argument(
        "--max-iter",
        "--max_iter",
        dest="max_iter",
        type=int,
        default=300,
        help="Maximum epochs.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="Relative L2-change stopping threshold.",
    )
    parser.add_argument(
        "--acceleration",
        "--acc",
        dest="acc",
        type=int,
        default=1,
        help="Retrospective undersampling factor.",
    )
    parser.add_argument(
        "--fov-scale",
        "--fov_scale",
        dest="fov_scale",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help="Reconstruction FOV scale factors in (z,y,x).",
    )
    parser.add_argument(
        "--crop-to-original",
        "--crop_to_orig",
        dest="crop_to_orig",
        action="store_true",
        help="Center-crop output back to the original matrix.",
    )
    parser.add_argument(
        "--echo-groups",
        "--echo_groups",
        dest="echo_groups",
        type=int,
        default=1,
        help="Number of echo groups (grid columns).",
    )
    parser.add_argument(
        "--motion-groups",
        "--motion_groups",
        dest="motion_groups",
        type=int,
        default=1,
        help="Number of motion groups (grid rows).",
    )
    parser.add_argument(
        "--l2-coupling",
        "--l2_coupling",
        dest="l2_coupling",
        action="store_true",
        help="Use L2 coupling across echoes.",
    )
    parser.add_argument(
        "--show-progress",
        "--show_pbar",
        dest="show_pbar",
        action="store_true",
        help="Show the progress bar.",
    )
    parser.add_argument("--device", type=int, default=0, help="GPU device.")
    parser.add_argument(
        "--multi-gpu",
        "--multi_gpu",
        dest="multi_gpu",
        action="store_true",
        help="Use one GPU per local MPI rank.",
    )
    parser.add_argument(
        "--reduce-per-bin",
        "--reduce_per_bin",
        dest="reduce_per_bin",
        action="store_true",
        help="Reduce one motion bin at a time to lower peak memory.",
    )
    parser.add_argument(
        "--fp16-communication",
        "--use_fp16_comm",
        dest="use_fp16_comm",
        action="store_true",
        help="Use fp16 where supported for intra-node reductions.",
    )
    parser.add_argument(
        "--nufft-backend",
        choices=("sigpy",),
        default="sigpy",
    )
    parser.add_argument(
        "--prepared-dir",
        type=str,
        default=None,
        help="Directory containing mps.hdr/.cfl and resp.hdr/.cfl.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for reconstructed CFL shards.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("input_dir", type=str, help="Raw MICCAI dataset directory.")
    parser.add_argument("img_file", type=str, help="Output filename stem.")

    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    world = MPI.COMM_WORLD
    intra = IntraNodeGroup(world)
    group_comm = intra.intra_node_comm

    world_rank = intra.world_rank
    world_size = intra.world_size
    group_rank = intra.group_rank
    group_size = intra.group_size
    num_nodes = intra.num_nodes
    group_ranks = intra.group_world_ranks

    motion_groups = int(args.motion_groups)
    echo_groups = int(args.echo_groups)
    if motion_groups <= 0 or echo_groups <= 0:
        if world_rank == 0:
            logging.error(
                "Invalid groups: motion_groups=%s, echo_groups=%s "
                "(must be positive).",
                motion_groups,
                echo_groups,
            )
        world.Abort(1)

    if num_nodes != motion_groups * echo_groups:
        if world_rank == 0:
            logging.error(
                "Node grid mismatch: num_nodes=%s != "
                "motion_groups*echo_groups=%s*%s=%s.",
                num_nodes,
                motion_groups,
                echo_groups,
                motion_groups * echo_groups,
            )
        world.Abort(1)

    grid = NodeGrid2D(
        intra,
        motion_groups=motion_groups,
        echo_groups=echo_groups,
    )
    leader_comm = grid.echo_axis_leader_comm
    tagger = make_comm_tagger(world, stride=512)

    if args.multi_gpu and CUPY_AVAILABLE:
        local_rank = intra.get_local_rank()
        device = sp.Device(local_rank)
        cp.cuda.Device(local_rank).use()
        logging.info(
            "World Rank %s (Node %s Grid %s Rank %s): binding to GPU %s",
            world_rank,
            intra.node_id,
            grid.coord,
            group_rank,
            local_rank,
        )
    else:
        device = sp.Device(args.device)
        logging.info(
            "World Rank %s (Node %s Grid %s Rank %s): using device %s",
            world_rank,
            intra.node_id,
            grid.coord,
            group_rank,
            args.device,
        )

    if world_rank == 0:
        logging.info(
            "[MPI] world_size=%s, num_nodes=%s",
            world_size,
            num_nodes,
        )
    if group_rank == 0:
        logging.info(
            "[Group %s] group_size=%s, ranks=%s",
            world_rank,
            group_size,
            group_ranks,
        )

    gpu_comm = (
        GpuComm(group_comm, prefer_fp16=args.use_fp16_comm)
        if args.multi_gpu
        else None
    )

    if world_rank == 0:
        logging.info("Reading data.")

    ksp_file = os.path.join(args.input_dir, "ksp")
    coord_file = os.path.join(args.input_dir, "ktraj")
    dcf_file = os.path.join(args.input_dir, "dens")
    img_shape_file = os.path.join(args.input_dir, "imageDim.txt")
    voxel_size_file = os.path.join(args.input_dir, "voxelSize.txt")
    prepared_dir = args.prepared_dir or args.input_dir
    mps_file = os.path.join(prepared_dir, "mps")
    resp_file = os.path.join(prepared_dir, "resp")

    with open(img_shape_file, "r", encoding="utf-8") as stream:
        line = stream.readline()
        img_shape = [int(value) for value in line.split(":")[::-1]]
    orig_img_shape = img_shape.copy()

    with open(voxel_size_file, "r", encoding="utf-8") as stream:
        line = stream.readline()
        voxel_size = [float(value) for value in line.split(":")[::-1]]

    if group_rank == 0:
        coord_full = np.squeeze(cfl.read_cfl(coord_file)).real[..., ::-1]
        dcf_full = np.squeeze(cfl.read_cfl(dcf_file)).real
        resp_full = np.squeeze(cfl.read_cfl(resp_file)).real

        reduction_factor = int(args.acc)
        num_tr_total = int(resp_full.size)
        if world_rank == 0:
            logging.info("Undersampling (ACC=%sX).", reduction_factor)

        tr_keep = np.arange(num_tr_total, dtype=np.int64)
        margin = 2
        lower = np.percentile(resp_full, margin)
        upper = np.percentile(resp_full, 100 - margin)
        tr_keep = tr_keep[(resp_full > lower) & (resp_full < upper)]
        target_count = int(num_tr_total / max(reduction_factor, 1))
        tr_keep = tr_keep[:target_count]

        num_ro = int(args.frac * coord_full.shape[-2])
        coord = coord_full[tr_keep, :num_ro]
        dcf = dcf_full[tr_keep, :num_ro]
        resp = resp_full[tr_keep]

        while coord.max() > 1000:
            num_ro -= 1
            coord = coord[:, :num_ro]
            dcf = dcf[:, :num_ro]

        if world_rank == 0:
            logging.info("Scaling coordinates.")
        for dimension in range(coord.shape[-1]):
            coord[..., dimension] *= voxel_size[dimension] * img_shape[dimension]

        fov_scale = np.asarray(args.fov_scale, dtype=float)
        if not np.allclose(fov_scale, 1.0):
            if np.any(fov_scale < 1.0):
                raise ValueError(
                    "FoV scale factors must be >= 1.0 for all axes (z y x)."
                )
            if world_rank == 0:
                logging.info(
                    "Scaling FoV by factors (x,y,z) = (%.2f, %.2f, %.2f).",
                    fov_scale[-1],
                    fov_scale[-2],
                    fov_scale[-3],
                )
            coord = coord.copy()
            coord[..., 0] *= fov_scale[-3]
            coord[..., 1] *= fov_scale[-2]
            coord[..., 2] *= fov_scale[-1]
            img_shape = [
                int(img_shape[-3] * fov_scale[-3]),
                int(img_shape[-2] * fov_scale[-2]),
                int(img_shape[-1] * fov_scale[-1]),
            ]

        dcf_maximum = float(np.max(dcf))
        if dcf_maximum > 0:
            dcf /= np.float32(dcf_maximum)

        metadata = {
            "num_ro": num_ro,
            "img_shape": img_shape,
            "orig_img_shape": orig_img_shape,
            "voxel_size": voxel_size,
            "fov_scale": fov_scale.tolist(),
        }
    else:
        coord = None
        dcf = None
        resp = None
        tr_keep = None
        metadata = None

    metadata = group_comm.bcast(metadata, root=0)
    num_ro = metadata["num_ro"]
    img_shape = metadata["img_shape"]
    orig_img_shape = metadata["orig_img_shape"]
    voxel_size = metadata["voxel_size"]
    fov_scale = np.asarray(metadata["fov_scale"], dtype=float)

    coord = group_comm.bcast(coord, root=0)
    dcf = group_comm.bcast(dcf, root=0)
    resp = group_comm.bcast(resp, root=0)
    tr_keep = group_comm.bcast(tr_keep, root=0)

    if group_rank == 0:
        del coord_full, dcf_full, resp_full
    if world_rank == 0:
        logging.info(
            "I/O small tensors done. num_ro=%s, img_shape=%s",
            num_ro,
            img_shape,
        )

    if group_rank == 0:
        ksp_header = cfl.read_cfl_header(ksp_file)
        E_total = KSpaceLayout.from_cfl_shape(ksp_header).echoes
    else:
        E_total = None
    E_total = group_comm.bcast(E_total, root=0)

    b0, b1, e0, e1 = grid.set_partitions(
        B_total=args.num_bins,
        E_total=E_total,
    )
    B_local = b1 - b0
    E_local = e1 - e0

    bin_margin = 0
    if group_rank == 0:
        bin_edges = np.percentile(
            resp,
            np.linspace(
                0 + bin_margin,
                100 - bin_margin,
                args.num_bins + 1,
            ),
        )
    else:
        bin_edges = None
    bin_edges = group_comm.bcast(bin_edges, root=0)

    if group_rank == 0:
        local_mask = np.zeros(resp.shape, dtype=bool)
        for global_bin in range(b0, b1):
            local_mask |= (resp >= bin_edges[global_bin]) & (
                resp < bin_edges[global_bin + 1]
            )
        idx_local = np.nonzero(local_mask)[0].astype(np.int64)
        tr_idx_local = tr_keep[idx_local].astype(np.int64)
    else:
        idx_local = None
        tr_idx_local = None

    idx_local = group_comm.bcast(idx_local, root=0)
    tr_idx_local = group_comm.bcast(tr_idx_local, root=0)
    coord = coord[idx_local]
    dcf = dcf[idx_local]
    resp = resp[idx_local]

    if group_rank == 0:
        mps_header = cfl.read_cfl_header(mps_file)
        C_total = int(mps_header[-4])
        logging.info(
            "Node#%s Grid %s echo %s-%s/%s, bins %s-%s/%s",
            intra.node_id,
            grid.coord,
            e0,
            e1 - 1,
            E_total,
            b0,
            b1 - 1,
            args.num_bins,
        )
        coil_indices = split_coils(C_total, group_size)
        shared_metadata = {
            "C_total": int(C_total),
            "E_local": int(E_local),
            "coil_indices": [indices.tolist() for indices in coil_indices],
        }
    else:
        shared_metadata = None

    shared_metadata = group_comm.bcast(shared_metadata, root=0)
    coil_indices = [
        np.array(indices, dtype=int)
        for indices in shared_metadata["coil_indices"]
    ]
    shared_rank = gpu_comm.shm_comm.rank if gpu_comm is not None else group_rank
    coil_idxs_local = coil_indices[shared_rank]

    iostart = time.perf_counter()
    ksp_local = cfl.read_kspace_shard(
        name=ksp_file,
        echo_start=e0,
        echo_end=e1,
        coil_indices=coil_idxs_local,
        readout=num_ro,
        trajectory_indices=tr_idx_local,
    )
    mps_local = cfl.read_mps_coils(
        name=mps_file,
        coil_indices=coil_idxs_local,
    )

    if tuple(mps_local.shape[-3:]) != tuple(img_shape):
        raise ValueError(
            f"MPS grid {tuple(mps_local.shape[-3:])} does not match "
            f"the requested reconstruction grid {tuple(img_shape)}. "
            "Prepare sensitivity maps with the same FoV scale."
        )

    local_ksp_maximum = max_abs_cplx_chunked(ksp_local)
    global_ksp_maximum = world.allreduce(local_ksp_maximum, op=MPI.MAX)
    if global_ksp_maximum > 0:
        ksp_local /= np.float32(global_ksp_maximum)

    local_sos_sq = np.sum(np.abs(mps_local) ** 2, axis=0, dtype=np.float32)
    sos_sq = np.ascontiguousarray(local_sos_sq)
    group_comm.Allreduce(MPI.IN_PLACE, sos_sq, op=MPI.SUM)
    mps_sos = np.sqrt(sos_sq)
    mps_sos[mps_sos == 0] = np.float32(1.0)
    mps_local /= mps_sos[None, ...]

    # Preserve the legacy post-sharding normalization as well as the earlier
    # normalization over retained trajectories. The second maximum is usually
    # one, but can differ when the trajectory carrying it falls outside the
    # selected motion bins.
    local_dcf_maximum = float(np.max(dcf)) if dcf.size else 0.0
    global_dcf_maximum = world.allreduce(local_dcf_maximum, op=MPI.MAX)
    if global_dcf_maximum > 0:
        dcf /= np.float32(global_dcf_maximum)

    ksp = ksp_local
    mps = mps_local
    C_local = mps.shape[0]

    ioend = time.perf_counter()
    if world_rank == 0:
        logging.info("Largest IO done in %.2f seconds.", ioend - iostart)
        logging.info(
            "Prepared shards (direct-IO): Node#%s Grid %s "
            "E_local=%s, C_local=%s, num_bins=%s",
            intra.node_id,
            grid.coord,
            E_local,
            C_local,
            args.num_bins,
        )

    dual_q = np.zeros_like(ksp, dtype=np.complex64)
    procstart = time.perf_counter()
    mrimg = TvmReconstructor(
        ksp,
        coord,
        dcf,
        mps,
        resp,
        dual_q,
        B_local,
        args.l2_coupling,
        max_iter=args.max_iter,
        lamda=args.lamda,
        sigma=1 / 6,
        tau=1 / 6,
        tol=args.tol,
        margin=0,
        device=device,
        B_total=args.num_bins,
        b0=b0,
        b1=b1,
        motion_left_peer=grid.motion_left_peer,
        motion_right_peer=grid.motion_right_peer,
        bin_edges=bin_edges,
        world_comm=world,
        group_comm=group_comm,
        leader_comm=leader_comm,
        nccl_group=gpu_comm,
        dist_grid=grid,
        tagger=tagger,
        show_pbar=args.show_pbar and world_rank == 0,
        reduce_per_bin=args.reduce_per_bin,
        nufft_backend=args.nufft_backend,
    ).run()

    procend = time.perf_counter()
    if world_rank == 0:
        logging.info(
            "Reconstruction done in %.2f seconds.",
            procend - procstart,
        )

    is_cupy_array = CUPY_AVAILABLE and isinstance(mrimg, cp.ndarray)
    mr_cpu = cp.asnumpy(mrimg) if is_cupy_array else np.asarray(mrimg)

    X, Y, Z = map(int, mps.shape[1:])
    B, E = int(mr_cpu.shape[0]), int(mr_cpu.shape[1])
    img_cpu = np.zeros(
        (B, 1, 1, E, 1, 1, X, Y, Z),
        dtype=mr_cpu.dtype,
    )
    img_cpu[:, 0, 0, :, 0, 0, :, :, :] = mr_cpu
    if args.crop_to_orig and not np.allclose(fov_scale, 1.0):
        img_cpu = center_crop_zyx(img_cpu, orig_img_shape)

    os.makedirs(args.output_dir, exist_ok=True)
    out_base = os.path.join(
        args.output_dir,
        f"{args.img_file}_e{e0}-{e1 - 1}_m{b0}-{b1 - 1}",
    )
    if group_rank == 0:
        if world_rank == 0:
            logging.info("Writing output to %s", out_base)
        cfl.write_cfl(out_base, img_cpu)

    if world_rank == 0:
        manifest = {
            "format_version": 1,
            "algorithm": "tvm",
            "nufft_backend": args.nufft_backend,
            "input_directory": os.path.abspath(args.input_dir),
            "prepared_directory": os.path.abspath(prepared_dir),
            "output_stem": args.img_file,
            "output_pattern": (
                f"{args.img_file}_e{{e0}}-{{e1}}_m{{b0}}-{{b1}}"
            ),
            "parameters": {
                "readout_fraction": args.frac,
                "num_bins": args.num_bins,
                "num_echoes": E_total,
                "lambda_motion": args.lamda,
                "max_iter": args.max_iter,
                "tol": args.tol,
                "acceleration": args.acc,
                "fov_scale_zyx": list(args.fov_scale),
                "crop_to_original": args.crop_to_orig,
                "l2_coupling": args.l2_coupling,
                "reduce_per_bin": args.reduce_per_bin,
            },
            "distributed": {
                "world_size": world_size,
                "num_nodes": num_nodes,
                "motion_groups": motion_groups,
                "echo_groups": echo_groups,
                "fp16_communication": args.use_fp16_comm,
            },
            "normalization": {
                "kspace": "global_max_magnitude",
                "mps": "coil_root_sum_of_squares",
                "density": "maximum_then_global_maximum_after_motion_sharding",
            },
        }
        manifest_path = os.path.join(args.output_dir, "run_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")

    mainend = time.perf_counter()
    if world_rank == 0:
        logging.info("Main done in %.2f seconds.", mainend - mainstart)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
