#!/usr/bin/env python3
"""Motion- and echo-TV reconstruction on a distributed node grid."""

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
    m = 0.0
    for i in range(0, x.size, chunk_elems):
        m_i = np.max(np.abs(x[i:i+chunk_elems]))
        if m_i > m:
            m = float(m_i)
    return m

def center_crop_zyx(arr, out_shape_zyx):
    Nz0, Ny0, Nx0 = out_shape_zyx
    Nz, Ny, Nx = arr.shape[-3:]
    if Nz0 > Nz or Ny0 > Ny or Nx0 > Nx:
        raise ValueError(f"Requested crop {out_shape_zyx} larger than input {(Nz,Ny,Nx)}")

    z0 = (Nz - Nz0) // 2
    y0 = (Ny - Ny0) // 2
    x0 = (Nx - Nx0) // 2
    return arr[..., z0:z0+Nz0, y0:y0+Ny0, x0:x0+Nx0]

class TvmeReconstructor:
    def __init__(self, ksp, coord, dcf, mps, resp, dual_q, B_local,
                 l2_coupling=False,
                 lamda_m=1e-5, lamda_e=1e-5, sigma=0.01, tau=0.01,
                 max_iter=300, tol=1e-3, margin=10, device=sp.cpu_device,
                 E_total=None, e0=None, e1=None, echo_left_peer=None, echo_right_peer=None,
                 B_total=None, b0=None, b1=None, motion_left_peer=None, motion_right_peer=None,
                 bin_edges=None,
                 world_comm=None, group_comm=None, leader_comm=None,
                 nccl_group=None, dist_grid=None, tagger=None,
                 show_pbar=True, nufft_backend="sigpy"):

        self.mps = mps
        self.B = B_local             # number of motion states
        self.C = len(mps)      # number of coils
        self.E = ksp.shape[0]  # number of echoes

        self.E_total = E_total if E_total is not None else self.E
        self.e0 = e0
        self.e1 = e1
        self.is_global_last = (self.e1 == self.E_total)
        self.is_global_first = (self.e0 == 0)
        self.echo_left_peer = echo_left_peer
        self.echo_right_peer = echo_right_peer
        self.B_total = B_total if B_total is not None else self.B
        self.b0 = b0
        self.b1 = b1
        self.is_motion_last = (self.b1 == self.B_total)
        self.is_motion_first = (self.b0 == 0)
        self.motion_left_peer = motion_left_peer
        self.motion_right_peer = motion_right_peer

        self.lamda_m = lamda_m # reg for motion
        self.lamda_e = lamda_e # reg for echo
        self.sigma = sigma     # step size for dual update
        self.tau = tau         # step size for primal update
        self.l2_coupling = l2_coupling
        self.max_iter = max_iter # stopping criteria: max iterations
        self.tol = tol           # stopping criteria: tolerance level
        self.dist_grid = dist_grid
        self.tagger = tagger

        # Device setup
        self.device = sp.Device(device)
        self.xp = self.device.xp
        self.world_comm = world_comm
        self.group_comm = group_comm
        self.leader_comm = leader_comm
        self.nccl_group = nccl_group

        self.need_coil_reduce = (self.group_comm is not None and self.group_comm.Get_size() > 1)
        self.use_nccl_reduce = (
            self.need_coil_reduce
            and CUPY_AVAILABLE
            and getattr(self, "nccl_group", None) is not None
            and getattr(self.nccl_group, "stream", None) is not None
        )

        self.group_rank = self.group_comm.Get_rank() if self.group_comm is not None else 0

        self.show_pbar = show_pbar
        if world_comm is not None:
            self.show_pbar = show_pbar and world_comm.rank == 0

        self.img_shape = list(mps.shape[-3:])

        if bin_edges is None:
            bins = np.percentile(resp, np.linspace(0 + margin, 100 - margin, self.B_total + 1))
        else:
            bins = np.asarray(bin_edges)
            assert bins.ndim == 1 and bins.size == self.B_total + 1, \
                f"bin_edges must have shape ({self.B_total + 1},), got {bins.shape}"

        self.bksp = []
        self.bcoord = []
        self.bdcf = []
        self.dual_q = []
        for gb in range(self.b0, self.b1):   # global bin id
            idx = (resp >= bins[gb]) & (resp < bins[gb + 1])
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

        # move data to device
        self.mps_dev = sp.to_device(self.mps, self.device)
        self.mps_conj = self.xp.conj(self.mps_dev)

    ## Forward Difference Operators
    def dxfc(self, x):
        diff = self.xp.zeros(x.shape, self.xp.complex64)
        diff[..., :-1] = x[..., 1:]
        diff[..., :-1] = diff[..., :-1] - x[..., :-1]
        return diff

    def dyfc(self, x):
        diff = self.xp.zeros(x.shape, self.xp.complex64)
        diff[..., :-1, :] = x[..., 1:, :]
        diff[..., :-1, :] = diff[..., :-1, :] - x[..., :-1, :]
        return diff

    def dzfc(self, x):
        diff = self.xp.zeros(x.shape, self.xp.complex64)
        diff[..., :-1, :, :] = x[..., 1:, :, :]
        diff[..., :-1, :, :] = diff[..., :-1, :, :] - x[..., :-1, :, :]
        return diff

    ## Adjoint of the forward difference operators (negative backward)
    def dxbc(self, x):
        diff = self.xp.zeros(x.shape, self.xp.complex64)
        diff[..., :-1] = -x[..., :-1]
        diff[..., 1:]  = -diff[..., :-1] + diff[..., 1:]
        return diff

    def dybc(self, x):
        diff = self.xp.zeros(x.shape, self.xp.complex64)
        diff[..., :-1, :] = -x[..., :-1, :]
        diff[..., 1:, :]  = -diff[..., :-1, :] + diff[..., 1:, :]
        return diff

    def dzbc(self, x):
        diff = self.xp.zeros(x.shape, self.xp.complex64)
        diff[..., :-1, :, :] = -x[..., :-1, :, :]
        diff[..., 1:, :, :]  = -diff[..., :-1, :, :] + diff[..., 1:, :, :]
        return diff

    ## Initialize Primal-Dual Algorithm
    def pdhg_init(self):

        # Primal Variables
        u_bar = self.xp.zeros([self.B] + [self.E] + self.img_shape, dtype=self.mps.dtype)

        # Dual Variables
        ## motion
        p_m  = self.xp.zeros_like(u_bar)
        ## echo
        p_ex = self.xp.zeros_like(u_bar)
        p_ey = self.xp.zeros_like(u_bar)
        p_ez = self.xp.zeros_like(u_bar)
        ## data
        q_data = self.dual_q

        return u_bar, p_m, p_ex, p_ey, p_ez, q_data

    ## Main PDHG function
    def pdhg(self, u_bar, u_now, p_m, p_ex, p_ey, p_ez, q_data, it):

        ##----------------------------##
        ## @Store the current iterate ##
        ##----------------------------##
        u_old = self.xp.copy(u_now)

        req_absp_sq = None

        # 1-1) Forward difference in the motion state dimension
        # 1-1-A） communicate u_0
        ghost_m_right_first = self.xp.empty_like(u_bar[0])  # (E, X, Y, Z)
        ghost_m_right_first = self.xp.ascontiguousarray(ghost_m_right_first)
        sendbuf_m = self.xp.ascontiguousarray(u_bar[0])     # send my first bin to left

        tag_mdelta = self.tagger(it, axis="motion", kind="delta")
        reqs_mdelta = self.dist_grid.halo_exchange(
            sendbuf_m, ghost_m_right_first,
            send_peer=self.motion_left_peer,
            recv_peer=self.motion_right_peer,
            tag=tag_mdelta
        )

        # 2-1) Forward difference in the echo dimension
        # (B, E_local, X, Y, Z)
        diff_e = self.xp.zeros_like(u_bar)

        # local forward difference along echo
        if self.E > 1:
            diff_e[:, :-1] = u_bar[:, 1:] - u_bar[:, :-1]

        # 2-1-A) start async Sendrecv for ghost_right_first
        # need ghost from right neighbor for the last echo slice
        ghost_right_first = self.xp.zeros_like(u_bar[:, 0]) # (B, X, Y, Z)
        ghost_right_first = self.xp.ascontiguousarray(ghost_right_first)
        sendbuf = self.xp.ascontiguousarray(u_bar[:, 0])

        if CUPY_AVAILABLE and isinstance(sendbuf, cp.ndarray):
            cp.cuda.get_current_stream().synchronize()

        tag_delta = self.tagger(it, axis="echo", kind="delta")
        reqs_delta = self.dist_grid.halo_exchange(
            sendbuf, ghost_right_first,
            send_peer=self.echo_left_peer,  # send to left
            recv_peer=self.echo_right_peer, # recv from right
            tag=tag_delta
        )

        # # 3-1) Computing (FSu^k - y)
        for b in range(self.B):
            tmp = self.xp.zeros_like(self.dual_q[b])
            for c in range(self.C):
                mps_c = self.mps_dev[c]
                for e in range(self.E):
                    tmp[e, c] = (
                        self.nufft.forward(u_bar[b, e] * mps_c, b)
                        - self.bksp[b][e][c]
                    )
            # 3-2) Proximal mapping
            q_data[b] = (q_data[b] + self.sigma * tmp) / (1 + self.sigma)

        # 1-1-B) compute diff_m
        diff_m = self.xp.zeros_like(u_bar)
        if self.B > 1:
            diff_m[:-1] = u_bar[1:] - u_bar[:-1]
        MPI.Request.Waitall(reqs_mdelta)
        if not self.is_motion_last:
            diff_m[-1] = ghost_m_right_first - u_bar[-1]
        else:
            diff_m[-1] = 0


        # 1-2) Gradient ascent update
        p_m = p_m + self.sigma * diff_m
        diff_m = None

        # 1-3-A) Proximal mapping: start async allreduce for L2-coupling
        if self.l2_coupling:  # L2 coupling

            # local sum of ||p_m||^2 over echoes
            absp_sq = self.xp.sum(self.xp.abs(p_m) ** 2, axis=1, keepdims=True)
            absp_sq = self.xp.ascontiguousarray(absp_sq)

            # global Allreduce over all echoes via leader_comm, then Bcast inside group
            if self.leader_comm is not None:
                if CUPY_AVAILABLE and isinstance(absp_sq, cp.ndarray):
                    cp.cuda.get_current_stream().synchronize()

                # non-blocking Allreduce
                req_absp_sq = self.leader_comm.Iallreduce(MPI.IN_PLACE, absp_sq, op=MPI.SUM)

        else: # L1-coupling
            absp = self.xp.abs(p_m)
            p_m = p_m/self.xp.maximum(1, absp/self.lamda_m)
            absp = None

        # 2-1-B) wait comm and fill in ghost cell
        MPI.Request.Waitall(reqs_delta)
        if not self.is_global_last:
            diff_e[:, -1] = ghost_right_first - u_bar[:, -1]
        else:
            # last global slice, no right neighbor
            diff_e[:, -1] = 0

        # 2-2) Forward gradient in the image space of diff_e and gradient ascent update
        p_ex = p_ex + self.sigma * self.dxfc(diff_e)
        p_ey = p_ey + self.sigma * self.dyfc(diff_e)
        p_ez = p_ez + self.sigma * self.dzfc(diff_e)
        diff_e = None

        # 2-3) Isotropic TV proximal mapping for each motion bin
        absp = (self.xp.abs(p_ex) ** 2 + self.xp.abs(p_ey) ** 2 + self.xp.abs(p_ez) ** 2) ** 0.5
        p_ex = p_ex/self.xp.maximum(1, absp/self.lamda_e)
        p_ey = p_ey/self.xp.maximum(1, absp/self.lamda_e)
        p_ez = p_ez/self.xp.maximum(1, absp/self.lamda_e)
        absp = None

        # 1-3-B) Proximal mapping: wait comm, Bcast and apply prox
        if self.l2_coupling:
            if self.leader_comm is not None and req_absp_sq is not None:
                req_absp_sq.Wait()
            if self.group_comm is not None:
                if CUPY_AVAILABLE and isinstance(absp_sq, cp.ndarray):
                    cp.cuda.get_current_stream().synchronize()
                self.group_comm.Bcast(absp_sq, root=0)
            absp = self.xp.sqrt(absp_sq)
            denom = self.xp.maximum(self.xp.asarray(1.0, dtype=absp.dtype), absp / self.lamda_m)
            p_m = p_m / denom
            absp_sq = None
            absp = None

        # 4-1) Compute divergene w/adjoint of the forward difference (-backward)
        # divp_m (motion)
        divp_m = self.xp.zeros_like(u_now)

        # 4-1-A) communicate p_m
        ghost_m_left_last = self.xp.empty_like(p_m[0])      # (E, X, Y, Z)
        ghost_m_left_last = self.xp.ascontiguousarray(ghost_m_left_last)
        sendbuf_m = self.xp.ascontiguousarray(p_m[-1])      # send my last p to right

        tag_mdiv = self.tagger(it, axis="motion", kind="div")
        reqs_mdiv = self.dist_grid.halo_exchange(
            sendbuf_m, ghost_m_left_last,
            send_peer=self.motion_right_peer,
            recv_peer=self.motion_left_peer,
            tag=tag_mdiv
        )

        # 4-1-2) Compute divp_e
        # divp_e (echo)
        divp = self.dxbc(p_ex) + self.dybc(p_ey) + self.dzbc(p_ez)
        divp_e = self.xp.zeros_like(u_now)

        # local divergence along echo
        if self.E > 1:
            divp_e[:, 1:] = divp[:, :-1] - divp[:, 1:]

        # 4-1-2-A) start async Sendrecv for ghost_left_last
        # ghost from left neighbor for echo = 0
        ghost_left_last = self.xp.empty_like(divp[:, 0]) # (B, X, Y, Z)
        ghost_left_last = self.xp.ascontiguousarray(ghost_left_last)
        sendbuf = self.xp.ascontiguousarray(divp[:, -1])

        if CUPY_AVAILABLE and isinstance(sendbuf, cp.ndarray):
            cp.cuda.get_current_stream().synchronize()

        tag_div = self.tagger(it, axis="echo", kind="div")
        reqs_div = self.dist_grid.halo_exchange(
            sendbuf, ghost_left_last,
            send_peer=self.echo_right_peer,  # send to right
            recv_peer=self.echo_left_peer,   # recv from left
            tag=tag_div
        )

        # 4-2) Compute (FS)^H q_data^k+1
        tmp = self.xp.zeros_like(u_now)
        for b in range(self.B):
            for c in range(self.C):
                mps_c_conj = self.mps_conj[c]
                for e in range(self.E):
                    tmp[b, e] += (
                        self.nufft.adjoint(
                            self.bdcf[b] * q_data[b][e][c],
                            b,
                        )
                        * mps_c_conj
                    )

        # group-wise Allreduce over coils for (FS)^H q_data
        evt2 = None
        req_coil = None
        buf = None
        if self.use_nccl_reduce and isinstance(tmp, cp.ndarray):
            # NCCL async allreduce with separate comm stream
            comp_stream = cp.cuda.get_current_stream()
            comm_stream = self.nccl_group.stream

            evt = cp.cuda.Event()
            evt.record(comp_stream)
            comm_stream.wait_event(evt)

            with comm_stream:
                self.nccl_group.alreduce_sum_(tmp)

            evt2 = cp.cuda.Event()
            evt2.record(comm_stream)

        elif self.need_coil_reduce:
            # CPU / MPI fallback
            # print("Using MPI Allreduce for (FS)^H q_data")
            buf = cp.asnumpy(tmp) if CUPY_AVAILABLE and isinstance(tmp, cp.ndarray) else np.asarray(tmp)
            req_coil = self.group_comm.Iallreduce(MPI.IN_PLACE, buf, op=MPI.SUM)

        # 4-1-B) compute div p_m
        divp_m = self.xp.zeros_like(u_now)
        if self.B > 1:
            divp_m[1:] = p_m[:-1] - p_m[1:]

        MPI.Request.Waitall(reqs_mdiv)

        if self.is_motion_first:
            divp_m[0] = -p_m[0]
        else:
            divp_m[0] = ghost_m_left_last - p_m[0]

        if self.is_motion_last:
            if self.B > 1:
                divp_m[-1] = p_m[-2]
            else:
                divp_m[0] = 0 if self.is_motion_first else ghost_m_left_last

        u_now -= self.tau * divp_m
        divp_m = None

        # 4-1-2-B) wait comm and fill in ghost cell
        MPI.Request.Waitall(reqs_div)
        # echo = 0
        if self.is_global_first:
            divp_e[:, 0] = -divp[:, 0]
        elif self.E == 1 and self.is_global_last:
            # only one local echo and it's also the last global echo
            divp_e[:, 0] = ghost_left_last
        else:
            divp_e[:, 0] = ghost_left_last - divp[:, 0]
        # global last echo (right boundary)
        if self.is_global_last and self.E > 1:
            divp_e[:, -1] = divp[:, -2]

        # in-place update
        u_now -= self.tau * divp_e
        divp_e = None
        divp = None

        # --- finish coil allreduce BEFORE using tmp ---
        if evt2 is not None:
            # NCCL path: wait until comm_stream finished the allreduce on tmp
            cp.cuda.get_current_stream().wait_event(evt2)

        elif req_coil is not None:
            # MPI path: wait and then copy reduced buf back to tmp
            req_coil.Wait()
            if CUPY_AVAILABLE and isinstance(tmp, cp.ndarray):
                tmp.set(buf)
            else:
                tmp[...] = buf

        # 4-3) Gradient descent
        u_now -= self.tau * tmp

        ##---------------------##
        ## @Extragradient step ##
        ##---------------------##
        u_bar[...] = 2*u_now - u_old

        return u_bar, u_now, p_m, p_ex, p_ey, p_ez, q_data


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
                with tqdm(total=self.max_iter, desc='ReconTVME',
                          disable=not self.show_pbar) as pbar:
                    with self.device:

                        # Initialize variables
                        mrimg, p_m, p_ex, p_ey, p_ez, q_data = self.pdhg_init()
                        primal_u = self.xp.zeros_like(mrimg)

                        self._sync_all()
                        t0 = time.perf_counter()

                        # PDHG iteration
                        for it in range(self.max_iter):
                            primal_u_old = self.xp.copy(primal_u)

                            mrimg, primal_u, p_m, p_ex, p_ey, p_ez, q_data = \
                                self.pdhg(mrimg, primal_u, p_m, p_ex, p_ey, p_ez, q_data, it)

                            # global tol
                            # tol = ||u^k - u^{k-1}||_2 / ||u^k||_2
                            du = (primal_u - primal_u_old)

                            num2_local = self.xp.vdot(du.ravel(), du.ravel()).real
                            den2_local = self.xp.vdot(primal_u.ravel(), primal_u.ravel()).real

                            buf = self.xp.empty((2,), dtype=self.xp.float32)
                            buf[0] = num2_local.astype(buf.dtype, copy=False)
                            buf[1] = den2_local.astype(buf.dtype, copy=False)

                            # avoid double-counting within a node: only group_rank==0 contributes
                            if (self.group_comm is not None) and (self.group_rank != 0):
                                buf[0] = self.xp.float32(0.0)
                                buf[1] = self.xp.float32(0.0)

                            # Stage tol scalars to host before MPI. The device-buffer
                            # Allreduce path can let ranks see inconsistent tol values
                            # and exit the iteration loop at different times.
                            if CUPY_AVAILABLE and self.xp is cp:
                                buf_host = cp.asnumpy(buf)
                            else:
                                buf_host = np.asarray(buf)

                            if self.world_comm is not None:
                                self.world_comm.Allreduce(MPI.IN_PLACE, buf_host, op=MPI.SUM)

                            num2 = float(buf_host[0])
                            den2 = float(buf_host[1])

                            if np.isfinite(num2) and np.isfinite(den2) and den2 > 0.0:
                                _tol = float(np.sqrt(max(num2, 0.0)) / np.sqrt(den2))
                            else:
                                _tol = float("inf")

                            pbar.set_postfix(tol=_tol)

                            if (_tol < self.tol):
                                logging.info('Converged in {} iterations.'.format(it))
                                break

                            pbar.update()

                        self._sync_all()
                        t1 = time.perf_counter()

                        t_local = t1 - t0
                        t_global = self.world_comm.allreduce(t_local, op=MPI.MAX) if self.world_comm is not None else t_local
                        if self.show_pbar:
                            logging.info(f"[PDHG iterations only] {t_global:.2f} s (max over ranks)")

                        done = True

            except OverflowError:
                self.sigma *= 0.9
                self.tau *= 0.9

        return mrimg


def main(argv=None) -> int:
    mainstart = time.perf_counter()

    parser = argparse.ArgumentParser(description='TVME topology-aware PDHG reconstruction.')
    parser.add_argument('--readout-fraction', '--frac', dest='frac', type=float, default=0.98,
                        help='Readout fractions.')
    parser.add_argument('--num-bins', '--num_bins', dest='num_bins', type=int, default=6,
                        help='Number of phases.')
    parser.add_argument('--lambda-motion', '--lamda_m', dest='lamda_m', type=float, default=1e-5,
                        help='Regularization for motion.')
    parser.add_argument('--lambda-echo', '--lamda_e', dest='lamda_e', type=float, default=1e-5,
                        help='Regularization for echo.')
    parser.add_argument('--max-iter', '--max_iter', dest='max_iter', type=int, default=300,
                        help='Maximum epochs.')
    parser.add_argument('--tol', type=float, default=1e-3,
                        help='Relative L2-change stopping threshold.')
    parser.add_argument('--acceleration', '--acc', dest='acc', type=int, default=1,
                        help='Reduction factor')
    parser.add_argument('--fov-scale', '--fov_scale', dest='fov_scale',
                        type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help="Reconstruction FOV scale factors in (z,y,x) (>1 increases FOV).")
    parser.add_argument("--crop-to-original", "--crop_to_orig",
                        dest="crop_to_orig", action="store_true",
                        help="Center-crop output back to original matrix implied by --fov_scale.")

    # grid configuration
    parser.add_argument('--echo-groups', '--echo_groups', dest='echo_groups', type=int, default=1,
                        help='Number of echo groups (grid cols). Default 1 = no echo split.')
    parser.add_argument('--motion-groups', '--motion_groups', dest='motion_groups', type=int, default=1,
                        help='Number of motion groups (grid rows). Default 1 = no motion split.')

    parser.add_argument('--l2-coupling', '--l2_coupling', dest='l2_coupling',
                        action='store_true', help='L2 coupling between echoes')
    parser.add_argument('--show-progress', '--show_pbar', dest='show_pbar',
                        action='store_true', help='Show progress bar.')
    parser.add_argument('--device', type=int, default=0, help='GPU device.')
    parser.add_argument('--multi-gpu', '--multi_gpu', dest='multi_gpu', action='store_true',
                        help='Toggle multi-gpu. Overrides device option.')
    parser.add_argument('--fp16-communication', '--use_fp16_comm',
                        dest='use_fp16_comm', action='store_true',
                        help='Cast to fp16 during allreduce to reduce bandwidth.')
    parser.add_argument('--nufft-backend', choices=('sigpy',), default='sigpy')
    parser.add_argument(
        '--prepared-dir',
        type=str,
        default=None,
        help='Directory containing mps.hdr/.cfl and resp.hdr/.cfl.',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Directory for reconstructed CFL shards.',
    )
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('input_dir', type=str, help='Raw MICCAI dataset directory.')
    parser.add_argument('img_file', type=str, help='Output filename stem.')

    args = parser.parse_args(argv)

    # Verbose
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # MPI
    world = MPI.COMM_WORLD

    # per-node intra communicator
    intra = IntraNodeGroup(world)
    group_comm = intra.intra_node_comm

    world_rank = intra.world_rank
    world_size = intra.world_size
    group_rank = intra.group_rank
    group_size = intra.group_size
    num_nodes = intra.num_nodes
    group_ranks = intra.group_world_ranks  # world ranks on this node

    # 2D grid
    motion_groups = int(args.motion_groups)
    echo_groups   = int(args.echo_groups)

    if motion_groups <= 0 or echo_groups <= 0:
        if world_rank == 0:
            logging.error(f"Invalid groups: motion_groups={motion_groups}, echo_groups={echo_groups} (must be positive).")
        world.Abort(1)

    if num_nodes != motion_groups * echo_groups:
        if world_rank == 0:
            logging.error(
                f"Node grid mismatch: num_nodes={num_nodes} != motion_groups*echo_groups="
                f"{motion_groups}*{echo_groups}={motion_groups*echo_groups}. "
                "Please set --motion_groups and --echo_groups explicitly."
            )
        world.Abort(1)

    grid = NodeGrid2D(intra, motion_groups=motion_groups, echo_groups=echo_groups)

    # leader comm across nodes along echo axis (leaders only)
    leader_comm = grid.echo_axis_leader_comm   # None on non-leaders

    tagger = make_comm_tagger(world, stride=512)

    # bind to GPU
    if args.multi_gpu and CUPY_AVAILABLE:
        local_rank = intra.get_local_rank()
        device = sp.Device(local_rank)
        if cp is not None:
           cp.cuda.Device(local_rank).use()  # set device for cupy
        logging.info(f'World Rank {world_rank} (Node {intra.node_id} Grid {grid.coord} Rank {group_rank}): binding to GPU {local_rank}')
    else:
        device = sp.Device(args.device)
        logging.info(f'World Rank {world_rank} (Node {intra.node_id} Grid {grid.coord} Rank {group_rank}): using device {args.device}')

    if world_rank == 0:
        logging.info(f'[MPI] world_size={world_size}, num_nodes={num_nodes}')
    if group_rank == 0:
        logging.info(f'[Group {world_rank}] group_size={group_size}, ranks={group_ranks}')

    # NCCL group
    gpu_comm = GpuComm(group_comm, prefer_fp16=args.use_fp16_comm) if args.multi_gpu else None

    # Reading data
    if world_rank == 0:
        logging.info('Reading data.')

    ksp_file       = os.path.join(args.input_dir, 'ksp')
    coord_file     = os.path.join(args.input_dir, 'ktraj')
    dcf_file       = os.path.join(args.input_dir, 'dens')
    img_shape_file = os.path.join(args.input_dir, 'imageDim.txt')
    prepared_dir = args.prepared_dir or args.input_dir
    mps_file = os.path.join(prepared_dir, 'mps')
    resp_file = os.path.join(prepared_dir, 'resp')

    voxel_size_file = os.path.join(args.input_dir, 'voxelSize.txt')


    with open(img_shape_file, 'r') as f:
        l = f.readline()
        img_shape = [int(i) for i in l.split(':')[::-1]]

    orig_img_shape = img_shape.copy()

    with open(voxel_size_file, 'r') as f:
        l = f.readline()
        voxel_size = [float(i) for i in l.split(':')[::-1]]

    if group_rank == 0:
        coord_full = np.squeeze(cfl.read_cfl(coord_file)).real[..., ::-1]
        dcf_full   = np.squeeze(cfl.read_cfl(dcf_file)).real
        resp_full = np.squeeze(cfl.read_cfl(resp_file)).real
        resp = resp_full

        # ----------------------------
        # Retrospective undersampling
        #   1) discard outliers with margin=2
        #   2) keep first int(num_tr_total / R) trajectories
        # ----------------------------
        reduction_factor = int(args.acc)
        num_tr_total = int(resp_full.size)

        if world_rank == 0:
            logging.info(f'Undersampling (ACC={reduction_factor}X).')

        tr_keep = np.arange(num_tr_total, dtype=np.int64)

        # 1) discard outliers from the entire data (margin=2)
        margin = 2
        lo = np.percentile(resp_full, 0 + margin)
        hi = np.percentile(resp_full, 100 - margin)
        keep_mask = (resp_full > lo) & (resp_full < hi)
        tr_keep = tr_keep[keep_mask]

        # 2) truncate to first int(num_tr_total / R)
        # (this matches new version, which uses the original num_tr in the truncation)
        n_target = int(num_tr_total / max(reduction_factor, 1))
        tr_keep = tr_keep[:n_target]

        # Hack for dealing with abnormally large image shape
        num_ro = int(args.frac * coord_full.shape[-2])

        coord  = coord_full[tr_keep, :num_ro]
        dcf    = dcf_full[tr_keep, :num_ro]
        resp   = resp_full[tr_keep]

        # Double-check the issue with large image shape
        while (coord.max() > 1000):
            num_ro -= 1
            # ksp    = ksp[..., :num_ro]
            coord  = coord[:, :num_ro]
            dcf    = dcf[:, :num_ro]

        if world_rank == 0:
            logging.info('Scaling coordinates.')

        ndim = coord.shape[-1]
        for d in range(ndim):
            coord[..., d] *= voxel_size[d] * img_shape[d]

        # ----------------------------
        # FoV scaling
        # ----------------------------
        fov_scale = np.asarray(args.fov_scale, dtype=float)  # [z, y, x]
        if not np.allclose(fov_scale, 1.0):
            if np.any(fov_scale < 1.0):
                raise ValueError("FoV scale factors must be >= 1.0 for all axes (z y x).")

            if world_rank == 0:
                logging.info("Scaling FoV by factors (x,y,z) = (%.2f, %.2f, %.2f).",
                            fov_scale[-1], fov_scale[-2], fov_scale[-3])

            coord = coord.copy()
            coord[..., 0] *= fov_scale[-3]  # z
            coord[..., 1] *= fov_scale[-2]  # y
            coord[..., 2] *= fov_scale[-1]  # x

            img_shape = [
                int(img_shape[-3] * fov_scale[-3]),
                int(img_shape[-2] * fov_scale[-2]),
                int(img_shape[-1] * fov_scale[-1]),
            ]
        # ----------------------------
        # Preserve the research implementation's DCF normalization here.
        # ----------------------------
        dcf_max = float(np.max(dcf))
        if dcf_max > 0:
            dcf /= np.float32(dcf_max)

        meta = {
            'num_ro': num_ro,
            'img_shape': img_shape,
            'orig_img_shape': orig_img_shape,
            'voxel_size': voxel_size,
            'fov_scale': fov_scale.tolist(),
        }

    else:
        coord = None; dcf = None; resp = None; tr_keep = None; meta = None

    meta = group_comm.bcast(meta, root=0)
    num_ro = meta['num_ro']
    img_shape = meta['img_shape']

    orig_img_shape = meta['orig_img_shape']
    voxel_size = meta['voxel_size']
    fov_scale = np.asarray(meta['fov_scale'], dtype=float)

    coord = group_comm.bcast(coord, root=0)
    dcf = group_comm.bcast(dcf, root=0)
    resp = group_comm.bcast(resp, root=0)

    tr_keep = group_comm.bcast(tr_keep, root=0)

    if group_rank == 0:
        del coord_full, dcf_full, resp_full

    if world_rank == 0:
        logging.info(f'I/O small tensors done. num_ro={num_ro}, img_shape={img_shape}')

    # node read ksp/mps, read echo slices for local group, distribute coils inside group
    if group_rank == 0:
        ksp_hdr = cfl.read_cfl_header(ksp_file)
        E_total = KSpaceLayout.from_cfl_shape(ksp_hdr).echoes

    else:
        E_total = None
    E_total = group_comm.bcast(E_total, root=0)

    b0, b1, e0, e1 = grid.set_partitions(B_total=args.num_bins, E_total=E_total)
    B_local = b1 - b0
    E_local = e1 - e0

    # ----------------------------
    # Motion bin -> trajectory selection (reduce IO/memory)
    # ----------------------------
    bin_margin = 0  # keep consistent with TvmeReconstructor(..., margin=0)
    # 1) compute global bin_edges ONCE, then broadcast
    if group_rank == 0:
        bin_edges = np.percentile(resp, np.linspace(0 + bin_margin, 100 - bin_margin, args.num_bins + 1))
    else:
        bin_edges = None
    bin_edges = group_comm.bcast(bin_edges, root=0)

    # 2) compute idx_local (indices into current resp/coord/dcf) and tr_idx_local (ORIGINAL file indices for ksp)
    if group_rank == 0:
        local_mask = np.zeros(resp.shape, dtype=bool)
        for gb in range(b0, b1):
            local_mask |= (resp >= bin_edges[gb]) & (resp < bin_edges[gb + 1])

        idx_local = np.nonzero(local_mask)[0].astype(np.int64)

        # tr_keep must exist and store ORIGINAL TR indices in the ksp file (broadcast earlier)
        tr_idx_local = tr_keep[idx_local].astype(np.int64)
    else:
        idx_local = None
        tr_idx_local = None

    idx_local = group_comm.bcast(idx_local, root=0)
    tr_idx_local = group_comm.bcast(tr_idx_local, root=0)

    # 3) slice small tensors consistently on every rank
    coord = coord[idx_local]
    dcf  = dcf[idx_local]
    resp = resp[idx_local]

    if group_rank == 0:
        mps_hdr = cfl.read_cfl_header(mps_file)
        C_total = int(mps_hdr[-4])
        logging.info(f"Node#{intra.node_id} Grid {grid.coord} "
                     f"echo {e0}-{e1-1}/{E_total}, "
                     f"bins {b0}-{b1-1}/{args.num_bins}")
        coil_indices = split_coils(C_total, group_size)
        meta_shm = {
            'C_total': int(C_total),
            'E_local': int(E_local),
            'coil_indices': [idx.tolist() for idx in coil_indices]
        }
    else:
        meta_shm = None

    meta_shm = group_comm.bcast(meta_shm, root=0)
    C_total = meta_shm['C_total']
    E_local = meta_shm['E_local']
    coil_indices = [np.array(x, dtype=int) for x in meta_shm['coil_indices']]
    shm_rank = (gpu_comm.shm_comm.rank if gpu_comm is not None else group_rank)
    coil_idxs_local = coil_indices[shm_rank]

    # io timing
    iostart = time.perf_counter()

    # each rank read its local_e local_m
    ksp_local = cfl.read_kspace_shard(
        name=ksp_file,
        echo_start=e0,
        echo_end=e1,
        coil_indices=coil_idxs_local,
        readout=num_ro,
        trajectory_indices=tr_idx_local,
    ) # (E_local, C_local, N_tr_local, N_ro_eff)

    mps_local = cfl.read_mps_coils(
        name=mps_file,
        coil_indices=coil_idxs_local,
    ) # (C_local, X, Y, Z)

    if tuple(mps_local.shape[-3:]) != tuple(img_shape):
        raise ValueError(
            f"MPS grid {tuple(mps_local.shape[-3:])} does not match "
            f"the requested reconstruction grid {tuple(img_shape)}. "
            "Prepare sensitivity maps with the same FoV scale."
        )

    # ----------------------------
    # Data normalization
    #   (1) ksp /= global_max(|ksp|)
    #   (2) mps /= mpsSOS over ALL coils
    #   (3) dcf already normalized earlier (do NOT touch here)
    # ----------------------------

    # (1) ksp: global max over ALL ranks
    local_ksp_max = max_abs_cplx_chunked(ksp_local)
    global_ksp_max = world.allreduce(local_ksp_max, op=MPI.MAX)
    if global_ksp_max > 0:
        ksp_local /= np.float32(global_ksp_max)

    # (2) mps: SOS over ALL coils (within the node group)
    local_sos_sq = np.sum(np.abs(mps_local)**2, axis=0, dtype=np.float32)
    sos_sq = np.ascontiguousarray(local_sos_sq)
    group_comm.Allreduce(MPI.IN_PLACE, sos_sq, op=MPI.SUM)
    mpsSOS = np.sqrt(sos_sq, dtype=np.float32)
    mpsSOS = np.maximum(mpsSOS, np.float32(1e-12))
    mps_local /= mpsSOS[None, ...]

    ksp = ksp_local
    mps= mps_local
    C_local = mps.shape[0]

    # io timing
    ioend = time.perf_counter()
    if world_rank == 0:
        logging.info(f'Largest IO done in {ioend - iostart:.2f} seconds.')
        logging.info(f'Prepared shards (direct-IO): Node#{intra.node_id} Grid {grid.coord} E_local={E_local}, C_local={C_local}, num_bins={args.num_bins}')

    # Split between MPI nodes: make sure we split COILS of ksp and mps
    dual_q = np.zeros_like(ksp, dtype=np.complex64)
    # timing
    procstart = time.perf_counter()

    mrimg = TvmeReconstructor(ksp, coord, dcf, mps, resp, dual_q, B_local, args.l2_coupling,
                            max_iter=args.max_iter, lamda_m=args.lamda_m, lamda_e=args.lamda_e,
                            sigma=1/6, tau=1/6, tol=args.tol, margin=0,
                            device=device, E_total=E_total, e0=e0, e1=e1,
                            echo_left_peer=grid.echo_left_peer, echo_right_peer=grid.echo_right_peer,
                            B_total=args.num_bins, b0=b0, b1=b1,
                            motion_left_peer=grid.motion_left_peer, motion_right_peer=grid.motion_right_peer,
                            bin_edges=bin_edges,
                            world_comm=world, group_comm=group_comm, leader_comm=leader_comm,
                            nccl_group=gpu_comm, dist_grid=grid, tagger=tagger,
                            show_pbar=(args.show_pbar and world_rank == 0),
                            nufft_backend=args.nufft_backend).run()

    procend = time.perf_counter()
    if world_rank == 0:
        logging.info(f'Reconstruction done in {procend - procstart:.2f} seconds.')

    # move to CPU
    _is_cupy = CUPY_AVAILABLE and isinstance(mrimg, cp.ndarray)

    # mriimg: (B, E, X, Y, Z)
    mr_cpu = cp.asnumpy(mrimg) if _is_cupy else np.asarray(mrimg)

    # mps (C_local, X, Y, Z)
    X, Y, Z = map(int, mps.shape[1:])
    B, E = int(mr_cpu.shape[0]), int(mr_cpu.shape[1])

    img_cpu = np.zeros((B, 1, 1, E, 1, 1, X, Y, Z), dtype=mr_cpu.dtype)
    img_cpu[:, 0, 0, :, 0, 0, :, :, :] = mr_cpu

    if args.crop_to_orig and (not np.allclose(fov_scale, 1.0)):
        img_cpu = center_crop_zyx(img_cpu, orig_img_shape)

    os.makedirs(args.output_dir, exist_ok=True)
    out_base = os.path.join(
        args.output_dir,
        f"{args.img_file}_e{e0}-{e1-1}_m{b0}-{b1-1}",
    )

    if group_rank == 0:
        if world_rank == 0:
            logging.info(f"Writing output to {out_base}")
        cfl.write_cfl(out_base, img_cpu)

    if world_rank == 0:
        manifest = {
            "format_version": 1,
            "algorithm": "tvme",
            "nufft_backend": args.nufft_backend,
            "input_directory": os.path.abspath(args.input_dir),
            "prepared_directory": os.path.abspath(prepared_dir),
            "output_stem": args.img_file,
            "output_pattern": f"{args.img_file}_e{{e0}}-{{e1}}_m{{b0}}-{{b1}}",
            "parameters": {
                "readout_fraction": args.frac,
                "num_bins": args.num_bins,
                "num_echoes": E_total,
                "lambda_motion": args.lamda_m,
                "lambda_echo": args.lamda_e,
                "max_iter": args.max_iter,
                "tol": args.tol,
                "acceleration": args.acc,
                "fov_scale_zyx": list(args.fov_scale),
                "crop_to_original": args.crop_to_orig,
                "l2_coupling": args.l2_coupling,
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
                "density": "maximum",
            },
        }
        manifest_path = os.path.join(args.output_dir, "run_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")

    mainend = time.perf_counter()
    if world_rank == 0:
        logging.info(f'Main done in {mainend - mainstart:.2f} seconds.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
