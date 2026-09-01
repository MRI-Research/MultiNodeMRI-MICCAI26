#!/usr/bin/env python3
"""TVMW topology-aware PDHG reconstruction.

The motion and wavelet PDHG updates, PTWT transforms, and communication
schedule remain equivalent to the research implementation while data access
and NUFFT calls use the release interfaces.
"""

import argparse
import json
import logging
import math
import os
import time

import numpy as np
import ptwt
import sigpy as sp
import torch
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

class PtwtWaveletOps:
    """
    STABLE version: correctness first.
    - No custom torch stream.
    - DLPack boundary uses device-wide synchronize on both sides.
    """

    def __init__(self, *, E, img_shape_zyx, w2="db6", mode="zero",
                 level_w2=None, disable_tf32=True):
        self.E = int(E)
        self.Z, self.Y, self.X = [int(v) for v in img_shape_zyx]
        self.w2 = w2
        self.mode = mode
        self.level_w2 = level_w2

        if disable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    # ---------- dlpack bridges (GPU, zero-copy) ----------
    @staticmethod
    def _cp_to_torch(cp_arr):
        if CUPY_AVAILABLE and isinstance(cp_arr, cp.ndarray):
            # strongest barrier: ensure all prior CuPy work is done
            cp.cuda.runtime.deviceSynchronize()

        dl = cp_arr.__dlpack__() if hasattr(cp_arr, "__dlpack__") else cp_arr.toDlpack()
        t = torch.utils.dlpack.from_dlpack(dl)

        # optional extra safety (usually not needed, but keep for stability)
        if torch.is_tensor(t) and t.is_cuda:
            torch.cuda.synchronize(device=t.device)

        return t

    @staticmethod
    def _torch_to_cp(t):
        if not (torch.is_tensor(t) and t.is_cuda):
            return cp.asarray(t) if CUPY_AVAILABLE else np.asarray(t)

        # strongest barrier: ensure ALL torch work on this device is done
        torch.cuda.synchronize(device=t.device)

        dl = t.__dlpack__() if hasattr(t, "__dlpack__") else torch.utils.dlpack.to_dlpack(t)
        arr = cp.from_dlpack(dl) if hasattr(cp, "from_dlpack") else cp.fromDlpack(dl)

        # strongest barrier: make sure CuPy side sees fully produced data
        cp.cuda.runtime.deviceSynchronize()
        return arr

    # ---------- complex <-> 2ch float ----------
    def _cp_complex_to_torch_2ch_EZYX(self, u_cp):
        cp = __import__("cupy")
        assert u_cp.dtype == cp.complex64, f"expect complex64, got {u_cp.dtype}"

        # ensure contiguous (important for view/reshape correctness)
        if not u_cp.flags.c_contiguous:
            u_cp = cp.ascontiguousarray(u_cp)

        u_ri = u_cp.view(cp.float32).reshape(self.E, self.Z, self.Y, self.X, 2)
        t = self._cp_to_torch(u_ri)                 # (E,Z,Y,X,2) float32
        t = t.permute(0, 4, 1, 2, 3).contiguous()   # (E,2,Z,Y,X)
        return t

    def _torch_2ch_EZYX_to_cp_complex(self, t):
        cp = __import__("cupy")
        t = t.permute(0, 2, 3, 4, 1).contiguous()   # (E,Z,Y,X,2)
        u_ri_cp = self._torch_to_cp(t)              # cupy float32 (E,Z,Y,X,2)
        u_cp = u_ri_cp.view(cp.complex64).reshape(self.E, self.Z, self.Y, self.X)
        return u_cp

    # ---------- tree utils ----------
    def _tree_map(self, obj, fn):
        if torch.is_tensor(obj):
            return fn(obj)
        if isinstance(obj, dict):
            return {k: self._tree_map(v, fn) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            out = [self._tree_map(v, fn) for v in obj]
            return type(obj)(out)
        raise TypeError(f"Unsupported coeff container: {type(obj)}")

    def _tree_axpy_(self, dst, a, src):
        if torch.is_tensor(dst):
            dst.add_(src, alpha=float(a))
            return dst
        if isinstance(dst, dict):
            for k in dst.keys():
                self._tree_axpy_(dst[k], a, src[k])
            return dst
        if isinstance(dst, (list, tuple)):
            for i in range(len(dst)):
                self._tree_axpy_(dst[i], a, src[i])
            return dst
        raise TypeError(f"Unsupported coeff container: {type(dst)}")

    def _tree_zero_(self, obj):
        def zero_(t):
            t.zero_()
            return t
        return self._tree_map(obj, zero_)

    def _tree_project_linf_complex2ch_(self, obj, lam):
        if lam <= 0:
            return self._tree_zero_(obj)

        def proj_tensor(t):
            re = t[:, 0]
            im = t[:, 1]
            mag = torch.sqrt(re * re + im * im + 1e-12)
            scale = torch.clamp(lam / mag, max=1.0)
            t[:, 0].mul_(scale)
            t[:, 1].mul_(scale)
            return t

        def walk(o):
            if torch.is_tensor(o):
                return proj_tensor(o)
            if isinstance(o, dict):
                for k, v in o.items():
                    o[k] = walk(v)
                return o
            if isinstance(o, list):
                for i in range(len(o)):
                    o[i] = walk(o[i])
                return o
            if isinstance(o, tuple):
                o = list(o)
                for i in range(len(o)):
                    o[i] = walk(o[i])
                return tuple(o)
            raise TypeError(type(o))

        return walk(obj)

    # ---------- W2: 3D wavelet along (Z,Y,X) ----------
    def w2_fwd(self, u_cp):
        t = self._cp_complex_to_torch_2ch_EZYX(u_cp)
        coeffs = ptwt.wavedec3(t, self.w2, mode=self.mode, level=self.level_w2, axes=(-3,-2,-1))
        # extra safety: ensure coeffs are fully produced before returning (stability)
        torch.cuda.synchronize(device=t.device)
        return coeffs

    def w2_adj(self, coeffs):
        t = ptwt.waverec3(coeffs, self.w2, axes=(-3,-2,-1))
        torch.cuda.synchronize(device=t.device)
        return self._torch_2ch_EZYX_to_cp_complex(t)

    def w2_zeros_like(self, u_cp_example):
        c = self.w2_fwd(u_cp_example)
        return self._tree_map(c, lambda t: torch.zeros_like(t))


class TvmwReconstructor:
    def __init__(self, ksp, coord, dcf, mps, resp, dual_q, B_local,
                 l2_coupling=False,
                 lambda1=1e-6, lambda2=1e-6, lambda3=1e-6, sigma=0.1, tau=0.1,
                 max_iter=10, tol=0.01,device=sp.cpu_device, margin=2,
                 E_total=None, e0=None, e1=None, echo_left_peer=None, echo_right_peer=None,
                 B_total=None, b0=None, b1=None, motion_left_peer=None, motion_right_peer=None,
                 bin_edges=None,
                 world_comm=None, group_comm=None, leader_comm=None,
                 nccl_group=None, dist_grid=None, tagger=None,
                 show_pbar=True, nufft_backend="sigpy"):
        if isinstance(device, sp.Device):
            self.device = device
        else:
            self.device = sp.Device(device)

        self.xp = self.device.xp
        self.B = B_local
        self.E = ksp.shape[0]
        self.mps = sp.to_device(mps, self.device) # move mps to device
        self.C = int(self.mps.shape[0])

        assert ksp.ndim == 4, f"ksp must be (E_local,C_local,Ntr,Nro), got {ksp.shape}"
        assert mps.ndim == 4, f"mps must be (C_local,*,*,*), got {mps.shape}"
        assert ksp.shape[1] == mps.shape[0], f"C_local mismatch: ksp C={ksp.shape[1]}, mps C={mps.shape[0]}"

        self.sigma = sigma # PDHG
        self.tau = tau # PDHG
        self.l2_coupling = l2_coupling
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.max_iter = max_iter
        self.tol = tol

        self.world_comm = world_comm
        self.group_comm = group_comm
        self.leader_comm = leader_comm
        self.nccl_group = nccl_group

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

        self.dist_grid = dist_grid
        self.tagger = tagger

        if world_comm is not None:
            self.show_pbar = show_pbar and (world_comm.Get_rank() == 0)
        else:
            self.show_pbar = show_pbar

        self.img_shape = list(self.mps.shape[1:])
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


        self.wv = PtwtWaveletOps(
            E=self.E,
            img_shape_zyx=self.img_shape,  # (Z,Y,X)
            w2="db6",
            mode="zero",
            level_w2=None,
            disable_tf32=True,
        )

    def _w1_db1_level1_fwd(self, u_BEZYX, ghost_left_last, ghost_right_first):
        """
        u_BEZYX: (B, E_local, Z, Y, X)
        ghost_left_last:  (B, Z, Y, X) = x_{e0-1}
        ghost_right_first:(B, Z, Y, X) = x_{e1}
        Return coeff_BEZYX: (B, E_local, Z, Y, X)
        global even g: stores a_g
        global odd  g: stores d_g
        """
        coeff = self.xp.zeros_like(u_BEZYX)
        inv_sqrt2 = (1.0 / math.sqrt(2.0))

        E_local = u_BEZYX.shape[1]
        for j in range(E_local):
            g = self.e0 + j
            if (g % 2) == 0:
                # a_g = (x_g + x_{g+1}) / sqrt2
                x0 = u_BEZYX[:, j]
                if j + 1 < E_local:
                    x1 = u_BEZYX[:, j + 1]
                else:
                    x1 = ghost_right_first if (not self.is_global_last) else 0.0
                coeff[:, j] = (x0 + x1) * inv_sqrt2
            else:
                # d_g = (-x_{g-1} + x_g) / sqrt2
                x1 = u_BEZYX[:, j]
                if j - 1 >= 0:
                    x0 = u_BEZYX[:, j - 1]
                else:
                    x0 = ghost_left_last if (not self.is_global_first) else 0.0
                coeff[:, j] = (-x0 + x1) * inv_sqrt2
        return coeff

    def _w1_db1_level1_adj(self, p_BEZYX, ghost_left_last, ghost_right_first):
        """
        p_BEZYX stores:
        global even g: a_g
        global odd  g: d_g

        Return x_BEZYX = W1^H p, shape (B, E_local, Z, Y, X)
        """
        out = self.xp.zeros_like(p_BEZYX)
        inv_sqrt2 = (1.0 / math.sqrt(2.0))
        E_local = p_BEZYX.shape[1]

        for j in range(E_local):
            g = self.e0 + j
            if (g % 2) == 0:
                # x_g = (a_g - d_{g+1})/sqrt2
                a = p_BEZYX[:, j]
                if j + 1 < E_local:
                    d_next = p_BEZYX[:, j + 1]
                else:
                    d_next = ghost_right_first if (not self.is_global_last) else 0.0
                out[:, j] = (a - d_next) * inv_sqrt2
            else:
                # x_g = (a_{g-1} + d_g)/sqrt2
                d = p_BEZYX[:, j]
                if j - 1 >= 0:
                    a_prev = p_BEZYX[:, j - 1]
                else:
                    a_prev = ghost_left_last if (not self.is_global_first) else 0.0
                out[:, j] = (a_prev + d) * inv_sqrt2
        return out

    # Primal-Dual Algorithm
    def pdinit(self, mrimg):
        dual_p_m = self.xp.zeros_like(mrimg)
        dual_p_w1 = self.xp.zeros_like(mrimg)   # (B, E_local, Z, Y, X) complex
        dual_p_w2 = [None] * self.B
        dual_q = self.dual_q
        return dual_p_m, dual_p_w1, dual_p_w2, dual_q

    def pdhg(self, primal_u, primal_u_old, primal_u_tmp, dual_p_m, dual_p_w1, dual_p_w2, dual_q, it):

        # p: dual variable for total variation
        # q: dual variable for data term
        # u: primal variable
        primal_u_old[...] = primal_u_tmp
        req_absp_sq = None

        # ---- motion halo for delta_m u ----
        # communicate primal_u for diff_m
        ghost_m_right_first = self.xp.empty_like(primal_u[0])   # (E_local, Z,Y,X)
        ghost_m_right_first = self.xp.ascontiguousarray(ghost_m_right_first)
        sendbuf_m = self.xp.ascontiguousarray(primal_u[0])      # send my first bin to left

        tag_mdelta = self.tagger(it, axis="motion", kind="delta")

        if CUPY_AVAILABLE and isinstance(sendbuf_m, cp.ndarray):
            cp.cuda.get_current_stream().synchronize()

        reqs_mdelta = self.dist_grid.halo_exchange(
            sendbuf_m, ghost_m_right_first,
            send_peer=self.motion_left_peer,
            recv_peer=self.motion_right_peer,
            tag=tag_mdelta
        )

        # ---- start halo for W1 forward (u) using dist_comm.halo_exchange ----
        # comm primal_u for w1
        ghost_left_last_u = self.xp.empty_like(primal_u[:, 0])   # (B, Z, Y, X)
        ghost_right_first_u = self.xp.empty_like(primal_u[:, 0])   # (B, Z, Y, X)
        ghost_left_last_u = self.xp.ascontiguousarray(ghost_left_last_u)
        ghost_right_first_u = self.xp.ascontiguousarray(ghost_right_first_u)

        # send my first echo -> left, recv right's first echo -> ghostR_u
        send_first_u = self.xp.ascontiguousarray(primal_u[:, 0])
        tag_w1u_rf = self.tagger(it, axis="echo", kind="w1u_rf")

        if CUPY_AVAILABLE and isinstance(send_first_u, cp.ndarray):
            cp.cuda.get_current_stream().synchronize()

        reqs_w1u = self.dist_grid.halo_exchange(
            send_first_u, ghost_right_first_u,
            send_peer=self.echo_left_peer,
            recv_peer=self.echo_right_peer,
            tag=tag_w1u_rf
        )

        # send my last echo -> right, recv left's last echo -> ghostL_u
        send_last_u = self.xp.ascontiguousarray(primal_u[:, -1])
        tag_w1u_ll = self.tagger(it, axis="echo", kind="w1u_ll")

        if CUPY_AVAILABLE and isinstance(send_last_u, cp.ndarray):
            cp.cuda.get_current_stream().synchronize()

        reqs_w1u += self.dist_grid.halo_exchange(
            send_last_u, ghost_left_last_u,
            send_peer=self.echo_right_peer,
            recv_peer=self.echo_left_peer,
            tag=tag_w1u_ll
        )

        # compute dual_p_w2
        for b in range(self.B):
            w2 = self.wv.w2_fwd(primal_u[b])       # coeff-tree (torch tensors on GPU)
            if dual_p_w2[b] is None:
                dual_p_w2[b] = self.wv.w2_zeros_like(primal_u[b])
            self.wv._tree_axpy_(dual_p_w2[b], self.sigma, w2)
            self.wv._tree_project_linf_complex2ch_(dual_p_w2[b], self.lambda3)

        for b in range(self.B):
            tmp =  self.xp.zeros_like(self.dual_q[b])
            for c in range(self.C):
                for e in range(self.E):
                    mps_c = self.mps[c]
                    tmp[e, c] = (
                        self.nufft.forward(primal_u[b, e] * mps_c, b)
                        - self.bksp[b][e][c]
                    )

            # proximal operator
            dual_q[b] = (dual_q[b] + self.sigma * tmp)/(1 + self.sigma)

        # compute diff_m
        diff_m = self.xp.zeros_like(dual_p_m)
        if self.B > 1:
            diff_m[:-1] = primal_u[1:] - primal_u[:-1]
        MPI.Request.Waitall(reqs_mdelta)
        if not self.is_motion_last:
            diff_m[-1] = ghost_m_right_first - primal_u[-1]
        else:
            diff_m[-1] = 0

        # proximal operator
        # compute dual_p_m
        dual_p_m = dual_p_m + self.sigma * diff_m

        # 02/23/2026: l2 coupling

        # 1-3-A) Proximal mapping: start async allreduce for L2-coupling
        if self.l2_coupling: # L2-coupling: # TODO: test
            # local sum of ||p_m||^2 over echoes
            absp_sq = self.xp.sum(self.xp.abs(dual_p_m) ** 2, axis=1, keepdims=True)
            absp_sq = self.xp.ascontiguousarray(absp_sq)

            # global Allreduce over all echoes via leader_comm, then Bcast inside group
            if self.leader_comm is not None:
                if CUPY_AVAILABLE and isinstance(absp_sq, cp.ndarray):
                    cp.cuda.get_current_stream().synchronize()

                # non-blocking Allreduce
                req_absp_sq = self.leader_comm.Iallreduce(MPI.IN_PLACE, absp_sq, op=MPI.SUM)

        else: # L1-coupling
                absp = self.xp.abs(dual_p_m)
                dual_p_m = dual_p_m/self.xp.maximum(1, absp/self.lambda1)

        # compute w1
        # ---- finish halo then do W1 forward/update ----
        MPI.Request.Waitall(reqs_w1u)
        # boundary safety (optional; ghost may be garbage if PROC_NULL doesn't write)
        if self.is_global_first:
            ghost_left_last_u[...] = 0
        if self.is_global_last:
            ghost_right_first_u[...] = 0
        w1_coeff = self._w1_db1_level1_fwd(primal_u, ghost_left_last_u, ghost_right_first_u)

        # Dual ascent
        # update dual_p_w1
        dual_p_w1 = dual_p_w1 + self.sigma * w1_coeff

        # Projection onto L_inf ball radius=lambda2 (complex)
        if self.lambda2 > 0:
            absw1 = self.xp.abs(dual_p_w1)
            dual_p_w1 = dual_p_w1 / self.xp.maximum(1, absw1 / self.lambda2)
        else:
            dual_p_w1 = self.xp.zeros_like(dual_p_w1)

        # ---- start halo for W1 adj (p) using dist_comm.halo_exchange ----
        # communicate dual_p_w1 for w1 adj
        ghost_left_last_p = self.xp.empty_like(dual_p_w1[:, 0])   # (B, Z, Y, X)
        ghost_right_first_p = self.xp.empty_like(dual_p_w1[:, 0])   # (B, Z, Y, X)
        ghost_left_last_p = self.xp.ascontiguousarray(ghost_left_last_p)
        ghost_right_first_p = self.xp.ascontiguousarray(ghost_right_first_p)

        send_first_p = self.xp.ascontiguousarray(dual_p_w1[:, 0])
        tag_w1p_rf = self.tagger(it, axis="echo", kind="w1p_rf")

        if CUPY_AVAILABLE and isinstance(send_first_p, cp.ndarray):
            cp.cuda.get_current_stream().synchronize()

        reqs_w1p = self.dist_grid.halo_exchange(
            send_first_p, ghost_right_first_p,
            send_peer=self.echo_left_peer,
            recv_peer=self.echo_right_peer,
            tag=tag_w1p_rf
        )

        send_last_p = self.xp.ascontiguousarray(dual_p_w1[:, -1])
        tag_w1p_ll = self.tagger(it, axis="echo", kind="w1p_ll")

        if CUPY_AVAILABLE and isinstance(send_last_p, cp.ndarray):
            cp.cuda.get_current_stream().synchronize()

        reqs_w1p += self.dist_grid.halo_exchange(
            send_last_p, ghost_left_last_p,
            send_peer=self.echo_right_peer,
            recv_peer=self.echo_left_peer,
            tag=tag_w1p_ll
        )

        # 1-3-B) Proximal mapping: wait comm, Bcast and apply prox
        if self.l2_coupling:
            if self.leader_comm is not None and req_absp_sq is not None:
                req_absp_sq.Wait()
            if self.group_comm is not None:
                if CUPY_AVAILABLE and isinstance(absp_sq, cp.ndarray):
                    cp.cuda.get_current_stream().synchronize()
                self.group_comm.Bcast(absp_sq, root=0)
            absp = self.xp.sqrt(absp_sq)
            denom = self.xp.maximum(self.xp.asarray(1.0, dtype=absp.dtype), absp / self.lambda1)
            dual_p_m = dual_p_m / denom

        # ---- motion halo for div: need ghost from left for first bin ----
        # communicate dual_p_m for div_m
        ghost_m_left_last = self.xp.empty_like(dual_p_m[0])     # (E_local,Z,Y,X)
        ghost_m_left_last = self.xp.ascontiguousarray(ghost_m_left_last)
        sendbuf_m = self.xp.ascontiguousarray(dual_p_m[-1])     # send my last p to right

        tag_mdiv = self.tagger(it, axis="motion", kind="div")

        if CUPY_AVAILABLE and isinstance(sendbuf_m, cp.ndarray):
            cp.cuda.get_current_stream().synchronize()

        reqs_mdiv = self.dist_grid.halo_exchange(
            sendbuf_m, ghost_m_left_last,
            send_peer=self.motion_right_peer,
            recv_peer=self.motion_left_peer,
            tag=tag_mdiv
        )

        # (FS)^H dual_q
        tmp = self.xp.zeros_like(primal_u)
        for b in range(self.B):
            for c in range(self.C):
                for e in range(self.E):
                    mps_c = self.mps[c]
                    tmp[b, e] += (
                        self.nufft.adjoint(
                            self.bdcf[b] * dual_q[b][e][c],
                            b,
                        )
                        * self.xp.conj(mps_c)
                    )

        # ------------------ (A) start coil-sum allreduce (do NOT wait here) ------------------
        # streams/events for overlap (NCCL path)
        compute_stream = cp.cuda.get_current_stream() if CUPY_AVAILABLE else None
        ev_tmp_ready = None
        ev_reduce_done = None

        need_reduce = (self.group_comm is not None and self.group_comm.Get_size() > 1)

        reduce_kind = None
        reduce_req  = None  # MPI request if using MPI Iallreduce

        if need_reduce:
            # ensure contiguous only if needed
            if CUPY_AVAILABLE and isinstance(tmp, cp.ndarray) and (not tmp.flags.c_contiguous):
                tmp = cp.ascontiguousarray(tmp)

            # Prefer NCCL if enabled
            if (self.nccl_group is not None
                    and getattr(self.nccl_group, "enabled", False)
                    and CUPY_AVAILABLE and isinstance(tmp, cp.ndarray)):

                # Record an event on compute stream AFTER tmp kernels are enqueued.
                # This does NOT block host; it just inserts a marker into the stream.
                ev_tmp_ready = cp.cuda.Event()
                ev_tmp_ready.record(compute_stream)

                # Make NCCL comm stream wait until tmp is ready (no host sync).
                self.nccl_group.stream.wait_event(ev_tmp_ready)

                # Enqueue NCCL allreduce on comm stream (async)
                self.nccl_group.alreduce_sum_(tmp)
                reduce_kind = "nccl"

                # Record "reduce done" event on NCCL stream (no host sync)
                ev_reduce_done = cp.cuda.Event()
                ev_reduce_done.record(self.nccl_group.stream)

            else:
                # MPI nonblocking allreduce
                if CUPY_AVAILABLE and isinstance(tmp, cp.ndarray):
                    # conservative: wait tmp kernels done before MPI touches device pointer
                    cp.cuda.get_current_stream().synchronize()

                reduce_req = self.group_comm.Iallreduce(MPI.IN_PLACE, tmp, op=MPI.SUM)
                reduce_kind = "mpi"
        # -------------------------------------------------------------------------------------


        divp_w2 = self.xp.zeros_like(primal_u)
        for b in range(self.B):
            divp_w2[b] = self.wv.w2_adj(dual_p_w2[b])     # cupy complex (E,Z,Y,X)

        # ---- finish halo then do W1 adj ----
        MPI.Request.Waitall(reqs_w1p)

        if self.is_global_first:
            ghost_left_last_p[...] = 0
        if self.is_global_last:
            ghost_right_first_p[...] = 0

        divp_w1 = self._w1_db1_level1_adj(dual_p_w1, ghost_left_last_p, ghost_right_first_p)

        # compute divp_m
        divp_m = self.xp.zeros_like(primal_u)
        # local part
        if self.B > 1:
            divp_m[1:] = dual_p_m[1:] - dual_p_m[:-1]
        MPI.Request.Waitall(reqs_mdiv)
        # first local bin
        if self.is_motion_first:
            divp_m[0] = dual_p_m[0]
        else:
            divp_m[0] = dual_p_m[0] - ghost_m_left_last

        # ------------------ (B) finish coil-sum allreduce (wait here) ------------------
        if reduce_kind == "nccl":
            # Make compute stream wait for NCCL completion (GPU-side dependency, no host block)
            compute_stream.wait_event(ev_reduce_done)
        elif reduce_kind == "mpi":
            reduce_req.Wait()
        # --------------------------------------------------------------------------------

        sp.axpy(primal_u_tmp, -self.tau, tmp  - divp_m  + divp_w1 + divp_w2)

        ### @AUXILIARY UPDATE ###
        primal_u = 2*primal_u_tmp - primal_u_old
        return primal_u, primal_u_old, primal_u_tmp, dual_p_m, dual_p_w1, dual_p_w2, dual_q

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
            with tqdm(total=self.max_iter, desc='ReconTVMW',
                      disable=not self.show_pbar) as pbar:

                with self.device:
                    mrimg = self.xp.zeros([self.B] + [self.E] + self.img_shape,
                                         dtype=self.mps.dtype)

                    dual_p_m, dual_p_w1,dual_p_w2, dual_q = self.pdinit(mrimg)

                    primal_u_old = self.xp.zeros_like(mrimg)
                    primal_u_tmp = self.xp.zeros_like(mrimg)
                    # timer
                    self._sync_all()
                    t0 = time.perf_counter()

                    for it in range(self.max_iter):
                        mrimg_od = mrimg.copy()

                        mrimg, primal_u_old, primal_u_tmp, dual_p_m, dual_p_w1, dual_p_w2, dual_q = \
                        self.pdhg(mrimg, primal_u_old, primal_u_tmp, dual_p_m, dual_p_w1, dual_p_w2, dual_q, it)
                        difference = mrimg_od - mrimg
                        numerator_local = self.xp.vdot(
                            difference.ravel(),
                            difference.ravel(),
                        ).real
                        denominator_local = self.xp.vdot(
                            mrimg_od.ravel(),
                            mrimg_od.ravel(),
                        ).real

                        buf = self.xp.empty((2,), dtype=self.xp.float32)
                        buf[0] = numerator_local.astype(buf.dtype, copy=False)
                        buf[1] = denominator_local.astype(buf.dtype, copy=False)

                        if (
                            self.world_comm is not None
                            and self.group_comm is not None
                            and self.group_comm.Get_rank() != 0
                        ):
                            buf[0] = self.xp.float32(0.0)
                            buf[1] = self.xp.float32(0.0)

                        # Stage through host memory so every rank observes the
                        # same completed reduction before deciding to stop.
                        if CUPY_AVAILABLE and self.xp is cp:
                            buf_host = cp.asnumpy(buf)
                        else:
                            buf_host = np.asarray(buf)

                        if self.world_comm is not None:
                            self.world_comm.Allreduce(
                                MPI.IN_PLACE,
                                buf_host,
                                op=MPI.SUM,
                            )

                        numerator = float(buf_host[0])
                        denominator = float(buf_host[1])
                        if (
                            np.isfinite(numerator)
                            and np.isfinite(denominator)
                        ):
                            numerator_norm = np.sqrt(max(numerator, 0.0))
                            denominator_norm = np.sqrt(max(denominator, 0.0))
                            global_tol = float(
                                numerator_norm / max(denominator_norm, 1e-12)
                            )
                        else:
                            global_tol = float("inf")
                        if self.show_pbar:
                            pbar.set_postfix(tol=global_tol)
                        if global_tol < self.tol:
                            break

                        pbar.update()

                    # timer
                    self._sync_all()
                    t1 = time.perf_counter()

                    t_local = t1 - t0
                    t_global = self.world_comm.allreduce(t_local, op=MPI.MAX) if self.world_comm is not None else t_local
                    if self.show_pbar:
                        logging.info(f"[PDHG iterations only] {t_global:.2f} s (max over ranks)")
                    done = True
        return mrimg


def main(argv=None) -> int:
    # timer
    mainstart = time.perf_counter()

    parser = argparse.ArgumentParser(description='TVMW topology-aware PDHG reconstruction.')
    parser.add_argument('--readout-fraction', '--frac', dest='frac', type=float, default=0.98,
                        help='Readout fractions.')
    parser.add_argument('--num-bins', '--num_bins', dest='num_bins', type=int, default=6,
                        help='Number of phases.')
    parser.add_argument('--lambda-motion', '--lambda1', dest='lambda1', type=float, default=1e-6,
                        help='Regularization for motion.')
    parser.add_argument('--lambda-echo-wavelet', '--lambda2', dest='lambda2', type=float, default=1e-6,
                        help='Regularization for the db1 echo wavelet.')
    parser.add_argument('--lambda-spatial-wavelet', '--lambda3', dest='lambda3', type=float, default=1e-6,
                        help='Regularization for the db6 spatial wavelet.')
    parser.add_argument('--max-iter', '--max_iter', dest='max_iter', type=int, default=300,
                        help='Maximum epochs.')
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

    # Match the stable PTWT reference path.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

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
        # Hack for dealing with abnormally large image shape
        num_ro = int(args.frac * coord_full.shape[-2])

        coord = coord_full[:, :num_ro]
        dcf = dcf_full[:, :num_ro]
        resp = resp_full

        # Double-check the issue with large image shape
        while (coord.max() > 1000):
            num_ro -= 1
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
        meta = {
            'num_ro': num_ro,
            'img_shape': img_shape,
            'orig_img_shape': orig_img_shape,
            'voxel_size': voxel_size,
            'fov_scale': fov_scale.tolist(),
        }

    else:
        coord = None; dcf = None; resp = None; meta = None

    meta = group_comm.bcast(meta, root=0)
    num_ro = meta['num_ro']
    img_shape = meta['img_shape']

    orig_img_shape = meta['orig_img_shape']
    voxel_size = meta['voxel_size']
    fov_scale = np.asarray(meta['fov_scale'], dtype=float)

    coord = group_comm.bcast(coord, root=0)
    dcf = group_comm.bcast(dcf, root=0)
    resp = group_comm.bcast(resp, root=0)

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
    bin_margin = 0  # keep consistent with TvmwReconstructor(..., margin=0)
    # 1) compute global bin_edges ONCE, then broadcast
    if group_rank == 0:
        bin_edges = np.percentile(resp, np.linspace(0 + bin_margin, 100 - bin_margin, args.num_bins + 1))
    else:
        bin_edges = None
    bin_edges = group_comm.bcast(bin_edges, root=0)

    # 2) Compute original trajectory indices for this node's motion shard.
    if group_rank == 0:
        local_mask = np.zeros(resp.shape, dtype=bool)
        for gb in range(b0, b1):
            local_mask |= (resp >= bin_edges[gb]) & (resp < bin_edges[gb + 1])

        idx_local = np.nonzero(local_mask)[0].astype(np.int64)

        tr_idx_local = idx_local.copy()
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
    #   (3) dcf /= global_max(dcf) after local motion-bin selection
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
    mpsSOS[mpsSOS == 0] = np.float32(1.0)
    mps_local /= mpsSOS[None, ...]

    # (3) dcf: preserve the legacy world-wide maximum after motion sharding
    local_dcf_max = float(np.max(dcf)) if dcf.size else 0.0
    global_dcf_max = world.allreduce(local_dcf_max, op=MPI.MAX)
    if global_dcf_max > 0:
        dcf /= np.float32(global_dcf_max)

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

    mrimg = TvmwReconstructor(ksp, coord, dcf, mps, resp, dual_q,
                            B_local, args.l2_coupling,
                            max_iter=args.max_iter, lambda1=args.lambda1, lambda2=args.lambda2,
                            lambda3=args.lambda3,
                            sigma=1/6, tau=1/6, tol=0.001, margin=0,
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

    # mrimg: (B, E, Z, Y, X)
    mr_cpu = cp.asnumpy(mrimg) if _is_cupy else np.asarray(mrimg)

    # mps: (C_local, Z, Y, X)
    Z, Y, X = map(int, mps.shape[1:])
    B, E = int(mr_cpu.shape[0]), int(mr_cpu.shape[1])

    img_cpu = np.zeros((B, 1, 1, E, 1, 1, Z, Y, X), dtype=mr_cpu.dtype)
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
            "algorithm": "tvmw",
            "nufft_backend": args.nufft_backend,
            "input_directory": os.path.abspath(args.input_dir),
            "prepared_directory": os.path.abspath(prepared_dir),
            "output_stem": args.img_file,
            "output_pattern": f"{args.img_file}_e{{e0}}-{{e1}}_m{{b0}}-{{b1}}",
            "parameters": {
                "readout_fraction": args.frac,
                "num_bins": args.num_bins,
                "lambda_motion": args.lambda1,
                "lambda_echo_wavelet": args.lambda2,
                "lambda_spatial_wavelet": args.lambda3,
                "max_iter": args.max_iter,
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
                "density": "global_maximum_after_motion_sharding",
            },
        }
        manifest_path = os.path.join(args.output_dir, "run_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")

    # timer
    mainend = time.perf_counter()
    if world_rank == 0:
        logging.info(f'Main done in {mainend - mainstart:.2f} seconds.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
