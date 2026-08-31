"""Topology-aware MPI/NCCL communication for the reconstruction algorithms.

- Multi-node / multi-GPU communication helpers for a 2D echo/motion grid.
- Naming:
    - egid: echo group id (grid column)
    - mgid: motion group id (grid row)
    - grid coord is (egid, mgid)
    - node_id is a stable per-host index (0..num_nodes-1)
"""

import os
import socket
import logging
import numpy as np
from mpi4py import MPI

# Optional CuPy/NCCL
try:
    import cupy as cp
    _CUPY_AVAILABLE = True
    try:
        from cupy.cuda import nccl
        _NCCL_AVAILABLE = hasattr(nccl, "get_unique_id")
        if not _NCCL_AVAILABLE:
            nccl = None
    except Exception:
        nccl = None
        _NCCL_AVAILABLE = False
except Exception:
    cp = None
    nccl = None
    _CUPY_AVAILABLE = False
    _NCCL_AVAILABLE = False


def split_range(N: int, G: int, g: int):
    """Split [0..N) into G groups, return g-th group's [start, end)."""
    if G <= 0:
        raise ValueError("G must be positive.")
    g = int(g)
    s = (N * g) // G
    e = (N * (g + 1)) // G
    return int(s), int(e)


def make_comm_tagger(comm: MPI.Comm, stride: int = 256):
    """
    Robust tag generator:
    - Avoid collisions among different (axis, kind) within the same iteration.
    - Prevent exceeding MPI_TAG_UB by using a modulo 'period'.

    Usage:
        tagger = make_comm_tagger(world)
        tag = tagger(it, axis="echo", kind="delta")
    """
    tag_ub = int(comm.Get_attr(MPI.TAG_UB))
    # Reserve a small 'off' space for (axis, kind)
    axis_id = {"echo": 0, "motion": 1}
    kind_id = {
                "delta": 1,
                "div": 2,
                "norm": 3,
                # W1 echo-halo: u (primal) and p (dual)
                "w1u_rf": 4,   # u: send first -> recv right-first
                "w1u_ll": 5,   # u: send last  -> recv left-last
                "w1p_rf": 6,   # p: send first -> recv right-first
                "w1p_ll": 7,   # p: send last  -> recv left-last
                }
    # off = axis*32 + kind  => < 64 for current set
    max_off = 64

    stride = int(stride)
    if stride <= max_off:
        stride = max_off + 1

    if stride > (tag_ub + 1):
        stride = tag_ub + 1

    if stride <= max_off:
        raise RuntimeError(f"MPI_TAG_UB too small: TAG_UB={tag_ub}, need > {max_off}")

    period = max(1, (tag_ub + 1) // stride)

    def tag(it: int, axis: str, kind: str) -> int:
        a = axis_id[axis]
        k = kind_id[kind]
        off = a * 32 + k
        return (int(it) % period) * stride + off

    return tag


class IntraNodeGroup:
    """
    Group MPI ranks by hostname (one group per physical node).
    Provides:
      - intra_node_comm: ranks on the same host
      - node_leader_comm: rank0 of each host (leader-only comm)
      - node_id: stable node index [0..num_nodes-1]
      - world_ranks_by_node_id: list of world ranks for each node_id
    """
    def __init__(self, world: MPI.Comm):
        self.world = world
        self.world_rank = world.Get_rank()
        self.world_size = world.Get_size()

        host = socket.gethostname()
        hosts = world.allgather(host)
        self._hosts = hosts

        # stable node_id assignment based on first occurrence order in hosts[]
        unique_hosts = []
        host2node = {}
        for h in hosts:
            if h not in host2node:
                host2node[h] = len(unique_hosts)
                unique_hosts.append(h)

        self.node_id = host2node[host]
        self.num_nodes = len(unique_hosts)

        # ranks on the same node
        self.intra_node_comm = world.Split(color=self.node_id, key=self.world_rank)
        self.group_rank = self.intra_node_comm.Get_rank()
        self.group_size = self.intra_node_comm.Get_size()

        # leader-only comm (leaders ordered by node_id for stability)
        if self.num_nodes > 1:
            color = 0 if self.group_rank == 0 else MPI.UNDEFINED
            raw = world.Split(color=color, key=self.node_id)
            self.node_leader_comm = None if raw == MPI.COMM_NULL else raw
        else:
            self.node_leader_comm = None

        # world ranks per node (built locally on every rank, no gather needed)
        world_ranks_by_node = [[] for _ in range(self.num_nodes)]
        for r, h in enumerate(hosts):
            nid = host2node[h]
            world_ranks_by_node[nid].append(r)
        self.world_ranks_by_node_id = world_ranks_by_node
        self.group_world_ranks = self.world_ranks_by_node_id[self.node_id]

    def get_local_rank(self) -> int:
        """GPU-local rank. Prefer env vars; fallback to host scan."""
        for k in ("OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID",
                  "MV2_COMM_WORLD_LOCAL_RANK", "LOCAL_RANK"):
            v = os.environ.get(k)
            if v is not None:
                return int(v)
        logging.warning("Cannot find local rank from env, falling back to host scan.")
        my_host = self._hosts[self.world_rank]
        return sum(1 for i, h in enumerate(self._hosts[:self.world_rank]) if h == my_host)

    def peer_world_rank(self, peer_node_id: int, idx_in_node: int = None) -> int:
        """
        Map (peer_node_id, idx_in_node) -> world rank.
        Default idx_in_node uses self.group_rank (i.e., same intra-node rank index).
        If peer node doesn't have that index, returns MPI.PROC_NULL.
        """
        if peer_node_id is None or peer_node_id == MPI.PROC_NULL:
            return MPI.PROC_NULL
        peer_node_id = int(peer_node_id)
        if peer_node_id < 0 or peer_node_id >= self.num_nodes:
            return MPI.PROC_NULL
        if idx_in_node is None:
            idx_in_node = self.group_rank
        idx_in_node = int(idx_in_node)
        ranks = self.world_ranks_by_node_id[peer_node_id]
        if idx_in_node < 0 or idx_in_node >= len(ranks):
            return MPI.PROC_NULL
        return int(ranks[idx_in_node])


class GpuComm:
    """
    NCCL allreduce within one node (intra_node_comm).
    """
    def __init__(self, intra_node_comm: MPI.Comm, prefer_fp16: bool = False):
        self.comm_mpi = intra_node_comm
        self.size = intra_node_comm.Get_size()
        self.rank = intra_node_comm.Get_rank()
        self.prefer_fp16 = bool(prefer_fp16)

        self.enabled = bool(_CUPY_AVAILABLE and _NCCL_AVAILABLE and self.size > 1)
        self.comm_nccl = None
        if self.enabled:
            if self.rank == 0:
                uid = nccl.get_unique_id()
            else:
                uid = None
            uid = intra_node_comm.bcast(uid, root=0)
            self.comm_nccl = nccl.NcclCommunicator(self.size, uid, self.rank)

        self.stream = cp.cuda.Stream(non_blocking=True) if self.enabled else None

        # shared memory comm (optional)
        try:
            self.shm_comm = intra_node_comm.Split_type(MPI.COMM_TYPE_SHARED, key=intra_node_comm.rank)
        except Exception:
            self.shm_comm = intra_node_comm

    def _dtype_to_nccl(self, dt):
        if dt == cp.float16:
            return nccl.NCCL_FLOAT16
        if dt == cp.float32:
            return nccl.NCCL_FLOAT32
        if dt == cp.float64:
            return nccl.NCCL_FLOAT64
        if dt == cp.int32:
            return nccl.NCCL_INT32
        return None

    def alreduce_sum_(self, arr, stream=None):
        """
        Stable in-place allreduce sum.
        - Fixes non-contiguous buffers.
        - Adds strong GPU sync before/after NCCL to avoid race (stable > performance).
        """
        if stream is None:
            stream = cp.cuda.get_current_stream()

        # fallback: CPU/MPI allreduce
        if (not self.enabled) or self.size == 1:
            buf = cp.asnumpy(arr) if (_CUPY_AVAILABLE and isinstance(arr, cp.ndarray)) \
                else np.ascontiguousarray(np.asarray(arr))
            self.comm_mpi.Allreduce(MPI.IN_PLACE, buf, op=MPI.SUM)
            if _CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
                arr.set(buf)
            else:
                arr[...] = buf
            return

        if not (_CUPY_AVAILABLE and isinstance(arr, cp.ndarray)):
            raise TypeError("NCCL allreduce requires cupy.ndarray.")

        # --------- CRITICAL: strong sync (per-rank GPU only) ----------
        # Make sure all prior GPU work that produced 'arr' is done,
        # even if upstream kernels used non-current streams.
        cp.cuda.Device().synchronize()

        # complex64
        if arr.dtype == cp.complex64:
            if self.prefer_fp16:
                tmp = cp.empty((2,) + arr.shape, dtype=cp.float16)
                with stream:
                    tmp[0] = arr.real.astype(cp.float16, copy=False)
                    tmp[1] = arr.imag.astype(cp.float16, copy=False)
                    self.comm_nccl.allReduce(
                        tmp.data.ptr, tmp.data.ptr,
                        tmp.size, nccl.NCCL_FLOAT16, nccl.NCCL_SUM,
                        stream.ptr
                    )
                # ensure NCCL done before writing back
                cp.cuda.Device().synchronize()
                arr.real[...] = tmp[0].astype(cp.float32)
                arr.imag[...] = tmp[1].astype(cp.float32)
                return

            if arr.flags.c_contiguous:
                view = arr.view(cp.float32)
                with stream:
                    self.comm_nccl.allReduce(
                        view.data.ptr, view.data.ptr,
                        view.size, nccl.NCCL_FLOAT32, nccl.NCCL_SUM,
                        stream.ptr
                    )
                cp.cuda.Device().synchronize()
                return

            tmp = cp.ascontiguousarray(arr)
            view = tmp.view(cp.float32)
            with stream:
                self.comm_nccl.allReduce(
                    view.data.ptr, view.data.ptr,
                    view.size, nccl.NCCL_FLOAT32, nccl.NCCL_SUM,
                    stream.ptr
                )
            cp.cuda.Device().synchronize()
            arr[...] = tmp
            return

        # float32 fp16 comm
        if arr.dtype == cp.float32 and self.prefer_fp16:
            tmp16 = arr.astype(cp.float16, copy=False)
            if not tmp16.flags.c_contiguous:
                tmp16 = cp.ascontiguousarray(tmp16)
            with stream:
                self.comm_nccl.allReduce(
                    tmp16.data.ptr, tmp16.data.ptr,
                    tmp16.size, nccl.NCCL_FLOAT16, nccl.NCCL_SUM,
                    stream.ptr
                )
            cp.cuda.Device().synchronize()
            arr[...] = tmp16.astype(cp.float32)
            return

        ntype = self._dtype_to_nccl(arr.dtype)
        if ntype is None:
            raise ValueError(f"Unsupported dtype for NCCL allreduce: {arr.dtype}")

        if not arr.flags.c_contiguous:
            tmp = cp.ascontiguousarray(arr)
            with stream:
                self.comm_nccl.allReduce(
                    tmp.data.ptr, tmp.data.ptr,
                    tmp.size, ntype, nccl.NCCL_SUM,
                    stream.ptr
                )
            cp.cuda.Device().synchronize()
            arr[...] = tmp
            return

        with stream:
            self.comm_nccl.allReduce(
                arr.data.ptr, arr.data.ptr,
                arr.size, ntype, nccl.NCCL_SUM,
                stream.ptr
            )
        cp.cuda.Device().synchronize()


    def synchronize(self):
        if self.enabled and self.stream is not None:
            self.stream.synchronize()


class NodeGrid2D:
    """
    2D node grid:
      - coord = (egid, mgid)  [col, row]
      - row = mgid (motion), col = egid (echo)
      - node_id mapping (row-major by motion):
            node_id = mgid * E_groups + egid

    Provides:
      - (b0,b1): motion bin range on this node
      - (e0,e1): echo range on this node
      - halo peers (world ranks) for echo and motion axes:
            echo_left_peer / echo_right_peer
            motion_left_peer / motion_right_peer
      - axis leader comms for leaders (optional):
            echo_axis_leader_comm   (same mgid, varying egid)
            motion_axis_leader_comm (same egid, varying mgid)
    """
    def __init__(self, intra: IntraNodeGroup, motion_groups: int, echo_groups: int):
        self.intra = intra
        self.world = intra.world

        self.M_groups = int(motion_groups)
        self.E_groups = int(echo_groups)
        if self.M_groups <= 0 or self.E_groups <= 0:
            raise ValueError("motion_groups and echo_groups must be positive.")

        self.node_id = intra.node_id
        self.num_nodes = intra.num_nodes

        self.num_grid_nodes = self.M_groups * self.E_groups
        self.is_active = (self.node_id < self.num_grid_nodes)

        # coord = (egid, mgid)
        if self.is_active:
            self.mgid = self.node_id // self.E_groups
            self.egid = self.node_id %  self.E_groups
        else:
            self.mgid = -1
            self.egid = -1
        self.coord = (self.egid, self.mgid)

        # partitions
        self.B_total = None
        self.E_total = None
        self.b0 = self.b1 = 0
        self.e0 = self.e1 = 0

        # world-rank peers for halo exchange (per-rank aligned by intra.group_rank)
        self.echo_left_peer = MPI.PROC_NULL
        self.echo_right_peer = MPI.PROC_NULL
        self.motion_left_peer = MPI.PROC_NULL
        self.motion_right_peer = MPI.PROC_NULL

        # leader-only comms
        self.active_node_leader_comm = None
        self.echo_axis_leader_comm = None
        self.motion_axis_leader_comm = None

        self._build_axis_leader_comms()

    def _build_axis_leader_comms(self):
        """
        Build leader comms only on node leaders.
        - active_node_leader_comm includes only active grid nodes.
        - echo_axis_leader_comm: same mgid (row), varying egid
        - motion_axis_leader_comm: same egid (col), varying mgid
        """
        leader = self.intra.node_leader_comm
        if leader is None:
            return

        if not self.is_active:
            color_active = MPI.UNDEFINED
        else:
            color_active = 0

        raw = leader.Split(color=color_active, key=self.node_id)
        self.active_node_leader_comm = None if raw == MPI.COMM_NULL else raw

        if self.active_node_leader_comm is None:
            return

        # same row (mgid): echo axis
        self.echo_axis_leader_comm = self.active_node_leader_comm.Split(
            color=int(self.mgid),
            key=int(self.egid)
        )
        # same col (egid): motion axis
        self.motion_axis_leader_comm = self.active_node_leader_comm.Split(
            color=int(self.egid),
            key=int(self.mgid)
        )

    def set_partitions(self, B_total: int, E_total: int):
        """
        Set motion and echo partitions for this node (only if active).
        """
        self.B_total = int(B_total)
        self.E_total = int(E_total)

        if not self.is_active:
            self.b0 = self.b1 = 0
            self.e0 = self.e1 = 0
            self._set_peers()
            return (0, 0, 0, 0)

        self.b0, self.b1 = split_range(self.B_total, self.M_groups, self.mgid)
        self.e0, self.e1 = split_range(self.E_total, self.E_groups, self.egid)

        self._set_peers()
        return (self.b0, self.b1, self.e0, self.e1)

    def _node_id_of(self, egid: int, mgid: int) -> int:
        """Return node_id for given (egid, mgid)."""
        return int(mgid) * self.E_groups + int(egid)

    def _set_peers(self):
        """
        Compute halo peer world ranks for current (egid, mgid).
        Peer mapping uses the same intra-node rank index (default: intra.group_rank).
        """
        if not self.is_active:
            self.echo_left_peer = MPI.PROC_NULL
            self.echo_right_peer = MPI.PROC_NULL
            self.motion_left_peer = MPI.PROC_NULL
            self.motion_right_peer = MPI.PROC_NULL
            return

        # echo neighbors: (egid-1, mgid) and (egid+1, mgid)
        left_egid = self.egid - 1
        right_egid = self.egid + 1

        if left_egid >= 0:
            nid = self._node_id_of(left_egid, self.mgid)
            self.echo_left_peer = self.intra.peer_world_rank(nid)
        else:
            self.echo_left_peer = MPI.PROC_NULL

        if right_egid < self.E_groups:
            nid = self._node_id_of(right_egid, self.mgid)
            self.echo_right_peer = self.intra.peer_world_rank(nid)
        else:
            self.echo_right_peer = MPI.PROC_NULL

        # motion neighbors: (egid, mgid-1) and (egid, mgid+1)
        left_mgid = self.mgid - 1
        right_mgid = self.mgid + 1

        if left_mgid >= 0:
            nid = self._node_id_of(self.egid, left_mgid)
            self.motion_left_peer = self.intra.peer_world_rank(nid)
        else:
            self.motion_left_peer = MPI.PROC_NULL

        if right_mgid < self.M_groups:
            nid = self._node_id_of(self.egid, right_mgid)
            self.motion_right_peer = self.intra.peer_world_rank(nid)
        else:
            self.motion_right_peer = MPI.PROC_NULL

    @staticmethod
    def _sync_if_cupy(arr):
        if _CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
            cp.cuda.get_current_stream().synchronize()

    def halo_exchange(self, sendbuf, recvbuf, send_peer, recv_peer, tag):
        self._sync_if_cupy(sendbuf)
        self._sync_if_cupy(recvbuf)

        if int(recv_peer) == MPI.PROC_NULL:
            recvbuf[...] = 0
            req_r = MPI.REQUEST_NULL
        else:
            req_r = self.world.Irecv(recvbuf, source=int(recv_peer), tag=int(tag))

        if int(send_peer) == MPI.PROC_NULL:
            req_s = MPI.REQUEST_NULL
        else:
            req_s = self.world.Isend(sendbuf, dest=int(send_peer), tag=int(tag))

        return [req_r, req_s]
