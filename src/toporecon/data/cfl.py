"""Read and write BART-style CFL/HDR arrays used by the research code."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


CFL_DTYPE = np.dtype(np.complex64)


def base_path(name: str | Path) -> Path:
    """Return a CFL base path with a possible ``.cfl``/``.hdr`` suffix removed."""

    path = Path(name)
    if path.suffix in {".cfl", ".hdr"}:
        return path.with_suffix("")
    return path


def read_cfl_header(name: str | Path) -> tuple[int, ...]:
    """Read the array shape in the same axis order as NumPy."""

    path = base_path(name).with_suffix(".hdr")
    dimensions: list[int] | None = None
    with path.open("r", encoding="utf-8") as header:
        for line in header:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            dimensions = [int(value) for value in stripped.split()]
            break

    if not dimensions:
        raise ValueError(f"No dimensions found in CFL header: {path}")
    if any(dimension < 1 for dimension in dimensions):
        raise ValueError(f"CFL dimensions must be positive: {dimensions}")
    return tuple(reversed(dimensions))


def expected_cfl_bytes(shape: Sequence[int]) -> int:
    """Return the exact byte count for a complex64 CFL payload."""

    return int(np.prod(tuple(shape), dtype=np.int64)) * CFL_DTYPE.itemsize


def validate_cfl_storage(name: str | Path, shape: Sequence[int] | None = None) -> None:
    """Raise when a CFL payload size does not match its header."""

    root = base_path(name)
    resolved_shape = tuple(shape) if shape is not None else read_cfl_header(root)
    expected = expected_cfl_bytes(resolved_shape)
    actual = root.with_suffix(".cfl").stat().st_size
    if actual != expected:
        raise ValueError(
            f"CFL payload size mismatch for {root}: expected {expected} bytes, "
            f"found {actual} bytes."
        )


def read_cfl(name: str | Path, *, mmap: bool = False) -> np.ndarray:
    """Read a CFL array.

    The historical reconstruction code stores payloads in C order while
    writing dimensions in reverse order in the HDR file. This function
    intentionally preserves that convention.
    """

    root = base_path(name)
    shape = read_cfl_header(root)
    validate_cfl_storage(root, shape)
    payload = root.with_suffix(".cfl")

    if mmap:
        return np.memmap(payload, mode="r", dtype=CFL_DTYPE, shape=shape)

    element_count = int(np.prod(shape, dtype=np.int64))
    with payload.open("rb") as stream:
        array = np.fromfile(stream, dtype=CFL_DTYPE, count=element_count)
    return array.reshape(shape)


def write_cfl(name: str | Path, array: np.ndarray) -> None:
    """Write an array using the historical CFL/HDR convention."""

    root = base_path(name)
    root.parent.mkdir(parents=True, exist_ok=True)
    contiguous = np.ascontiguousarray(array, dtype=CFL_DTYPE)

    with root.with_suffix(".hdr").open("w", encoding="utf-8") as header:
        header.write("# Dimensions\n")
        header.write(" ".join(str(dimension) for dimension in contiguous.shape[::-1]))
        header.write("\n")

    with root.with_suffix(".cfl").open("wb") as stream:
        contiguous.tofile(stream)


def _kspace_dimensions(shape: Sequence[int]) -> tuple[int, int, int, int]:
    """Return ``(echo, coil, trajectory, readout)`` from a legacy CFL shape."""

    dimensions = tuple(int(value) for value in shape)
    if len(dimensions) < 8:
        raise ValueError(f"Unsupported k-space CFL shape: {dimensions}.")
    result = (
        dimensions[-6],
        dimensions[-4],
        dimensions[-3],
        dimensions[-2],
    )
    if int(np.prod(result, dtype=np.int64)) != int(
        np.prod(dimensions, dtype=np.int64)
    ):
        raise ValueError(
            "Unsupported non-singleton dimensions in k-space CFL shape "
            f"{dimensions}."
        )
    return result


def read_kspace_shard(
    name: str | Path,
    *,
    echo_start: int,
    echo_end: int,
    coil_indices: Sequence[int],
    readout: int | None = None,
    trajectory_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """Read one echo/coil/trajectory shard without loading the full CFL.

    Returns canonical shape ``[echo, coil, trajectory, readout]``.
    The order of ``trajectory_indices`` is preserved.
    """

    root = base_path(name)
    stored_shape = read_cfl_header(root)
    validate_cfl_storage(root, stored_shape)
    echoes, coils, trajectories, full_readout = _kspace_dimensions(stored_shape)

    if not 0 <= echo_start < echo_end <= echoes:
        raise ValueError(
            f"Invalid echo range [{echo_start}, {echo_end}) for {echoes} echoes."
        )
    selected_coils = np.asarray(coil_indices, dtype=np.int64).reshape(-1)
    if selected_coils.size == 0:
        raise ValueError("At least one coil index is required.")
    if selected_coils.min() < 0 or selected_coils.max() >= coils:
        raise IndexError(
            f"Coil indices [{selected_coils.min()}, {selected_coils.max()}] "
            f"are outside [0, {coils})."
        )

    resolved_readout = full_readout if readout is None else int(readout)
    if not 1 <= resolved_readout <= full_readout:
        raise ValueError(
            f"readout must be in [1, {full_readout}], got {resolved_readout}."
        )

    selected_trajectories: np.ndarray | None = None
    if trajectory_indices is not None:
        selected_trajectories = np.asarray(
            trajectory_indices, dtype=np.int64
        ).reshape(-1)
        if selected_trajectories.size:
            if (
                selected_trajectories.min() < 0
                or selected_trajectories.max() >= trajectories
            ):
                raise IndexError(
                    "Trajectory indices are outside "
                    f"[0, {trajectories})."
                )

    mapped = np.memmap(
        root.with_suffix(".cfl"),
        mode="r",
        dtype=CFL_DTYPE,
        shape=(echoes, coils, trajectories, full_readout),
    )
    block = mapped[echo_start:echo_end]
    block = np.take(block, selected_coils, axis=1)
    if selected_trajectories is not None:
        block = np.take(block, selected_trajectories, axis=2)
    return np.array(block[..., :resolved_readout], dtype=CFL_DTYPE, order="C")


def read_mps_coils(
    name: str | Path,
    *,
    coil_indices: Sequence[int],
) -> np.ndarray:
    """Read selected sensitivity maps as ``[coil,z,y,x]``."""

    root = base_path(name)
    shape = read_cfl_header(root)
    validate_cfl_storage(root, shape)
    if len(shape) < 4:
        raise ValueError(f"Unsupported MPS CFL shape: {shape}.")

    coils = int(shape[-4])
    spatial_shape = tuple(int(value) for value in shape[-3:])
    if coils * int(np.prod(spatial_shape, dtype=np.int64)) != int(
        np.prod(shape, dtype=np.int64)
    ):
        raise ValueError(
            f"Unsupported non-singleton dimensions in MPS CFL shape {shape}."
        )

    selected_coils = np.asarray(coil_indices, dtype=np.int64).reshape(-1)
    if selected_coils.size == 0:
        raise ValueError("At least one coil index is required.")
    if selected_coils.min() < 0 or selected_coils.max() >= coils:
        raise IndexError(
            f"Coil indices [{selected_coils.min()}, {selected_coils.max()}] "
            f"are outside [0, {coils})."
        )

    mapped = np.memmap(
        root.with_suffix(".cfl"),
        mode="r",
        dtype=CFL_DTYPE,
        shape=shape,
    )
    coil_first = np.moveaxis(mapped, -4, 0)
    selected = coil_first[selected_coils]
    return np.array(
        selected.reshape((selected_coils.size,) + spatial_shape),
        dtype=CFL_DTYPE,
        order="C",
    )
