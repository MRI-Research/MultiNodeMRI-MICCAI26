"""Respiratory self-navigation extracted from the historical CLI script."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from toporecon.data.cfl import write_cfl
from toporecon.data.dataset import RawDataset


@dataclass(frozen=True)
class RespiratoryConfig:
    """Band-pass and normalization settings matching the research script."""

    low_cut_hz: float = 0.1
    high_cut_hz: float = 2.0
    low_pass_order: int = 10
    high_pass_order: int = 5

    def validate(self, sampling_rate_hz: float) -> None:
        if self.low_cut_hz <= 0:
            raise ValueError("low_cut_hz must be positive.")
        if self.high_cut_hz <= self.low_cut_hz:
            raise ValueError("high_cut_hz must be greater than low_cut_hz.")
        if self.high_cut_hz >= sampling_rate_hz / 2:
            raise ValueError(
                f"high_cut_hz={self.high_cut_hz} must be below the Nyquist "
                f"frequency {sampling_rate_hz / 2}."
            )


def estimate_respiratory_signal(
    kspace: np.ndarray,
    tr_seconds: float,
    *,
    config: RespiratoryConfig | None = None,
) -> np.ndarray:
    """Estimate a normalized respiratory signal from canonical k-space.

    Input k-space must have shape ``[echo, coil, trajectory, readout]``.
    The first readout sample supplies the self-navigation signal. A PCA is
    applied over coils per echo and then over echoes.
    """

    try:
        from scipy import signal
    except ImportError as exc:  # pragma: no cover - depends on release environment.
        raise RuntimeError(
            "SciPy is required for respiratory signal estimation."
        ) from exc

    data = np.asarray(kspace)
    if data.ndim != 4:
        raise ValueError(
            "k-space must have shape [echo,coil,trajectory,readout], "
            f"got {data.shape}."
        )
    if tr_seconds <= 0:
        raise ValueError("tr_seconds must be positive.")

    resolved = config or RespiratoryConfig()
    sampling_rate = 1.0 / float(tr_seconds)
    resolved.validate(sampling_rate)

    dc = np.abs(data[..., 0])
    low_pass = signal.butter(
        resolved.low_pass_order,
        resolved.high_cut_hz,
        "low",
        fs=sampling_rate,
        output="sos",
    )
    high_pass = signal.butter(
        resolved.high_pass_order,
        resolved.low_cut_hz,
        "high",
        fs=sampling_rate,
        output="sos",
    )

    filtered = np.empty(dc.shape, dtype=np.float32)
    for echo in range(dc.shape[0]):
        for coil in range(dc.shape[1]):
            trace = dc[echo, coil] - np.mean(dc[echo, coil])
            trace = signal.sosfiltfilt(low_pass, trace)
            trace = signal.sosfiltfilt(high_pass, trace)
            filtered[echo, coil] = trace

    per_echo = np.empty((data.shape[0], data.shape[2]), dtype=np.float32)
    for echo in range(data.shape[0]):
        _, singular_values, right_vectors = np.linalg.svd(
            filtered[echo],
            full_matrices=False,
        )
        per_echo[echo] = singular_values[0] * right_vectors[0]

    if data.shape[0] == 1:
        respiratory = per_echo[0].copy()
    else:
        _, singular_values, right_vectors = np.linalg.svd(
            per_echo,
            full_matrices=False,
        )
        respiratory = singular_values[0] * right_vectors[0]

    minimum = float(np.min(respiratory))
    maximum = float(np.max(respiratory))
    extent = maximum - minimum
    if extent <= np.finfo(np.float32).eps:
        raise ValueError("Respiratory signal is constant and cannot be normalized.")
    return np.asarray((respiratory - minimum) / extent, dtype=np.float32)


def estimate_respiratory_from_dataset(
    input_directory: str | Path,
    output_directory: str | Path,
    *,
    config: RespiratoryConfig | None = None,
) -> Path:
    """Estimate and write ``resp.hdr/.cfl`` without modifying raw data."""

    dataset = RawDataset.open(input_directory)
    respiratory = estimate_respiratory_signal(
        dataset.read_kspace(),
        dataset.tr_seconds,
        config=config,
    )
    output_base = Path(output_directory) / "resp"
    write_cfl(output_base, respiratory[:, None])
    return output_base
