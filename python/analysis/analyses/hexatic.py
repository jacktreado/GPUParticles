"""
hexatic — per-particle bond-orientational order parameter psi6.

For each particle i with PBC Delaunay neighbours j, define the bond angles
theta_ij from the minimum-image displacement vector and accumulate

    psi6_i = (1 / Z_i) * sum_j exp(i * 6 * theta_ij)

Particles with no neighbours yield NaN (handled via nanmean for summaries).
"""

from __future__ import annotations

import numpy as np

from bdtrajectory import BDTrajectory

from ..io import stack_dataset
from ..registry import register_analysis
from ._pbc_delaunay import compute_pbc_delaunay_neighbors


@register_analysis(
    name="hexatic",
    version=1,
    requires=("positions",),
    outputs={
        "abs_psi6_mean":    {"shape": "(num_frames,)",   "dtype": "f8", "axes": ["frame"]},
        "abs_global_psi6":  {"shape": "(num_frames,)",   "dtype": "f8", "axes": ["frame"]},
        "times":            {"shape": "(num_frames,)",   "dtype": "f8", "axes": ["frame"]},
        "psi6_re":          {"shape": "(num_frames, N)", "dtype": "f4", "axes": ["frame", "particle"]},
        "psi6_im":          {"shape": "(num_frames, N)", "dtype": "f4", "axes": ["frame", "particle"]},
    },
)
def analyze(
    traj: BDTrajectory,
    *,
    with_per_particle: bool = False,
) -> dict[str, np.ndarray]:
    pos = stack_dataset(traj, "positions")  # (T, N, 2)
    if pos is None:
        raise RuntimeError("positions dataset is missing — cannot compute hexatic")

    T, N, _ = pos.shape
    times = traj.all_times().astype(np.float64)

    abs_psi6_mean = np.empty(T, dtype=np.float64)
    abs_global_psi6 = np.empty(T, dtype=np.float64)

    if with_per_particle:
        psi6_re = np.empty((T, N), dtype=np.float32)
        psi6_im = np.empty((T, N), dtype=np.float32)

    for k in range(T):
        Lx, Ly = traj.frame_box(k)
        psi6 = _psi6_one_frame(pos[k], Lx, Ly)
        abs_psi6_mean[k] = float(np.nanmean(np.abs(psi6)))
        # Global psi6: average the complex per-particle values, then take |.|.
        # Use nanmean to ignore particles with no neighbours.
        abs_global_psi6[k] = float(np.abs(np.nanmean(psi6)))
        if with_per_particle:
            psi6_re[k] = psi6.real.astype(np.float32)
            psi6_im[k] = psi6.imag.astype(np.float32)

    out: dict[str, np.ndarray] = {
        "abs_psi6_mean": abs_psi6_mean,
        "abs_global_psi6": abs_global_psi6,
        "times": times,
    }
    if with_per_particle:
        out["psi6_re"] = psi6_re
        out["psi6_im"] = psi6_im
    return out


def _psi6_one_frame(pos: np.ndarray, Lx: float, Ly: float) -> np.ndarray:
    """Per-particle complex psi6 for one frame; returns shape (N,) complex128."""
    N = pos.shape[0]
    x_ctr = np.mod(pos[:, 0], Lx)
    y_ctr = np.mod(pos[:, 1], Ly)
    centered = np.stack([x_ctr, y_ctr], axis=1)
    neighbors = compute_pbc_delaunay_neighbors(centered, Lx, Ly)

    psi6 = np.zeros(N, dtype=np.complex128)
    for i in range(N):
        nb = neighbors[i]
        if not nb:
            psi6[i] = np.nan + 1j * np.nan
            continue
        nb_arr = np.asarray(nb, dtype=np.int64)
        dx = centered[nb_arr, 0] - centered[i, 0]
        dy = centered[nb_arr, 1] - centered[i, 1]
        dx -= Lx * np.round(dx / Lx)
        dy -= Ly * np.round(dy / Ly)
        angles = np.arctan2(dy, dx)
        psi6[i] = np.sum(np.exp(1j * 6.0 * angles)) / len(nb)
    return psi6
