"""
force_orientation — alignment of pair-interaction force with self-propulsion.

For each ABP particle at every frame we have:
  - F_i(t) : net pair-interaction force (the trajectory's `forces` dataset),
  - n̂_i(t): orientation unit vector along which the active force acts.

This analysis computes:
  - per-particle time-averaged F.n̂ (and population mean / SEM),
  - particle-averaged F.n̂(t) signal vs time,
  - FFT-based cross-correlation C(τ) = <F(t+τ).n̂(t)>_{i,t} via scipy.signal.correlate,
  - optionally, F.n̂ binned against per-particle Voronoi local packing fraction
    (gated by `with_density=True` because Voronoi-per-frame is expensive at large N).

Only meaningful for active (f0 > 0) trajectories with `forces` and `orientations`.
Passive runs raise via the requires=("forces", "orientations") gate.
"""

from __future__ import annotations

import numpy as np

from bdtrajectory import BDTrajectory

from ..io import stack_dataset
from ..registry import register_analysis


@register_analysis(
    name="force_orientation",
    version=1,
    requires=("forces", "orientations"),
    outputs={
        "s_per_particle": {"shape": "(N,)",          "dtype": "f8", "axes": ["particle"]},
        "s_mean":         {"shape": "()",            "dtype": "f8", "axes": []},
        "s_sem":          {"shape": "()",            "dtype": "f8", "axes": []},
        "s_vs_time":      {"shape": "(num_frames,)", "dtype": "f8", "axes": ["frame"]},
        "times":          {"shape": "(num_frames,)", "dtype": "f8", "axes": ["frame"]},
        "lags":           {"shape": "(2T-1,)",       "dtype": "i8", "axes": ["lag"]},
        "tau":            {"shape": "(2T-1,)",       "dtype": "f8", "axes": ["lag"]},
        "C_tau":          {"shape": "(2T-1,)",       "dtype": "f8", "axes": ["lag"]},
        "phi_bin_edges":  {"shape": "(n_bins+1,)",   "dtype": "f8", "axes": ["bin_edge"]},
        "s_vs_phi":       {"shape": "(n_bins,)",     "dtype": "f8", "axes": ["bin"]},
        "s_vs_phi_sem":   {"shape": "(n_bins,)",     "dtype": "f8", "axes": ["bin"]},
        "s_vs_phi_count": {"shape": "(n_bins,)",     "dtype": "i8", "axes": ["bin"]},
    },
)
def analyze(
    traj: BDTrajectory,
    *,
    with_density: bool = False,
    n_bins: int = 30,
) -> dict[str, np.ndarray]:
    if not traj.is_active:
        raise RuntimeError(
            "trajectory is passive (f0 = 0); F.n is not physically meaningful"
        )

    forces = stack_dataset(traj, "forces")          # (T, N, 2)
    theta = stack_dataset(traj, "orientations")     # (T, N)
    if forces is None or theta is None:
        raise RuntimeError("forces or orientations missing from trajectory")

    T, N, _ = forces.shape
    times = traj.all_times()
    nhat = np.stack((np.cos(theta), np.sin(theta)), axis=-1)  # (T, N, 2)

    # Per-frame, per-particle scalar s_i(t) = F_i(t) . n̂_i(t) — shape (T, N).
    s = np.einsum("tnk,tnk->tn", forces, nhat)

    s_per_particle = s.mean(axis=0).astype(np.float64)              # (N,)
    s_mean = float(s_per_particle.mean())
    s_sem = (
        float(s_per_particle.std(ddof=1) / np.sqrt(N)) if N > 1 else 0.0
    )
    s_vs_time = s.mean(axis=1).astype(np.float64)                   # (T,)

    # FFT cross-correlation per particle, summed over xy components, averaged
    # over particles. Unbiased normalization by overlap count per lag.
    from scipy.signal import correlate, correlation_lags

    lags = correlation_lags(T, T, mode="full")                      # (2T-1,)
    norm = T - np.abs(lags)
    cc = np.zeros_like(lags, dtype=np.float64)
    for k in range(2):
        for i in range(N):
            cc += correlate(forces[:, i, k], nhat[:, i, k], mode="full")
    cc /= (N * norm)

    dt_out = float(
        traj.params.get(
            "output_dt",
            (times[1] - times[0]) if T > 1 else 1.0,
        )
    )
    tau = (lags * dt_out).astype(np.float64)

    out: dict[str, np.ndarray] = {
        "s_per_particle": s_per_particle,
        "s_mean": np.float64(s_mean),
        "s_sem": np.float64(s_sem),
        "s_vs_time": s_vs_time,
        "times": times.astype(np.float64),
        "lags": lags.astype(np.int64),
        "tau": tau,
        "C_tau": cc,
    }

    if with_density:
        phi_per = _local_voronoi_phi_all_frames(traj, T, N)
        bin_edges, mean_in_bin, sem_in_bin, count = _bin_against(
            x=phi_per.ravel(),
            y=s.ravel(),
            n_bins=n_bins,
        )
        out["phi_bin_edges"] = bin_edges
        out["s_vs_phi"] = mean_in_bin
        out["s_vs_phi_sem"] = sem_in_bin
        out["s_vs_phi_count"] = count

    return out


# ---------------------------------------------------------------------------
# Helpers (Voronoi local density + binning) — only used when with_density=True
# ---------------------------------------------------------------------------


def _local_voronoi_phi_all_frames(traj: BDTrajectory, T: int, N: int) -> np.ndarray:
    """Per-particle local packing fraction phi_i for every frame: shape (T, N)."""
    out = np.empty((T, N), dtype=np.float64)
    sigma = float(traj.sigma)
    for k in range(T):
        pos = traj.positions(k)
        Lx, Ly = traj.frame_box(k)
        out[k] = _local_voronoi_phi(pos, sigma, Lx, Ly)
    return out


def _local_voronoi_phi(positions: np.ndarray, sigma: float, Lx: float, Ly: float) -> np.ndarray:
    """phi_i = pi (sigma/2)^2 / A_voronoi_i with 9-image periodic tiling."""
    from scipy.spatial import Voronoi

    N = positions.shape[0]
    # Trajectory positions are unwrapped; fold to the primary cell before the
    # 9-image tiling or the Voronoi diagram explodes over an unbounded domain.
    pos = np.column_stack([np.mod(positions[:, 0], Lx),
                           np.mod(positions[:, 1], Ly)])
    shifts = np.array(
        [(dx * Lx, dy * Ly) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    )
    images = (pos[None, :, :] + shifts[:, None, :]).reshape(-1, 2)

    vor = Voronoi(images)
    particle_area = np.pi * (sigma / 2.0) ** 2
    phi = np.empty(N, dtype=np.float64)
    for i in range(N):
        region = vor.regions[vor.point_region[i]]
        if not region or -1 in region or len(region) < 3:
            phi[i] = np.nan
            continue
        v = vor.vertices[region]
        x, y = v[:, 0], v[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        phi[i] = particle_area / area
    return phi


def _bin_against(
    x: np.ndarray, y: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean / SEM / count of y within equal-width bins of x (1st-99th pctile)."""
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size == 0:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        zeros = np.zeros(n_bins, dtype=np.float64)
        return edges, zeros, zeros, np.zeros(n_bins, dtype=np.int64)

    lo, hi = np.percentile(x, [1.0, 99.0])
    if hi <= lo:
        hi = lo + 1e-12
    edges = np.linspace(lo, hi, n_bins + 1)

    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    mean = np.zeros(n_bins, dtype=np.float64)
    sem = np.zeros(n_bins, dtype=np.float64)
    count = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        sel = y[idx == b]
        if sel.size:
            mean[b] = sel.mean()
            sem[b] = sel.std(ddof=1) / np.sqrt(sel.size) if sel.size > 1 else 0.0
            count[b] = sel.size
    return edges.astype(np.float64), mean, sem, count
