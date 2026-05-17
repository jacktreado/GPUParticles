"""
field_observables.py — small helpers for comparing trajectory data
to the viscoelastic Dean-Kawasaki field theory.

What this module provides
-------------------------
1. Parameter mapping between input.json fields (R, De, Pe) and the
   notes' dimensionless groups (Pi1, Pi2, Pe), plus derived speeds
   v0 = f0/gamma, v_inf = v0/(1+Pi1), and time scales.

2. Closure-C run-mean prediction:
        v_eff(Pi1, Pi2) / v0 = (v_inf/v0 + 1/((1+Pi1) Pi2)) /
                               (1   + 1/((1+Pi1) Pi2))
   reducing to v0 at Pi2 -> 0 and v_inf at Pi2 -> infinity.

3. read_run(h5path)
       Open a trajectory HDF5, read root attrs (R, De, Pe, f0, ...),
       derive the dimensionless groups, and if /correlations exists
       also return tau, C_vn, C_Fn, C_vv arrays. <v . n_hat> at lag 0
       is the dilute-limit effective swim speed measurement needed for
       Priority 1.

4. veff_vs_phi_from_traj(traj, ...)
       Per-frame loop: compute per-particle Voronoi local packing
       fraction phi_i and v_i . n_hat_i, then bin <v . n_hat> against
       phi_i (1st-99th percentile). Returns bin centres, mean, SEM,
       count. Reuses _local_voronoi_phi and _bin_against from
       analyses/force_orientation.py.

5. fit_linear_veff(phi, vbar, sem)
       Weighted least-squares fit of v_eff(phi) = v0_eff (1 - phi/phi_*)
       (returns v0_eff, phi_*, sigma_v0_eff, sigma_phi_*).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np

# Reuse the already-validated Voronoi local-density and binning helpers.
from analysis.analyses.force_orientation import (
    _local_voronoi_phi,
    _bin_against,
)


# ============================================================================
# 1. Parameter mapping
# ============================================================================


def map_params(R: float, De: float, Pe: float, f0: float, gamma: float):
    """
    Map (R, De, Pe) [input.json convention] to (Pi1, Pi2, Pe, v0, v_inf).

        Pi1   = gamma' / gamma                = R
        Pi2   = tau_p / tau_M                 = 1 / De
        Pe    = f0 * tau_p / (gamma * sigma)  = Pe (passed through)
        v0    = f0 / gamma
        v_inf = v0 / (1 + Pi1)
    """
    Pi1 = float(R)
    Pi2 = 1.0 / float(De) if De > 0 else np.inf
    v0 = float(f0) / float(gamma)
    v_inf = v0 / (1.0 + Pi1)
    return {
        "Pi1": Pi1,
        "Pi2": Pi2,
        "Pe": float(Pe),
        "v0": v0,
        "v_inf": v_inf,
    }


# ============================================================================
# 2. Closure-C run-mean prediction
# ============================================================================


def closure_C_veff(Pi1: float, Pi2, v0: float = 1.0):
    """
    Closure-C (Section 5.3 of the viscoelastic notes) run-mean
    effective swim speed in the dilute limit:

        v_eff / v0 = ( v_inf/v0 + 1/[(1+Pi1) Pi2] )
                     ----------------------------------
                       ( 1     + 1/[(1+Pi1) Pi2] )

    with v_inf/v0 = 1/(1+Pi1).

    Parameters
    ----------
    Pi1   : scalar
    Pi2   : scalar or numpy array (broadcasts)
    v0    : scalar prefactor (default 1, so output is v_eff/v0).

    Returns
    -------
    same shape as Pi2.
    """
    Pi2 = np.asarray(Pi2, dtype=float)
    v_inf_over_v0 = 1.0 / (1.0 + Pi1)
    Drtau = (1.0 + Pi1) * Pi2  # = D_r tau_*
    inv = 1.0 / Drtau
    return v0 * (v_inf_over_v0 + inv) / (1.0 + inv)


def closure_A_veff(Pi1: float, v0: float = 1.0):
    """Persistent-limit (Pi2 -> infty) closure: v_eff = v_inf."""
    return v0 / (1.0 + Pi1)


def closure_B_veff(Pi1: float, v0: float = 1.0):  # noqa: ARG001
    """Tumbling-limit (Pi2 -> 0) closure: v_eff = v0."""
    return v0


# ============================================================================
# 3. Read a trajectory + on-the-fly correlator
# ============================================================================


def read_run(h5path: str | Path) -> dict:
    """
    Open a GPUParticles trajectory HDF5 and return a dict with
    physical parameters, dimensionless groups, and (if present)
    the on-the-fly correlator (tau, C_vn, C_Fn, C_vv).

    The dilute-limit measurement of <v . n_hat> is C_vn[0]; this is
    exactly the quantity the notes' Priority 1 asks for.

    Returned keys:
        N, phi, sigma, L, gamma, gamma_a, f0, kT, tau_theta, k_a,
        R, De, Pe,
        Pi1, Pi2, v0, v_inf,
        has_corr (bool), tau, C_vn, C_Fn, C_vv, n_samples, t_warm, corr_dt
    """
    with h5py.File(h5path, "r") as f:
        a = dict(f.attrs)
        out = {
            "path": str(h5path),
            "N": int(a.get("N", 0)),
            "phi": float(a.get("phi", 0.0)),
            "sigma": float(a.get("sigma", 1.0)),
            "L": float(a.get("L", 0.0)),
            "gamma": float(a.get("gamma", 1.0)),
            "gamma_a": float(a.get("gamma_a", 0.0)),
            "f0": float(a.get("f0", 0.0)),
            "kT": float(a.get("kT", 0.0)),
            "tau_theta": float(a.get("tau_theta", 0.0)),
            "k_a": float(a.get("k_a", 0.0)),
            "potential": str(a.get("potential", "")),
        }
        # Sweep-level dimensionless groups (when set by psweep.py)
        out["R"] = float(a.get("R", out["gamma_a"] / out["gamma"]
                                  if out["gamma"] > 0 else 0.0))
        out["De"] = float(a.get("De", np.nan))
        out["Pe"] = float(a.get("Pe", np.nan))

        out.update(map_params(out["R"], out["De"], out["Pe"],
                              out["f0"], out["gamma"]))

        # On-the-fly correlator (optional)
        out["has_corr"] = "correlations" in f
        if out["has_corr"]:
            g = f["correlations"]
            out["tau"] = g["tau"][:]
            out["C_vn"] = g["C_vn"][:]
            out["C_Fn"] = g["C_Fn"][:]
            out["C_vv"] = g["C_vv"][:]
            out["n_samples"] = int(g.attrs.get("n_samples", 0))
            out["t_warm"] = float(g.attrs.get("t_warm", 0.0))
            out["corr_dt"] = float(g.attrs.get("corr_dt", 0.0))
            # The dilute-limit / spatial-average effective swim speed:
            out["vn_lag0"] = float(out["C_vn"][0])
        else:
            out["vn_lag0"] = float("nan")

    return out


# ============================================================================
# 4. v_eff(phi) from a single trajectory (Phase B)
# ============================================================================


def veff_vs_phi_from_traj(
    traj,
    n_bins: int = 20,
    frame_stride: int = 1,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
):
    """
    For each frame in [frame_start, frame_stop) stride frame_stride:
      - Compute per-particle Voronoi local packing fraction phi_i
        on the periodic box (9-image tiling).
      - Compute v_i . n_hat(theta_i) using stored velocities and
        orientations.
    Pool all (phi_i, v_i . n_hat_i) across frames and bin v.n_hat
    against phi (1st-99th percentile, equal-width bins).

    Parameters
    ----------
    traj : BDTrajectory
        Must have velocities, orientations, anchors stored.
    n_bins : int
        Number of phi bins.
    frame_stride : int
        Use every Nth frame (default: all).
    frame_start, frame_stop : int
        Frame range. Default: full range.

    Returns
    -------
    dict with keys:
        phi_edges  : (n_bins+1,)  bin edges
        phi_centres: (n_bins,)
        vn_mean    : (n_bins,)    <v . n_hat>(phi)
        vn_sem     : (n_bins,)
        count      : (n_bins,)    samples per bin
        n_frames   : int          number of frames processed
        phi_all    : (n_frames*N,) pooled phi
        vn_all     : (n_frames*N,) pooled v.n_hat
    """
    Lx, Ly = float(traj.Lx), float(traj.Ly)
    sigma = float(traj.sigma)
    nf_total = traj.num_frames
    if frame_stop is None:
        frame_stop = nf_total
    frames = range(frame_start, frame_stop, frame_stride)

    phi_pool = []
    vn_pool = []

    for k in frames:
        pos = traj.positions(k)
        vel = traj.velocities(k)
        theta = traj.orientations(k)
        if vel is None or theta is None:
            continue
        n_hat = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        vn = np.einsum("ij,ij->i", vel, n_hat)
        phi = _local_voronoi_phi(pos, sigma, Lx, Ly)
        phi_pool.append(phi)
        vn_pool.append(vn)

    phi_all = np.concatenate(phi_pool) if phi_pool else np.array([])
    vn_all = np.concatenate(vn_pool) if vn_pool else np.array([])

    edges, mean, sem, count = _bin_against(phi_all, vn_all, n_bins)
    centres = 0.5 * (edges[:-1] + edges[1:])

    return {
        "phi_edges": edges,
        "phi_centres": centres,
        "vn_mean": mean,
        "vn_sem": sem,
        "count": count,
        "n_frames": len(phi_pool),
        "phi_all": phi_all,
        "vn_all": vn_all,
    }


# ============================================================================
# 5. Linear veff(phi) fit
# ============================================================================


def fit_linear_veff(phi: np.ndarray,
                    vbar: np.ndarray,
                    sem: np.ndarray,
                    min_count: Optional[np.ndarray] = None,
                    min_n: int = 10) -> dict:
    """
    Weighted linear fit  v_eff(phi) = v0_eff * (1 - phi / phi_star).

    Equivalent to  vbar(phi) = a + b * phi  with  v0_eff = a,
    phi_star = -a / b.

    Parameters
    ----------
    phi  : bin centres
    vbar : <v . n_hat>(phi)
    sem  : SEM of vbar
    min_count : optional bin counts; bins with count < min_n are dropped.

    Returns
    -------
    dict with v0_eff, phi_star, sigma_v0_eff, sigma_phi_star,
    plus the underlying (a, b, cov).
    """
    phi = np.asarray(phi, dtype=float)
    vbar = np.asarray(vbar, dtype=float)
    sem = np.asarray(sem, dtype=float)

    keep = np.isfinite(phi) & np.isfinite(vbar) & np.isfinite(sem) & (sem > 0)
    if min_count is not None:
        keep &= (np.asarray(min_count) >= min_n)
    if keep.sum() < 3:
        return {"v0_eff": np.nan, "phi_star": np.nan,
                "sigma_v0_eff": np.nan, "sigma_phi_star": np.nan,
                "a": np.nan, "b": np.nan, "cov": None, "n_used": int(keep.sum())}

    x = phi[keep]
    y = vbar[keep]
    w = 1.0 / sem[keep] ** 2  # weights = 1/var
    # Design matrix [1, phi]
    X = np.column_stack([np.ones_like(x), x])
    W = np.diag(w)
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y
    beta = np.linalg.solve(XtWX, XtWy)
    cov = np.linalg.inv(XtWX)
    a, b = float(beta[0]), float(beta[1])
    sa = float(np.sqrt(cov[0, 0]))
    sb = float(np.sqrt(cov[1, 1]))
    if b >= 0:
        # No physical phi_star (no density quenching detected).
        phi_star, sigma_phi_star = np.nan, np.nan
    else:
        phi_star = -a / b
        # error propagation:  sigma(phi*)^2 = (1/b^2)*sa^2 + (a/b^2)^2*sb^2
        #                                     - 2*(a/b^3)*cov[0,1]
        cov01 = float(cov[0, 1])
        var_ps = (1.0 / b ** 2) * sa ** 2 + (a / b ** 2) ** 2 * sb ** 2 \
            - 2.0 * (a / b ** 3) * cov01
        sigma_phi_star = float(np.sqrt(max(var_ps, 0.0)))
    return {
        "v0_eff": a,
        "phi_star": phi_star,
        "sigma_v0_eff": sa,
        "sigma_phi_star": sigma_phi_star,
        "a": a, "b": b, "cov": cov,
        "n_used": int(keep.sum()),
    }
