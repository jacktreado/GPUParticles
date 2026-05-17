"""
correlations — ensemble average of the C++ on-the-fly correlator.

Each trajectory's HDF5 contains a top-level /correlations group written by
the simulator (tau, C_vn, C_Fn, C_vv, count). Those arrays are already
averaged over particles and time windows within a single run; this
analysis just exposes them to the framework so the reduce step computes
the across-seed mean/SEM in the standard <analysis>.h5 layout.
"""

from __future__ import annotations

import numpy as np

from bdtrajectory import BDTrajectory

from ..registry import register_analysis


@register_analysis(
    name="correlations",
    version=1,
    requires=(),
    outputs={
        "tau":   {"shape": "(n_lags,)", "dtype": "f8", "axes": ["lag"]},
        "C_vn":  {"shape": "(n_lags,)", "dtype": "f8", "axes": ["lag"]},
        "C_Fn":  {"shape": "(n_lags,)", "dtype": "f8", "axes": ["lag"]},
        "C_vv":  {"shape": "(n_lags,)", "dtype": "f8", "axes": ["lag"]},
        "count": {"shape": "(n_lags,)", "dtype": "i8", "axes": ["lag"]},
    },
)
def analyze(traj: BDTrajectory) -> dict[str, np.ndarray]:
    f = traj._file
    if "correlations" not in f:
        raise RuntimeError(
            "trajectory has no /correlations group — "
            "re-run the simulation with the on-the-fly correlator enabled"
        )
    g = f["correlations"]
    return {
        "tau":   np.asarray(g["tau"][:],   dtype=np.float64),
        "C_vn":  np.asarray(g["C_vn"][:],  dtype=np.float64),
        "C_Fn":  np.asarray(g["C_Fn"][:],  dtype=np.float64),
        "C_vv":  np.asarray(g["C_vv"][:],  dtype=np.float64),
        "count": np.asarray(g["count"][:], dtype=np.int64),
    }
