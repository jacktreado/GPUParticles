"""
contact_duration — ensemble average of pair-contact lifetime statistics.

Each trajectory's HDF5 carries a top-level /contact_durations group written
by the simulator's ContactDurationAccumulator (geometric contact:
r_ij < contact_cutoff, default sigma; streaming Welford over *completed*
contact lifetimes). That group is attribute-only: count, mean, stddev,
min, max, in_progress, in_progress_sum_duration, n_samples, contact_cutoff.

This analysis exposes the per-trajectory statistics as 0-d arrays so the
reduce step stacks them across seeds and computes the standard
grid / mean / sem / n_valid layout — uniform with msd.h5, correlations.h5,
etc.

Note: the source HDF5 group is `/contact_durations` (plural) but the
registered analysis is `contact_duration` (singular) to parallel
`force_orientation`, `msd`, etc.

The seed-average of `mean` is a mean-of-means (fine for equal-weight
seeds); the seed-averages of `stddev`/`min`/`max` are not the pooled
quantities a strict statistician would compute. Reach for `count`-weighted
recombinations if you need true ensemble statistics.
"""

from __future__ import annotations

import numpy as np

from bdtrajectory import BDTrajectory

from ..registry import register_analysis


@register_analysis(
    name="contact_duration",
    version=1,
    requires=(),
    outputs={
        "count":                    {"shape": "()", "dtype": "i8"},
        "mean":                     {"shape": "()", "dtype": "f8"},
        "stddev":                   {"shape": "()", "dtype": "f8"},
        "min":                      {"shape": "()", "dtype": "f8"},
        "max":                      {"shape": "()", "dtype": "f8"},
        "in_progress":              {"shape": "()", "dtype": "i8"},
        "in_progress_sum_duration": {"shape": "()", "dtype": "f8"},
    },
)
def analyze(traj: BDTrajectory) -> dict[str, np.ndarray]:
    f = traj._file
    if "contact_durations" not in f:
        raise RuntimeError(
            "trajectory has no /contact_durations group — "
            "re-run the simulation with compute_contact_durations=true"
        )
    a = f["contact_durations"].attrs
    return {
        "count":                    np.asarray(a["count"],                    dtype=np.int64),
        "mean":                     np.asarray(a["mean"],                     dtype=np.float64),
        "stddev":                   np.asarray(a["stddev"],                   dtype=np.float64),
        "min":                      np.asarray(a["min"],                      dtype=np.float64),
        "max":                      np.asarray(a["max"],                      dtype=np.float64),
        "in_progress":              np.asarray(a["in_progress"],              dtype=np.int64),
        "in_progress_sum_duration": np.asarray(a["in_progress_sum_duration"], dtype=np.float64),
    }
