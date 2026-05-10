"""
io.py — bulk-load helpers for trajectory data.

The simulation writer stores one HDF5 group per output frame, so reading T
frames means T group lookups + T dataset opens. For T~1000 this is several
hundred milliseconds of pure h5py overhead per array on top of the actual
read. `stack_dataset` collapses that into a single (T, *shape) NumPy array
with one read-loop, so analyses that need full time series (MSD, FFT
correlations) only pay the cost once per trajectory.

Returns None when the requested dataset is absent on every frame (matches
BDTrajectory's per-frame None semantics for forces/orientations on legacy
files).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from bdtrajectory import BDTrajectory


def stack_dataset(traj: BDTrajectory, name: str) -> Optional[np.ndarray]:
    """
    Stack a per-frame dataset across all frames into one ndarray.

    Parameters
    ----------
    traj : BDTrajectory
        Open trajectory.
    name : str
        One of "positions", "velocities", "orientations", "anchors", "forces".

    Returns
    -------
    np.ndarray of shape (T, *frame_shape), or None if the dataset is missing
    from frame 0 (older trajectory format).
    """
    accessor = getattr(traj, name, None)
    if accessor is None or not callable(accessor):
        raise AttributeError(
            f"BDTrajectory has no per-frame accessor named {name!r}"
        )

    first = accessor(0)
    if first is None:
        return None

    out = np.empty((traj.num_frames,) + first.shape, dtype=first.dtype)
    out[0] = first
    for i in range(1, traj.num_frames):
        out[i] = accessor(i)
    return out


def all_times(traj: BDTrajectory) -> np.ndarray:
    """1-D array of physical times across all frames (cached convenience)."""
    return traj.all_times()
