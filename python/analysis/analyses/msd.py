"""
msd — ensemble-averaged mean squared displacement vs reference frame.

MSD(i) = < |r_i - r_ref|^2 >_particles. Trajectory positions are stored
unwrapped by the engine (the integrator does not fold positions back into the
primary cell), so the naive difference is the true displacement at any lag.
"""

from __future__ import annotations

import numpy as np

from bdtrajectory import BDTrajectory

from ..io import stack_dataset
from ..registry import register_analysis


@register_analysis(
    name="msd",
    version=1,
    requires=("positions",),
    outputs={
        "msd":      {"shape": "(num_frames,)", "dtype": "f8", "axes": ["frame"]},
        "lag_time": {"shape": "(num_frames,)", "dtype": "f8", "axes": ["frame"]},
    },
)
def analyze(traj: BDTrajectory, *, reference_frame: int = 0) -> dict[str, np.ndarray]:
    pos = stack_dataset(traj, "positions")  # (T, N, 2)
    if pos is None:
        raise RuntimeError("positions dataset is missing — cannot compute MSD")

    if not 0 <= reference_frame < pos.shape[0]:
        raise IndexError(
            f"reference_frame={reference_frame} out of range [0, {pos.shape[0]})"
        )

    delta = pos - pos[reference_frame]                 # (T, N, 2)
    msd = np.mean(np.sum(delta * delta, axis=-1), axis=-1)  # (T,)

    times = traj.all_times()
    lag = times - times[reference_frame]

    return {"msd": msd.astype(np.float64), "lag_time": lag.astype(np.float64)}
