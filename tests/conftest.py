"""Test config: makes `python/` importable and exposes a tiny-sweep fixture."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

# Defends against ghost lock files on shared filesystems; harmless locally.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


@pytest.fixture
def mini_sweep(tmp_path):
    """Build a tiny synthetic sweep mirroring the layout psweep.py produces.

    3 combos x 2 seeds x 5 frames x N=8 particles, single loop_var=phi.
    Returns the sweep directory Path.
    """
    return _build_mini_sweep(tmp_path / "psweep_TEST_mini")


def _build_mini_sweep(sweep_dir: Path) -> Path:
    sweep_dir.mkdir(parents=True, exist_ok=True)
    sweep_name = sweep_dir.name

    n_combos = 3
    num_seeds = 2
    start_seed = 0
    num_frames = 5
    N = 8
    output_dt = 0.5
    t_end = output_dt * (num_frames - 1)
    sigma = 1.0
    L = 10.0
    dt = 0.001
    f0 = 1.0
    tau_theta = 1.0

    phi_axis = np.linspace(0.1, 0.5, n_combos)

    log = {
        "psweep_name": sweep_name,
        "date": "2026-05-10",
        "tag": "test",
        "note": "synthetic mini sweep",
        "base_json": "",
        "binary": "",
        "overrides": {},
        "loop_info": {
            "phi": {"start": 0.1, "stop": 0.5, "num": n_combos, "format": "lin"}
        },
        "loop_vars": ["phi"],
        "n_combos": n_combos,
        "start_seed": start_seed,
        "num_seeds": num_seeds,
        "scratch_dir": None,
    }
    psweep_map = {
        "psweep_name": sweep_name,
        "loop_vars": ["phi"],
        "start_seed": start_seed,
        "num_seeds": num_seeds,
        "combos": {
            str(i): {"phi": float(phi_axis[i])} for i in range(n_combos)
        },
    }
    psweep_info = {
        "phi": {"start": 0.1, "stop": 0.5, "num": n_combos, "format": "lin"}
    }
    import json

    (sweep_dir / "log.json").write_text(json.dumps(log, indent=2))
    (sweep_dir / "psweep_map.json").write_text(json.dumps(psweep_map, indent=2))
    (sweep_dir / "psweep_info.json").write_text(json.dumps(psweep_info, indent=2))

    # Per-combo run dirs + per-seed input JSONs + trajectory .h5 files.
    rng = np.random.default_rng(0)
    for ci in range(n_combos):
        run_name = f"{sweep_name}_{ci:04d}"
        run_dir = sweep_dir / run_name
        run_dir.mkdir(exist_ok=True)
        for seed in range(start_seed, start_seed + num_seeds):
            cfg = {
                "N": N,
                "phi": float(phi_axis[ci]),
                "sigma": sigma,
                "L": L,
                "dt": dt,
                "f0": f0,
                "tau_theta": tau_theta,
                "kT": 1.0,
                "gamma": 1.0,
                "output_dt": output_dt,
                "t_end": t_end,
                "seed": int(seed),
                "output_file": str(run_dir / f"{run_name}_seed{seed}.h5"),
                "potential": "WCA",
                "init_mode": "lattice",
                "integrator": "heun",
            }
            (run_dir / f"input_seed{seed}.json").write_text(json.dumps(cfg, indent=2))
            _write_traj(
                Path(cfg["output_file"]),
                cfg=cfg,
                num_frames=num_frames,
                rng=rng,
            )
    return sweep_dir


def _write_traj(path: Path, *, cfg: dict, num_frames: int, rng: np.random.Generator) -> None:
    """Write a single trajectory .h5 matching the bdtrajectory schema."""
    N = cfg["N"]
    L = cfg["L"]
    output_dt = cfg["output_dt"]

    # Lattice initial positions, then small random walk so MSD is non-zero.
    side = int(np.ceil(np.sqrt(N)))
    spacing = L / side
    x = (np.arange(N) % side + 0.5) * spacing
    y = (np.arange(N) // side + 0.5) * spacing
    pos = np.stack([x, y], axis=-1)
    theta = rng.uniform(0, 2 * np.pi, size=N)

    with h5py.File(path, "w") as f:
        # Root attrs (subset of what BDTrajectory expects).
        for k in ("N", "phi", "sigma", "L", "dt", "f0", "tau_theta", "kT", "gamma", "output_dt"):
            f.attrs[k] = cfg[k]
        f.attrs["seed"] = int(cfg["seed"])

        for k in range(num_frames):
            grp = f.create_group(f"frame_{k:08d}")
            grp.attrs["step"] = int(k * output_dt / cfg["dt"])
            grp.attrs["time"] = float(k * output_dt)
            grp.attrs["frame"] = int(k)
            grp.attrs["Lx"] = float(L)
            grp.attrs["Ly"] = float(L)

            pos = pos + rng.normal(scale=0.05, size=pos.shape)
            grp.create_dataset("positions", data=pos)
            # Velocities, orientations, anchors, forces — present so all
            # registered analyses run end-to-end. Forces deliberately
            # anti-aligned with orientation so F.n is negative on average.
            nhat = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
            forces = -0.3 * nhat + 0.05 * rng.normal(size=nhat.shape)
            grp.create_dataset("velocities", data=rng.normal(size=pos.shape))
            grp.create_dataset("orientations", data=theta)
            grp.create_dataset("anchors", data=pos)
            grp.create_dataset("forces", data=forces)
