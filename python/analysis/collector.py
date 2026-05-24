"""
collector.py — PsweepCollector: read-only navigator over a parameter sweep.

The collector parses the three JSON manifests written by psweep.py
(log.json, psweep_map.json, psweep_info.json) and exposes:

  - axes (canonical 1-D arrays per loop variable, recomputed from the loop
    spec — float-roundoff-safe)
  - per-(combo, seed) trajectory paths and cache paths
  - lazy trajectory open via .open(combo_idx, seed) -> BDTrajectory
  - cache-status enumeration for map/reduce orchestration
  - grid assemblers that pivot combo-indexed values into N-D phase-diagram arrays

It does not open trajectory files eagerly. Callers control file lifetime so
multi-process workers can safely instantiate their own BDTrajectory after
spawn.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

# Allow `from bdtrajectory import BDTrajectory` regardless of how the package
# is imported. python/ is on sys.path via the existing notebook idiom.
_PYTHON_DIR = Path(__file__).resolve().parent.parent
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from bdtrajectory import BDTrajectory  # noqa: E402


CACHE_SUFFIX = ".cache.h5"


class PsweepCollector:
    """
    Read-only navigator over a sweep directory produced by psweep.py.

    Examples
    --------
    >>> coll = PsweepCollector("/path/to/psweep_2026-05-07_phi_dense")
    >>> coll.loop_vars
    ['phi']
    >>> coll.axes['phi']
    array([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> for combo_idx, seed, path in coll:
    ...     with coll.open(combo_idx, seed) as traj:
    ...         ...
    """

    def __init__(self, sweep_dir: str | Path) -> None:
        self.sweep_dir = Path(sweep_dir).resolve()
        if not self.sweep_dir.is_dir():
            raise FileNotFoundError(f"Sweep directory not found: {self.sweep_dir}")

        self.sweep_name = self.sweep_dir.name

        log_path = self.sweep_dir / "log.json"
        map_path = self.sweep_dir / "psweep_map.json"
        info_path = self.sweep_dir / "psweep_info.json"
        for p in (log_path, map_path, info_path):
            if not p.is_file():
                raise FileNotFoundError(
                    f"Manifest missing: {p} — is this really a psweep directory?"
                )

        with open(log_path) as f:
            self.log: dict[str, Any] = json.load(f)
        with open(map_path) as f:
            self.map: dict[str, Any] = json.load(f)
        with open(info_path) as f:
            self.loop_info: dict[str, dict[str, Any]] = json.load(f)

        self.loop_vars: list[str] = list(self.log["loop_vars"])
        self.n_combos: int = int(self.log["n_combos"])
        self.start_seed: int = int(self.log["start_seed"])
        self.num_seeds: int = int(self.log["num_seeds"])
        self.seeds: list[int] = list(
            range(self.start_seed, self.start_seed + self.num_seeds)
        )
        self.scratch_dir: Optional[str] = self.log.get("scratch_dir")

        # Canonical axes — derived from the loop spec, NOT from psweep_map.json,
        # which stores np.linspace float-roundoff artifacts.
        self.axes: dict[str, np.ndarray] = {
            v: _axis_from_spec(self.loop_info[v]) for v in self.loop_vars
        }
        self.grid_shape: tuple[int, ...] = tuple(
            len(self.axes[v]) for v in self.loop_vars
        )
        if int(np.prod(self.grid_shape)) != self.n_combos:
            warnings.warn(
                f"Sweep {self.sweep_name}: prod(grid_shape)={int(np.prod(self.grid_shape))} "
                f"!= n_combos={self.n_combos}. Manifests may be inconsistent.",
                stacklevel=2,
            )

        # Detect combo-id padding overflow (caveat #8 in the plan).
        if self.n_combos > 9999:
            warnings.warn(
                f"Sweep {self.sweep_name}: n_combos={self.n_combos} exceeds the "
                "4-digit zero-padding used by psweep.py — trajectory paths may "
                "collide. Bump psweep.py to :05d.",
                stacklevel=2,
            )

        # Cache parsed input JSONs so expected_num_frames is cheap.
        self._input_json_cache: dict[tuple[int, int], dict] = {}

    # ------------------------------------------------------------------
    # Identity / paths
    # ------------------------------------------------------------------

    def _run_dir(self, combo_idx: int) -> Path:
        self._check_combo(combo_idx)
        return self.sweep_dir / f"{self.sweep_name}_{combo_idx:04d}"

    def traj_path(self, combo_idx: int, seed: int) -> Path:
        """Absolute path to the trajectory .h5 for (combo_idx, seed)."""
        self._check_seed(seed)
        return (
            self._run_dir(combo_idx)
            / f"{self.sweep_name}_{combo_idx:04d}_seed{seed}.h5"
        )

    def cache_path(self, combo_idx: int, seed: int) -> Path:
        """Absolute path to the per-trajectory cache (.cache.h5).

        We strip only the trailing '.h5' with traj.stem rather than chaining
        with_suffix() calls, because the sweep name may contain dots (e.g.
        'phi0.5_del0.95') and with_suffix('') would eat those too.
        """
        traj = self.traj_path(combo_idx, seed)
        return traj.parent / (traj.stem + CACHE_SUFFIX)

    def input_json_path(self, combo_idx: int, seed: int) -> Path:
        self._check_seed(seed)
        return self._run_dir(combo_idx) / f"input_seed{seed}.json"

    def processed_dir(self) -> Path:
        """Directory where per-analysis `<name>.h5` files are written."""
        return self.sweep_dir

    def analysis_output_path(self, analysis_name: str) -> Path:
        """Path to the reduced `<analysis>.h5` for one registered analysis."""
        return self.processed_dir() / f"{analysis_name}.h5"

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    def combo_params(self, combo_idx: int) -> dict[str, float]:
        """Display-quality parameter values for combo (from psweep_map.json)."""
        self._check_combo(combo_idx)
        return dict(self.map["combos"][str(combo_idx)])

    def combo_indices_grid(self) -> np.ndarray:
        """Return an array of shape `grid_shape` whose entries are combo_idx.

        Maps an axis-tuple position to the original combo index produced by
        psweep.py's `itertools.product` order (the first variable in
        loop_vars is the outermost loop).
        """
        out = np.arange(self.n_combos, dtype=np.int64).reshape(self.grid_shape)
        return out

    def _input_json(self, combo_idx: int, seed: int) -> dict:
        key = (combo_idx, seed)
        if key not in self._input_json_cache:
            with open(self.input_json_path(combo_idx, seed)) as f:
                self._input_json_cache[key] = json.load(f)
        return self._input_json_cache[key]

    def expected_num_frames(self, combo_idx: int, seed: Optional[int] = None) -> int:
        """
        Expected number of frames in trajectories for (combo_idx, seed).

        Inferred from the per-run input JSON's `t_end` and `output_dt`. Frame 0
        plus N output_dt steps yields `floor(t_end / output_dt) + 1`.
        """
        if seed is None:
            seed = self.start_seed
        cfg = self._input_json(combo_idx, seed)
        t_end = float(cfg["t_end"])
        out_dt = float(cfg["output_dt"])
        if out_dt <= 0:
            raise ValueError(
                f"output_dt={out_dt} in {self.input_json_path(combo_idx, seed)}"
            )
        return int(np.floor(t_end / out_dt + 1e-9)) + 1

    # ------------------------------------------------------------------
    # Iteration / filtering
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[tuple[int, int, Path]]:
        for ci in range(self.n_combos):
            for seed in self.seeds:
                yield ci, seed, self.traj_path(ci, seed)

    def __len__(self) -> int:
        return self.n_combos * self.num_seeds

    def iter_combos(self) -> Iterator[tuple[int, dict[str, float]]]:
        for ci in range(self.n_combos):
            yield ci, self.combo_params(ci)

    def filter(
        self,
        *,
        combo_idx: Optional[int | list[int]] = None,
        seed: Optional[int | list[int]] = None,
        **idx_kwargs: int | list[int],
    ) -> Iterator[tuple[int, int, Path]]:
        """
        Filter (combo, seed) pairs.

        Use `combo_idx` and `seed` for direct integer matching. For
        parameter-axis filtering, pass `<varname>_idx=N` (NOT `<varname>=value`
        — float roundoff makes value matching unreliable).

        Examples
        --------
        >>> for c, s, p in coll.filter(phi_idx=2):  # all seeds at the 3rd phi
        ...     ...
        >>> list(coll.filter(combo_idx=[0, 1, 2], seed=0))
        """
        combo_set = _as_set(combo_idx, range(self.n_combos))
        seed_set = _as_set(seed, self.seeds)

        for var, sel in idx_kwargs.items():
            if not var.endswith("_idx") or var[:-4] not in self.loop_vars:
                raise KeyError(
                    f"filter kwarg {var!r} must be one of: "
                    f"{[v + '_idx' for v in self.loop_vars]}"
                )
            vname = var[:-4]
            allowed_axis = _as_set(sel, range(len(self.axes[vname])))
            grid = self.combo_indices_grid()
            axis_pos = self.loop_vars.index(vname)
            mask = np.zeros(self.n_combos, dtype=bool)
            sl: list[Any] = [slice(None)] * len(self.grid_shape)
            sl[axis_pos] = list(allowed_axis)
            mask[grid[tuple(sl)].ravel()] = True
            combo_set &= set(np.flatnonzero(mask).tolist())

        for ci in sorted(combo_set):
            for s in sorted(seed_set):
                yield ci, s, self.traj_path(ci, s)

    # ------------------------------------------------------------------
    # File / cache lifecycle
    # ------------------------------------------------------------------

    def open(self, combo_idx: int, seed: int) -> BDTrajectory:
        """Return a freshly-opened BDTrajectory. Caller is responsible for closing.

        Designed to be safe to call inside multiprocessing workers — it opens
        the HDF5 file in the calling process, never inheriting an open handle
        across fork.
        """
        return BDTrajectory(self.traj_path(combo_idx, seed))

    def find_existing_trajectories(self) -> list[tuple[int, int]]:
        """Return all (combo_idx, seed) pairs whose .h5 file currently exists."""
        return [(c, s) for c, s, p in self if p.exists()]

    def find_missing_trajectories(self) -> list[tuple[int, int]]:
        return [(c, s) for c, s, p in self if not p.exists()]

    # ------------------------------------------------------------------
    # Grid assembly
    # ------------------------------------------------------------------

    def assemble_grid_per_combo(
        self,
        values: dict[int, np.ndarray],
        *,
        fill: float | int = np.nan,
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        """
        Reshape a {combo_idx -> ndarray-or-scalar} mapping into an N-D grid.

        Output shape: grid_shape + tail_shape, where tail_shape is the shape
        of one entry. Missing combos are filled with `fill`. Useful for
        reducing combine()-style outputs (one value per combo, already
        averaged over seeds) into a phase diagram.
        """
        if not values:
            raise ValueError("assemble_grid_per_combo: empty values dict")
        sample = next(iter(values.values()))
        sample = np.asarray(sample)
        out_dtype = dtype if dtype is not None else _grid_dtype(sample, fill)
        tail_shape = sample.shape

        grid = np.full(self.grid_shape + tail_shape, fill, dtype=out_dtype)
        ci_grid = self.combo_indices_grid()
        for combo_idx, val in values.items():
            arr = np.asarray(val)
            if arr.shape != tail_shape:
                raise ValueError(
                    f"combo {combo_idx}: shape {arr.shape} != reference {tail_shape}"
                )
            pos = tuple(np.argwhere(ci_grid == combo_idx)[0])
            grid[pos] = arr
        return grid

    def assemble_grid_per_seed(
        self,
        values: dict[tuple[int, int], np.ndarray],
        *,
        fill: float | int = np.nan,
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        """
        Reshape a {(combo_idx, seed) -> ndarray} mapping into an N-D grid.

        Output shape: grid_shape + (num_seeds,) + tail_shape. The seeds axis
        follows ``self.seeds`` ordering. Missing entries are NaN-filled.
        """
        if not values:
            raise ValueError("assemble_grid_per_seed: empty values dict")
        sample = np.asarray(next(iter(values.values())))
        out_dtype = dtype if dtype is not None else _grid_dtype(sample, fill)
        tail_shape = sample.shape

        grid = np.full(
            self.grid_shape + (self.num_seeds,) + tail_shape,
            fill,
            dtype=out_dtype,
        )
        ci_grid = self.combo_indices_grid()
        seed_pos = {s: i for i, s in enumerate(self.seeds)}
        for (combo_idx, seed), val in values.items():
            arr = np.asarray(val)
            if arr.shape != tail_shape:
                raise ValueError(
                    f"({combo_idx}, {seed}): shape {arr.shape} != reference {tail_shape}"
                )
            pos = tuple(np.argwhere(ci_grid == combo_idx)[0]) + (seed_pos[seed],)
            grid[pos] = arr
        return grid

    # ------------------------------------------------------------------
    # Internal validators
    # ------------------------------------------------------------------

    def _check_combo(self, combo_idx: int) -> None:
        if not 0 <= combo_idx < self.n_combos:
            raise IndexError(
                f"combo_idx {combo_idx} out of range [0, {self.n_combos})"
            )

    def _check_seed(self, seed: int) -> None:
        if seed not in self.seeds:
            raise IndexError(
                f"seed {seed} not in sweep seeds {self.seeds}"
            )

    def __repr__(self) -> str:
        return (
            f"PsweepCollector('{self.sweep_name}', "
            f"loop_vars={self.loop_vars}, "
            f"grid_shape={self.grid_shape}, "
            f"num_seeds={self.num_seeds})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _axis_from_spec(spec: dict[str, Any]) -> np.ndarray:
    """Recompute the canonical axis values from a {start, stop, num, format} spec."""
    start = float(spec["start"])
    stop = float(spec["stop"])
    num = int(spec["num"])
    fmt = str(spec["format"])
    if num < 1:
        raise ValueError(f"num={num} must be >= 1")
    if fmt == "lin":
        return np.linspace(start, stop, num)
    if fmt == "log":
        if start <= 0 or stop <= 0:
            raise ValueError("log spacing requires start > 0 and stop > 0")
        return np.logspace(np.log10(start), np.log10(stop), num)
    raise ValueError(f"Unknown format {fmt!r} (expected 'lin' or 'log')")


def _as_set(sel: Any, default_iter) -> set:
    if sel is None:
        return set(default_iter)
    if isinstance(sel, (int, np.integer)):
        return {int(sel)}
    return set(int(x) for x in sel)


def _grid_dtype(sample: np.ndarray, fill) -> np.dtype:
    """Pick a dtype that can hold both the sample values and the fill value."""
    base = sample.dtype
    # NaN fill requires a float dtype.
    if isinstance(fill, float) and np.isnan(fill):
        if np.issubdtype(base, np.floating):
            return base
        return np.dtype(np.float64)
    return base
