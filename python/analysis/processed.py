"""
processed.py — ProcessedSweep: reader for per-analysis output files.

Each registered analysis writes a self-contained `<name>.h5` at the sweep
root. `ProcessedSweep` opens either:

  - a directory: discovers every `<name>.h5` whose root attr `@analysis`
    matches the filename stem, exposing them as a unified phase-diagram
    reader.
  - a single `.h5` file: loads just that one analysis.

Examples
--------
>>> ps = ProcessedSweep("/path/to/sweep")        # directory
>>> ps.analyses
['clusters', 'force_orientation', 'hexatic', 'msd']
>>> ps.loop_vars
['Pe', 'De']
>>> ps.get('msd', 'msd', kind='mean').shape
(20, 20, 51)

>>> ps = ProcessedSweep("/path/to/sweep/msd.h5")  # single file
>>> ps.analyses
['msd']
>>> ps.get('msd', 'lag_time', kind='mean').shape
(20, 20, 51)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np


class ProcessedSweep:
    """Read-only loader for `<analysis>.h5` files produced by `reduce_sweep`."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Path not found: {self.path}")

        self._files: dict[str, h5py.File] = {}

        if self.path.is_dir():
            self._discover_in_directory(self.path)
        else:
            self._open_single_file(self.path)

        if not self._files:
            raise FileNotFoundError(
                f"No analysis files found under {self.path}"
            )

        # Loop-vars, axes, and seeds must be identical across files in the
        # same sweep — pick any one as canonical and validate the rest.
        first_name = next(iter(self._files))
        first = self._files[first_name]
        self.loop_vars: list[str] = _strlist(first.attrs["loop_vars"])
        self.axes: dict[str, np.ndarray] = {
            v: first["axes"][v][()] for v in self.loop_vars
        }
        self.seeds: np.ndarray = first["axes"]["seeds"][()]
        self.grid_shape: tuple[int, ...] = tuple(
            self.axes[v].size for v in self.loop_vars
        )
        self._validate_consistent_axes(first_name)

        self.analyses: list[str] = sorted(self._files.keys())
        # Convenience: merged attrs (per-file provenance lives on each file).
        self.attrs: dict[str, Any] = {
            k: _attr_value(v) for k, v in first.attrs.items()
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        for f in self._files.values():
            if f.id.valid:
                f.close()
        self._files.clear()

    def __enter__(self) -> "ProcessedSweep":
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self) -> str:
        return (
            f"ProcessedSweep('{self.path.name}', "
            f"loop_vars={self.loop_vars}, "
            f"grid_shape={self.grid_shape}, "
            f"analyses={self.analyses})"
        )

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def datasets(self, analysis: str) -> list[str]:
        """Return dataset keys under `<analysis>.h5` (sorted)."""
        f = self._file_for(analysis)
        return sorted(k for k in f.keys() if k != "axes")

    def axis_order(self, analysis: str, dataset: str) -> list[str]:
        grp = self._dgrp(analysis, dataset)
        return _strlist(grp.attrs.get("axis_order", []))

    def file_path(self, analysis: str) -> Path:
        """Filesystem path of the `<analysis>.h5` file for `analysis`."""
        f = self._file_for(analysis)
        return Path(f.filename)

    def analysis_attrs(self, analysis: str) -> dict[str, Any]:
        """Root-level attrs for one analysis file (provenance + versioning)."""
        f = self._file_for(analysis)
        return {k: _attr_value(v) for k, v in f.attrs.items()}

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get(
        self,
        analysis: str,
        dataset: str,
        *,
        kind: str = "mean",
    ) -> np.ndarray:
        """Return the array at `<analysis>.h5:/<dataset>/<kind>`.

        kind ∈ {"grid", "mean", "sem", "n_valid", "combined"}.
        """
        grp = self._dgrp(analysis, dataset)
        if kind not in grp:
            raise KeyError(
                f"{analysis}.h5:/{dataset}/{kind} not present "
                f"(have: {sorted(grp.keys())})"
            )
        return grp[kind][()]

    def slice(
        self,
        analysis: str,
        dataset: str,
        *,
        kind: str = "mean",
        **idx_kwargs: int,
    ) -> np.ndarray:
        """Slice `<analysis>.h5:/<dataset>/<kind>` by loop-var index selectors.

        Pass selectors as `<var>_idx=N`. Example for a (phi, Pe) sweep::

            ps.slice("msd", "msd", phi_idx=2)
        """
        full = self.get(analysis, dataset, kind=kind)
        sl: list[Any] = [slice(None)] * full.ndim
        for var, val in idx_kwargs.items():
            if not var.endswith("_idx"):
                raise KeyError(f"slice() kwargs must be <var>_idx, not {var}")
            vname = var[:-4]
            if vname not in self.loop_vars:
                raise KeyError(f"unknown loop var {vname!r}; have {self.loop_vars}")
            axis_pos = self.loop_vars.index(vname)
            if not 0 <= val < self.grid_shape[axis_pos]:
                raise IndexError(
                    f"{vname}_idx={val} out of range [0, {self.grid_shape[axis_pos]})"
                )
            sl[axis_pos] = int(val)
        return full[tuple(sl)]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _discover_in_directory(self, dir_path: Path) -> None:
        for cand in sorted(dir_path.glob("*.h5")):
            try:
                f = h5py.File(cand, "r")
            except OSError:
                continue
            try:
                analysis = _attr_value(f.attrs.get("analysis"))
            except Exception:
                f.close()
                continue
            # Treat files without an @analysis attr (or where it doesn't match
            # the stem) as foreign — never auto-discover them.
            if not isinstance(analysis, str) or analysis != cand.stem:
                f.close()
                continue
            self._files[analysis] = f

    def _open_single_file(self, file_path: Path) -> None:
        f = h5py.File(file_path, "r")
        analysis = _attr_value(f.attrs.get("analysis"))
        if not isinstance(analysis, str):
            f.close()
            raise ValueError(
                f"{file_path} has no `@analysis` root attr; not a reduced output."
            )
        self._files[analysis] = f

    def _validate_consistent_axes(self, ref_name: str) -> None:
        """All files in a discovered set must share loop_vars + axes + seeds."""
        ref_loop = self.loop_vars
        ref_axes = self.axes
        ref_seeds = self.seeds
        for name, f in self._files.items():
            if name == ref_name:
                continue
            their_loop = _strlist(f.attrs["loop_vars"])
            if their_loop != ref_loop:
                raise ValueError(
                    f"{name}.h5 loop_vars={their_loop} disagree with "
                    f"{ref_name}.h5 loop_vars={ref_loop}"
                )
            for v in ref_loop:
                if not np.array_equal(f["axes"][v][()], ref_axes[v]):
                    raise ValueError(
                        f"{name}.h5 axis {v!r} disagrees with {ref_name}.h5"
                    )
            if not np.array_equal(f["axes"]["seeds"][()], ref_seeds):
                raise ValueError(
                    f"{name}.h5 seeds disagree with {ref_name}.h5"
                )

    def _file_for(self, analysis: str) -> h5py.File:
        if analysis not in self._files:
            raise KeyError(
                f"analysis {analysis!r} not loaded; have {self.analyses}"
            )
        return self._files[analysis]

    def _dgrp(self, analysis: str, dataset: str) -> h5py.Group:
        f = self._file_for(analysis)
        if dataset not in f:
            raise KeyError(
                f"dataset {dataset!r} not under {analysis}.h5 "
                f"(have: {sorted(k for k in f.keys() if k != 'axes')})"
            )
        return f[dataset]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strlist(v) -> list[str]:
    if v is None:
        return []
    arr = np.asarray(v).ravel().tolist()
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


def _attr_value(v):
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray) and v.dtype.kind in ("U", "O", "S"):
        return _strlist(v)
    if isinstance(v, np.ndarray) and v.ndim == 0:
        return v.item()
    return v
