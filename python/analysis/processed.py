"""
processed.py — ProcessedSweep: reader for the final processed.h5.

Symmetric to PsweepCollector but for downloaded post-processed files. Lets
notebooks pull a single phase-diagram array out without hand-decoding the
schema.

Examples
--------
>>> ps = ProcessedSweep("processed.h5")
>>> ps.loop_vars
['phi']
>>> ps.axes['phi']
array([0.1, 0.2, 0.3, 0.4, 0.5])
>>> ps.analyses
['msd']
>>> ps.datasets('msd')
['lag_time', 'msd']
>>> ps.get('msd', 'msd', kind='mean').shape
(5, 1000)                       # (axis_phi, frame)
>>> ps.get('msd', 'msd', kind='grid').shape
(5, 10, 1000)                   # (axis_phi, seed, frame)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np


class ProcessedSweep:
    """Read-only loader for processed.h5 files produced by `reduce_sweep`."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Processed sweep not found: {self.path}")
        self._file = h5py.File(self.path, "r")

        self.loop_vars: list[str] = _strlist(self._file.attrs["loop_vars"])
        self.analyses: list[str] = _strlist(self._file.attrs["analyses"])
        self.axes: dict[str, np.ndarray] = {
            v: self._file["axes"][v][()] for v in self.loop_vars
        }
        self.seeds: np.ndarray = self._file["axes"]["seeds"][()]
        self.grid_shape: tuple[int, ...] = tuple(
            self.axes[v].size for v in self.loop_vars
        )
        self.attrs: dict[str, Any] = {k: _attr_value(v) for k, v in self._file.attrs.items()}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._file.id.valid:
            self._file.close()

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
        """Return dataset keys under /<analysis>/, in alphabetic order."""
        if analysis not in self._file:
            raise KeyError(f"analysis {analysis!r} not in {self.path}")
        return sorted(self._file[analysis].keys())

    def axis_order(self, analysis: str, dataset: str) -> list[str]:
        grp = self._dgrp(analysis, dataset)
        return _strlist(grp.attrs.get("axis_order", []))

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
        """
        Return the array at /<analysis>/<dataset>/<kind>.

        kind ∈ {"grid", "mean", "sem", "n_valid", "combined"}.
        """
        grp = self._dgrp(analysis, dataset)
        if kind not in grp:
            raise KeyError(
                f"/{analysis}/{dataset}/{kind} not present "
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
        """
        Return a slice of /<analysis>/<dataset>/<kind> selected by axis indices.

        Pass loop-variable index selectors as `<var>_idx=N`. Example for a
        2-D (phi, Pe) sweep::

            ps.slice("msd", "msd", phi_idx=2)  # returns (Pe-axis, seed?, frame) slice
        """
        full = self.get(analysis, dataset, kind=kind)
        order = self.axis_order(analysis, dataset)
        # Strip the trailing axes that aren't sweep axes.
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

    def _dgrp(self, analysis: str, dataset: str) -> h5py.Group:
        if analysis not in self._file:
            raise KeyError(f"analysis {analysis!r} not in {self.path}")
        agroup = self._file[analysis]
        if dataset not in agroup:
            raise KeyError(
                f"dataset {dataset!r} not under /{analysis} "
                f"(have: {sorted(agroup.keys())})"
            )
        return agroup[dataset]


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
