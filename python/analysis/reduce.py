"""
reduce.py — combine per-trajectory caches into one <analysis>.h5 per analysis.

Each registered analysis writes its reduce output to `<sweep_dir>/<name>.h5`
(or under `output_dir` when supplied). Adding, re-running, or removing one
analysis does not disturb the others.

Per-file schema:

    /                          root
      @sweep_dir, @analysis, @analysis_version, @inputs_hash, @analysis_kwargs
      @loop_vars
      @processed_at, @code_git_sha, @reduce_args
      @n_trajectories_used, @n_trajectories_missing
    /axes/<var_name>           1-D canonical axis (from psweep_info.json)
    /axes/seeds                1-D explicit seed values
    /<dataset_key>/
      @axis_order              ["loop_var_1", ..., "loop_var_k", "seed", *output_axes]
      /grid                    full per-seed grid (gzip)
      /mean                    mean over seeds (np.nanmean)
      /sem                     SEM over seeds (np.nanstd / sqrt(n_valid))
      /n_valid                 finite-seed count per phase point
      /combined                only when an analysis defines combine()

Atomic write per file: tmp + os.replace. Returns the directory holding the
files so callers (ProcessedSweep, tests) can consume them uniformly.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

from . import cache as _cache
from .collector import PsweepCollector
from .registry import get_analysis


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def reduce_sweep(
    coll: PsweepCollector,
    *,
    analysis_specs: list[tuple[str, dict[str, Any]]],
    output_dir: Optional[Path] = None,
    code_git_sha: str = "",
    reduce_args: str = "",
) -> Path:
    """
    Read every cache under `coll`, combine per-analysis, write one
    `<analysis>.h5` per registered analysis.

    Returns the directory containing the written files (sweep root by default).
    """
    out_dir = Path(output_dir) if output_dir is not None else coll.processed_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    for analysis_name, kwargs in analysis_specs:
        _reduce_one_analysis(
            coll,
            analysis_name=analysis_name,
            kwargs=kwargs,
            out_path=out_dir / f"{analysis_name}.h5",
            code_git_sha=code_git_sha,
            reduce_args=reduce_args,
        )

    return out_dir


def _reduce_one_analysis(
    coll: PsweepCollector,
    *,
    analysis_name: str,
    kwargs: dict[str, Any],
    out_path: Path,
    code_git_sha: str,
    reduce_args: str,
) -> None:
    desc = get_analysis(analysis_name)
    inputs_hash = desc.inputs_hash(kwargs)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    # Pull all per-(combo, seed) caches for this analysis up front.
    per_seed: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    used: set[tuple[int, int]] = set()
    missing: set[tuple[int, int]] = set()
    for combo_idx, seed, _ in coll:
        cp = coll.cache_path(combo_idx, seed)
        data = _cache.read_analysis(cp, analysis_name)
        if data is None:
            missing.add((combo_idx, seed))
            continue
        used.add((combo_idx, seed))
        per_seed[(combo_idx, seed)] = data

    with h5py.File(tmp_path, "w") as f:
        # Axes group.
        ax = f.create_group("axes")
        for v in coll.loop_vars:
            ax.create_dataset(v, data=coll.axes[v])
        ax.create_dataset("seeds", data=np.asarray(coll.seeds, dtype=np.int64))

        if per_seed:
            # Per-dataset reductions. analyze()'s output shapes must be
            # consistent across (combo, seed) for a given key.
            keys = sorted({k for v in per_seed.values() for k in v.keys()})
            for key in keys:
                values = {
                    pair: per_seed[pair][key]
                    for pair in per_seed
                    if key in per_seed[pair]
                }
                if not values:
                    continue
                _write_dataset_group(
                    f, key, coll, values, axis_order_prefix=coll.loop_vars
                )

            # Optional combine() — per-combo aggregate written under /<key>/combined.
            if desc.combine is not None:
                per_combo_combined: dict[int, dict[str, np.ndarray]] = {}
                for combo_idx in range(coll.n_combos):
                    seeds_present = [
                        per_seed[(combo_idx, s)]
                        for s in coll.seeds
                        if (combo_idx, s) in per_seed
                    ]
                    if not seeds_present:
                        continue
                    try:
                        per_combo_combined[combo_idx] = desc.combine(seeds_present)
                    except Exception as e:
                        f.attrs[f"combine_error__combo_{combo_idx:04d}"] = (
                            f"{type(e).__name__}: {e}"
                        )

                if per_combo_combined:
                    combine_keys = sorted(
                        {k for v in per_combo_combined.values() for k in v.keys()}
                    )
                    for key in combine_keys:
                        per_combo_values = {
                            ci: per_combo_combined[ci][key]
                            for ci in per_combo_combined
                            if key in per_combo_combined[ci]
                        }
                        if not per_combo_values:
                            continue
                        target = (
                            f[key] if key in f else f.create_group(key)
                        )
                        grid = coll.assemble_grid_per_combo(per_combo_values)
                        target.create_dataset(
                            "combined",
                            data=grid,
                            compression="gzip",
                            compression_opts=4,
                        )
        else:
            f.attrs["empty"] = True

        # Provenance attrs.
        f.attrs["sweep_dir"] = str(coll.sweep_dir)
        f.attrs["analysis"] = analysis_name
        f.attrs["analysis_version"] = int(desc.version)
        f.attrs["inputs_hash"] = inputs_hash
        f.attrs["analysis_kwargs"] = json.dumps(kwargs, sort_keys=True, default=str)
        f.attrs["loop_vars"] = np.array(coll.loop_vars, dtype=h5py.string_dtype())
        f.attrs["processed_at"] = _now_iso()
        f.attrs["code_git_sha"] = code_git_sha
        f.attrs["reduce_args"] = reduce_args
        f.attrs["n_trajectories_used"] = len(used)
        f.attrs["n_trajectories_missing"] = len(missing)

    os.replace(tmp_path, out_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_dataset_group(
    parent: h5py.Group,
    key: str,
    coll: PsweepCollector,
    values: dict[tuple[int, int], np.ndarray],
    *,
    axis_order_prefix: list[str],
) -> None:
    """Assemble grid + mean + sem + n_valid for one analysis dataset under `parent`."""
    sample = np.asarray(next(iter(values.values())))
    tail_shape = sample.shape

    grid = coll.assemble_grid_per_seed(values, fill=np.nan)
    # grid shape: grid_shape + (num_seeds,) + tail_shape

    seed_axis = len(coll.grid_shape)
    finite = np.isfinite(grid)
    n_valid = finite.sum(axis=seed_axis).astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(grid, axis=seed_axis)
        std = np.nanstd(grid, axis=seed_axis, ddof=1)
        denom = np.sqrt(np.where(n_valid > 1, n_valid, 1)).astype(np.float64)
        sem = np.where(n_valid > 1, std / denom, np.nan)

    target = parent.create_group(key)
    target.attrs["axis_order"] = np.array(
        list(axis_order_prefix) + ["seed"] + [f"axis_{i}" for i in range(len(tail_shape))],
        dtype=h5py.string_dtype(),
    )

    target.create_dataset(
        "grid",
        data=grid,
        compression="gzip",
        compression_opts=4,
    )
    target.create_dataset("mean", data=mean)
    target.create_dataset("sem", data=sem)
    target.create_dataset("n_valid", data=n_valid)


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
