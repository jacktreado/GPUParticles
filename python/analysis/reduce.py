"""
reduce.py — combine per-trajectory caches into a single processed.h5.

Schema (see plan):

    /                          root
      @sweep_dir, @loop_vars, @analyses
      @processed_at, @code_git_sha, @reduce_args
      @n_trajectories_used, @n_trajectories_missing
    /axes/<var_name>           1-D canonical axis (from psweep_info.json)
    /axes/seeds                1-D explicit seed values
    /<analysis_name>/<dataset>/
      @axis_order              ["loop_var_1", ..., "loop_var_k", "seed", *output_axes]
      /grid                    full per-seed grid
      /mean                    mean over seeds (np.nanmean)
      /sem                     SEM over seeds (np.nanstd / sqrt(n_valid))
      /n_valid                 finite-seed count per phase point
      /combined                only when an analysis defines combine()

We compress /grid with gzip; mean/sem/n_valid stay uncompressed (small).
Atomic write: tmp file + os.replace.
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
    output_path: Optional[Path] = None,
    code_git_sha: str = "",
    reduce_args: str = "",
) -> Path:
    """
    Read every cache under `coll`, combine per-analysis, write the final h5.

    Returns the path of the written processed.h5.
    """
    out_path = (
        Path(output_path)
        if output_path is not None
        else coll.processed_path("processed.h5")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    used: set[tuple[int, int]] = set()
    missing: set[tuple[int, int]] = set()

    with h5py.File(tmp_path, "w") as f:
        # Axes group.
        ax = f.create_group("axes")
        for v in coll.loop_vars:
            ax.create_dataset(v, data=coll.axes[v])
        ax.create_dataset("seeds", data=np.asarray(coll.seeds, dtype=np.int64))

        # Per-analysis output.
        for analysis_name, kwargs in analysis_specs:
            desc = get_analysis(analysis_name)
            inputs_hash = desc.inputs_hash(kwargs)
            agroup = f.create_group(analysis_name)
            agroup.attrs["analysis_version"] = int(desc.version)
            agroup.attrs["inputs_hash"] = inputs_hash
            agroup.attrs["analysis_kwargs"] = json.dumps(
                kwargs, sort_keys=True, default=str
            )

            # Collect per-seed dicts keyed by (combo, seed).
            per_seed: dict[tuple[int, int], dict[str, np.ndarray]] = {}
            for combo_idx, seed, traj_path in coll:
                cp = coll.cache_path(combo_idx, seed)
                data = _cache.read_analysis(cp, analysis_name)
                if data is None:
                    missing.add((combo_idx, seed))
                    continue
                used.add((combo_idx, seed))
                per_seed[(combo_idx, seed)] = data

            if not per_seed:
                # Mark group as empty but still record it so consumers see "we tried".
                agroup.attrs["empty"] = True
                continue

            # Each dataset key is processed independently. The datasets
            # produced by analyze() must have consistent shapes across
            # successful (combo, seed) entries — if not, the assemble call
            # will raise.
            keys = sorted({k for v in per_seed.values() for k in v.keys()})
            for key in keys:
                values = {
                    pair: per_seed[pair][key]
                    for pair in per_seed
                    if key in per_seed[pair]
                }
                if not values:
                    continue
                _write_dataset_group(agroup, key, coll, values, axis_order_prefix=coll.loop_vars)

            # Optional combine().
            if desc.combine is not None:
                # Group per-seed results by combo, run combine() per combo.
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
                        agroup.attrs[f"combine_error__combo_{combo_idx:04d}"] = (
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
                            agroup[key] if key in agroup else agroup.create_group(key)
                        )
                        grid = coll.assemble_grid_per_combo(per_combo_values)
                        target.create_dataset(
                            "combined",
                            data=grid,
                            compression="gzip",
                            compression_opts=4,
                        )

        # Top-level provenance.
        f.attrs["sweep_dir"] = str(coll.sweep_dir)
        f.attrs["loop_vars"] = np.array(coll.loop_vars, dtype=h5py.string_dtype())
        f.attrs["analyses"] = np.array(
            [n for n, _ in analysis_specs], dtype=h5py.string_dtype()
        )
        f.attrs["processed_at"] = _now_iso()
        f.attrs["code_git_sha"] = code_git_sha
        f.attrs["reduce_args"] = reduce_args
        f.attrs["n_trajectories_used"] = len(used)
        f.attrs["n_trajectories_missing"] = len(missing)

    os.replace(tmp_path, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_dataset_group(
    agroup: h5py.Group,
    key: str,
    coll: PsweepCollector,
    values: dict[tuple[int, int], np.ndarray],
    *,
    axis_order_prefix: list[str],
) -> None:
    """Assemble grid + mean + sem + n_valid for one analysis dataset."""
    sample = np.asarray(next(iter(values.values())))
    tail_shape = sample.shape

    grid = coll.assemble_grid_per_seed(values, fill=np.nan)
    # grid shape: grid_shape + (num_seeds,) + tail_shape

    # Compute seed-axis statistics (handle NaN for missing seeds).
    seed_axis = len(coll.grid_shape)
    finite = np.isfinite(grid)
    n_valid = finite.sum(axis=seed_axis).astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(grid, axis=seed_axis)
        std = np.nanstd(grid, axis=seed_axis, ddof=1)
        # SEM = std / sqrt(n_valid). Where n_valid <= 1 the SEM is undefined.
        denom = np.sqrt(np.where(n_valid > 1, n_valid, 1)).astype(np.float64)
        sem = np.where(n_valid > 1, std / denom, np.nan)

    target = agroup.create_group(key)
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
