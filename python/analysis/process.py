#!/usr/bin/env python3
"""
process.py — CLI for the GPUParticles cluster-side analysis pipeline.

Subcommands:
    map     <sweep_dir> --analyses ... [--backend local|slurm] ...
    reduce  <sweep_dir> --analyses ... [--output processed.h5]
    one     <sweep_dir> <combo_idx> <seed> --analyses ...   (SLURM array entry)
    status  <sweep_dir> [--analyses ...]
    clean   <sweep_dir> [--analyses ...] [--stale-only]

See the plan doc in /Users/jacktreado/.claude/plans for the design rationale.
"""

from __future__ import annotations

# IMPORTANT: Set HDF5_USE_FILE_LOCKING before importing h5py anywhere. Same
# defense as analysis/__init__.py — repeated here so that running
# `python process.py one ...` directly (without a parent __init__) also wins.
import os as _os

_os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

# Bootstrap: allow `python process.py ...` from inside python/analysis/ even if
# the parent python/ folder hasn't been added to sys.path. Tests, notebooks,
# and CLI users converge on the same import shape this way.
_HERE = Path(__file__).resolve().parent
_PYTHON_DIR = _HERE.parent
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from analysis import backends, cache as _cache, reduce as _reduce  # noqa: E402
from analysis.collector import PsweepCollector  # noqa: E402
from analysis.registry import (  # noqa: E402
    AnalysisDescriptor,
    get_analysis,
    list_analyses,
    resolve_analysis_names,
)


# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="process.py",
        description=(
            "Map-reduce analysis pipeline for GPUParticles parameter sweeps."
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- map ---
    m = sub.add_parser("map", help="Run analyses across all (combo, seed) trajectories.")
    m.add_argument("sweep_dir")
    m.add_argument(
        "--analyses",
        required=True,
        help="Comma-separated analysis names, or 'all'.",
    )
    m.add_argument("--backend", choices=["local", "slurm"], default="local")
    m.add_argument(
        "--workers",
        default="auto",
        help="Local worker count (int or 'auto'). Default: auto.",
    )
    m.add_argument(
        "--rerun",
        choices=_cache.RERUN_POLICIES,
        default="skip",
    )
    m.add_argument(
        "--on-partial",
        choices=["skip", "trim", "error"],
        default="skip",
        help="What to do for trajectories with fewer than the expected num_frames.",
    )
    m.add_argument(
        "--only-missing",
        action="store_true",
        help="Limit task list to (combo, seed) pairs that don't have a complete cache.",
    )
    m.add_argument("--no-reduce", action="store_true")
    m.add_argument("--reduce-anyway", action="store_true")
    m.add_argument("--dry-run", action="store_true")
    # SLURM-only options:
    m.add_argument("--partition")
    m.add_argument("--time", dest="walltime", default="04:00:00")
    m.add_argument("--mem", default="4G")
    m.add_argument("--cpus", type=int, default=1)
    m.add_argument(
        "--no-submit",
        action="store_true",
        help="With --backend slurm: write task+slurm files but do not sbatch.",
    )

    # --- reduce ---
    r = sub.add_parser("reduce", help="Combine per-trajectory caches into processed.h5.")
    r.add_argument("sweep_dir")
    r.add_argument("--analyses", required=True)
    r.add_argument("--output", default=None)

    # --- one ---
    o = sub.add_parser(
        "one",
        help="Run analyses for a single (combo_idx, seed). Used by SLURM array tasks.",
    )
    o.add_argument("sweep_dir")
    o.add_argument("combo_idx", type=int)
    o.add_argument("seed", type=int)
    o.add_argument("--analyses", required=True)
    o.add_argument("--rerun", choices=_cache.RERUN_POLICIES, default="skip")
    o.add_argument("--on-partial", choices=["skip", "trim", "error"], default="skip")

    # --- status ---
    s = sub.add_parser("status", help="Print per-analysis completion table.")
    s.add_argument("sweep_dir")
    s.add_argument("--analyses", default="all")

    # --- clean ---
    c = sub.add_parser("clean", help="Delete cached analyses (use with care).")
    c.add_argument("sweep_dir")
    c.add_argument("--analyses", required=True)
    c.add_argument(
        "--stale-only",
        action="store_true",
        help="Only delete groups whose version/hash no longer match the registry.",
    )
    c.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt.",
    )

    return p


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def cmd_map(args: argparse.Namespace) -> int:
    coll = PsweepCollector(args.sweep_dir)
    analysis_specs = backends.parse_analyses_arg(args.analyses)
    code_sha = _git_sha(_HERE)

    only_pairs: Optional[list[tuple[int, int]]]
    if args.only_missing:
        only_pairs = _collect_incomplete_pairs(coll, analysis_specs)
        if not only_pairs:
            print("All (combo, seed) caches are complete — nothing to do.")
            return 0
    else:
        only_pairs = None

    tasks = backends.build_tasks(
        coll,
        analysis_specs=analysis_specs,
        rerun_policy=args.rerun,
        on_partial=args.on_partial,
        code_git_sha=code_sha,
        only_pairs=only_pairs,
    )
    print(
        f"Sweep:    {coll.sweep_name}\n"
        f"Analyses: {[n for n, _ in analysis_specs]}\n"
        f"Tasks:    {len(tasks)}  ({coll.n_combos} combos x {coll.num_seeds} seeds)"
        + (f"  [filtered to incomplete: {len(tasks)}]" if only_pairs is not None else "")
    )

    if args.dry_run:
        print("Dry run — no tasks executed.")
        return 0

    if args.backend == "local":
        return _run_local_map(coll, tasks, analysis_specs, args, code_sha)
    return _run_slurm_map(coll, tasks, args)


def _run_local_map(
    coll: PsweepCollector,
    tasks: list[backends.MapTask],
    analysis_specs: list[tuple[str, dict[str, Any]]],
    args: argparse.Namespace,
    code_sha: str,
) -> int:
    workers = (
        max(1, (os.cpu_count() or 1))
        if args.workers == "auto"
        else int(args.workers)
    )

    t0 = time.time()
    results: list[backends.MapResult] = backends.run_local(
        tasks,
        workers=workers,
        progress=_print_progress,
    )
    elapsed = time.time() - t0
    n_ok = sum(1 for r in results if r.ok)
    n_err = len(results) - n_ok
    print(
        f"\nMap done: {n_ok}/{len(results)} ok, {n_err} failed in {elapsed:.1f}s"
    )
    if n_err:
        for r in results[:200]:
            if r.ok:
                continue
            err = r.error or next(
                (v for v in r.statuses.values() if v.startswith("error")),
                "(unknown error)",
            )
            print(f"  - combo={r.combo_idx:04d} seed={r.seed}: {err}")
        if n_err > 200:
            print(f"  ... ({n_err - 200} more)")

    should_reduce = (not args.no_reduce) and (n_err == 0 or args.reduce_anyway)
    if not should_reduce:
        print(
            "Skipping reduce (errors present)."
            if n_err and not args.reduce_anyway
            else "Skipping reduce (--no-reduce)."
        )
        print(
            f"  To run later: process.py reduce {shlex.quote(str(coll.sweep_dir))} "
            f"--analyses {shlex.quote(args.analyses)}"
        )
        return 0 if n_err == 0 else 1

    out_path = _reduce.reduce_sweep(
        coll,
        analysis_specs=analysis_specs,
        code_git_sha=code_sha,
        reduce_args=" ".join(sys.argv),
    )
    print(f"Reduce wrote: {out_path}")
    return 0 if n_err == 0 else 1


def _run_slurm_map(
    coll: PsweepCollector,
    tasks: list[backends.MapTask],
    args: argparse.Namespace,
) -> int:
    if not args.partition:
        sys.exit("--partition is required with --backend slurm")
    process_py_path = Path(__file__).resolve()

    task_file, slurm_file = backends.write_slurm_array(
        coll,
        tasks,
        partition=args.partition,
        walltime=args.walltime,
        mem=args.mem,
        cpus=args.cpus,
        process_py_path=process_py_path,
    )
    print(f"Wrote {task_file}\nWrote {slurm_file}")
    if args.no_submit:
        print(f"\nNot submitted (--no-submit). Submit with: sbatch {slurm_file}")
    else:
        out = backends.submit_slurm(slurm_file)
        print(f"sbatch: {out}")
    print(
        f"\nWhen the array completes, run reduce:\n"
        f"  process.py reduce {shlex.quote(str(coll.sweep_dir))} "
        f"--analyses {shlex.quote(args.analyses)}"
    )
    return 0


def cmd_one(args: argparse.Namespace) -> int:
    coll = PsweepCollector(args.sweep_dir)
    analysis_specs = backends.parse_analyses_arg(args.analyses)
    task = backends.MapTask(
        sweep_dir=str(coll.sweep_dir),
        combo_idx=args.combo_idx,
        seed=args.seed,
        analyses=analysis_specs,
        rerun_policy=args.rerun,
        code_git_sha=_git_sha(_HERE),
        on_partial=args.on_partial,
    )
    res = backends.run_one_task(task)
    if res.error:
        print(f"FAIL combo={res.combo_idx:04d} seed={res.seed}: {res.error}")
        return 1
    for name, status in res.statuses.items():
        print(f"  {name}: {status}")
    return 0 if res.ok else 1


def cmd_reduce(args: argparse.Namespace) -> int:
    coll = PsweepCollector(args.sweep_dir)
    analysis_specs = backends.parse_analyses_arg(args.analyses)
    out_path = _reduce.reduce_sweep(
        coll,
        analysis_specs=analysis_specs,
        output_path=Path(args.output) if args.output else None,
        code_git_sha=_git_sha(_HERE),
        reduce_args=" ".join(sys.argv),
    )
    print(f"Reduce wrote: {out_path}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    coll = PsweepCollector(args.sweep_dir)
    names = (
        list_analyses() if args.analyses == "all" else resolve_analysis_names(args.analyses)
    )
    counters: dict[str, dict[str, int]] = {
        n: {"complete": 0, "stale": 0, "error": 0, "missing": 0} for n in names
    }
    for combo_idx, seed, _ in coll:
        cp = coll.cache_path(combo_idx, seed)
        for n in names:
            desc = get_analysis(n)
            st = _cache.cache_status(cp, n, desc.inputs_hash({}), desc.version)
            counters[n][st] += 1
    total = coll.n_combos * coll.num_seeds
    print(f"Sweep: {coll.sweep_name}  ({total} trajectories total)")
    print(f"  {'analysis':<24} {'complete':>10} {'stale':>10} {'error':>10} {'missing':>10}")
    for n, c in counters.items():
        print(
            f"  {n:<24} {c['complete']:>10} {c['stale']:>10} "
            f"{c['error']:>10} {c['missing']:>10}"
        )
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    coll = PsweepCollector(args.sweep_dir)
    names = resolve_analysis_names(args.analyses)
    if not args.yes:
        prompt = (
            f"About to delete cache groups {names} from caches under "
            f"{coll.sweep_dir}. Type 'yes' to proceed: "
        )
        try:
            if input(prompt).strip().lower() != "yes":
                print("Aborted.")
                return 1
        except EOFError:
            print("No tty; pass --yes to confirm.")
            return 1

    n_deleted = 0
    for combo_idx, seed, _ in coll:
        cp = coll.cache_path(combo_idx, seed)
        if not cp.exists():
            continue
        with h5py.File(cp, "a") as f:
            for n in names:
                desc = get_analysis(n)
                if n not in f:
                    continue
                if args.stale_only:
                    grp = f[n]
                    same_v = int(grp.attrs.get("analysis_version", -1)) == desc.version
                    same_h = (
                        _cache._attr_str(grp.attrs.get("inputs_hash", ""))
                        == desc.inputs_hash({})
                    )
                    if same_v and same_h and "error" not in grp.attrs:
                        continue
                del f[n]
                n_deleted += 1
    print(f"Deleted {n_deleted} cache group(s).")
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_progress(res: backends.MapResult, i: int, n: int) -> None:
    tag = "ok " if res.ok else "ERR"
    print(
        f"[{i:>5}/{n}] {tag} combo={res.combo_idx:04d} seed={res.seed} "
        + (res.error or "; ".join(f"{k}={v}" for k, v in res.statuses.items())),
        flush=True,
    )


def _git_sha(path: Path) -> str:
    """Best-effort short SHA of HEAD for `path`; empty string if not in a git repo."""
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip()
    except FileNotFoundError:
        return ""


def _collect_incomplete_pairs(
    coll: PsweepCollector,
    analysis_specs: list[tuple[str, dict[str, Any]]],
) -> list[tuple[int, int]]:
    """Return (combo, seed) pairs that need at least one analysis to (re)run."""
    out: list[tuple[int, int]] = []
    descs = [(get_analysis(n), kw) for n, kw in analysis_specs]
    for combo_idx, seed, _ in coll:
        cp = coll.cache_path(combo_idx, seed)
        for desc, kw in descs:
            st = _cache.cache_status(cp, desc.name, desc.inputs_hash(kw), desc.version)
            if st != "complete":
                out.append((combo_idx, seed))
                break
    return out


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    args = _make_parser().parse_args(argv)
    return {
        "map": cmd_map,
        "reduce": cmd_reduce,
        "one": cmd_one,
        "status": cmd_status,
        "clean": cmd_clean,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
