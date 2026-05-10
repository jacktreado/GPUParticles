"""
backends.py — execute the map step locally (multiprocessing) or via SLURM.

The local backend uses a `spawn` Pool with imap_unordered, sentinel-returning
workers, and live progress. The SLURM backend writes a task file (one
`process.py one ...` invocation per line) and a SLURM array script, then
optionally sbatchs it.
"""

from __future__ import annotations

import multiprocessing as _mp
import os
import shlex
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from . import cache as _cache
from .collector import PsweepCollector
from .registry import resolve_analysis_names


# ---------------------------------------------------------------------------
# Task description
# ---------------------------------------------------------------------------


@dataclass
class MapTask:
    sweep_dir: str
    combo_idx: int
    seed: int
    analyses: list[tuple[str, dict[str, Any]]]
    rerun_policy: str
    code_git_sha: str
    on_partial: str = "skip"           # "skip" | "trim" | "error"
    expected_num_frames: Optional[int] = None


@dataclass
class MapResult:
    combo_idx: int
    seed: int
    statuses: dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and not any(
            v.startswith("error") for v in self.statuses.values()
        )


# ---------------------------------------------------------------------------
# Worker (used by both local pool and the `process.py one` SLURM entry)
# ---------------------------------------------------------------------------


def run_one_task(task: MapTask) -> MapResult:
    """Execute a single (combo_idx, seed) map task. Never raises."""
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    try:
        coll = PsweepCollector(task.sweep_dir)
        traj_path = coll.traj_path(task.combo_idx, task.seed)
        if not traj_path.exists():
            return MapResult(
                task.combo_idx, task.seed, error=f"trajectory missing: {traj_path}"
            )

        # Open the trajectory inside the worker — never inherited across fork.
        try:
            traj = coll.open(task.combo_idx, task.seed)
        except (OSError, KeyError, RuntimeError) as e:
            return MapResult(
                task.combo_idx,
                task.seed,
                error=f"trajectory open failed ({type(e).__name__}): {e}",
            )

        try:
            expected = (
                task.expected_num_frames
                if task.expected_num_frames is not None
                else coll.expected_num_frames(task.combo_idx, task.seed)
            )
            if traj.num_frames < expected:
                if task.on_partial == "error":
                    return MapResult(
                        task.combo_idx,
                        task.seed,
                        error=(
                            f"partial trajectory: {traj.num_frames}/{expected} frames"
                        ),
                    )
                if task.on_partial == "skip":
                    return MapResult(
                        task.combo_idx,
                        task.seed,
                        error=(
                            f"skipped partial: {traj.num_frames}/{expected} frames"
                        ),
                    )
                # 'trim': fall through and analyze whatever frames we have.

            statuses = _cache.run_and_write(
                traj,
                cache_path=coll.cache_path(task.combo_idx, task.seed),
                sweep_dir=coll.sweep_dir,
                combo_idx=task.combo_idx,
                seed=task.seed,
                analyses=task.analyses,
                code_git_sha=task.code_git_sha,
                rerun_policy=task.rerun_policy,
            )
            return MapResult(task.combo_idx, task.seed, statuses=statuses)
        finally:
            traj.close()
    except Exception as e:  # last-resort guard so the pool never sees an exception
        return MapResult(
            task.combo_idx,
            task.seed,
            error=f"unhandled {type(e).__name__}: {e}",
        )


# ---------------------------------------------------------------------------
# Local backend
# ---------------------------------------------------------------------------


def run_local(
    tasks: list[MapTask],
    *,
    workers: int,
    progress: Callable[[MapResult, int, int], None] | None = None,
) -> list[MapResult]:
    """Run `tasks` in parallel using a spawn Pool. Returns all results."""
    if not tasks:
        return []
    workers = max(1, min(workers, len(tasks)))
    ctx = _mp.get_context("spawn")
    chunksize = max(1, len(tasks) // (workers * 8))
    results: list[MapResult] = []
    n = len(tasks)
    with ctx.Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_one_task, tasks, chunksize)):
            results.append(res)
            if progress is not None:
                progress(res, i + 1, n)
    return results


# ---------------------------------------------------------------------------
# SLURM array backend
# ---------------------------------------------------------------------------


def write_slurm_array(
    coll: PsweepCollector,
    tasks: list[MapTask],
    *,
    partition: str,
    walltime: str,
    mem: str,
    cpus: int,
    process_py_path: Path,
    extra_env: Optional[dict[str, str]] = None,
) -> tuple[Path, Path]:
    """
    Write a task file + .slurm array script under <sweep>/analysis/slurm/.

    Returns (task_file, slurm_script).
    """
    if not tasks:
        raise ValueError("no SLURM tasks to write")

    slurm_dir = coll.sweep_dir / "analysis" / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    job_name = f"{coll.sweep_name}_proc_{stamp}"
    task_file = slurm_dir / f"{job_name}.task"
    slurm_file = slurm_dir / f"{job_name}.slurm"
    log_dir = slurm_dir / job_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Each task line is a fully self-contained `process.py one` invocation —
    # no shared state, so SLURM can re-run any task on retry.
    with open(task_file, "w") as f:
        for t in tasks:
            analyses_arg = ",".join(name for name, _ in t.analyses)
            cmd = (
                f"{shlex.quote(sys.executable)} {shlex.quote(str(process_py_path))} "
                f"one {shlex.quote(str(coll.sweep_dir))} "
                f"{t.combo_idx} {t.seed} "
                f"--analyses {shlex.quote(analyses_arg)} "
                f"--rerun {shlex.quote(t.rerun_policy)} "
                f"--on-partial {shlex.quote(t.on_partial)}"
            )
            f.write(cmd + "\n")

    env_lines = "".join(
        f"export {k}={shlex.quote(v)}\n"
        for k, v in (extra_env or {}).items()
    )

    script = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --partition={partition}
        #SBATCH --time={walltime}
        #SBATCH --mem={mem}
        #SBATCH --cpus-per-task={cpus}
        #SBATCH --ntasks=1
        #SBATCH --job-name={job_name}
        #SBATCH --array=1-{len(tasks)}
        #SBATCH --output={log_dir}/task-%a.out

        set -eo pipefail
        export HDF5_USE_FILE_LOCKING=FALSE
        {env_lines}
        cmd=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "{task_file}")
        echo "+ $cmd"
        eval "$cmd"
    """)
    with open(slurm_file, "w") as f:
        f.write(script)

    return task_file, slurm_file


class SbatchError(RuntimeError):
    """sbatch returned non-zero. Carries the rejected script and sbatch's stderr."""

    def __init__(self, slurm_file: Path, returncode: int, stdout: str, stderr: str):
        self.slurm_file = slurm_file
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        msg = (
            f"sbatch rejected {slurm_file} (exit {returncode}).\n"
            f"--- sbatch stderr ---\n{stderr.rstrip() or '(empty)'}\n"
            f"--- sbatch stdout ---\n{stdout.rstrip() or '(empty)'}\n"
            "Inspect the file directly and try `sbatch <file>` by hand to see "
            "the same error in its native context."
        )
        super().__init__(msg)


def submit_slurm(slurm_file: Path) -> str:
    """sbatch the script; return the captured stdout (contains job id).

    On a non-zero sbatch exit, raises SbatchError carrying both stderr and
    stdout — sbatch reports the *reason* for rejection on stderr (bad
    partition, missing account, malformed walltime, array size limit, etc.),
    so we surface it instead of swallowing it inside CalledProcessError.
    """
    out = subprocess.run(
        ["sbatch", str(slurm_file)],
        check=False,
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise SbatchError(slurm_file, out.returncode, out.stdout, out.stderr)
    return out.stdout.strip()


# ---------------------------------------------------------------------------
# Helpers shared with the CLI
# ---------------------------------------------------------------------------


def build_tasks(
    coll: PsweepCollector,
    *,
    analysis_specs: list[tuple[str, dict[str, Any]]],
    rerun_policy: str,
    on_partial: str,
    code_git_sha: str,
    only_pairs: Optional[Iterable[tuple[int, int]]] = None,
) -> list[MapTask]:
    pairs: Iterable[tuple[int, int]]
    if only_pairs is None:
        pairs = ((c, s) for c, s, _ in coll)
    else:
        pairs = only_pairs

    tasks: list[MapTask] = []
    for combo_idx, seed in pairs:
        tasks.append(
            MapTask(
                sweep_dir=str(coll.sweep_dir),
                combo_idx=combo_idx,
                seed=seed,
                analyses=list(analysis_specs),
                rerun_policy=rerun_policy,
                code_git_sha=code_git_sha,
                on_partial=on_partial,
            )
        )
    return tasks


def parse_analyses_arg(spec: str) -> list[tuple[str, dict[str, Any]]]:
    """Resolve a CLI --analyses string into [(name, {}), ...].

    Per-analysis kwargs are not parsed from the CLI in v1; pass them by
    editing process.py or by calling the API directly. The framework still
    tracks an inputs_hash so cache invalidation works.
    """
    names = resolve_analysis_names(spec)
    return [(n, {}) for n in names]
