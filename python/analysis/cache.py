"""
cache.py — per-trajectory analysis cache (<traj>.cache.h5).

One sibling cache file per trajectory. Stores the result of each registered
analysis under /<analysis_name>/, with provenance attrs that let the reduce
step (and reruns of map) decide whether the cached entry is still valid.

Layout:
    /                       root
      @sweep_dir, @combo_idx, @seed
      @traj_path             relative to sweep_dir
      @traj_mtime            float seconds, for staleness detection
      @traj_num_frames
      @code_git_sha
      @bdtrajectory_schema_version
      @analyses_run          variable-length list[str]
    /<analysis_name>/
      @analysis_version
      @inputs_hash           short hash of analyze() kwargs
      @analysis_kwargs       JSON dump
      @timestamp             ISO 8601
      @error                 (only on failure; group has no datasets)
      /<dataset_1>, /<dataset_2>, ...

Idempotency:
  - skip:    cached version+hash match -> noop
  - stale:   mismatch -> rename existing group to <name>__stale_<ts>, rerun
  - force:   delete existing group, rerun
  - rebuild: nuke whole cache file, rerun all
"""

from __future__ import annotations

import datetime as _dt
import errno
import fcntl
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

from .registry import AnalysisDescriptor, get_analysis


BDTRAJECTORY_SCHEMA_VERSION = 1
RERUN_POLICIES = ("skip", "stale", "force", "rebuild")


# ---------------------------------------------------------------------------
# File-level locking (cross-process map jobs against the same cache)
# ---------------------------------------------------------------------------


@contextmanager
def _lock(cache_path: Path):
    """Advisory exclusive lock on <cache>.lock — survives the cache being unlinked.

    Uses fcntl.flock so concurrent map workers on the same trajectory serialise
    cleanly. flock is no-op on most cluster filesystems but reliable on a
    single host, which is the common collision case.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


# ---------------------------------------------------------------------------
# Cache status enumeration (consumed by reduce / status)
# ---------------------------------------------------------------------------


def cache_status(
    cache_path: Path,
    analysis_name: str,
    expected_inputs_hash: str,
    expected_version: int,
) -> str:
    """Return one of: 'missing', 'stale', 'error', 'complete'.

    'missing' — no cache file, or analysis group not present.
    'error'   — analysis group present with @error attr.
    'stale'   — version or inputs hash mismatch.
    'complete'— ok.
    """
    if not cache_path.exists():
        return "missing"
    try:
        with h5py.File(cache_path, "r") as f:
            if analysis_name not in f:
                return "missing"
            grp = f[analysis_name]
            if "error" in grp.attrs:
                return "error"
            v = int(grp.attrs.get("analysis_version", -1))
            h = _attr_str(grp.attrs.get("inputs_hash", ""))
            if v != expected_version or h != expected_inputs_hash:
                return "stale"
            return "complete"
    except (OSError, KeyError):
        return "missing"


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_root_attrs(
    f: h5py.File,
    *,
    sweep_dir: Path,
    combo_idx: int,
    seed: int,
    traj_path: Path,
    traj_mtime: float,
    traj_num_frames: int,
    code_git_sha: str,
) -> None:
    """Set/refresh root-level provenance attrs."""
    a = f.attrs
    a["sweep_dir"] = str(sweep_dir)
    a["combo_idx"] = int(combo_idx)
    a["seed"] = int(seed)
    a["traj_path"] = str(traj_path.relative_to(sweep_dir))
    a["traj_mtime"] = float(traj_mtime)
    a["traj_num_frames"] = int(traj_num_frames)
    a["code_git_sha"] = code_git_sha
    a["bdtrajectory_schema_version"] = BDTRAJECTORY_SCHEMA_VERSION


def _write_analysis_group(
    f: h5py.File,
    desc: AnalysisDescriptor,
    kwargs: dict[str, Any],
    inputs_hash: str,
    result: Optional[dict[str, Any]],
    error: Optional[str],
    rerun_policy: str,
) -> None:
    """Write one analysis group.

    On rerun_policy=='stale', if a group already exists with mismatched
    version/hash, it is renamed to <name>__stale_<ts> first.
    On rerun_policy in ('force', 'rebuild'), it is deleted first.
    """
    name = desc.name

    if name in f:
        if rerun_policy == "stale":
            stale_name = f"{name}__stale_{int(time.time())}"
            f.move(name, stale_name)
        else:  # 'force', 'rebuild', or 'skip' (caller already decided to write)
            del f[name]

    grp = f.create_group(name)
    grp.attrs["analysis_version"] = int(desc.version)
    grp.attrs["inputs_hash"] = inputs_hash
    grp.attrs["analysis_kwargs"] = json.dumps(kwargs, sort_keys=True, default=str)
    grp.attrs["timestamp"] = _now_iso()

    if error is not None:
        grp.attrs["error"] = error
        return

    assert result is not None
    for key, val in result.items():
        arr = np.asarray(val)
        # gzip is cheap; helps a lot for repeated patterns (e.g., constant time
        # axes). Auto-chunk leaves chunk shape to h5py.
        if arr.size >= 64 and arr.ndim >= 1:
            grp.create_dataset(key, data=arr, compression="gzip", compression_opts=4)
        else:
            grp.create_dataset(key, data=arr)


def _atomic_replace(src: Path, dst: Path) -> None:
    """Atomic POSIX rename. Both paths must be on the same filesystem."""
    os.replace(src, dst)


def run_and_write(
    traj,
    cache_path: Path,
    sweep_dir: Path,
    combo_idx: int,
    seed: int,
    analyses: list[tuple[str, dict[str, Any]]],
    *,
    code_git_sha: str = "",
    rerun_policy: str = "skip",
) -> dict[str, str]:
    """
    Execute `analyses` on `traj` and merge their outputs into `cache_path`.

    Each entry of `analyses` is (analysis_name, kwargs). The function honors
    `rerun_policy` per-analysis: a 'skip' policy lets us mix already-cached
    analyses with new ones in the same call.

    Returns a dict {analysis_name: status} with status in
    {"skipped", "ok", "error: <msg>"}.
    """
    if rerun_policy not in RERUN_POLICIES:
        raise ValueError(
            f"rerun_policy={rerun_policy!r} not in {RERUN_POLICIES}"
        )

    cache_path = Path(cache_path)
    sweep_dir = Path(sweep_dir)
    traj_path = Path(traj._path)
    traj_mtime = traj_path.stat().st_mtime

    # 'rebuild' nukes the cache file outright before doing anything else.
    if rerun_policy == "rebuild" and cache_path.exists():
        cache_path.unlink()

    statuses: dict[str, str] = {}

    with _lock(cache_path):
        # Write into a temp file alongside the final path, then atomic replace.
        # If the cache already exists, copy its content into the temp first —
        # this preserves cache groups for analyses we are skipping this run.
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        # Always remove a stale .tmp before proceeding — h5py opens new files
        # with O_EXCL and will fail with FileExistsError if one is left over
        # from a process killed between _copy_h5 and _atomic_replace.
        if tmp_path.exists():
            tmp_path.unlink()
        if cache_path.exists():
            # Copy the existing cache file by re-opening and rewriting via
            # h5py — bytewise copy would also work, but using h5py keeps the
            # logic uniform with the create-from-scratch case.
            try:
                _copy_h5(cache_path, tmp_path)
            except OSError as e:
                # Corrupt cache (typically a process killed mid-write). Nuke
                # it and start fresh — analyses requested this run will get
                # recomputed; ones we'd have skipped will simply rerun.
                print(
                    f"WARNING: corrupt cache file, deleting and rebuilding: "
                    f"{cache_path} ({e})",
                    file=sys.stderr,
                )
                cache_path.unlink()

        try:
            with h5py.File(tmp_path, "a") as f:
                _ensure_root_attrs(
                    f,
                    sweep_dir=sweep_dir,
                    combo_idx=combo_idx,
                    seed=seed,
                    traj_path=traj_path,
                    traj_mtime=traj_mtime,
                    traj_num_frames=int(traj.num_frames),
                    code_git_sha=code_git_sha,
                )

                for analysis_name, kwargs in analyses:
                    desc = get_analysis(analysis_name)
                    inputs_hash = desc.inputs_hash(kwargs)

                    # Decide whether to skip based on existing group's metadata.
                    if analysis_name in f and rerun_policy != "force":
                        existing = f[analysis_name]
                        same_v = (
                            int(existing.attrs.get("analysis_version", -1))
                            == desc.version
                        )
                        same_h = (
                            _attr_str(existing.attrs.get("inputs_hash", ""))
                            == inputs_hash
                        )
                        if same_v and same_h and "error" not in existing.attrs:
                            statuses[analysis_name] = "skipped"
                            continue

                    # Pre-check required datasets so analyses fail cleanly.
                    missing = _missing_requirements(traj, desc.requires)
                    if missing:
                        _write_analysis_group(
                            f,
                            desc,
                            kwargs,
                            inputs_hash,
                            result=None,
                            error=f"trajectory missing required datasets: {missing}",
                            rerun_policy=rerun_policy,
                        )
                        statuses[analysis_name] = (
                            f"error: missing datasets {missing}"
                        )
                        continue

                    try:
                        result = desc(traj, **kwargs)
                    except Exception as e:
                        _write_analysis_group(
                            f,
                            desc,
                            kwargs,
                            inputs_hash,
                            result=None,
                            error=f"{type(e).__name__}: {e}",
                            rerun_policy=rerun_policy,
                        )
                        statuses[analysis_name] = f"error: {type(e).__name__}: {e}"
                        continue

                    _write_analysis_group(
                        f,
                        desc,
                        kwargs,
                        inputs_hash,
                        result=result,
                        error=None,
                        rerun_policy=rerun_policy,
                    )
                    statuses[analysis_name] = "ok"

                # Update analyses_run tally.
                f.attrs["analyses_run"] = sorted(
                    set(_attr_strlist(f.attrs.get("analyses_run", [])))
                    | set(analyses_named(analyses))
                )

            _atomic_replace(tmp_path, cache_path)
        except Exception:
            # Don't leave a half-written tmp behind on unexpected exit.
            try:
                tmp_path.unlink()
            except OSError as cleanup_err:
                if cleanup_err.errno != errno.ENOENT:
                    raise
            raise

    return statuses


# ---------------------------------------------------------------------------
# Read path (used by reduce + status)
# ---------------------------------------------------------------------------


def cleanup_caches(
    cache_paths,
    analysis_names: list[str],
) -> tuple[int, int, list[str]]:
    """
    Delete the named analysis groups from each cache file.

    If a cache file ends up with no remaining analysis groups, delete the
    file itself (and its sibling .lock). Used by `process.py reduce
    --cleanup` after a successful merge into the per-analysis `<name>.h5`.

    Returns (n_groups_deleted, n_files_deleted, warnings).
    """
    n_groups = 0
    n_files = 0
    warnings_: list[str] = []
    name_set = set(analysis_names)

    for cp in cache_paths:
        cp = Path(cp)
        if not cp.exists():
            continue
        try:
            with h5py.File(cp, "a") as f:
                for name in analysis_names:
                    if name in f:
                        del f[name]
                        n_groups += 1
                # Trim analyses_run attr so it stays in sync with what's left.
                if "analyses_run" in f.attrs:
                    remaining = [
                        n for n in _attr_strlist(f.attrs["analyses_run"])
                        if n not in name_set
                    ]
                    if remaining:
                        f.attrs["analyses_run"] = remaining
                    else:
                        del f.attrs["analyses_run"]
                empty = len(list(f.keys())) == 0
            if empty:
                cp.unlink()
                lock_path = cp.with_suffix(cp.suffix + ".lock")
                if lock_path.exists():
                    lock_path.unlink()
                n_files += 1
        except OSError as e:
            warnings_.append(f"{cp}: {e}")
    return n_groups, n_files, warnings_


def read_analysis(
    cache_path: Path, analysis_name: str
) -> Optional[dict[str, np.ndarray]]:
    """Read all datasets under /<analysis_name>/ as a dict of ndarrays.

    Returns None when the analysis group is missing, contains an @error, or
    when the cache file itself is unreadable (e.g. truncated by a killed job).
    """
    if not cache_path.exists():
        return None
    try:
        with h5py.File(cache_path, "r") as f:
            if analysis_name not in f:
                return None
            grp = f[analysis_name]
            if "error" in grp.attrs:
                return None
            return {k: grp[k][()] for k in grp.keys()}
    except OSError as e:
        # Truncated/corrupt cache file — typically a process killed mid-write.
        # Treat as missing so reduce can proceed; user can re-run process for
        # the affected seed.
        print(
            f"WARNING: corrupt cache file, skipping: {cache_path} ({e})",
            file=sys.stderr,
        )
        return None


def read_analysis_attrs(
    cache_path: Path, analysis_name: str
) -> Optional[dict[str, Any]]:
    if not cache_path.exists():
        return None
    try:
        with h5py.File(cache_path, "r") as f:
            if analysis_name not in f:
                return None
            grp = f[analysis_name]
            return {k: _attr_value(v) for k, v in grp.attrs.items()}
    except OSError as e:
        print(
            f"WARNING: corrupt cache file, skipping: {cache_path} ({e})",
            file=sys.stderr,
        )
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def analyses_named(items: list[tuple[str, dict[str, Any]]]) -> list[str]:
    return [name for name, _ in items]


def _missing_requirements(traj, requires: tuple[str, ...]) -> list[str]:
    """Check that each required dataset has a non-None frame-0 value."""
    missing = []
    for r in requires:
        if r == "positions":
            continue  # always present
        accessor = getattr(traj, r, None)
        if accessor is None or accessor(0) is None:
            missing.append(r)
    return missing


def _copy_h5(src: Path, dst: Path) -> None:
    """Bytewise file copy (atomic-replace prep). Shutil.copy is fine here."""
    import shutil

    shutil.copyfile(src, dst)


def _attr_str(v) -> str:
    if isinstance(v, bytes):
        return v.decode()
    return str(v)


def _attr_value(v):
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray) and v.dtype.kind in ("U", "O", "S"):
        return [
            x.decode() if isinstance(x, bytes) else str(x)
            for x in v.tolist()
        ]
    if isinstance(v, np.ndarray) and v.ndim == 0:
        return v.item()
    return v


def _attr_strlist(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, (bytes, str)):
        return [_attr_str(v)]
    return [_attr_str(x) for x in np.asarray(v).ravel().tolist()]
