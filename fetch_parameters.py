#!/usr/bin/env python3
"""
fetch_parameters.py — extract bare simulation parameters from a psweep's .h5
files, entirely off the remote cluster, into (i, j) grids + flat instance
arrays. Nothing is copied down except the final result.

Usage:
    ./fetch_parameters.py <sweep_name> [options]

A single SSH connection runs a small Python (h5py) script on the remote host
that reads the sweep's manifests (log.json, psweep_map.json, psweep_info.json)
and every instance's simulation .h5 attributes directly off the cluster's
filesystem, and prints one JSON blob back. Locally, that JSON feeds
PsweepCollector (for grid_shape / combo_idx -> (i, j)) and the attribute
assembly. The only thing written to local disk is the resulting
bare_parameters.pkl.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_PYTHON_DIR = Path(__file__).resolve().parent / "python" / "analysis"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from collector import PsweepCollector  # noqa: E402

# Executed on the remote host via `ssh ... python3 -` (piped over stdin).
# Reads the three manifests plus every instance's .h5 attrs directly off the
# cluster's filesystem and prints one JSON blob to stdout; per-combo progress
# goes to stderr so it doesn't corrupt the JSON. __SWEEP_NAME__ / __REMOTE_BASE_DIR__
# are substituted locally (as JSON string literals, which are also valid
# Python string literals) before sending.
_REMOTE_SCRIPT = r"""
import json, os, glob, sys

sweep_name = __SWEEP_NAME__
remote_base_dir = __REMOTE_BASE_DIR__
CACHE_SUFFIX = ".cache.h5"

sweep_dir = os.path.join(os.path.expanduser(remote_base_dir), sweep_name)

def _read_json(name):
    with open(os.path.join(sweep_dir, name)) as f:
        return json.load(f)

log = _read_json("log.json")
pmap = _read_json("psweep_map.json")
pinfo = _read_json("psweep_info.json")
n_combos = int(log["n_combos"])

def _jsonable(v):
    if hasattr(v, "item"):
        v = v.item()
    return v

def _find_h5(instance_dir):
    if not os.path.isdir(instance_dir):
        return None
    candidates = [
        p for p in glob.glob(os.path.join(instance_dir, "*_seed*.h5"))
        if not p.endswith(CACHE_SUFFIX)
    ]
    if not candidates:
        return None
    def _seed_of(p):
        stem = os.path.splitext(os.path.basename(p))[0]
        return int(stem.rsplit("_seed", 1)[1])
    return min(candidates, key=_seed_of)

import h5py

attrs_by_combo = {}
for combo_idx in range(n_combos):
    instance_dir = os.path.join(sweep_dir, "{0}_{1:04d}".format(sweep_name, combo_idx))
    h5_path = _find_h5(instance_dir)
    if h5_path is None:
        attrs_by_combo[str(combo_idx)] = None
        print("[{0}/{1}] {2}: no simulation .h5 found, skipping".format(
            combo_idx + 1, n_combos, os.path.basename(instance_dir)), file=sys.stderr)
        continue
    with h5py.File(h5_path, "r") as f:
        attrs_by_combo[str(combo_idx)] = {k: _jsonable(v) for k, v in f.attrs.items()}
    print("[{0}/{1}] {2}: read {3} attrs".format(
        combo_idx + 1, n_combos, os.path.basename(h5_path), len(attrs_by_combo[str(combo_idx)])),
        file=sys.stderr)

print(json.dumps({"log": log, "psweep_map": pmap, "psweep_info": pinfo, "attrs": attrs_by_combo}))
"""


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Extract per-instance bare simulation parameters from a psweep's "
            ".h5 files, read directly off the remote cluster, into (i, j) "
            "grids + flat instance arrays."
        )
    )
    p.add_argument("sweep_name", help="Name of the psweep (e.g. psweep_2026-09-24_wca_N500_corr_comp)")
    p.add_argument(
        "--local-output-dir", default="./output",
        help="Local output root, matching fetch_h5_files.sh (default: ./output)",
    )
    p.add_argument("--remote-user", default="treado", help="Remote SSH user (default: treado)")
    p.add_argument("--remote-host", default="vesta", help="Remote SSH host (default: vesta)")
    p.add_argument(
        "--remote-base-dir", default="~/data/GPUParticles",
        help="Remote base directory containing the sweep (default: ~/data/GPUParticles)",
    )
    p.add_argument(
        "-o", "--output", default=None,
        help="Output pickle path (default: <local-output-dir>/<sweep_name>/bare_parameters.pkl)",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Print status updates while reading")
    return p


def _fetch_everything_over_ssh(
    user: str, host: str, remote_base_dir: str, sweep_name: str, verbose: bool
) -> tuple[dict, dict, dict, dict]:
    """Run the remote h5py script over one SSH connection.

    Returns (log, psweep_map, psweep_info, attrs_by_combo). attrs_by_combo maps
    the string combo_idx to that instance's .h5 attrs dict, or None if no
    simulation .h5 was found for it. Nothing from this call is ever written to
    local disk directly by the caller.
    """
    script = (
        _REMOTE_SCRIPT
        .replace("__SWEEP_NAME__", json.dumps(sweep_name))
        .replace("__REMOTE_BASE_DIR__", json.dumps(remote_base_dir))
    )
    if verbose:
        print(f"[ssh] running remote h5py script on {user}@{host} for sweep {sweep_name!r} ...")
    result = subprocess.run(
        ["ssh", f"{user}@{host}", "python3", "-"],
        input=script, capture_output=True, text=True,
    )
    if verbose and result.stderr:
        for line in result.stderr.splitlines():
            print(f"  [remote] {line}")
    if result.returncode != 0:
        raise RuntimeError(
            f"Remote script failed on {user}@{host} (exit {result.returncode}): "
            f"{result.stderr.strip()}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Could not parse remote script output as JSON ({e}). "
            f"stdout was: {result.stdout[:500]!r}"
        ) from e
    return payload["log"], payload["psweep_map"], payload["psweep_info"], payload["attrs"]


def _collector_from_manifests(log: dict, pmap: dict, pinfo: dict) -> PsweepCollector:
    """Build a PsweepCollector purely from in-memory manifest dicts.

    PsweepCollector.__init__ only ever reads the three manifest files from the
    directory it's given, so writing them into a throwaway temp dir (deleted
    before this function returns) is safe and keeps nothing on local disk.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "log.json").write_text(json.dumps(log))
        (tmp_path / "psweep_map.json").write_text(json.dumps(pmap))
        (tmp_path / "psweep_info.json").write_text(json.dumps(pinfo))
        coll = PsweepCollector(tmp_path)
        _ = (coll.n_combos, coll.grid_shape, coll.loop_vars, coll.axes, coll.combo_indices_grid())
    return coll


def _assemble_bare_parameters(attrs_by_combo: dict, coll: PsweepCollector) -> dict:
    per_instance = [attrs_by_combo.get(str(i)) for i in range(coll.n_combos)]

    attr_names: set[str] = set()
    for attrs in per_instance:
        if attrs is not None:
            attr_names.update(attrs.keys())

    ci_grid = coll.combo_indices_grid()
    bare_parameters: dict = {}
    for name in sorted(attr_names):
        raw_values = [attrs.get(name) if attrs is not None else None for attrs in per_instance]
        all_numeric = all(
            v is None or isinstance(v, (int, float, np.integer, np.floating))
            for v in raw_values
        )
        if all_numeric:
            instance_arr = np.array(
                [np.nan if v is None else float(v) for v in raw_values], dtype=np.float64
            )
        else:
            instance_arr = np.array(raw_values, dtype=object)

        grid_arr = instance_arr[ci_grid]
        bare_parameters[name] = {"grid": grid_arr, "instance": instance_arr}

    bare_parameters["_meta"] = {
        "loop_vars": coll.loop_vars,
        "axes": coll.axes,
        "grid_shape": coll.grid_shape,
        "n_combos": coll.n_combos,
    }
    return bare_parameters


def main(argv: list[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)

    try:
        log, pmap, pinfo, attrs_by_combo = _fetch_everything_over_ssh(
            args.remote_user, args.remote_host, args.remote_base_dir, args.sweep_name, args.verbose
        )
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    coll = _collector_from_manifests(log, pmap, pinfo)
    if args.verbose:
        print(f"[grid] loop_vars={coll.loop_vars}, grid_shape={coll.grid_shape}, n_combos={coll.n_combos}")

    bare_parameters = _assemble_bare_parameters(attrs_by_combo, coll)

    output_path = (
        Path(args.output) if args.output
        else Path(args.local_output_dir) / args.sweep_name / "bare_parameters.pkl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(bare_parameters, f)

    n_missing = sum(1 for v in attrs_by_combo.values() if v is None)
    n_attrs = len(bare_parameters) - 1  # exclude _meta
    print(
        f"Wrote {output_path} "
        f"(grid_shape={coll.grid_shape}, {n_attrs} parameters, "
        f"{coll.n_combos - n_missing}/{coll.n_combos} instances found on cluster)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
