"""End-to-end test of the analysis pipeline against the mini_sweep fixture."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

# Imports from the package live here so the fixture's path setup runs first.
from analysis import (
    PsweepCollector,
    ProcessedSweep,
    list_analyses,
    register_analysis,
    get_analysis,
)
from analysis import backends, cache as _cache, reduce as _reduce


# ---------------------------------------------------------------------------
# PsweepCollector
# ---------------------------------------------------------------------------


def test_collector_basic(mini_sweep):
    coll = PsweepCollector(mini_sweep)
    assert coll.loop_vars == ["phi"]
    assert coll.n_combos == 3
    assert coll.num_seeds == 2
    assert coll.seeds == [0, 1]
    np.testing.assert_allclose(coll.axes["phi"], np.linspace(0.1, 0.5, 3))
    assert coll.grid_shape == (3,)


def test_collector_iter_and_paths(mini_sweep):
    coll = PsweepCollector(mini_sweep)
    pairs = [(c, s) for c, s, _ in coll]
    assert pairs == [(c, s) for c in range(3) for s in (0, 1)]
    for combo_idx, seed, path in coll:
        assert path.exists()
        assert path == coll.traj_path(combo_idx, seed)


def test_collector_filter_by_axis_idx(mini_sweep):
    coll = PsweepCollector(mini_sweep)
    pairs = [(c, s) for c, s, _ in coll.filter(phi_idx=2)]
    assert pairs == [(2, 0), (2, 1)]


def test_collector_expected_num_frames(mini_sweep):
    coll = PsweepCollector(mini_sweep)
    assert coll.expected_num_frames(0, 0) == 5  # 0..t_end = 4*output_dt inclusive


def test_collector_assemble_grid_per_combo(mini_sweep):
    coll = PsweepCollector(mini_sweep)
    values = {0: np.array([1.0]), 1: np.array([2.0]), 2: np.array([3.0])}
    grid = coll.assemble_grid_per_combo(values)
    assert grid.shape == (3, 1)
    np.testing.assert_allclose(grid[:, 0], [1.0, 2.0, 3.0])


def test_collector_assemble_grid_fills_missing_with_nan(mini_sweep):
    coll = PsweepCollector(mini_sweep)
    values = {0: np.array([1.0]), 2: np.array([3.0])}  # combo 1 missing
    grid = coll.assemble_grid_per_combo(values)
    assert np.isnan(grid[1, 0])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_has_builtins():
    names = list_analyses()
    assert "msd" in names
    assert "force_orientation" in names


def test_inputs_hash_changes_with_kwargs():
    desc = get_analysis("msd")
    h_default = desc.inputs_hash({})
    h_other = desc.inputs_hash({"reference_frame": 1})
    assert h_default != h_other


# ---------------------------------------------------------------------------
# Map step (local backend)
# ---------------------------------------------------------------------------


def _run_map(coll, names):
    specs = [(n, {}) for n in names]
    tasks = backends.build_tasks(
        coll,
        analysis_specs=specs,
        rerun_policy="skip",
        on_partial="skip",
        code_git_sha="testsha",
    )
    # Run sequentially to avoid spawn overhead in the test (correctness only).
    return [backends.run_one_task(t) for t in tasks]


def test_map_writes_caches(mini_sweep):
    coll = PsweepCollector(mini_sweep)
    results = _run_map(coll, ["msd", "force_orientation"])
    assert all(r.ok for r in results), [r for r in results if not r.ok]
    for combo_idx, seed, _ in coll:
        cp = coll.cache_path(combo_idx, seed)
        assert cp.exists()
        with h5py.File(cp, "r") as f:
            assert "msd" in f
            assert "force_orientation" in f
            assert "msd" in f["msd"]
            assert "lag_time" in f["msd"]
            assert f["msd"].attrs["analysis_version"] == 1


def test_map_skips_when_cache_is_complete(mini_sweep):
    coll = PsweepCollector(mini_sweep)
    _run_map(coll, ["msd"])
    results = _run_map(coll, ["msd"])  # second pass
    assert all(r.statuses["msd"] == "skipped" for r in results)


def test_map_records_error_on_corrupted_traj(mini_sweep):
    coll = PsweepCollector(mini_sweep)
    # Truncate one trajectory file.
    bad = coll.traj_path(1, 0)
    with open(bad, "wb") as f:
        f.write(b"junk")
    results = _run_map(coll, ["msd"])
    bad_res = next(r for r in results if (r.combo_idx, r.seed) == (1, 0))
    assert not bad_res.ok
    # Other trajectories should still succeed.
    others = [r for r in results if (r.combo_idx, r.seed) != (1, 0)]
    assert all(r.ok for r in others)


def test_map_only_missing(mini_sweep):
    coll = PsweepCollector(mini_sweep)
    _run_map(coll, ["msd"])
    # Delete one cache file, then the --only-missing logic should pick exactly that pair.
    target = coll.cache_path(2, 1)
    target.unlink()
    descs = [(get_analysis("msd"), {})]
    incomplete = []
    for combo_idx, seed, _ in coll:
        for desc, kw in descs:
            st = _cache.cache_status(coll.cache_path(combo_idx, seed), desc.name, desc.inputs_hash(kw), desc.version)
            if st != "complete":
                incomplete.append((combo_idx, seed))
                break
    assert incomplete == [(2, 1)]


# ---------------------------------------------------------------------------
# Reduce step
# ---------------------------------------------------------------------------


def test_reduce_writes_processed(mini_sweep):
    coll = PsweepCollector(mini_sweep)
    _run_map(coll, ["msd", "force_orientation"])
    out_dir = _reduce.reduce_sweep(
        coll,
        analysis_specs=[("msd", {}), ("force_orientation", {})],
        code_git_sha="testsha",
        reduce_args="test",
    )
    assert out_dir.is_dir()

    with h5py.File(out_dir / "msd.h5", "r") as f:
        assert "axes" in f
        assert "msd" in f
        # Shape: grid_shape + (num_seeds,) + tail_shape ; here T=5 frames.
        assert f["msd/grid"].shape == (3, 2, 5)
        assert f["msd/mean"].shape == (3, 5)
        assert f["msd/sem"].shape == (3, 5)
        # n_valid is per-element (matches mean shape) — distinguishes per-frame NaN.
        assert f["msd/n_valid"].shape == (3, 5)
    with h5py.File(out_dir / "force_orientation.h5", "r") as f:
        # Force-orientation has scalar s_mean per trajectory.
        assert f["s_mean/mean"].shape == (3,)
        assert f["s_mean/n_valid"].shape == (3,)


def test_reduce_partial_sweep_fills_nan(mini_sweep):
    coll = PsweepCollector(mini_sweep)
    _run_map(coll, ["msd"])
    # Wipe one cache so reduce sees a missing seed.
    coll.cache_path(2, 1).unlink()
    out_dir = _reduce.reduce_sweep(
        coll,
        analysis_specs=[("msd", {})],
        code_git_sha="testsha",
    )
    with h5py.File(out_dir / "msd.h5", "r") as f:
        # n_valid has shape (3 combos, 5 frames). Sum over the frame axis
        # (or pick any frame — they're all equal here) tells you how many
        # seeds contributed.
        n_valid = f["msd/n_valid"][()]
        assert n_valid[0, 0] == 2
        assert n_valid[1, 0] == 2
        assert n_valid[2, 0] == 1  # one seed missing


# ---------------------------------------------------------------------------
# ProcessedSweep reader
# ---------------------------------------------------------------------------


def test_processed_reader(mini_sweep):
    coll = PsweepCollector(mini_sweep)
    _run_map(coll, ["msd"])
    out = _reduce.reduce_sweep(coll, analysis_specs=[("msd", {})])

    with ProcessedSweep(out) as ps:
        assert ps.loop_vars == ["phi"]
        assert ps.analyses == ["msd"]
        np.testing.assert_allclose(ps.axes["phi"], np.linspace(0.1, 0.5, 3))
        assert ps.get("msd", "msd", kind="grid").shape == (3, 2, 5)
        assert ps.slice("msd", "msd", kind="mean", phi_idx=1).shape == (5,)


# ---------------------------------------------------------------------------
# Cache status enumeration
# ---------------------------------------------------------------------------


def test_cache_status_transitions(mini_sweep):
    coll = PsweepCollector(mini_sweep)
    desc = get_analysis("msd")
    h0 = desc.inputs_hash({})

    cp = coll.cache_path(0, 0)
    assert _cache.cache_status(cp, "msd", h0, desc.version) == "missing"

    _run_map(coll, ["msd"])
    assert _cache.cache_status(cp, "msd", h0, desc.version) == "complete"

    # Different inputs_hash should mark stale.
    h_other = desc.inputs_hash({"reference_frame": 1})
    assert _cache.cache_status(cp, "msd", h_other, desc.version) == "stale"

    # Wrong version -> stale.
    assert _cache.cache_status(cp, "msd", h0, 99) == "stale"
