# The GPUParticles Analysis Pipeline

A map-reduce pipeline for turning a parameter sweep of Brownian-dynamics trajectories into a per-analysis phase-diagram file. This document explains the moving parts, how to drive them from the command line, and how to plug in a new analysis.

---

## 1. Overview

A parameter sweep produced by `psweep.py` is a directory containing one trajectory `.h5` per `(combo, seed)` pair, plus three JSON manifests (`log.json`, `psweep_map.json`, `psweep_info.json`). The pipeline does two things:

1. **Map** — for every trajectory, run a set of registered Python analyses. Each analysis writes its result to a sibling cache file (`<traj>.cache.h5`) so repeated runs are skippable.
2. **Reduce** — once the caches are populated, combine them seed-by-seed and combo-by-combo into one phase-diagram file per analysis: `<sweep_dir>/<analysis>.h5`.

The two stages are independent: map runs locally or under SLURM, reduce runs in one process and is cheap. Adding, re-running, or removing one analysis touches only that analysis's outputs — all other reduced files stay put.

---

## 2. Architecture

```
sweep_dir/
├── log.json                   sweep-level manifest (loop_vars, n_combos, seeds, …)
├── psweep_map.json            combo_idx → parameter dict
├── psweep_info.json           loop-var axis specs (start / stop / num / format)
├── <sweep>_0000/              one folder per combo
│   ├── <sweep>_0000_seed0.h5  trajectory (BDTrajectory)
│   ├── <sweep>_0000_seed0.cache.h5   per-trajectory analysis cache
│   ├── input_seed0.json       per-run input parameters
│   └── …
├── <sweep>_0001/
│   └── …
├── msd.h5                     reduced output for "msd"
├── force_orientation.h5       reduced output for "force_orientation"
├── hexatic.h5                 reduced output for "hexatic"
└── clusters.h5                reduced output for "clusters"
```

### 2.1 `PsweepCollector`

`analysis.collector.PsweepCollector` is the read-only navigator over a sweep directory. It parses the three JSONs, derives canonical axes (re-computing the loop-var ranges from `psweep_info.json` to avoid float roundoff in `psweep_map.json`), and exposes `traj_path(combo_idx, seed)`, `cache_path(combo_idx, seed)`, `analysis_output_path(name)`, plus `assemble_grid_per_seed` / `assemble_grid_per_combo` for pivoting `{(combo, seed) → array}` dictionaries into N-D grids.

### 2.2 The analysis registry

Every analysis is a Python function decorated with `@register_analysis` (see `python/analysis/registry.py`). The decorator stores a descriptor with:

- `name`, `version` — used together to invalidate caches when the algorithm changes.
- `requires` — tuple of trajectory dataset names; the framework gates execution if any are missing.
- `outputs` — descriptive dict (shape strings + dtypes). Actual shapes are taken from the returned dict at run time.
- `combine` — optional callable that aggregates per-seed results into one per-combo summary, written into a `/<dataset>/combined` group in the reduced file.

A short `inputs_hash` (SHA1 over name + version + kwargs) tags cache entries so that re-running with different kwargs produces a new cache, not a silent collision.

### 2.3 Per-trajectory cache: `<traj>.cache.h5`

```
/                       root
  @sweep_dir, @combo_idx, @seed
  @traj_path             relative to sweep_dir
  @traj_mtime            for staleness checks
  @code_git_sha          short SHA of the analysis code
  @analyses_run          list[str]
/<analysis_name>/
  @analysis_version
  @inputs_hash           8-char SHA1 of the kwargs payload
  @analysis_kwargs       JSON dump
  @timestamp             ISO 8601
  @error                 only present on failure (group has no datasets)
  /<dataset_1>, /<dataset_2>, …
```

`run_and_write` in `python/analysis/cache.py` writes atomically (temp file + `os.replace`) under an advisory `fcntl` lock so concurrent map workers on the same trajectory don't clobber each other. The four `--rerun` policies tune what happens when an analysis group already exists: `skip` (default), `stale` (rename old group to `<name>__stale_<ts>` and rerun), `force` (delete and rerun the one group), `rebuild` (nuke the whole cache file before doing anything).

### 2.4 Per-analysis reduced output: `<analysis>.h5`

```
/                                root
  @sweep_dir, @analysis           analysis name matches the filename stem
  @analysis_version, @inputs_hash, @analysis_kwargs
  @loop_vars                      list[str]
  @processed_at, @code_git_sha, @reduce_args
  @n_trajectories_used, @n_trajectories_missing
/axes/<var>                       1-D canonical loop-var axis
/axes/seeds                       1-D explicit seed values
/<dataset>/
  @axis_order                     ["loop_var_1", …, "seed", *output_axes]
  /grid                           full per-seed grid (gzip-compressed)
  /mean                           seed-axis mean (np.nanmean)
  /sem                            seed-axis SEM (std / sqrt(n_valid))
  /n_valid                        finite-seed count per phase point
  /combined                       optional, populated only when combine() is defined
```

The `/<analysis>/` wrapper that older monolithic `processed.h5` files used is gone — the filename already names the analysis. Read it back through `ProcessedSweep`:

```python
from analysis import ProcessedSweep

with ProcessedSweep("/path/to/sweep") as ps:          # directory → discover
    print(ps.analyses)                                # ['clusters', 'force_orientation', 'hexatic', 'msd']
    msd = ps.get("msd", "msd", kind="mean")           # (Pe, De, frame)
    s_phase = ps.get("force_orientation", "s_mean")   # (Pe, De)

with ProcessedSweep("/path/to/sweep/msd.h5") as ps:   # single file
    print(ps.analyses)                                # ['msd']
```

The constructor accepts a directory (auto-discovers every `<name>.h5` whose root `@analysis` attr matches the filename stem) or a single `.h5` file. Files in the same directory must agree on `loop_vars` / `axes` / `seeds`, or the constructor raises.

---

## 3. Running the pipeline

The CLI is `python/analysis/process.py`. Five subcommands.

### 3.1 First run: `map`

```bash
python python/analysis/process.py map  <sweep_dir>  --analyses msd,hexatic,clusters
```

This iterates over every `(combo, seed)`, runs each named analysis, writes the per-trajectory caches, and (unless `--no-reduce` is passed) auto-reduces into the per-analysis files at the sweep root. `--analyses all` resolves to every registered name.

Useful flags:

- `--workers N` — local parallelism (default: `os.cpu_count()`).
- `--rerun {skip,stale,force,rebuild}` — default `skip` (only run when cache is missing/incomplete). Use `force` after editing analysis code.
- `--only-missing` — limit the task list to `(combo, seed)` pairs that don't have a complete cache for every requested analysis. Handy for resuming after partial failures.
- `--no-reduce` / `--reduce-anyway` — skip the auto-reduce step / run it even if some map tasks errored.
- `--dry-run` — list the task count without executing.

### 3.2 SLURM workflow

```bash
python python/analysis/process.py map <sweep_dir> --analyses all \
    --backend slurm --partition my_partition --time 04:00:00 --mem 4G
```

This writes a task list and a SLURM array script under `<sweep_dir>/_process/` and submits via `sbatch`. Each array task is `process.py one <sweep_dir> <combo_idx> <seed> --analyses …`, running one trajectory.

When the array finishes, reduce separately:

```bash
python python/analysis/process.py reduce <sweep_dir> --analyses all --cleanup
```

### 3.3 Reduce only

```bash
python python/analysis/process.py reduce <sweep_dir> --analyses msd,hexatic
python python/analysis/process.py reduce <sweep_dir> --analyses all
```

Writes `<sweep_dir>/<analysis>.h5` for each named analysis. `--output-dir DIR` places them elsewhere.

### 3.4 Cleanup: `--cleanup`

`--cleanup` (on both `map` and `reduce`) deletes the per-trajectory cache groups for the analyses you just reduced. Cache files that hold no remaining analyses are unlinked along with their `.lock` siblings; caches still containing other analyses are kept intact. The flag is opt-in — caches never disappear on their own.

### 3.5 Status, clean, one

```bash
python python/analysis/process.py status <sweep_dir>
python python/analysis/process.py status <sweep_dir> --analyses msd

python python/analysis/process.py clean  <sweep_dir> --analyses msd
python python/analysis/process.py clean  <sweep_dir> --analyses all --stale-only

python python/analysis/process.py one    <sweep_dir> 0 0 --analyses msd
```

`status` reports per-analysis completion counts (`complete / stale / error / missing`). `clean` is the standalone version of `--cleanup`. `one` runs the analyses for a single `(combo, seed)` — used by SLURM array tasks but also handy for local debugging.

---

## 4. Adding a new analysis

The whole template:

```python
# python/analysis/analyses/myparam.py
"""myparam — one-line headline of what this computes."""

from __future__ import annotations

import numpy as np

from bdtrajectory import BDTrajectory

from ..io import stack_dataset
from ..registry import register_analysis


@register_analysis(
    name="myparam",
    version=1,
    requires=("positions",),                         # tuple of dataset names
    outputs={
        "summary":   {"shape": "(num_frames,)", "dtype": "f8", "axes": ["frame"]},
        "spectrum":  {"shape": "(n_bins,)",     "dtype": "f8", "axes": ["bin"]},
    },
)
def analyze(traj: BDTrajectory, *, threshold: float = 0.5) -> dict[str, np.ndarray]:
    pos = stack_dataset(traj, "positions")  # (T, N, 2)
    if pos is None:
        raise RuntimeError("positions dataset is missing")

    summary = (pos[..., 0] > threshold).mean(axis=-1)  # (T,)
    edges, hist = np.histogram(pos.ravel(), bins=32)
    return {
        "summary":  summary.astype(np.float64),
        "spectrum": hist.astype(np.float64),
    }
```

Then wire the module into the registry by adding one line to `python/analysis/analyses/__init__.py`:

```python
from . import myparam as _myparam  # noqa: F401
```

Things to know:

- **Shape contract.** Every `(combo, seed)` must produce identically-shaped arrays per dataset key. The reduce step assembles them into `grid_shape + (num_seeds,) + tail_shape`; mismatched shapes raise. If your output is genuinely variable-length (e.g. cluster sizes per frame), bin it into a fixed-size histogram before returning — see `clusters.py` for the pattern.
- **`outputs` strings are descriptive.** The framework does not enforce them; actual shapes come from the returned ndarray. Use `(num_frames,)`, `(N,)`, `(n_bins,)` as documentation only. Future tooling may parse them for sanity-checks.
- **Kwargs participate in cache identity.** Every kwarg you accept feeds into `inputs_hash`, so running with different defaults produces a separate cache entry. Bump `version` when you change the algorithm itself (not its kwargs) and want existing caches invalidated as stale.
- **`requires`.** List any dataset accessor beyond `positions` that you call on the trajectory. The runtime calls `_missing_requirements` to skip work cleanly when, say, `orientations` is missing from a passive run; the cache records an error group instead of crashing.
- **Optional `combine`.** Pass `combine=my_combine` to `register_analysis` (or set the attribute on the returned descriptor). The framework calls `my_combine(seeds_present_for_one_combo)` per combo, then assembles the result via `coll.assemble_grid_per_combo` into a `/<dataset>/combined` group. Use it for aggregations that genuinely need access to all seeds (e.g. cross-seed correlations).
- **Performance.** The map stage parallelises across `(combo, seed)`, so per-trajectory work scales linearly with the number of workers. Heavy NumPy / scipy is fine inside `analyze`; avoid touching disk again from inside the function — pull everything from the `BDTrajectory` object the framework hands you.

### 4.1 Smoke test recipe

Before launching across the whole sweep, run on one trajectory and inspect the cache:

```bash
python python/analysis/process.py one <sweep_dir> 0 0 --analyses myparam
python -c "
import h5py, sys
from pathlib import Path
sys.path.insert(0, 'python')
from analysis.collector import PsweepCollector
coll = PsweepCollector('<sweep_dir>')
with h5py.File(coll.cache_path(0, 0), 'r') as f:
    print(list(f['myparam'].keys()))
"
```

Then reduce against a slice:

```bash
python python/analysis/process.py reduce <sweep_dir> --analyses myparam
```

and load the output:

```python
from analysis import ProcessedSweep
with ProcessedSweep("<sweep_dir>") as ps:
    print(ps.datasets("myparam"))
    print(ps.get("myparam", "summary", kind="mean").shape)
```

---

## 5. Reading reduced output

Five `kind`s are available per dataset, accessed through `ProcessedSweep.get(analysis, dataset, kind=…)`:

| kind | shape | content |
|---|---|---|
| `grid` | `grid_shape + (num_seeds,) + tail_shape` | full per-seed grid, NaN-filled where seeds are missing |
| `mean` | `grid_shape + tail_shape` | seed-axis `np.nanmean` |
| `sem` | `grid_shape + tail_shape` | seed-axis SEM, `nanstd / sqrt(n_valid)` (NaN where `n_valid ≤ 1`) |
| `n_valid` | `grid_shape + tail_shape` (int64) | count of finite seeds per phase point |
| `combined` | `grid_shape + tail_shape` | per-combo `combine()` aggregation (only present if the analysis declares one) |

`ProcessedSweep.slice(analysis, dataset, kind=…, <loop_var>_idx=N)` reads any single `kind` and slices along loop-var axes by index, so the typical "fix all axes but one" plotting pattern is one line:

```python
msd_at_pe_idx_5 = ps.slice("msd", "msd", kind="mean", Pe_idx=5)   # (De, frame)
```

`ps.axis_order(analysis, dataset)` lists the named axes for one dataset, and `ps.file_path(analysis)` returns the underlying `<name>.h5` path if you need raw h5py access.
