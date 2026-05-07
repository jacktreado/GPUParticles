#!/usr/bin/env python3
"""
psweep.py — Parameter sweeps for the GPUParticles Brownian sim.

For each Cartesian-product combination of the swept variables, and for each
random seed, this script writes one input JSON, optionally generates a SLURM
array job per combination, and (with --submit) sbatch's them.

Sweep variables are restricted to the *physical* knobs of the model:

    N, phi, delta, Pe, De, R, C

Anything else in the input JSON (the integrator, its tolerances/skin/timestep,
output cadence, init mode, file paths, RNG seed, simulation duration,
potential type) can be set as a constant for the whole sweep via
`--<key> <value>`, but cannot be looped over. The integrator in particular
is a numerical-method choice, not a scientific parameter, so attempting to
sweep it raises an error.

Examples:

    # Print eligible sweep parameters and exit.
    ./psweep.py --info

    # 1D sweep over phi at 5 linearly spaced points, 10 seeds per combo.
    # Generates files under psweeps/ but does not submit.
    ./psweep.py --psweep --tag phi_dense --note "phi sweep" \
        --loop --var phi --start 0.1 --stop 0.5 --num 5 --format lin \
        --start-seed 0 --num-seeds 10

    # 2D sweep (phi x Pe) defined in a JSON spec, submitted to SLURM.
    ./psweep.py --psweep --tag big --note "phi x Pe" --ljson sweep.json \
        --start-seed 0 --num-seeds 8 \
        --submit --partition short --time 02:00:00 --mem 2G

    # Override non-swept parameters as constants for the whole sweep.
    ./psweep.py --psweep --tag heun --note "heun all" \
        --loop --var Pe --start 1 --stop 100 --num 4 --format log \
        --integrator heun --t_end 50 --output_dt 0.5 \
        --start-seed 0 --num-seeds 4
"""

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

import numpy as np


# ---- Paths -------------------------------------------------------------------

REPO_ROOT       = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE    = os.path.join(REPO_ROOT, "examples", "input.json")
BUILD_DIR       = os.path.join(REPO_ROOT, "build")
DEFAULT_BINARY  = os.path.join(BUILD_DIR, "sim")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "psweeps")


# ---- What may and may not be swept ------------------------------------------
#
# Sweep variables are restricted to the physical/scientific knobs of the
# model. Numerical-method parameters (integrator method, tolerances, skin,
# timestep, drift cap, output cadence) and I/O/RNG parameters (file names,
# init mode, seed) are *not* sweepable: looping over them is almost never
# what a user actually means, and conflating physical and numerical scans
# leads to misleading plots.

SWEEPABLE = {
    "N":     "particle count (cast to int)",
    "phi":   "2D packing fraction in (0, 1)",
    "delta": "typical pair overlap r*/sigma at steady contact",
    "Pe":    "active Peclet f0 * tau_theta / (gamma * sigma)",
    "De":    "active Deborah (gamma_a / k_a) / tau_theta",
    "R":     "friction ratio gamma_a / gamma",
    "C":     "dimensionless temperature kT / epsilon",
}

# Cast to int when applying the swept value back into the JSON.
INT_PARAMS = {"N"}

# Why each non-sweepable JSON field is rejected as a loop variable.
NON_SWEEPABLE = {
    "integrator":  "a numerical-method choice, not a physical parameter",
    "potential":   "a categorical choice; set it once via --potential <name>",
    "init_mode":   "an initialization mode, not a continuous parameter",
    "output_file": "an I/O target — psweep auto-names outputs per run",
    "seed":        "controlled separately via --start-seed / --num-seeds",
    "dt_init":     "an integrator timestep, not a physical parameter",
    "abs_tol":     "an integrator tolerance, not a physical parameter",
    "rel_tol":     "an integrator tolerance, not a physical parameter",
    "max_drift":   "an integrator stability knob, not a physical parameter",
    "r_skin":      "a cell-list buffer, not a physical parameter",
    "t_end":       "a simulation duration, not a physical parameter",
    "output_dt":   "an output cadence, not a physical parameter",
}

# Every JSON key the script knows about. Used to validate --<key> overrides.
ALL_INPUT_KEYS = set(SWEEPABLE) | set(NON_SWEEPABLE)

# Constrained string fields: validate override values.
ALLOWED_VALUES = {
    "integrator": {"euler_maruyama", "heun", "adaptive_strang"},
    "potential":  {"WCA", "soft_sphere"},
    "init_mode":  {"lattice", "random"},
}

# String-typed input fields (don't try to cast their override values to float).
STRING_FIELDS = {"integrator", "potential", "init_mode", "output_file"}


# =============================================================================
# Argument parsing
# =============================================================================

def make_parser():
    p = argparse.ArgumentParser(
        prog="psweep.py",
        description="Parameter sweep driver for the GPUParticles Brownian sim.",
        epilog=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--info",   action="store_true",
                      help="Print eligible sweep parameters and exit.")
    mode.add_argument("--psweep", action="store_true",
                      help="Construct (and optionally submit) a parameter sweep.")

    sweep = p.add_argument_group("Sweep identity")
    sweep.add_argument("--tag",  type=str,
                       help="Short tag distinguishing this sweep (required with --psweep).")
    sweep.add_argument("--note", type=str,
                       help="Free-form note saved in log.json (required with --psweep).")
    sweep.add_argument("--base", type=str, default=DEFAULT_BASE,
                       help=f"Path to the base input.json (default: {DEFAULT_BASE}).")
    sweep.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR,
                       help=f"Output directory for sweep results (default: {DEFAULT_OUT_DIR}).")
    sweep.add_argument("--scratch-dir", type=str, default=None,
                       help="Optional scratch directory. If set, the binary writes "
                            "trajectories there and the SLURM job moves them back to "
                            "the sweep directory on exit.")

    loop = p.add_argument_group("Loop specification (use --loop OR --ljson)")
    loop.add_argument("--loop",  action="store_true",
                      help="Single-variable loop on the CLI.")
    loop.add_argument("--var",   type=str, help="Variable name (with --loop).")
    loop.add_argument("--start", type=float, help="Loop start (with --loop).")
    loop.add_argument("--stop",  type=float, help="Loop stop  (with --loop).")
    loop.add_argument("--num",   type=int,   help="Number of points (with --loop).")
    loop.add_argument("--format", type=str, choices=["lin", "log"],
                      help="Spacing between start and stop (with --loop).")
    loop.add_argument("--ljson", type=str,
                      help="Multi-variable loop spec: a JSON file mapping each loop "
                           "variable to {start, stop, num, format}. See --info.")

    seeds = p.add_argument_group("Random seeds")
    seeds.add_argument("-s", "--start-seed", type=int, default=0,
                       help="First seed value (default: 0).")
    seeds.add_argument("-n", "--num-seeds",  type=int, default=1,
                       help="Number of seeds per combination (default: 1).")

    build = p.add_argument_group("Build")
    build.add_argument("--no-build", action="store_true",
                       help="Skip the cmake/make step. Default is to (re)build "
                            "the sim binary before generating the sweep.")
    build.add_argument("--build-type", type=str, default="Release",
                       help="CMAKE_BUILD_TYPE (default: Release).")

    slurm = p.add_argument_group("SLURM submission")
    slurm.add_argument("--submit", action="store_true",
                       help="Actually call sbatch on the generated SLURM files. "
                            "Default just generates files for inspection.")
    slurm.add_argument("--partition", type=str, default=None,
                       help="SLURM partition (required with --submit).")
    slurm.add_argument("-t", "--time", type=str, default="06:00:00",
                       help="SLURM walltime (default: 06:00:00).")
    slurm.add_argument("-m", "--mem",  type=str, default="2G",
                       help="SLURM memory per task (default: 2G).")
    slurm.add_argument("--cpus", type=int, default=1,
                       help="SLURM cpus-per-task (default: 1).")

    p.add_argument("--list-only", action="store_true",
                   help="Print the resolved per-run parameters and exit "
                        "without writing any files.")
    return p


def assert_psweep_args(known):
    if not known.tag:
        raise ValueError("--tag is required with --psweep")
    if not known.note:
        raise ValueError("--note is required with --psweep")
    if known.submit and not known.partition:
        raise ValueError("--partition is required when --submit is set")
    if known.num_seeds < 1:
        raise ValueError("--num-seeds must be >= 1")


def parse_overrides(unknown):
    """Parse `--key value` pairs from leftover args.

    Each key must be a known JSON field. Override values for STRING_FIELDS
    are taken verbatim and validated against ALLOWED_VALUES; everything else
    is parsed as a number (with int cast for INT_PARAMS).
    """
    if len(unknown) % 2 != 0:
        raise ValueError(
            f"Override args must come in --key value pairs; got: {unknown}"
        )

    overrides = {}
    for i in range(0, len(unknown), 2):
        key = unknown[i].lstrip("-")
        raw = unknown[i + 1]
        if key not in ALL_INPUT_KEYS:
            raise ValueError(
                f'Unknown override "--{key}". Allowed keys: '
                f"{sorted(ALL_INPUT_KEYS)}"
            )

        if key in STRING_FIELDS:
            if key in ALLOWED_VALUES and raw not in ALLOWED_VALUES[key]:
                raise ValueError(
                    f'Invalid value "{raw}" for --{key}; '
                    f"allowed: {sorted(ALLOWED_VALUES[key])}"
                )
            overrides[key] = raw
        elif key in INT_PARAMS:
            overrides[key] = int(round(float(raw)))
        else:
            overrides[key] = float(raw)

    return overrides


# =============================================================================
# Loop spec validation & expansion
# =============================================================================

def print_info():
    print("Eligible sweep variables (numeric):")
    for k, desc in SWEEPABLE.items():
        print(f"  {k:<8} — {desc}")
    print()
    print("Forbidden as sweep variables (kept fixed across the sweep):")
    for k, why in NON_SWEEPABLE.items():
        print(f"  {k:<12} — {why}")
    print()
    print("--ljson file schema (one entry per loop variable):")
    print(json.dumps({
        "phi": {"start": 0.1, "stop": 0.5, "num": 5, "format": "lin"},
        "Pe":  {"start": 1.0, "stop": 100, "num": 4, "format": "log"},
    }, indent=2))


def validate_loop_var(name):
    """Raise a helpful error if `name` is not a sweepable parameter."""
    if name in SWEEPABLE:
        return
    if name in NON_SWEEPABLE:
        raise ValueError(
            f'Cannot sweep over "{name}": {NON_SWEEPABLE[name]}.\n'
            f"Run with --info to see eligible sweep parameters."
        )
    raise ValueError(
        f'Cannot sweep over "{name}": not a known input parameter.\n'
        f"Run with --info to see eligible sweep parameters."
    )


def loop_info_from_cli(args):
    if not args.var:
        raise ValueError("--loop requires --var")
    missing = [k for k in ("start", "stop", "num", "format")
               if getattr(args, k) is None]
    if missing:
        raise ValueError(f"--loop requires {missing} (in addition to --var)")
    validate_loop_var(args.var)
    return {args.var: {
        "start":  float(args.start),
        "stop":   float(args.stop),
        "num":    int(args.num),
        "format": str(args.format),
    }}


def loop_info_from_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"--ljson file not found: {path}")
    with open(path) as f:
        spec = json.load(f)
    if not isinstance(spec, dict) or not spec:
        raise ValueError(f"--ljson file must be a non-empty JSON object: {path}")

    out = {}
    for name, entry in spec.items():
        validate_loop_var(name)
        for req in ("start", "stop", "num", "format"):
            if req not in entry:
                raise ValueError(
                    f'Loop "{name}": missing required field "{req}" '
                    "(start, stop, num, format are all required)"
                )
        fmt = str(entry["format"])
        if fmt not in ("lin", "log"):
            raise ValueError(f'Loop "{name}": format must be "lin" or "log"')
        out[name] = {
            "start":  float(entry["start"]),
            "stop":   float(entry["stop"]),
            "num":    int(entry["num"]),
            "format": fmt,
        }
    return out


def expand_combinations(loop_info):
    """Cartesian product over loop variables. Returns (var_names, combos)."""
    names, grids = [], []
    for n, e in loop_info.items():
        names.append(n)
        if e["num"] < 1:
            raise ValueError(f'Loop "{n}": num must be >= 1')
        if e["format"] == "lin":
            grids.append(list(np.linspace(e["start"], e["stop"], e["num"])))
        else:  # log
            if e["start"] <= 0 or e["stop"] <= 0:
                raise ValueError(
                    f'Loop "{n}": log spacing requires start > 0 and stop > 0'
                )
            grids.append(list(np.logspace(np.log10(e["start"]),
                                          np.log10(e["stop"]),
                                          e["num"])))
    return names, list(itertools.product(*grids))


def cast_value(name, v):
    return int(round(float(v))) if name in INT_PARAMS else float(v)


# =============================================================================
# Build
# =============================================================================

def build_binary(build_type):
    """Configure (if needed) and build the `sim` target. Raises on failure."""
    print(f"Configuring with cmake (build type: {build_type})...")
    subprocess.run(
        ["cmake", "-S", REPO_ROOT, "-B", BUILD_DIR,
         f"-DCMAKE_BUILD_TYPE={build_type}"],
        check=True,
    )
    print("Building target 'sim'...")
    subprocess.run(
        ["cmake", "--build", BUILD_DIR, "-j", "--target", "sim"],
        check=True,
    )
    if not os.path.exists(DEFAULT_BINARY):
        raise FileNotFoundError(
            f"Build reported success but binary not found at {DEFAULT_BINARY}"
        )


# =============================================================================
# JSON base + per-run materialization
# =============================================================================

def load_base_json(path, overrides):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Base input JSON not found: {path}")
    with open(path) as f:
        base = json.load(f)
    if not isinstance(base, dict):
        raise ValueError(f"Base JSON must be a flat object: {path}")
    base.update(overrides)
    return base


def write_run_inputs(psweep_dir, psweep_name, base, var_names, combos,
                     start_seed, num_seeds, scratch_dir):
    """For each combo, create a run dir; for each seed, write an input JSON.

    Returns:
        runs:  list of (combo_idx, run_dir, [(seed, json_path), ...])
        seeds: range of seed values
    """
    seeds = list(range(start_seed, start_seed + num_seeds))

    psweep_map = {
        "psweep_name": psweep_name,
        "loop_vars":   var_names,
        "start_seed":  start_seed,
        "num_seeds":   num_seeds,
        "combos":      {},
    }

    runs = []
    for ci, combo in enumerate(combos):
        run_name = f"{psweep_name}_{ci:04d}"
        run_dir  = os.path.join(psweep_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        run_base = dict(base)
        combo_map = {}
        for vi, vname in enumerate(var_names):
            v = cast_value(vname, combo[vi])
            run_base[vname] = v
            combo_map[vname] = v
        psweep_map["combos"][str(ci)] = combo_map

        per_seed = []
        for seed in seeds:
            run_seed = dict(run_base)
            run_seed["seed"] = int(seed)
            output_dir = scratch_dir if scratch_dir else run_dir
            run_seed["output_file"] = os.path.join(
                output_dir, f"{run_name}_seed{seed}.h5"
            )
            jp = os.path.join(run_dir, f"input_seed{seed}.json")
            with open(jp, "w") as f:
                json.dump(run_seed, f, indent=2)
            per_seed.append((seed, jp))

        runs.append((ci, run_dir, per_seed))

    with open(os.path.join(psweep_dir, "psweep_map.json"), "w") as f:
        json.dump(psweep_map, f, indent=2)

    return runs, seeds


# =============================================================================
# SLURM file generation & submission
# =============================================================================

def write_slurm(psweep_dir, psweep_name, runs, args, binary_path, scratch_dir):
    slurm_root = os.path.join(psweep_dir, "slurm")
    os.makedirs(slurm_root, exist_ok=True)

    slurm_files = []
    for ci, run_dir, per_seed in runs:
        run_name      = f"{psweep_name}_{ci:04d}"
        run_slurm_dir = os.path.join(slurm_root, run_name)
        os.makedirs(run_slurm_dir, exist_ok=True)

        task_file  = os.path.join(run_slurm_dir, run_name + ".task")
        slurm_file = os.path.join(run_slurm_dir, run_name + ".slurm")

        with open(task_file, "w") as f:
            for seed, jp in per_seed:
                cmd = f'{binary_path} run "{jp}"'
                if scratch_dir:
                    src = os.path.join(scratch_dir, f"{run_name}_seed{seed}.h5")
                    dst = os.path.join(run_dir,   f"{run_name}_seed{seed}.h5")
                    cmd = (f'mkdir -p "{scratch_dir}" && {cmd} && '
                           f'mv -v "{src}" "{dst}"')
                f.write(cmd + "\n")

        with open(slurm_file, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"#SBATCH --partition={args.partition}\n")
            f.write(f"#SBATCH --time={args.time}\n")
            f.write(f"#SBATCH --mem={args.mem}\n")
            f.write(f"#SBATCH --cpus-per-task={args.cpus}\n")
            f.write("#SBATCH --ntasks=1\n")
            f.write(f"#SBATCH --job-name={run_name}\n")
            f.write(f"#SBATCH --array=1-{len(per_seed)}\n")
            f.write(f"#SBATCH --output={run_slurm_dir}/{run_name}-%a.out\n")
            f.write("\n")
            f.write("set -eo pipefail\n")
            f.write('echo "Job started on $(date) on $(hostname)"\n')
            f.write(f'cmd=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "{task_file}")\n')
            f.write('echo "+ $cmd"\n')
            f.write('eval "$cmd"\n')
            f.write('echo "Job finished on $(date)"\n')

        slurm_files.append(slurm_file)

    return slurm_files


def submit(slurm_files):
    for sf in slurm_files:
        print(f"sbatch {sf}")
        subprocess.run(["sbatch", sf], check=True)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = make_parser()
    known, unknown = parser.parse_known_args()

    if known.info:
        print_info()
        return 0

    # ---- --psweep branch ---------------------------------------------------
    assert_psweep_args(known)

    # Build (auto unless --no-build).
    if not known.no_build:
        build_binary(known.build_type)
    if not os.path.exists(DEFAULT_BINARY):
        raise FileNotFoundError(
            f"Binary not found at {DEFAULT_BINARY}. Re-run without --no-build "
            f"or build manually:\n"
            f"  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release\n"
            f"  cmake --build build -j --target sim"
        )

    overrides = parse_overrides(unknown)

    if known.ljson:
        loop_info = loop_info_from_json(known.ljson)
    elif known.loop:
        loop_info = loop_info_from_cli(known)
    else:
        raise ValueError("Provide either --loop (CLI) or --ljson (file).")

    for k in overrides:
        if k in loop_info:
            raise ValueError(
                f'"{k}" is both swept (in --loop/--ljson) and overridden via '
                f"--{k}; choose one."
            )

    var_names, combos = expand_combinations(loop_info)

    # Resolve names (still useful for --list-only).
    date_str    = datetime.today().strftime("%Y-%m-%d")
    psweep_name = f"psweep_{date_str}_{known.tag}"
    psweep_dir  = os.path.join(known.out_dir, psweep_name)

    base = load_base_json(known.base, overrides)

    if known.list_only:
        print(f"\nSweep:  {psweep_name}  (dry run, no files written)")
        print(f"  loop_vars   : {var_names}")
        print(f"  combos      : {len(combos)}")
        print(f"  seeds/combo : {known.num_seeds} "
              f"(seeds={known.start_seed}..{known.start_seed + known.num_seeds - 1})")
        for ci, combo in enumerate(combos):
            print(f"  [{ci:04d}]  " +
                  ", ".join(f"{n}={cast_value(n, v)}"
                            for n, v in zip(var_names, combo)))
        return 0

    if os.path.exists(psweep_dir):
        print(f"Warning: {psweep_dir} already exists — overwriting contents.")
    os.makedirs(psweep_dir, exist_ok=True)

    log = {
        "psweep_name": psweep_name,
        "date":        date_str,
        "tag":         known.tag,
        "note":        known.note,
        "base_json":   os.path.abspath(known.base),
        "binary":      os.path.abspath(DEFAULT_BINARY),
        "overrides":   overrides,
        "loop_info":   loop_info,
        "loop_vars":   var_names,
        "n_combos":    len(combos),
        "start_seed":  known.start_seed,
        "num_seeds":   known.num_seeds,
        "scratch_dir": known.scratch_dir,
    }
    with open(os.path.join(psweep_dir, "log.json"), "w") as f:
        json.dump(log, f, indent=2)
    with open(os.path.join(psweep_dir, "psweep_info.json"), "w") as f:
        json.dump(loop_info, f, indent=2)

    runs, seeds = write_run_inputs(
        psweep_dir, psweep_name, base, var_names, combos,
        known.start_seed, known.num_seeds, known.scratch_dir,
    )

    slurm_files = write_slurm(
        psweep_dir, psweep_name, runs, known, DEFAULT_BINARY, known.scratch_dir
    )

    n_tasks = len(combos) * known.num_seeds
    print(f"\nSweep ready: {psweep_dir}")
    print(f"  loop_vars   : {var_names}")
    print(f"  combos      : {len(combos)}")
    print(f"  seeds/combo : {known.num_seeds}")
    print(f"  total runs  : {n_tasks}")
    print(f"  slurm files : {len(slurm_files)} "
          f"(under {os.path.join(psweep_dir, 'slurm')})")

    if known.submit:
        submit(slurm_files)
    else:
        print("\nNot submitted (use --submit to sbatch).")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
