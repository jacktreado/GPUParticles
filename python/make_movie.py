#!/usr/bin/env python3
"""
make_movie.py — render a GPUParticles trajectory to an .mp4 movie.

Usage
-----
    python make_movie.py <h5_path> [options]

The particle color, frame range, and camera (zoom) window are all
configurable via CLI flags. See `--help` for the full list.
"""

from __future__ import annotations

# IMPORTANT: Set HDF5_USE_FILE_LOCKING before importing h5py anywhere (same
# defense used in python/analysis/process.py).
import os as _os

_os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import PatchCollection
from tqdm import tqdm

# Bootstrap: allow `python make_movie.py ...` regardless of cwd.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from bdtrajectory import BDTrajectory  # noqa: E402
from cluster_utils import ClusterUtils  # noqa: E402

_REPO_ROOT = _HERE.parent

COLOR_BY_CHOICES = [
    "cluster_fraction",
    "cluster_id",
    "orientation",
    "speed",
    "force",
    "hexatic",
    "density",
]

# Dataset each color_by option needs, for fail-fast validation. None = always
# available (only needs positions, which every trajectory file has).
_REQUIRED_DATASET = {
    "cluster_fraction": None,
    "cluster_id": None,
    "orientation": "orientations",
    "speed": "velocities",
    "force": "forces",
    "hexatic": None,
    "density": None,
}

_DEFAULT_CMAP = {
    "cluster_fraction": "inferno",
    "cluster_id": "tab20",
    "orientation": "twilight",
    "speed": "viridis",
    "force": "viridis",
    "hexatic": "viridis",
    "density": "viridis",
}

_COLORBAR_LABEL = {
    "cluster_fraction": "cluster fraction",
    "cluster_id": "cluster id",
    "orientation": "orientation (rad)",
    "speed": "speed |v|",
    "force": "force |F|",
    "hexatic": "hexatic order |psi6|",
    "density": "local packing fraction",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="make_movie.py",
        description="Render a GPUParticles HDF5 trajectory to an .mp4 movie.",
    )
    p.add_argument("h5_path", help="Path to the trajectory HDF5 file.")

    p.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Folder to write the movie into. Default: "
            "<repo_root>/output/<YYYY>/<YYYY-MM>/<YYYY-MM-DD>/ "
            "(created if it doesn't exist)."
        ),
    )
    p.add_argument(
        "--filename",
        default=None,
        help="Output file name. Default: '<input_stem>_<color_by>.mp4'.",
    )
    p.add_argument(
        "--extra-tag",
        default=None,
        help="Extra tag to append to the output filename.",
    )

    p.add_argument("--start-frame", type=int, default=None, help="First frame index (default: 0).")
    p.add_argument("--end-frame", type=int, default=None, help="Last frame index, inclusive (default: last frame).")
    p.add_argument(
        "--frame-skip",
        type=int,
        default=0,
        help="Number of frames to skip between rendered frames (default: 0 = every frame).",
    )

    p.add_argument("--zoom-x", type=float, default=0.5, help="Camera center x, in units of L (default: 0.5).")
    p.add_argument("--zoom-y", type=float, default=0.5, help="Camera center y, in units of L (default: 0.5).")
    p.add_argument(
        "--zoom-frac",
        type=float,
        default=1.0,
        help="Camera window width/height, in units of L (default: 1.0 = whole box).",
    )

    p.add_argument(
        "--color-by",
        choices=COLOR_BY_CHOICES,
        default="cluster_fraction",
        help="Per-particle quantity to color by (default: cluster_fraction).",
    )
    p.add_argument(
        "--cluster-threshold",
        type=float,
        default=1.05,
        help="Link distance for cluster_fraction/cluster_id, in units of sigma (default: 1.05).",
    )

    p.add_argument("--vmin", type=float, default=None, help="Override the lower color-scale bound.")
    p.add_argument("--vmax", type=float, default=None, help="Override the upper color-scale bound.")
    p.add_argument("--cmap", default=None, help="Override the default colormap for the chosen --color-by.")

    p.add_argument("--fps", type=int, default=20, help="Output frame rate (default: 20).")
    p.add_argument("--dpi", type=int, default=150, help="Output resolution in dots per inch (default: 150).")
    p.add_argument("--figsize", type=float, nargs=2, default=(6.0, 6.0), metavar=("W", "H"), help="Figure size in inches (default: 6 6).")

    p.add_argument(
        "--show-orientations",
        action="store_true",
        help="Overlay orientation arrows regardless of --color-by (requires the orientations dataset).",
    )
    p.add_argument("--no-box", action="store_true", help="Don't draw the periodic-cell boundary rectangle.")

    p.add_argument("--edgecolor", default="navy", help="Particle edge color (default: navy).")
    p.add_argument("--alpha", type=float, default=0.85, help="Particle fill opacity (default: 0.85).")
    p.add_argument("--linewidth", type=float, default=0.5, help="Particle edge line width (default: 0.5).")
    p.add_argument("--title", default=None, help="Fixed title. Default: per-frame 'frame i  step s  t=...'.")
    p.add_argument(
        "--no-labels",
        action="store_true",
        help=(
            "Strip the title (frame/step/time) and colorbar, and fill the whole "
            "frame with the simulation view (no surrounding whitespace) — useful "
            "for websites and supplemental-material clips."
        ),
    )

    return p


# ---------------------------------------------------------------------------
# Output path resolution
# ---------------------------------------------------------------------------
def _resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output_dir is None:
        today = date.today()
        out_dir = (
            _REPO_ROOT
            / "output"
            / f"{today.year:04d}"
            / f"{today.year:04d}-{today.month:02d}"
            / f"{today.year:04d}-{today.month:02d}-{today.day:02d}"
        )
    else:
        out_dir = Path(args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.filename is None:
        stem = Path(args.h5_path).stem
        if args.extra_tag is not None:
            stem += f"_{args.extra_tag}"
        filename = f"{stem}_{args.color_by}.mp4"
    else:
        filename = args.filename
        if not filename.endswith(".mp4"):
            filename += ".mp4"

    return out_dir / filename


# ---------------------------------------------------------------------------
# Frame selection / validation
# ---------------------------------------------------------------------------
def _resolve_frame_list(traj: BDTrajectory, args: argparse.Namespace) -> list[int]:
    start = 0 if args.start_frame is None else args.start_frame
    end = traj.num_frames - 1 if args.end_frame is None else args.end_frame
    if start < 0 or start >= traj.num_frames:
        raise ValueError(f"--start-frame {start} out of range [0, {traj.num_frames})")
    if end < start or end >= traj.num_frames:
        raise ValueError(f"--end-frame {end} out of range [{start}, {traj.num_frames})")
    if args.frame_skip < 0:
        raise ValueError("--frame-skip must be >= 0")
    stride = args.frame_skip + 1
    return list(range(start, end + 1, stride))


def _validate_color_by(traj: BDTrajectory, color_by: str) -> None:
    required = _REQUIRED_DATASET[color_by]
    if required is None:
        return
    getter = getattr(traj, required)
    if getter(0) is None:
        available = [c for c in COLOR_BY_CHOICES if _REQUIRED_DATASET[c] is None]
        raise ValueError(
            f"--color-by {color_by} requires the '{required}' dataset, which this "
            f"trajectory file does not have. Options that don't need it: {available}"
        )


# ---------------------------------------------------------------------------
# Color-value computation
# ---------------------------------------------------------------------------
def _compute_color_arrays(
    traj: BDTrajectory, frames: list[int], color_by: str, cluster_threshold: float
) -> dict[int, np.ndarray]:
    sigma = traj.sigma
    values: dict[int, np.ndarray] = {}
    for fidx in tqdm(frames, desc=f"computing {color_by}"):
        pos = traj.positions(fidx)
        Lx, _Ly = traj.frame_box(fidx)
        x, y = pos[:, 0], pos[:, 1]

        if color_by == "cluster_fraction":
            labels, sizes = ClusterUtils.find_clusters(x, y, cluster_threshold * sigma, Lx)
            values[fidx] = sizes[labels] / traj.N
        elif color_by == "cluster_id":
            labels, _sizes = ClusterUtils.find_clusters(x, y, cluster_threshold * sigma, Lx)
            values[fidx] = labels.astype(float)
        elif color_by == "orientation":
            values[fidx] = traj.orientations(fidx)
        elif color_by == "speed":
            v = traj.velocities(fidx)
            values[fidx] = np.hypot(v[:, 0], v[:, 1])
        elif color_by == "force":
            f = traj.forces(fidx)
            values[fidx] = np.hypot(f[:, 0], f[:, 1])
        elif color_by == "hexatic":
            values[fidx] = np.abs(ClusterUtils.compute_hexatic_order(x, y, Lx))
        elif color_by == "density":
            values[fidx] = ClusterUtils.local_voronoi_phi(x, y, sigma, Lx)
        else:
            raise ValueError(f"Unknown color_by: {color_by}")
    return values


def _color_range(
    color_by: str, values: dict[int, np.ndarray], vmin: Optional[float], vmax: Optional[float]
) -> tuple[float, float]:
    if color_by in ("cluster_fraction", "hexatic"):
        lo, hi = 0.0, 1.0
    elif color_by == "orientation":
        lo, hi = -np.pi, np.pi
    elif color_by == "cluster_id":
        lo, hi = 0.0, 19.0
    else:
        all_vals = np.concatenate([v[np.isfinite(v)] for v in values.values()])
        lo, hi = float(np.percentile(all_vals, 1)), float(np.percentile(all_vals, 99))
    if vmin is not None:
        lo = vmin
    if vmax is not None:
        hi = vmax
    return lo, hi


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _tile_and_filter(pos: np.ndarray, color_vals: np.ndarray, Lx: float, Ly: float, sigma: float, xlim, ylim):
    """Wrap positions into [0, L), tile 3x3 periodically, and keep only the
    particles (and their color values) whose circle overlaps the zoom window."""
    x = np.mod(pos[:, 0], Lx)
    y = np.mod(pos[:, 1], Ly)

    margin = sigma / 2.0
    xs, ys, cs = [], [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            tx = x + dx * Lx
            ty = y + dy * Ly
            mask = (
                (tx >= xlim[0] - margin)
                & (tx <= xlim[1] + margin)
                & (ty >= ylim[0] - margin)
                & (ty <= ylim[1] + margin)
            )
            if np.any(mask):
                xs.append(tx[mask])
                ys.append(ty[mask])
                cs.append(color_vals[mask])

    if not xs:
        return np.empty(0), np.empty(0), np.empty(0)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(cs)


def render_movie(
    traj: BDTrajectory,
    frames: list[int],
    color_arrays: dict[int, np.ndarray],
    color_by: str,
    cmap_name: str,
    vmin: float,
    vmax: float,
    args: argparse.Namespace,
    out_path: Path,
) -> None:
    sigma = traj.sigma
    r = sigma / 2.0
    L = traj.Lx  # square box

    zoom_half = args.zoom_frac * L / 2.0
    cx, cy = args.zoom_x * L, args.zoom_y * L
    xlim = (cx - zoom_half, cx + zoom_half)
    ylim = (cy - zoom_half, cy + zoom_half)

    if color_by == "cluster_id":
        cmap = plt.get_cmap(cmap_name, 20)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        def to_color(vals):
            return cmap((vals.astype(int) % 20) / 19.0)
    else:
        cmap = plt.get_cmap(cmap_name)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        def to_color(vals):
            return cmap(norm(vals))

    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    if args.no_labels:
        # No colorbar, and the axes fill the entire canvas so the saved
        # frames have no surrounding whitespace/margins.
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.set_position([0, 0, 1, 1])
    else:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=_COLORBAR_LABEL[color_by])

    writer = FFMpegWriter(fps=args.fps)
    with writer.saving(fig, str(out_path), dpi=args.dpi):
        for fidx in tqdm(frames, desc="rendering"):
            ax.cla()

            pos = traj.positions(fidx)
            vals = color_arrays[fidx]
            tx, ty, tvals = _tile_and_filter(pos, vals, L, L, sigma, xlim, ylim)

            if tx.size > 0:
                facecolors = to_color(tvals)
                circles = [mpatches.Circle((tx[i], ty[i]), radius=r) for i in range(tx.size)]
                col = PatchCollection(
                    circles,
                    facecolor=facecolors,
                    edgecolor=args.edgecolor,
                    alpha=args.alpha,
                    linewidth=args.linewidth,
                )
                ax.add_collection(col)

            if args.show_orientations:
                theta = traj.orientations(fidx)
                if theta is not None:
                    x0 = np.mod(pos[:, 0], L)
                    y0 = np.mod(pos[:, 1], L)
                    arrow_len = r * 0.4
                    for i in range(traj.N):
                        ax.annotate(
                            "",
                            xy=(x0[i] + arrow_len * np.cos(theta[i]), y0[i] + arrow_len * np.sin(theta[i])),
                            xytext=(x0[i], y0[i]),
                            arrowprops=dict(arrowstyle="-|>", color="white", lw=0.6),
                        )

            if not args.no_box:
                ax.add_patch(
                    mpatches.Rectangle((0, 0), L, L, fill=False, edgecolor="black", linewidth=1.5)
                )

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")
            ax.set_axis_off()

            if not args.no_labels:
                if args.title is not None:
                    title = args.title
                else:
                    s = traj.step(fidx)
                    t = traj.time(fidx)
                    title = f"frame {fidx}   step {s}   t = {t:.4g}"
                ax.set_title(title, fontsize=10)

            writer.grab_frame()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)

    with BDTrajectory(args.h5_path) as traj:
        _validate_color_by(traj, args.color_by)
        frames = _resolve_frame_list(traj, args)

        color_arrays = _compute_color_arrays(traj, frames, args.color_by, args.cluster_threshold)
        vmin, vmax = _color_range(args.color_by, color_arrays, args.vmin, args.vmax)
        cmap_name = args.cmap or _DEFAULT_CMAP[args.color_by]

        out_path = _resolve_output_path(args)
        render_movie(traj, frames, color_arrays, args.color_by, cmap_name, vmin, vmax, args, out_path)

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
