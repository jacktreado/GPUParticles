"""
clusters — connected-component statistics under PBC.

Build the PBC Delaunay neighbour graph, keep only edges shorter than
`link_factor * sigma` (minimum-image distance), union-find the components
(Newman-Ziff: path compression + union by rank). Per frame we report the
largest-cluster size / fraction, the number of components, the mean
component size, the fraction of particles in components above
`min_cluster_size`, and a histogram of component sizes against
N-independent log-spaced bin edges.

Optional per-particle component label is gated by `with_per_particle=True`.
"""

from __future__ import annotations

import numpy as np

from bdtrajectory import BDTrajectory

from ..io import stack_dataset
from ..registry import register_analysis
from ._pbc_delaunay import compute_pbc_delaunay_neighbors


# Hardcoded so all sweeps share the same axis regardless of N.
# Sizes above the last edge fall into the final (overflow) bin.
_CLUSTER_SIZE_BIN_EDGES = np.array(
    [
        1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384,
        512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384,
    ],
    dtype=np.float64,
)
_N_BINS = _CLUSTER_SIZE_BIN_EDGES.size - 1


@register_analysis(
    name="clusters",
    version=1,
    requires=("positions",),
    outputs={
        "largest_cluster_size":     {"shape": "(num_frames,)",         "dtype": "i8", "axes": ["frame"]},
        "largest_cluster_fraction": {"shape": "(num_frames,)",         "dtype": "f8", "axes": ["frame"]},
        "n_components_total":       {"shape": "(num_frames,)",         "dtype": "i8", "axes": ["frame"]},
        "n_clusters_above_min":     {"shape": "(num_frames,)",         "dtype": "i8", "axes": ["frame"]},
        "mean_cluster_size":        {"shape": "(num_frames,)",         "dtype": "f8", "axes": ["frame"]},
        "fraction_in_clusters":     {"shape": "(num_frames,)",         "dtype": "f8", "axes": ["frame"]},
        "cluster_size_bin_edges":   {"shape": "(n_bins+1,)",           "dtype": "f8", "axes": ["bin_edge"]},
        "cluster_size_hist":        {"shape": "(num_frames, n_bins)",  "dtype": "i8", "axes": ["frame", "bin"]},
        "times":                    {"shape": "(num_frames,)",         "dtype": "f8", "axes": ["frame"]},
        "cluster_label":            {"shape": "(num_frames, N)",       "dtype": "i8", "axes": ["frame", "particle"]},
    },
)
def analyze(
    traj: BDTrajectory,
    *,
    link_factor: float = 0.99,
    min_cluster_size: int = 4,
    with_per_particle: bool = False,
) -> dict[str, np.ndarray]:
    pos = stack_dataset(traj, "positions")  # (T, N, 2)
    if pos is None:
        raise RuntimeError("positions dataset is missing — cannot compute clusters")

    T, N, _ = pos.shape
    sigma = float(traj.sigma)
    r_link = float(link_factor) * sigma
    times = traj.all_times().astype(np.float64)

    largest_size = np.empty(T, dtype=np.int64)
    largest_frac = np.empty(T, dtype=np.float64)
    n_total = np.empty(T, dtype=np.int64)
    n_above = np.empty(T, dtype=np.int64)
    mean_size = np.empty(T, dtype=np.float64)
    frac_in = np.empty(T, dtype=np.float64)
    size_hist = np.zeros((T, _N_BINS), dtype=np.int64)
    if with_per_particle:
        labels = np.empty((T, N), dtype=np.int64)

    for k in range(T):
        Lx, Ly = traj.frame_box(k)
        labels_k, sizes_k = _cluster_one_frame(pos[k], Lx, Ly, r_link)

        largest_size[k] = int(sizes_k.max())
        largest_frac[k] = float(largest_size[k]) / N
        n_total[k] = int(sizes_k.size)
        above = sizes_k >= int(min_cluster_size)
        n_above[k] = int(above.sum())
        mean_size[k] = float(sizes_k.mean())
        frac_in[k] = float(sizes_k[above].sum()) / N
        size_hist[k] = _hist_with_overflow(sizes_k, _CLUSTER_SIZE_BIN_EDGES)
        if with_per_particle:
            labels[k] = labels_k

    out: dict[str, np.ndarray] = {
        "largest_cluster_size": largest_size,
        "largest_cluster_fraction": largest_frac,
        "n_components_total": n_total,
        "n_clusters_above_min": n_above,
        "mean_cluster_size": mean_size,
        "fraction_in_clusters": frac_in,
        "cluster_size_bin_edges": _CLUSTER_SIZE_BIN_EDGES.copy(),
        "cluster_size_hist": size_hist,
        "times": times,
    }
    if with_per_particle:
        out["cluster_label"] = labels
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cluster_one_frame(
    pos: np.ndarray, Lx: float, Ly: float, r_link: float
) -> tuple[np.ndarray, np.ndarray]:
    """Union-find over PBC Delaunay edges shorter than r_link.

    Returns (labels, cluster_sizes) where labels is shape (N,) with values
    in [0, n_clusters) and cluster_sizes[c] is the size of cluster c.
    """
    N = pos.shape[0]
    x_ctr = np.mod(pos[:, 0], Lx)
    y_ctr = np.mod(pos[:, 1], Ly)
    centered = np.stack([x_ctr, y_ctr], axis=1)
    neighbors = compute_pbc_delaunay_neighbors(centered, Lx, Ly)

    parent = np.arange(N, dtype=np.int64)
    rank = np.zeros(N, dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    r_link_sq = r_link * r_link
    for i in range(N):
        for j in neighbors[i]:
            if j <= i:
                continue
            dx = centered[j, 0] - centered[i, 0]
            dy = centered[j, 1] - centered[i, 1]
            dx -= Lx * round(dx / Lx)
            dy -= Ly * round(dy / Ly)
            if dx * dx + dy * dy < r_link_sq:
                union(i, j)

    root_to_label: dict[int, int] = {}
    labels = np.empty(N, dtype=np.int64)
    next_label = 0
    for i in range(N):
        r = find(i)
        if r not in root_to_label:
            root_to_label[r] = next_label
            next_label += 1
        labels[i] = root_to_label[r]

    sizes = np.bincount(labels, minlength=next_label).astype(np.int64)
    return labels, sizes


def _hist_with_overflow(sizes: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Histogram of integer cluster sizes against `edges`, with overflow into the last bin."""
    n_bins = edges.size - 1
    out = np.zeros(n_bins, dtype=np.int64)
    if sizes.size == 0:
        return out
    # np.digitize returns indices in [0, len(edges)]; bin index = digitize-1.
    idx = np.digitize(sizes, edges) - 1
    # Anything >= edges[-1] folds into the last bin (overflow).
    idx = np.clip(idx, 0, n_bins - 1)
    np.add.at(out, idx, 1)
    return out
