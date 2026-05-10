"""
_pbc_delaunay — shared helper: Delaunay neighbour lists under 2-D periodic BCs.

Tile the central cell to its 8 neighbours, Delaunay-triangulate the 9N point
cloud, then for each particle in the central tile collect the simplex
neighbours and remap any image-tile indices via `% N`. Returns one
deduplicated neighbour list per particle.

Mirrors the 9-image trick used by `_local_voronoi_phi` in force_orientation.py,
but generalised to rectangular boxes (Lx, Ly).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay


def compute_pbc_delaunay_neighbors(
    positions: np.ndarray,
    Lx: float,
    Ly: float,
) -> list[list[int]]:
    """
    Parameters
    ----------
    positions : (N, 2) array
        Particle positions. Wrapped into [0, Lx) x [0, Ly) internally.
    Lx, Ly : float
        Periodic-cell side lengths.

    Returns
    -------
    neighbors : list of length N
        `neighbors[i]` is a sorted list of unique central-tile indices that
        share a Delaunay edge with particle `i` under PBC.
    """
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError(f"positions must be (N, 2); got {pos.shape}")
    N = pos.shape[0]

    x_ctr = np.mod(pos[:, 0], Lx)
    y_ctr = np.mod(pos[:, 1], Ly)
    central = np.stack([x_ctr, y_ctr], axis=1)

    # Central tile first so its vertex indices in the tiled cloud are [0, N).
    shifts = np.array(
        [(0.0, 0.0)]
        + [(dx * Lx, dy * Ly) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
    )
    images = (central[None, :, :] + shifts[:, None, :]).reshape(-1, 2)

    tri = Delaunay(images)
    indptr, indices = tri.vertex_neighbor_vertices

    neighbors: list[list[int]] = [[] for _ in range(N)]
    seen: list[set[int]] = [set() for _ in range(N)]
    for i in range(N):
        for j in indices[indptr[i]:indptr[i + 1]]:
            j_mod = int(j) % N
            if j_mod == i or j_mod in seen[i]:
                continue
            seen[i].add(j_mod)
            neighbors[i].append(j_mod)
    for i in range(N):
        neighbors[i].sort()
    return neighbors
