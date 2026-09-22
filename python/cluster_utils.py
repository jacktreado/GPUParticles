import numpy as np
from scipy.spatial import Delaunay

class ClusterUtils:
    @staticmethod
    def compute_pbc_delaunay_neighbors(x, y, box_length=1.0):
        # first, recenter the points in the box to [0, box_length)
        x_ctr = np.mod(x, box_length)
        y_ctr = np.mod(y, box_length)
        N = len(x)
        
        # replicate points in neighboring boxes for Delaunay triangulation
        pts = np.stack([x_ctr, y_ctr], axis=1)
        for xx in np.arange(-1, 2):
            for yy in np.arange(-1, 2):
                if xx == 0 and yy == 0:
                    continue
                pts = np.append(pts, np.stack([x_ctr + xx * box_length, y_ctr + yy * box_length], axis=1), axis=0)
        tri_orig = Delaunay(pts)
        
        # loop over the neighbors of the first N points (the original points). If a neighbor is in
        # the original set, add to the list of neighbors for that particle. If not, find the 
        # image in the middle box and add that as a neighbor.
        neighbors = [[] for _ in range(N)]
        for i in range(N):
            for j in tri_orig.vertex_neighbor_vertices[1][tri_orig.vertex_neighbor_vertices[0][i]:tri_orig.vertex_neighbor_vertices[0][i+1]]:
                if j < N:
                    neighbors[i].append(j)
                else:
                    # find the image in the middle box
                    j_mod = j % N
                    neighbors[i].append(j_mod)

        
        return neighbors
    
    @staticmethod
    def compute_hexatic_order(x, y, box_length=1.0):
        # compute hexatic order parameter for each particle
        N = len(x)
        x_ctr = np.mod(x, box_length)
        y_ctr = np.mod(y, box_length)
        
        neighbors = ClusterUtils.compute_pbc_delaunay_neighbors(x_ctr, y_ctr, box_length)
        
        psi6 = np.zeros(N, dtype=complex)
        for i in range(N):
            for j in neighbors[i]:
                dx = x_ctr[j] - x_ctr[i]
                dy = y_ctr[j] - y_ctr[i]
                # Minimum image convention
                dx -= box_length * np.round(dx / box_length)
                dy -= box_length * np.round(dy / box_length)
                angle = np.arctan2(dy, dx)
                psi6[i] += np.exp(1j * 6 * angle)
            if len(neighbors[i]) > 0:
                psi6[i] /= len(neighbors[i])
        
        return psi6
    
    @staticmethod
    def find_clusters(x, y, sigma, box_length=1.0):
        """
        Find connected clusters of particles using Newman-Ziff algorithm.
        
        Parameters:
            x, y: 1D arrays of particle positions
            sigma: distance threshold for connectivity
            box_length: side length of the periodic box (assume square, default 1.0)
        
        Returns:
            labels: array of cluster labels for each particle
            cluster_sizes: array of cluster sizes (indexed by cluster label)
        """
        N = len(x)
        
        # center positions in box
        x_ctr = np.mod(x, box_length)
        y_ctr = np.mod(y, box_length)
        
        # Union-Find data structure
        parent = np.arange(N, dtype=int)
        rank = np.zeros(N, dtype=int)
        
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]  # path compression
                a = parent[a]
            return a
        
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
        
        # Use Delaunay triangulation to avoid O(N^2) brute force (optional optimization)
        neighbors = ClusterUtils.compute_pbc_delaunay_neighbors(x_ctr, y_ctr, box_length)
        
        # loop over neighbors and union if within distance
        for i in range(N):
            for j in neighbors[i]:
                if j > i:  # avoid double counting
                    dx = x_ctr[j] - x_ctr[i]
                    dy = y_ctr[j] - y_ctr[i]
                    # Minimum image convention
                    dx -= box_length * np.round(dx / box_length)
                    dy -= box_length * np.round(dy / box_length)
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < sigma:
                        union(i, j)
        
        # Assign cluster labels
        root_to_label = {}
        labels = np.zeros(N, dtype=int)
        current_label = 0
        for i in range(N):
            root = find(i)
            if root not in root_to_label:
                root_to_label[root] = current_label
                current_label += 1
            labels[i] = root_to_label[root]
        
        # Compute cluster sizes
        num_clusters = current_label
        cluster_sizes = np.zeros(num_clusters, dtype=int)
        for i in range(N):
            cluster_sizes[labels[i]] += 1
        
        return labels, cluster_sizes

    @staticmethod
    def local_voronoi_phi(x, y, sigma, box_length=1.0):
        """
        Compute the local Voronoi packing fraction phi_i = pi*(sigma/2)^2 / A_i
        for each particle, using a 9-image periodic tiling so that cells near
        the box edges are computed correctly.

        Parameters:
            x, y: 1D arrays of particle positions (need not be pre-wrapped)
            sigma: particle diameter (assumes monodisperse particles)
            box_length: side length of the periodic box (assume square, default 1.0)

        Returns:
            phi: array of per-particle local packing fractions, shape (N,).
                 NaN for particles whose Voronoi cell is unbounded/degenerate.
        """
        from scipy.spatial import Voronoi

        N = len(x)
        x_ctr = np.mod(x, box_length)
        y_ctr = np.mod(y, box_length)
        pos = np.stack([x_ctr, y_ctr], axis=1)

        # Same 3x3 tiling convention as compute_pbc_delaunay_neighbors: the
        # unshifted (0, 0) block is placed first, so rows [0:N] of `images`
        # are always the true central copies of the particles.
        shifts = [(0, 0)]
        for xx in np.arange(-1, 2):
            for yy in np.arange(-1, 2):
                if xx == 0 and yy == 0:
                    continue
                shifts.append((xx, yy))
        shifts = np.array(shifts, dtype=float) * box_length

        images = (pos[None, :, :] + shifts[:, None, :]).reshape(-1, 2)

        vor = Voronoi(images)
        particle_area = np.pi * (sigma / 2.0) ** 2
        phi = np.empty(N, dtype=np.float64)
        for i in range(N):
            region = vor.regions[vor.point_region[i]]
            if not region or -1 in region or len(region) < 3:
                phi[i] = np.nan
                continue
            v = vor.vertices[region]
            vx, vy = v[:, 0], v[:, 1]
            area = 0.5 * np.abs(np.dot(vx, np.roll(vy, 1)) - np.dot(vy, np.roll(vx, 1)))
            phi[i] = particle_area / area
        return phi
