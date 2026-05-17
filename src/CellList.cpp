#include "CellList.hpp"

#include "Box.hpp"
#include "System.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

CellList::CellList(double r_cut, double r_skin, std::size_t N, const Box& box) {
    setup(r_cut, r_skin, N, box);
}

void CellList::setup(double r_cut, double r_skin, std::size_t N, const Box& box) {
    r_cut_      = r_cut;
    r_skin_     = r_skin;
    r_verlet_   = r_cut_ + r_skin_;
    r_verlet2_  = r_verlet_ * r_verlet_;
    half_skin2_ = 0.25 * r_skin_ * r_skin_;
    N_          = N;

    // Choose nx, ny so that an integer number of cells exactly tiles the box.
    // The resulting cell side is >= r_verlet_, which is exactly what the 3x3
    // stencil needs. The alternative (cell_size == r_verlet_ exactly, with a
    // non-integer cell count) would break the indexing scheme.
    const double Lx = box.getLx();
    const double Ly = box.getLy();
    nx_ = std::max<std::int32_t>(1, static_cast<std::int32_t>(std::floor(Lx / r_verlet_)));
    ny_ = std::max<std::int32_t>(1, static_cast<std::int32_t>(std::floor(Ly / r_verlet_)));

    cell_size_x_ = Lx / static_cast<double>(nx_);
    cell_size_y_ = Ly / static_cast<double>(ny_);
    inv_cell_x_  = 1.0 / cell_size_x_;
    inv_cell_y_  = 1.0 / cell_size_y_;

    // With fewer than 3 cells along an axis, the 3x3 PBC stencil revisits the
    // same cell via wraparound and would list the same neighbor multiple
    // times. Rather than complicating the stencil walk (which won't translate
    // cleanly to CUDA), drop into brute-force mode and let ForceCalculator's
    // O(N^2) overload do the work.
    use_brute_force_ = (nx_ < 3 || ny_ < 3);
    if (use_brute_force_) {
        std::cerr << "CellList: box too small for 3x3 stencil (nx=" << nx_
                  << ", ny=" << ny_ << "); falling back to brute-force forces.\n";
    }

    // Allocate. assign() zero-initializes and resizes in one pass; for vectors
    // that get fully overwritten on every rebuild (cell_id_, sorted_*) the
    // initial values don't matter, but we keep them deterministic.
    cell_id_.assign(N_, 0);
    sorted_particle_ids_.assign(N_, 0);
    sorted_cell_id_.assign(N_, 0);

    const std::size_t numCells = static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_);
    cell_start_.assign(numCells, -1);
    cell_end_.assign(numCells, -1);

    nlist_start_.assign(N_ + 1, 0);
    nlist_.clear();
    // Heuristic reservation: in 2D at moderate density a particle has O(10)
    // neighbors within r_verlet. A factor of 12 keeps the first few rebuilds
    // from reallocating, without burning much memory. On GPU this becomes a
    // hard cap with overflow detection; on CPU std::vector grows as needed.
    nlist_.reserve(static_cast<std::size_t>(12) * N_);

    // Empty snapshot => needsRebuild() returns true on its first call.
    x_snapshot_.clear();
    y_snapshot_.clear();

    num_rebuilds_ = 0;
}

bool CellList::needsRebuild(const System& sys, const Box& box) const {
    if (use_brute_force_) return false;
    if (x_snapshot_.empty()) return true;          // never built yet

    const double* __restrict__ x  = sys.xData();
    const double* __restrict__ y  = sys.yData();
    const double* __restrict__ xs = x_snapshot_.data();
    const double* __restrict__ ys = y_snapshot_.data();
    const double thresh2 = half_skin2_;

    // Per-particle predicate, with PBC: a particle that wrapped through a
    // boundary appears displaced by ~L if we don't apply minimumImage.
    // On GPU this is a per-thread predicate followed by thrust::any_of (or a
    // ballot reduction). The CPU early-exit is a strict superset of that
    // semantics — once any particle is over threshold, the answer is fixed.
    for (std::size_t i = 0; i < N_; ++i) {
        double dx = x[i] - xs[i];
        double dy = y[i] - ys[i];
        box.minimumImage(dx, dy);
        if (dx * dx + dy * dy > thresh2) return true;
    }
    return false;
}

void CellList::rebuild(const System& sys, const Box& box) {
    if (use_brute_force_) return;

    computeCellIds(sys);
    sortByCellId();
    findCellBoundaries();
    buildVerletList(sys, box);
    captureSnapshot(sys);                          // snapshot AT THE END

    ++num_rebuilds_;
}

// =============================================================================
// Stage 1 — computeCellIds
// -----------------------------------------------------------------------------
// One independent operation per particle. Becomes one CUDA thread per i.
// Stored positions are NOT wrapped by the integrator (see Box.hpp); cellOf()
// applies a floor-mod so any out-of-box position is mapped to its primary
// image cell. The defensive edge clamp inside cellOf() guards against the
// floating-point case where x[i] == Lx - eps rounds up to nx_.
// =============================================================================
void CellList::computeCellIds(const System& sys) {
    const double* __restrict__ x   = sys.xData();
    const double* __restrict__ y   = sys.yData();
    std::int32_t* __restrict__ cid = cell_id_.data();
    const std::int32_t nx = nx_;

    for (std::size_t i = 0; i < N_; ++i) {
        std::int32_t cx, cy;
        cellOf(x[i], y[i], cx, cy);
        cid[i] = cx + nx * cy;
    }
}

// =============================================================================
// Stage 2 — sortByCellId
// -----------------------------------------------------------------------------
// Co-sort (cell_id, particle_id) by cell_id. We avoid building an explicit
// vector of pairs by sorting a permutation of [0, N) using cell_id_[a] as the
// key. After sorting, sorted_particle_ids_[k] is the original particle index
// that ends up in slot k. We then materialize sorted_cell_id_ as a parallel
// array for stage 3 (this corresponds to the "second half" of thrust's
// sort_by_key, which produces both arrays together).
//
// CUDA: thrust::sequence(sorted_particle_ids_) +
//       thrust::sort_by_key(cell_id_, sorted_particle_ids_).
//
// Cannot fuse with stage 1: a global rearrangement requires the full
// cell_id_ array realized.
// =============================================================================
void CellList::sortByCellId() {
    std::iota(sorted_particle_ids_.begin(), sorted_particle_ids_.end(), 0);

    const std::int32_t* cid = cell_id_.data();
    std::sort(sorted_particle_ids_.begin(), sorted_particle_ids_.end(),
              [cid](std::int32_t a, std::int32_t b) {
                  return cid[a] < cid[b];
              });

    for (std::size_t k = 0; k < N_; ++k) {
        sorted_cell_id_[k] = cid[sorted_particle_ids_[k]];
    }
}

// =============================================================================
// Stage 3 — findCellBoundaries
// -----------------------------------------------------------------------------
// Walk the sorted cell_ids; whenever the value changes, we've crossed a cell
// boundary. The cell that just ended gets cell_end_[prev] = k, and the cell
// that just started gets cell_start_[curr] = k. Empty cells keep their -1
// sentinel from setup(), which the stencil walk in stage 4 uses to skip them.
//
// CUDA: one thread per sorted slot k, each thread writes at most one
// cell_end_[prev] and one cell_start_[curr] slot. Atomic-free because each
// boundary is touched by exactly one thread.
// =============================================================================
void CellList::findCellBoundaries() {
    std::fill(cell_start_.begin(), cell_start_.end(), -1);
    std::fill(cell_end_.begin(),   cell_end_.end(),   -1);
    if (N_ == 0) return;

    const std::int32_t* scid = sorted_cell_id_.data();

    cell_start_[scid[0]] = 0;
    for (std::size_t k = 1; k < N_; ++k) {
        if (scid[k] != scid[k - 1]) {
            cell_end_[scid[k - 1]] = static_cast<std::int32_t>(k);
            cell_start_[scid[k]]   = static_cast<std::int32_t>(k);
        }
    }
    cell_end_[scid[N_ - 1]] = static_cast<std::int32_t>(N_);
}

// =============================================================================
// Stage 4 — buildVerletList
// -----------------------------------------------------------------------------
// Three passes — count, exclusive scan, fill — each independent per particle.
// This is the canonical GPU-friendly pattern: pass A and C are 1-thread-per-i
// kernels with no atomics; pass B is thrust::exclusive_scan.
//
// Stencil walk: 3x3 cells in 2D, including the home cell. PBC is handled by
// (cx + dx + nx_) % nx_ — modular arithmetic, branch-free. We also still
// apply Box::minimumImage to the separation vector because the cell wrap
// only handles the indexing; the geometry of a periodic image vector still
// has to be resolved before a distance test.
//
// Self-exclusion (i == j) is done HERE so the force loop in
// ForceCalculator::compute(System&, const Box&, const CellList&) doesn't
// need to re-check.
// =============================================================================
void CellList::buildVerletList(const System& sys, const Box& box) {
    const double* __restrict__ x = sys.xData();
    const double* __restrict__ y = sys.yData();

    const std::int32_t  nx = nx_;
    const std::int32_t  ny = ny_;
    const std::int32_t* cs = cell_start_.data();
    const std::int32_t* ce = cell_end_.data();
    const std::int32_t* sp = sorted_particle_ids_.data();

    // ---- Pass A: count neighbors per particle -------------------------------
    // nlist_start_[i+1] holds the count for particle i; index 0 stays zero.
    // After pass B's prefix sum it becomes the offset.
    std::fill(nlist_start_.begin(), nlist_start_.end(), 0);

    for (std::size_t i = 0; i < N_; ++i) {
        const double xi = x[i];
        const double yi = y[i];
        std::int32_t cx, cy;
        cellOf(xi, yi, cx, cy);

        std::int32_t count = 0;
        for (std::int32_t dy_off = -1; dy_off <= 1; ++dy_off) {
            const std::int32_t ny_ix = (cy + dy_off + ny) % ny;
            for (std::int32_t dx_off = -1; dx_off <= 1; ++dx_off) {
                const std::int32_t nx_ix = (cx + dx_off + nx) % nx;
                const std::int32_t c     = nx_ix + nx * ny_ix;
                const std::int32_t s     = cs[c];
                if (s < 0) continue;
                const std::int32_t e     = ce[c];
                for (std::int32_t k = s; k < e; ++k) {
                    const std::int32_t j = sp[k];
                    if (j == static_cast<std::int32_t>(i)) continue;
                    double rx = x[j] - xi;
                    double ry = y[j] - yi;
                    box.minimumImage(rx, ry);
                    if (rx * rx + ry * ry < r_verlet2_) ++count;
                }
            }
        }
        nlist_start_[i + 1] = count;
    }

    // ---- Pass B: exclusive prefix sum (sequential on CPU, scan on GPU) ------
    for (std::size_t i = 1; i <= N_; ++i) {
        nlist_start_[i] += nlist_start_[i - 1];
    }
    nlist_.resize(static_cast<std::size_t>(nlist_start_[N_]));

    // ---- Pass C: fill -------------------------------------------------------
    // Identical stencil walk; instead of incrementing a counter we write into
    // nlist_[w++] starting at nlist_start_[i]. On GPU each thread writes its
    // own contiguous range, so no atomics needed.
    for (std::size_t i = 0; i < N_; ++i) {
        const double xi = x[i];
        const double yi = y[i];
        std::int32_t cx, cy;
        cellOf(xi, yi, cx, cy);

        std::int32_t w = nlist_start_[i];
        for (std::int32_t dy_off = -1; dy_off <= 1; ++dy_off) {
            const std::int32_t ny_ix = (cy + dy_off + ny) % ny;
            for (std::int32_t dx_off = -1; dx_off <= 1; ++dx_off) {
                const std::int32_t nx_ix = (cx + dx_off + nx) % nx;
                const std::int32_t c     = nx_ix + nx * ny_ix;
                const std::int32_t s     = cs[c];
                if (s < 0) continue;
                const std::int32_t e     = ce[c];
                for (std::int32_t k = s; k < e; ++k) {
                    const std::int32_t j = sp[k];
                    if (j == static_cast<std::int32_t>(i)) continue;
                    double rx = x[j] - xi;
                    double ry = y[j] - yi;
                    box.minimumImage(rx, ry);
                    if (rx * rx + ry * ry < r_verlet2_) {
                        nlist_[w++] = j;
                    }
                }
            }
        }
    }
}

// =============================================================================
// captureSnapshot — record the positions used to build the current Verlet list
// =============================================================================
void CellList::captureSnapshot(const System& sys) {
    const double* x = sys.xData();
    const double* y = sys.yData();
    x_snapshot_.assign(x, x + N_);
    y_snapshot_.assign(y, y + N_);
}
