#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

class System;
class Box;

// =============================================================================
// CellList
// -----------------------------------------------------------------------------
// Sort-based cell list + Verlet (skin) neighbor list for short-range pair
// interactions in a 2D periodic box. Replaces the O(N^2) inner loop in
// ForceCalculator with an O(N * <neighbors>) loop driven by a packed (CSR)
// neighbor list. The Verlet list is reused across many timesteps and rebuilt
// only when any particle has drifted by more than r_skin/2 since the last
// rebuild — at which point an outdated entry could in principle have crossed
// into the cutoff radius without us seeing it.
//
// PIPELINE
// --------
// Each stage is its own private method, no fusion across stages, because each
// one becomes a single CUDA kernel in the GPU port and they have hard data
// dependencies that survive that translation:
//
//   1. computeCellIds()      cell_id[i] = cx + nx_ * cy
//                            -> CUDA: one thread per particle, embarrassingly
//                               parallel.
//   2. sortByCellId()        co-sort (cell_id, particle_id) by cell_id.
//                            -> CUDA: thrust::sort_by_key on device. Cannot
//                               fuse with stage 1: a global rearrangement
//                               needs the whole cell_id_ array realized.
//   3. findCellBoundaries()  derive cell_start[c], cell_end[c] from the
//                            sorted cell_ids (boundary-finding pass).
//                            -> CUDA: one thread per sorted slot, atomic-free
//                               because each cell boundary is written by
//                               exactly one thread.
//   4. buildVerletList()     three passes — count per particle, exclusive
//                            prefix-sum into nlist_start_, fill nlist_ — by
//                            walking the 3x3 PBC stencil through cell_start /
//                            cell_end.
//                            -> CUDA: kernel A counts, thrust::exclusive_scan,
//                               kernel B fills. Atomic-free per particle.
//
// REBUILD TRIGGER
// ---------------
// needsRebuild() does an O(N) PBC-aware reduction over per-particle drift
// from the snapshot taken at the last rebuild. It applies Box::minimumImage()
// before the squared-distance test so particles that wrapped through a
// periodic boundary are handled correctly.
//
// On GPU this is a per-particle predicate followed by thrust::any_of (or a
// __ballot_sync ballot-and-or). The algebra is unchanged.
//
// SMALL-BOX FALLBACK
// ------------------
// When min(Lx, Ly) < 3*(r_cut + r_skin) the 3x3 PBC stencil double-counts
// neighbors via wraparound (the same cell appears twice in the stencil). To
// keep the GPU translation clean we *do not* try to dedupe in the stencil
// walk; instead setup() detects this and sets use_brute_force_. In that
// mode rebuild() is a no-op and the caller must use the brute-force
// ForceCalculator::compute(System&, const Box&) overload.
//
// LAYOUT / OWNERSHIP
// ------------------
// All bookkeeping arrays are flat std::vector<int32_t> (never nested
// vectors). The same handles are exposed as raw pointers via *Data()
// accessors so the GPU port can swap in device pointers with no API churn.
// int32 keeps memory traffic low (half of int64) and matches the index
// width that conventional CUDA codes assume. The Verlet list is CSR:
// nlist_start_[i]..nlist_start_[i+1] is the neighbor range of particle i.
// =============================================================================
class CellList {
public:
    CellList() = default;

    // Construct + allocate. Does NOT do an initial rebuild — call rebuild()
    // explicitly after the System is initialized but before the first force
    // evaluation.
    CellList(double r_cut, double r_skin, std::size_t N, const Box& box);

    // (Re)allocate against a new box / particle count. Recomputes geometry,
    // sets use_brute_force_ if the box is too small for the standard 3x3
    // stencil. Safe to call repeatedly.
    void setup(double r_cut, double r_skin, std::size_t N, const Box& box);

    // ---- Rebuild trigger (O(N) reduction) -----------------------------------
    // Returns true iff any particle has moved > r_skin/2 since the last
    // snapshot. PBC-aware: applies Box::minimumImage to (r_i - r_i^snapshot)
    // so particles that wrapped through a boundary are handled correctly.
    // Always returns true on the very first call (no snapshot yet).
    // In brute-force mode returns false (nothing to maintain).
    bool needsRebuild(const System& sys, const Box& box) const;

    // ---- Full rebuild: cell list + Verlet list + snapshot -------------------
    // Stages 1–4 in order, then captures the snapshot of current positions.
    // No-op in brute-force mode.
    void rebuild(const System& sys, const Box& box);

    // ---- Mode ---------------------------------------------------------------
    bool useBruteForce() const { return use_brute_force_; }

    // ---- Raw pointer accessors (mirror System::*Data()) ---------------------
    // The GPU on-ramp: when we add a CUDA backend these pointers can be
    // replaced by device pointers and the rest of the engine is untouched.
    const std::int32_t* cellIdData()              const { return cell_id_.data();             }
    const std::int32_t* sortedParticleIdsData()   const { return sorted_particle_ids_.data(); }
    const std::int32_t* cellStartData()           const { return cell_start_.data();          }
    const std::int32_t* cellEndData()             const { return cell_end_.data();            }
    const std::int32_t* nlistData()               const { return nlist_.data();               }
    const std::int32_t* nlistStartData()          const { return nlist_start_.data();         }

    // ---- Geometry / sizes ---------------------------------------------------
    std::size_t  getNumParticles() const { return N_;            }
    std::int32_t getNx()           const { return nx_;           }
    std::int32_t getNy()           const { return ny_;           }
    std::int32_t getNumCells()     const { return nx_ * ny_;     }
    double       getCellSizeX()    const { return cell_size_x_;  }
    double       getCellSizeY()    const { return cell_size_y_;  }
    double       getRcut()         const { return r_cut_;        }
    double       getRskin()        const { return r_skin_;       }
    double       getRverlet()      const { return r_verlet_;     }   // r_cut + r_skin

    // ---- Diagnostics --------------------------------------------------------
    std::size_t getNumRebuilds()         const { return num_rebuilds_;     }
    std::size_t getTotalNeighborEntries()const { return nlist_.size();     }
    double      getMeanNeighbors()       const {
        return N_ ? static_cast<double>(nlist_.size()) / static_cast<double>(N_) : 0.0;
    }

private:
    // ---- Geometry — set in setup(), constant until the next setup() --------
    double r_cut_      = 0.0;
    double r_skin_     = 0.0;
    double r_verlet_   = 0.0;          // r_cut + r_skin
    double r_verlet2_  = 0.0;          // (r_cut + r_skin)^2
    double half_skin2_ = 0.0;          // (r_skin / 2)^2

    std::int32_t nx_   = 0;            // cells along x
    std::int32_t ny_   = 0;            // cells along y
    double cell_size_x_ = 0.0;         // Lx / nx_  (>= r_verlet_)
    double cell_size_y_ = 0.0;         // Ly / ny_
    double inv_cell_x_  = 0.0;         // 1 / cell_size_x_
    double inv_cell_y_  = 0.0;

    std::size_t N_ = 0;
    bool use_brute_force_ = false;

    // ---- Cell-list bookkeeping (length-N or length-numCells) ---------------
    std::vector<std::int32_t> cell_id_;              // size N
    std::vector<std::int32_t> sorted_particle_ids_;  // size N
    // Parallel array of cell_ids in sorted order. Held as a member (not a
    // stack local) so rebuild() does zero allocation in steady state.
    std::vector<std::int32_t> sorted_cell_id_;       // size N
    std::vector<std::int32_t> cell_start_;           // size numCells, -1 if empty
    std::vector<std::int32_t> cell_end_;             // size numCells, exclusive end

    // ---- Verlet list (CSR layout) ------------------------------------------
    std::vector<std::int32_t> nlist_start_;          // size N+1, prefix sum
    std::vector<std::int32_t> nlist_;                // size = nlist_start_[N]

    // ---- Rebuild snapshot --------------------------------------------------
    std::vector<double> x_snapshot_;                 // size N
    std::vector<double> y_snapshot_;                 // size N

    // ---- Diagnostics -------------------------------------------------------
    std::size_t num_rebuilds_ = 0;

    // ---- Private pipeline stages (each becomes one CUDA kernel) -----------
    // Reads:  sys.x_, sys.y_                Writes: cell_id_
    void computeCellIds(const System& sys);

    // Reads:  cell_id_                      Writes: sorted_particle_ids_, sorted_cell_id_
    void sortByCellId();

    // Reads:  sorted_cell_id_               Writes: cell_start_, cell_end_
    void findCellBoundaries();

    // Reads:  sys.x_, sys.y_, cell_start_, cell_end_, sorted_particle_ids_
    // Writes: nlist_start_, nlist_
    void buildVerletList(const System& sys, const Box& box);

    // Reads sys.x_, sys.y_; writes x_snapshot_, y_snapshot_
    void captureSnapshot(const System& sys);

    // ---- PBC-safe integer cell index ---------------------------------------
    // Maps an absolute world position (which may lie outside [0, L)) to the
    // (cx, cy) cell index in [0, nx_) x [0, ny_). Uses floor-mod so that
    // negative or out-of-box positions are mapped correctly to their primary
    // image cell. Defensive clamps guard against the floating-point edge case
    // where the result rounds up to nx_ exactly. Branch-free except for the
    // edge clamp (CUDA-friendly for the eventual GPU port).
    inline void cellOf(double x, double y,
                       std::int32_t& cx, std::int32_t& cy) const {
        cx = static_cast<std::int32_t>(std::floor(x * inv_cell_x_));
        cy = static_cast<std::int32_t>(std::floor(y * inv_cell_y_));
        cx -= nx_ * static_cast<std::int32_t>(
                       std::floor(static_cast<double>(cx) / nx_));
        cy -= ny_ * static_cast<std::int32_t>(
                       std::floor(static_cast<double>(cy) / ny_));
        if (cx >= nx_) cx = nx_ - 1;
        if (cy >= ny_) cy = ny_ - 1;
        if (cx < 0)    cx = 0;
        if (cy < 0)    cy = 0;
    }
};
