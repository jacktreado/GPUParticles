#include "Box.hpp"
#include "CellList.hpp"
#include "ForceCalculator.hpp"
#include "System.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <set>
#include <vector>

// =============================================================================
// CellList unit tests
// =============================================================================

// ---- Setup / geometry -------------------------------------------------------

TEST(CellListTest, GridDimensionsFloorDivision) {
    // nx = floor(Lx / (r_cut + r_skin))
    const double r_cut  = std::pow(2.0, 1.0 / 6.0);  // ~1.122
    const double r_skin = 0.2;
    const double L      = 10.0;
    const std::size_t N = 4;

    Box box(L);
    CellList cl(r_cut, r_skin, N, box);

    const std::int32_t expected_n = static_cast<std::int32_t>(std::floor(L / (r_cut + r_skin)));
    EXPECT_EQ(cl.getNx(), expected_n);
    EXPECT_EQ(cl.getNy(), expected_n);
}

TEST(CellListTest, CellSizeCoversTilesBox) {
    const double r_cut = std::pow(2.0, 1.0 / 6.0);
    const double r_skin = 0.2;
    const double L = 10.0;
    Box box(L);
    CellList cl(r_cut, r_skin, 4, box);

    // Cell grid must tile the box exactly.
    EXPECT_DOUBLE_EQ(cl.getCellSizeX() * cl.getNx(), L);
    EXPECT_DOUBLE_EQ(cl.getCellSizeY() * cl.getNy(), L);
}

TEST(CellListTest, CellSizeAtLeastRverlet) {
    const double r_cut = std::pow(2.0, 1.0 / 6.0);
    const double r_skin = 0.2;
    Box box(10.0);
    CellList cl(r_cut, r_skin, 4, box);

    // Each cell side must cover at least one interaction radius.
    EXPECT_GE(cl.getCellSizeX(), cl.getRverlet());
    EXPECT_GE(cl.getCellSizeY(), cl.getRverlet());
}

TEST(CellListTest, RverletEqualsRcutPlusRskin) {
    const double r_cut = 1.0, r_skin = 0.3;
    Box box(20.0);
    CellList cl(r_cut, r_skin, 4, box);
    EXPECT_DOUBLE_EQ(cl.getRverlet(), r_cut + r_skin);
}

// ---- Small-box fallback -----------------------------------------------------

TEST(CellListTest, SmallBoxTriggersBruteForce) {
    // A box smaller than 3*(r_cut+r_skin) in at least one axis.
    const double r_cut  = std::pow(2.0, 1.0 / 6.0);
    const double r_skin = 0.2;
    const double tiny_L = 2.0 * (r_cut + r_skin);    // < 3 * r_verlet
    Box box(tiny_L);
    CellList cl(r_cut, r_skin, 2, box);

    EXPECT_TRUE(cl.useBruteForce());
}

TEST(CellListTest, NormalBoxDoesNotTriggerBruteForce) {
    const double r_cut  = std::pow(2.0, 1.0 / 6.0);
    const double r_skin = 0.2;
    const double L      = 20.0;   // well above 3 * r_verlet
    Box box(L);
    CellList cl(r_cut, r_skin, 16, box);

    EXPECT_FALSE(cl.useBruteForce());
}

TEST(CellListTest, SmallBoxRebuildIsNoop) {
    // rebuild() must not crash and must keep useBruteForce() true.
    const double r_cut  = std::pow(2.0, 1.0 / 6.0);
    const double r_skin = 0.2;
    const double tiny_L = 2.0 * (r_cut + r_skin);
    Box box(tiny_L);
    System sys(2);
    sys.setPosition(0, 0.5, 0.5);
    sys.setPosition(1, 1.0, 1.0);
    CellList cl(r_cut, r_skin, 2, box);

    cl.rebuild(sys, box);    // must not crash
    EXPECT_TRUE(cl.useBruteForce());
    EXPECT_EQ(cl.getNumRebuilds(), std::size_t{0});   // no-op, so no counter
}

// ---- needsRebuild -----------------------------------------------------------

TEST(CellListTest, NeedsRebuildTrueBeforeFirstRebuild) {
    const double r_cut = std::pow(2.0, 1.0 / 6.0);
    Box box(20.0);
    System sys(4);
    sys.setPosition(0, 1.0, 1.0);
    CellList cl(r_cut, 0.2, 4, box);

    EXPECT_TRUE(cl.needsRebuild(sys, box));
}

TEST(CellListTest, NeedsRebuildFalseAfterRebuildNoMotion) {
    const double r_cut = std::pow(2.0, 1.0 / 6.0);
    Box box(20.0);
    System sys(4);
    for (std::size_t i = 0; i < 4; ++i) sys.setPosition(i, double(i) * 4.0 + 1.0, 1.0);

    CellList cl(r_cut, 0.2, 4, box);
    cl.rebuild(sys, box);

    EXPECT_FALSE(cl.needsRebuild(sys, box));
}

TEST(CellListTest, NeedsRebuildTrueAfterLargeDisplacement) {
    const double r_cut = std::pow(2.0, 1.0 / 6.0);
    const double r_skin = 0.4;
    Box box(20.0);
    System sys(4);
    for (std::size_t i = 0; i < 4; ++i) sys.setPosition(i, double(i) * 4.0 + 1.0, 1.0);

    CellList cl(r_cut, r_skin, 4, box);
    cl.rebuild(sys, box);

    // Move particle 2 by more than r_skin/2 = 0.2.
    sys.setX(2, sys.getX(2) + 0.25);

    EXPECT_TRUE(cl.needsRebuild(sys, box));
}

TEST(CellListTest, NeedsRebuildFalseAfterSmallDisplacement) {
    const double r_cut = std::pow(2.0, 1.0 / 6.0);
    const double r_skin = 0.4;
    Box box(20.0);
    System sys(4);
    for (std::size_t i = 0; i < 4; ++i) sys.setPosition(i, double(i) * 4.0 + 1.0, 1.0);

    CellList cl(r_cut, r_skin, 4, box);
    cl.rebuild(sys, box);

    // Move particle 0 by less than r_skin/2 = 0.2.
    sys.setX(0, sys.getX(0) + 0.1);

    EXPECT_FALSE(cl.needsRebuild(sys, box));
}

TEST(CellListTest, NeedsRebuildPBCAware) {
    // Particle near the right boundary wraps to the left: the raw
    // displacement looks large (~L) but the PBC displacement is small.
    const double r_cut  = std::pow(2.0, 1.0 / 6.0);
    const double r_skin = 0.4;
    const double L      = 20.0;
    Box box(L);

    System sys(2);
    // Place particle 0 just inside the right boundary.
    sys.setPosition(0, L - 0.05, 5.0);
    sys.setPosition(1, 5.0, 5.0);

    CellList cl(r_cut, r_skin, 2, box);
    cl.rebuild(sys, box);

    // Simulate wrap: particle 0 moves slightly to the right, wraps to ~0.05.
    sys.setX(0, 0.05);   // ~0.1 total displacement (crosses boundary)

    // 0.1 < r_skin/2 = 0.2: should NOT trigger rebuild.
    EXPECT_FALSE(cl.needsRebuild(sys, box));
}

// ---- rebuild counter --------------------------------------------------------

TEST(CellListTest, NumRebuildsIncrementsOnRebuild) {
    const double r_cut = std::pow(2.0, 1.0 / 6.0);
    Box box(20.0);
    System sys(4);
    for (std::size_t i = 0; i < 4; ++i) sys.setPosition(i, double(i) * 4.0 + 1.0, 1.0);

    CellList cl(r_cut, 0.2, 4, box);
    EXPECT_EQ(cl.getNumRebuilds(), std::size_t{0});

    cl.rebuild(sys, box);
    EXPECT_EQ(cl.getNumRebuilds(), std::size_t{1});

    cl.rebuild(sys, box);
    EXPECT_EQ(cl.getNumRebuilds(), std::size_t{2});
}

// ---- Neighbor list correctness ----------------------------------------------

// Build a system with known positions and verify that every pair within r_verlet
// appears in each other's neighbor list, and no pair outside r_verlet appears.
TEST(CellListTest, NeighborListMatchesBruteForce) {
    const double r_cut  = std::pow(2.0, 1.0 / 6.0);
    const double r_skin = 0.5;   // generous skin so we can test easily
    const double r_v    = r_cut + r_skin;
    const double L      = 20.0;
    const std::size_t N = 16;

    Box box(L);
    System sys(N);
    // Place particles on a sparse grid.
    for (std::size_t i = 0; i < N; ++i) {
        sys.setPosition(i,
            (static_cast<double>(i % 4) + 0.5) * (L / 4.0),
            (static_cast<double>(i / 4) + 0.5) * (L / 4.0));
    }

    CellList cl(r_cut, r_skin, N, box);
    cl.rebuild(sys, box);

    const std::int32_t* nlist       = cl.nlistData();
    const std::int32_t* nlist_start = cl.nlistStartData();

    for (std::size_t i = 0; i < N; ++i) {
        // Collect cell-list neighbors of i.
        std::set<int> cl_neighbors;
        for (std::int32_t k = nlist_start[i]; k < nlist_start[i + 1]; ++k) {
            cl_neighbors.insert(nlist[k]);
        }

        // Brute-force: all j != i within r_verlet.
        for (std::size_t j = 0; j < N; ++j) {
            if (j == i) continue;
            double dx = sys.getX(j) - sys.getX(i);
            double dy = sys.getY(j) - sys.getY(i);
            box.minimumImage(dx, dy);
            const double r2 = dx * dx + dy * dy;
            const bool within = r2 < r_v * r_v;

            if (within) {
                EXPECT_TRUE(cl_neighbors.count(static_cast<int>(j))) << "i=" << i << " j=" << j;
            } else {
                EXPECT_FALSE(cl_neighbors.count(static_cast<int>(j))) << "i=" << i << " j=" << j;
            }
        }

        // Self must not be in the list.
        EXPECT_EQ(cl_neighbors.count(static_cast<int>(i)), std::size_t{0});
    }
}

// ---- Neighbor list across periodic boundary ---------------------------------

TEST(CellListTest, NeighborAcrossPeriodicBoundary) {
    // Two particles: one near left wall, one near right wall.
    // They are separated by 0.2 through PBC, well within r_verlet.
    const double r_cut  = std::pow(2.0, 1.0 / 6.0);
    const double r_skin = 0.4;
    const double L      = 10.0;
    const double g      = 0.1;

    Box box(L);
    System sys(2);
    sys.setPosition(0, g,       5.0);
    sys.setPosition(1, L - g,   5.0);

    CellList cl(r_cut, r_skin, 2, box);
    cl.rebuild(sys, box);

    const std::int32_t* nlist       = cl.nlistData();
    const std::int32_t* nlist_start = cl.nlistStartData();

    // Particle 0 must see particle 1 and vice versa.
    bool zero_sees_one = false, one_sees_zero = false;
    for (std::int32_t k = nlist_start[0]; k < nlist_start[1]; ++k) {
        if (nlist[k] == 1) zero_sees_one = true;
    }
    for (std::int32_t k = nlist_start[1]; k < nlist_start[2]; ++k) {
        if (nlist[k] == 0) one_sees_zero = true;
    }

    EXPECT_TRUE(zero_sees_one) << "particle 0 should see particle 1 across PBC";
    EXPECT_TRUE(one_sees_zero) << "particle 1 should see particle 0 across PBC";
}

// ---- Out-of-box positions ---------------------------------------------------

// After the integrator stopped wrapping stored positions, particles can drift
// outside [0, L). The cell list must still produce the same Verlet list as
// the equivalent in-box configuration.
TEST(CellListTest, NeighborListInvariantUnderImageShift) {
    const double r_cut  = std::pow(2.0, 1.0 / 6.0);
    const double r_skin = 0.5;
    const double L      = 20.0;
    const std::size_t N = 16;

    Box box(L);

    // Reference: all positions in [0, L).
    System sys_in(N);
    for (std::size_t i = 0; i < N; ++i) {
        sys_in.setPosition(i,
            (static_cast<double>(i % 4) + 0.5) * (L / 4.0),
            (static_cast<double>(i / 4) + 0.5) * (L / 4.0));
    }
    CellList cl_in(r_cut, r_skin, N, box);
    cl_in.rebuild(sys_in, box);

    // Shifted: shove each particle by a different integer multiple of (Lx, Ly)
    // so positions land far outside [0, L) (some negative, some > L).
    System sys_out(N);
    for (std::size_t i = 0; i < N; ++i) {
        const int kx = (static_cast<int>(i) % 3) - 1;  // -1, 0, 1, -1, 0, 1, ...
        const int ky = (static_cast<int>(i) / 3) - 2;  // a mix of negative and positive
        sys_out.setPosition(i,
            sys_in.getX(i) + kx * L,
            sys_in.getY(i) + ky * L);
    }
    CellList cl_out(r_cut, r_skin, N, box);
    cl_out.rebuild(sys_out, box);

    // Per-particle Verlet lists must match as sets.
    const std::int32_t* nlist_in        = cl_in.nlistData();
    const std::int32_t* nlist_start_in  = cl_in.nlistStartData();
    const std::int32_t* nlist_out       = cl_out.nlistData();
    const std::int32_t* nlist_start_out = cl_out.nlistStartData();

    for (std::size_t i = 0; i < N; ++i) {
        std::set<std::int32_t> set_in, set_out;
        for (std::int32_t k = nlist_start_in[i];  k < nlist_start_in[i + 1];  ++k)
            set_in.insert(nlist_in[k]);
        for (std::int32_t k = nlist_start_out[i]; k < nlist_start_out[i + 1]; ++k)
            set_out.insert(nlist_out[k]);
        EXPECT_EQ(set_in, set_out) << "neighbor sets diverge for i=" << i;
    }
}

// ---- Mean neighbors / diagnostics -------------------------------------------

TEST(CellListTest, DiagnosticsAfterRebuild) {
    const double r_cut  = std::pow(2.0, 1.0 / 6.0);
    const double r_skin = 0.2;
    const std::size_t N = 16;
    Box box(20.0);
    System sys(N);
    for (std::size_t i = 0; i < N; ++i)
        sys.setPosition(i, (static_cast<double>(i % 4) + 0.5) * 5.0,
                            (static_cast<double>(i / 4) + 0.5) * 5.0);

    CellList cl(r_cut, r_skin, N, box);
    cl.rebuild(sys, box);

    EXPECT_EQ(cl.getNumParticles(), N);
    EXPECT_GE(cl.getMeanNeighbors(), 0.0);
    EXPECT_EQ(cl.getTotalNeighborEntries(), static_cast<std::size_t>(cl.nlistStartData()[N]));
}
