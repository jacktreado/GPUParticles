#include "Box.hpp"
#include "CellList.hpp"
#include "ForceCalculator.hpp"
#include "System.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <cstddef>

// =============================================================================
// ForceCalculator unit tests
// =============================================================================

// ---- Helpers ----------------------------------------------------------------

static System makeTwoParticles(double x0, double y0, double x1, double y1) {
    System s(2);
    s.setPosition(0, x0, y0);
    s.setPosition(1, x1, y1);
    return s;
}

// ---- Cutoff and cached constants --------------------------------------------

// The default constructor uses = default and leaves cached constants at 0
// (recomputeCachedConstants is only called by the parametric constructor).
// This test documents that behavior and verifies the parametric ctor.
TEST(ForceCalculatorTest, DefaultConstructorCutoffIsZero) {
    ForceCalculator fc;
    EXPECT_DOUBLE_EQ(fc.getCutoff(), 0.0);
}

TEST(ForceCalculatorTest, ParametricCutoffSigmaOne) {
    ForceCalculator fc(1.0, 1.0);
    const double expected = std::pow(2.0, 1.0 / 6.0);
    EXPECT_DOUBLE_EQ(fc.getCutoff(), expected);
}

TEST(ForceCalculatorTest, CutoffScalesWithSigma) {
    ForceCalculator fc(1.0, 2.0);
    const double expected = std::pow(2.0, 1.0 / 6.0) * 2.0;
    EXPECT_DOUBLE_EQ(fc.getCutoff(), expected);
}

TEST(ForceCalculatorTest, SetSigmaUpdatesCutoff) {
    ForceCalculator fc;
    fc.setSigma(3.0);
    const double expected = std::pow(2.0, 1.0 / 6.0) * 3.0;
    EXPECT_DOUBLE_EQ(fc.getCutoff(), expected);
}

TEST(ForceCalculatorTest, CutoffSquaredConsistent) {
    ForceCalculator fc(1.0, 1.5);
    const double rc  = fc.getCutoff();
    EXPECT_DOUBLE_EQ(fc.getCutoffSquared(), rc * rc);
}

// ---- Force magnitude at known separations (sigma = 1, eps = 1) -------------

// At r = sigma: F(r)/r = 24/r^2 * (sr6 * (2*sr6 - 1)) with sr6=1
//   => f_over_r2 = 24/1 * (2-1) = 24
//   Force on particle 0 (j=1 is at +dx): fxi -= 24 * dx = -24
TEST(ForceCalculatorTest, ForceAtSigmaSeparation) {
    const double L = 20.0;           // large box — no PBC effect
    System s = makeTwoParticles(5.0, 5.0, 6.0, 5.0);   // separation = 1 = sigma
    Box box(L);
    ForceCalculator fc(1.0, 1.0);

    fc.compute(s, box);

    EXPECT_DOUBLE_EQ(s.getFx(0), -24.0);    // pushed left (away from particle 1)
    EXPECT_DOUBLE_EQ(s.getFy(0),  0.0);
    EXPECT_DOUBLE_EQ(s.getFx(1),  24.0);    // pushed right (away from particle 0)
    EXPECT_DOUBLE_EQ(s.getFy(1),  0.0);
}

// Newton's 3rd law: forces are equal and opposite (though computed redundantly).
TEST(ForceCalculatorTest, ForcesEqualAndOpposite) {
    System s = makeTwoParticles(0.0, 0.0, 0.8, 0.0);
    Box box(20.0);
    ForceCalculator fc;

    fc.compute(s, box);

    EXPECT_DOUBLE_EQ(s.getFx(0), -s.getFx(1));
    EXPECT_DOUBLE_EQ(s.getFy(0), -s.getFy(1));
}

// Particles beyond r_cut receive zero force.
TEST(ForceCalculatorTest, ZeroForceBeyondCutoff) {
    ForceCalculator fc;
    const double beyond = fc.getCutoff() + 0.01;
    System s = makeTwoParticles(0.0, 0.0, beyond, 0.0);
    Box box(20.0);

    fc.compute(s, box);

    EXPECT_DOUBLE_EQ(s.getFx(0), 0.0);
    EXPECT_DOUBLE_EQ(s.getFy(0), 0.0);
    EXPECT_DOUBLE_EQ(s.getFx(1), 0.0);
    EXPECT_DOUBLE_EQ(s.getFy(1), 0.0);
}

// Force is repulsive (pushes particles apart) for r < r_cut.
TEST(ForceCalculatorTest, ForceIsRepulsive) {
    System s = makeTwoParticles(0.0, 0.0, 0.9, 0.0);
    Box box(20.0);
    ForceCalculator fc(1.0, 1.0);

    fc.compute(s, box);

    // Particle 0 is to the left — must be pushed left (negative x).
    EXPECT_LT(s.getFx(0), 0.0);
    // Particle 1 is to the right — must be pushed right (positive x).
    EXPECT_GT(s.getFx(1), 0.0);
}

// Force in y for a vertical pair.
TEST(ForceCalculatorTest, VerticalPairForce) {
    System s = makeTwoParticles(0.0, 0.0, 0.0, 1.0);   // separation = sigma along y
    Box box(20.0);
    ForceCalculator fc(1.0, 1.0);

    fc.compute(s, box);

    EXPECT_DOUBLE_EQ(s.getFx(0), 0.0);
    EXPECT_DOUBLE_EQ(s.getFy(0), -24.0);
    EXPECT_DOUBLE_EQ(s.getFx(1), 0.0);
    EXPECT_DOUBLE_EQ(s.getFy(1),  24.0);
}

// compute() resets forces before evaluating.
TEST(ForceCalculatorTest, ComputeResetsForces) {
    const double L = 20.0;
    System s = makeTwoParticles(0.0, 0.0, 5.0, 0.0);   // beyond cutoff
    s.setForce(0, 99.0, 99.0);
    s.setForce(1, 99.0, 99.0);
    Box box(L);
    ForceCalculator fc;

    fc.compute(s, box);

    EXPECT_DOUBLE_EQ(s.getFx(0), 0.0);
    EXPECT_DOUBLE_EQ(s.getFy(0), 0.0);
    EXPECT_DOUBLE_EQ(s.getFx(1), 0.0);
    EXPECT_DOUBLE_EQ(s.getFy(1), 0.0);
}

// ---- Periodic boundary conditions -------------------------------------------

// Particle at 0.1 and particle at L-0.1 are 0.2 apart through PBC.
TEST(ForceCalculatorTest, ForceAcrossPeriodicBoundary) {
    const double L  = 10.0;
    const double g  = 0.1;    // 2*g = 0.2 separation, well within r_cut~1.12
    System s = makeTwoParticles(g, 5.0, L - g, 5.0);
    Box box(L);
    ForceCalculator fc(1.0, 1.0);

    fc.compute(s, box);

    // Particle 0 at x=0.1: nearest image of particle 1 is at x=L-0.1 = 9.9,
    // so dx = 9.9-0.1 = 9.8 -> minimumImage => 9.8-10 = -0.2.
    // Force on 0 is -f_over_r2 * dx, dx < 0 => force is positive (pushed right).
    EXPECT_GT(s.getFx(0), 0.0);
    EXPECT_LT(s.getFx(1), 0.0);
}

// ---- computeEnergy ----------------------------------------------------------

// Energy between two particles at r = sigma is eps = 1.
// WCA: U(sigma) = 4*(1-1)+1 = 1.
TEST(ForceCalculatorTest, EnergyAtSigmaSeparation) {
    System s = makeTwoParticles(0.0, 0.0, 1.0, 0.0);
    Box box(20.0);
    ForceCalculator fc(1.0, 1.0);

    const double U = fc.computeEnergy(s, box);
    EXPECT_DOUBLE_EQ(U, 1.0);
}

TEST(ForceCalculatorTest, EnergyZeroBeyondCutoff) {
    ForceCalculator fc;
    const double beyond = fc.getCutoff() + 0.01;
    System s = makeTwoParticles(0.0, 0.0, beyond, 0.0);
    Box box(20.0);

    EXPECT_DOUBLE_EQ(fc.computeEnergy(s, box), 0.0);
}

TEST(ForceCalculatorTest, EnergyIsNonNegative) {
    // WCA is purely repulsive; energy is always >= 0.
    System s = makeTwoParticles(0.0, 0.0, 0.8, 0.0);
    Box box(20.0);
    ForceCalculator fc;

    EXPECT_GE(fc.computeEnergy(s, box), 0.0);
}

TEST(ForceCalculatorTest, EnergyCountedOncePerPair) {
    // Four particles, all far apart except particles 0-1 which are at separation 1.
    System s(4);
    s.setPosition(0,  0.0, 0.0);
    s.setPosition(1,  1.0, 0.0);
    s.setPosition(2, 10.0, 0.0);
    s.setPosition(3, 15.0, 0.0);
    Box box(100.0);
    ForceCalculator fc(1.0, 1.0);

    const double U = fc.computeEnergy(s, box);
    EXPECT_DOUBLE_EQ(U, 1.0);          // only the 0-1 pair contributes
}

// ---- Cell-list overload matches brute force ---------------------------------

class FCEquivalenceTest : public ::testing::Test {
protected:
    static constexpr std::size_t N = 64;
    static constexpr double      L = 15.0;
    static constexpr double      r_skin = 0.2;

    System          sys{N};
    Box             box{L};
    ForceCalculator fc{1.0, 1.0};

    void SetUp() override {
        // Place particles on a grid — well spaced, some within r_cut of each other.
        const int side = static_cast<int>(std::ceil(std::sqrt(double(N))));
        const double spacing = L / static_cast<double>(side);
        for (std::size_t i = 0; i < N; ++i) {
            const double px = (static_cast<double>(i % static_cast<std::size_t>(side)) + 0.5) * spacing;
            const double py = (static_cast<double>(i / static_cast<std::size_t>(side)) + 0.5) * spacing;
            sys.setPosition(i, px, py);
        }
    }
};

TEST_F(FCEquivalenceTest, ForcesMatchBruteForce) {
    // Brute force.
    fc.compute(sys, box);
    std::vector<double> fx_bf(N), fy_bf(N);
    for (std::size_t i = 0; i < N; ++i) {
        fx_bf[i] = sys.getFx(i);
        fy_bf[i] = sys.getFy(i);
    }

    // Cell list.
    CellList cl(fc.getCutoff(), r_skin, N, box);
    cl.rebuild(sys, box);
    fc.compute(sys, box, cl);

    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(sys.getFx(i), fx_bf[i], 1e-12) << "i=" << i;
        EXPECT_NEAR(sys.getFy(i), fy_bf[i], 1e-12) << "i=" << i;
    }
}

TEST_F(FCEquivalenceTest, ForcesMatchAfterRebuild) {
    // Verify equivalence still holds on a second rebuild (tests that rebuild
    // clears and repopulates the list, not accumulates).
    CellList cl(fc.getCutoff(), r_skin, N, box);
    cl.rebuild(sys, box);
    cl.rebuild(sys, box);    // second rebuild

    fc.compute(sys, box);
    std::vector<double> fx_bf(N), fy_bf(N);
    for (std::size_t i = 0; i < N; ++i) { fx_bf[i] = sys.getFx(i); fy_bf[i] = sys.getFy(i); }

    fc.compute(sys, box, cl);

    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(sys.getFx(i), fx_bf[i], 1e-12) << "i=" << i;
        EXPECT_NEAR(sys.getFy(i), fy_bf[i], 1e-12) << "i=" << i;
    }
}

// =============================================================================
// Pluggable potential — switching, soft sphere, static helpers
// =============================================================================

// ---- Default / switching ----------------------------------------------------

TEST(ForceCalculatorTest, DefaultPotentialTypeIsWCA) {
    ForceCalculator fc(1.0, 1.0);
    EXPECT_EQ(fc.getPotentialType(), PotentialType::WCA);
}

TEST(ForceCalculatorTest, SetPotentialSoftSphereCutoffEqualsSigma) {
    ForceCalculator fc(1.0, 1.5);
    fc.setPotentialType(PotentialType::SoftSphere);
    EXPECT_EQ(fc.getPotentialType(), PotentialType::SoftSphere);
    EXPECT_DOUBLE_EQ(fc.getCutoff(), 1.5);
    EXPECT_DOUBLE_EQ(fc.getCutoffSquared(), 1.5 * 1.5);
}

TEST(ForceCalculatorTest, SwitchPotentialBackToWCARestoresCutoff) {
    ForceCalculator fc(1.0, 2.0);
    fc.setPotentialType(PotentialType::SoftSphere);
    fc.setPotentialType(PotentialType::WCA);
    const double expected = std::pow(2.0, 1.0 / 6.0) * 2.0;
    EXPECT_DOUBLE_EQ(fc.getCutoff(), expected);
}

TEST(ForceCalculatorTest, ConstructWithSoftSphereSetsSigmaCutoff) {
    ForceCalculator fc(1.0, 1.0, PotentialType::SoftSphere);
    EXPECT_DOUBLE_EQ(fc.getCutoff(), 1.0);
}

// Switching potentials on the same object must change the computed force,
// confirming the function-pointer dispatch is actually being used.
TEST(ForceCalculatorTest, SwitchingPotentialChangesForce) {
    System s = makeTwoParticles(0.0, 0.0, 0.5, 0.0);   // r = 0.5
    Box box(20.0);
    ForceCalculator fc(1.0, 1.0);                       // WCA

    fc.compute(s, box);
    const double fx_wca = s.getFx(0);

    fc.setPotentialType(PotentialType::SoftSphere);
    fc.compute(s, box);
    const double fx_ss = s.getFx(0);

    // WCA at r = 0.5 sigma is huge (~10^4); SoftSphere is order eps/sigma.
    // The two must differ substantially.
    EXPECT_GT(std::abs(fx_wca), std::abs(fx_ss) * 100.0);
    EXPECT_LT(fx_wca, 0.0);     // both repulsive
    EXPECT_LT(fx_ss,  0.0);
}

// ---- Soft-sphere static helpers --------------------------------------------

// At r = sigma/2 with eps = sigma = 1:
//   U = (1/2) * (1 - 0.5)^2 = 0.125
//   F = (1/1) * (1 - 0.5)   = 0.5
//   f_over_r2 = F / r = 0.5 / 0.5 = 1.0
TEST(ForceCalculatorTest, SoftSphereForceHelperAtHalfSigma) {
    double f_over_r2 = 0.0;
    ForceCalculator::softSphereForce(0.25, 1.0, 1.0, 1.0, f_over_r2);
    EXPECT_DOUBLE_EQ(f_over_r2, 1.0);
}

TEST(ForceCalculatorTest, SoftSphereEnergyHelperAtHalfSigma) {
    const double U = ForceCalculator::softSphereEnergy(0.25, 1.0, 1.0, 1.0);
    EXPECT_DOUBLE_EQ(U, 0.125);
}

// Soft sphere is a finite-energy potential: U(0) = eps/2, NOT divergent.
TEST(ForceCalculatorTest, SoftSphereEnergyAtZeroSeparation) {
    const double U = ForceCalculator::softSphereEnergy(1e-30, 2.0, 1.0, 1.0);
    EXPECT_NEAR(U, 1.0, 1e-10);    // eps/2 = 1.0
}

// Force/energy consistency: F = -dU/dr. Verify by central difference.
TEST(ForceCalculatorTest, SoftSphereForceMatchesEnergyDerivative) {
    const double eps = 1.0, sigma = 1.0, sigma2 = 1.0;
    const double r   = 0.6;
    const double h   = 1e-6;
    const double Up  = ForceCalculator::softSphereEnergy((r + h) * (r + h), eps, sigma, sigma2);
    const double Um  = ForceCalculator::softSphereEnergy((r - h) * (r - h), eps, sigma, sigma2);
    const double F_fd = -(Up - Um) / (2.0 * h);

    double f_over_r2 = 0.0;
    ForceCalculator::softSphereForce(r * r, eps, sigma, sigma2, f_over_r2);
    const double F_analytic = f_over_r2 * r;     // F = (F/r) * r

    EXPECT_NEAR(F_analytic, F_fd, 1e-6);
}

// Same sanity check for WCA — guards against accidental refactor breakage.
TEST(ForceCalculatorTest, WCAForceMatchesEnergyDerivative) {
    const double eps = 1.0, sigma = 1.0, sigma2 = 1.0;
    const double r   = 0.95;
    const double h   = 1e-7;
    const double Up  = ForceCalculator::wcaEnergy((r + h) * (r + h), eps, sigma, sigma2);
    const double Um  = ForceCalculator::wcaEnergy((r - h) * (r - h), eps, sigma, sigma2);
    const double F_fd = -(Up - Um) / (2.0 * h);

    double f_over_r2 = 0.0;
    ForceCalculator::wcaForce(r * r, eps, sigma, sigma2, f_over_r2);
    const double F_analytic = f_over_r2 * r;

    EXPECT_NEAR(F_analytic, F_fd, 1e-4);   // looser: WCA is steep, f.d. is noisier
}

// ---- Soft-sphere ForceCalculator end-to-end --------------------------------

// At r = sigma/2: force on particle 0 along -x is -0.5 (eps = sigma = 1).
TEST(ForceCalculatorTest, SoftSphereForceEndToEnd) {
    System s = makeTwoParticles(0.0, 0.0, 0.5, 0.0);
    Box box(20.0);
    ForceCalculator fc(1.0, 1.0, PotentialType::SoftSphere);

    fc.compute(s, box);

    EXPECT_DOUBLE_EQ(s.getFx(0), -0.5);
    EXPECT_DOUBLE_EQ(s.getFy(0),  0.0);
    EXPECT_DOUBLE_EQ(s.getFx(1),  0.5);
    EXPECT_DOUBLE_EQ(s.getFy(1),  0.0);
}

// Beyond cutoff = sigma: zero force.
TEST(ForceCalculatorTest, SoftSphereZeroForceBeyondSigma) {
    System s = makeTwoParticles(0.0, 0.0, 1.01, 0.0);
    Box box(20.0);
    ForceCalculator fc(1.0, 1.0, PotentialType::SoftSphere);

    fc.compute(s, box);

    EXPECT_DOUBLE_EQ(s.getFx(0), 0.0);
    EXPECT_DOUBLE_EQ(s.getFx(1), 0.0);
}

// Energy at r = sigma/2: U = 0.125 (eps = sigma = 1).
TEST(ForceCalculatorTest, SoftSphereEnergyEndToEnd) {
    System s = makeTwoParticles(0.0, 0.0, 0.5, 0.0);
    Box box(20.0);
    ForceCalculator fc(1.0, 1.0, PotentialType::SoftSphere);

    EXPECT_DOUBLE_EQ(fc.computeEnergy(s, box), 0.125);
}

TEST(ForceCalculatorTest, SoftSphereEnergyZeroBeyondSigma) {
    System s = makeTwoParticles(0.0, 0.0, 1.01, 0.0);
    Box box(20.0);
    ForceCalculator fc(1.0, 1.0, PotentialType::SoftSphere);

    EXPECT_DOUBLE_EQ(fc.computeEnergy(s, box), 0.0);
}

TEST(ForceCalculatorTest, SoftSphereEnergyIsNonNegative) {
    System s = makeTwoParticles(0.0, 0.0, 0.3, 0.0);
    Box box(20.0);
    ForceCalculator fc(1.0, 1.0, PotentialType::SoftSphere);

    EXPECT_GE(fc.computeEnergy(s, box), 0.0);
}

// Soft sphere is still repulsive (forces apart for r < sigma).
TEST(ForceCalculatorTest, SoftSphereForceIsRepulsive) {
    System s = makeTwoParticles(0.0, 0.0, 0.7, 0.0);
    Box box(20.0);
    ForceCalculator fc(1.0, 1.0, PotentialType::SoftSphere);

    fc.compute(s, box);

    EXPECT_LT(s.getFx(0), 0.0);
    EXPECT_GT(s.getFx(1), 0.0);
}

// ---- f0ForOverlap (force-balance helper) ------------------------------------

// Soft sphere has the closed form f0 = (eps/sigma) * (1 - delta).
TEST(ForceCalculatorTest, F0ForOverlapSoftSphereClosedForm) {
    EXPECT_DOUBLE_EQ(
        ForceCalculator::f0ForOverlap(PotentialType::SoftSphere, 0.5, 1.0, 1.0),
        0.5);
    EXPECT_DOUBLE_EQ(
        ForceCalculator::f0ForOverlap(PotentialType::SoftSphere, 0.9, 2.0, 1.0),
        0.2);   // (eps/sigma) * 0.1 = 2 * 0.1
    EXPECT_DOUBLE_EQ(
        ForceCalculator::f0ForOverlap(PotentialType::SoftSphere, 0.5, 1.0, 2.0),
        0.25);  // (eps/sigma) * (1 - 0.5) = 0.5 * 0.5
}

// At delta >= cutoff/sigma, no contact force is needed — passive limit.
TEST(ForceCalculatorTest, F0ForOverlapPassiveLimitSoftSphere) {
    EXPECT_DOUBLE_EQ(
        ForceCalculator::f0ForOverlap(PotentialType::SoftSphere, 1.0, 1.0, 1.0),
        0.0);
    EXPECT_DOUBLE_EQ(
        ForceCalculator::f0ForOverlap(PotentialType::SoftSphere, 1.5, 1.0, 1.0),
        0.0);
}

TEST(ForceCalculatorTest, F0ForOverlapPassiveLimitWCA) {
    const double cutoff = std::pow(2.0, 1.0 / 6.0);
    EXPECT_DOUBLE_EQ(
        ForceCalculator::f0ForOverlap(PotentialType::WCA, cutoff, 1.0, 1.0),
        0.0);
    EXPECT_DOUBLE_EQ(
        ForceCalculator::f0ForOverlap(PotentialType::WCA, cutoff + 0.01, 1.0, 1.0),
        0.0);
}

// At delta = 1 (r = sigma), WCA force is 24 eps/sigma exactly.
TEST(ForceCalculatorTest, F0ForOverlapWCAAtSigma) {
    EXPECT_DOUBLE_EQ(
        ForceCalculator::f0ForOverlap(PotentialType::WCA, 1.0, 1.0, 1.0),
        24.0);
    EXPECT_DOUBLE_EQ(
        ForceCalculator::f0ForOverlap(PotentialType::WCA, 1.0, 1.0, 2.0),
        12.0);   // 24 * eps / sigma = 24 / 2
}

// Closed-form sanity: f0_WCA(delta) = 24 * delta^-7 * (2 delta^-6 - 1).
TEST(ForceCalculatorTest, F0ForOverlapWCAClosedForm) {
    const double delta = 0.95;
    const double expected = 24.0 * std::pow(delta, -7.0) *
                            (2.0 * std::pow(delta, -6.0) - 1.0);
    EXPECT_NEAR(
        ForceCalculator::f0ForOverlap(PotentialType::WCA, delta, 1.0, 1.0),
        expected, 1e-12);
}

// Round-trip check: configuring the calculator with f0 = f0ForOverlap(delta)
// and placing two particles at r = delta*sigma gives a per-particle radial
// force of magnitude f0 (within FP precision).
TEST(ForceCalculatorTest, F0ForOverlapMatchesPairForceSoftSphere) {
    const double delta = 0.7;
    const double f0 = ForceCalculator::f0ForOverlap(
        PotentialType::SoftSphere, delta, 1.0, 1.0);

    System s = makeTwoParticles(0.0, 0.0, delta, 0.0);
    Box box(20.0);
    ForceCalculator fc(1.0, 1.0, PotentialType::SoftSphere);
    fc.compute(s, box);

    // Repulsive radial force on particle 0 along -x has magnitude f0.
    EXPECT_NEAR(std::abs(s.getFx(0)), f0, 1e-12);
}

TEST(ForceCalculatorTest, F0ForOverlapMatchesPairForceWCA) {
    const double delta = 1.05;
    const double f0 = ForceCalculator::f0ForOverlap(
        PotentialType::WCA, delta, 1.0, 1.0);

    System s = makeTwoParticles(0.0, 0.0, delta, 0.0);
    Box box(20.0);
    ForceCalculator fc(1.0, 1.0, PotentialType::WCA);
    fc.compute(s, box);

    EXPECT_NEAR(std::abs(s.getFx(0)), f0, 1e-12);
}

// ---- Soft-sphere cell-list equivalence -------------------------------------

TEST(ForceCalculatorTest, SoftSphereCellListMatchesBruteForce) {
    constexpr std::size_t N = 64;
    constexpr double      L = 15.0;
    constexpr double      r_skin = 0.2;

    System          sys(N);
    Box             box(L);
    ForceCalculator fc(1.0, 1.0, PotentialType::SoftSphere);

    const int side = static_cast<int>(std::ceil(std::sqrt(double(N))));
    const double spacing = L / static_cast<double>(side);
    for (std::size_t i = 0; i < N; ++i) {
        const double px = (static_cast<double>(i % static_cast<std::size_t>(side)) + 0.5) * spacing;
        const double py = (static_cast<double>(i / static_cast<std::size_t>(side)) + 0.5) * spacing;
        sys.setPosition(i, px, py);
    }

    // Brute force.
    fc.compute(sys, box);
    std::vector<double> fx_bf(N), fy_bf(N);
    for (std::size_t i = 0; i < N; ++i) {
        fx_bf[i] = sys.getFx(i);
        fy_bf[i] = sys.getFy(i);
    }

    // Cell list.
    CellList cl(fc.getCutoff(), r_skin, N, box);
    cl.rebuild(sys, box);
    fc.compute(sys, box, cl);

    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(sys.getFx(i), fx_bf[i], 1e-12) << "i=" << i;
        EXPECT_NEAR(sys.getFy(i), fy_bf[i], 1e-12) << "i=" << i;
    }
}
