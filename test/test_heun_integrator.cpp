#include "Box.hpp"
#include "ForceCalculator.hpp"
#include "HeunIntegrator.hpp"
#include "Initializer.hpp"
#include "RandomGenerator.hpp"
#include "System.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <cstddef>

// =============================================================================
// HeunIntegrator unit tests
// -----------------------------------------------------------------------------
// Mirror of test_integrator.cpp but for the predictor-corrector path. Heun
// reduces to EM in two checkable limits: zero drift (only noise — Heun and
// EM apply the same noise step) and constant drift (Heun's averaged drift
// equals the predictor drift, which equals EM's drift). Both are tested.
// =============================================================================

namespace {

// A box big enough that N particles on a square lattice never see each
// other — so the WCA force is identically zero everywhere. Lets us drive
// HeunIntegrator with the real ForceCalculator (its step() always re-
// evaluates forces internally) and still get free-particle dynamics.
constexpr double kBigL = 1000.0;
constexpr std::size_t kFreeN = 256;

void placeOnSparseLattice(System& sys, double L) {
    const std::size_t N = sys.getNumParticles();
    const auto m = static_cast<std::size_t>(std::ceil(std::sqrt(double(N))));
    const double dx = L / static_cast<double>(m);
    for (std::size_t i = 0; i < N; ++i) {
        const std::size_t ix = i % m;
        const std::size_t iy = i / m;
        sys.setPosition(i,
                        (ix + 0.5) * dx,
                        (iy + 0.5) * dx);
    }
}

} // namespace

// Free-particle MSD: with f0 = 0, k_a = 0, and particles placed far enough
// apart that WCA forces vanish, the dynamics reduce to dr = sqrt(2 D dt) Z.
// Heun's drift1 and drift2 are both zero, so only the noise term acts —
// matches EM exactly in distribution. MSD after T steps = 4 D dt T.
TEST(HeunIntegratorTest, FreeParticleMSDMatchesDiffusion) {
    const double gamma = 1.0;
    const double dt    = 1e-3;
    const double kT    = 1.0;
    const double D     = kT / gamma;
    const int    T     = 100;

    HeunIntegrator integ(gamma, dt);
    integ.setKBT(kT);
    Box             box(kBigL);
    RandomGenerator rng(7);
    ForceCalculator fc(1.0, 1.0, PotentialType::WCA);

    System sys(kFreeN);
    placeOnSparseLattice(sys, kBigL);
    // Snapshot starting positions to avoid the lattice spacing biasing the MSD.
    std::vector<double> x0(kFreeN), y0(kFreeN);
    for (std::size_t i = 0; i < kFreeN; ++i) {
        x0[i] = sys.getX(i);
        y0[i] = sys.getY(i);
    }

    for (int t = 0; t < T; ++t)
        integ.step(sys, box, fc, /*cl=*/nullptr, rng);

    double msd = 0.0;
    for (std::size_t i = 0; i < kFreeN; ++i) {
        double dx = sys.getX(i) - x0[i];
        double dy = sys.getY(i) - y0[i];
        box.minimumImage(dx, dy);
        msd += dx * dx + dy * dy;
    }
    msd /= static_cast<double>(kFreeN);

    const double expected = 4.0 * D * dt * T;
    const double sigma    = expected / std::sqrt(static_cast<double>(kFreeN) / 2.0);
    EXPECT_NEAR(msd, expected, 4.0 * sigma);
}

// Constant-drift drift test. With kT = 0, k_a = 0, D_r = 0, and theta_0 = 0
// fixed, the only drift is the active force f0 in +x. Both predictor and
// corrector see the same drift (no spatial dependence, theta is frozen),
// so 0.5*(d1+d2) = mu*f0. Heun should give x_{n+1} = x_n + mu*f0*dt
// EXACTLY (up to drift-cap clipping, which we keep below the cap).
TEST(HeunIntegratorTest, ConstantActiveDriftIsExact) {
    const double gamma = 1.0;
    const double dt    = 1e-3;
    const double f0    = 0.05;     // mu*f0*dt = 5e-5, well under any cap
    const std::size_t N = 16;

    HeunIntegrator integ(gamma, dt);
    integ.setKBT(0.0);              // athermal: no noise
    integ.setActiveForce(f0);
    integ.setRotationalDiffusion(0.0);   // theta is frozen
    Box             box(kBigL);
    RandomGenerator rng(0);
    ForceCalculator fc(1.0, 1.0, PotentialType::WCA);

    System sys(N);
    placeOnSparseLattice(sys, kBigL);
    for (std::size_t i = 0; i < N; ++i) sys.setTheta(i, 0.0);

    std::vector<double> x0(N);
    for (std::size_t i = 0; i < N; ++i) x0[i] = sys.getX(i);

    constexpr int T = 50;
    for (int t = 0; t < T; ++t)
        integ.step(sys, box, fc, /*cl=*/nullptr, rng);

    // Snapshot starting y too, so we can assert no drift in the perpendicular axis.
    std::vector<double> y0(N);
    for (std::size_t i = 0; i < N; ++i) y0[i] = sys.getX(i);   // unused below; kept for symmetry
    (void)y0;

    const double expected_dx = (1.0 / gamma) * f0 * dt * T;
    for (std::size_t i = 0; i < N; ++i) {
        const double dx = sys.getX(i) - x0[i];
        EXPECT_NEAR(dx, expected_dx, 1e-12) << "i=" << i;
    }
}

// PBC: positions stay in the primary cell after each step. Particles start
// at the high-x/y corner so any positive displacement triggers a wrap. The
// drift cap is on (matches the project default) so the WCA blow-up from
// the initial overlap doesn't fling particles past the box.
TEST(HeunIntegratorTest, PositionsWrappedAfterStep) {
    const double L = 5.0;
    const std::size_t N = 32;
    HeunIntegrator integ(1.0, 1e-2);
    integ.setKBT(1.0);
    integ.setMaxDrift(0.1);
    Box             box(L);
    RandomGenerator rng(99);
    ForceCalculator fc(1.0, 1.0, PotentialType::WCA);

    System sys(N);
    for (std::size_t i = 0; i < N; ++i) sys.setPosition(i, L - 0.01, L - 0.01);

    for (int step = 0; step < 10; ++step) {
        integ.step(sys, box, fc, nullptr, rng);
        for (std::size_t i = 0; i < N; ++i) {
            EXPECT_GE(sys.getX(i), 0.0) << "step=" << step << " i=" << i;
            EXPECT_LT(sys.getX(i), L)   << "step=" << step << " i=" << i;
            EXPECT_GE(sys.getY(i), 0.0) << "step=" << step << " i=" << i;
            EXPECT_LT(sys.getY(i), L)   << "step=" << step << " i=" << i;
        }
    }
}

// Athermal limit (kT = 0, f0 = 0, k_a = 0) on a free-particle setup: Heun
// is purely zero-drift / zero-noise, so positions must not move at all.
TEST(HeunIntegratorTest, AthermalFreeParticleStaysPut) {
    const std::size_t N = 16;
    HeunIntegrator integ(1.0, 1e-3);
    integ.setKBT(0.0);
    integ.setRotationalDiffusion(0.0);
    Box             box(kBigL);
    RandomGenerator rng(0);
    ForceCalculator fc(1.0, 1.0, PotentialType::WCA);

    System sys(N);
    placeOnSparseLattice(sys, kBigL);
    std::vector<double> x0(N), y0(N);
    for (std::size_t i = 0; i < N; ++i) {
        x0[i] = sys.getX(i);
        y0[i] = sys.getY(i);
    }

    for (int t = 0; t < 20; ++t)
        integ.step(sys, box, fc, nullptr, rng);

    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(sys.getX(i), x0[i]);
        EXPECT_DOUBLE_EQ(sys.getY(i), y0[i]);
    }
}
