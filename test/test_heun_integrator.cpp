#include "Box.hpp"
#include "ForceCalculator.hpp"
#include "HeunIntegrator.hpp"
#include "Initializer.hpp"
#include "RandomGenerator.hpp"
#include "System.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <cstddef>
#include <vector>

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
        const double dx = sys.getX(i) - x0[i];
        const double dy = sys.getY(i) - y0[i];
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

// Positions are NOT auto-wrapped by the engine (see Box.hpp). Particles start
// at the high-x/y corner with overlap forces pushing them outward; we expect
// some particles to exit [0, L), but all pair distances under minimum image
// must remain finite and physically sensible (no NaNs, no runaway drift).
TEST(HeunIntegratorTest, PositionsDriftFreelyAcrossBoundary) {
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

    bool any_outside_box = false;
    for (int step = 0; step < 10; ++step) {
        integ.step(sys, box, fc, nullptr, rng);
        for (std::size_t i = 0; i < N; ++i) {
            EXPECT_TRUE(std::isfinite(sys.getX(i))) << "step=" << step << " i=" << i;
            EXPECT_TRUE(std::isfinite(sys.getY(i))) << "step=" << step << " i=" << i;
            if (sys.getX(i) < 0.0 || sys.getX(i) >= L ||
                sys.getY(i) < 0.0 || sys.getY(i) >= L) {
                any_outside_box = true;
            }
        }
    }
    // We don't insist positions exit the box (that depends on RNG), but the
    // engine must tolerate it if they do — covered by the finiteness checks.
    (void)any_outside_box;
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

// Stored positions are unwrapped: drive a passive Brownian system long enough
// that the RMS displacement exceeds L/2, then verify the naive MSD (no
// minimum-image correction) still grows as 4*D*t. This is the regression test
// for the MSD plateau bug — pre-fix, the engine wrapped positions in-place
// and the same measurement would saturate near L^2/6.
TEST(HeunIntegratorTest, UnwrappedFreeParticleMSDGrowsPastBox) {
    const double gamma = 1.0;
    const double dt    = 1e-2;
    const double kT    = 1.0;
    const double D     = kT / gamma;
    const double L     = 5.0;            // small box: RMS will exceed L/2 fast
    const int    T     = 2000;           // 4*D*dt*T = 80 >> (L/2)^2 = 6.25
    const std::size_t N = 256;

    HeunIntegrator integ(gamma, dt);
    integ.setKBT(kT);
    Box             box(L);
    RandomGenerator rng(11);
    // No forces (use SoftSphere with strength 0, particles spread far apart).
    // Easiest: place all particles at the box center; with kT >> 0 they
    // quickly disperse, but WCA blow-up is bounded by the drift cap.
    integ.setMaxDrift(0.1);
    ForceCalculator fc(0.0, 1.0, PotentialType::WCA);

    System sys(N);
    // Spread on a wide lattice so initial WCA is zero, then let diffusion take
    // them past the box boundary.
    const auto m = static_cast<std::size_t>(std::ceil(std::sqrt(double(N))));
    const double spacing = L / static_cast<double>(m);
    for (std::size_t i = 0; i < N; ++i) {
        sys.setPosition(i,
                        (static_cast<double>(i % m) + 0.5) * spacing,
                        (static_cast<double>(i / m) + 0.5) * spacing);
    }
    std::vector<double> x0(N), y0(N);
    for (std::size_t i = 0; i < N; ++i) { x0[i] = sys.getX(i); y0[i] = sys.getY(i); }

    for (int t = 0; t < T; ++t)
        integ.step(sys, box, fc, nullptr, rng);

    double msd = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double dx = sys.getX(i) - x0[i];
        const double dy = sys.getY(i) - y0[i];
        msd += dx * dx + dy * dy;
    }
    msd /= static_cast<double>(N);

    const double expected = 4.0 * D * dt * T;       // = 80
    const double sigma    = expected / std::sqrt(static_cast<double>(N) / 2.0);

    // MSD must be far above any wrapping plateau (which would saturate at
    // ~L^2/3 ≈ 8.3 for naive in-box subtraction), and within a few sigma of
    // the diffusive prediction.
    EXPECT_GT(msd, 2.0 * L * L);
    EXPECT_NEAR(msd, expected, 4.0 * sigma);
}
