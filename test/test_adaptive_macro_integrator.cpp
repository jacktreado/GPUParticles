#include "AdaptiveMacroIntegrator.hpp"

#include "Box.hpp"
#include "ForceCalculator.hpp"
#include "RandomGenerator.hpp"
#include "System.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>

// =============================================================================
// AdaptiveMacroIntegrator unit tests
// -----------------------------------------------------------------------------
// The integrator does Strang(N - D - N) operator splitting where N is the
// pure-noise operator (translational + anchor + rotational additive Brownian
// increments) and D is the deterministic flow integrated by the inner
// Cash-Karp 5(4) AdaptiveIntegrator. The macro Δt is chosen by step doubling
// + a PI controller; Brownian bridges keep coarse and fine noise consistent.
// =============================================================================

// ---- Construction / getters / setters --------------------------------------

TEST(AdaptiveMacroIntegratorTest, DefaultConstruction) {
    AdaptiveMacroIntegrator ai(1.0);
    EXPECT_DOUBLE_EQ(ai.getFriction(),       1.0);
    EXPECT_DOUBLE_EQ(ai.getKBT(),            0.0);
    EXPECT_DOUBLE_EQ(ai.getRotationalDiffusion(), 0.0);
    EXPECT_DOUBLE_EQ(ai.getActiveForce(),    0.0);
    EXPECT_DOUBLE_EQ(ai.getSpringStiffness(),0.0);
    EXPECT_DOUBLE_EQ(ai.getAnchorFriction(), 0.0);
    EXPECT_DOUBLE_EQ(ai.getAbsTol(),         1.0e-3);
    EXPECT_DOUBLE_EQ(ai.getRelTol(),         1.0e-2);
    EXPECT_DOUBLE_EQ(ai.getDt(),             1.0e-3);
}

TEST(AdaptiveMacroIntegratorTest, AllSettersWork) {
    AdaptiveMacroIntegrator ai(1.0);
    ai.setFriction(0.5);
    ai.setKBT(0.1);
    ai.setRotationalDiffusion(2.0);
    ai.setActiveForce(0.3);
    ai.setSpringStiffness(1.5);
    ai.setAnchorFriction(0.7);
    ai.setAbsTol(1.0e-5);
    ai.setRelTol(1.0e-4);
    ai.setDt(2.5e-4);
    EXPECT_DOUBLE_EQ(ai.getFriction(),       0.5);
    EXPECT_DOUBLE_EQ(ai.getKBT(),            0.1);
    EXPECT_DOUBLE_EQ(ai.getRotationalDiffusion(), 2.0);
    EXPECT_DOUBLE_EQ(ai.getActiveForce(),    0.3);
    EXPECT_DOUBLE_EQ(ai.getSpringStiffness(),1.5);
    EXPECT_DOUBLE_EQ(ai.getAnchorFriction(), 0.7);
    EXPECT_DOUBLE_EQ(ai.getAbsTol(),         1.0e-5);
    EXPECT_DOUBLE_EQ(ai.getRelTol(),         1.0e-4);
    EXPECT_DOUBLE_EQ(ai.getDt(),             2.5e-4);
}

// ---- step() returns positive dt and advances the system --------------------

TEST(AdaptiveMacroIntegratorTest, StepReturnsPositiveDt) {
    System sys(1);
    sys.setPosition(0, 5.0, 5.0);
    Box             box(10.0);
    ForceCalculator fc(1.0, 1.0);
    RandomGenerator rng(0);

    AdaptiveMacroIntegrator ai(1.0);
    const double dt = ai.step(sys, box, fc, rng);
    EXPECT_GT(dt, 0.0);
    EXPECT_TRUE(std::isfinite(dt));
}

// With NO noise (kT = 0, D_r = 0) and no active/spring, the macro step should
// be effectively deterministic — coarse and fine should agree to high
// accuracy and the integrator should grow dt aggressively.
TEST(AdaptiveMacroIntegratorTest, AthermalNoForceTrajectoryStays) {
    System sys(1);
    sys.setPosition(0, 5.0, 5.0);
    Box             box(10.0);
    ForceCalculator fc(1.0, 1.0);
    RandomGenerator rng(0);

    AdaptiveMacroIntegrator ai(1.0);
    ai.setKBT(0.0);
    ai.setRotationalDiffusion(0.0);

    for (int i = 0; i < 5; ++i) ai.step(sys, box, fc, rng);

    EXPECT_NEAR(sys.getX(0), 5.0, 1.0e-9);
    EXPECT_NEAR(sys.getY(0), 5.0, 1.0e-9);
}

// Pure translational diffusion (no force, no rotation): MSD scales as 4*D*t,
// D = kT/gamma. Particles are placed on a lattice with enough spacing that
// WCA never engages (>> r_cut), so D-step is identity and the macro step
// reduces to pure noise. Use many particles for statistical averaging.
TEST(AdaptiveMacroIntegratorTest, MSDScalesWithDiffusionCoefficient) {
    const double gamma = 1.0;
    const double kT    = 1.0;
    const double D     = kT / gamma;
    const std::size_t N = 256;

    // Lattice with spacing 4*sigma so particles never enter each other's
    // WCA cutoff (~1.12 sigma) and the deterministic flow stays trivial.
    const int    side    = static_cast<int>(std::ceil(std::sqrt(double(N))));
    const double spacing = 4.0;
    const double L       = spacing * side;

    System sys(N);
    for (std::size_t i = 0; i < N; ++i) {
        const double px = (static_cast<double>(i % static_cast<std::size_t>(side)) + 0.5) * spacing;
        const double py = (static_cast<double>(i / static_cast<std::size_t>(side)) + 0.5) * spacing;
        sys.setPosition(i, px, py);
    }
    // Snapshot the initial positions for the MSD computation; the Box wraps
    // post-step so we need to track unwrapped displacements ourselves.
    std::vector<double> x0(N), y0(N);
    for (std::size_t i = 0; i < N; ++i) { x0[i] = sys.getX(i); y0[i] = sys.getY(i); }

    Box             box(L);
    ForceCalculator fc(1.0, 1.0);
    RandomGenerator rng(7);

    AdaptiveMacroIntegrator ai(gamma);
    ai.setKBT(kT);

    double t = 0.0;
    const double T_end = 0.05;
    while (t < T_end) {
        const double dt = ai.step(sys, box, fc, rng);
        t += dt;
    }

    double msd = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        double dx = sys.getX(i) - x0[i];
        double dy = sys.getY(i) - y0[i];
        box.minimumImage(dx, dy);
        msd += dx * dx + dy * dy;
    }
    msd /= static_cast<double>(N);

    const double expected = 4.0 * D * t;
    // Statistical: stddev / mean ~ sqrt(2/N) for chi-squared-like distribution
    // of MSD; widen to 4 sigma to keep test robust at modest N.
    const double slack = expected * 4.0 * std::sqrt(2.0 / static_cast<double>(N));
    EXPECT_NEAR(msd, expected, slack);
}

// Pure rotational diffusion: angles drift with variance 2*D_r*t, mean ~ 0.
TEST(AdaptiveMacroIntegratorTest, RotationalDiffusionVariance) {
    const double D_r   = 1.0;
    const std::size_t N = 1024;

    System sys(N);
    for (std::size_t i = 0; i < N; ++i) {
        sys.setPosition(i, 50.0, 50.0);
        sys.setTheta(i, 0.0);
    }
    Box             box(100.0);
    ForceCalculator fc(1.0, 1.0);
    RandomGenerator rng(11);

    AdaptiveMacroIntegrator ai(1.0);
    ai.setKBT(0.0);            // disable translational noise (so positions don't move)
    ai.setRotationalDiffusion(D_r);

    double t = 0.0;
    const double T_end = 0.1;
    while (t < T_end) t += ai.step(sys, box, fc, rng);

    double m1 = 0.0, m2 = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double th = sys.getTheta(i);
        m1 += th;
        m2 += th * th;
    }
    m1 /= static_cast<double>(N);
    m2 /= static_cast<double>(N);
    const double var = m2 - m1 * m1;
    const double expected = 2.0 * D_r * t;
    EXPECT_NEAR(var, expected, expected * 0.30);   // 30% slack for finite N
}

// Active force only (no noise): a single particle drifts along its
// orientation at speed mu * f0 = f0 (gamma = 1).
TEST(AdaptiveMacroIntegratorTest, ActiveForceDeterministicDrift) {
    System sys(1);
    sys.setPosition(0, 50.0, 50.0);
    sys.setTheta(0, 0.0);
    Box             box(100.0);
    ForceCalculator fc(1.0, 1.0);
    RandomGenerator rng(0);

    AdaptiveMacroIntegrator ai(1.0);
    ai.setActiveForce(0.5);

    double t = 0.0;
    const double T_end = 0.5;
    while (t < T_end) t += ai.step(sys, box, fc, rng);

    EXPECT_NEAR(sys.getX(0), 50.0 + 0.5 * t, 1.0e-9);
    EXPECT_NEAR(sys.getY(0), 50.0,           1.0e-9);
}

// Spring-only relaxation (anchor frozen): exponential decay toward the
// anchor at rate k_a / gamma.
TEST(AdaptiveMacroIntegratorTest, SpringRelaxationToAnchor) {
    const double gamma = 1.0;
    const double k_a   = 4.0;
    const double d0    = 0.3;

    System sys(1);
    sys.setPosition(0, 50.0 + d0, 50.0);
    sys.setAnchor(0, 50.0, 50.0);
    Box             box(100.0);
    ForceCalculator fc(1.0, 1.0);
    RandomGenerator rng(0);

    AdaptiveMacroIntegrator ai(gamma);
    ai.setSpringStiffness(k_a);
    ai.setAnchorFriction(0.0);     // anchor frozen

    double t = 0.0;
    const double T_end = 0.05;
    while (t < T_end) t += ai.step(sys, box, fc, rng);

    const double expected = d0 * std::exp(-(k_a / gamma) * t);
    EXPECT_NEAR(sys.getX(0) - 50.0, expected, expected * 0.05);   // 5% tol
    EXPECT_NEAR(sys.getY(0),        50.0,    1.0e-9);
}

// ---- Macro-step adaptation -------------------------------------------------

// Tighter tolerances should produce more rejections AND/OR a smaller dt.
TEST(AdaptiveMacroIntegratorTest, TighterToleranceShrinksDt) {
    const double L = 100.0;
    System sys_loose(2);
    sys_loose.setPosition(0, L / 2.0, L / 2.0 + 0.5);
    sys_loose.setPosition(1, L / 2.0, L / 2.0 - 0.5);
    System sys_tight = sys_loose;

    Box             box(L);
    ForceCalculator fc(1.0, 1.0);
    RandomGenerator rng_loose(42), rng_tight(42);

    AdaptiveMacroIntegrator ai_loose(1.0, 1.0e-2, 1.0e-1);
    ai_loose.setKBT(1.0);
    AdaptiveMacroIntegrator ai_tight(1.0, 1.0e-6, 1.0e-5);
    ai_tight.setKBT(1.0);

    // Run both for the same number of accepted macro steps.
    constexpr int K = 20;
    double t_loose = 0.0, t_tight = 0.0;
    for (int i = 0; i < K; ++i) t_loose += ai_loose.step(sys_loose, box, fc, rng_loose);
    for (int i = 0; i < K; ++i) t_tight += ai_tight.step(sys_tight, box, fc, rng_tight);

    // For the tight run the controller should have shrunk dt enough that the
    // total simulated time is shorter (more steps per unit time).
    EXPECT_LT(t_tight, t_loose);
}

// Calling step() many times keeps total accumulated time strictly increasing.
TEST(AdaptiveMacroIntegratorTest, AccumulatedTimeIsMonotone) {
    System sys(1);
    sys.setPosition(0, 5.0, 5.0);
    Box             box(10.0);
    ForceCalculator fc(1.0, 1.0);
    RandomGenerator rng(0);

    AdaptiveMacroIntegrator ai(1.0);
    ai.setKBT(0.1);
    ai.setRotationalDiffusion(0.5);

    double t = 0.0;
    for (int i = 0; i < 10; ++i) {
        const double dt = ai.step(sys, box, fc, rng);
        EXPECT_GT(dt, 0.0);
        t += dt;
    }
    EXPECT_GT(t, 0.0);
}

// ---- Bridge sanity check ---------------------------------------------------

// The Brownian bridge formula: η_a = (η + ~η)/sqrt(2), η_b = (η - ~η)/sqrt(2)
// must give unit-variance Gaussians and zero cross-correlation. Test via the
// public API: with kT > 0, gamma = 1, the per-step T-noise variance must
// match 2*kT*dt regardless of how the controller chose to split. We do this
// by running step() once on a single particle with no force / no rotation
// and verifying that <r²>/<dt> ≈ 4*kT/gamma over many independent macro
// steps.
TEST(AdaptiveMacroIntegratorTest, MacroStepNoiseVarianceIsConsistent) {
    const double gamma = 1.0;
    const double kT    = 1.0;
    const std::size_t M = 5000;       // independent macro-step samples
    const double L = 1.0e6;            // huge box so PBC never wraps

    Box             box(L);
    ForceCalculator fc(1.0, 1.0);
    RandomGenerator rng(13);

    double sum_r2_over_dt = 0.0;
    for (std::size_t k = 0; k < M; ++k) {
        System sys(1);
        sys.setPosition(0, L / 2.0, L / 2.0);
        AdaptiveMacroIntegrator ai(gamma);
        ai.setKBT(kT);

        const double dt = ai.step(sys, box, fc, rng);
        const double dx = sys.getX(0) - L / 2.0;
        const double dy = sys.getY(0) - L / 2.0;
        sum_r2_over_dt += (dx * dx + dy * dy) / dt;
    }
    const double mean = sum_r2_over_dt / static_cast<double>(M);
    const double expected = 4.0 * kT / gamma;
    // 4-sigma slack at this M.
    EXPECT_NEAR(mean, expected, expected * 4.0 / std::sqrt(static_cast<double>(M)));
}
