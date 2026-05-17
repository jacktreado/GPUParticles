#include "Box.hpp"
#include "Integrator.hpp"
#include "RandomGenerator.hpp"
#include "System.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <cstddef>

// =============================================================================
// Integrator unit tests
// -----------------------------------------------------------------------------
// Parametrization: gamma (friction) is the input.  Mobility mu = 1/gamma is
// independent of kT; noise amplitude = sqrt(2*kT/gamma*dt) (Einstein relation).
// =============================================================================

// ---- Cached-constant derivations -------------------------------------------

// The default constructor uses = default and leaves cached constants at 0.
// Cached constants are only populated by the parametric constructor (which
// calls recomputeCachedConstants).
TEST(IntegratorTest, DefaultConstructorCachedConstantsAreZero) {
    Integrator integ;
    EXPECT_DOUBLE_EQ(integ.getMobilityTimesDt(), 0.0);
    EXPECT_DOUBLE_EQ(integ.getNoiseAmplitude(),  0.0);
}

TEST(IntegratorTest, ParametricCachedConstants) {
    Integrator integ(1.0, 1e-4);   // gamma=1, dt=1e-4, kT=1 (default)
    EXPECT_DOUBLE_EQ(integ.getMobilityTimesDt(), (1.0 / 1.0) * 1e-4);
    EXPECT_DOUBLE_EQ(integ.getNoiseAmplitude(), std::sqrt(2.0 * 1.0 / 1.0 * 1e-4));
}

TEST(IntegratorTest, ConstructorCachedConstants) {
    const double gamma = 0.4, dt = 0.001, kT = 1.0;
    Integrator integ(gamma, dt);
    EXPECT_DOUBLE_EQ(integ.getMobilityTimesDt(), (1.0 / gamma) * dt);
    EXPECT_DOUBLE_EQ(integ.getNoiseAmplitude(), std::sqrt(2.0 * kT / gamma * dt));
}

TEST(IntegratorTest, SetFrictionUpdatesCache) {
    Integrator integ(1.0, 1e-4);
    integ.setFriction(0.25);   // gamma = 0.25 => mobility = 4
    EXPECT_DOUBLE_EQ(integ.getMobilityTimesDt(), (1.0 / 0.25) * 1e-4);
    EXPECT_DOUBLE_EQ(integ.getNoiseAmplitude(), std::sqrt(2.0 * 1.0 / 0.25 * 1e-4));
}

TEST(IntegratorTest, SetTimestepUpdatesCache) {
    Integrator integ(1.0, 1e-4);
    integ.setTimestep(1e-3);
    EXPECT_DOUBLE_EQ(integ.getMobilityTimesDt(), (1.0 / 1.0) * 1e-3);
    EXPECT_DOUBLE_EQ(integ.getNoiseAmplitude(), std::sqrt(2.0 * 1.0 / 1.0 * 1e-3));
}

TEST(IntegratorTest, SetKBTUpdatesCache) {
    Integrator integ(1.0, 1e-4);
    integ.setKBT(2.0);
    // mobility_dt = (1/gamma) * dt — independent of kT
    EXPECT_DOUBLE_EQ(integ.getMobilityTimesDt(), (1.0 / 1.0) * 1e-4);
    // noise amplitude DOES depend on kT via the Einstein relation
    EXPECT_DOUBLE_EQ(integ.getNoiseAmplitude(), std::sqrt(2.0 * 2.0 / 1.0 * 1e-4));
}

TEST(IntegratorTest, AthermalLimitNoNoise) {
    // kT = 0 is the athermal limit: mobility well-defined, noise vanishes.
    Integrator integ(1.0, 1e-4);
    integ.setKBT(0.0);
    EXPECT_DOUBLE_EQ(integ.getMobilityTimesDt(), (1.0 / 1.0) * 1e-4);
    EXPECT_DOUBLE_EQ(integ.getNoiseAmplitude(), 0.0);
}

// ---- Drift term (F=0 only verifies noise is applied, F!=0 verifies drift) --

TEST(IntegratorTest, DriftFromForce) {
    // A single particle with a known force, ensemble-averaged over many trials.
    const double gamma = 1.0;
    const double dt    = 1e-6;     // tiny dt => tiny noise, drift dominates
    const double F     = 100.0;
    const double kT    = 1.0;
    const double L     = 100.0;

    Integrator integ(gamma, dt);
    integ.setKBT(kT);
    Box box(L);
    RandomGenerator rng(42);

    const int N_trials = 1000;
    double sum_dx = 0.0;
    for (int t = 0; t < N_trials; ++t) {
        System sys(1);
        sys.setPosition(0, 50.0, 50.0);
        sys.setForce(0, F, 0.0);

        integ.step(sys, box, rng);

        // Positions are unwrapped, so the raw difference is the true displacement.
        const double dx = sys.getX(0) - 50.0;
        sum_dx += dx;
    }
    const double mean_dx  = sum_dx / N_trials;
    const double expected = (1.0 / gamma) * dt * F;   // mobility_dt * F

    // Standard error of the mean: sigma_noise / sqrt(N_trials).
    const double sigma_noise = std::sqrt(2.0 * kT / gamma * dt);
    const double tol = 5.0 * sigma_noise / std::sqrt(double(N_trials));
    EXPECT_NEAR(mean_dx, expected, tol);
}

// ---- Positions drift freely across the boundary ----------------------------
// Stored positions are NOT auto-wrapped (see Box.hpp). Particles started near
// the box corner with kT > 0 will eventually leak across the boundary; the
// engine must tolerate that without producing NaNs or runaway state.
TEST(IntegratorTest, PositionsDriftFreelyAcrossBoundary) {
    const double L = 5.0;
    const std::size_t N = 32;
    Integrator integ(1.0, 1e-2);   // gamma=1, large dt so noise easily exits
    integ.setKBT(1.0);
    Box box(L);
    RandomGenerator rng(99);

    System sys(N);
    for (std::size_t i = 0; i < N; ++i) sys.setPosition(i, L - 0.01, L - 0.01);

    for (int step = 0; step < 10; ++step) {
        integ.step(sys, box, rng);
        for (std::size_t i = 0; i < N; ++i) {
            EXPECT_TRUE(std::isfinite(sys.getX(i))) << "step=" << step << " i=" << i;
            EXPECT_TRUE(std::isfinite(sys.getY(i))) << "step=" << step << " i=" << i;
        }
    }
}

// ---- Mean-squared displacement (statistical) --------------------------------
// With zero force, expected 2D MSD after T steps is 4*D*T*dt with D = kT/gamma.

TEST(IntegratorTest, MSDMatchesDiffusionCoefficient) {
    const double gamma = 1.0;
    const double dt    = 1e-3;
    const double kT    = 1.0;
    const double D     = kT / gamma;   // Einstein relation
    const double L     = 1000.0;
    const std::size_t N = 500;
    const int    T  = 100;

    Integrator integ(gamma, dt);
    integ.setKBT(kT);
    Box box(L);
    RandomGenerator rng(7);

    System sys(N);
    for (std::size_t i = 0; i < N; ++i) sys.setPosition(i, L / 2.0, L / 2.0);

    for (int t = 0; t < T; ++t) integ.step(sys, box, rng);

    double msd = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double dx = sys.getX(i) - L / 2.0;
        const double dy = sys.getY(i) - L / 2.0;
        msd += dx * dx + dy * dy;
    }
    msd /= static_cast<double>(N);

    const double expected_msd = 4.0 * D * dt * T;
    const double sigma = expected_msd / std::sqrt(static_cast<double>(N) / 2.0);
    EXPECT_NEAR(msd, expected_msd, 3.0 * sigma);
}

// ---- getters ----------------------------------------------------------------

TEST(IntegratorTest, Getters) {
    Integrator integ(2.0, 0.001);   // gamma=2
    integ.setKBT(1.5);
    EXPECT_DOUBLE_EQ(integ.getFriction(),   2.0);
    EXPECT_DOUBLE_EQ(integ.getMobility(),   1.0 / 2.0);
    EXPECT_DOUBLE_EQ(integ.getDiffusionCoefficient(), 1.5 / 2.0);   // D = kT/gamma
    EXPECT_DOUBLE_EQ(integ.getTimestep(),   0.001);
    EXPECT_DOUBLE_EQ(integ.getKBT(),        1.5);
}

// =============================================================================
// kickStep — operator-split companion (active + spring + noise + rotation +
//            anchor at a caller-supplied dt, no WCA).
// =============================================================================

// dt = 0 is a no-op.
TEST(IntegratorTest, KickStepZeroDtIsNoOp) {
    Integrator integ(1.0, 1.0);     // member dt is irrelevant for kickStep
    integ.setActiveForce(1.0);
    integ.setKBT(1.0);
    Box box(100.0);
    RandomGenerator rng(0);

    System sys(1);
    sys.setPosition(0, 50.0, 50.0);
    sys.setTheta(0, 0.0);

    integ.kickStep(sys, box, rng, 0.0);
    EXPECT_DOUBLE_EQ(sys.getX(0), 50.0);
    EXPECT_DOUBLE_EQ(sys.getY(0), 50.0);
}

// Active force kick: with kT=0 (no noise), one step at dt advances along
// the orientation by mu * f0 * dt.
TEST(IntegratorTest, KickStepActiveForceDriftIsDeterministic) {
    const double gamma = 1.0;
    const double f0    = 0.5;
    const double dt    = 1e-3;
    Integrator integ(gamma, 1.0);    // member dt is unused by kickStep
    integ.setKBT(0.0);
    integ.setActiveForce(f0);

    Box box(100.0);
    RandomGenerator rng(0);
    System sys(1);
    sys.setPosition(0, 50.0, 50.0);
    sys.setTheta(0, 0.0);            // pointing +x

    integ.kickStep(sys, box, rng, dt);
    EXPECT_DOUBLE_EQ(sys.getX(0), 50.0 + f0 * dt / gamma);
    EXPECT_DOUBLE_EQ(sys.getY(0), 50.0);
}

// kickStep does NOT read fx/fy: stale or absent WCA force in the System
// must not affect the kick.
TEST(IntegratorTest, KickStepIgnoresWCAForce) {
    Integrator integ(1.0, 1.0);
    integ.setKBT(0.0);
    integ.setActiveForce(0.0);       // no active drift either

    Box box(100.0);
    RandomGenerator rng(0);
    System sys(1);
    sys.setPosition(0, 50.0, 50.0);
    sys.setForce(0, 999.0, 999.0);   // bogus stale forces

    integ.kickStep(sys, box, rng, 1e-3);
    // No active, no spring, no kT, no rotation -> nothing should move.
    EXPECT_DOUBLE_EQ(sys.getX(0), 50.0);
    EXPECT_DOUBLE_EQ(sys.getY(0), 50.0);
}

// Noise amplitude is computed from the supplied dt, not the member dt_.
// Increasing dt by 4x increases the noise variance by 4x (amplitude by 2x).
TEST(IntegratorTest, KickStepNoiseAmplitudeUsesSuppliedDt) {
    const double gamma = 1.0;
    const double kT    = 1.0;
    const std::size_t N = 5000;

    auto run = [&](double dt) {
        Integrator integ(gamma, 1.0);   // member dt is unused
        integ.setKBT(kT);
        Box box(1e6);
        RandomGenerator rng(123);
        System sys(N);
        for (std::size_t i = 0; i < N; ++i) sys.setPosition(i, 5e5, 5e5);
        integ.kickStep(sys, box, rng, dt);
        double msd = 0.0;
        for (std::size_t i = 0; i < N; ++i) {
            const double dx = sys.getX(i) - 5e5;
            const double dy = sys.getY(i) - 5e5;
            msd += dx * dx + dy * dy;
        }
        return msd / static_cast<double>(N);
    };

    const double msd_small = run(1e-4);
    const double msd_big   = run(4e-4);
    // Variance scales linearly with dt, so msd ratio should be ~4. Allow 25%
    // statistical slack for finite N.
    EXPECT_NEAR(msd_big / msd_small, 4.0, 1.0);
}

// Rotational kick at supplied dt: mean theta drift is zero, variance grows
// linearly with dt and matches 2 * D_r * dt.
TEST(IntegratorTest, KickStepRotationalDiffusion) {
    const double D_r = 1.0;
    const double dt  = 0.01;
    const std::size_t N = 4000;

    Integrator integ(1.0, 1.0);
    integ.setKBT(0.0);
    integ.setRotationalDiffusion(D_r);

    Box box(1e3);
    RandomGenerator rng(7);
    System sys(N);
    for (std::size_t i = 0; i < N; ++i) {
        sys.setPosition(i, 500.0, 500.0);
        sys.setTheta(i, 0.0);
    }

    integ.kickStep(sys, box, rng, dt);

    double m1 = 0.0, m2 = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double th = sys.getTheta(i);
        m1 += th;
        m2 += th * th;
    }
    m1 /= static_cast<double>(N);
    m2 /= static_cast<double>(N);
    EXPECT_NEAR(m1, 0.0, 0.1);
    EXPECT_NEAR(m2 - m1 * m1, 2.0 * D_r * dt, 0.5 * 2.0 * D_r * dt);
}

// Spring kick: with kT=0, particle on spring relaxes toward anchor.
TEST(IntegratorTest, KickStepSpringRelaxation) {
    const double gamma = 1.0;
    const double k_a   = 10.0;
    const double dt    = 1e-3;

    Integrator integ(gamma, 1.0);
    integ.setKBT(0.0);
    integ.setSpringStiffness(k_a);
    integ.setAnchorFriction(1.0);    // anchor-on flag; gamma_a > 0 enables anchor evolution

    Box box(100.0);
    RandomGenerator rng(0);
    System sys(1);
    sys.setPosition(0, 50.5, 50.0);
    sys.setAnchor  (0, 50.0, 50.0);  // particle 0.5 to the right of its anchor

    const double x0_before = sys.getX(0);
    integ.kickStep(sys, box, rng, dt);

    // Spring force on particle: F = -k_a * (r - a) = -10 * 0.5 = -5 (pulls left).
    // With kT=0, displacement is mu * F * dt = -5 * 1e-3.
    EXPECT_DOUBLE_EQ(sys.getX(0), x0_before - k_a * 0.5 * dt / gamma);
}
