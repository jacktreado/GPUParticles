#include "Box.hpp"
#include "ForceCalculator.hpp"
#include "Initializer.hpp"
#include "Integrator.hpp"
#include "RandomGenerator.hpp"
#include "System.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <cstddef>

// =============================================================================
// Active Brownian Particle (ABP) integrator unit tests
// =============================================================================

// ---- Getter/setter round-trip -----------------------------------------------

TEST(ActiveIntegratorTest, SetGetActiveForce) {
    Integrator integ(1.0, 1e-4);
    EXPECT_DOUBLE_EQ(integ.getActiveForce(), 0.0);
    integ.setActiveForce(5.0);
    EXPECT_DOUBLE_EQ(integ.getActiveForce(), 5.0);
}

TEST(ActiveIntegratorTest, SetGetRotationalDiffusion) {
    Integrator integ(1.0, 1e-4);
    EXPECT_DOUBLE_EQ(integ.getRotationalDiffusion(), 0.0);
    integ.setRotationalDiffusion(2.0);
    EXPECT_DOUBLE_EQ(integ.getRotationalDiffusion(), 2.0);
    EXPECT_DOUBLE_EQ(integ.getRotNoiseAmplitude(),
                     std::sqrt(2.0 * 2.0 * 1e-4));
}

TEST(ActiveIntegratorTest, RotNoiseAmplitudeUpdatesWithDt) {
    Integrator integ(1.0, 1e-4);
    integ.setRotationalDiffusion(3.0);
    integ.setTimestep(1e-3);
    EXPECT_DOUBLE_EQ(integ.getRotNoiseAmplitude(),
                     std::sqrt(2.0 * 3.0 * 1e-3));
}

// ---- Passive limit: f0=0 gives same behavior as original integrator ---------

TEST(ActiveIntegratorTest, ZeroActiveForceNoExtraDrift) {
    // With f0=0 the active term is zero. Two runs from the same seed must be
    // identical regardless of theta values.
    const double L  = 50.0;
    const std::size_t N = 4;

    System sys1(N), sys2(N);
    for (std::size_t i = 0; i < N; ++i) {
        sys1.setPosition(i, 5.0 * (i + 1), 5.0 * (i + 1));
        sys2.setPosition(i, 5.0 * (i + 1), 5.0 * (i + 1));
        // Different theta to confirm it's ignored when f0=0.
        sys1.setTheta(i, 0.0);
        sys2.setTheta(i, 1.234);
    }

    Box box(L);
    Integrator integ(1.0, 1e-4);   // f0 defaults to 0
    RandomGenerator rng1(42), rng2(42);

    integ.step(sys1, box, rng1);
    integ.step(sys2, box, rng2);

    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(sys1.getX(i), sys2.getX(i)) << "i=" << i;
        EXPECT_DOUBLE_EQ(sys1.getY(i), sys2.getY(i)) << "i=" << i;
    }
}

// ---- Active force moves particles in the correct direction ------------------

TEST(ActiveIntegratorTest, ActiveForceDriftDirection) {
    // A single particle with theta=0 (pointing in +x). With zero WCA force and
    // large f0, the particle should drift in +x on average over many trials.
    const double gamma = 1.0;
    const double dt    = 1e-5;
    const double f0    = 100.0;
    const double kT    = 1.0;
    const double L     = 1000.0;

    Integrator integ(gamma, dt);
    integ.setKBT(kT);
    integ.setActiveForce(f0);

    const int N_trials = 500;
    double mean_dx = 0.0;

    for (int t = 0; t < N_trials; ++t) {
        System sys(1);
        sys.setPosition(0, L / 2.0, L / 2.0);
        sys.setTheta(0, 0.0);   // pointing in +x
        sys.setForce(0, 0.0, 0.0);

        Box box(L);
        RandomGenerator rng(t + 1);
        integ.step(sys, box, rng);

        mean_dx += sys.getX(0) - L / 2.0;
    }
    mean_dx /= N_trials;

    // Expected drift = mu * f0 * dt = (1/gamma) * f0 * dt = 1e-3
    const double expected = (1.0 / gamma) * f0 * dt;
    const double noise_sigma = std::sqrt(2.0 * kT / gamma * dt) / std::sqrt(double(N_trials));
    EXPECT_NEAR(mean_dx, expected, 5.0 * noise_sigma);
}

TEST(ActiveIntegratorTest, ActiveForceThetaPiDriftsNegativeX) {
    // theta = pi => e_theta = (-1, 0) => drift in -x
    const double gamma = 1.0;
    const double dt = 1e-5, f0 = 100.0, kT = 1.0, L = 1000.0;

    Integrator integ(gamma, dt);
    integ.setKBT(kT);
    integ.setActiveForce(f0);

    const int N_trials = 500;
    double mean_dx = 0.0;
    const double pi = std::acos(-1.0);

    for (int t = 0; t < N_trials; ++t) {
        System sys(1);
        sys.setPosition(0, L / 2.0, L / 2.0);
        sys.setTheta(0, pi);   // pointing in -x
        sys.setForce(0, 0.0, 0.0);
        Box box(L);
        RandomGenerator rng(t + 100);
        integ.step(sys, box, rng);
        mean_dx += sys.getX(0) - L / 2.0;
    }
    mean_dx /= N_trials;

    const double expected = -(1.0 / gamma) * f0 * dt;
    const double noise_sigma = std::sqrt(2.0 * kT / gamma * dt) / std::sqrt(double(N_trials));
    EXPECT_NEAR(mean_dx, expected, 5.0 * noise_sigma);
}

// ---- Velocity is stored correctly ------------------------------------------

TEST(ActiveIntegratorTest, VelocityEqualsDisplacementOverDt) {
    // The stored velocity vx[i] must match (x_new - x_old) / dt to within a
    // few ULPs. Exact bit-equality cannot be guaranteed because (a + b) - a
    // differs from b by one ULP in IEEE 754, and 1.0/dt * dt != 1.0 for
    // non-power-of-two dt values. A relative tolerance of 1e-12 is tight
    // enough to catch any logic errors while tolerating fp rounding.
    const double L = 100.0;
    System sys(2);
    sys.setPosition(0, 20.0, 20.0);
    sys.setPosition(1, 80.0, 80.0);
    sys.setTheta(0, 0.3);
    sys.setTheta(1, 1.1);
    sys.setForce(0, 1.0, -0.5);
    sys.setForce(1, -2.0, 3.0);

    const double dt = 1e-4;
    Box box(L);
    Integrator integ(1.0, dt);
    integ.setKBT(1.0);
    integ.setActiveForce(2.0);
    RandomGenerator rng(7);

    const double x0_0 = sys.getX(0), y0_0 = sys.getY(0);
    const double x0_1 = sys.getX(1), y0_1 = sys.getY(1);

    integ.step(sys, box, rng);

    // Compute actual displacement (no wrap expected: max |Δr| << L/2)
    const double dx0 = sys.getX(0) - x0_0;
    const double dy0 = sys.getY(0) - y0_0;
    const double dx1 = sys.getX(1) - x0_1;
    const double dy1 = sys.getY(1) - y0_1;

    const double tol = 1e-10;   // relative: velocity ~ O(1), tol ~ 1e-10
    EXPECT_NEAR(sys.getVx(0), dx0 / dt, tol * std::abs(dx0 / dt) + 1e-15);
    EXPECT_NEAR(sys.getVy(0), dy0 / dt, tol * std::abs(dy0 / dt) + 1e-15);
    EXPECT_NEAR(sys.getVx(1), dx1 / dt, tol * std::abs(dx1 / dt) + 1e-15);
    EXPECT_NEAR(sys.getVy(1), dy1 / dt, tol * std::abs(dy1 / dt) + 1e-15);
}

// ---- Rotational diffusion ---------------------------------------------------

TEST(ActiveIntegratorTest, ZeroRotDiffThetaUnchanged) {
    // With D_r = 0, theta must remain exactly constant.
    const std::size_t N = 16;
    System sys(N);
    Box box(50.0);
    for (std::size_t i = 0; i < N; ++i) {
        sys.setPosition(i, 5.0, 5.0);
        sys.setTheta(i, 0.1 * static_cast<double>(i));
    }

    Integrator integ(1.0, 1e-4);
    integ.setRotationalDiffusion(0.0);   // no tumbling
    RandomGenerator rng(99);

    for (int step = 0; step < 10; ++step) {
        integ.step(sys, box, rng);
    }

    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(sys.getTheta(i), 0.1 * static_cast<double>(i))
            << "i=" << i;
    }
}

TEST(ActiveIntegratorTest, NonzeroRotDiffChangesTheta) {
    // With D_r > 0, theta values must change after enough steps.
    const std::size_t N = 8;
    System sys(N);
    Box box(50.0);
    for (std::size_t i = 0; i < N; ++i) {
        sys.setPosition(i, 5.0, 5.0);
        sys.setTheta(i, 0.0);
    }

    Integrator integ(1.0, 1e-4);
    integ.setRotationalDiffusion(1.0);   // D_r = 1
    RandomGenerator rng(55);

    integ.step(sys, box, rng);

    bool any_changed = false;
    for (std::size_t i = 0; i < N; ++i) {
        if (sys.getTheta(i) != 0.0) { any_changed = true; break; }
    }
    EXPECT_TRUE(any_changed);
}

TEST(ActiveIntegratorTest, ThetaMSDMatchesRotDiffusion) {
    // Over many particles and steps, the MSD of theta should approach
    // 2 * D_r * T * dt.
    const std::size_t N   = 500;
    const int         T   = 100;
    const double      D_r = 2.0;
    const double      dt  = 1e-3;
    const double      L   = 1000.0;

    System sys(N);
    Box box(L);
    for (std::size_t i = 0; i < N; ++i) {
        sys.setPosition(i, L / 2.0, L / 2.0);
        sys.setTheta(i, 0.0);
    }

    Integrator integ(1.0, dt);
    integ.setRotationalDiffusion(D_r);
    RandomGenerator rng(3);

    for (int t = 0; t < T; ++t) integ.step(sys, box, rng);

    double msd_theta = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double dtheta = sys.getTheta(i);   // started at 0
        msd_theta += dtheta * dtheta;
    }
    msd_theta /= static_cast<double>(N);

    const double expected = 2.0 * D_r * dt * T;
    // 3-sigma tolerance
    const double sigma = expected / std::sqrt(static_cast<double>(N) / 2.0);
    EXPECT_NEAR(msd_theta, expected, 3.0 * sigma);
}

// ---- Positions remain in box after ABP step --------------------------------

// Stored positions are NOT auto-wrapped (see Box.hpp). With strong active
// drift from the near-corner start, particles will exit the box quickly; the
// engine must remain finite and well-behaved.
TEST(ActiveIntegratorTest, PositionsDriftFreelyAfterActiveStep) {
    const double L = 5.0;
    const std::size_t N = 32;
    Integrator integ(1.0, 1e-2);
    integ.setActiveForce(10.0);
    integ.setRotationalDiffusion(1.0);
    Box box(L);
    RandomGenerator rng(7);

    System sys(N);
    for (std::size_t i = 0; i < N; ++i) {
        sys.setPosition(i, L - 0.01, L - 0.01);
        sys.setTheta(i, 0.0);
    }

    for (int step = 0; step < 10; ++step) {
        integ.step(sys, box, rng);
        for (std::size_t i = 0; i < N; ++i) {
            EXPECT_TRUE(std::isfinite(sys.getX(i))) << "step=" << step << " i=" << i;
            EXPECT_TRUE(std::isfinite(sys.getY(i))) << "step=" << step << " i=" << i;
        }
    }
}

// ---- randomizeOrientations fills [0, 2*pi) ----------------------------------

TEST(ActiveIntegratorTest, RandomizeOrientationsInRange) {
    const std::size_t N = 256;
    System sys(N);
    Box box(20.0);
    Initializer::placeOnLattice(sys, box);

    RandomGenerator rng(42);
    Initializer::randomizeOrientations(sys, rng);

    const double two_pi = 2.0 * std::acos(-1.0);
    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_GE(sys.getTheta(i), 0.0)     << "i=" << i;
        EXPECT_LT(sys.getTheta(i), two_pi)  << "i=" << i;
    }
}

TEST(ActiveIntegratorTest, RandomizeOrientationsNotAllZero) {
    const std::size_t N = 64;
    System sys(N);
    Box box(20.0);
    Initializer::placeOnLattice(sys, box);

    RandomGenerator rng(17);
    Initializer::randomizeOrientations(sys, rng);

    bool all_zero = true;
    for (std::size_t i = 0; i < N; ++i) {
        if (sys.getTheta(i) != 0.0) { all_zero = false; break; }
    }
    EXPECT_FALSE(all_zero);
}

// =============================================================================
// Anchor (spring tether) tests
// =============================================================================

TEST(AnchoredIntegratorTest, PlaceAnchorsAtParticles) {
    const std::size_t N = 16;
    System sys(N);
    Box box(20.0);
    for (std::size_t i = 0; i < N; ++i) {
        sys.setPosition(i, 1.5 * (i + 1), 2.0 * (i + 1));
    }
    Initializer::placeAnchorsAtParticles(sys);
    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(sys.getAx(i), sys.getX(i)) << "i=" << i;
        EXPECT_DOUBLE_EQ(sys.getAy(i), sys.getY(i)) << "i=" << i;
    }
}

TEST(AnchoredIntegratorTest, GettersDefaults) {
    Integrator integ(1.0, 1e-4);
    EXPECT_DOUBLE_EQ(integ.getSpringStiffness(), 0.0);
    EXPECT_DOUBLE_EQ(integ.getAnchorFriction(),  0.0);
    EXPECT_DOUBLE_EQ(integ.getAnchorMobilityTimesDt(),  0.0);
    EXPECT_DOUBLE_EQ(integ.getAnchorNoiseAmplitude(),   0.0);
}

TEST(AnchoredIntegratorTest, AnchorCachesUpdate) {
    Integrator integ(1.0, 1e-3);
    integ.setKBT(2.0);
    integ.setAnchorFriction(4.0);
    EXPECT_DOUBLE_EQ(integ.getAnchorMobilityTimesDt(), (1.0 / 4.0) * 1e-3);
    EXPECT_DOUBLE_EQ(integ.getAnchorNoiseAmplitude(),
                     std::sqrt(2.0 * (2.0 / 4.0) * 1e-3));
}

// ---- Spring force pulls particle toward anchor ------------------------------

TEST(AnchoredIntegratorTest, SpringPullsParticleTowardAnchor) {
    // Particle at (5, 0), anchor at (0, 0). With k_a > 0 and gamma_a = 0
    // (frozen anchor), no active force, and tiny noise (small dt), the
    // particle must drift in -x (toward the anchor).
    const double L  = 100.0;
    const double dt = 1e-5;       // tiny dt => small noise
    const double k_a = 1.0;

    Integrator integ(1.0, dt);
    integ.setKBT(1.0);
    integ.setSpringStiffness(k_a);
    integ.setAnchorFriction(0.0);   // frozen anchor

    const int N_trials = 500;
    double mean_dx = 0.0;
    for (int t = 0; t < N_trials; ++t) {
        System sys(1);
        sys.setPosition(0, L / 2.0 + 5.0, L / 2.0);
        sys.setAnchor  (0, L / 2.0,        L / 2.0);
        sys.setForce   (0, 0.0, 0.0);

        Box box(L);
        RandomGenerator rng(t + 1);
        integ.step(sys, box, rng);

        mean_dx += sys.getX(0) - (L / 2.0 + 5.0);
    }
    mean_dx /= N_trials;

    // Expected drift = mu * (-k_a * 5) * dt = -5e-5
    const double expected = -(1.0 / 1.0) * k_a * 5.0 * dt;
    const double noise_sigma = std::sqrt(2.0 * 1.0 * dt) / std::sqrt(double(N_trials));
    EXPECT_NEAR(mean_dx, expected, 5.0 * noise_sigma);
}

// ---- Newton's third law: anchor is pulled toward particle -------------------

TEST(AnchoredIntegratorTest, AnchorPulledTowardParticle) {
    // Same geometry as above but anchor mobile (gamma_a > 0).  Take the mean
    // anchor displacement; should drift in +x (toward particle at +5).
    const double L   = 100.0;
    const double dt  = 1e-5;
    const double k_a = 1.0;
    const double gamma_a = 1.0;

    Integrator integ(1.0, dt);
    integ.setKBT(1.0);
    integ.setSpringStiffness(k_a);
    integ.setAnchorFriction(gamma_a);

    const int N_trials = 500;
    double mean_dax = 0.0;
    for (int t = 0; t < N_trials; ++t) {
        System sys(1);
        sys.setPosition(0, L / 2.0 + 5.0, L / 2.0);
        sys.setAnchor  (0, L / 2.0,        L / 2.0);

        Box box(L);
        RandomGenerator rng(t + 100);
        integ.step(sys, box, rng);

        mean_dax += sys.getAx(0) - (L / 2.0);
    }
    mean_dax /= N_trials;

    // Expected anchor drift = (1/gamma_a) * k_a * 5 * dt
    const double expected = (1.0 / gamma_a) * k_a * 5.0 * dt;
    const double anchor_noise = std::sqrt(2.0 * (1.0 / gamma_a) * dt)
                              / std::sqrt(double(N_trials));
    EXPECT_NEAR(mean_dax, expected, 5.0 * anchor_noise);
}

// ---- Frozen anchor (gamma_a = 0) doesn't move -------------------------------

TEST(AnchoredIntegratorTest, FrozenAnchorDoesNotMove) {
    const std::size_t N = 8;
    System sys(N);
    Box box(50.0);
    for (std::size_t i = 0; i < N; ++i) {
        sys.setPosition(i, 25.0 + 0.5 * i, 25.0);
        sys.setAnchor  (i, 1.0 + 0.1 * i,  2.0);   // far from particles
    }

    Integrator integ(1.0, 1e-3);
    integ.setSpringStiffness(0.5);    // active spring
    integ.setAnchorFriction(0.0);     // frozen
    RandomGenerator rng(13);

    for (int s = 0; s < 50; ++s) integ.step(sys, box, rng);

    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(sys.getAx(i), 1.0 + 0.1 * static_cast<double>(i)) << "i=" << i;
        EXPECT_DOUBLE_EQ(sys.getAy(i), 2.0)                                 << "i=" << i;
    }
}

// ---- Free anchor (k_a = 0, gamma_a > 0) diffuses with the right MSD ---------

TEST(AnchoredIntegratorTest, FreeAnchorMSDMatchesItsDiffusion) {
    // With k_a = 0 the anchor is decoupled from the particle and diffuses
    // freely with D_a = kT / gamma_a.  Check ensemble MSD ~ 4 * D_a * T * dt.
    const double kT      = 1.0;
    const double gamma_a = 2.0;
    const double dt      = 1e-3;
    const double L       = 1000.0;
    const std::size_t N = 500;
    const int T          = 100;

    System sys(N);
    Box box(L);
    for (std::size_t i = 0; i < N; ++i) {
        sys.setPosition(i, L / 2.0, L / 2.0);
        sys.setAnchor  (i, L / 2.0, L / 2.0);
    }

    Integrator integ(1.0, dt);
    integ.setKBT(kT);
    integ.setSpringStiffness(0.0);          // decoupled
    integ.setAnchorFriction(gamma_a);
    RandomGenerator rng(2026);

    for (int t = 0; t < T; ++t) integ.step(sys, box, rng);

    double msd = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        double dx = sys.getAx(i) - L / 2.0;
        double dy = sys.getAy(i) - L / 2.0;
        box.minimumImage(dx, dy);
        msd += dx * dx + dy * dy;
    }
    msd /= static_cast<double>(N);

    const double D_a      = kT / gamma_a;
    const double expected = 4.0 * D_a * dt * T;
    const double sigma    = expected / std::sqrt(static_cast<double>(N) / 2.0);
    EXPECT_NEAR(msd, expected, 3.0 * sigma);
}

// ---- Equipartition: <|r-a|^2> = 2*kT/k_a in 2D for frozen anchor ------------

TEST(AnchoredIntegratorTest, EquipartitionParticleSpread) {
    // Many particles, each tethered to a frozen anchor at the same point.
    // After equilibration, the variance of (r - a) should be 2 * kT / k_a
    // (sum of x and y variances, each = kT/k_a).
    const double kT  = 1.0;
    const double k_a = 4.0;            // tight spring => fast equilibration
    const double dt  = 1e-3;
    const double L   = 100.0;
    const std::size_t N = 800;
    const int    T_burn   = 5000;
    const int    T_sample = 5000;

    System sys(N);
    Box box(L);
    for (std::size_t i = 0; i < N; ++i) {
        sys.setPosition(i, L / 2.0, L / 2.0);
        sys.setAnchor  (i, L / 2.0, L / 2.0);
    }

    Integrator integ(1.0, dt);
    integ.setKBT(kT);
    integ.setSpringStiffness(k_a);
    integ.setAnchorFriction(0.0);   // frozen anchors at L/2
    RandomGenerator rng(99);

    // Burn-in to thermalize the spring DOF.
    for (int t = 0; t < T_burn; ++t) integ.step(sys, box, rng);

    // Time-average of <|r-a|^2> over many samples.
    double sum = 0.0;
    int    cnt = 0;
    for (int t = 0; t < T_sample; ++t) {
        integ.step(sys, box, rng);
        if (t % 50 == 0) {     // sub-sample to reduce correlations
            for (std::size_t i = 0; i < N; ++i) {
                double dx = sys.getX(i) - sys.getAx(i);
                double dy = sys.getY(i) - sys.getAy(i);
                box.minimumImage(dx, dy);
                sum += dx * dx + dy * dy;
                ++cnt;
            }
        }
    }
    const double mean = sum / static_cast<double>(cnt);
    const double expected = 2.0 * kT / k_a;   // 2 DOFs * kT/k_a
    // 5% tolerance — equipartition is statistical, and our integrator
    // discretization introduces a small O(dt*k_a*mu) bias.
    EXPECT_NEAR(mean, expected, 0.05 * expected);
}

// ---- Spring respects PBC minimum image -------------------------------------

TEST(AnchoredIntegratorTest, SpringForceUsesMinimumImage) {
    // Particle just inside the +x boundary, anchor just inside the -x
    // boundary.  Bare separation is L - 0.2; minimum-image separation is 0.2
    // (across the periodic wall).  The spring should pull the particle in +x
    // (across the wall) toward the anchor — *not* pull it back across the box.
    const double L   = 10.0;
    const double dt  = 1e-5;
    const double k_a = 1.0;

    Integrator integ(1.0, dt);
    integ.setKBT(1.0);
    integ.setSpringStiffness(k_a);
    integ.setAnchorFriction(0.0);   // freeze anchor

    const int N_trials = 500;
    double mean_dx = 0.0;
    for (int t = 0; t < N_trials; ++t) {
        System sys(1);
        sys.setPosition(0, L - 0.1, L / 2.0);   // just left of +x wall
        sys.setAnchor  (0, 0.1,     L / 2.0);   // just right of -x wall
        sys.setForce   (0, 0.0, 0.0);

        Box box(L);
        RandomGenerator rng(t + 500);

        const double x0 = sys.getX(0);
        integ.step(sys, box, rng);

        // Displacement using minimum image (handles wrap if it occurred)
        double dx = sys.getX(0) - x0;
        box.minimumImage(dx, dx);
        mean_dx += dx;
    }
    mean_dx /= N_trials;

    // Minimum-image separation = +0.2 (anchor is "to the right" through the
    // boundary), so spring force on particle is -k*(+0.2) = -0.2.
    // Particle drift = mu * F * dt = -0.2 * dt = -2e-6.
    const double expected = -k_a * 0.2 * dt;
    const double noise_sigma = std::sqrt(2.0 * 1.0 * dt) / std::sqrt(double(N_trials));
    EXPECT_NEAR(mean_dx, expected, 5.0 * noise_sigma);
}

// ---- k_a = 0 yields no spring effect ---------------------------------------

TEST(AnchoredIntegratorTest, KaZeroNoSpringEffect) {
    // Two systems, identical except one has anchor far away and k_a = 0.
    // With same RNG seed they must give identical particle trajectories.
    const double L   = 100.0;
    const double dt  = 1e-4;

    Integrator integ_no_spring(1.0, dt);
    integ_no_spring.setSpringStiffness(0.0);
    integ_no_spring.setAnchorFriction(0.0);

    System sysA(2), sysB(2);
    Box box(L);
    for (std::size_t i = 0; i < 2; ++i) {
        sysA.setPosition(i, 10.0 + 5.0 * i, 20.0);
        sysB.setPosition(i, 10.0 + 5.0 * i, 20.0);
        sysA.setAnchor  (i, 10.0 + 5.0 * i, 20.0);   // anchor at particle
        sysB.setAnchor  (i, 50.0,           50.0);   // anchor far away (k_a=0 so irrelevant)
    }

    RandomGenerator rngA(1234), rngB(1234);
    for (int s = 0; s < 50; ++s) {
        integ_no_spring.step(sysA, box, rngA);
        integ_no_spring.step(sysB, box, rngB);
    }

    for (std::size_t i = 0; i < 2; ++i) {
        EXPECT_DOUBLE_EQ(sysA.getX(i), sysB.getX(i)) << "i=" << i;
        EXPECT_DOUBLE_EQ(sysA.getY(i), sysB.getY(i)) << "i=" << i;
    }
}

// ---- Anchors drift freely across boundaries (positions are not auto-wrapped)

TEST(AnchoredIntegratorTest, AnchorsDriftFreelyAcrossBoundary) {
    // Stored anchor coordinates are unwrapped (see Box.hpp). A free anchor
    // (k_a=0) starting near the boundary diffuses across without being
    // remapped; the engine must remain numerically well-behaved.
    const double L  = 5.0;
    const std::size_t N = 32;
    System sys(N);
    Box box(L);
    for (std::size_t i = 0; i < N; ++i) {
        sys.setPosition(i, 2.5, 2.5);
        sys.setAnchor  (i, L - 0.01, L - 0.01);
    }

    Integrator integ(1.0, 1e-2);   // large dt to push anchors past the boundary
    integ.setSpringStiffness(0.0);
    integ.setAnchorFriction(1.0);  // free diffusing anchors
    RandomGenerator rng(77);

    for (int s = 0; s < 50; ++s) {
        integ.step(sys, box, rng);
        for (std::size_t i = 0; i < N; ++i) {
            EXPECT_TRUE(std::isfinite(sys.getAx(i))) << "step=" << s << " i=" << i;
            EXPECT_TRUE(std::isfinite(sys.getAy(i))) << "step=" << s << " i=" << i;
        }
    }
}
