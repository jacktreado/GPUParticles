#include "AdaptiveIntegrator.hpp"
#include "Box.hpp"
#include "ForceCalculator.hpp"
#include "RandomGenerator.hpp"
#include "System.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

// =============================================================================
// AdaptiveIntegrator unit tests
//
// Physics recap: the integrator solves  dr/dt = mu * F(r),  mu = 1 / gamma.
// (kT does not enter the deterministic drift; it only sets noise amplitude
// for the stochastic Euler-Maruyama integrator, which lives in Integrator.)
// All tests use sigma = epsilon = 1.0 (WCA defaults) unless stated otherwise.
// WCA cutoff: r_cut = 2^(1/6) ≈ 1.1225.  At r = sigma = 1.0 the pair force
// magnitude per unit separation is f_over_r2 = 24 * eps * (2 - 1) / r^2 = 24.
// =============================================================================

// ---- Constructor / getters --------------------------------------------------

TEST(AdaptiveIntegratorTest, DefaultTolerances) {
    AdaptiveIntegrator ai(1.0);
    EXPECT_DOUBLE_EQ(ai.getFriction(), 1.0);
    EXPECT_DOUBLE_EQ(ai.getMobility(), 1.0);
    EXPECT_DOUBLE_EQ(ai.getAbsTol(),   1.0e-6);
    EXPECT_DOUBLE_EQ(ai.getRelTol(),   1.0e-6);
    EXPECT_DOUBLE_EQ(ai.getDt(),       1.0e-3);   // default dt_init
}

TEST(AdaptiveIntegratorTest, ParametricConstructor) {
    AdaptiveIntegrator ai(0.25, 1.0e-8, 1.0e-9, 5.0e-4);
    EXPECT_DOUBLE_EQ(ai.getFriction(), 0.25);
    EXPECT_DOUBLE_EQ(ai.getMobility(), 4.0);
    EXPECT_DOUBLE_EQ(ai.getAbsTol(),   1.0e-8);
    EXPECT_DOUBLE_EQ(ai.getRelTol(),   1.0e-9);
    EXPECT_DOUBLE_EQ(ai.getDt(),       5.0e-4);
}

TEST(AdaptiveIntegratorTest, Setters) {
    AdaptiveIntegrator ai(1.0);
    ai.setFriction(0.5);
    ai.setAbsTol(1.0e-10);
    ai.setRelTol(1.0e-11);
    ai.setDt(2.5e-4);
    EXPECT_DOUBLE_EQ(ai.getFriction(), 0.5);
    EXPECT_DOUBLE_EQ(ai.getMobility(), 2.0);
    EXPECT_DOUBLE_EQ(ai.getAbsTol(),   1.0e-10);
    EXPECT_DOUBLE_EQ(ai.getRelTol(),   1.0e-11);
    EXPECT_DOUBLE_EQ(ai.getDt(),       2.5e-4);
}

// ---- Zero-force: a lone particle should not move ----------------------------

TEST(AdaptiveIntegratorTest, ZeroForceMeansNoMotion) {
    const double L = 50.0;
    System sys(1);
    sys.setPosition(0, L / 2.0, L / 2.0);

    Box             box(L);
    ForceCalculator fc(1.0, 1.0);
    AdaptiveIntegrator ai(1.0, 1.0e-10, 1.0e-10);

    ai.advance(sys, box, fc, 0.01);

    EXPECT_NEAR(sys.getX(0), L / 2.0, 1.0e-9);
    EXPECT_NEAR(sys.getY(0), L / 2.0, 1.0e-9);
}

TEST(AdaptiveIntegratorTest, ZeroForceReturnsAtLeastOneSubStep) {
    const double L = 50.0;
    System sys(1);
    sys.setPosition(0, L / 2.0, L / 2.0);
    Box box(L);
    ForceCalculator fc(1.0, 1.0);
    AdaptiveIntegrator ai(1.0);

    const std::size_t n = ai.advance(sys, box, fc, 0.01);
    EXPECT_GE(n, std::size_t{1});
}

// ---- WCA repulsion: correct drift direction --------------------------------

TEST(AdaptiveIntegratorTest, WCARepulsionDriftDirection) {
    const double L     = 100.0;
    const double d     = 1.0;
    const double T     = 1.0e-6;
    const double gamma = 1.0;
    const double mu    = 1.0 / gamma;

    System sys(2);
    sys.setPosition(0, L / 2.0, L / 2.0 + d / 2.0);
    sys.setPosition(1, L / 2.0, L / 2.0 - d / 2.0);

    Box             box(L);
    ForceCalculator fc(1.0, 1.0);
    AdaptiveIntegrator ai(gamma, 1.0e-10, 1.0e-10);

    const double x0_before = sys.getX(0);
    const double y0_before = sys.getY(0);
    const double y1_before = sys.getY(1);

    ai.advance(sys, box, fc, T);

    const double dy0 = sys.getY(0) - y0_before;
    const double dy1 = sys.getY(1) - y1_before;

    EXPECT_GT(dy0, 0.0) << "particle 0 should be pushed upward";
    EXPECT_LT(dy1, 0.0) << "particle 1 should be pushed downward";
    EXPECT_NEAR(sys.getX(0), x0_before, 1.0e-12);

    const double expected_dy0 = mu * 24.0 * T;
    EXPECT_NEAR(dy0, expected_dy0, 1.0e-2 * std::abs(expected_dy0));
}

// ---- Symmetry: equal-and-opposite displacements ----------------------------

TEST(AdaptiveIntegratorTest, WCARepulsionSymmetric) {
    const double L = 100.0;
    const double d = 1.0;
    const double T = 1.0e-5;

    System sys(2);
    sys.setPosition(0, L / 2.0, L / 2.0 + d / 2.0);
    sys.setPosition(1, L / 2.0, L / 2.0 - d / 2.0);

    const double y0_i = sys.getY(0);
    const double y1_i = sys.getY(1);

    Box             box(L);
    ForceCalculator fc(1.0, 1.0);
    AdaptiveIntegrator ai(1.0, 1.0e-10, 1.0e-10);
    ai.advance(sys, box, fc, T);

    const double dy0 = sys.getY(0) - y0_i;
    const double dy1 = sys.getY(1) - y1_i;

    EXPECT_NEAR(dy0, -dy1, 1.0e-10 * std::abs(dy0));
}

// ---- Tighter tolerance requires more (or equal) sub-steps ------------------

TEST(AdaptiveIntegratorTest, TighterToleranceTakesMoreOrEqualSubSteps) {
    const double L = 100.0;
    const double d = 0.95;
    const double T = 1.0e-3;

    auto make_sys = [&]() {
        System s(2);
        s.setPosition(0, L / 2.0, L / 2.0 + d / 2.0);
        s.setPosition(1, L / 2.0, L / 2.0 - d / 2.0);
        return s;
    };

    Box             box(L);
    ForceCalculator fc(1.0, 1.0);

    System sys_loose = make_sys();
    AdaptiveIntegrator ai_loose(1.0, 1.0e-4, 1.0e-4);
    const std::size_t n_loose = ai_loose.advance(sys_loose, box, fc, T);

    System sys_tight = make_sys();
    AdaptiveIntegrator ai_tight(1.0, 1.0e-8, 1.0e-8);
    const std::size_t n_tight = ai_tight.advance(sys_tight, box, fc, T);

    EXPECT_GE(n_tight, n_loose);
}

// ---- Accuracy: tight tolerance is more accurate than coarse Euler -----------

TEST(AdaptiveIntegratorTest, MoreAccurateThanEulerForSameDt) {
    const double L     = 100.0;
    const double d     = 1.0;
    const double T     = 5.0e-4;
    const double gamma = 1.0;
    const double mu    = 1.0 / gamma;

    Box             box(L);
    ForceCalculator fc(1.0, 1.0);

    System sys_ref(2);
    sys_ref.setPosition(0, L / 2.0, L / 2.0 + d / 2.0);
    sys_ref.setPosition(1, L / 2.0, L / 2.0 - d / 2.0);
    AdaptiveIntegrator ai_ref(gamma, 1.0e-11, 1.0e-11);
    ai_ref.advance(sys_ref, box, fc, T);
    const double y0_ref = sys_ref.getY(0);

    System sys_euler(2);
    sys_euler.setPosition(0, L / 2.0, L / 2.0 + d / 2.0);
    sys_euler.setPosition(1, L / 2.0, L / 2.0 - d / 2.0);
    fc.compute(sys_euler, box);
    sys_euler.setY(0, sys_euler.getY(0) + mu * sys_euler.getFy(0) * T);
    sys_euler.setY(1, sys_euler.getY(1) + mu * sys_euler.getFy(1) * T);

    System sys_ck(2);
    sys_ck.setPosition(0, L / 2.0, L / 2.0 + d / 2.0);
    sys_ck.setPosition(1, L / 2.0, L / 2.0 - d / 2.0);
    AdaptiveIntegrator ai_ck(gamma);
    ai_ck.advance(sys_ck, box, fc, T);

    const double err_euler = std::abs(sys_euler.getY(0) - y0_ref);
    const double err_ck    = std::abs(sys_ck.getY(0)    - y0_ref);

    EXPECT_LT(err_ck, err_euler);
}

// ---- PBC: positions stay in [0, L) after advance ---------------------------

TEST(AdaptiveIntegratorTest, PositionsWrappedAfterAdvance) {
    const double L = 5.0;
    const double d = 1.0;
    const double T = 0.1;

    System sys(2);
    sys.setPosition(0, L / 2.0, L - 0.01);
    sys.setPosition(1, L / 2.0, L - 0.01 - d);

    Box             box(L);
    ForceCalculator fc(1.0, 1.0);
    AdaptiveIntegrator ai(0.2);
    ai.advance(sys, box, fc, T);

    for (std::size_t i = 0; i < 2; ++i) {
        EXPECT_GE(sys.getX(i), 0.0);
        EXPECT_LT(sys.getX(i), L);
        EXPECT_GE(sys.getY(i), 0.0);
        EXPECT_LT(sys.getY(i), L);
    }
}

// ---- Two integrators starting from the same state agree -------------------

TEST(AdaptiveIntegratorTest, ConsecutiveCallsDeterministic) {
    const double L = 100.0;
    const double d = 1.0;
    const double T = 1.0e-4;

    auto make_sys = [&]() {
        System s(2);
        s.setPosition(0, L / 2.0, L / 2.0 + d / 2.0);
        s.setPosition(1, L / 2.0, L / 2.0 - d / 2.0);
        return s;
    };

    Box             box(L);
    ForceCalculator fc(1.0, 1.0);

    System sys_a = make_sys();
    System sys_b = make_sys();
    AdaptiveIntegrator ai_a(1.0, 1.0e-8, 1.0e-8);
    AdaptiveIntegrator ai_b(1.0, 1.0e-8, 1.0e-8);

    ai_a.advance(sys_a, box, fc, T);
    ai_b.advance(sys_b, box, fc, T);

    EXPECT_DOUBLE_EQ(sys_a.getY(0), sys_b.getY(0));
    EXPECT_DOUBLE_EQ(sys_a.getY(1), sys_b.getY(1));
}

// ---- Energy decreases when particles start overlapping ---------------------

TEST(AdaptiveIntegratorTest, EnergyDecreasesForOverlappingParticles) {
    const double L = 50.0;
    const double d = 0.5;
    const double T = 1.0e-4;

    System sys(2);
    sys.setPosition(0, L / 2.0, L / 2.0 + d / 2.0);
    sys.setPosition(1, L / 2.0, L / 2.0 - d / 2.0);

    Box             box(L);
    ForceCalculator fc(1.0, 1.0);

    const double U_before = fc.computeEnergy(sys, box);

    AdaptiveIntegrator ai(1.0, 1.0e-8, 1.0e-8);
    ai.advance(sys, box, fc, T);

    const double U_after = fc.computeEnergy(sys, box);

    EXPECT_LT(U_after, U_before);
}

// ---- Particles outside r_cut do not interact -------------------------------

TEST(AdaptiveIntegratorTest, NoInteractionBeyondCutoff) {
    const double L     = 100.0;
    const double sigma = 1.0;
    const double r_cut = std::pow(2.0, 1.0 / 6.0) * sigma;
    const double d     = r_cut + 0.5;
    const double T     = 0.01;

    System sys(2);
    sys.setPosition(0, L / 2.0, L / 2.0 + d / 2.0);
    sys.setPosition(1, L / 2.0, L / 2.0 - d / 2.0);

    const double y0_before = sys.getY(0);
    const double y1_before = sys.getY(1);

    Box             box(L);
    ForceCalculator fc(1.0, sigma);
    AdaptiveIntegrator ai(1.0, 1.0e-10, 1.0e-10);
    ai.advance(sys, box, fc, T);

    EXPECT_NEAR(sys.getY(0), y0_before, 1.0e-9);
    EXPECT_NEAR(sys.getY(1), y1_before, 1.0e-9);
}

// =============================================================================
// New API: single-adaptive-sub-step step() and dt management
// =============================================================================

// step() should report the dt it actually accepted (positive, finite, and
// for a benign zero-force system, not absurdly different from dt_init).
TEST(AdaptiveIntegratorTest, StepReturnsPositiveDt) {
    System sys(1);
    sys.setPosition(0, 5.0, 5.0);
    Box             box(10.0);
    ForceCalculator fc(1.0, 1.0);
    AdaptiveIntegrator ai(1.0, 1.0e-6, 1.0e-6, 1.0e-3);

    const double dt_taken = ai.step(sys, box, fc);
    EXPECT_GT(dt_taken, 0.0);
    EXPECT_TRUE(std::isfinite(dt_taken));
}

// step() must actually advance the system when there's a non-zero force.
TEST(AdaptiveIntegratorTest, StepAdvancesState) {
    System sys(2);
    sys.setPosition(0, 50.0, 50.5);
    sys.setPosition(1, 50.0, 49.5);   // separation = 1 = sigma
    Box             box(100.0);
    ForceCalculator fc(1.0, 1.0);
    AdaptiveIntegrator ai(1.0, 1.0e-8, 1.0e-8, 1.0e-5);

    const double y0_before = sys.getY(0);
    const double y1_before = sys.getY(1);

    ai.step(sys, box, fc);

    EXPECT_NE(sys.getY(0), y0_before);
    EXPECT_NE(sys.getY(1), y1_before);
}

// After step(), getDt() returns the controller's suggested next sub-step,
// which for a smooth zero-force system should generally grow above dt_init.
TEST(AdaptiveIntegratorTest, StepUpdatesDtSuggestion) {
    System sys(1);
    sys.setPosition(0, 5.0, 5.0);
    Box             box(10.0);
    ForceCalculator fc(1.0, 1.0);

    AdaptiveIntegrator ai(1.0, 1.0e-6, 1.0e-6, 1.0e-3);
    const double dt_initial = ai.getDt();
    ai.step(sys, box, fc);
    // For zero force the controller has every reason to grow dt; at minimum
    // it must not stay exactly at the initial value (it is being updated by
    // the controller).
    EXPECT_GE(ai.getDt(), dt_initial);
}

// setDt() seeds the next attempt; that exact value should be the dt actually
// used on the next step (for a smooth low-force problem the first try_step
// will accept it).
TEST(AdaptiveIntegratorTest, SetDtSeedsNextStep) {
    System sys(1);
    sys.setPosition(0, 5.0, 5.0);
    Box             box(10.0);
    ForceCalculator fc(1.0, 1.0);

    AdaptiveIntegrator ai(1.0, 1.0e-4, 1.0e-4);
    ai.setDt(7.5e-4);

    const double dt_taken = ai.step(sys, box, fc);
    EXPECT_DOUBLE_EQ(dt_taken, 7.5e-4);
}

// advance() should consist of exactly the same total time as repeatedly
// calling step() until the cumulative dt reaches the target.
TEST(AdaptiveIntegratorTest, AdvanceMatchesAccumulatedSteps) {
    const double L = 100.0;
    const double d = 1.0;
    const double T = 1.0e-3;

    auto make_sys = [&]() {
        System s(2);
        s.setPosition(0, L / 2.0, L / 2.0 + d / 2.0);
        s.setPosition(1, L / 2.0, L / 2.0 - d / 2.0);
        return s;
    };
    Box             box(L);
    ForceCalculator fc(1.0, 1.0);

    System sys_advance = make_sys();
    AdaptiveIntegrator ai_a(1.0, 1.0e-8, 1.0e-8);
    ai_a.advance(sys_advance, box, fc, T);

    System sys_step = make_sys();
    AdaptiveIntegrator ai_b(1.0, 1.0e-8, 1.0e-8);
    double t = 0.0;
    while (t < T * (1.0 - 1.0e-10)) {
        const double remaining = T - t;
        if (ai_b.getDt() > remaining) ai_b.setDt(remaining);
        t += ai_b.step(sys_step, box, fc);
    }

    EXPECT_NEAR(sys_advance.getY(0), sys_step.getY(0), 1.0e-12);
    EXPECT_NEAR(sys_advance.getY(1), sys_step.getY(1), 1.0e-12);
}

// Changing tolerances rebuilds the stepper; the old internal state doesn't
// carry over but the integrator must still produce a valid step.
TEST(AdaptiveIntegratorTest, SettingTolerancesAfterStepStillWorks) {
    System sys(2);
    sys.setPosition(0, 50.0, 50.5);
    sys.setPosition(1, 50.0, 49.5);
    Box             box(100.0);
    ForceCalculator fc(1.0, 1.0);

    AdaptiveIntegrator ai(1.0, 1.0e-4, 1.0e-4);
    ai.step(sys, box, fc);

    ai.setAbsTol(1.0e-9);
    ai.setRelTol(1.0e-9);

    const double dt_taken = ai.step(sys, box, fc);
    EXPECT_GT(dt_taken, 0.0);
    EXPECT_TRUE(std::isfinite(sys.getY(0)));
    EXPECT_TRUE(std::isfinite(sys.getY(1)));
}

// =============================================================================
// Extended deterministic flow: active force, spring, anchor evolution.
//
// The defaults (f0 = k_a = gamma_a = 0) reduce the integrator to its earlier
// WCA-only form (verified by all the tests above). These tests exercise the
// new degrees of freedom one at a time.
// =============================================================================

// Active force, no WCA, no spring: a single particle with theta = 0 should
// drift along +x at speed mu * f0. Use a far-away dummy second particle so
// the WCA force is zero (>> r_cut).
TEST(AdaptiveIntegratorTest, ActiveForceDriftsAlongOrientation) {
    const double gamma = 1.0;
    const double f0    = 0.5;
    const double T     = 1.0e-3;

    System sys(2);
    sys.setPosition(0, 50.0, 50.0);
    sys.setPosition(1, 90.0, 90.0);    // far away
    sys.setTheta(0, 0.0);              // points +x
    sys.setTheta(1, 0.0);

    Box             box(100.0);
    ForceCalculator fc(1.0, 1.0);
    AdaptiveIntegrator ai(gamma, 1.0e-10, 1.0e-10);
    ai.setActiveForce(f0);

    ai.advance(sys, box, fc, T);

    // dx = mu * f0 * T to first order; CK is 5th-order so the correction
    // (which only comes from changes in F_WCA, here zero) is negligible.
    EXPECT_NEAR(sys.getX(0), 50.0 + f0 * T / gamma, 1.0e-12);
    EXPECT_NEAR(sys.getY(0), 50.0,                   1.0e-12);
}

// Spring relaxation, no WCA, no active: a particle starting at distance d
// from its anchor (with anchor frozen, gamma_a = 0) decays exponentially.
TEST(AdaptiveIntegratorTest, SpringRelaxationFrozenAnchor) {
    const double gamma = 1.0;
    const double k_a   = 4.0;
    const double d0    = 0.3;
    const double T     = 0.05;

    System sys(2);
    sys.setPosition(0, 50.0 + d0, 50.0);
    sys.setPosition(1, 90.0, 90.0);    // far away
    sys.setAnchor  (0, 50.0,      50.0);

    Box             box(100.0);
    ForceCalculator fc(1.0, 1.0);
    AdaptiveIntegrator ai(gamma, 1.0e-10, 1.0e-10);
    ai.setSpringStiffness(k_a);
    ai.setAnchorFriction(0.0);   // anchor frozen

    ai.advance(sys, box, fc, T);

    // Solution: d(t) = d0 * exp(-(k_a/gamma) * t).
    const double d_expected = d0 * std::exp(-(k_a / gamma) * T);
    EXPECT_NEAR(sys.getX(0), 50.0 + d_expected, 1.0e-9);
    EXPECT_NEAR(sys.getY(0), 50.0,              1.0e-12);
    EXPECT_NEAR(sys.getAx(0), 50.0,             1.0e-12);   // anchor unchanged
}

// Spring co-evolution: with gamma_a > 0 and k_a > 0, particle and anchor
// approach a common drift; their separation decays as exp(-(k_a/gamma_eff) t)
// where 1/gamma_eff = 1/gamma + 1/gamma_a. Center of mass (mu-weighted)
// is conserved by the deterministic spring dynamics.
TEST(AdaptiveIntegratorTest, SpringCoEvolutionParticleAndAnchor) {
    const double gamma   = 1.0;
    const double gamma_a = 2.0;
    const double k_a     = 5.0;
    const double d0      = 0.4;
    const double T       = 0.03;

    System sys(2);
    sys.setPosition(0, 50.0 + d0, 50.0);
    sys.setPosition(1, 90.0, 90.0);
    sys.setAnchor  (0, 50.0,      50.0);

    Box             box(100.0);
    ForceCalculator fc(1.0, 1.0);
    AdaptiveIntegrator ai(gamma, 1.0e-10, 1.0e-10);
    ai.setSpringStiffness(k_a);
    ai.setAnchorFriction(gamma_a);

    ai.advance(sys, box, fc, T);

    // Reduced rate: separation r-a satisfies d/dt(r-a) = -k_a (1/gamma + 1/gamma_a)(r-a).
    const double rate = k_a * (1.0 / gamma + 1.0 / gamma_a);
    const double sep_expected = d0 * std::exp(-rate * T);
    const double sep = sys.getX(0) - sys.getAx(0);
    EXPECT_NEAR(sep, sep_expected, 1.0e-7);

    // Conserved quantity: d/dt(mu_a * r + mu * a) = 0 because dr/dt and da/dt
    // are equal and opposite up to the mobility ratio (Newton's third law on
    // the spring). Verify the (mu_a-weighted r + mu-weighted a) doesn't drift.
    const double mu  = 1.0 / gamma;
    const double mua = 1.0 / gamma_a;
    const double com_init = ((50.0 + d0) * mua + 50.0 * mu) / (mua + mu);
    const double com_now  = (sys.getX(0)  * mua + sys.getAx(0) * mu) / (mua + mu);
    EXPECT_NEAR(com_now, com_init, 1.0e-9);
}

// Active force + WCA: two particles oriented head-on. With small dt the
// active drift dominates (no WCA contact yet); with large dt WCA repulsion
// must keep them apart.
TEST(AdaptiveIntegratorTest, ActivePlusWCAStaysSeparated) {
    System sys(2);
    sys.setPosition(0, 50.0,           50.0);
    sys.setPosition(1, 50.0 + 1.05, 50.0);   // just outside r_cut
    sys.setTheta(0, 0.0);                       // points toward particle 1
    sys.setTheta(1, 3.14159265358979323846);   // points toward particle 0

    Box             box(100.0);
    ForceCalculator fc(1.0, 1.0);
    AdaptiveIntegrator ai(1.0, 1.0e-9, 1.0e-9);
    ai.setActiveForce(1.0);

    ai.advance(sys, box, fc, 0.5);   // long enough for them to meet

    // Final separation should be at most r_cut (~1.122) but well above zero —
    // WCA prevents collapse.
    double dx = sys.getX(1) - sys.getX(0);
    double dy = sys.getY(1) - sys.getY(0);
    box.minimumImage(dx, dy);
    const double r = std::sqrt(dx * dx + dy * dy);
    EXPECT_GT(r, 0.5);    // far from collapsing
    EXPECT_LT(r, 1.2);    // they did approach
}

// Active force getter / setter.
TEST(AdaptiveIntegratorTest, ActiveForceGetterSetter) {
    AdaptiveIntegrator ai(1.0);
    EXPECT_DOUBLE_EQ(ai.getActiveForce(), 0.0);
    ai.setActiveForce(2.5);
    EXPECT_DOUBLE_EQ(ai.getActiveForce(), 2.5);
}

TEST(AdaptiveIntegratorTest, SpringAnchorGettersSetters) {
    AdaptiveIntegrator ai(1.0);
    EXPECT_DOUBLE_EQ(ai.getSpringStiffness(), 0.0);
    EXPECT_DOUBLE_EQ(ai.getAnchorFriction(),  0.0);
    ai.setSpringStiffness(3.0);
    ai.setAnchorFriction(0.5);
    EXPECT_DOUBLE_EQ(ai.getSpringStiffness(), 3.0);
    EXPECT_DOUBLE_EQ(ai.getAnchorFriction(),  0.5);
}

// =============================================================================
// Strang(N-D-N) macro step tests
//
// step(sys, box, fc, rng) runs N(Δt/2) - D(Δt) - N(Δt/2) at a fixed user-set
// macro Δt. N is the pure-noise kick on positions/anchors/orientations; D is
// the deterministic flow integrated by the inner Cash-Karp 5(4) sub-stepper.
// =============================================================================

TEST(AdaptiveIntegratorTest, StochasticGettersSetters) {
    AdaptiveIntegrator ai(1.0);
    EXPECT_DOUBLE_EQ(ai.getKBT(),                 0.0);
    EXPECT_DOUBLE_EQ(ai.getRotationalDiffusion(), 0.0);
    ai.setKBT(0.7);
    ai.setRotationalDiffusion(2.5);
    ai.setMacroDt(5.0e-3);
    EXPECT_DOUBLE_EQ(ai.getKBT(),                 0.7);
    EXPECT_DOUBLE_EQ(ai.getRotationalDiffusion(), 2.5);
    EXPECT_DOUBLE_EQ(ai.getMacroDt(),             5.0e-3);
}

// Strang step at zero noise / zero force degenerates to N(Δt/2) = identity,
// D(Δt) = no drift, N(Δt/2) = identity. Particle should not move.
TEST(AdaptiveIntegratorTest, StrangAthermalNoForceParticleStaysPut) {
    System sys(1);
    sys.setPosition(0, 5.0, 5.0);
    Box             box(10.0);
    ForceCalculator fc(1.0, 1.0);
    RandomGenerator rng(0);

    AdaptiveIntegrator ai(1.0);
    ai.setMacroDt(1.0e-3);
    ai.setKBT(0.0);
    ai.setRotationalDiffusion(0.0);

    for (int i = 0; i < 5; ++i) ai.step(sys, box, fc, rng);

    EXPECT_NEAR(sys.getX(0), 5.0, 1.0e-9);
    EXPECT_NEAR(sys.getY(0), 5.0, 1.0e-9);
}

// Strang step returns its macro Δt and advances time monotonically.
TEST(AdaptiveIntegratorTest, StrangStepReturnsMacroDt) {
    System sys(1);
    sys.setPosition(0, 5.0, 5.0);
    Box             box(10.0);
    ForceCalculator fc(1.0, 1.0);
    RandomGenerator rng(0);

    AdaptiveIntegrator ai(1.0);
    const double dt = 1.0e-3;
    ai.setMacroDt(dt);
    ai.setKBT(0.5);

    const double dt_taken = ai.step(sys, box, fc, rng);
    EXPECT_DOUBLE_EQ(dt_taken, dt);
    EXPECT_TRUE(std::isfinite(sys.getX(0)));
    EXPECT_TRUE(std::isfinite(sys.getY(0)));
}

// Pure translational diffusion (no force, no rotation). MSD at time t should
// be 4*D*t with D = kT/gamma. Spread particles across a lattice that's far
// from contact so D-step is identity (WCA never engages).
TEST(AdaptiveIntegratorTest, StrangMSDScalesWithDiffusionCoefficient) {
    const double gamma = 1.0;
    const double kT    = 1.0;
    const double D     = kT / gamma;
    const std::size_t N = 256;

    const int    side    = static_cast<int>(std::ceil(std::sqrt(double(N))));
    const double spacing = 4.0;
    const double L       = spacing * side;

    System sys(N);
    for (std::size_t i = 0; i < N; ++i) {
        const double px = (static_cast<double>(i % static_cast<std::size_t>(side)) + 0.5) * spacing;
        const double py = (static_cast<double>(i / static_cast<std::size_t>(side)) + 0.5) * spacing;
        sys.setPosition(i, px, py);
    }
    std::vector<double> x0(N), y0(N);
    for (std::size_t i = 0; i < N; ++i) { x0[i] = sys.getX(i); y0[i] = sys.getY(i); }

    Box             box(L);
    ForceCalculator fc(1.0, 1.0);
    RandomGenerator rng(7);

    AdaptiveIntegrator ai(gamma);
    ai.setMacroDt(1.0e-3);
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
    const double slack = expected * 4.0 * std::sqrt(2.0 / static_cast<double>(N));
    EXPECT_NEAR(msd, expected, slack);
}

// Pure rotational diffusion (no translational noise): theta variance grows
// as 2*D_r*t about its initial value.
TEST(AdaptiveIntegratorTest, StrangRotationalDiffusionVariance) {
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

    AdaptiveIntegrator ai(1.0);
    ai.setMacroDt(1.0e-3);
    ai.setKBT(0.0);
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
    EXPECT_NEAR(var, expected, expected * 0.30);   // 30% slack
}

// Active force without noise: deterministic drift along the orientation.
TEST(AdaptiveIntegratorTest, StrangActiveForceDeterministicDrift) {
    System sys(1);
    sys.setPosition(0, 50.0, 50.0);
    sys.setTheta(0, 0.0);
    Box             box(100.0);
    ForceCalculator fc(1.0, 1.0);
    RandomGenerator rng(0);

    AdaptiveIntegrator ai(1.0);
    ai.setMacroDt(1.0e-3);
    ai.setActiveForce(0.5);

    double t = 0.0;
    const double T_end = 0.5;
    while (t < T_end) t += ai.step(sys, box, fc, rng);

    EXPECT_NEAR(sys.getX(0), 50.0 + 0.5 * t, 1.0e-9);
    EXPECT_NEAR(sys.getY(0), 50.0,            1.0e-9);
}

// Spring-only relaxation, anchor frozen: exponential decay at rate k_a/gamma.
TEST(AdaptiveIntegratorTest, StrangSpringRelaxationToAnchor) {
    const double gamma = 1.0;
    const double k_a   = 4.0;
    const double d0    = 0.3;

    System sys(1);
    sys.setPosition(0, 50.0 + d0, 50.0);
    sys.setAnchor(0, 50.0, 50.0);
    Box             box(100.0);
    ForceCalculator fc(1.0, 1.0);
    RandomGenerator rng(0);

    AdaptiveIntegrator ai(gamma);
    ai.setMacroDt(1.0e-3);
    ai.setSpringStiffness(k_a);
    ai.setAnchorFriction(0.0);

    double t = 0.0;
    const double T_end = 0.05;
    while (t < T_end) t += ai.step(sys, box, fc, rng);

    const double expected = d0 * std::exp(-(k_a / gamma) * t);
    EXPECT_NEAR(sys.getX(0) - 50.0, expected, expected * 0.05);
    EXPECT_NEAR(sys.getY(0),        50.0,    1.0e-9);
}

// One Strang macro step with kT > 0 and no force should give per-step
// position variance <r²>/Δt → 4*kT/gamma. This pins the noise amplitude
// (the 1/2 factors in N(Δt/2) sum to give the full Δt variance).
TEST(AdaptiveIntegratorTest, StrangSingleStepNoiseVarianceMatches4Dt) {
    const double gamma = 1.0;
    const double kT    = 1.0;
    const std::size_t M = 5000;
    const double L     = 1.0e6;
    const double dt    = 1.0e-3;

    Box             box(L);
    ForceCalculator fc(1.0, 1.0);
    RandomGenerator rng(13);

    double sum_r2 = 0.0;
    for (std::size_t k = 0; k < M; ++k) {
        System sys(1);
        sys.setPosition(0, L / 2.0, L / 2.0);
        AdaptiveIntegrator ai(gamma);
        ai.setMacroDt(dt);
        ai.setKBT(kT);

        ai.step(sys, box, fc, rng);
        const double dx = sys.getX(0) - L / 2.0;
        const double dy = sys.getY(0) - L / 2.0;
        sum_r2 += dx * dx + dy * dy;
    }
    const double mean = sum_r2 / static_cast<double>(M) / dt;
    const double expected = 4.0 * kT / gamma;
    EXPECT_NEAR(mean, expected, expected * 4.0 / std::sqrt(static_cast<double>(M)));
}

// Strang advance() loops step()s until total time ≥ T and returns macro count.
TEST(AdaptiveIntegratorTest, StrangAdvanceMatchesAccumulatedSteps) {
    System sys_a(1), sys_b(1);
    sys_a.setPosition(0, 5.0, 5.0);
    sys_b.setPosition(0, 5.0, 5.0);
    Box             box(10.0);
    ForceCalculator fc(1.0, 1.0);
    RandomGenerator rng_a(42), rng_b(42);

    AdaptiveIntegrator ai_a(1.0), ai_b(1.0);
    ai_a.setMacroDt(1.0e-3); ai_b.setMacroDt(1.0e-3);
    ai_a.setKBT(0.0);        ai_b.setKBT(0.0);    // deterministic so paths match exactly

    const double T = 5.0e-3;
    const std::size_t n_a = ai_a.advance(sys_a, box, fc, rng_a, T);
    EXPECT_GE(n_a, std::size_t{5});

    double t = 0.0;
    std::size_t n_b = 0;
    while (t < T * (1.0 - 1.0e-10)) {
        const double remaining = T - t;
        ai_b.setMacroDt(std::min(1.0e-3, remaining));
        t += ai_b.step(sys_b, box, fc, rng_b);
        ++n_b;
    }
    EXPECT_EQ(n_a, n_b);
    EXPECT_NEAR(sys_a.getX(0), sys_b.getX(0), 1.0e-12);
    EXPECT_NEAR(sys_a.getY(0), sys_b.getY(0), 1.0e-12);
}

// Cell list bookkeeping: with enough particles in a box that admits a 3x3
// stencil, the integrator should rebuild the owned cell list at least once.
TEST(AdaptiveIntegratorTest, StrangCellListRebuildsAtLeastOnce) {
    const std::size_t N = 64;
    const double      L = 16.0;
    System sys(N);
    const int    side    = 8;
    const double spacing = L / side;
    for (std::size_t i = 0; i < N; ++i) {
        const double px = (static_cast<double>(i % static_cast<std::size_t>(side)) + 0.5) * spacing;
        const double py = (static_cast<double>(i / static_cast<std::size_t>(side)) + 0.5) * spacing;
        sys.setPosition(i, px, py);
    }
    Box             box(L);
    ForceCalculator fc(1.0, 1.0);
    RandomGenerator rng(3);

    AdaptiveIntegrator ai(1.0);
    ai.setMacroDt(1.0e-3);
    ai.setKBT(0.1);

    EXPECT_FALSE(ai.cellListIsBruteForce());     // not initialized yet
    EXPECT_EQ(ai.getCellListRebuilds(), std::size_t{0});

    for (int i = 0; i < 10; ++i) ai.step(sys, box, fc, rng);

    // Either we used a real cell list (rebuild count > 0) or geometry forced
    // brute force; the box here is 16x16 with r_verlet ~ 1.6, so the cell
    // grid is 10x10 — well above the 3x3 minimum.
    EXPECT_FALSE(ai.cellListIsBruteForce());
    EXPECT_GE(ai.getCellListRebuilds(), std::size_t{1});
}

// Cell-list brute-force fallback when the box is too small for a 3x3 stencil.
TEST(AdaptiveIntegratorTest, StrangCellListFallsBackToBruteForceForSmallBox) {
    System sys(2);
    sys.setPosition(0, 1.0, 1.0);
    sys.setPosition(1, 1.5, 1.5);
    Box             box(2.5);                    // too small for a 3x3 stencil
    ForceCalculator fc(1.0, 1.0);
    RandomGenerator rng(0);

    AdaptiveIntegrator ai(1.0);
    ai.setMacroDt(1.0e-3);
    ai.setKBT(0.0);

    ai.step(sys, box, fc, rng);

    EXPECT_TRUE(ai.cellListIsBruteForce());
    EXPECT_EQ(ai.getCellListRebuilds(), std::size_t{0});
}
