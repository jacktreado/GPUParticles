#pragma once

#include <cstddef>
#include <memory>

class System;
class Box;
class ForceCalculator;
class RandomGenerator;

// =============================================================================
// AdaptiveMacroIntegrator
// -----------------------------------------------------------------------------
// Adaptive-macro-step Strang operator splitting for the full overdamped
// active-Brownian + anchor SDE:
//
//   dr_i = mu * [F_WCA(r) + f0 * e_theta_i - k_a * (r_i - a_i)] dt
//          + sqrt(2 kT / gamma) dW_i^t
//   da_i = (1/gamma_a) k_a (r_i - a_i) dt + sqrt(2 kT / gamma_a) dW_i^a
//   dtheta_i = sqrt(2 D_r) dW_i^r
//
// The Strang composition over a macro step Δt is
//
//      Φ(Δt) ≈ N(Δt/2) ∘ D(Δt) ∘ N(Δt/2)
//
// where N is the pure-noise operator (additive Brownian increments on r, a,
// and theta — they commute since they act on independent state components),
// and D is the deterministic flow integrated by AdaptiveIntegrator (Cash-Karp
// 5(4) with internal PI control).
//
// THE MACRO Δt IS AN OUTPUT, NOT AN INPUT.
//
// On every call to step(), the integrator picks Δt by **step doubling**:
// it runs one Strang step at trial Δt (the "coarse" path) and two Strang
// steps at Δt/2 each (the "fine" path), then compares them in a weighted-RMS
// norm. To avoid the noise term dominating the comparison (which would force
// Δt -> 0), the fine noises are sampled from the conditional **Brownian
// bridge** of the coarse noises:
//
//      η_a = (η_c + ~η) / sqrt(2),    η_b = (η_c - ~η) / sqrt(2)
//
// (no OU bridge is needed because our overdamped model has no momentum and
// hence no friction-decayed velocity propagator — only simple additive
// Brownian increments on positions, anchors, and angles).
//
// A standard PI controller (Gustafsson) scales Δt by err^(-1/3) * err_prev^(1/9)
// — the order-2 splitting exponents — clamped to a growth/shrink window.
// Trial steps are accepted when err <= 1 (in units of atol + rtol*|state|);
// rejected steps are retried with a smaller Δt without advancing time.
//
// The user-facing knobs are abs_tol, rel_tol, the initial Δt guess, and the
// physical parameters (gamma, kT, f0, k_a, gamma_a, D_r). The first call
// from a given seed will spend a few extra rejections discovering the right
// scale; thereafter the controller tracks the problem's stiffness automatically.
//
// References:
//   Foster, dos Reis, Strange (2023) — "no-skip" property and Lévy-area bias.
//   Hairer, Nørsett, Wanner (1993)   — PI step-size controllers.
//   Leimkuhler, Matthews (2013)      — BAOAB and configurational sampling.
// =============================================================================
class AdaptiveMacroIntegrator {
public:
    AdaptiveMacroIntegrator(double gamma,
                            double abs_tol = 1.0e-3,
                            double rel_tol = 1.0e-2,
                            double dt_init = 1.0e-3);
    ~AdaptiveMacroIntegrator();
    AdaptiveMacroIntegrator(const AdaptiveMacroIntegrator&)            = delete;
    AdaptiveMacroIntegrator& operator=(const AdaptiveMacroIntegrator&) = delete;
    AdaptiveMacroIntegrator(AdaptiveMacroIntegrator&&) noexcept;
    AdaptiveMacroIntegrator& operator=(AdaptiveMacroIntegrator&&) noexcept;

    // Take ONE adaptive macro step. Loops trial-and-retry internally until
    // the controller accepts an error. Returns the Δt that was accepted.
    // After the call, getDt() returns the controller's suggested next Δt.
    double step(System& sys, const Box& box,
                const ForceCalculator& fc, RandomGenerator& rng);

    // ---- Stochastic params -------------------------------------------------
    double getKBT()                  const;
    void   setKBT(double kT);
    double getRotationalDiffusion()  const;     // D_r
    void   setRotationalDiffusion(double D_r);

    // ---- Frictions (gamma drives both D and T noise; gamma_a drives both
    // the anchor D-flow and A-noise) ----------------------------------------
    double getFriction()        const;
    void   setFriction(double g);
    double getAnchorFriction()  const;
    void   setAnchorFriction(double g_a);

    // ---- Deterministic-flow extras (forwarded to inner D-stepper) ----------
    double getActiveForce()     const;
    void   setActiveForce(double f0);
    double getSpringStiffness() const;
    void   setSpringStiffness(double k_a);

    // ---- Adaptive-macro-step controller ------------------------------------
    double getAbsTol()         const;
    void   setAbsTol(double a);
    double getRelTol()         const;
    void   setRelTol(double r);
    double getDt()             const;     // current macro suggestion
    void   setDt(double d);
    double getLastErr()        const;     // most-recent accepted err norm
    std::size_t getRejections() const;    // total rejections (lifetime)

    // ---- Cell list / Verlet neighbor map -----------------------------------
    // The macro integrator owns a CellList that the inner deterministic
    // (Cash-Karp) sub-stepper consults via fc.compute(sys, box, *cl). The
    // list is reset up automatically when the particle count, box dimensions,
    // or the ForceCalculator's cutoff change, and is rebuilt at the START of
    // every step() call when the standard r_skin/2 drift criterion fires —
    // i.e. *before* the coarse + fine Strang trials, so both trials share an
    // identical, FIXED neighbor connectivity. r_skin is the only knob; it
    // trades off rebuild frequency vs. neighbor-list size and must be large
    // enough that no particle drifts more than r_skin/2 within a single
    // accepted macro step (otherwise the list is stale when the controller
    // reads it). When the box is too small for a 3x3 stencil the cell list
    // silently falls back to brute force.
    double getCellListSkin()         const;
    void   setCellListSkin(double r_skin);
    std::size_t getCellListRebuilds() const;
    bool        cellListIsBruteForce() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
