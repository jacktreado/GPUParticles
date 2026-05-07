#pragma once

#include <cstddef>
#include <memory>

class System;
class Box;
class ForceCalculator;
class CellList;
class RandomGenerator;

// =============================================================================
// AdaptiveIntegrator — Cash-Karp 5(4) adaptive deterministic stepper, plus
//                      a normal Strang(N-D-N) stochastic macro step at a
//                      user-fixed Δt.
// -----------------------------------------------------------------------------
// Two integration paths share one underlying object:
//
//   (1) Deterministic only (drift-only flow with theta frozen):
//
//       dr_i/dt = mu * [F_WCA(r) + f0 * e_theta_i - k_a * (r_i - a_i)]
//       da_i/dt = mu_a * k_a * (r_i - a_i)
//
//       Driven by a controlled Cash-Karp 5(4) stepper with PI step-size
//       control. Use step()/advance() (no RandomGenerator argument) — the
//       inner controller picks every sub-step itself.
//
//   (2) Strang(N-D-N) stochastic macro step at a user-fixed Δt:
//
//       Φ(Δt) ≈ N(Δt/2) ∘ D(Δt) ∘ N(Δt/2)
//
//       N is the pure-noise operator (additive Brownian increments on r, a,
//       and theta), D is the deterministic flow above integrated by the
//       inner Cash-Karp 5(4). The macro Δt is FIXED — set via setMacroDt() —
//       and the integrator does not adapt it. This is the "normal" Strang
//       scheme; the older AdaptiveMacroIntegrator additionally step-doubles
//       and Brownian-bridges to adapt the macro Δt, which is more accurate
//       per step but ~3x slower.
//
//       Use step(...rng) / advance(...rng, T) — both refresh the owned cell
//       list at the start of the macro step (so the inner D-step sees a
//       fixed neighbor connectivity), then run the Strang composition.
//
// API levels for path (1) — DETERMINISTIC
// ---------------------------------------
//   step(sys, box, fc)
//       One adaptive sub-step. Wraps positions. Returns dt taken.
//   advance(sys, box, fc, T)
//       Adaptive sub-steps until total time ≥ T. Returns sub-step count.
//
// API levels for path (2) — STRANG STOCHASTIC
// -------------------------------------------
//   step(sys, box, fc, rng)
//       One Strang macro step at the current macro Δt. Refreshes the
//       owned cell list at the start. Returns the macro Δt taken.
//   advance(sys, box, fc, rng, T)
//       Strang macro steps until total time ≥ T. Caps the last macro Δt
//       so total time lands on T (does NOT permanently shrink macro Δt).
//       Returns macro-step count.
//
// STOCHASTIC PARAMETERS:
//   kT, D_r, gamma_a together determine the noise amplitudes:
//     position noise sigma per step = sqrt(2 * kT / gamma * dt)
//     anchor noise sigma per step   = sqrt(2 * kT / gamma_a * dt)  (gamma_a > 0)
//     theta noise sigma per step    = sqrt(2 * D_r * dt)
//   kT = 0 turns off translational + anchor noise. D_r = 0 freezes theta.
//   gamma_a = 0 freezes anchors entirely (no drift, no noise).
//
// CELL LIST:
//   The integrator owns a CellList that the inner deterministic stepper
//   consults via fc.compute(sys, box, *cl). It is reset up automatically
//   on geometry changes (N, Lx, Ly, r_cut) and rebuilt at the START of
//   each Strang macro step when the standard r_skin/2 drift criterion
//   fires. r_skin is the only knob; default 0.5. When the box is too
//   small for a 3x3 stencil the cell list silently falls back to brute
//   force.
//
//   Path (1) does NOT manage the cell list itself; callers (e.g. the
//   AdaptiveMacroIntegrator) install one via setCellList(). When path (2)
//   is in use, setCellList() should not be called externally.
//
// GPU PORTABILITY:
//   The expensive part — force evaluation via ForceCalculator — is already
//   GPU-ready (independent-particle outer loop, SoA memory). The boost
//   stepper infrastructure is CPU-only and is hidden behind pImpl, so a CUDA
//   port replaces only AdaptiveIntegrator.cpp's Impl.
// =============================================================================
class AdaptiveIntegrator {
public:
    // abs_tol / rel_tol set the Cash-Karp 5(4) per-component error tolerance:
    //     |err[i]| <= abs_tol + rel_tol * |state[i]|
    //
    // dt_init is the first inner sub-step the controller tries (and the
    // initial macro Δt for the Strang path; users typically override it
    // via setMacroDt()).
    explicit AdaptiveIntegrator(double gamma,
                                double abs_tol = 1.0e-6,
                                double rel_tol = 1.0e-6,
                                double dt_init = 1.0e-3,
                                // Deterministic flow extras:
                                double f0      = 0.0,
                                double k_a     = 0.0,
                                double gamma_a = 0.0);
    ~AdaptiveIntegrator();

    AdaptiveIntegrator(const AdaptiveIntegrator&)            = delete;
    AdaptiveIntegrator& operator=(const AdaptiveIntegrator&) = delete;
    AdaptiveIntegrator(AdaptiveIntegrator&&) noexcept;
    AdaptiveIntegrator& operator=(AdaptiveIntegrator&&) noexcept;

    // ---- Path (1): deterministic adaptive sub-stepping ---------------------
    double      step   (System& sys, const Box& box, const ForceCalculator& fc);
    std::size_t advance(System& sys, const Box& box,
                        const ForceCalculator& fc, double T);

    // ---- Path (2): Strang(N-D-N) macro step at user-fixed Δt ---------------
    double      step   (System& sys, const Box& box, const ForceCalculator& fc,
                        RandomGenerator& rng);
    std::size_t advance(System& sys, const Box& box, const ForceCalculator& fc,
                        RandomGenerator& rng, double T);

    // ---- Deterministic flow getters ----------------------------------------
    double getFriction()        const;
    double getMobility()        const;     // mu = 1/gamma
    double getAbsTol()          const;
    double getRelTol()          const;
    double getDt()              const;     // current inner adaptive sub-step
    double getActiveForce()     const;
    double getSpringStiffness() const;
    double getAnchorFriction()  const;

    // ---- Deterministic flow setters ----------------------------------------
    void setFriction(double g);
    void setAbsTol(double a);               // rebuilds inner stepper
    void setRelTol(double r);               // rebuilds inner stepper
    void setDt(double d);                   // override inner adaptive sub-step
    void setActiveForce(double f0);
    void setSpringStiffness(double k_a);
    void setAnchorFriction(double gamma_a);

    // ---- Stochastic params (Strang path only) ------------------------------
    double getKBT()                 const;
    void   setKBT(double kT);
    double getRotationalDiffusion() const;  // D_r
    void   setRotationalDiffusion(double D_r);

    // ---- Macro Δt (Strang path) --------------------------------------------
    double getMacroDt() const;
    void   setMacroDt(double dt);

    // ---- Cell list -----------------------------------------------------------
    // External (non-owning) override — used by AdaptiveMacroIntegrator. Pass
    // nullptr to revert to the owned cell list (if any) / brute force.
    void               setCellList(const CellList* cl);
    const CellList*    getCellList() const;

    // Owned cell-list controls used by the Strang path.
    double      getCellListSkin()      const;
    void        setCellListSkin(double r_skin);
    std::size_t getCellListRebuilds() const;
    bool        cellListIsBruteForce() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
