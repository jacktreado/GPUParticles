#pragma once

class System;
class Box;
class ForceCalculator;
class RandomGenerator;

// =============================================================================
// Integrator (Overdamped Langevin / Active Brownian Dynamics with Anchors)
// -----------------------------------------------------------------------------
// Implements the Euler-Maruyama scheme for the overdamped Langevin equation
// with an optional active self-propulsion force AND an optional per-particle
// harmonic anchor (spring tether):
//
//   particle:  dr/dt     = mu * (F_WCA + f0 * e_theta - k_a*(r-a)) + sqrt(2D)*eta_T
//   orientation: dtheta/dt = sqrt(2 * D_r) * eta_r(t)
//   anchor:    da/dt     = (1/gamma_a) * k_a*(r-a) + sqrt(2*kT/gamma_a) * eta_a
//
// Parametrization
// ---------------
//   The primary free parameter is the *friction* gamma.  The diffusion
//   coefficient is derived via the Einstein relation:
//
//       D = kT / gamma           (so noise amplitude is sqrt(2*kT/gamma*dt))
//       mu = 1 / gamma           (mobility — independent of kT)
//
//   This makes the athermal limit kT -> 0 well-defined: mobility stays finite
//   while the noise vanishes.  Users supply gamma directly; D is never an
//   input.  The deprecated getDiffusionCoefficient() is retained as a derived
//   getter for diagnostics.
//
// Symbols
// -------
//   gamma     = particle translational friction
//   mu        = 1 / gamma         (translational mobility)
//   e_theta   = (cos theta, sin theta)
//   f0        = self-propulsion force magnitude (0 = passive)
//   D_r       = 1 / tau_theta     (rotational diffusion, 0 = no tumbling)
//   k_a       = spring stiffness coupling particle to its anchor (0 = no spring)
//   gamma_a   = anchor friction coefficient (0 = anchor frozen)
//
// Spring forces use the periodic minimum-image convention so PBC is respected.
//
// Drift cap (numerical stability):
//   The combined drift (WCA + active + spring) is capped to max_drift before
//   noise is added.  This prevents Euler-Maruyama blow-up at high mobility
//   (small gamma) or stiff potentials.  Noise is never rescaled, so equilibrium
//   statistics remain correct when the cap is not active.  The same cap is
//   applied to anchor drift.
//
// GPU READINESS:
//   - step() reads/writes only index i in its main loop => trivially parallel.
//   - The RNG is the only sequential bit; replace with Philox/cuRAND per-thread
//     in the GPU port.
// =============================================================================
class Integrator {
public:
    Integrator() = default;
    Integrator(double gamma, double dt);

    // Take one BD/ABP step. Assumes WCA forces are already up-to-date.
    // Updates particle positions, velocities, orientations, and (when
    // gamma_a > 0) anchor positions on sys.
    void step(System& sys, const Box& box, RandomGenerator& rng) const;

    // Convenience: refresh forces, then step. Use this in the main loop.
    void stepWithForces(System& sys, const Box& box,
                        const ForceCalculator& fc,
                        RandomGenerator& rng) const;

    // Apply the non-WCA Euler-Maruyama terms at a caller-supplied dt:
    //   - active drift   :  r += mu * f0 * e_theta * dt
    //   - spring drift   :  r += mu * (-k_a * (r - a)) * dt
    //   - translational noise: r += sqrt(2 * kT/gamma * dt) * eta
    //   - rotational diffusion: theta += sqrt(2 * D_r * dt) * eta_r
    //   - anchor evolution: a += (1/gamma_a) * k_a * (r-a) * dt + sqrt(2*kT/gamma_a*dt)*eta_a
    //
    // The WCA drift is assumed to have been advanced already by an external
    // integrator (e.g. AdaptiveIntegrator). Wraps positions at the end.
    //
    // Coefficients are derived from the supplied dt at call time, so the
    // member dt_ is irrelevant here — the integrator can be driven from a
    // controller-chosen dt that varies between calls.
    void kickStep(System& sys, const Box& box, RandomGenerator& rng,
                  double dt) const;

    // ---- Getters ------------------------------------------------------------
    double getFriction()             const { return gamma_;     }
    double getTimestep()             const { return dt_;        }
    double getKBT()                  const { return kT_;        }
    double getActiveForce()          const { return f0_;        }
    double getRotationalDiffusion()  const { return D_r_;       }
    double getMaxDrift()             const { return max_drift_; }
    double getSpringStiffness()      const { return k_a_;       }
    double getAnchorFriction()       const { return gamma_a_;   }

    // Derived quantities.
    double getDiffusionCoefficient() const { return kT_ / gamma_; }   // D = kT/gamma
    double getMobility()             const { return 1.0 / gamma_; }   // mu = 1/gamma

    // Cached coefficients (also accessible for tests / diagnostics):
    double getMobilityTimesDt()      const { return mobility_dt_;          }
    double getNoiseAmplitude()       const { return noise_amplitude_;       }
    double getRotNoiseAmplitude()    const { return rot_noise_amplitude_;   }
    double getAnchorMobilityTimesDt()const { return anchor_mobility_dt_;    }
    double getAnchorNoiseAmplitude() const { return anchor_noise_amplitude_;}

    // ---- Setters (re-derive cached coefficients) ----------------------------
    void setFriction            (double g)   { gamma_    = g;   recomputeCachedConstants(); }
    void setTimestep            (double dt)  { dt_       = dt;  recomputeCachedConstants(); }
    void setKBT                 (double kT)  { kT_       = kT;  recomputeCachedConstants(); }
    void setActiveForce         (double f0)  { f0_       = f0;  }
    // Set D_r directly (pass 1.0/tau_theta from Config when tau_theta > 0).
    void setRotationalDiffusion (double Dr)  { D_r_      = Dr;  recomputeCachedConstants(); }
    // 0.0 = disabled (no cap).  Recommended default: 0.1 * sigma.
    void setMaxDrift            (double md)  { max_drift_= md;  }
    // Anchor coupling.
    void setSpringStiffness     (double ka)  { k_a_      = ka;  }
    void setAnchorFriction      (double ga)  { gamma_a_  = ga;  recomputeCachedConstants(); }

private:
    double gamma_     = 1.0;    // particle friction coefficient (Einstein: D = kT/gamma)
    double dt_        = 1.0e-4;
    double kT_        = 1.0;
    double f0_        = 0.0;    // active force magnitude (0 = passive)
    double D_r_       = 0.0;    // rotational diffusion coefficient (0 = no tumbling)
    double max_drift_ = 0.0;    // drift cap in length units (0 = disabled)
    double k_a_       = 0.0;    // anchor spring stiffness (0 = no coupling)
    double gamma_a_   = 0.0;    // anchor friction (0 = anchor frozen)

    // Cached values used in the inner loop.
    double mobility_dt_           = 0.0;   // (1 / gamma) * dt           [particle drift coef]
    double noise_amplitude_       = 0.0;   // sqrt(2 * kT/gamma * dt)    [particle noise]
    double rot_noise_amplitude_   = 0.0;   // sqrt(2 * D_r * dt)         [orientation noise]
    double anchor_mobility_dt_    = 0.0;   // (1 / gamma_a) * dt         [anchor drift coef]
    double anchor_noise_amplitude_= 0.0;   // sqrt(2 * kT/gamma_a*dt)    [anchor noise]

    void recomputeCachedConstants();
};
