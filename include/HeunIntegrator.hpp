#pragma once

#include <cstddef>
#include <vector>

class System;
class Box;
class ForceCalculator;
class CellList;
class RandomGenerator;

// =============================================================================
// HeunIntegrator (Stochastic Heun / predictor-corrector for overdamped Langevin)
// -----------------------------------------------------------------------------
// Two-stage explicit predictor-corrector for the same overdamped active
// Brownian system as Integrator. Both stages share the SAME noise sample, the
// condition under which Heun attains the higher order:
//
//   Predictor (Euler-Maruyama):
//     r_pred  = r_n + b(r_n, theta_n, a_n) * dt + sigma_r * sqrt(dt) * Z_r
//     theta_p = theta_n + sigma_theta * sqrt(dt) * Z_theta
//     a_pred  = a_n + b_a(r_n, a_n) * dt + sigma_a * sqrt(dt) * Z_a
//
//   Corrector (averaged drift; SAME Z):
//     r_{n+1} = r_n + 0.5 * (b(r_n, theta_n, a_n) + b(r_pred, theta_p, a_pred)) * dt
//                  + sigma_r * sqrt(dt) * Z_r
//     a_{n+1} = a_n + 0.5 * (b_a(r_n, a_n) + b_a(r_pred, a_pred))           * dt
//                  + sigma_a * sqrt(dt) * Z_a
//     theta_{n+1} = theta_p   (no drift on theta in this model)
//
// ORDER OF ACCURACY
// -----------------
// Our noise is additive (sigma_r, sigma_a, sigma_theta do not depend on the
// state). For SDEs with additive noise:
//    Euler-Maruyama  : weak 1, strong 1
//    Stochastic Heun : weak 2, strong 1
// At a fixed dt Heun is more accurate; in practice dt can be raised by 3-5x
// at matched mean-observable error, partially compensating for the 2x
// per-step force-evaluation cost. Net: usually a 1.5-2.5x speedup vs EM at
// the same accuracy budget.
//
// CELL-LIST INTERACTION
// ---------------------
// step() does two compute() calls. The caller supplies an (optional) cell
// list and is responsible for the rebuild trigger (same convention as the
// EM Integrator). Between the two stages we DO NOT rebuild — the predictor
// displacement is bounded by max_drift + |Z| * sqrt(2 D dt), comfortably
// under the standard r_skin/2 = 0.25 trigger for default skin = 0.5.
//
// DRIFT CAP
// ---------
// max_drift_ caps the predictor drift AND the corrector drift independently
// (each before being scaled by dt). Since |0.5*(a+b)| <= 0.5*|a| + 0.5*|b|,
// the averaged displacement remains bounded by max_drift in magnitude — the
// EM stability guarantee carries through.
// =============================================================================
class HeunIntegrator {
public:
    HeunIntegrator() = default;
    HeunIntegrator(double gamma, double dt);

    // Take one Heun step. Both force evaluations happen inside. If cl is
    // non-null and not in brute-force mode, both go through the cell list;
    // otherwise both fall back to brute force. The caller owns the rebuild
    // decision before this call (matches the EM pattern).
    void step(System& sys, const Box& box, const ForceCalculator& fc,
              const CellList* cl, RandomGenerator& rng);

    // ---- Same parameter API as Integrator -----------------------------------
    double getFriction()             const { return gamma_;     }
    double getTimestep()             const { return dt_;        }
    double getKBT()                  const { return kT_;        }
    double getActiveForce()          const { return f0_;        }
    double getRotationalDiffusion()  const { return D_r_;       }
    double getMaxDrift()             const { return max_drift_; }
    double getSpringStiffness()      const { return k_a_;       }
    double getAnchorFriction()       const { return gamma_a_;   }
    double getDiffusionCoefficient() const { return kT_ / gamma_; }
    double getMobility()             const { return 1.0 / gamma_; }

    void setFriction            (double g)   { gamma_    = g;   recomputeCachedConstants(); }
    void setTimestep            (double dt)  { dt_       = dt;  recomputeCachedConstants(); }
    void setKBT                 (double kT)  { kT_       = kT;  recomputeCachedConstants(); }
    void setActiveForce         (double f0)  { f0_       = f0;  }
    void setRotationalDiffusion (double Dr)  { D_r_      = Dr;  recomputeCachedConstants(); }
    void setMaxDrift            (double md)  { max_drift_= md;  }
    void setSpringStiffness     (double ka)  { k_a_      = ka;  }
    void setAnchorFriction      (double ga)  { gamma_a_  = ga;  recomputeCachedConstants(); }

private:
    double gamma_     = 1.0;
    double dt_        = 1.0e-4;
    double kT_        = 1.0;
    double f0_        = 0.0;
    double D_r_       = 0.0;
    double max_drift_ = 0.0;
    double k_a_       = 0.0;
    double gamma_a_   = 0.0;

    // Cached at setter time so the inner loops don't redo sqrt/divide.
    double mobility_dt_           = 0.0;
    double noise_amplitude_       = 0.0;
    double rot_noise_amplitude_   = 0.0;
    double anchor_mobility_dt_    = 0.0;
    double anchor_noise_amplitude_= 0.0;

    // Scratch held as members so step() does zero allocation in steady state.
    // Resized on first call (and whenever N or anchor_on changes).
    std::vector<double> r_orig_x_,    r_orig_y_;
    std::vector<double> a_orig_x_,    a_orig_y_;
    std::vector<double> drift1_dx_,   drift1_dy_;        // capped drift1 * dt
    std::vector<double> drift1_a_dx_, drift1_a_dy_;
    std::vector<double> noise_x_,     noise_y_;          // unit Gaussians
    std::vector<double> noise_a_x_,   noise_a_y_;

    void recomputeCachedConstants();
    void ensureScratch(std::size_t N, bool anchor_on);
};
