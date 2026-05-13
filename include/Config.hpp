#pragma once

#include "ForceCalculator.hpp"

#include <cstdint>
#include <string>

// =============================================================================
// Config
// -----------------------------------------------------------------------------
// Parses a JSON input file and holds every parameter the simulation needs.
//
// Working units (frozen at 1):
//
//   sigma     = 1   (length unit: particle diameter)
//   epsilon   = 1   (energy unit: pair-interaction scale)
//   gamma     = 1   (particle friction; sets the time unit
//                    tau = gamma * sigma^2 / epsilon = 1)
//
// All physical microscopic parameters are then determined by SIX dimensionless
// inputs plus N and the potential type:
//
//   N         particle count                              (required)
//   phi       2D packing fraction in (0, 1)               (required)
//   delta     typical pair overlap r*/sigma in steady contact
//                  (sets f0 by force balance F_pot(r*) = f0)
//   Pe        active Peclet f0 * tau_theta / (gamma * sigma)
//                  (sets persistence; tau_theta = Pe * gamma * sigma / f0)
//   De        active Deborah (gamma_a / k_a) / tau_theta
//                  (sets anchor memory time relative to persistence;
//                   k_a = gamma_a / (De * tau_theta))
//   R         friction ratio gamma_a / gamma
//                  (sets anchor friction; gamma_a = R * gamma)
//   C         dimensionless temperature kT / epsilon
//                  (kT = C * epsilon; small C = athermal limit)
//   potential pair potential, "WCA" or "soft_sphere"
//
// Geometry: phi = N * pi * sigma^2 / (4 * L^2) gives the box side L.
//
// Edge cases handled by recompute():
//   - delta at/beyond the potential's cutoff -> f0 = 0 (passive limit);
//     tau_theta and k_a are set to 0 because they are degenerate without
//     active drive.
//   - Pe <= 0 -> tau_theta = 0 (no rotational diffusion). k_a then = 0.
//   - De <= 0 or tau_theta = 0 -> k_a = 0 (no anchor coupling).
//
// Time integration. Three methods, selected by the "integrator" JSON field:
//   - "euler_maruyama" (default): fixed-Δt Euler-Maruyama. Cheapest per
//     step (1 force eval). Weak/strong order 1 in dt for additive noise.
//   - "heun": fixed-Δt stochastic Heun (predictor-corrector). 2 force evals
//     per step but weak order 2 for additive noise — typically lets you
//     raise dt by 3-5x at matched accuracy. The accuracy/cost sweet spot
//     for overdamped active Brownian dynamics.
//   - "adaptive_strang": Strang(N-D-N) at fixed macro Δt = dt_init, with an
//     inner Cash-Karp 5(4) sub-stepper controlled by (abs_tol, rel_tol).
// All three share dt_init, max_drift, r_skin (the cell-list skin). Output
// cadence is time-based (output_dt) for every method.
// =============================================================================
enum class IntegratorMethod {
    EulerMaruyama,
    Heun,
    AdaptiveStrang,
};

class Config {
public:
    // ---- Required fields ----------------------------------------------------
    std::size_t N   = 0;
    double      phi = 0.5;

    // ---- Dimensionless control parameters (with defaults) -------------------
    double        Pe        = 1.0;
    double        De        = 1.0;
    double        R         = 1.0;
    double        C         = 1.0e-3;
    double        delta     = 1.0;
    PotentialType potential = PotentialType::WCA;

    // ---- Frozen working units -----------------------------------------------
    // These are physical fields the rest of the engine consumes; in the new
    // parameterization they are constants, but we keep them as members so
    // every existing reader (ForceCalculator, Integrator, main.cpp, tests)
    // works without change.
    double sigma   = 1.0;
    // double epsilon = 1.0;
    double tau_elastic = 1.0;
    double gamma_hat = 1.0;
    
    // ---- Derived microscopic parameters (set by recompute()) ----------------
    double epsilon   = 1.0;     // from delta + potential via f0ForOverlap
    double kT        = 0.0;     // = C * epsilon
    double f0        = 0.0;     // from delta + potential via f0ForOverlap
    double tau_theta = 0.0;     // = Pe * gamma_hat * sigma / f0
    double gamma_a   = 0.5;     // = gamma_hat - gamma
    double gamma     = 0.5;     // = gamma_hat - gamma_a
    double k_a       = 0.0;     // = gamma_a / (De * tau_theta)
    double L         = 0.0;     // from N, phi, sigma

    // ---- Integrator selection + per-method knobs ----------------------------
    IntegratorMethod integrator = IntegratorMethod::EulerMaruyama;

    // EM-only knobs. Ignored for adaptive_strang.
    double      max_drift = 0.1;          // drift cap in length units (0 = disabled)
    double      r_skin    = 0.5;          // Verlet-list skin (cell-list rebuild trigger)

    // Time-based loop: simulate from t = 0 to t_end, writing a frame every
    // output_dt of accumulated time. dt_init is the (fixed) Δt for both
    // methods; for adaptive_strang it's the macro Δt and the inner controller
    // chooses sub-steps from (abs_tol, rel_tol). For EM the tolerances are
    // unused.
    double      t_end     = 1.0;          // total simulation time
    double      output_dt = 0.01;         // trajectory write interval
    double      abs_tol   = 1.0e-3;       // (adaptive only) inner-step abs tol
    double      rel_tol   = 1.0e-2;       // (adaptive only) inner-step rel tol
    double      dt_init   = 1.0e-3;       // (fixed) Δt for both methods

    // ---- I/O ----------------------------------------------------------------
    std::string output_file  = "trajectory.h5";
    std::string init_mode    = "lattice";   // "lattice" or "random"
    std::uint64_t seed       = 12345;

    // ---- Methods ------------------------------------------------------------
    static Config fromFile(const std::string& path);

    // Recompute every derived quantity (kT, f0, tau_theta, gamma_a, k_a, L)
    // from the current inputs. Call after mutating any input field.
    void recompute();

    void print() const;

    // Einstein relation: D = kT / gamma. Useful for diagnostics.
    double diffusionCoefficient() const { return kT / gamma; }

    // Convert the potential enum to its JSON-input string ("WCA" / "soft_sphere").
    std::string potentialName() const;

    // Convert the integrator enum to its JSON-input string.
    std::string integratorName() const;
};
