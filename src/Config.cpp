#include "Config.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {

inline double pi() { return std::acos(-1.0); }

PotentialType potentialFromString(const std::string& name) {
    if (name == "WCA"        || name == "wca")         return PotentialType::WCA;
    if (name == "soft_sphere"|| name == "softsphere"||
        name == "SoftSphere" || name == "soft-sphere") return PotentialType::SoftSphere;
    throw std::runtime_error(
        "Config: unknown potential \"" + name +
        "\" (allowed: \"WCA\", \"soft_sphere\")");
}

IntegratorMethod integratorFromString(const std::string& name) {
    if (name == "euler_maruyama" || name == "EulerMaruyama" ||
        name == "EM"             || name == "em"            ||
        name == "euler-maruyama" || name == "eulermaruyama")
        return IntegratorMethod::EulerMaruyama;
    if (name == "heun"             || name == "Heun"             ||
        name == "stochastic_heun"  || name == "stochastic-heun"  ||
        name == "predictor_corrector"|| name == "predictor-corrector")
        return IntegratorMethod::Heun;
    if (name == "adaptive_strang" || name == "AdaptiveStrang" ||
        name == "adaptive"        || name == "strang"         ||
        name == "adaptive-strang")
        return IntegratorMethod::AdaptiveStrang;
    throw std::runtime_error(
        "Config: unknown integrator \"" + name +
        "\" (allowed: \"euler_maruyama\", \"heun\", \"adaptive_strang\")");
}

} // namespace

using json = nlohmann::json;

std::string Config::potentialName() const {
    switch (potential) {
        case PotentialType::WCA:        return "WCA";
        case PotentialType::SoftSphere: return "soft_sphere";
    }
    return "WCA";   // unreachable; silences -Wreturn-type
}

std::string Config::integratorName() const {
    switch (integrator) {
        case IntegratorMethod::EulerMaruyama:  return "euler_maruyama";
        case IntegratorMethod::Heun:           return "heun";
        case IntegratorMethod::AdaptiveStrang: return "adaptive_strang";
    }
    return "euler_maruyama";   // unreachable; silences -Wreturn-type
}

Config Config::fromFile(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error(
            "Config::fromFile: could not open input file '" + path + "'");
    }

    json j;
    try {
        in >> j;
    } catch (const json::parse_error& e) {
        throw std::runtime_error(
            std::string("Config::fromFile: JSON parse error — ") + e.what());
    }

    Config cfg;

    // ---- Required fields ----------------------------------------------------
    if (!j.contains("N") || !j.contains("phi")) {
        throw std::runtime_error(
            "Config::fromFile: missing required field(s); "
            "the input must contain N and phi.");
    }
    cfg.N   = j.at("N").get<std::size_t>();
    cfg.phi = j.at("phi").get<double>();

    if (cfg.N == 0)
        throw std::runtime_error("Config::fromFile: N must be > 0");
    if (cfg.phi <= 0.0 || cfg.phi >= 1.0)
        throw std::runtime_error("Config::fromFile: phi must lie in (0, 1)");

    // ---- Dimensionless control parameters (with defaults) -------------------
    if (j.contains("Pe"))    cfg.Pe    = j["Pe"].get<double>();
    if (j.contains("De"))    cfg.De    = j["De"].get<double>();
    if (j.contains("R"))     cfg.R     = j["R"].get<double>();
    if (j.contains("C"))     cfg.C     = j["C"].get<double>();
    if (j.contains("delta")) cfg.delta = j["delta"].get<double>();
    if (j.contains("potential"))
        cfg.potential = potentialFromString(j["potential"].get<std::string>());

    if (cfg.Pe    < 0.0) throw std::runtime_error("Config: Pe must be >= 0");
    if (cfg.De    < 0.0) throw std::runtime_error("Config: De must be >= 0");
    if (cfg.R     < 0.0) throw std::runtime_error("Config: R must be >= 0");
    if (cfg.C     < 0.0) throw std::runtime_error("Config: C must be >= 0");
    if (cfg.delta <= 0.0) throw std::runtime_error("Config: delta must be > 0");

    // ---- Integrator + I/O ---------------------------------------------------
    if (j.contains("integrator"))
        cfg.integrator = integratorFromString(j["integrator"].get<std::string>());
    if (j.contains("max_drift"))   cfg.max_drift   = j["max_drift"].get<double>();
    if (j.contains("r_skin"))      cfg.r_skin      = j["r_skin"].get<double>();
    if (j.contains("t_end"))       cfg.t_end       = j["t_end"].get<double>();
    if (j.contains("output_dt"))   cfg.output_dt   = j["output_dt"].get<double>();
    if (j.contains("abs_tol"))     cfg.abs_tol     = j["abs_tol"].get<double>();
    if (j.contains("rel_tol"))     cfg.rel_tol     = j["rel_tol"].get<double>();
    if (j.contains("dt_init"))     cfg.dt_init     = j["dt_init"].get<double>();
    if (j.contains("output_file")) cfg.output_file = j["output_file"].get<std::string>();
    if (j.contains("init_mode"))   cfg.init_mode   = j["init_mode"].get<std::string>();
    if (j.contains("seed"))        cfg.seed        = j["seed"].get<std::uint64_t>();

    if (cfg.max_drift < 0.0) throw std::runtime_error("Config: max_drift must be >= 0 (0 disables the cap)");
    if (cfg.r_skin    < 0.0) throw std::runtime_error("Config: r_skin must be >= 0");
    if (cfg.t_end     <= 0.0) throw std::runtime_error("Config: t_end must be > 0");
    if (cfg.output_dt <= 0.0) throw std::runtime_error("Config: output_dt must be > 0");
    if (cfg.abs_tol   <= 0.0) throw std::runtime_error("Config: abs_tol must be > 0");
    if (cfg.rel_tol   <= 0.0) throw std::runtime_error("Config: rel_tol must be > 0");
    if (cfg.dt_init   <= 0.0) throw std::runtime_error("Config: dt_init must be > 0");
    if (cfg.init_mode != "lattice" && cfg.init_mode != "random")
        throw std::runtime_error(
            "Config: init_mode must be \"lattice\" or \"random\" (got \"" +
            cfg.init_mode + "\")");

    cfg.recompute();
    return cfg;
}

void Config::recompute() {
    // Anchor friction: gamma_a / gamma = R.
    gamma = gamma_hat / (1.0 + R);
    gamma_a = R * gamma;

    // set epsilon based off of gamma and tau_elastic
    epsilon = (gamma * (sigma * sigma)) / tau_elastic;

    // 2D packing: phi = N * pi * sigma^2 / (4 L^2).
    L = std::sqrt(static_cast<double>(N) * pi() * sigma * sigma / (4.0 * phi));
    
    // Thermal energy.
    kT = C * epsilon;

    // Active force from typical overlap (force balance F_pot(delta * sigma) = f0).
    f0 = ForceCalculator::f0ForOverlap(potential, delta, epsilon, sigma);

    // Persistence time from Pe = f0 * tau_theta / (gamma * sigma).
    // Degenerate when there's no active drive (f0 = 0) or no persistence
    // requested (Pe = 0): no rotational diffusion to model.
    if (f0 > 0.0 && Pe > 0.0) {
        tau_theta = (Pe * gamma_hat * sigma) / f0;
    } else {
        tau_theta = 0.0;
    }

    // Anchor stiffness from De = (gamma_a / k_a) / tau_theta.
    // Degenerate when there's no persistence (tau_theta = 0) or De = 0
    // (which would imply k_a = infinity); decouple anchors in those cases.
    if (De > 0.0 && tau_theta > 0.0 && gamma_a > 0.0) {
        k_a = gamma_a / (De * tau_theta);
    } else {
        k_a = 0.0;
    }
}

void Config::print() const {
    std::cout
        << "Configuration:\n"
        << "  N            = " << N << "\n"
        << "  phi          = " << phi << "\n"
        << "  potential    = " << potentialName() << "\n"
        << "  --- inputs (dimensionless) ---\n"
        << "  delta        = " << delta << "\n"
        << "  Pe           = " << Pe << "\n"
        << "  De           = " << De << "\n"
        << "  R            = " << R << "\n"
        << "  C (kT/eps)   = " << C << "\n"
        << "  --- working units (frozen) ---\n"
        << "  sigma        = " << sigma << "\n"
        << "  epsilon      = " << epsilon << "\n"
        << "  gamma_hat    = " << gamma_hat << "\n"
        << "  --- derived ---\n"
        << "  L            = " << L << "\n"
        << "  kT           = " << kT << "\n"
        << "  gamma        = " << gamma << "\n"
        << "  f0           = " << f0 << (f0 == 0.0 ? " (passive)\n" : "\n")
        << "  tau_theta    = " << tau_theta
        << (tau_theta == 0.0 ? " (no rotational diffusion)\n" : "\n")
        << "  gamma_a      = " << gamma_a
        << (gamma_a == 0.0 ? " (anchors frozen)\n" : "\n")
        << "  k_a          = " << k_a
        << (k_a == 0.0 ? " (no anchor coupling)\n" : "\n")
        << "  D = kT/gamma = " << diffusionCoefficient() << "\n"
        << "  --- integrator + I/O ---\n"
        << "  integrator   = " << integratorName() << "\n"
        << "  dt           = " << dt_init << "\n"
        << "  t_end        = " << t_end << "\n"
        << "  output_dt    = " << output_dt << "\n";
    if (integrator == IntegratorMethod::AdaptiveStrang) {
        std::cout
            << "  abs_tol      = " << abs_tol << "\n"
            << "  rel_tol      = " << rel_tol << "\n";
    } else {
        // EM and Heun share the same fixed-dt knobs.
        std::cout
            << "  max_drift    = " << max_drift << "\n"
            << "  r_skin       = " << r_skin << "\n";
    }
    std::cout
        << "  output_file  = " << output_file << "\n"
        << "  init_mode    = " << init_mode << "\n"
        << "  seed         = " << seed << "\n";
}
