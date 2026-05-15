// =============================================================================
// main.cpp
// -----------------------------------------------------------------------------
// Single entry point for every kind of run. Subcommand-style CLI:
//
//      sim run   <input.json>     Run a Brownian-dynamics simulation.
//      sim info  <input.json>     Parse the input and print the resolved config.
//      sim help                   Show usage.
//
// Two integrator backends, picked by the JSON "integrator" field:
//
//   "euler_maruyama" (default):  fixed-Δt Euler-Maruyama with a drift cap
//                                (cfg.max_drift) and cell-list pair forces
//                                (cfg.r_skin). Δt = cfg.dt_init.
//   "adaptive_strang":           Strang(N-D-N) at a fixed macro Δt = cfg.dt_init,
//                                with an inner Cash-Karp 5(4) sub-stepper
//                                controlled by (cfg.abs_tol, cfg.rel_tol).
//                                AdaptiveIntegrator manages its own cell list.
//
// Output cadence is time-based for both: simulate from t = 0 to cfg.t_end and
// write a frame every cfg.output_dt. The last step before each output boundary
// is shortened so frames land exactly on multiples of output_dt.
// =============================================================================

#include "AdaptiveIntegrator.hpp"
#include "Box.hpp"
#include "CellList.hpp"
#include "Config.hpp"
#include "ContactDurationAccumulator.hpp"
#include "CorrelationAccumulator.hpp"
#include "ForceCalculator.hpp"
#include "HeunIntegrator.hpp"
#include "Initializer.hpp"
#include "Integrator.hpp"
#include "RandomGenerator.hpp"
#include "System.hpp"
#include "TrajectoryWriter.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

void printUsage(const char* progname) {
    std::cout
        << "Usage:\n"
        << "  " << progname << " run   <input.json>   Run a Brownian-dynamics simulation.\n"
        << "  " << progname << " info  <input.json>   Print the parsed configuration.\n"
        << "  " << progname << " help                 Show this message.\n";
}

int cmdInfo(const std::string& jsonPath) {
    Config cfg = Config::fromFile(jsonPath);
    cfg.print();
    return EXIT_SUCCESS;
}

// Run the time loop with the adaptive Strang(N-D-N) integrator. Macro Δt is
// fixed but the *last* macro step before each output boundary is shortened
// so frames land exactly on multiples of output_dt; the cap is restored after.
void runAdaptive(const Config& cfg, System& sys, Box& box,
                 ForceCalculator& forces, RandomGenerator& rng,
                 TrajectoryWriter& writer,
                 CorrelationAccumulator* corr, double corr_dt,
                 ContactDurationAccumulator* contacts) {
    AdaptiveIntegrator integ(cfg.gamma, cfg.abs_tol, cfg.rel_tol, cfg.dt_init);
    integ.setMacroDt(cfg.dt_init);
    integ.setKBT(cfg.kT);
    integ.setActiveForce(cfg.f0);
    if (cfg.tau_theta > 0.0)
        integ.setRotationalDiffusion(1.0 / cfg.tau_theta);
    integ.setSpringStiffness(cfg.k_a);
    integ.setAnchorFriction(cfg.gamma_a);

    std::cout << "Running Strang(N-D-N) at macro Δt = " << cfg.dt_init
              << " until t = " << cfg.t_end
              << ", output every " << cfg.output_dt << " of t...\n";

    constexpr double rel_eps    = 1.0e-10;
    const double     macro_dt   = cfg.dt_init;
    double           t          = 0.0;
    double           next_output= cfg.output_dt;
    double           next_corr_sample = corr
        ? std::max(cfg.t_warm, corr_dt)
        : std::numeric_limits<double>::infinity();
    std::size_t      n_steps    = 0;
    std::size_t      n_frames   = 1;

    while (t < cfg.t_end * (1.0 - rel_eps)) {
        const double next_target = std::min({next_output, next_corr_sample, cfg.t_end});
        const double remaining   = next_target - t;
        const double step_dt     = std::min(macro_dt, remaining);
        integ.setMacroDt(step_dt);

        const double dt_taken = integ.step(sys, box, forces, rng);
        t += dt_taken;
        ++n_steps;

        if (contacts) contacts->sample(sys, box, dt_taken);

        const bool need_corr  = corr && t >= cfg.t_warm * (1.0 - rel_eps)
                                     && t >= next_corr_sample * (1.0 - rel_eps);
        const bool need_write = t >= next_output * (1.0 - rel_eps);

        if (need_corr) {
            forces.compute(sys, box);
            corr->sample(sys);
            next_corr_sample += corr_dt;
        }

        if (need_write) {
            forces.compute(sys, box);
            writer.writeFrame(sys, box, n_steps, t);
            const double U = forces.computeEnergy(sys, box);
            std::cout << "  t = " << t << " / " << cfg.t_end
                      << "    U/N = " << U / static_cast<double>(cfg.N)
                      << "    macro_steps = " << n_steps
                      << "    cl_rebuilds = " << integ.getCellListRebuilds()
                      << '\n';
            ++n_frames;
            next_output += cfg.output_dt;
        }
    }

    std::cout << "Done. " << n_steps << " macro steps, "
              << n_frames << " frames written.\n";
}

// Run the time loop with stochastic Heun (predictor-corrector). Two force
// evaluations per step share the same cell list (predictor displacement is
// bounded by max_drift + |Z| sqrt(2 D dt) << r_skin/2 for default skin), so
// we rebuild the list at most once per full step. Boundary-snapping mirrors
// the EM runner.
void runHeun(const Config& cfg, System& sys, Box& box,
             ForceCalculator& forces, RandomGenerator& rng,
             TrajectoryWriter& writer,
             CorrelationAccumulator* corr, double corr_dt,
             ContactDurationAccumulator* contacts) {
    HeunIntegrator integ(cfg.gamma, cfg.dt_init);
    integ.setKBT(cfg.kT);
    integ.setActiveForce(cfg.f0);
    if (cfg.tau_theta > 0.0)
        integ.setRotationalDiffusion(1.0 / cfg.tau_theta);
    integ.setSpringStiffness(cfg.k_a);
    integ.setAnchorFriction(cfg.gamma_a);
    integ.setMaxDrift(cfg.max_drift);

    CellList cl(forces.getCutoff(), cfg.r_skin, cfg.N, box);
    if (!cl.useBruteForce())
        cl.rebuild(sys, box);

    std::cout << "Running stochastic Heun at Δt = " << cfg.dt_init
              << " until t = " << cfg.t_end
              << ", output every " << cfg.output_dt << " of t"
              << (cl.useBruteForce()
                      ? " (brute-force forces; box too small for cell list).\n"
                      : ".\n");

    constexpr double rel_eps    = 1.0e-10;
    const double     dt_full    = cfg.dt_init;
    double           current_dt = dt_full;
    double           t          = 0.0;
    double           next_output= cfg.output_dt;
    double           next_corr_sample = corr
        ? std::max(cfg.t_warm, corr_dt)
        : std::numeric_limits<double>::infinity();
    std::size_t      n_steps    = 0;
    std::size_t      n_frames   = 1;

    while (t < cfg.t_end * (1.0 - rel_eps)) {
        const double next_target = std::min({next_output, next_corr_sample, cfg.t_end});
        const double remaining   = next_target - t;
        const double step_dt     = std::min(dt_full, remaining);
        if (step_dt != current_dt) {
            integ.setTimestep(step_dt);
            current_dt = step_dt;
        }

        if (!cl.useBruteForce() && cl.needsRebuild(sys, box))
            cl.rebuild(sys, box);

        integ.step(sys, box, forces, &cl, rng);
        t += step_dt;
        ++n_steps;

        if (contacts) contacts->sample(sys, box, step_dt);

        const bool need_corr  = corr && t >= cfg.t_warm * (1.0 - rel_eps)
                                     && t >= next_corr_sample * (1.0 - rel_eps);
        const bool need_write = t >= next_output * (1.0 - rel_eps);

        if (need_corr) {
            forces.compute(sys, box);
            corr->sample(sys);
            next_corr_sample += corr_dt;
        }

        if (need_write) {
            forces.compute(sys, box);    // brute force for the energy diagnostic
            writer.writeFrame(sys, box, n_steps, t);
            const double U = forces.computeEnergy(sys, box);
            std::cout << "  t = " << t << " / " << cfg.t_end
                      << "    U/N = " << U / static_cast<double>(cfg.N)
                      << "    steps = " << n_steps
                      << "    cl_rebuilds = " << cl.getNumRebuilds()
                      << '\n';
            ++n_frames;
            next_output += cfg.output_dt;
        }
    }

    std::cout << "Done. " << n_steps << " Heun steps, "
              << n_frames << " frames written.\n";
}

// Run the time loop with plain Euler-Maruyama using a Verlet/cell list for
// the pair forces. Same boundary-snapping policy as runAdaptive: the last EM
// step before each output is shortened (we only re-cache integrator
// coefficients when the dt actually changes, since recomputing involves a
// few sqrts per call).
void runEulerMaruyama(const Config& cfg, System& sys, Box& box,
                      ForceCalculator& forces, RandomGenerator& rng,
                      TrajectoryWriter& writer,
                      CorrelationAccumulator* corr, double corr_dt,
                      ContactDurationAccumulator* contacts) {
    Integrator integ(cfg.gamma, cfg.dt_init);
    integ.setKBT(cfg.kT);
    integ.setActiveForce(cfg.f0);
    if (cfg.tau_theta > 0.0)
        integ.setRotationalDiffusion(1.0 / cfg.tau_theta);
    integ.setSpringStiffness(cfg.k_a);
    integ.setAnchorFriction(cfg.gamma_a);
    integ.setMaxDrift(cfg.max_drift);

    CellList cl(forces.getCutoff(), cfg.r_skin, cfg.N, box);
    if (!cl.useBruteForce())
        cl.rebuild(sys, box);

    auto computeForces = [&] {
        if (cl.useBruteForce()) {
            forces.compute(sys, box);
        } else {
            if (cl.needsRebuild(sys, box)) cl.rebuild(sys, box);
            forces.compute(sys, box, cl);
        }
    };

    std::cout << "Running Euler-Maruyama at Δt = " << cfg.dt_init
              << " until t = " << cfg.t_end
              << ", output every " << cfg.output_dt << " of t"
              << (cl.useBruteForce()
                      ? " (brute-force forces; box too small for cell list).\n"
                      : ".\n");

    constexpr double rel_eps    = 1.0e-10;
    const double     dt_full    = cfg.dt_init;
    double           current_dt = dt_full;
    double           t          = 0.0;
    double           next_output= cfg.output_dt;
    double           next_corr_sample = corr
        ? std::max(cfg.t_warm, corr_dt)
        : std::numeric_limits<double>::infinity();
    std::size_t      n_steps    = 0;
    std::size_t      n_frames   = 1;

    while (t < cfg.t_end * (1.0 - rel_eps)) {
        const double next_target = std::min({next_output, next_corr_sample, cfg.t_end});
        const double remaining   = next_target - t;
        const double step_dt     = std::min(dt_full, remaining);
        if (step_dt != current_dt) {
            integ.setTimestep(step_dt);
            current_dt = step_dt;
        }

        computeForces();
        integ.step(sys, box, rng);
        t += step_dt;
        ++n_steps;

        if (contacts) contacts->sample(sys, box, step_dt);

        const bool need_corr  = corr && t >= cfg.t_warm * (1.0 - rel_eps)
                                     && t >= next_corr_sample * (1.0 - rel_eps);
        const bool need_write = t >= next_output * (1.0 - rel_eps);

        if (need_corr) {
            forces.compute(sys, box);
            corr->sample(sys);
            next_corr_sample += corr_dt;
        }

        if (need_write) {
            forces.compute(sys, box);    // brute force for the energy diagnostic
            writer.writeFrame(sys, box, n_steps, t);
            const double U = forces.computeEnergy(sys, box);
            std::cout << "  t = " << t << " / " << cfg.t_end
                      << "    U/N = " << U / static_cast<double>(cfg.N)
                      << "    steps = " << n_steps
                      << "    cl_rebuilds = " << cl.getNumRebuilds()
                      << '\n';
            ++n_frames;
            next_output += cfg.output_dt;
        }
    }

    std::cout << "Done. " << n_steps << " EM steps, "
              << n_frames << " frames written.\n";
}

int cmdRun(const std::string& jsonPath) {
    Config cfg = Config::fromFile(jsonPath);
    cfg.print();

    // ---- Build the engine objects ------------------------------------------
    System          sys(cfg.N);
    sys.setSigma(cfg.sigma);

    Box             box(cfg.L);

    ForceCalculator forces(cfg.epsilon, cfg.sigma, cfg.potential);

    RandomGenerator rng(cfg.seed);

    // ---- Initial configuration ---------------------------------------------
    if (cfg.init_mode == "lattice") {
        Initializer::placeOnLattice(sys, box);
    } else {
        Initializer::placeRandomly(sys, box, rng, cfg.sigma);
    }
    if (cfg.f0 > 0.0)
        Initializer::randomizeOrientations(sys, rng);
    Initializer::placeAnchorsAtParticles(sys);

    // ---- Output stream + initial frame -------------------------------------
    TrajectoryWriter writer(cfg.output_file);
    writer.writeSimulationConfig(cfg);
    forces.compute(sys, box);            // for the energy diagnostic only
    writer.writeFrame(sys, box, /*step*/ 0, /*time*/ 0.0);

    const double U0 = forces.computeEnergy(sys, box);
    std::cout
        << "\nInitial energy U0 = " << U0
        << " (per particle: " << U0 / static_cast<double>(cfg.N) << ")\n"
        << (cfg.f0 > 0.0 ? "Active Brownian motion ENABLED  " : "")
        << (cfg.f0 > 0.0 ? ("f0=" + std::to_string(cfg.f0) + "  tau_theta=" +
                            std::to_string(cfg.tau_theta) + "\n") : "");

    // ---- Optional on-the-fly correlations ----------------------------------
    std::unique_ptr<CorrelationAccumulator> corr;
    double corr_dt = 0.0;
    if (cfg.compute_correlations) {
        corr_dt = cfg.corr_dt_max / static_cast<double>(cfg.n_corr_steps);
        corr = std::make_unique<CorrelationAccumulator>(
            cfg.N, cfg.n_corr_steps, corr_dt);
        std::cout << "On-the-fly correlations ENABLED: "
                  << "n_corr_steps=" << cfg.n_corr_steps
                  << ", corr_dt=" << corr_dt
                  << ", corr_dt_max=" << cfg.corr_dt_max
                  << ", t_warm=" << cfg.t_warm << "\n";
    }

    // ---- Optional on-the-fly contact-duration statistics -------------------
    std::unique_ptr<ContactDurationAccumulator> contacts;
    if (cfg.compute_contact_durations) {
        contacts = std::make_unique<ContactDurationAccumulator>(
            cfg.N, cfg.sigma, cfg.r_skin, box);
        std::cout << "Contact-duration tracking ENABLED "
                  << "(contact_cutoff = sigma = " << cfg.sigma << ")\n";
    }

    // ---- Time loop ----------------------------------------------------------
    switch (cfg.integrator) {
        case IntegratorMethod::EulerMaruyama:
            runEulerMaruyama(cfg, sys, box, forces, rng, writer,
                             corr.get(), corr_dt, contacts.get());
            break;
        case IntegratorMethod::Heun:
            runHeun(cfg, sys, box, forces, rng, writer,
                    corr.get(), corr_dt, contacts.get());
            break;
        case IntegratorMethod::AdaptiveStrang:
            runAdaptive(cfg, sys, box, forces, rng, writer,
                        corr.get(), corr_dt, contacts.get());
            break;
    }

    if (corr) {
        writer.writeCorrelations(*corr, cfg.t_warm);
        std::cout << "Wrote /correlations: "
                  << corr->numSamplesTaken() << " samples on a "
                  << corr->nLags() << "-point grid.\n";
    }

    if (contacts) {
        writer.writeContactDurations(*contacts);
        std::cout << "Wrote /contact_durations: count=" << contacts->count()
                  << ", mean=" << contacts->mean()
                  << ", stddev=" << contacts->stddev()
                  << ", in_progress=" << contacts->inProgressCount() << "\n";
    }

    writer.close();
    std::cout << "Trajectory at '" << cfg.output_file << "'.\n";
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    const std::string cmd = argv[1];

    try {
        if (cmd == "help" || cmd == "--help" || cmd == "-h") {
            printUsage(argv[0]);
            return EXIT_SUCCESS;
        }
        if (cmd == "info") {
            if (argc < 3) { printUsage(argv[0]); return EXIT_FAILURE; }
            return cmdInfo(argv[2]);
        }
        if (cmd == "run") {
            if (argc < 3) { printUsage(argv[0]); return EXIT_FAILURE; }
            return cmdRun(argv[2]);
        }
        std::cerr << "Unknown command: '" << cmd << "'\n";
        printUsage(argv[0]);
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
