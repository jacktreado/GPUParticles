// =============================================================================
// SimulationGUI.cpp — interactive viewer for active Brownian particles
//                     coupled to viscoelastic anchors.
// -----------------------------------------------------------------------------
// Working units (frozen at 1):
//
//   sigma   = 1   (length unit: particle diameter)
//   epsilon = 1   (energy unit: pair-interaction scale)
//   gamma   = 1   (particle friction; tau = gamma sigma^2 / eps = 1)
//
// Five dimensionless control parameters drive the dynamics:
//
//   delta = r*/sigma  typical pair overlap; sets f0 by force balance
//                     F_pot(r*) = f0  (see README "Typical pair overlap")
//   Pe = f0*tau_theta/(gamma*sigma)  active Peclet — sets persistence
//                     -> tau_theta = Pe * gamma * sigma / f0
//   De = (gamma_a/k_a)/tau_theta     active Deborah — anchor memory time
//                     -> k_a = gamma_a / (De * tau_theta)
//   R  = gamma_a/gamma               anchor friction ratio
//                     -> gamma_a = R * gamma
//   C  = kT/epsilon                  dimensionless temperature
//                     -> kT = C * epsilon (frozen small for the athermal limit)
//
// Resolution is delegated to Config::recompute() so the GUI and the CLI
// driver use exactly the same formulas.
//
// Time integration: three methods, switched at runtime.
//   * Euler-Maruyama (default): plain fixed-Δt EM with a drift cap. Cheapest
//     per step (1 force eval). Weak/strong order 1 in Δt for additive noise.
//   * Stochastic Heun: fixed-Δt predictor-corrector. 2 force evals per step
//     but weak order 2 — typically a 3-5x larger Δt is admissible at the
//     same accuracy, making it the practical sweet spot for moderate-Δt
//     interactive runs.
//   * Adaptive Strang(N-D-N): N(Δt/2) noise kick, D(Δt) deterministic flow
//     integrated by an inner adaptive Cash-Karp 5(4) sub-stepper controlled
//     by (abs_tol, rel_tol), N(Δt/2) noise kick. Cost grows with Δt because
//     more inner sub-steps are required to bridge the macro interval.
// Switching the method resets the simulation. The slower step-doubling /
// Brownian-bridge macro-adaptive variant lives in AdaptiveMacroIntegrator
// and is not used here — too expensive for the interactive frame rate.
//
// Visualization
// -------------
//   - particles: large circles (radius sigma/2), filled
//   - orientation: fixed-length white arrow per particle
//   - anchors: smaller circles (red), drawn at the minimum-image position
//     so particle + anchor + spring stay visually grouped
//   - springs: thin gray segment between particle and its anchor
//   - box: white outline
//
// No file I/O — the simulation runs entirely in memory.
// =============================================================================

#include "AdaptiveIntegrator.hpp"
#include "Box.hpp"
#include "CellList.hpp"
#include "Config.hpp"
#include "ForceCalculator.hpp"
#include "HeunIntegrator.hpp"
#include "Initializer.hpp"
#include "Integrator.hpp"
#include "RandomGenerator.hpp"
#include "System.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#ifndef GL_SILENCE_DEPRECATION
#  define GL_SILENCE_DEPRECATION
#endif
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>

// ---------------------------------------------------------------------------
// Parameters and resolution
// ---------------------------------------------------------------------------
namespace {

// Athermal-limit C used by the GUI. Small enough that thermal fluctuations
// don't wash out the deterministic force-balance overlap, but nonzero so the
// noise term is exercised.
constexpr float kAthermalC = 1.0e-8f;

struct Params {
    int   N           = 100;
    float phi         = 0.30f;

    // Dimensionless control parameters.
    float delta = 0.95f;    // typical pair overlap
    float Pe    = 10.0f;    // active Peclet  (persistence)
    float De    = 1.0f;     // active Deborah (anchor memory)
    float R     = 1.0f;     // gamma_a / gamma

    // Cash-Karp 5(4) tolerances — only used when the Adaptive Strang method
    // is selected. Greyed out for Euler-Maruyama.
    float abs_tol  = 1.0e-5f;
    float rel_tol  = 1.0e-5f;
    // Integration time step. For Euler-Maruyama this is the EM dt; for
    // Adaptive Strang it is the fixed macro Δt bridged by the inner Cash-Karp
    // sub-stepper. Default chosen so that one step is well below the
    // rotational and spring timescales for typical (Pe, De) settings.
    float macro_dt = 1.0e-2f;

    int   steps_per_frame = 5;
    int   seed            = 42;

    // Integration method: index into kIntegratorNames below. 0 = EM (default).
    int   integrator_idx = 0;

    // Pair potential: index into kPotentialNames / kPotentialValues below.
    int   potential_idx = 0;

    // Visualization toggles (purely visual — no effect on dynamics).
    bool  show_anchors      = true;
    bool  show_orientations = true;
};

// Order matters: the index is what we store in Params and what ImGui::Combo
// uses. Keep names and values in lockstep.
constexpr const char* kPotentialNames[]  = { "WCA", "Soft sphere" };
constexpr PotentialType kPotentialValues[] = {
    PotentialType::WCA,
    PotentialType::SoftSphere,
};

// Integration methods. Index 0 (EM) is the default — see Params::integrator_idx.
constexpr const char* kIntegratorNames[] = {
    "Euler-Maruyama",
    "Stochastic Heun",
    "Adaptive Strang(N-D-N)",
};
enum IntegratorKind : int {
    INTEG_EM       = 0,
    INTEG_HEUN     = 1,
    INTEG_ADAPTIVE = 2,
};

// Drift cap for Euler-Maruyama: prevents blow-up at small kT / stiff WCA.
// 0.1 sigma matches the project default (see CLI driver).
constexpr double kEMMaxDrift = 0.1;

// Verlet skin used by the Euler-Maruyama path's cell list. 0.5 sigma matches
// AdaptiveIntegrator's default; rebuilds are triggered at r_skin/2 drift.
constexpr double kEMCellListSkin = 0.5;

// How often to refresh the U/N status display. Energy is O(N^2) and was the
// dominant per-frame cost at moderate N once forces moved to the cell list.
constexpr double kEnergyRefreshSeconds = 0.25;

// Use Config::recompute() so the GUI and CLI go through identical formulas.
// The returned Config holds both the inputs and every derived microscopic
// quantity (f0, tau_theta, gamma_a, k_a, kT, L).
Config resolveAsConfig(const Params& p) {
    Config cfg;
    cfg.N         = static_cast<std::size_t>(p.N);
    cfg.phi       = p.phi;
    cfg.Pe        = p.Pe;
    cfg.De        = p.De;
    cfg.R         = p.R;
    cfg.C         = kAthermalC;
    cfg.delta     = p.delta;
    cfg.potential = kPotentialValues[p.potential_idx];
    // dt is not a user input here — the adaptive integrator picks it. Leave
    // Config's dt at its default; it isn't read by anything the GUI uses.
    cfg.recompute();
    return cfg;
}

// ---------------------------------------------------------------------------
// Simulation state
// ---------------------------------------------------------------------------
//
// Exactly one of em_integ / adaptive_integ is non-null, set by rebuild().
// The Euler-Maruyama integrator runs at a fixed user-set Δt; the adaptive
// integrator owns its own cell list and runs Strang(N-D-N) with an inner
// Cash-Karp 5(4) sub-stepper (controlled by abs_tol/rel_tol).
// ---------------------------------------------------------------------------
struct Sim {
    std::unique_ptr<System>              sys;
    std::unique_ptr<Box>                 box;
    std::unique_ptr<ForceCalculator>     forces;
    std::unique_ptr<Integrator>          em_integ;
    std::unique_ptr<HeunIntegrator>      heun_integ;
    std::unique_ptr<AdaptiveIntegrator>  adaptive_integ;
    std::unique_ptr<CellList>            em_cl;     // shared by EM/Heun; adaptive owns its own
    std::unique_ptr<RandomGenerator>     rng;

    Config      cfg{};
    long long   step_count        = 0;
    double      sim_time          = 0.0;  // accumulated dt across all (macro) steps
    double      last_dt           = 0.0;  // dt of the most recently accepted step
    std::size_t em_cl_rebuilds    = 0;    // EM cell-list rebuild counter

    // Throttled U/N cache for the status display. Recomputed at most once per
    // kEnergyRefreshSeconds; energy is O(N^2) and was the bottleneck after
    // forces moved to the cell list.
    double      cached_U_per_N    = 0.0;
    double      next_energy_time  = 0.0;  // ImGui::GetTime() at which to refresh
};

void rebuild(Sim& sim, const Params& p) {
    sim.cfg              = resolveAsConfig(p);
    sim.step_count       = 0;
    sim.sim_time         = 0.0;
    sim.last_dt          = 0.0;
    sim.em_cl_rebuilds   = 0;
    sim.cached_U_per_N   = 0.0;
    sim.next_energy_time = 0.0;

    sim.sys    = std::make_unique<System>(sim.cfg.N);
    sim.sys->setSigma(sim.cfg.sigma);
    sim.box    = std::make_unique<Box>(sim.cfg.L);
    sim.forces = std::make_unique<ForceCalculator>(
        sim.cfg.epsilon, sim.cfg.sigma, sim.cfg.potential);

    // Build exactly one integrator; tear down whichever was previously alive.
    sim.em_integ.reset();
    sim.heun_integ.reset();
    sim.adaptive_integ.reset();
    sim.em_cl.reset();

    const double D_r = (sim.cfg.tau_theta > 0.0) ? 1.0 / sim.cfg.tau_theta : 0.0;
    const bool   fixed_dt_method = (p.integrator_idx == INTEG_EM ||
                                    p.integrator_idx == INTEG_HEUN);

    if (p.integrator_idx == INTEG_EM) {
        sim.em_integ = std::make_unique<Integrator>(sim.cfg.gamma, p.macro_dt);
        sim.em_integ->setKBT(sim.cfg.kT);
        sim.em_integ->setActiveForce(sim.cfg.f0);
        sim.em_integ->setRotationalDiffusion(D_r);
        sim.em_integ->setSpringStiffness(sim.cfg.k_a);
        sim.em_integ->setAnchorFriction(sim.cfg.gamma_a);
        sim.em_integ->setMaxDrift(kEMMaxDrift);
    } else if (p.integrator_idx == INTEG_HEUN) {
        sim.heun_integ = std::make_unique<HeunIntegrator>(sim.cfg.gamma, p.macro_dt);
        sim.heun_integ->setKBT(sim.cfg.kT);
        sim.heun_integ->setActiveForce(sim.cfg.f0);
        sim.heun_integ->setRotationalDiffusion(D_r);
        sim.heun_integ->setSpringStiffness(sim.cfg.k_a);
        sim.heun_integ->setAnchorFriction(sim.cfg.gamma_a);
        sim.heun_integ->setMaxDrift(kEMMaxDrift);
    } else {
        sim.adaptive_integ = std::make_unique<AdaptiveIntegrator>(
            sim.cfg.gamma, p.abs_tol, p.rel_tol, /*dt_init*/ p.macro_dt);
        sim.adaptive_integ->setMacroDt(p.macro_dt);
        sim.adaptive_integ->setKBT(sim.cfg.kT);
        sim.adaptive_integ->setActiveForce(sim.cfg.f0);
        sim.adaptive_integ->setRotationalDiffusion(D_r);
        sim.adaptive_integ->setSpringStiffness(sim.cfg.k_a);
        sim.adaptive_integ->setAnchorFriction(sim.cfg.gamma_a);
    }

    // Cell list shared by both fixed-dt methods (EM, Heun). Falls back to
    // brute force when the box is too small for the 3x3 PBC stencil.
    if (fixed_dt_method) {
        sim.em_cl = std::make_unique<CellList>(
            sim.forces->getCutoff(), kEMCellListSkin, sim.cfg.N, *sim.box);
    }

    sim.rng = std::make_unique<RandomGenerator>(p.seed);

    Initializer::placeOnLattice(*sim.sys, *sim.box);
    Initializer::randomizeOrientations(*sim.sys, *sim.rng);
    Initializer::placeAnchorsAtParticles(*sim.sys);

    if (sim.em_cl && !sim.em_cl->useBruteForce()) {
        sim.em_cl->rebuild(*sim.sys, *sim.box);
        ++sim.em_cl_rebuilds;
    }
}

// Update integrator parameters in-place without resetting positions. Method
// changes go through rebuild() instead, since the two integrators don't share
// state.
void updateIntegrator(Sim& sim, const Params& p) {
    sim.cfg = resolveAsConfig(p);

    const double D_r = (sim.cfg.tau_theta > 0.0) ? 1.0 / sim.cfg.tau_theta : 0.0;

    if (sim.em_integ) {
        sim.em_integ->setKBT(sim.cfg.kT);
        sim.em_integ->setActiveForce(sim.cfg.f0);
        sim.em_integ->setRotationalDiffusion(D_r);
        sim.em_integ->setSpringStiffness(sim.cfg.k_a);
        sim.em_integ->setAnchorFriction(sim.cfg.gamma_a);
        sim.em_integ->setTimestep(p.macro_dt);
    } else if (sim.heun_integ) {
        sim.heun_integ->setKBT(sim.cfg.kT);
        sim.heun_integ->setActiveForce(sim.cfg.f0);
        sim.heun_integ->setRotationalDiffusion(D_r);
        sim.heun_integ->setSpringStiffness(sim.cfg.k_a);
        sim.heun_integ->setAnchorFriction(sim.cfg.gamma_a);
        sim.heun_integ->setTimestep(p.macro_dt);
    } else if (sim.adaptive_integ) {
        sim.adaptive_integ->setKBT(sim.cfg.kT);
        sim.adaptive_integ->setActiveForce(sim.cfg.f0);
        sim.adaptive_integ->setRotationalDiffusion(D_r);
        sim.adaptive_integ->setSpringStiffness(sim.cfg.k_a);
        sim.adaptive_integ->setAnchorFriction(sim.cfg.gamma_a);
        sim.adaptive_integ->setMacroDt(p.macro_dt);
        sim.adaptive_integ->setAbsTol(p.abs_tol);
        sim.adaptive_integ->setRelTol(p.rel_tol);
    }
}

void stepSim(Sim& sim, int n) {
    // Shared cell-list rebuild check for the fixed-dt methods. Heun's
    // predictor displacement is bounded under r_skin/2, so one rebuild check
    // per full step (not per stage) is sufficient.
    auto refreshCellList = [&]() -> bool {
        const bool use_cl = sim.em_cl && !sim.em_cl->useBruteForce();
        if (use_cl && sim.em_cl->needsRebuild(*sim.sys, *sim.box)) {
            sim.em_cl->rebuild(*sim.sys, *sim.box);
            ++sim.em_cl_rebuilds;
        }
        return use_cl;
    };

    for (int s = 0; s < n; ++s) {
        if (sim.em_integ) {
            // Plain Euler-Maruyama: one cell-list force eval, then EM step.
            // Cost is independent of dt — this is the smooth-FPS path.
            const bool use_cl = refreshCellList();
            if (use_cl) sim.forces->compute(*sim.sys, *sim.box, *sim.em_cl);
            else        sim.forces->compute(*sim.sys, *sim.box);
            sim.em_integ->step(*sim.sys, *sim.box, *sim.rng);
            sim.last_dt   = sim.em_integ->getTimestep();
            sim.sim_time += sim.last_dt;
        } else if (sim.heun_integ) {
            // Stochastic Heun: predictor + corrector. HeunIntegrator does
            // both force evals internally, sharing the cell list across
            // stages. Cost ~2x EM per step but weak order 2.
            refreshCellList();
            sim.heun_integ->step(*sim.sys, *sim.box, *sim.forces,
                                 sim.em_cl.get(), *sim.rng);
            sim.last_dt   = sim.heun_integ->getTimestep();
            sim.sim_time += sim.last_dt;
        } else {
            // One Strang(N-D-N) macro step at the user-fixed Δt.
            const double dt_taken = sim.adaptive_integ->step(
                *sim.sys, *sim.box, *sim.forces, *sim.rng);
            sim.last_dt   = dt_taken;
            sim.sim_time += dt_taken;
        }
    }
    sim.step_count += n;
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------
void drawSimulation(ImDrawList* dl, ImVec2 p0, ImVec2 p1, const Sim& sim,
                    bool show_anchors, bool show_orientations) {
    const float pad      = 8.0f;
    const float canvas_w = (p1.x - p0.x) - 2 * pad;
    const float canvas_h = (p1.y - p0.y) - 2 * pad;
    const float scale    = std::min(canvas_w, canvas_h) /
                           static_cast<float>(std::max(sim.cfg.L, 1e-9));
    const float cx0      = p0.x + pad;
    const float cy_bot   = p0.y + pad + canvas_h;     // y=0 at bottom

    auto worldToScreen = [&](double wx, double wy) {
        return ImVec2(cx0 + scale * static_cast<float>(wx),
                      cy_bot - scale * static_cast<float>(wy));
    };

    // Filled background for the canvas.
    dl->AddRectFilled(p0, p1, IM_COL32(28, 28, 46, 255));

    // Box outline.
    {
        ImVec2 bl = worldToScreen(0.0,        0.0);
        ImVec2 tr = worldToScreen(sim.cfg.L,  sim.cfg.L);
        dl->AddRect(ImVec2(bl.x, tr.y), ImVec2(tr.x, bl.y),
                    IM_COL32(255, 255, 255, 220), 0.0f, 0, 1.5f);
    }

    const float r_particle = 0.50f * scale;
    const float r_anchor   = 0.18f * scale;
    const float arrow_len  = 0.70f * scale;     // fixed: 0.7 sigma in world units

    const ImU32 part_col   = IM_COL32( 70, 130, 180, 230);   // steelblue
    const ImU32 part_edge  = IM_COL32( 25,  60, 110, 255);
    const ImU32 anchor_col = IM_COL32(220,  85,  85, 230);
    const ImU32 spring_col = IM_COL32(170, 170, 170, 180);
    const ImU32 arrow_col  = IM_COL32(255, 255, 255, 255);

    const std::size_t N = sim.sys->getNumParticles();
    const bool        anchors_visible = show_anchors && (sim.cfg.k_a > 0.0);

    // Pass 1: springs.
    if (anchors_visible) {
        for (std::size_t i = 0; i < N; ++i) {
            double dx = sim.sys->getX(i) - sim.sys->getAx(i);
            double dy = sim.sys->getY(i) - sim.sys->getAy(i);
            sim.box->minimumImage(dx, dy);
            const double ax_eff = sim.sys->getX(i) - dx;
            const double ay_eff = sim.sys->getY(i) - dy;

            ImVec2 ps = worldToScreen(sim.sys->getX(i), sim.sys->getY(i));
            ImVec2 as = worldToScreen(ax_eff, ay_eff);
            dl->AddLine(ps, as, spring_col, 1.0f);
        }
    }

    // Pass 2: anchors.
    if (anchors_visible) {
        for (std::size_t i = 0; i < N; ++i) {
            double dx = sim.sys->getX(i) - sim.sys->getAx(i);
            double dy = sim.sys->getY(i) - sim.sys->getAy(i);
            sim.box->minimumImage(dx, dy);
            const double ax_eff = sim.sys->getX(i) - dx;
            const double ay_eff = sim.sys->getY(i) - dy;
            ImVec2 as = worldToScreen(ax_eff, ay_eff);
            dl->AddCircleFilled(as, r_anchor, anchor_col);
        }
    }

    // Pass 3: particles + (optional) orientation arrows.
    for (std::size_t i = 0; i < N; ++i) {
        ImVec2 sp = worldToScreen(sim.sys->getX(i), sim.sys->getY(i));
        dl->AddCircleFilled(sp, r_particle, part_col, 24);
        dl->AddCircle      (sp, r_particle, part_edge, 24, 1.0f);

        if (!show_orientations) continue;

        const float ct = std::cos(static_cast<float>(sim.sys->getTheta(i)));
        const float st = std::sin(static_cast<float>(sim.sys->getTheta(i)));
        ImVec2 tip(sp.x + arrow_len * ct, sp.y - arrow_len * st);
        dl->AddLine(sp, tip, arrow_col, 1.5f);

        const float head_len = 0.30f * arrow_len;
        const float ca = std::cos(2.5f), sa = std::sin(2.5f);
        ImVec2 left ( tip.x + head_len * (ct * ca - st * sa),
                      tip.y - head_len * (st * ca + ct * sa) );
        ImVec2 right( tip.x + head_len * (ct * ca + st * sa),
                      tip.y - head_len * (st * ca - ct * sa) );
        dl->AddLine(tip, left,  arrow_col, 1.5f);
        dl->AddLine(tip, right, arrow_col, 1.5f);
    }
}

double meanEnergyPerParticle(const Sim& sim) {
    // Energy is a brute-force O(N^2) reduction over pairs. The GUI runs at a
    // small enough N that there's no win in routing it through a cell list,
    // and AdaptiveIntegrator already does brute force internally.
    const double U = sim.forces->computeEnergy(*sim.sys, *sim.box);
    return U / static_cast<double>(sim.sys->getNumParticles());
}

// Throttled wrapper: returns the cached value unless the refresh interval has
// elapsed. Energy is the per-frame status display only — there's no need to
// recompute it on every frame at moderate-to-large N.
double cachedMeanEnergyPerParticle(Sim& sim) {
    const double now = ImGui::GetTime();
    if (now >= sim.next_energy_time) {
        sim.cached_U_per_N   = meanEnergyPerParticle(sim);
        sim.next_energy_time = now + kEnergyRefreshSeconds;
    }
    return sim.cached_U_per_N;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
int main(int /*argc*/, char** /*argv*/) {
    glfwSetErrorCallback([](int code, const char* msg) {
        std::fprintf(stderr, "GLFW error %d: %s\n", code, msg);
    });
    if (!glfwInit()) return 1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,        GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(
        1280, 820, "GPUParticles — interactive ABP+anchor", nullptr, nullptr);
    if (!window) { glfwTerminate(); return 1; }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 150");

    Params params;
    Sim    sim;
    rebuild(sim, params);

    bool running           = false;
    bool needs_rebuild     = false;
    bool needs_integ_update = false;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // ----- Controls panel ----------------------------------------------
        ImGui::SetNextWindowPos (ImVec2(10, 10),  ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(440, 800), ImGuiCond_FirstUseEver);
        ImGui::Begin("Controls");

        ImGui::TextColored(ImVec4(0.7f, 0.85f, 1.0f, 1.0f), "System");
        ImGui::SliderInt("N (particles)", &params.N, 4, 4096);
        if (ImGui::IsItemDeactivatedAfterEdit()) needs_rebuild = true;
        ImGui::SliderFloat("phi (packing)", &params.phi, 0.05f, 0.70f, "%.3f");
        if (ImGui::IsItemDeactivatedAfterEdit()) needs_rebuild = true;
        ImGui::Text("L (box) = %.3f sigma", sim.cfg.L);

        // Potential type. Cutoff differs per potential, so the cell list has
        // to be rebuilt from scratch when this changes.
        if (ImGui::Combo("Pair potential", &params.potential_idx,
                         kPotentialNames, IM_ARRAYSIZE(kPotentialNames))) {
            needs_rebuild = true;
        }

        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.7f, 0.85f, 1.0f, 1.0f),
                           "Dimensionless parameters");
        ImGui::TextDisabled("(sigma = epsilon = gamma = 1)");

        if (ImGui::SliderFloat("delta = r* / sigma  (overlap)",
                                &params.delta, 0.50f, 1.122f, "%.4f"))
            needs_integ_update = true;
        if (ImGui::SliderFloat("Pe = f0 tau_theta / (gamma sigma)",
                                &params.Pe, 0.01f, 1000.0f, "%.4f",
                                ImGuiSliderFlags_Logarithmic))
            needs_integ_update = true;
        if (ImGui::SliderFloat("De = (gamma_a/k_a) / tau_theta",
                                &params.De, 0.001f, 1000.0f, "%.4f",
                                ImGuiSliderFlags_Logarithmic))
            needs_integ_update = true;
        if (ImGui::SliderFloat("R = gamma_a / gamma",
                                &params.R, 0.01f, 100.0f, "%.4f",
                                ImGuiSliderFlags_Logarithmic))
            needs_integ_update = true;
        ImGui::Text("C = kT / epsilon = %.0e  (frozen, athermal)",
                    static_cast<double>(kAthermalC));

        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.7f, 0.85f, 1.0f, 1.0f), "Time integration");
        if (ImGui::Combo("Method", &params.integrator_idx,
                         kIntegratorNames, IM_ARRAYSIZE(kIntegratorNames))) {
            needs_rebuild = true;
        }
        const bool fixed_dt_method = (params.integrator_idx == INTEG_EM ||
                                      params.integrator_idx == INTEG_HEUN);
        if (params.integrator_idx == INTEG_EM) {
            ImGui::TextDisabled("Fixed-Δt Euler-Maruyama with drift cap "
                                "(%.2g sigma). Weak order 1.", kEMMaxDrift);
        } else if (params.integrator_idx == INTEG_HEUN) {
            ImGui::TextDisabled("Fixed-Δt predictor-corrector with drift cap "
                                "(%.2g sigma). Weak order 2 (~2x EM cost / "
                                "step).", kEMMaxDrift);
        } else {
            ImGui::TextDisabled(
                "Strang(N-D-N) macro step; inner Cash-Karp 5(4) "
                "uses (abs_tol, rel_tol).");
        }
        // Format: %g with explicit precision (NOT %e). ImGui's logarithmic
        // slider derives its zero-epsilon from the format's precision, but
        // ImParseFormatPrecision() forces precision = -1 for any %e/%E format,
        // which yields epsilon = 10 — larger than the entire slider range here,
        // collapsing the log mapping so the handle only snaps to min/max.
        if (ImGui::SliderFloat("Δt", &params.macro_dt,
                                1e-5f, 1e-1f, "%.7g",
                                ImGuiSliderFlags_Logarithmic))
            needs_integ_update = true;

        // Tolerances are only meaningful for the adaptive method.
        if (fixed_dt_method) ImGui::BeginDisabled();
        if (ImGui::SliderFloat("abs_tol", &params.abs_tol,
                                1e-10f, 1e-3f, "%.12g",
                                ImGuiSliderFlags_Logarithmic))
            needs_integ_update = true;
        if (ImGui::SliderFloat("rel_tol", &params.rel_tol,
                                1e-10f, 1e-3f, "%.12g",
                                ImGuiSliderFlags_Logarithmic))
            needs_integ_update = true;
        if (fixed_dt_method) ImGui::EndDisabled();
        ImGui::SliderInt("steps per frame", &params.steps_per_frame, 1, 200);
        ImGui::SliderInt("seed", &params.seed, 0, 100000);
        if (ImGui::IsItemDeactivatedAfterEdit()) needs_rebuild = true;

        ImGui::Separator();
        ImGui::Checkbox("Run", &running);
        ImGui::SameLine();
        if (ImGui::Button("Step once")) stepSim(sim, params.steps_per_frame);
        ImGui::SameLine();
        if (ImGui::Button("Reset")) needs_rebuild = true;

        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.7f, 0.85f, 1.0f, 1.0f), "Visualization");
        ImGui::Checkbox("Show anchors + springs", &params.show_anchors);
        ImGui::SameLine();
        ImGui::TextDisabled("(only drawn when k_a > 0)");
        ImGui::Checkbox("Show orientation arrows", &params.show_orientations);

        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.85f, 0.85f, 0.85f, 1.0f),
                           "Resolved physical parameters");
        ImGui::Text("  f0        = %.4g  (from delta)",     sim.cfg.f0);
        ImGui::Text("  tau_theta = %.4g  (= Pe/f0)",        sim.cfg.tau_theta);
        ImGui::Text("  gamma_a   = %.4g  (= R)",            sim.cfg.gamma_a);
        ImGui::Text("  k_a       = %.4g  (= R/(De tau_th))",sim.cfg.k_a);
        ImGui::Text("  kT        = %.4g  (= C)",            sim.cfg.kT);

        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.85f, 0.85f, 0.85f, 1.0f), "Status");
        ImGui::Text("step        = %lld", sim.step_count);
        ImGui::Text("time        = %.4g (in units of tau = gamma sigma^2 / eps)",
                    sim.sim_time);
        if (sim.em_integ) {
            ImGui::Text("Δt          = %.4g  (Euler-Maruyama)", sim.last_dt);
            ImGui::Text("cl rebuilds = %zu", sim.em_cl_rebuilds);
        } else if (sim.heun_integ) {
            ImGui::Text("Δt          = %.4g  (stochastic Heun)", sim.last_dt);
            ImGui::Text("cl rebuilds = %zu", sim.em_cl_rebuilds);
        } else if (sim.adaptive_integ) {
            ImGui::Text("macro Δt    = %.4g  (fixed, user-set)", sim.last_dt);
            ImGui::Text("inner dt    = %.4g  (Cash-Karp suggestion)",
                        sim.adaptive_integ->getDt());
            ImGui::Text("cl rebuilds = %zu",
                        sim.adaptive_integ->getCellListRebuilds());
        }
        ImGui::Text("U/N         = %.4g", cachedMeanEnergyPerParticle(sim));
        ImGui::Text("FPS         = %.1f", io.Framerate);

        ImGui::End();

        // Apply pending changes.
        if (needs_rebuild) {
            rebuild(sim, params);
            needs_rebuild      = false;
            needs_integ_update = false;
        } else if (needs_integ_update) {
            updateIntegrator(sim, params);
            needs_integ_update = false;
        }

        // Run the simulation.
        if (running) stepSim(sim, params.steps_per_frame);

        // ----- Visualization window ----------------------------------------
        ImGui::SetNextWindowPos (ImVec2(460, 10),  ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(800, 800), ImGuiCond_FirstUseEver);
        ImGui::Begin("Simulation");
        ImVec2 cv_pos   = ImGui::GetCursorScreenPos();
        ImVec2 cv_avail = ImGui::GetContentRegionAvail();
        ImVec2 cv_end(cv_pos.x + cv_avail.x, cv_pos.y + cv_avail.y);

        ImDrawList* dl = ImGui::GetWindowDrawList();
        drawSimulation(dl, cv_pos, cv_end, sim,
                       params.show_anchors, params.show_orientations);
        ImGui::InvisibleButton("canvas", cv_avail);
        ImGui::End();

        // ----- Render -------------------------------------------------------
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.10f, 0.10f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
