#include "AdaptiveIntegrator.hpp"

#include "Box.hpp"
#include "CellList.hpp"
#include "ForceCalculator.hpp"
#include "RandomGenerator.hpp"
#include "System.hpp"

// boost::numeric::odeint is header-only; all Boost includes stay confined to
// this .cpp (via pImpl) so callers never need Boost on their include path.
#include <boost/numeric/odeint.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace odeint = boost::numeric::odeint;

// Flat SoA state vector of length 4N:
//   [0   .. N)    rx
//   [N   .. 2N)   ry
//   [2N  .. 3N)   ax
//   [3N  .. 4N)   ay
using State = std::vector<double>;

namespace {

// ---------------------------------------------------------------------------
// DriftRHS — full deterministic flow with theta frozen.
//
//   dr/dt = mu * [F_WCA(r) + f0 * e_theta - k_a * (r - a)]
//   da/dt = mu_a * k_a * (r - a)        (mu_a = 1/gamma_a; 0 when gamma_a = 0)
// ---------------------------------------------------------------------------
struct DriftRHS {
    System&                sys;
    const Box&             box;
    const ForceCalculator& fc;
    const CellList*        cl;          // optional, non-owning; nullptr -> brute force
    double                 mu;          // 1 / gamma
    double                 mu_a;        // 1 / gamma_a, 0 if anchor frozen
    double                 f0;
    double                 k_a;

    void operator()(const State& q, State& dqdt, double /*t*/) const {
        const std::size_t N = sys.getNumParticles();

        double* __restrict__ x  = sys.xData();
        double* __restrict__ y  = sys.yData();
        double* __restrict__ ax = sys.axData();
        double* __restrict__ ay = sys.ayData();
        for (std::size_t i = 0; i < N; ++i) {
            x[i]  = q[i];
            y[i]  = q[N + i];
            ax[i] = q[2 * N + i];
            ay[i] = q[3 * N + i];
        }

        if (cl) fc.compute(sys, box, *cl);
        else    fc.compute(sys, box);

        const double* __restrict__ fx    = sys.fxData();
        const double* __restrict__ fy    = sys.fyData();
        const double* __restrict__ theta = sys.thetaData();

        const bool spring_on = (k_a > 0.0);
        const bool anchor_on = (mu_a > 0.0) && spring_on;

        for (std::size_t i = 0; i < N; ++i) {
            const double fx_act = f0 * std::cos(theta[i]);
            const double fy_act = f0 * std::sin(theta[i]);

            double sep_x = 0.0, sep_y = 0.0;
            double fx_spring = 0.0, fy_spring = 0.0;
            if (spring_on) {
                sep_x = x[i]  - ax[i];
                sep_y = y[i]  - ay[i];
                box.minimumImage(sep_x, sep_y);
                fx_spring = -k_a * sep_x;
                fy_spring = -k_a * sep_y;
            }

            dqdt[i]         = mu * (fx[i] + fx_act + fx_spring);
            dqdt[N + i]     = mu * (fy[i] + fy_act + fy_spring);

            if (anchor_on) {
                dqdt[2 * N + i] = mu_a * k_a * sep_x;
                dqdt[3 * N + i] = mu_a * k_a * sep_y;
            } else {
                dqdt[2 * N + i] = 0.0;
                dqdt[3 * N + i] = 0.0;
            }
        }
    }
};

// Apply the noise operator N(tau): pure additive Brownian increments to r,
// a, theta. Each call draws fresh standard Gaussians from rng. kT = 0 / D_r =
// 0 / gamma_a = 0 short-circuit the corresponding pieces.
void applyNoiseOperator(System& sys, double tau,
                        double kT, double gamma, double gamma_a, double D_r,
                        RandomGenerator& rng) {
    if (tau <= 0.0) return;
    const std::size_t N = sys.getNumParticles();

    if (kT > 0.0 && gamma > 0.0) {
        const double sigma = std::sqrt(2.0 * kT / gamma * tau);
        double* x = sys.xData();
        double* y = sys.yData();
        for (std::size_t i = 0; i < N; ++i) {
            x[i] += sigma * rng.gaussian();
            y[i] += sigma * rng.gaussian();
        }
    }

    if (kT > 0.0 && gamma_a > 0.0) {
        const double sigma_a = std::sqrt(2.0 * kT / gamma_a * tau);
        double* ax = sys.axData();
        double* ay = sys.ayData();
        for (std::size_t i = 0; i < N; ++i) {
            ax[i] += sigma_a * rng.gaussian();
            ay[i] += sigma_a * rng.gaussian();
        }
    }

    if (D_r > 0.0) {
        const double sigma_r = std::sqrt(2.0 * D_r * tau);
        double* th = sys.thetaData();
        for (std::size_t i = 0; i < N; ++i) {
            th[i] += sigma_r * rng.gaussian();
        }
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Impl: holds the cached controlled stepper plus all integrator state.
// ---------------------------------------------------------------------------
struct AdaptiveIntegrator::Impl {
    using ErrorStepper = odeint::runge_kutta_cash_karp54<State>;
    using Stepper      = odeint::controlled_runge_kutta<ErrorStepper>;

    // Tolerances + inner adaptive sub-step.
    double  gamma;
    double  abs_tol;
    double  rel_tol;
    double  dt;            // inner Cash-Karp adaptive sub-step suggestion

    // Deterministic flow extras.
    double  f0;
    double  k_a;
    double  gamma_a;

    // Stochastic-flow params (Strang path).
    double  kT;
    double  D_r;

    // Macro Δt for the Strang path (user-set, FIXED).
    double  macro_dt;

    // Cell list. cl_external is the pointer the inner RHS dereferences. When
    // setCellList() is called by an external caller (e.g. AdaptiveMacroIntegrator)
    // we route through that pointer; otherwise the Strang path manages the
    // owned cl_owned and points cl_external at it.
    CellList        cl_owned;
    const CellList* cl_external;            // what DriftRHS sees
    bool            cl_external_user_set;   // true if setCellList was called by
                                            // a non-Strang caller — in that case
                                            // Strang step() leaves cl_external alone

    double      r_skin;
    std::size_t last_N;
    double      last_Lx, last_Ly, last_rcut;
    bool        cl_initialized;
    bool        cl_brute;

    Stepper stepper;
    State   q;             // resized lazily

    Impl(double g, double a, double r, double d,
         double f, double ka, double ga)
        : gamma(g), abs_tol(a), rel_tol(r), dt(d),
          f0(f), k_a(ka), gamma_a(ga),
          kT(0.0), D_r(0.0),
          macro_dt(d),
          cl_owned(),
          cl_external(nullptr),
          cl_external_user_set(false),
          r_skin(0.5),
          last_N(0), last_Lx(0.0), last_Ly(0.0), last_rcut(0.0),
          cl_initialized(false),
          cl_brute(false),
          stepper(odeint::make_controlled<ErrorStepper>(a, r))
    {}

    void rebuildStepper() {
        stepper = odeint::make_controlled<ErrorStepper>(abs_tol, rel_tol);
    }

    // Refresh cl_owned for the current System / Box / cutoff if any of them
    // changed. Returns true if the cell list is in cell-list mode (false =>
    // brute force, caller should null cl_external).
    bool ensureOwnedCellListSetup(std::size_t N, const Box& box, double r_cut) {
        const double Lx = box.getLx();
        const double Ly = box.getLy();
        const bool need_setup = !cl_initialized
                                || N      != last_N
                                || Lx     != last_Lx
                                || Ly     != last_Ly
                                || r_cut  != last_rcut;
        if (need_setup) {
            const double r_verlet = r_cut + r_skin;
            const double approx_cells = (Lx / r_verlet) * (Ly / r_verlet);
            const double cell_cap = static_cast<double>(64 * std::max<std::size_t>(N, 1));
            const bool too_sparse = approx_cells > cell_cap;
            const bool too_small  = (Lx / r_verlet < 3.0) || (Ly / r_verlet < 3.0);

            cl_brute = too_sparse || too_small;
            if (!cl_brute) cl_owned.setup(r_cut, r_skin, N, box);

            last_N         = N;
            last_Lx        = Lx;
            last_Ly        = Ly;
            last_rcut      = r_cut;
            cl_initialized = true;
        }
        return !cl_brute;
    }

    // Pack (sys.{x,y,ax,ay}) -> q. Used at the start of every inner step().
    void packState(const System& sys) {
        const std::size_t N = sys.getNumParticles();
        q.resize(4 * N);
        const double* xp  = sys.xData();
        const double* yp  = sys.yData();
        const double* axp = sys.axData();
        const double* ayp = sys.ayData();
        for (std::size_t i = 0; i < N; ++i) {
            q[i]         = xp[i];
            q[N + i]     = yp[i];
            q[2 * N + i] = axp[i];
            q[3 * N + i] = ayp[i];
        }
    }

    // q -> (sys.{x,y,ax,ay}). Stored positions are NOT wrapped — they diffuse
    // freely; pair forces and cell binning handle PBC on their own (see
    // Box.hpp). Kept named for diff continuity with earlier "AndWrap" callers.
    void unpackStateAndWrap(System& sys, const Box& /*box*/) {
        const std::size_t N = sys.getNumParticles();
        double* x  = sys.xData();
        double* y  = sys.yData();
        double* ax = sys.axData();
        double* ay = sys.ayData();
        for (std::size_t i = 0; i < N; ++i) {
            x[i]  = q[i];
            y[i]  = q[N + i];
            ax[i] = q[2 * N + i];
            ay[i] = q[3 * N + i];
        }
    }
};

// ---------------------------------------------------------------------------
// Construction / destruction.
// ---------------------------------------------------------------------------

AdaptiveIntegrator::AdaptiveIntegrator(double gamma,
                                       double abs_tol, double rel_tol,
                                       double dt_init,
                                       double f0, double k_a, double gamma_a)
    : impl_(std::make_unique<Impl>(gamma, abs_tol, rel_tol, dt_init,
                                   f0, k_a, gamma_a)) {}

AdaptiveIntegrator::~AdaptiveIntegrator() = default;
AdaptiveIntegrator::AdaptiveIntegrator(AdaptiveIntegrator&&) noexcept = default;
AdaptiveIntegrator& AdaptiveIntegrator::operator=(AdaptiveIntegrator&&) noexcept = default;

// ---------------------------------------------------------------------------
// Path (1): deterministic adaptive sub-step.
// ---------------------------------------------------------------------------

double AdaptiveIntegrator::step(System& sys, const Box& box,
                                const ForceCalculator& fc) {
    impl_->packState(sys);

    const double mu_a = (impl_->gamma_a > 0.0) ? (1.0 / impl_->gamma_a) : 0.0;
    DriftRHS rhs{sys, box, fc, impl_->cl_external,
                 1.0 / impl_->gamma, mu_a, impl_->f0, impl_->k_a};

    double t = 0.0;
    while (impl_->stepper.try_step(rhs, impl_->q, t, impl_->dt) != odeint::success) {
        // dt was reduced by the controller; retry.
    }
    const double dt_taken = t;

    impl_->unpackStateAndWrap(sys, box);
    return dt_taken;
}

std::size_t AdaptiveIntegrator::advance(System& sys, const Box& box,
                                        const ForceCalculator& fc,
                                        double T) {
    if (T <= 0.0) return 0;

    constexpr double rel_eps = 1.0e-10;
    const double dt_saved = impl_->dt;

    double      t_done  = 0.0;
    std::size_t n_steps = 0;

    while (t_done < T * (1.0 - rel_eps)) {
        const double remaining = T - t_done;
        if (impl_->dt > remaining) impl_->dt = remaining;

        const double dt_taken = step(sys, box, fc);
        t_done += dt_taken;
        ++n_steps;
    }

    if (impl_->dt < dt_saved) impl_->dt = dt_saved;

    return n_steps;
}

// ---------------------------------------------------------------------------
// Path (2): Strang(N-D-N) macro step at user-fixed Δt.
// ---------------------------------------------------------------------------

double AdaptiveIntegrator::step(System& sys, const Box& box,
                                const ForceCalculator& fc,
                                RandomGenerator& rng) {
    const std::size_t N = sys.getNumParticles();
    const double dt     = impl_->macro_dt;

    // ---- Cell list bookkeeping (BEFORE the Strang composition) ------------
    // When the caller has installed an external cell list, we leave it alone
    // — they're managing rebuilds themselves. Otherwise we own the list and
    // refresh it according to the standard r_skin/2 drift trigger.
    if (!impl_->cl_external_user_set) {
        const bool active = impl_->ensureOwnedCellListSetup(N, box, fc.getCutoff());
        if (active) {
            if (impl_->cl_owned.needsRebuild(sys, box))
                impl_->cl_owned.rebuild(sys, box);
            impl_->cl_external = &impl_->cl_owned;
        } else {
            impl_->cl_external = nullptr;
        }
    }

    // ---- N(dt/2) ----------------------------------------------------------
    applyNoiseOperator(sys, 0.5 * dt,
                       impl_->kT, impl_->gamma, impl_->gamma_a, impl_->D_r,
                       rng);

    // ---- D(dt) — adaptive Cash-Karp 5(4) inner sub-stepping ---------------
    advance(sys, box, fc, dt);

    // ---- N(dt/2) ----------------------------------------------------------
    applyNoiseOperator(sys, 0.5 * dt,
                       impl_->kT, impl_->gamma, impl_->gamma_a, impl_->D_r,
                       rng);

    return dt;
}

std::size_t AdaptiveIntegrator::advance(System& sys, const Box& box,
                                        const ForceCalculator& fc,
                                        RandomGenerator& rng,
                                        double T) {
    if (T <= 0.0) return 0;

    constexpr double rel_eps = 1.0e-10;
    const double dt_saved = impl_->macro_dt;

    double      t_done  = 0.0;
    std::size_t n_steps = 0;

    while (t_done < T * (1.0 - rel_eps)) {
        const double remaining = T - t_done;
        if (impl_->macro_dt > remaining) impl_->macro_dt = remaining;

        const double dt_taken = step(sys, box, fc, rng);
        t_done += dt_taken;
        ++n_steps;
    }

    impl_->macro_dt = dt_saved;
    return n_steps;
}

// ---------------------------------------------------------------------------
// Getters / setters.
// ---------------------------------------------------------------------------

double AdaptiveIntegrator::getFriction()        const { return impl_->gamma; }
double AdaptiveIntegrator::getMobility()        const { return 1.0 / impl_->gamma; }
double AdaptiveIntegrator::getAbsTol()          const { return impl_->abs_tol; }
double AdaptiveIntegrator::getRelTol()          const { return impl_->rel_tol; }
double AdaptiveIntegrator::getDt()              const { return impl_->dt; }
double AdaptiveIntegrator::getActiveForce()     const { return impl_->f0; }
double AdaptiveIntegrator::getSpringStiffness() const { return impl_->k_a; }
double AdaptiveIntegrator::getAnchorFriction()  const { return impl_->gamma_a; }

void AdaptiveIntegrator::setFriction(double g) { impl_->gamma = g; }

void AdaptiveIntegrator::setAbsTol(double a) {
    impl_->abs_tol = a;
    impl_->rebuildStepper();
}
void AdaptiveIntegrator::setRelTol(double r) {
    impl_->rel_tol = r;
    impl_->rebuildStepper();
}

void AdaptiveIntegrator::setDt(double d)             { impl_->dt      = d;  }
void AdaptiveIntegrator::setActiveForce(double f)    { impl_->f0      = f;  }
void AdaptiveIntegrator::setSpringStiffness(double k){ impl_->k_a     = k;  }
void AdaptiveIntegrator::setAnchorFriction(double g) { impl_->gamma_a = g;  }

double AdaptiveIntegrator::getKBT()                 const { return impl_->kT;  }
void   AdaptiveIntegrator::setKBT(double kT)              { impl_->kT  = kT;  }
double AdaptiveIntegrator::getRotationalDiffusion() const { return impl_->D_r; }
void   AdaptiveIntegrator::setRotationalDiffusion(double Dr) { impl_->D_r = Dr; }

double AdaptiveIntegrator::getMacroDt() const         { return impl_->macro_dt; }
void   AdaptiveIntegrator::setMacroDt(double dt)      { impl_->macro_dt = dt;   }

void AdaptiveIntegrator::setCellList(const CellList* cl) {
    impl_->cl_external          = cl;
    impl_->cl_external_user_set = (cl != nullptr);
}
const CellList* AdaptiveIntegrator::getCellList() const  { return impl_->cl_external; }

double AdaptiveIntegrator::getCellListSkin() const       { return impl_->r_skin; }
void   AdaptiveIntegrator::setCellListSkin(double r_skin) {
    impl_->r_skin         = r_skin;
    impl_->cl_initialized = false;        // force re-setup at the next step()
}
std::size_t AdaptiveIntegrator::getCellListRebuilds() const {
    return impl_->cl_owned.getNumRebuilds();
}
bool AdaptiveIntegrator::cellListIsBruteForce() const {
    return impl_->cl_initialized && impl_->cl_brute;
}
