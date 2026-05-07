#include "AdaptiveMacroIntegrator.hpp"

#include "AdaptiveIntegrator.hpp"
#include "Box.hpp"
#include "CellList.hpp"
#include "ForceCalculator.hpp"
#include "RandomGenerator.hpp"
#include "System.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// PI controller exponents and clamps for an order-2 (Strang) splitting.
// Standard Gustafsson choices.
// ---------------------------------------------------------------------------
constexpr double kPIalpha = 1.0 / 3.0;
constexpr double kPIbeta  = 1.0 / 9.0;
constexpr double kSafety  = 0.9;
constexpr double kFmin    = 0.2;
constexpr double kFmax    = 5.0;
constexpr double kErrFloor = 1.0e-4;     // floor for err_prev so it doesn't
                                          // produce runaway growth scales

// ---------------------------------------------------------------------------
// Snapshot of the System state for save/restore around trial macro steps.
// We snapshot positions, anchors, and orientations — every component the
// macro step touches. Velocities (vx, vy) are diagnostic-only; we leave
// them alone.
// ---------------------------------------------------------------------------
struct Snapshot {
    std::vector<double> x, y, ax, ay, theta;

    void capture(const System& sys) {
        const std::size_t N = sys.getNumParticles();
        x.resize(N); y.resize(N); ax.resize(N); ay.resize(N); theta.resize(N);
        const double* xp  = sys.xData();
        const double* yp  = sys.yData();
        const double* axp = sys.axData();
        const double* ayp = sys.ayData();
        const double* tp  = sys.thetaData();
        for (std::size_t i = 0; i < N; ++i) {
            x[i] = xp[i]; y[i] = yp[i];
            ax[i] = axp[i]; ay[i] = ayp[i];
            theta[i] = tp[i];
        }
    }

    void restore(System& sys) const {
        const std::size_t N = sys.getNumParticles();
        double* xp  = sys.xData();
        double* yp  = sys.yData();
        double* axp = sys.axData();
        double* ayp = sys.ayData();
        double* tp  = sys.thetaData();
        for (std::size_t i = 0; i < N; ++i) {
            xp[i] = x[i]; yp[i] = y[i];
            axp[i] = ax[i]; ayp[i] = ay[i];
            tp[i] = theta[i];
        }
    }
};

// Brownian bridge: given a unit-Gaussian eta_c used over an interval of
// length tau, produce two unit Gaussians eta_a, eta_b for two consecutive
// intervals of tau/2 such that the SUM of the half-step increments matches
// the full-step increment in distribution. tilde is a fresh independent
// unit Gaussian.
inline void brownianBridge(double eta_c, double tilde,
                           double& eta_a, double& eta_b) {
    constexpr double kInvSqrt2 = 0.7071067811865475;   // 1/sqrt(2)
    eta_a = (eta_c + tilde) * kInvSqrt2;
    eta_b = (eta_c - tilde) * kInvSqrt2;
}

// Apply the noise operator N(tau): pure additive Brownian increments to r,
// a, theta with externally-supplied unit Gaussians. Sizes:
//   eta_T_x, eta_T_y : N each (translational position noise components)
//   eta_A_x, eta_A_y : N each (anchor noise components)
//   eta_R            : N      (rotational noise)
// kT = 0 / D_r = 0 / gamma_a = 0 short-circuit the corresponding pieces.
void applyNoiseOperator(System& sys, double tau,
                        double kT, double gamma, double gamma_a, double D_r,
                        const double* eta_T_x, const double* eta_T_y,
                        const double* eta_A_x, const double* eta_A_y,
                        const double* eta_R) {
    const std::size_t N = sys.getNumParticles();

    if (kT > 0.0 && gamma > 0.0 && tau > 0.0) {
        const double sigma = std::sqrt(2.0 * kT / gamma * tau);
        double* x = sys.xData();
        double* y = sys.yData();
        for (std::size_t i = 0; i < N; ++i) {
            x[i] += sigma * eta_T_x[i];
            y[i] += sigma * eta_T_y[i];
        }
    }

    if (kT > 0.0 && gamma_a > 0.0 && tau > 0.0) {
        const double sigma_a = std::sqrt(2.0 * kT / gamma_a * tau);
        double* ax = sys.axData();
        double* ay = sys.ayData();
        for (std::size_t i = 0; i < N; ++i) {
            ax[i] += sigma_a * eta_A_x[i];
            ay[i] += sigma_a * eta_A_y[i];
        }
    }

    if (D_r > 0.0 && tau > 0.0) {
        const double sigma_r = std::sqrt(2.0 * D_r * tau);
        double* th = sys.thetaData();
        for (std::size_t i = 0; i < N; ++i) {
            th[i] += sigma_r * eta_R[i];
        }
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Impl: holds the inner deterministic D-stepper, the stochastic parameters,
// and the macro-step PI controller state.
// ---------------------------------------------------------------------------
struct AdaptiveMacroIntegrator::Impl {
    // Inner D-step integrator (Cash-Karp 5(4) on the 4N deterministic flow).
    // Its own internal abs/rel tolerances default tight; deterministic
    // accuracy isn't what gates the macro step.
    AdaptiveIntegrator d_step;

    // Stochastic / kinetic parameters.
    double gamma;
    double gamma_a;
    double kT;
    double D_r;

    // Macro-step PI controller.
    double abs_tol;
    double rel_tol;
    double dt;
    double err_prev;
    double last_err;
    std::size_t rejections;

    // Cell list / Verlet neighbor map (owned). Rebuilt at the start of every
    // step() when needsRebuild() fires; FIXED for the duration of the macro
    // step (coarse + both fine sub-steps). Setup is lazy: we re-call
    // cl_.setup() whenever (N, Lx, Ly, r_cut) change, with last_* tracking
    // those values so we don't redo it unnecessarily. cl_brute is true when
    // the geometry forces brute force (small box or extremely sparse system);
    // in that case we never call cl.setup() and the inner stepper sees a null
    // cell-list pointer.
    CellList    cl;
    double      r_skin;
    std::size_t last_N;
    double      last_Lx, last_Ly, last_rcut;
    bool        cl_initialized;
    bool        cl_brute;

    // Buffers (reused across calls to avoid reallocation per macro step).
    std::vector<double> eta_T_x_c, eta_T_y_c;
    std::vector<double> eta_A_x_c, eta_A_y_c;
    std::vector<double> eta_R_c;
    std::vector<double> eta_T_x_c2, eta_T_y_c2;
    std::vector<double> eta_A_x_c2, eta_A_y_c2;
    std::vector<double> eta_R_c2;

    std::vector<double> tilde_T_x_1, tilde_T_y_1;
    std::vector<double> tilde_A_x_1, tilde_A_y_1;
    std::vector<double> tilde_R_1;
    std::vector<double> tilde_T_x_2, tilde_T_y_2;
    std::vector<double> tilde_A_x_2, tilde_A_y_2;
    std::vector<double> tilde_R_2;

    std::vector<double> eta_T_x_a, eta_T_y_a, eta_T_x_b, eta_T_y_b;
    std::vector<double> eta_A_x_a, eta_A_y_a, eta_A_x_b, eta_A_y_b;
    std::vector<double> eta_R_a,  eta_R_b;
    std::vector<double> eta_T_x_a2, eta_T_y_a2, eta_T_x_b2, eta_T_y_b2;
    std::vector<double> eta_A_x_a2, eta_A_y_a2, eta_A_x_b2, eta_A_y_b2;
    std::vector<double> eta_R_a2,  eta_R_b2;

    Snapshot snap_init;
    Snapshot snap_coarse;

    Impl(double g, double a, double r, double d)
        : d_step(g, /*abs_tol*/ 1.0e-7, /*rel_tol*/ 1.0e-7, d),
          gamma(g), gamma_a(0.0), kT(0.0), D_r(0.0),
          abs_tol(a), rel_tol(r), dt(d),
          err_prev(1.0), last_err(0.0), rejections(0),
          cl(),
          r_skin(0.5),
          last_N(0), last_Lx(0.0), last_Ly(0.0), last_rcut(0.0),
          cl_initialized(false),
          cl_brute(false)
    {}

    // Ensure the cell list is configured for the current System / Box / cutoff.
    // (Re)calls cl.setup(...) only when a tracked geometry parameter changes.
    // Returns true if the list is in cell-list mode, false if the cell list
    // is degenerate (box too small for a 3x3 stencil, or so sparse that the
    // grid would burn far more memory than brute force) and the caller
    // should null the inner d_step's cell pointer.
    bool ensureCellListSetup(std::size_t N, const Box& box, double r_cut) {
        const double Lx = box.getLx();
        const double Ly = box.getLy();
        const bool need_setup = !cl_initialized
                                || N        != last_N
                                || Lx       != last_Lx
                                || Ly       != last_Ly
                                || r_cut    != last_rcut;
        if (need_setup) {
            // Sparse-system guard: in a hugely-oversized box (think synthetic
            // tests with N=1 and L=1e6) the natural nx*ny cell count would
            // run into the billions — just allocating cell_start_/cell_end_
            // alone would OOM. The cell list is also algorithmically pointless
            // there (most cells empty, expected neighbor count ~ 0). Skip
            // setup() entirely in that case and tell the caller to fall back.
            // Likewise drop to brute force when the natural grid is < 3x3,
            // which is exactly the case CellList's setup() handles internally
            // but we'd rather not even allocate for.
            const double r_verlet = r_cut + r_skin;
            const double approx_cells = (Lx / r_verlet) * (Ly / r_verlet);
            const double cell_cap = static_cast<double>(64 * std::max<std::size_t>(N, 1));
            const bool too_sparse = approx_cells > cell_cap;
            const bool too_small  = (Lx / r_verlet < 3.0) || (Ly / r_verlet < 3.0);

            cl_brute = too_sparse || too_small;
            if (!cl_brute) cl.setup(r_cut, r_skin, N, box);

            last_N         = N;
            last_Lx        = Lx;
            last_Ly        = Ly;
            last_rcut      = r_cut;
            cl_initialized = true;
        }
        return !cl_brute;
    }

    // Resize all noise + bridge buffers for current particle count.
    void resizeNoiseBuffers(std::size_t N) {
        const auto resize_vec = [&](std::vector<double>& v) { v.resize(N); };
        for (auto* p : { &eta_T_x_c,  &eta_T_y_c,
                         &eta_A_x_c,  &eta_A_y_c, &eta_R_c,
                         &eta_T_x_c2, &eta_T_y_c2,
                         &eta_A_x_c2, &eta_A_y_c2, &eta_R_c2,
                         &tilde_T_x_1, &tilde_T_y_1,
                         &tilde_A_x_1, &tilde_A_y_1, &tilde_R_1,
                         &tilde_T_x_2, &tilde_T_y_2,
                         &tilde_A_x_2, &tilde_A_y_2, &tilde_R_2,
                         &eta_T_x_a, &eta_T_y_a, &eta_T_x_b, &eta_T_y_b,
                         &eta_A_x_a, &eta_A_y_a, &eta_A_x_b, &eta_A_y_b,
                         &eta_R_a,  &eta_R_b,
                         &eta_T_x_a2, &eta_T_y_a2, &eta_T_x_b2, &eta_T_y_b2,
                         &eta_A_x_a2, &eta_A_y_a2, &eta_A_x_b2, &eta_A_y_b2,
                         &eta_R_a2,  &eta_R_b2 }) resize_vec(*p);
    }

    void drawCoarseNoise(RandomGenerator& rng, std::size_t N) {
        for (std::size_t i = 0; i < N; ++i) {
            eta_T_x_c[i]  = rng.gaussian();  eta_T_y_c[i]  = rng.gaussian();
            eta_A_x_c[i]  = rng.gaussian();  eta_A_y_c[i]  = rng.gaussian();
            eta_R_c[i]    = rng.gaussian();
            eta_T_x_c2[i] = rng.gaussian();  eta_T_y_c2[i] = rng.gaussian();
            eta_A_x_c2[i] = rng.gaussian();  eta_A_y_c2[i] = rng.gaussian();
            eta_R_c2[i]   = rng.gaussian();
        }
    }

    // Draw bridge auxiliaries and split each coarse noise into a/b halves.
    void drawAndBridge(RandomGenerator& rng, std::size_t N) {
        for (std::size_t i = 0; i < N; ++i) {
            tilde_T_x_1[i] = rng.gaussian(); tilde_T_y_1[i] = rng.gaussian();
            tilde_A_x_1[i] = rng.gaussian(); tilde_A_y_1[i] = rng.gaussian();
            tilde_R_1[i]   = rng.gaussian();
            tilde_T_x_2[i] = rng.gaussian(); tilde_T_y_2[i] = rng.gaussian();
            tilde_A_x_2[i] = rng.gaussian(); tilde_A_y_2[i] = rng.gaussian();
            tilde_R_2[i]   = rng.gaussian();

            brownianBridge(eta_T_x_c[i],  tilde_T_x_1[i], eta_T_x_a[i],  eta_T_x_b[i]);
            brownianBridge(eta_T_y_c[i],  tilde_T_y_1[i], eta_T_y_a[i],  eta_T_y_b[i]);
            brownianBridge(eta_A_x_c[i],  tilde_A_x_1[i], eta_A_x_a[i],  eta_A_x_b[i]);
            brownianBridge(eta_A_y_c[i],  tilde_A_y_1[i], eta_A_y_a[i],  eta_A_y_b[i]);
            brownianBridge(eta_R_c[i],    tilde_R_1[i],   eta_R_a[i],    eta_R_b[i]);

            brownianBridge(eta_T_x_c2[i], tilde_T_x_2[i], eta_T_x_a2[i], eta_T_x_b2[i]);
            brownianBridge(eta_T_y_c2[i], tilde_T_y_2[i], eta_T_y_a2[i], eta_T_y_b2[i]);
            brownianBridge(eta_A_x_c2[i], tilde_A_x_2[i], eta_A_x_a2[i], eta_A_x_b2[i]);
            brownianBridge(eta_A_y_c2[i], tilde_A_y_2[i], eta_A_y_a2[i], eta_A_y_b2[i]);
            brownianBridge(eta_R_c2[i],   tilde_R_2[i],   eta_R_a2[i],   eta_R_b2[i]);
        }
    }

    // One coarse Strang step at trial dt: N(dt/2) - D(dt) - N(dt/2),
    // using the *_c and *_c2 noise vectors. Mutates sys.
    void runCoarseStrang(System& sys, const Box& box, const ForceCalculator& fc,
                         double trial_dt) {
        applyNoiseOperator(sys, 0.5 * trial_dt, kT, gamma, gamma_a, D_r,
                           eta_T_x_c.data(),  eta_T_y_c.data(),
                           eta_A_x_c.data(),  eta_A_y_c.data(),
                           eta_R_c.data());
        d_step.advance(sys, box, fc, trial_dt);
        applyNoiseOperator(sys, 0.5 * trial_dt, kT, gamma, gamma_a, D_r,
                           eta_T_x_c2.data(), eta_T_y_c2.data(),
                           eta_A_x_c2.data(), eta_A_y_c2.data(),
                           eta_R_c2.data());
    }

    // Two fine Strang steps each at trial_dt/2, using the bridged a/b noise
    // vectors. The first half uses (a, a2), the second half uses (b, b2).
    void runFineStrangs(System& sys, const Box& box, const ForceCalculator& fc,
                        double trial_dt) {
        const double half = 0.5 * trial_dt;
        // ---- First half-Strang at half ------------------------------------
        applyNoiseOperator(sys, 0.5 * half, kT, gamma, gamma_a, D_r,
                           eta_T_x_a.data(),  eta_T_y_a.data(),
                           eta_A_x_a.data(),  eta_A_y_a.data(),
                           eta_R_a.data());
        d_step.advance(sys, box, fc, half);
        applyNoiseOperator(sys, 0.5 * half, kT, gamma, gamma_a, D_r,
                           eta_T_x_a2.data(), eta_T_y_a2.data(),
                           eta_A_x_a2.data(), eta_A_y_a2.data(),
                           eta_R_a2.data());

        // ---- Second half-Strang at half -----------------------------------
        applyNoiseOperator(sys, 0.5 * half, kT, gamma, gamma_a, D_r,
                           eta_T_x_b.data(),  eta_T_y_b.data(),
                           eta_A_x_b.data(),  eta_A_y_b.data(),
                           eta_R_b.data());
        d_step.advance(sys, box, fc, half);
        applyNoiseOperator(sys, 0.5 * half, kT, gamma, gamma_a, D_r,
                           eta_T_x_b2.data(), eta_T_y_b2.data(),
                           eta_A_x_b2.data(), eta_A_y_b2.data(),
                           eta_R_b2.data());
    }

    // Weighted-RMS error norm between the (current System state) "fine" and
    // the snap_coarse "coarse". Both have positions wrapped into [0, L), so
    // we compute differences via Box::minimumImage to handle PBC correctly.
    double errorNorm(const System& sys, const Box& box) const {
        const std::size_t N = sys.getNumParticles();
        double sum_sq = 0.0;
        std::size_t count = 0;

        for (std::size_t i = 0; i < N; ++i) {
            // Positions: PBC-aware difference.
            double dx = sys.getX(i) - snap_coarse.x[i];
            double dy = sys.getY(i) - snap_coarse.y[i];
            box.minimumImage(dx, dy);
            const double sx = abs_tol + rel_tol * std::max(std::abs(sys.getX(i)),
                                                           std::abs(snap_coarse.x[i]));
            const double sy = abs_tol + rel_tol * std::max(std::abs(sys.getY(i)),
                                                           std::abs(snap_coarse.y[i]));
            sum_sq += (dx / sx) * (dx / sx);
            sum_sq += (dy / sy) * (dy / sy);
            count += 2;

            // Anchors: same treatment.
            double dax = sys.getAx(i) - snap_coarse.ax[i];
            double day = sys.getAy(i) - snap_coarse.ay[i];
            box.minimumImage(dax, day);
            const double sax = abs_tol + rel_tol * std::max(std::abs(sys.getAx(i)),
                                                            std::abs(snap_coarse.ax[i]));
            const double say = abs_tol + rel_tol * std::max(std::abs(sys.getAy(i)),
                                                            std::abs(snap_coarse.ay[i]));
            sum_sq += (dax / sax) * (dax / sax);
            sum_sq += (day / say) * (day / say);
            count += 2;

            // Angles: O(1) scale, no PBC weighting issue.
            const double dth = sys.getTheta(i) - snap_coarse.theta[i];
            const double sth = abs_tol + rel_tol;
            sum_sq += (dth / sth) * (dth / sth);
            count += 1;
        }

        return (count > 0) ? std::sqrt(sum_sq / static_cast<double>(count))
                           : 0.0;
    }
};

// ---------------------------------------------------------------------------
// Construction / destruction.
// ---------------------------------------------------------------------------
AdaptiveMacroIntegrator::AdaptiveMacroIntegrator(double gamma,
                                                 double abs_tol, double rel_tol,
                                                 double dt_init)
    : impl_(std::make_unique<Impl>(gamma, abs_tol, rel_tol, dt_init)) {}

AdaptiveMacroIntegrator::~AdaptiveMacroIntegrator() = default;
AdaptiveMacroIntegrator::AdaptiveMacroIntegrator(AdaptiveMacroIntegrator&&) noexcept = default;
AdaptiveMacroIntegrator& AdaptiveMacroIntegrator::operator=(AdaptiveMacroIntegrator&&) noexcept = default;

// ---------------------------------------------------------------------------
// step(): one accepted adaptive macro step.
// ---------------------------------------------------------------------------
double AdaptiveMacroIntegrator::step(System& sys, const Box& box,
                                     const ForceCalculator& fc,
                                     RandomGenerator& rng) {
    const std::size_t N = sys.getNumParticles();
    impl_->resizeNoiseBuffers(N);

    // ---- Cell list bookkeeping (BEFORE the Strang trials) -----------------
    // Set up / refresh the cell-list geometry if needed, then check the
    // r_skin/2 drift trigger. A rebuild happens HERE — outside the macro
    // step's coarse and fine trials — so both trials evaluate forces against
    // the SAME fixed neighbor connectivity. If the cell list is in
    // brute-force mode (box too small for a 3x3 stencil), null the inner
    // stepper's pointer so it falls back to the O(N^2) overload.
    const bool cell_list_active =
        impl_->ensureCellListSetup(N, box, fc.getCutoff());
    if (cell_list_active) {
        if (impl_->cl.needsRebuild(sys, box)) impl_->cl.rebuild(sys, box);
        impl_->d_step.setCellList(&impl_->cl);
    } else {
        impl_->d_step.setCellList(nullptr);
    }

    // Snapshot the starting state so we can restore on rejection / before
    // the fine trial.
    impl_->snap_init.capture(sys);

    while (true) {
        const double trial_dt = impl_->dt;

        // ---- Coarse trial ---------------------------------------------------
        impl_->drawCoarseNoise(rng, N);
        impl_->runCoarseStrang(sys, box, fc, trial_dt);
        impl_->snap_coarse.capture(sys);

        // Restore initial state for the fine trial.
        impl_->snap_init.restore(sys);

        // ---- Fine trial: bridge the coarse noises and run two halves -------
        impl_->drawAndBridge(rng, N);
        impl_->runFineStrangs(sys, box, fc, trial_dt);

        // ---- Error norm ----------------------------------------------------
        const double err = impl_->errorNorm(sys, box);

        // ---- PI controller -------------------------------------------------
        if (err <= 1.0) {
            // ACCEPT. The fine state is what's currently in sys.
            const double scale = (err > 1.0e-12)
                ? kSafety * std::pow(err, -kPIalpha)
                          * std::pow(impl_->err_prev, kPIbeta)
                : kFmax;
            const double clamped = std::clamp(scale, kFmin, kFmax);

            const double dt_taken = trial_dt;
            impl_->dt        = trial_dt * clamped;
            impl_->err_prev  = std::max(err, kErrFloor);
            impl_->last_err  = err;
            return dt_taken;
        } else {
            // REJECT. Restore initial state, shrink dt, retry.
            impl_->snap_init.restore(sys);
            ++impl_->rejections;
            const double scale = kSafety * std::pow(err, -kPIalpha);
            const double clamped = std::clamp(scale, kFmin, 1.0);   // never grow on reject
            impl_->dt = trial_dt * clamped;
        }
    }
}

// ---------------------------------------------------------------------------
// Getters / setters.
// ---------------------------------------------------------------------------
double AdaptiveMacroIntegrator::getKBT() const                 { return impl_->kT; }
void   AdaptiveMacroIntegrator::setKBT(double kT)              { impl_->kT = kT; }

double AdaptiveMacroIntegrator::getRotationalDiffusion() const { return impl_->D_r; }
void   AdaptiveMacroIntegrator::setRotationalDiffusion(double Dr) { impl_->D_r = Dr; }

double AdaptiveMacroIntegrator::getFriction() const            { return impl_->gamma; }
void   AdaptiveMacroIntegrator::setFriction(double g) {
    impl_->gamma = g;
    impl_->d_step.setFriction(g);
}

double AdaptiveMacroIntegrator::getAnchorFriction() const      { return impl_->gamma_a; }
void   AdaptiveMacroIntegrator::setAnchorFriction(double ga) {
    impl_->gamma_a = ga;
    impl_->d_step.setAnchorFriction(ga);
}

double AdaptiveMacroIntegrator::getActiveForce() const         { return impl_->d_step.getActiveForce(); }
void   AdaptiveMacroIntegrator::setActiveForce(double f0)      { impl_->d_step.setActiveForce(f0); }

double AdaptiveMacroIntegrator::getSpringStiffness() const     { return impl_->d_step.getSpringStiffness(); }
void   AdaptiveMacroIntegrator::setSpringStiffness(double ka)  { impl_->d_step.setSpringStiffness(ka); }

double AdaptiveMacroIntegrator::getAbsTol() const              { return impl_->abs_tol; }
void   AdaptiveMacroIntegrator::setAbsTol(double a)            { impl_->abs_tol = a; }

double AdaptiveMacroIntegrator::getRelTol() const              { return impl_->rel_tol; }
void   AdaptiveMacroIntegrator::setRelTol(double r)            { impl_->rel_tol = r; }

double AdaptiveMacroIntegrator::getDt() const                  { return impl_->dt; }
void   AdaptiveMacroIntegrator::setDt(double d)                { impl_->dt = d; }

double AdaptiveMacroIntegrator::getLastErr() const             { return impl_->last_err; }
std::size_t AdaptiveMacroIntegrator::getRejections() const     { return impl_->rejections; }

double AdaptiveMacroIntegrator::getCellListSkin() const        { return impl_->r_skin; }
void   AdaptiveMacroIntegrator::setCellListSkin(double r_skin) {
    impl_->r_skin = r_skin;
    // Force a re-setup at the next step() so the new skin takes effect.
    impl_->cl_initialized = false;
}
std::size_t AdaptiveMacroIntegrator::getCellListRebuilds() const {
    return impl_->cl.getNumRebuilds();
}
bool AdaptiveMacroIntegrator::cellListIsBruteForce() const {
    return impl_->cl_initialized && impl_->cl_brute;
}
