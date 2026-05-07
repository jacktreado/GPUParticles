#include "Integrator.hpp"

#include "Box.hpp"
#include "ForceCalculator.hpp"
#include "RandomGenerator.hpp"
#include "System.hpp"

#include <cmath>
#include <cstddef>

Integrator::Integrator(double gamma, double dt) : gamma_(gamma), dt_(dt) {
    recomputeCachedConstants();
}

void Integrator::recomputeCachedConstants() {
    // Particle: mu = 1/gamma; noise amplitude = sqrt(2*kT/gamma*dt) (Einstein).
    mobility_dt_         = (1.0 / gamma_) * dt_;
    noise_amplitude_     = std::sqrt(2.0 * (kT_ / gamma_) * dt_);
    rot_noise_amplitude_ = std::sqrt(2.0 * D_r_ * dt_);

    // Anchor: gamma_a > 0 enables anchor dynamics.  When gamma_a = 0 we set
    // both to zero so the anchor is frozen with no noise (and the integrator
    // skips the anchor noise draws).
    if (gamma_a_ > 0.0) {
        anchor_mobility_dt_     = (1.0 / gamma_a_) * dt_;
        anchor_noise_amplitude_ = std::sqrt(2.0 * (kT_ / gamma_a_) * dt_);
    } else {
        anchor_mobility_dt_     = 0.0;
        anchor_noise_amplitude_ = 0.0;
    }
}

void Integrator::step(System& sys, const Box& box, RandomGenerator& rng) const {
    const std::size_t N = sys.getNumParticles();
    double* __restrict__ x     = sys.xData();
    double* __restrict__ y     = sys.yData();
    double* __restrict__ vx    = sys.vxData();
    double* __restrict__ vy    = sys.vyData();
    double* __restrict__ theta = sys.thetaData();
    double* __restrict__ ax    = sys.axData();
    double* __restrict__ ay    = sys.ayData();
    const double* __restrict__ fx = sys.fxData();
    const double* __restrict__ fy = sys.fyData();

    const double inv_dt     = 1.0 / dt_;
    const double max_drift2 = max_drift_ * max_drift_;
    const bool   anchor_on  = (gamma_a_ > 0.0);

    // ---- Translational update (Euler-Maruyama) ------------------------------
    // The particle feels: WCA force + active force + spring force from its
    // anchor.  Spring contribution uses minimum-image so PBC is respected.
    // The drift (sum of all force contributions * mobility * dt) is optionally
    // capped to max_drift_ before noise is added.
    for (std::size_t i = 0; i < N; ++i) {
        const double xi_noise = rng.gaussian();
        const double yi_noise = rng.gaussian();

        const double fx_act = f0_ * std::cos(theta[i]);
        const double fy_act = f0_ * std::sin(theta[i]);

        // Spring force on particle: -k_a * (r - a) using minimum image.
        double fx_spring = 0.0;
        double fy_spring = 0.0;
        if (k_a_ > 0.0) {
            double sep_x = x[i] - ax[i];
            double sep_y = y[i] - ay[i];
            box.minimumImage(sep_x, sep_y);
            fx_spring = -k_a_ * sep_x;
            fy_spring = -k_a_ * sep_y;
        }

        double drift_x = mobility_dt_ * (fx[i] + fx_act + fx_spring);
        double drift_y = mobility_dt_ * (fy[i] + fy_act + fy_spring);

        if (max_drift_ > 0.0) {
            const double d2 = drift_x * drift_x + drift_y * drift_y;
            if (d2 > max_drift2) {
                const double scale = max_drift_ / std::sqrt(d2);
                drift_x *= scale;
                drift_y *= scale;
            }
        }

        const double dx = drift_x + noise_amplitude_ * xi_noise;
        const double dy = drift_y + noise_amplitude_ * yi_noise;

        x[i]  += dx;
        y[i]  += dy;
        vx[i]  = dx * inv_dt;
        vy[i]  = dy * inv_dt;
    }

    // ---- Rotational diffusion -----------------------------------------------
    if (rot_noise_amplitude_ > 0.0) {
        for (std::size_t i = 0; i < N; ++i) {
            theta[i] += rot_noise_amplitude_ * rng.gaussian();
        }
    }

    // ---- Anchor update ------------------------------------------------------
    // Spring force on the anchor is +k_a*(r-a) (Newton's third law).  The
    // anchor sees its own thermal bath with friction gamma_a, so noise
    // amplitude is sqrt(2*kT/gamma_a*dt).  Uses the *pre-step* particle
    // position consistent with Euler-Maruyama; we use post-step x[i] here as
    // a one-step-lagged approximation, which is acceptable for small dt.
    // (Switching to pre-step requires an extra buffer; keep simple for now.)
    if (anchor_on) {
        for (std::size_t i = 0; i < N; ++i) {
            const double xi_a = rng.gaussian();
            const double yi_a = rng.gaussian();

            double drift_ax = 0.0;
            double drift_ay = 0.0;
            if (k_a_ > 0.0) {
                double sep_x = x[i] - ax[i];
                double sep_y = y[i] - ay[i];
                box.minimumImage(sep_x, sep_y);
                drift_ax = anchor_mobility_dt_ * k_a_ * sep_x;
                drift_ay = anchor_mobility_dt_ * k_a_ * sep_y;
            }

            if (max_drift_ > 0.0) {
                const double d2 = drift_ax * drift_ax + drift_ay * drift_ay;
                if (d2 > max_drift2) {
                    const double scale = max_drift_ / std::sqrt(d2);
                    drift_ax *= scale;
                    drift_ay *= scale;
                }
            }

            ax[i] += drift_ax + anchor_noise_amplitude_ * xi_a;
            ay[i] += drift_ay + anchor_noise_amplitude_ * yi_a;
        }
    }

    // ---- Periodic boundary conditions ---------------------------------------
    // Wrap both particle and (when active) anchor positions into the primary
    // cell.  Spring forces use minimum-image, so this is always consistent.
    for (std::size_t i = 0; i < N; ++i) {
        double xi = x[i];
        double yi = y[i];
        box.wrap(xi, yi);
        x[i] = xi;
        y[i] = yi;
    }
    if (anchor_on) {
        for (std::size_t i = 0; i < N; ++i) {
            double xi = ax[i];
            double yi = ay[i];
            box.wrap(xi, yi);
            ax[i] = xi;
            ay[i] = yi;
        }
    }
}

void Integrator::stepWithForces(System& sys, const Box& box,
                                const ForceCalculator& fc,
                                RandomGenerator& rng) const {
    fc.compute(sys, box);
    step(sys, box, rng);
}

void Integrator::kickStep(System& sys, const Box& box, RandomGenerator& rng,
                          double dt) const {
    if (dt <= 0.0) return;

    const std::size_t N = sys.getNumParticles();
    double* __restrict__ x     = sys.xData();
    double* __restrict__ y     = sys.yData();
    double* __restrict__ vx    = sys.vxData();
    double* __restrict__ vy    = sys.vyData();
    double* __restrict__ theta = sys.thetaData();
    double* __restrict__ ax    = sys.axData();
    double* __restrict__ ay    = sys.ayData();

    // Derive dt-dependent coefficients here (member caches use member dt_,
    // which is the wrong dt when this is called from an adaptive driver).
    const double mobility_dt        = (1.0 / gamma_) * dt;
    const double noise_amplitude    = std::sqrt(2.0 * (kT_ / gamma_) * dt);
    const double rot_noise_amp      = std::sqrt(2.0 * D_r_ * dt);
    const bool   anchor_on          = (gamma_a_ > 0.0);
    const double anchor_mobility_dt = anchor_on ? (1.0 / gamma_a_) * dt : 0.0;
    const double anchor_noise_amp   = anchor_on
                                          ? std::sqrt(2.0 * (kT_ / gamma_a_) * dt)
                                          : 0.0;

    const double inv_dt     = 1.0 / dt;
    const double max_drift2 = max_drift_ * max_drift_;

    // ---- Translational kick (active + spring + noise; WCA omitted) ----------
    for (std::size_t i = 0; i < N; ++i) {
        const double xi_noise = rng.gaussian();
        const double yi_noise = rng.gaussian();

        const double fx_act = f0_ * std::cos(theta[i]);
        const double fy_act = f0_ * std::sin(theta[i]);

        double fx_spring = 0.0;
        double fy_spring = 0.0;
        if (k_a_ > 0.0) {
            double sep_x = x[i] - ax[i];
            double sep_y = y[i] - ay[i];
            box.minimumImage(sep_x, sep_y);
            fx_spring = -k_a_ * sep_x;
            fy_spring = -k_a_ * sep_y;
        }

        double drift_x = mobility_dt * (fx_act + fx_spring);
        double drift_y = mobility_dt * (fy_act + fy_spring);

        if (max_drift_ > 0.0) {
            const double d2 = drift_x * drift_x + drift_y * drift_y;
            if (d2 > max_drift2) {
                const double scale = max_drift_ / std::sqrt(d2);
                drift_x *= scale;
                drift_y *= scale;
            }
        }

        const double dx = drift_x + noise_amplitude * xi_noise;
        const double dy = drift_y + noise_amplitude * yi_noise;

        x[i]  += dx;
        y[i]  += dy;
        // Velocity here reflects only the kick contribution; the drift
        // integrator that ran before this call is responsible for whatever
        // velocity bookkeeping it needs.
        vx[i]  = dx * inv_dt;
        vy[i]  = dy * inv_dt;
    }

    // ---- Rotational diffusion -----------------------------------------------
    if (rot_noise_amp > 0.0) {
        for (std::size_t i = 0; i < N; ++i) {
            theta[i] += rot_noise_amp * rng.gaussian();
        }
    }

    // ---- Anchor evolution ---------------------------------------------------
    if (anchor_on) {
        for (std::size_t i = 0; i < N; ++i) {
            const double xi_a = rng.gaussian();
            const double yi_a = rng.gaussian();

            double drift_ax = 0.0;
            double drift_ay = 0.0;
            if (k_a_ > 0.0) {
                double sep_x = x[i] - ax[i];
                double sep_y = y[i] - ay[i];
                box.minimumImage(sep_x, sep_y);
                drift_ax = anchor_mobility_dt * k_a_ * sep_x;
                drift_ay = anchor_mobility_dt * k_a_ * sep_y;
            }

            if (max_drift_ > 0.0) {
                const double d2 = drift_ax * drift_ax + drift_ay * drift_ay;
                if (d2 > max_drift2) {
                    const double scale = max_drift_ / std::sqrt(d2);
                    drift_ax *= scale;
                    drift_ay *= scale;
                }
            }

            ax[i] += drift_ax + anchor_noise_amp * xi_a;
            ay[i] += drift_ay + anchor_noise_amp * yi_a;
        }
    }

    // ---- Periodic boundary conditions ---------------------------------------
    for (std::size_t i = 0; i < N; ++i) {
        double xi = x[i];
        double yi = y[i];
        box.wrap(xi, yi);
        x[i] = xi;
        y[i] = yi;
    }
    if (anchor_on) {
        for (std::size_t i = 0; i < N; ++i) {
            double xi = ax[i];
            double yi = ay[i];
            box.wrap(xi, yi);
            ax[i] = xi;
            ay[i] = yi;
        }
    }
}
