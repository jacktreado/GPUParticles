#include "HeunIntegrator.hpp"

#include "Box.hpp"
#include "CellList.hpp"
#include "ForceCalculator.hpp"
#include "RandomGenerator.hpp"
#include "System.hpp"

#include <cmath>
#include <cstddef>

HeunIntegrator::HeunIntegrator(double gamma, double dt) : gamma_(gamma), dt_(dt) {
    recomputeCachedConstants();
}

void HeunIntegrator::recomputeCachedConstants() {
    mobility_dt_         = (1.0 / gamma_) * dt_;
    noise_amplitude_     = std::sqrt(2.0 * (kT_ / gamma_) * dt_);
    rot_noise_amplitude_ = std::sqrt(2.0 * D_r_ * dt_);
    if (gamma_a_ > 0.0) {
        anchor_mobility_dt_     = (1.0 / gamma_a_) * dt_;
        anchor_noise_amplitude_ = std::sqrt(2.0 * (kT_ / gamma_a_) * dt_);
    } else {
        anchor_mobility_dt_     = 0.0;
        anchor_noise_amplitude_ = 0.0;
    }
}

void HeunIntegrator::ensureScratch(std::size_t N, bool anchor_on) {
    if (r_orig_x_.size() != N) {
        r_orig_x_.resize(N);
        r_orig_y_.resize(N);
        drift1_dx_.resize(N);
        drift1_dy_.resize(N);
        noise_x_.resize(N);
        noise_y_.resize(N);
    }
    if (anchor_on) {
        if (a_orig_x_.size() != N) {
            a_orig_x_.resize(N);
            a_orig_y_.resize(N);
            drift1_a_dx_.resize(N);
            drift1_a_dy_.resize(N);
            noise_a_x_.resize(N);
            noise_a_y_.resize(N);
        }
    }
}

void HeunIntegrator::step(System& sys, const Box& box,
                          const ForceCalculator& fc,
                          const CellList* cl,
                          RandomGenerator& rng) {
    const std::size_t N = sys.getNumParticles();
    const bool   anchor_on  = (gamma_a_ > 0.0);
    const bool   spring_on  = (k_a_ > 0.0);
    const bool   use_cl     = (cl != nullptr) && !cl->useBruteForce();

    ensureScratch(N, anchor_on);

    double* __restrict__ x     = sys.xData();
    double* __restrict__ y     = sys.yData();
    double* __restrict__ vx    = sys.vxData();
    double* __restrict__ vy    = sys.vyData();
    double* __restrict__ theta = sys.thetaData();
    double* __restrict__ ax    = sys.axData();
    double* __restrict__ ay    = sys.ayData();
    const double* __restrict__ fx = sys.fxData();
    const double* __restrict__ fy = sys.fyData();

    const double mu_dt      = mobility_dt_;
    const double mu_a_dt    = anchor_mobility_dt_;
    const double dw_amp     = noise_amplitude_;
    const double dw_a_amp   = anchor_noise_amplitude_;
    const double rot_amp    = rot_noise_amplitude_;
    const double max_drift2 = max_drift_ * max_drift_;
    const double inv_dt     = 1.0 / dt_;

    // ---- (1) F1 at current state -------------------------------------------
    if (use_cl) fc.compute(sys, box, *cl);
    else        fc.compute(sys, box);

    // ---- (2) Save originals; sample noise; compute drift1; advance to predictor.
    // We fuse all per-particle work into one pass for cache locality. Per-DOF
    // RNG ordering inside the loop: (Z_x, Z_y) translation, Z_theta rotation
    // (when enabled), then (Z_ax, Z_ay) anchor (when enabled).
    for (std::size_t i = 0; i < N; ++i) {
        const double rx_n  = x[i];
        const double ry_n  = y[i];
        const double th_n  = theta[i];
        r_orig_x_[i] = rx_n;
        r_orig_y_[i] = ry_n;

        const double zx = rng.gaussian();
        const double zy = rng.gaussian();
        noise_x_[i] = zx;
        noise_y_[i] = zy;

        // Spring sep at current state (used by both r and a drift1).
        double sep_x = 0.0, sep_y = 0.0;
        if (spring_on) {
            sep_x = rx_n - ax[i];
            sep_y = ry_n - ay[i];
            box.minimumImage(sep_x, sep_y);
        }

        const double fx_act = f0_ * std::cos(th_n);
        const double fy_act = f0_ * std::sin(th_n);

        double dx1 = mu_dt * (fx[i] + fx_act + (-k_a_ * sep_x));
        double dy1 = mu_dt * (fy[i] + fy_act + (-k_a_ * sep_y));
        if (max_drift_ > 0.0) {
            const double d2 = dx1 * dx1 + dy1 * dy1;
            if (d2 > max_drift2) {
                const double s = max_drift_ / std::sqrt(d2);
                dx1 *= s; dy1 *= s;
            }
        }
        drift1_dx_[i] = dx1;
        drift1_dy_[i] = dy1;

        // Rotational predictor (theta has only noise — Z_theta becomes
        // theta_{n+1} as well, since there's no drift to correct).
        if (rot_amp > 0.0) {
            theta[i] = th_n + rot_amp * rng.gaussian();
        }

        // Anchor predictor (when enabled).
        if (anchor_on) {
            const double zax = rng.gaussian();
            const double zay = rng.gaussian();
            noise_a_x_[i] = zax;
            noise_a_y_[i] = zay;

            const double ax_n = ax[i];
            const double ay_n = ay[i];
            a_orig_x_[i] = ax_n;
            a_orig_y_[i] = ay_n;

            double dax1 = 0.0, day1 = 0.0;
            if (spring_on) {
                dax1 = mu_a_dt * k_a_ * sep_x;
                day1 = mu_a_dt * k_a_ * sep_y;
                if (max_drift_ > 0.0) {
                    const double d2 = dax1 * dax1 + day1 * day1;
                    if (d2 > max_drift2) {
                        const double s = max_drift_ / std::sqrt(d2);
                        dax1 *= s; day1 *= s;
                    }
                }
            }
            drift1_a_dx_[i] = dax1;
            drift1_a_dy_[i] = day1;

            ax[i] = ax_n + dax1 + dw_a_amp * zax;
            ay[i] = ay_n + day1 + dw_a_amp * zay;
        }

        // Translational predictor — write last so the spring sep above used
        // the un-stepped position (and so x[i]/y[i] reading in the loop is
        // consistent).
        x[i] = rx_n + dx1 + dw_amp * zx;
        y[i] = ry_n + dy1 + dw_amp * zy;
    }

    // ---- (4) F2 at predictor state -----------------------------------------
    // Skipping a CellList rebuild between stages is safe: predictor delta is
    // bounded by max_drift + |Z| * dw_amp, well under r_skin/2 for default
    // skin = 0.5. This is the assumption documented in the header.
    if (use_cl) fc.compute(sys, box, *cl);
    else        fc.compute(sys, box);

    // ---- (5) Compute drift2 at the predictor state and apply the corrector.
    // Strict ordering inside the loop: read predictor state for particle i,
    // compute drift2, write final state for i, advance to i+1. Other
    // particles' positions are untouched, so spring sep computations remain
    // consistent.
    for (std::size_t i = 0; i < N; ++i) {
        const double rx_p = x[i];
        const double ry_p = y[i];
        const double th_p = theta[i];     // already theta_{n+1}

        double sep_x = 0.0, sep_y = 0.0;
        if (spring_on) {
            sep_x = rx_p - ax[i];
            sep_y = ry_p - ay[i];
            box.minimumImage(sep_x, sep_y);
        }

        const double fx_act_p = f0_ * std::cos(th_p);
        const double fy_act_p = f0_ * std::sin(th_p);

        double dx2 = mu_dt * (fx[i] + fx_act_p + (-k_a_ * sep_x));
        double dy2 = mu_dt * (fy[i] + fy_act_p + (-k_a_ * sep_y));
        if (max_drift_ > 0.0) {
            const double d2 = dx2 * dx2 + dy2 * dy2;
            if (d2 > max_drift2) {
                const double s = max_drift_ / std::sqrt(d2);
                dx2 *= s; dy2 *= s;
            }
        }

        // Anchor drift2 uses the SAME predictor-state spring sep above.
        if (anchor_on) {
            double dax2 = 0.0, day2 = 0.0;
            if (spring_on) {
                dax2 = mu_a_dt * k_a_ * sep_x;
                day2 = mu_a_dt * k_a_ * sep_y;
                if (max_drift_ > 0.0) {
                    const double d2 = dax2 * dax2 + day2 * day2;
                    if (d2 > max_drift2) {
                        const double s = max_drift_ / std::sqrt(d2);
                        dax2 *= s; day2 *= s;
                    }
                }
            }
            ax[i] = a_orig_x_[i]
                  + 0.5 * (drift1_a_dx_[i] + dax2)
                  + dw_a_amp * noise_a_x_[i];
            ay[i] = a_orig_y_[i]
                  + 0.5 * (drift1_a_dy_[i] + day2)
                  + dw_a_amp * noise_a_y_[i];
        }

        const double dx_total = 0.5 * (drift1_dx_[i] + dx2) + dw_amp * noise_x_[i];
        const double dy_total = 0.5 * (drift1_dy_[i] + dy2) + dw_amp * noise_y_[i];
        x[i]  = r_orig_x_[i] + dx_total;
        y[i]  = r_orig_y_[i] + dy_total;
        vx[i] = dx_total * inv_dt;
        vy[i] = dy_total * inv_dt;
    }
}
