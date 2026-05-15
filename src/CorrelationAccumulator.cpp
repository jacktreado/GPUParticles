#include "CorrelationAccumulator.hpp"

#include "System.hpp"

#include <cmath>
#include <stdexcept>

CorrelationAccumulator::CorrelationAccumulator(std::size_t N,
                                               std::size_t n_lags,
                                               double      corr_dt)
    : N_(N),
      n_lags_(n_lags),
      corr_dt_(corr_dt),
      nx_buf_(N * n_lags, 0.0),
      ny_buf_(N * n_lags, 0.0),
      vx_buf_(N * n_lags, 0.0),
      vy_buf_(N * n_lags, 0.0),
      sum_vn_(n_lags, 0.0),
      sum_Fn_(n_lags, 0.0),
      sum_vv_(n_lags, 0.0),
      count_(n_lags, 0)
{
    if (N == 0)      throw std::runtime_error("CorrelationAccumulator: N must be > 0");
    if (n_lags == 0) throw std::runtime_error("CorrelationAccumulator: n_lags must be > 0");
    if (corr_dt <= 0.0) throw std::runtime_error("CorrelationAccumulator: corr_dt must be > 0");
}

void CorrelationAccumulator::sample(const System& sys) {
    // Advance the ring head. First sample writes slot 0; subsequent samples
    // advance by one (modulo n_lags_).
    if (n_samples_taken_ > 0) {
        head_ = (head_ + 1) % n_lags_;
    }

    const double* theta = sys.getOrientations().data();
    const double* vx    = sys.getVelocitiesX().data();
    const double* vy    = sys.getVelocitiesY().data();
    const double* fx    = sys.getForcesX().data();
    const double* fy    = sys.getForcesY().data();

    // Write current n_hat and v into the new slot.
    double* nx_now = nx_buf_.data() + head_ * N_;
    double* ny_now = ny_buf_.data() + head_ * N_;
    double* vx_now = vx_buf_.data() + head_ * N_;
    double* vy_now = vy_buf_.data() + head_ * N_;
    for (std::size_t i = 0; i < N_; ++i) {
        nx_now[i] = std::cos(theta[i]);
        ny_now[i] = std::sin(theta[i]);
        vx_now[i] = vx[i];
        vy_now[i] = vy[i];
    }

    // Number of lags that are valid right now: at sample j (0-indexed) the lags
    // 0..min(j, n_lags-1) all have a corresponding past slot in the buffer.
    const std::size_t valid_lags =
        (n_samples_taken_ + 1 < n_lags_) ? (n_samples_taken_ + 1) : n_lags_;

    // Accumulate v(t).n(t-tau_k), F(t).n(t-tau_k), and v(t).v(t-tau_k) for
    // each valid lag.
    for (std::size_t k = 0; k < valid_lags; ++k) {
        const std::size_t past_slot = (head_ + n_lags_ - k) % n_lags_;
        const double* nx_p = nx_buf_.data() + past_slot * N_;
        const double* ny_p = ny_buf_.data() + past_slot * N_;
        const double* vx_p = vx_buf_.data() + past_slot * N_;
        const double* vy_p = vy_buf_.data() + past_slot * N_;

        double acc_vn = 0.0;
        double acc_Fn = 0.0;
        double acc_vv = 0.0;
        for (std::size_t i = 0; i < N_; ++i) {
            acc_vn += vx[i] * nx_p[i] + vy[i] * ny_p[i];
            acc_Fn += fx[i] * nx_p[i] + fy[i] * ny_p[i];
            acc_vv += vx[i] * vx_p[i] + vy[i] * vy_p[i];
        }
        sum_vn_[k] += acc_vn;
        sum_Fn_[k] += acc_Fn;
        sum_vv_[k] += acc_vv;
        count_[k]  += static_cast<std::uint64_t>(N_);
    }

    ++n_samples_taken_;
}
