#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

class System;

// =============================================================================
// CorrelationAccumulator
// -----------------------------------------------------------------------------
// On-the-fly ensemble-averaged correlation functions on a uniform lag grid
// tau_k = k * corr_dt, k = 0..n_lags-1. Three correlations are accumulated:
//
//   C_vn(tau_k) = < v_i(t) . n_hat_i(t - tau_k) >_{i, t}   (cross)
//   C_Fn(tau_k) = < F_i(t) . n_hat_i(t - tau_k) >_{i, t}   (cross)
//   C_vv(tau_k) = < v_i(t) . v_i    (t - tau_k) >_{i, t}   (auto)
//
// where n_hat = (cos theta, sin theta) and the average runs over all particles
// and all valid sample times.
//
// Memory: two ring buffers (past orientations and past velocities), i.e.
// 4 * N * n_lags doubles. Forces are NOT buffered; they are dotted against the
// orientation buffer at sample time. Per-call cost: O(N * n_lags) flops once
// the buffer is full.
//
// The caller is responsible for ensuring System holds current velocities AND
// pair forces at the sample instant (see TrajectoryWriter::writeFrame for the
// same contract).
// =============================================================================
class CorrelationAccumulator {
public:
    CorrelationAccumulator(std::size_t N, std::size_t n_lags, double corr_dt);

    // Take one sample at the current simulation time. Reads theta_, vx_, vy_,
    // fx_, fy_ from sys; advances the ring buffer and updates all valid lag
    // accumulators.
    void sample(const System& sys);

    std::size_t numSamplesTaken() const { return n_samples_taken_; }
    std::size_t nLags()           const { return n_lags_; }
    std::size_t N()               const { return N_; }
    double      corrDt()          const { return corr_dt_; }

    const std::vector<double>&        sumVn() const { return sum_vn_; }
    const std::vector<double>&        sumFn() const { return sum_Fn_; }
    const std::vector<double>&        sumVv() const { return sum_vv_; }
    const std::vector<std::uint64_t>& count() const { return count_; }

private:
    std::size_t N_;
    std::size_t n_lags_;
    double      corr_dt_;

    // Ring buffers of past n_hat = (cos theta, sin theta) and past velocity.
    // Element for slot s, particle i lives at index s * N_ + i. Slots run
    // 0..n_lags_-1, advancing by one (mod n_lags_) each sample.
    std::vector<double>        nx_buf_;
    std::vector<double>        ny_buf_;
    std::vector<double>        vx_buf_;
    std::vector<double>        vy_buf_;

    // Per-lag running sums and contribution counts (= samples_with_lag * N_).
    std::vector<double>        sum_vn_;
    std::vector<double>        sum_Fn_;
    std::vector<double>        sum_vv_;
    std::vector<std::uint64_t> count_;

    std::size_t head_            = 0;   // slot index of most-recent write
    std::size_t n_samples_taken_ = 0;   // total samples since construction
};
