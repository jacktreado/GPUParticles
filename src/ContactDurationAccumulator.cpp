#include "ContactDurationAccumulator.hpp"

#include "Box.hpp"
#include "System.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>

ContactDurationAccumulator::ContactDurationAccumulator(std::size_t N,
                                                       double contact_cutoff,
                                                       double r_skin,
                                                       const Box& box)
    : contact_cutoff_(contact_cutoff),
      contact_r2_(contact_cutoff * contact_cutoff),
      cl_(contact_cutoff, r_skin, N, box)
{
    if (N == 0)
        throw std::runtime_error(
            "ContactDurationAccumulator: N must be > 0");
    if (contact_cutoff <= 0.0)
        throw std::runtime_error(
            "ContactDurationAccumulator: contact_cutoff must be > 0");
    if (r_skin < 0.0)
        throw std::runtime_error(
            "ContactDurationAccumulator: r_skin must be >= 0");
}

void ContactDurationAccumulator::sample(const System& sys, const Box& box,
                                        double dt)
{
    ++step_;
    ++n_samples_;

    if (!cl_initialized_ || cl_.needsRebuild(sys, box)) {
        cl_.rebuild(sys, box);
        cl_initialized_ = true;
    }

    const std::size_t N = sys.getNumParticles();
    const double* __restrict__ x = sys.xData();
    const double* __restrict__ y = sys.yData();
    const double rc2 = contact_r2_;

    auto bump = [&](std::size_t i, std::size_t j) {
        double dx = x[j] - x[i];
        double dy = y[j] - y[i];
        box.minimumImage(dx, dy);
        const double r2 = dx * dx + dy * dy;
        if (r2 >= rc2 || r2 == 0.0) return;
        const std::uint64_t key =
            (static_cast<std::uint64_t>(i) << 32) | static_cast<std::uint64_t>(j);
        auto [it, inserted] = active_.try_emplace(key);
        it->second.duration  += dt;
        it->second.last_step  = step_;
    };

    if (cl_.useBruteForce()) {
        // Small-box fallback. Iterate unique pairs once.
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = i + 1; j < N; ++j) {
                bump(i, j);
            }
        }
    } else {
        const std::int32_t* __restrict__ nlist       = cl_.nlistData();
        const std::int32_t* __restrict__ nlist_start = cl_.nlistStartData();
        for (std::size_t i = 0; i < N; ++i) {
            const std::int32_t kb = nlist_start[i];
            const std::int32_t ke = nlist_start[i + 1];
            for (std::int32_t k = kb; k < ke; ++k) {
                const std::int32_t j = nlist[k];
                if (j <= static_cast<std::int32_t>(i)) continue;  // unique pairs
                bump(i, static_cast<std::size_t>(j));
            }
        }
    }

    // Sweep: any entry not seen this step ended; record and erase.
    for (auto it = active_.begin(); it != active_.end(); ) {
        if (it->second.last_step != step_) {
            addCompleted(it->second.duration);
            it = active_.erase(it);
        } else {
            ++it;
        }
    }
}

void ContactDurationAccumulator::addCompleted(double dur) {
    ++n_completed_;
    if (n_completed_ == 1) {
        mean_ = dur;
        M2_   = 0.0;
        min_  = dur;
        max_  = dur;
    } else {
        // Welford update.
        const double delta  = dur - mean_;
        mean_ += delta / static_cast<double>(n_completed_);
        const double delta2 = dur - mean_;
        M2_   += delta * delta2;
        if (dur < min_) min_ = dur;
        if (dur > max_) max_ = dur;
    }
}

double ContactDurationAccumulator::mean() const {
    return n_completed_ ? mean_ : nan();
}

double ContactDurationAccumulator::stddev() const {
    if (n_completed_ < 2) return nan();
    return std::sqrt(M2_ / static_cast<double>(n_completed_ - 1));
}

double ContactDurationAccumulator::inProgressSumDuration() const {
    double s = 0.0;
    for (const auto& kv : active_) s += kv.second.duration;
    return s;
}
