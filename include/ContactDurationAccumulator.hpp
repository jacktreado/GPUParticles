#pragma once

#include "CellList.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>

class System;
class Box;

// =============================================================================
// ContactDurationAccumulator
// -----------------------------------------------------------------------------
// Streaming statistics of pair contact durations: how long, on average, two
// particles stay within r < contact_cutoff before separating again.
//
// Data structures:
//   - active_   : sparse map keyed by uint64 pair-id, holding the running
//                 duration of every currently-active contact + the sample step
//                 it was last observed on. ~24 B per active pair.
//   - cl_       : private cell list for pair traversal; rebuilt on demand.
//   - Welford state (count/mean/M2/min/max): 5 doubles, holds running mean
//                 and sample standard deviation of every COMPLETED contact.
//
// Algorithm per sample:
//   1. Iterate cell-list neighbors; for each pair with r2 < contact_r2:
//        active_[key].duration += dt
//        active_[key].last_step = step_
//   2. Sweep active_: any entry with last_step != step_ ended this sample;
//      add its duration to the Welford accumulator and erase.
//
// Contacts still active at simulation end are NOT folded into completed-stats
// — they're reported separately (count + sum of durations) so the caller can
// notice if censoring is biasing the mean.
// =============================================================================
class ContactDurationAccumulator {
public:
    // r_skin and box are used to size and seed the private cell list. The
    // contact threshold is stored as squared distance to avoid sqrt in the
    // inner loop.
    ContactDurationAccumulator(std::size_t N, double contact_cutoff,
                               double r_skin, const Box& box);

    // Sample at the current state. dt is the simulation time step that just
    // completed; it is added to every ongoing-contact's duration accumulator.
    void sample(const System& sys, const Box& box, double dt);

    // ---- Completed-contact statistics ---------------------------------------
    std::uint64_t count()  const { return n_completed_; }
    double        mean()   const;
    double        stddev() const;
    double        min()    const { return n_completed_ ? min_ : nan(); }
    double        max()    const { return n_completed_ ? max_ : nan(); }

    // ---- In-progress diagnostics --------------------------------------------
    std::size_t inProgressCount()       const { return active_.size(); }
    double      inProgressSumDuration() const;

    // ---- Bookkeeping --------------------------------------------------------
    std::uint64_t sampleCount()   const { return n_samples_; }
    double        contactCutoff() const { return contact_cutoff_; }

private:
    struct ContactEntry {
        double        duration  = 0.0;
        std::uint64_t last_step = 0;
    };

    static double nan() { return std::numeric_limits<double>::quiet_NaN(); }

    double   contact_cutoff_;
    double   contact_r2_;
    CellList cl_;
    bool     cl_initialized_ = false;

    std::unordered_map<std::uint64_t, ContactEntry> active_;

    // Welford state for the completed-durations distribution.
    std::uint64_t n_completed_ = 0;
    double        mean_        = 0.0;
    double        M2_          = 0.0;
    double        min_         = 0.0;
    double        max_         = 0.0;

    std::uint64_t n_samples_ = 0;
    std::uint64_t step_      = 0;

    void addCompleted(double dur);
};
