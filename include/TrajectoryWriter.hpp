#pragma once

#include <H5Cpp.h>
#include <cstddef>
#include <memory>
#include <string>

class System;
class Box;
class Config;
class CorrelationAccumulator;
class ContactDurationAccumulator;

// =============================================================================
// TrajectoryWriter
// -----------------------------------------------------------------------------
// Writes a multi-frame trajectory to an HDF5 file.
//
// File layout
// -----------
//   /                          root group
//     (attributes: all Config fields — written by writeSimulationConfig)
//
//   /frame_00000000            one group per frame, indexed by frame number
//     attr  step  : uint64    simulation step number
//     attr  time  : double    physical time = step * dt
//     attr  frame : uint64    sequential frame index (0, 1, 2, …)
//     dataset  positions      shape (N, 2), float64, row-major [xi, yi]
//     dataset  velocities     shape (N, 2), float64, row-major [vx_i, vy_i]
//     dataset  orientations   shape (N,),   float64, theta_i in radians
//     dataset  anchors        shape (N, 2), float64, row-major [ax_i, ay_i]
//     dataset  forces         shape (N, 2), float64, row-major [fx_i, fy_i]
//                             — net pair-interaction force on each particle
//                             (System::fx_/fy_, set by ForceCalculator::compute).
//                             Active and spring contributions are NOT included;
//                             those live only inside the integrator step.
//
//   /frame_00000001
//     ...
//
// Usage
// -----
//   TrajectoryWriter writer("out.h5");
//   writer.writeSimulationConfig(cfg);   // root attributes; call once
//   writer.writeFrame(sys, box, 0);      // step 0
//   writer.writeFrame(sys, box, 1000);   // step 1000
//   writer.close();                      // or let the destructor do it
// =============================================================================
class TrajectoryWriter {
public:
    explicit TrajectoryWriter(const std::string& path);
    ~TrajectoryWriter();

    // Write all Config fields as attributes on the root group.
    // Also stores cfg.dt so that writeFrame() can compute time = step * dt.
    // Call once before any writeFrame() calls.
    void writeSimulationConfig(const Config& cfg);

    // Append one frame to the file.
    //   step   — simulation step number / counter (stored as attribute)
    //   time   — accumulated physical time at this frame (stored as attribute)
    //   frame  — sequential write index, incremented automatically
    //   dataset "positions" — shape (N, 2), values [xi, yi] per row
    void writeFrame(const System& sys, const Box& box,
                    std::size_t step, double time);

    // Backward-compatible overload: emits time = 0.0. Useful for unit tests
    // and snapshots where a meaningful time is not available.
    void writeFrame(const System& sys, const Box& box, std::size_t step) {
        writeFrame(sys, box, step, 0.0);
    }

    // Write the final normalized correlation functions C_vn(tau), C_Fn(tau),
    // and C_vv(tau) into a /correlations group at the root. Call once after
    // the time loop; the accumulator's running sums and per-lag contribution
    // counts are read and normalized inline (NaN where count == 0).
    //
    // Layout:
    //   /correlations
    //     attr  corr_dt        : double
    //     attr  corr_dt_max    : double
    //     attr  n_corr_steps   : uint64
    //     attr  t_warm         : double
    //     attr  n_samples      : uint64    (snapshots taken since t_warm)
    //     dataset tau   (n_corr_steps,) float64    tau_k = k * corr_dt
    //     dataset C_vn  (n_corr_steps,) float64    <v . n_past>
    //     dataset C_Fn  (n_corr_steps,) float64    <F . n_past>
    //     dataset C_vv  (n_corr_steps,) float64    <v . v_past>
    //     dataset count (n_corr_steps,) uint64     contribution count per lag
    void writeCorrelations(const CorrelationAccumulator& corr, double t_warm);

    // Write the streaming statistics of pair contact durations into a
    // /contact_durations group at the root. Attribute-only group, no datasets.
    //
    // Layout:
    //   /contact_durations
    //     attr  contact_cutoff           : double
    //     attr  count                    : uint64  (completed contacts)
    //     attr  mean                     : double  (NaN if count == 0)
    //     attr  stddev                   : double  (sample, NaN if count < 2)
    //     attr  min                      : double
    //     attr  max                      : double
    //     attr  in_progress              : uint64  (active at sim end)
    //     attr  in_progress_sum_duration : double
    //     attr  n_samples                : uint64  (times sample() was called)
    void writeContactDurations(const ContactDurationAccumulator& cd);

    void close();

    bool               isOpen()  const { return file_ != nullptr; }
    const std::string& getPath() const { return path_; }

private:
    std::string                 path_;
    std::unique_ptr<H5::H5File> file_;
    std::size_t                 frame_ = 0;     // incremented by writeFrame
};
