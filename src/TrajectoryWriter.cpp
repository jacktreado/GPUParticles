#include "TrajectoryWriter.hpp"

#include "Box.hpp"
#include "Config.hpp"
#include "ContactDurationAccumulator.hpp"
#include "CorrelationAccumulator.hpp"
#include "System.hpp"

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers: write a single scalar attribute of a given HDF5 type onto any
// HDF5 object (file, group, dataset).  Defined here to avoid cluttering the
// implementation below with repetitive dataspace setup.
// ---------------------------------------------------------------------------
namespace {

void writeDoubleAttr(H5::H5Object& obj, const char* name, double val) {
    H5::DataSpace sp(H5S_SCALAR);
    H5::Attribute a = obj.createAttribute(name, H5::PredType::NATIVE_DOUBLE, sp);
    a.write(H5::PredType::NATIVE_DOUBLE, &val);
}

void writeUInt64Attr(H5::H5Object& obj, const char* name, std::uint64_t val) {
    H5::DataSpace sp(H5S_SCALAR);
    H5::Attribute a = obj.createAttribute(name, H5::PredType::NATIVE_UINT64, sp);
    a.write(H5::PredType::NATIVE_UINT64, &val);
}

void writeStringAttr(H5::H5Object& obj, const char* name, const std::string& val) {
    H5::StrType   st(H5::PredType::C_S1, H5T_VARIABLE);
    H5::DataSpace sp(H5S_SCALAR);
    H5::Attribute a = obj.createAttribute(name, st, sp);
    const char*   cstr = val.c_str();
    a.write(st, &cstr);
}

} // anonymous namespace

// ---------------------------------------------------------------------------

TrajectoryWriter::TrajectoryWriter(const std::string& path) : path_(path) {
    // Turn off the default HDF5 error printing so exceptions come through
    // cleanly instead of accompanied by a stack dump to stderr.
    H5::Exception::dontPrint();

    try {
        file_ = std::make_unique<H5::H5File>(path, H5F_ACC_TRUNC);
    } catch (const H5::Exception& e) {
        throw std::runtime_error(
            "TrajectoryWriter: could not create '" + path + "': " + e.getCDetailMsg());
    }
}

TrajectoryWriter::~TrajectoryWriter() {
    // unique_ptr destructor calls H5File::~H5File(), which flushes and closes.
    // Explicit close() is a no-op if already called.
}

void TrajectoryWriter::writeSimulationConfig(const Config& cfg) {
    if (!file_) return;
    // Frame timestamps are now passed in explicitly per frame (no fixed dt).

    // ---- Required + dimensionless inputs (the parameter set the user controls) -
    writeUInt64Attr(*file_, "N",          static_cast<std::uint64_t>(cfg.N));
    writeDoubleAttr(*file_, "phi",        cfg.phi);
    writeDoubleAttr(*file_, "Pe",         cfg.Pe);
    writeDoubleAttr(*file_, "De",         cfg.De);
    writeDoubleAttr(*file_, "R",          cfg.R);
    writeDoubleAttr(*file_, "C",          cfg.C);
    writeDoubleAttr(*file_, "delta",      cfg.delta);
    writeStringAttr(*file_, "potential",  cfg.potentialName());

    // ---- Frozen working units ----------------------------------------------
    writeDoubleAttr(*file_, "sigma",      cfg.sigma);
    writeDoubleAttr(*file_, "epsilon",    cfg.epsilon);
    writeDoubleAttr(*file_, "gamma",      cfg.gamma);

    // ---- Derived microscopic parameters (handy for postprocessing) ---------
    writeDoubleAttr(*file_, "L",          cfg.L);
    writeDoubleAttr(*file_, "kT",         cfg.kT);
    writeDoubleAttr(*file_, "f0",         cfg.f0);
    writeDoubleAttr(*file_, "tau_theta",  cfg.tau_theta);
    writeDoubleAttr(*file_, "gamma_a",    cfg.gamma_a);
    writeDoubleAttr(*file_, "k_a",        cfg.k_a);

    // ---- Adaptive integrator + I/O -----------------------------------------
    writeDoubleAttr(*file_, "t_end",      cfg.t_end);
    writeDoubleAttr(*file_, "output_dt",  cfg.output_dt);
    writeDoubleAttr(*file_, "abs_tol",    cfg.abs_tol);
    writeDoubleAttr(*file_, "rel_tol",    cfg.rel_tol);
    writeDoubleAttr(*file_, "dt_init",    cfg.dt_init);
    writeStringAttr(*file_, "output_file", cfg.output_file);
    writeStringAttr(*file_, "init_mode",  cfg.init_mode);
    writeUInt64Attr(*file_, "seed",       static_cast<std::uint64_t>(cfg.seed));
}

void TrajectoryWriter::writeFrame(const System& sys, const Box& box,
                                  std::size_t step, double time) {
    if (!file_) return;

    const std::size_t N = sys.getNumParticles();

    // Group name: /frame_00000000, /frame_00000001, ...
    std::ostringstream oss;
    oss << "/frame_" << std::setfill('0') << std::setw(8) << frame_;
    H5::Group grp = file_->createGroup(oss.str());

    // Per-frame attributes: step, time, frame index.
    writeUInt64Attr(grp, "step",  static_cast<std::uint64_t>(step));
    writeDoubleAttr(grp, "time",  time);
    writeUInt64Attr(grp, "frame", static_cast<std::uint64_t>(frame_));

    // Box dimensions as attributes on the frame group so the reader can
    // reconstruct the periodic cell without parsing extra files.
    writeDoubleAttr(grp, "Lx", box.getLx());
    writeDoubleAttr(grp, "Ly", box.getLy());

    const hsize_t dims2[2] = {static_cast<hsize_t>(N), 2};
    const hsize_t dims1[1] = {static_cast<hsize_t>(N)};

    // Positions dataset: shape (N, 2), row-major, each row = [xi, yi].
    {
        std::vector<double> buf(2 * N);
        for (std::size_t i = 0; i < N; ++i) {
            buf[2 * i]     = sys.getX(i);
            buf[2 * i + 1] = sys.getY(i);
        }
        H5::DataSpace dspace(2, dims2);
        H5::DataSet   dset = grp.createDataSet(
            "positions", H5::PredType::NATIVE_DOUBLE, dspace);
        dset.write(buf.data(), H5::PredType::NATIVE_DOUBLE);
    }

    // Velocities dataset: shape (N, 2), each row = [vxi, vyi].
    {
        std::vector<double> buf(2 * N);
        for (std::size_t i = 0; i < N; ++i) {
            buf[2 * i]     = sys.getVx(i);
            buf[2 * i + 1] = sys.getVy(i);
        }
        H5::DataSpace dspace(2, dims2);
        H5::DataSet   dset = grp.createDataSet(
            "velocities", H5::PredType::NATIVE_DOUBLE, dspace);
        dset.write(buf.data(), H5::PredType::NATIVE_DOUBLE);
    }

    // Orientations dataset: shape (N,), each element = theta_i in radians.
    {
        std::vector<double> buf(N);
        for (std::size_t i = 0; i < N; ++i) buf[i] = sys.getTheta(i);
        H5::DataSpace dspace(1, dims1);
        H5::DataSet   dset = grp.createDataSet(
            "orientations", H5::PredType::NATIVE_DOUBLE, dspace);
        dset.write(buf.data(), H5::PredType::NATIVE_DOUBLE);
    }

    // Anchors dataset: shape (N, 2), each row = [ax_i, ay_i].
    {
        std::vector<double> buf(2 * N);
        for (std::size_t i = 0; i < N; ++i) {
            buf[2 * i]     = sys.getAx(i);
            buf[2 * i + 1] = sys.getAy(i);
        }
        H5::DataSpace dspace(2, dims2);
        H5::DataSet   dset = grp.createDataSet(
            "anchors", H5::PredType::NATIVE_DOUBLE, dspace);
        dset.write(buf.data(), H5::PredType::NATIVE_DOUBLE);
    }

    // Forces dataset: shape (N, 2), each row = [fx_i, fy_i]. These are the
    // net pair-interaction forces stored on System (written by
    // ForceCalculator::compute). Callers responsible for ensuring forces are
    // up to date with current positions before writing — every writeFrame
    // call site in main.cpp precedes this with forces.compute(sys, box).
    {
        std::vector<double> buf(2 * N);
        for (std::size_t i = 0; i < N; ++i) {
            buf[2 * i]     = sys.getFx(i);
            buf[2 * i + 1] = sys.getFy(i);
        }
        H5::DataSpace dspace(2, dims2);
        H5::DataSet   dset = grp.createDataSet(
            "forces", H5::PredType::NATIVE_DOUBLE, dspace);
        dset.write(buf.data(), H5::PredType::NATIVE_DOUBLE);
    }

    ++frame_;
}

void TrajectoryWriter::writeCorrelations(const CorrelationAccumulator& corr,
                                         double t_warm) {
    if (!file_) return;

    const std::size_t n_lags    = corr.nLags();
    const double      corr_dt   = corr.corrDt();
    const std::size_t n_samples = corr.numSamplesTaken();
    const double      dt_max    = corr_dt * static_cast<double>(n_lags);

    H5::Group grp = file_->createGroup("/correlations");

    writeDoubleAttr(grp, "corr_dt",      corr_dt);
    writeDoubleAttr(grp, "corr_dt_max",  dt_max);
    writeUInt64Attr(grp, "n_corr_steps", static_cast<std::uint64_t>(n_lags));
    writeDoubleAttr(grp, "t_warm",       t_warm);
    writeUInt64Attr(grp, "n_samples",    static_cast<std::uint64_t>(n_samples));

    const hsize_t dims1[1] = {static_cast<hsize_t>(n_lags)};

    const auto& sum_vn = corr.sumVn();
    const auto& sum_Fn = corr.sumFn();
    const auto& sum_vv = corr.sumVv();
    const auto& count  = corr.count();

    std::vector<double> tau(n_lags);
    std::vector<double> C_vn(n_lags);
    std::vector<double> C_Fn(n_lags);
    std::vector<double> C_vv(n_lags);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t k = 0; k < n_lags; ++k) {
        tau[k]  = static_cast<double>(k) * corr_dt;
        if (count[k] == 0) {
            C_vn[k] = nan;
            C_Fn[k] = nan;
            C_vv[k] = nan;
        } else {
            const double inv = 1.0 / static_cast<double>(count[k]);
            C_vn[k] = sum_vn[k] * inv;
            C_Fn[k] = sum_Fn[k] * inv;
            C_vv[k] = sum_vv[k] * inv;
        }
    }

    {
        H5::DataSpace dspace(1, dims1);
        H5::DataSet   dset = grp.createDataSet(
            "tau", H5::PredType::NATIVE_DOUBLE, dspace);
        dset.write(tau.data(), H5::PredType::NATIVE_DOUBLE);
    }
    {
        H5::DataSpace dspace(1, dims1);
        H5::DataSet   dset = grp.createDataSet(
            "C_vn", H5::PredType::NATIVE_DOUBLE, dspace);
        dset.write(C_vn.data(), H5::PredType::NATIVE_DOUBLE);
    }
    {
        H5::DataSpace dspace(1, dims1);
        H5::DataSet   dset = grp.createDataSet(
            "C_Fn", H5::PredType::NATIVE_DOUBLE, dspace);
        dset.write(C_Fn.data(), H5::PredType::NATIVE_DOUBLE);
    }
    {
        H5::DataSpace dspace(1, dims1);
        H5::DataSet   dset = grp.createDataSet(
            "C_vv", H5::PredType::NATIVE_DOUBLE, dspace);
        dset.write(C_vv.data(), H5::PredType::NATIVE_DOUBLE);
    }
    {
        H5::DataSpace dspace(1, dims1);
        H5::DataSet   dset = grp.createDataSet(
            "count", H5::PredType::NATIVE_UINT64, dspace);
        dset.write(count.data(), H5::PredType::NATIVE_UINT64);
    }
}

void TrajectoryWriter::writeContactDurations(const ContactDurationAccumulator& cd) {
    if (!file_) return;

    H5::Group grp = file_->createGroup("/contact_durations");

    writeDoubleAttr(grp, "contact_cutoff",           cd.contactCutoff());
    writeUInt64Attr(grp, "count",                    cd.count());
    writeDoubleAttr(grp, "mean",                     cd.mean());
    writeDoubleAttr(grp, "stddev",                   cd.stddev());
    writeDoubleAttr(grp, "min",                      cd.min());
    writeDoubleAttr(grp, "max",                      cd.max());
    writeUInt64Attr(grp, "in_progress",
                    static_cast<std::uint64_t>(cd.inProgressCount()));
    writeDoubleAttr(grp, "in_progress_sum_duration", cd.inProgressSumDuration());
    writeUInt64Attr(grp, "n_samples",                cd.sampleCount());
}

void TrajectoryWriter::close() {
    file_.reset();   // flushes and closes the HDF5 file
}
