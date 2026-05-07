#include "Box.hpp"
#include "CellList.hpp"
#include "Config.hpp"
#include "ForceCalculator.hpp"
#include "Initializer.hpp"
#include "Integrator.hpp"
#include "RandomGenerator.hpp"
#include "System.hpp"
#include "TrajectoryWriter.hpp"

#include <H5Cpp.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <string>

// =============================================================================
// End-to-end simulation behavior tests
// These tests wire together multiple modules and verify aggregate behavior:
//   - Initializer places particles correctly.
//   - Energy is zero on a well-spaced lattice (no overlaps).
//   - Repulsive forces push overlapping particles apart.
//   - The Verlet-list integration loop produces the same trajectory as
//     the brute-force loop from the same seed.
//   - TrajectoryWriter produces a well-formed HDF5 file.
// =============================================================================

// ---- Lattice initializer ----------------------------------------------------

TEST(InitializerTest, LatticePositionsInBox) {
    const std::size_t N = 64;
    const double L = 20.0;
    System sys(N);
    Box box(L);

    Initializer::placeOnLattice(sys, box);

    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_GE(sys.getX(i), 0.0)  << "i=" << i;
        EXPECT_LT(sys.getX(i), L)    << "i=" << i;
        EXPECT_GE(sys.getY(i), 0.0)  << "i=" << i;
        EXPECT_LT(sys.getY(i), L)    << "i=" << i;
    }
}

TEST(InitializerTest, LatticePositionsAreUnique) {
    const std::size_t N = 36;
    const double L = 15.0;
    System sys(N);
    Box box(L);

    Initializer::placeOnLattice(sys, box);

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i + 1; j < N; ++j) {
            const bool same = (sys.getX(i) == sys.getX(j) && sys.getY(i) == sys.getY(j));
            EXPECT_FALSE(same) << "i=" << i << " j=" << j;
        }
    }
}

TEST(InitializerTest, RandomPositionsInBox) {
    const std::size_t N = 16;
    const double L = 20.0;
    System sys(N);
    Box box(L);
    RandomGenerator rng(42);

    Initializer::placeRandomly(sys, box, rng, 1.0);

    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_GE(sys.getX(i), 0.0) << "i=" << i;
        EXPECT_LT(sys.getX(i), L)   << "i=" << i;
        EXPECT_GE(sys.getY(i), 0.0) << "i=" << i;
        EXPECT_LT(sys.getY(i), L)   << "i=" << i;
    }
}

TEST(InitializerTest, RandomPositionsRespectMinSep) {
    const std::size_t N   = 16;
    const double L        = 20.0;
    const double min_sep  = 1.0;
    System sys(N);
    Box box(L);
    RandomGenerator rng(17);

    Initializer::placeRandomly(sys, box, rng, min_sep);

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i + 1; j < N; ++j) {
            double dx = sys.getX(j) - sys.getX(i);
            double dy = sys.getY(j) - sys.getY(i);
            box.minimumImage(dx, dy);
            EXPECT_GE(std::sqrt(dx * dx + dy * dy), min_sep) << "i=" << i << " j=" << j;
        }
    }
}

// ---- Energy on a well-spaced lattice is zero --------------------------------

TEST(SimulationTest, LatticeEnergyZeroForLowDensity) {
    // With phi = 0.1 the lattice spacing is much larger than r_cut => U = 0.
    const std::size_t N = 64;
    const double phi    = 0.1;
    const double sigma  = 1.0;
    const double pi     = std::acos(-1.0);
    const double L      = std::sqrt(N * pi * sigma * sigma / (4.0 * phi));

    System sys(N); sys.setSigma(sigma);
    Box box(L);
    ForceCalculator forces(1.0, sigma);

    Initializer::placeOnLattice(sys, box);

    EXPECT_DOUBLE_EQ(forces.computeEnergy(sys, box), 0.0);
}

// ---- Repulsive forces push overlapping particles apart ----------------------

TEST(SimulationTest, RepulsiveForcesDecreaseEnergyOverTime) {
    // Place two particles with significant overlap. Running a few overdamped
    // steps with large D (quick relaxation) should decrease the WCA energy.
    const double L = 20.0;
    System sys(2);
    sys.setPosition(0, 9.5, 10.0);   // separation 1.0 < r_cut ~ 1.12 in x
    sys.setPosition(1, 10.5, 10.0);
    Box box(L);
    ForceCalculator forces(1.0, 1.0);

    const double U_initial = forces.computeEnergy(sys, box);
    ASSERT_GT(U_initial, 0.0) << "setup: particles should overlap";

    Integrator integ(0.1, 1e-4);     // gamma=0.1 => mu=10 for fast relaxation
    integ.setKBT(1.0);
    RandomGenerator rng(1);

    // 200 steps should move the particles apart.
    for (int step = 0; step < 200; ++step) {
        forces.compute(sys, box);
        integ.step(sys, box, rng);
    }

    const double U_final = forces.computeEnergy(sys, box);
    EXPECT_LT(U_final, U_initial);
}

// ---- Cell-list vs brute-force give the same trajectory ----------------------

TEST(SimulationTest, CellListTrajectoryMatchesBruteForce) {
    const std::size_t N = 64;
    const double phi    = 0.3;
    const double sigma  = 1.0;
    const double pi     = std::acos(-1.0);
    const double L      = std::sqrt(N * pi * sigma * sigma / (4.0 * phi));

    System sys_bf(N),  sys_cl(N);
    sys_bf.setSigma(sigma);
    sys_cl.setSigma(sigma);

    Box box(L);
    ForceCalculator forces(1.0, sigma);

    // Identical initial conditions.
    RandomGenerator rng_init(42);
    Initializer::placeOnLattice(sys_bf, box);
    Initializer::placeOnLattice(sys_cl, box);

    CellList cells(forces.getCutoff(), 0.2, N, box);
    cells.rebuild(sys_cl, box);

    Integrator integ(1.0, 1e-4);
    integ.setKBT(1.0);

    const int T = 50;
    for (int step = 0; step < T; ++step) {
        // Brute force.
        RandomGenerator rng_bf(step + 100);
        forces.compute(sys_bf, box);
        integ.step(sys_bf, box, rng_bf);

        // Cell list.
        RandomGenerator rng_cl(step + 100);   // same seed each step
        if (cells.needsRebuild(sys_cl, box)) cells.rebuild(sys_cl, box);
        forces.compute(sys_cl, box, cells);
        integ.step(sys_cl, box, rng_cl);
    }

    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(sys_bf.getX(i), sys_cl.getX(i), 1e-12) << "i=" << i;
        EXPECT_NEAR(sys_bf.getY(i), sys_cl.getY(i), 1e-12) << "i=" << i;
    }
}

// ---- TrajectoryWriter HDF5 output -------------------------------------------
// These tests open the written HDF5 file with the H5Cpp API to verify the
// exact structure: group names, attribute names/types/values, dataset shape,
// and dataset values.

class TrajectoryWriterTest : public ::testing::Test {
protected:
    std::filesystem::path tempPath;

    void SetUp() override {
        tempPath = std::filesystem::temp_directory_path() / "bd_test_traj.h5";
        H5::Exception::dontPrint();
    }
    void TearDown() override {
        std::filesystem::remove(tempPath);
    }

    // Read a scalar uint64 attribute from any HDF5 object.
    static std::uint64_t readU64(H5::H5Object& obj, const std::string& name) {
        std::uint64_t v = 0;
        obj.openAttribute(name).read(H5::PredType::NATIVE_UINT64, &v);
        return v;
    }

    // Read a scalar double attribute.
    static double readDouble(H5::H5Object& obj, const std::string& name) {
        double v = 0.0;
        obj.openAttribute(name).read(H5::PredType::NATIVE_DOUBLE, &v);
        return v;
    }

    // Read a variable-length string attribute.
    static std::string readString(H5::H5Object& obj, const std::string& name) {
        H5::StrType st(H5::PredType::C_S1, H5T_VARIABLE);
        char* buf = nullptr;
        obj.openAttribute(name).read(st, &buf);
        std::string s(buf);
        H5free_memory(buf);
        return s;
    }
};

TEST_F(TrajectoryWriterTest, FileIsCreated) {
    System sys(4);
    Box    box(10.0);
    TrajectoryWriter writer(tempPath.string());
    writer.writeFrame(sys, box, 0);
    writer.close();
    EXPECT_TRUE(std::filesystem::exists(tempPath));
}

TEST_F(TrajectoryWriterTest, IsOpenAfterConstruct) {
    TrajectoryWriter writer(tempPath.string());
    EXPECT_TRUE(writer.isOpen());
}

TEST_F(TrajectoryWriterTest, IsClosedAfterClose) {
    TrajectoryWriter writer(tempPath.string());
    writer.close();
    EXPECT_FALSE(writer.isOpen());
}

TEST_F(TrajectoryWriterTest, FrameGroupExists) {
    System sys(2);
    Box    box(10.0);
    TrajectoryWriter writer(tempPath.string());
    writer.writeFrame(sys, box, 0);
    writer.close();

    H5::H5File file(tempPath.string(), H5F_ACC_RDONLY);
    EXPECT_TRUE(file.nameExists("/frame_00000000"));
}

TEST_F(TrajectoryWriterTest, FrameStepAttribute) {
    System sys(2);
    Box    box(10.0);
    TrajectoryWriter writer(tempPath.string());
    writer.writeFrame(sys, box, 42);
    writer.close();

    H5::H5File file(tempPath.string(), H5F_ACC_RDONLY);
    H5::Group  grp = file.openGroup("/frame_00000000");
    EXPECT_EQ(readU64(grp, "step"), std::uint64_t{42});
}

TEST_F(TrajectoryWriterTest, FrameTimeAttributeMatchesArgument) {
    // writeFrame(step, time): the time attribute should equal whatever the
    // caller passed in (no dt-based bookkeeping anymore).
    System sys(2);
    Box    box(10.0);

    TrajectoryWriter writer(tempPath.string());
    writer.writeFrame(sys, box, /*step*/ 500, /*time*/ 0.5);
    writer.close();

    H5::H5File file(tempPath.string(), H5F_ACC_RDONLY);
    H5::Group  grp = file.openGroup("/frame_00000000");
    EXPECT_DOUBLE_EQ(readDouble(grp, "time"), 0.5);
    EXPECT_EQ       (readU64   (grp, "step"), std::uint64_t{500});
}

TEST_F(TrajectoryWriterTest, FrameIndexAttribute) {
    System sys(2);
    Box    box(10.0);
    TrajectoryWriter writer(tempPath.string());
    writer.writeFrame(sys, box, 0);
    writer.writeFrame(sys, box, 1000);
    writer.close();

    H5::H5File file(tempPath.string(), H5F_ACC_RDONLY);
    H5::Group  g0 = file.openGroup("/frame_00000000");
    H5::Group  g1 = file.openGroup("/frame_00000001");
    EXPECT_EQ(readU64(g0, "frame"), std::uint64_t{0});
    EXPECT_EQ(readU64(g1, "frame"), std::uint64_t{1});
}

TEST_F(TrajectoryWriterTest, FrameBoxAttributes) {
    System sys(2);
    Box    box(7.0, 8.0);
    TrajectoryWriter writer(tempPath.string());
    writer.writeFrame(sys, box, 0);
    writer.close();

    H5::H5File file(tempPath.string(), H5F_ACC_RDONLY);
    H5::Group  grp = file.openGroup("/frame_00000000");
    EXPECT_DOUBLE_EQ(readDouble(grp, "Lx"), 7.0);
    EXPECT_DOUBLE_EQ(readDouble(grp, "Ly"), 8.0);
}

TEST_F(TrajectoryWriterTest, PositionsDatasetShape) {
    const std::size_t N = 5;
    System sys(N);
    Box    box(10.0);
    TrajectoryWriter writer(tempPath.string());
    writer.writeFrame(sys, box, 0);
    writer.close();

    H5::H5File  file(tempPath.string(), H5F_ACC_RDONLY);
    H5::Group   grp  = file.openGroup("/frame_00000000");
    H5::DataSet dset = grp.openDataSet("positions");
    H5::DataSpace sp = dset.getSpace();

    ASSERT_EQ(sp.getSimpleExtentNdims(), 2);
    hsize_t dims[2] = {};
    sp.getSimpleExtentDims(dims);
    EXPECT_EQ(dims[0], static_cast<hsize_t>(N));
    EXPECT_EQ(dims[1], static_cast<hsize_t>(2));
}

TEST_F(TrajectoryWriterTest, PositionsValuesCorrect) {
    const std::size_t N = 2;
    System sys(N);
    sys.setPosition(0, 1.5, 2.5);
    sys.setPosition(1, 3.0, 4.0);
    Box box(10.0);

    TrajectoryWriter writer(tempPath.string());
    writer.writeFrame(sys, box, 0);
    writer.close();

    H5::H5File  file(tempPath.string(), H5F_ACC_RDONLY);
    H5::Group   grp  = file.openGroup("/frame_00000000");
    H5::DataSet dset = grp.openDataSet("positions");

    std::vector<double> buf(2 * N);
    dset.read(buf.data(), H5::PredType::NATIVE_DOUBLE);

    EXPECT_DOUBLE_EQ(buf[0], 1.5);   // x[0]
    EXPECT_DOUBLE_EQ(buf[1], 2.5);   // y[0]
    EXPECT_DOUBLE_EQ(buf[2], 3.0);   // x[1]
    EXPECT_DOUBLE_EQ(buf[3], 4.0);   // y[1]
}

TEST_F(TrajectoryWriterTest, MultipleFramesCreateGroups) {
    System sys(2);
    Box    box(10.0);
    TrajectoryWriter writer(tempPath.string());
    for (std::size_t step = 0; step < 3; ++step)
        writer.writeFrame(sys, box, step * 1000);
    writer.close();

    H5::H5File file(tempPath.string(), H5F_ACC_RDONLY);
    EXPECT_TRUE(file.nameExists("/frame_00000000"));
    EXPECT_TRUE(file.nameExists("/frame_00000001"));
    EXPECT_TRUE(file.nameExists("/frame_00000002"));
    EXPECT_FALSE(file.nameExists("/frame_00000003"));
}

TEST_F(TrajectoryWriterTest, RootAttributesFromConfig) {
    System sys(2);
    Box    box(10.0);

    Config cfg;
    cfg.N         = 2;
    cfg.phi       = 0.2;
    cfg.Pe        = 5.0;
    cfg.De        = 0.5;
    cfg.R         = 2.0;
    cfg.C         = 0.05;
    cfg.delta     = 0.7;
    cfg.potential = PotentialType::SoftSphere;
    cfg.t_end       = 2.5;
    cfg.output_dt   = 0.05;
    cfg.abs_tol     = 5.0e-4;
    cfg.rel_tol     = 5.0e-3;
    cfg.dt_init     = 2.0e-3;
    cfg.output_file = "trajectory.h5";
    cfg.init_mode   = "lattice";
    cfg.seed        = 99;
    cfg.recompute();

    TrajectoryWriter writer(tempPath.string());
    writer.writeSimulationConfig(cfg);
    writer.writeFrame(sys, box, 0, 0.0);
    writer.close();

    H5::H5File file(tempPath.string(), H5F_ACC_RDONLY);

    // Inputs (the user-controlled parameter set).
    EXPECT_EQ       (readU64   (file, "N"),         std::uint64_t{2});
    EXPECT_DOUBLE_EQ(readDouble(file, "phi"),       0.2);
    EXPECT_DOUBLE_EQ(readDouble(file, "Pe"),        5.0);
    EXPECT_DOUBLE_EQ(readDouble(file, "De"),        0.5);
    EXPECT_DOUBLE_EQ(readDouble(file, "R"),         2.0);
    EXPECT_DOUBLE_EQ(readDouble(file, "C"),         0.05);
    EXPECT_DOUBLE_EQ(readDouble(file, "delta"),     0.7);
    EXPECT_EQ       (readString(file, "potential"), std::string{"soft_sphere"});

    // Frozen working units.
    EXPECT_DOUBLE_EQ(readDouble(file, "sigma"),     1.0);
    EXPECT_DOUBLE_EQ(readDouble(file, "epsilon"),   1.0);
    EXPECT_DOUBLE_EQ(readDouble(file, "gamma"),     1.0);

    // Derived microscopic params.
    EXPECT_DOUBLE_EQ(readDouble(file, "kT"),        0.05);    // C * epsilon
    EXPECT_DOUBLE_EQ(readDouble(file, "f0"),        0.3);     // SoftSphere: 1 - delta
    EXPECT_DOUBLE_EQ(readDouble(file, "tau_theta"), 5.0 / 0.3);
    EXPECT_DOUBLE_EQ(readDouble(file, "gamma_a"),   2.0);     // R * gamma

    // Adaptive integrator + I/O.
    EXPECT_DOUBLE_EQ(readDouble(file, "t_end"),       2.5);
    EXPECT_DOUBLE_EQ(readDouble(file, "output_dt"),   0.05);
    EXPECT_DOUBLE_EQ(readDouble(file, "abs_tol"),     5.0e-4);
    EXPECT_DOUBLE_EQ(readDouble(file, "rel_tol"),     5.0e-3);
    EXPECT_DOUBLE_EQ(readDouble(file, "dt_init"),     2.0e-3);
    EXPECT_EQ       (readString(file, "output_file"), std::string{"trajectory.h5"});
    EXPECT_EQ       (readString(file, "init_mode"),   std::string{"lattice"});
    EXPECT_EQ       (readU64   (file, "seed"),        std::uint64_t{99});
}
