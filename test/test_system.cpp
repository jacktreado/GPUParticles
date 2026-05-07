#include "System.hpp"

#include <gtest/gtest.h>
#include <cstddef>

// =============================================================================
// System unit tests
// =============================================================================

// ---- Construction -----------------------------------------------------------

TEST(SystemTest, DefaultConstructorIsEmpty) {
    System s;
    EXPECT_EQ(s.getNumParticles(), std::size_t{0});
}

TEST(SystemTest, SizedConstructorSetsN) {
    System s(64);
    EXPECT_EQ(s.getNumParticles(), std::size_t{64});
}

TEST(SystemTest, SizedConstructorZeroInitializes) {
    System s(10);
    for (std::size_t i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(s.getX(i),  0.0) << "i=" << i;
        EXPECT_DOUBLE_EQ(s.getY(i),  0.0) << "i=" << i;
        EXPECT_DOUBLE_EQ(s.getFx(i), 0.0) << "i=" << i;
        EXPECT_DOUBLE_EQ(s.getFy(i), 0.0) << "i=" << i;
    }
}

// ---- resize -----------------------------------------------------------------

TEST(SystemTest, ResizeChangesN) {
    System s(10);
    s.resize(30);
    EXPECT_EQ(s.getNumParticles(), std::size_t{30});
}

TEST(SystemTest, ResizeZeroInitializes) {
    System s(5);
    s.resize(8);
    for (std::size_t i = 0; i < 8; ++i) {
        EXPECT_DOUBLE_EQ(s.getX(i),  0.0);
        EXPECT_DOUBLE_EQ(s.getY(i),  0.0);
        EXPECT_DOUBLE_EQ(s.getFx(i), 0.0);
        EXPECT_DOUBLE_EQ(s.getFy(i), 0.0);
    }
}

// ---- sigma ------------------------------------------------------------------

TEST(SystemTest, DefaultSigmaIsOne) {
    System s;
    EXPECT_DOUBLE_EQ(s.getSigma(), 1.0);
}

TEST(SystemTest, SetSigmaRoundTrip) {
    System s;
    s.setSigma(2.5);
    EXPECT_DOUBLE_EQ(s.getSigma(), 2.5);
}

// ---- Per-particle setters / getters -----------------------------------------

TEST(SystemTest, SetPositionGetX) {
    System s(4);
    s.setX(2, 3.14);
    EXPECT_DOUBLE_EQ(s.getX(2), 3.14);
}

TEST(SystemTest, SetPositionGetY) {
    System s(4);
    s.setY(1, 2.71);
    EXPECT_DOUBLE_EQ(s.getY(1), 2.71);
}

TEST(SystemTest, SetForceGetFx) {
    System s(4);
    s.setFx(0, -1.5);
    EXPECT_DOUBLE_EQ(s.getFx(0), -1.5);
}

TEST(SystemTest, SetForceGetFy) {
    System s(4);
    s.setFy(3, 7.0);
    EXPECT_DOUBLE_EQ(s.getFy(3), 7.0);
}

TEST(SystemTest, SetPositionTwoArg) {
    System s(4);
    s.setPosition(1, 2.0, 5.0);
    EXPECT_DOUBLE_EQ(s.getX(1), 2.0);
    EXPECT_DOUBLE_EQ(s.getY(1), 5.0);
}

TEST(SystemTest, SetForceTwoArg) {
    System s(4);
    s.setForce(3, 1.1, -2.2);
    EXPECT_DOUBLE_EQ(s.getFx(3),  1.1);
    EXPECT_DOUBLE_EQ(s.getFy(3), -2.2);
}

// ---- zeroForces -------------------------------------------------------------

TEST(SystemTest, ZeroForcesDoesNotTouchPositions) {
    System s(4);
    s.setPosition(0, 1.0, 2.0);
    s.setPosition(1, 3.0, 4.0);
    s.setForce(0, 5.0, 6.0);

    s.zeroForces();

    EXPECT_DOUBLE_EQ(s.getFx(0), 0.0);
    EXPECT_DOUBLE_EQ(s.getFy(0), 0.0);
    // Positions untouched.
    EXPECT_DOUBLE_EQ(s.getX(0), 1.0);
    EXPECT_DOUBLE_EQ(s.getY(0), 2.0);
    EXPECT_DOUBLE_EQ(s.getX(1), 3.0);
    EXPECT_DOUBLE_EQ(s.getY(1), 4.0);
}

TEST(SystemTest, ZeroForcesAllParticles) {
    const std::size_t N = 32;
    System s(N);
    for (std::size_t i = 0; i < N; ++i) s.setForce(i, 1.0, 1.0);

    s.zeroForces();

    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(s.getFx(i), 0.0) << "i=" << i;
        EXPECT_DOUBLE_EQ(s.getFy(i), 0.0) << "i=" << i;
    }
}

// ---- Raw-pointer accessors --------------------------------------------------

TEST(SystemTest, XDataPointsToVectorData) {
    System s(8);
    EXPECT_EQ(s.xData(),  s.getPositionsX().data());
    EXPECT_EQ(s.yData(),  s.getPositionsY().data());
    EXPECT_EQ(s.fxData(), s.getForcesX().data());
    EXPECT_EQ(s.fyData(), s.getForcesY().data());
}

TEST(SystemTest, ConstXDataPointsToVectorData) {
    const System s(8);
    EXPECT_EQ(s.xData(),  s.getPositionsX().data());
    EXPECT_EQ(s.yData(),  s.getPositionsY().data());
    EXPECT_EQ(s.fxData(), s.getForcesX().data());
    EXPECT_EQ(s.fyData(), s.getForcesY().data());
}

TEST(SystemTest, RawPointerWriteVisibleThroughGetter) {
    System s(4);
    s.xData()[2] = 9.9;
    EXPECT_DOUBLE_EQ(s.getX(2), 9.9);
}

// ---- Whole-array getters ----------------------------------------------------

TEST(SystemTest, WholeArraySizeMatchesN) {
    const std::size_t N = 16;
    System s(N);
    EXPECT_EQ(s.getPositionsX().size(), N);
    EXPECT_EQ(s.getPositionsY().size(), N);
    EXPECT_EQ(s.getForcesX().size(),    N);
    EXPECT_EQ(s.getForcesY().size(),    N);
}

TEST(SystemTest, MutableWholeArrayGetterAllowsWrite) {
    System s(4);
    s.getPositionsX()[1] = 42.0;
    EXPECT_DOUBLE_EQ(s.getX(1), 42.0);
}
