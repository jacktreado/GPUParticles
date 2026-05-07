#include "Box.hpp"

#include <gtest/gtest.h>
#include <cmath>

// =============================================================================
// Box unit tests
// =============================================================================

// ---- Construction -----------------------------------------------------------

TEST(BoxTest, DefaultConstructor) {
    Box b;
    EXPECT_EQ(b.getLx(), 0.0);
    EXPECT_EQ(b.getLy(), 0.0);
}

TEST(BoxTest, SquareConstructor) {
    Box b(10.0);
    EXPECT_DOUBLE_EQ(b.getLx(), 10.0);
    EXPECT_DOUBLE_EQ(b.getLy(), 10.0);
    EXPECT_DOUBLE_EQ(b.getArea(), 100.0);
}

TEST(BoxTest, RectangularConstructor) {
    Box b(4.0, 6.0);
    EXPECT_DOUBLE_EQ(b.getLx(), 4.0);
    EXPECT_DOUBLE_EQ(b.getLy(), 6.0);
    EXPECT_DOUBLE_EQ(b.getArea(), 24.0);
}

TEST(BoxTest, Setters) {
    Box b;
    b.setLx(3.0);
    b.setLy(5.0);
    EXPECT_DOUBLE_EQ(b.getLx(), 3.0);
    EXPECT_DOUBLE_EQ(b.getLy(), 5.0);

    b.setSize(7.0, 11.0);
    EXPECT_DOUBLE_EQ(b.getLx(), 7.0);
    EXPECT_DOUBLE_EQ(b.getLy(), 11.0);
}

// ---- minimumImage -----------------------------------------------------------

// With L = 10, dx values and expected minimum images:
//   dx =  4.0  ->  4.0  (|4| < L/2, no change)
//   dx =  5.5  -> -4.5  (shifted by -L: nearbyint(5.5/10)=1)
//   dx = -5.5  ->  4.5  (shifted by +L: nearbyint(-5.5/10)=-1)
//   dx = -4.0  -> -4.0  (no change)
//   dx =  0.0  ->  0.0

TEST(BoxTest, MinimumImageNoShift) {
    Box b(10.0);
    double dx = 4.0, dy = 0.0;
    b.minimumImage(dx, dy);
    EXPECT_DOUBLE_EQ(dx,  4.0);
    EXPECT_DOUBLE_EQ(dy,  0.0);
}

TEST(BoxTest, MinimumImagePositiveShift) {
    Box b(10.0);
    double dx = 5.5, dy = 0.0;
    b.minimumImage(dx, dy);
    EXPECT_DOUBLE_EQ(dx, -4.5);
}

TEST(BoxTest, MinimumImageNegativeShift) {
    Box b(10.0);
    double dx = -5.5, dy = 0.0;
    b.minimumImage(dx, dy);
    EXPECT_DOUBLE_EQ(dx,  4.5);
}

TEST(BoxTest, MinimumImageNegativeNoShift) {
    Box b(10.0);
    double dx = -4.0, dy = 0.0;
    b.minimumImage(dx, dy);
    EXPECT_DOUBLE_EQ(dx, -4.0);
}

TEST(BoxTest, MinimumImageZero) {
    Box b(10.0);
    double dx = 0.0, dy = 0.0;
    b.minimumImage(dx, dy);
    EXPECT_DOUBLE_EQ(dx, 0.0);
    EXPECT_DOUBLE_EQ(dy, 0.0);
}

TEST(BoxTest, MinimumImageBothComponents) {
    Box b(10.0, 20.0);
    double dx = 5.5, dy = 12.0;
    b.minimumImage(dx, dy);
    EXPECT_DOUBLE_EQ(dx, -4.5);        // 5.5 - 10 = -4.5
    EXPECT_DOUBLE_EQ(dy, -8.0);        // 12.0 - 20 = -8.0
}

TEST(BoxTest, MinimumImageDistanceIsAlwaysMinimal) {
    Box b(10.0);
    // The minimum-image distance must always be <= L/2.
    for (double sep = 0.01; sep < 10.0; sep += 0.1) {
        double dx = sep, dy = 0.0;
        b.minimumImage(dx, dy);
        EXPECT_LE(std::abs(dx), 5.0) << "sep=" << sep;
    }
}

// ---- wrap -------------------------------------------------------------------

TEST(BoxTest, WrapNoChange) {
    Box b(10.0);
    double x = 5.0, y = 3.0;
    b.wrap(x, y);
    EXPECT_DOUBLE_EQ(x, 5.0);
    EXPECT_DOUBLE_EQ(y, 3.0);
}

TEST(BoxTest, WrapPositiveOverflow) {
    Box b(10.0);
    double x = 10.5, y = 0.0;
    b.wrap(x, y);
    EXPECT_DOUBLE_EQ(x, 0.5);
}

TEST(BoxTest, WrapNegativeCoordinate) {
    Box b(10.0);
    double x = -0.5, y = 0.0;
    b.wrap(x, y);
    EXPECT_DOUBLE_EQ(x, 9.5);
}

TEST(BoxTest, WrapExactlyAtBoundary) {
    Box b(10.0);
    // x = 10.0 is out of [0,10) — floor(10/10)=1, so x becomes 0.
    double x = 10.0, y = 0.0;
    b.wrap(x, y);
    EXPECT_DOUBLE_EQ(x, 0.0);
}

TEST(BoxTest, WrapBothComponents) {
    Box b(10.0, 20.0);
    double x = 15.0, y = -1.0;
    b.wrap(x, y);
    EXPECT_DOUBLE_EQ(x,  5.0);
    EXPECT_DOUBLE_EQ(y, 19.0);
}

TEST(BoxTest, WrapResultAlwaysInPrimaryCell) {
    Box b(10.0);
    for (double v = -25.0; v <= 35.0; v += 0.37) {
        double x = v, y = 0.0;
        b.wrap(x, y);
        EXPECT_GE(x, 0.0)  << "v=" << v;
        EXPECT_LT(x, 10.0) << "v=" << v;
    }
}

// ---- Round-trip consistency -------------------------------------------------

TEST(BoxTest, MinimumImageConsistentWithWrap) {
    // Wrapping two positions into the primary cell and then computing their
    // separation should match minimumImage applied to the raw difference.
    Box b(10.0);
    double x1 = 1.5, y1 = 8.5;
    double x2 = 9.0, y2 = 0.5;

    double dxRaw = x2 - x1;
    double dyRaw = y2 - y1;
    b.minimumImage(dxRaw, dyRaw);

    // Force x1,x2 into [0,L) first (they already are, but let's be explicit).
    b.wrap(x1, y1);
    b.wrap(x2, y2);
    double dxWrapped = x2 - x1;
    double dyWrapped = y2 - y1;
    b.minimumImage(dxWrapped, dyWrapped);

    EXPECT_DOUBLE_EQ(dxRaw, dxWrapped);
    EXPECT_DOUBLE_EQ(dyRaw, dyWrapped);
}
