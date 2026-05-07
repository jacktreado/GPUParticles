#pragma once

#include <cmath>

// =============================================================================
// Box
// -----------------------------------------------------------------------------
// A rectangular 2D simulation cell with periodic boundary conditions.
//
//   Lx_, Ly_  side lengths
//   origin    (0, 0); positions are stored in [0, Lx) x [0, Ly)
//
// Methods are inline because they will be hammered every step, by every
// particle, by every pair. minimumImage() is the most-called function in the
// whole engine — it MUST be inlined and branch-free.
// =============================================================================
class Box {
public:
    Box() = default;
    explicit Box(double L)              : Lx_(L), Ly_(L)  {}
    Box(double Lx, double Ly)           : Lx_(Lx), Ly_(Ly) {}

    // ---- Getters / setters --------------------------------------------------
    double getLx()   const { return Lx_; }
    double getLy()   const { return Ly_; }
    double getArea() const { return Lx_ * Ly_; }

    void setLx(double L) { Lx_ = L; }
    void setLy(double L) { Ly_ = L; }
    void setSize(double Lx, double Ly) { Lx_ = Lx; Ly_ = Ly; }

    // ---- Core periodic-boundary operations ----------------------------------
    // Apply the minimum-image convention to a separation vector (dx, dy).
    // After the call, the components are the shortest periodic image.
    inline void minimumImage(double& dx, double& dy) const {
        dx -= Lx_ * std::nearbyint(dx / Lx_);
        dy -= Ly_ * std::nearbyint(dy / Ly_);
    }

    // Wrap a position into the primary cell [0, L) x [0, L).
    // std::floor handles negative values correctly (e.g. floor(-0.3) == -1).
    inline void wrap(double& x, double& y) const {
        x -= Lx_ * std::floor(x / Lx_);
        y -= Ly_ * std::floor(y / Ly_);
    }

private:
    double Lx_ = 0.0;
    double Ly_ = 0.0;
};
