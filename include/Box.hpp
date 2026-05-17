#pragma once

#include <cmath>

// =============================================================================
// Box
// -----------------------------------------------------------------------------
// A rectangular 2D simulation cell with periodic boundary conditions.
//
//   Lx_, Ly_  side lengths
//   origin    (0, 0)
//
// Position contract:
//   Stored particle and anchor coordinates are NOT auto-wrapped by the engine.
//   They diffuse freely over all of R^2 so that downstream analyses (MSD,
//   intermediate scattering functions, etc.) see true unwrapped trajectories.
//
//   Pair forces, spring forces, and cell-list rebuild triggers all operate on
//   displacement vectors via minimumImage(), which is image-invariant. The
//   wrap() helper below remains available for the few callers that need a
//   primary-image copy (cell-list binning, initialization, display).
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
    // Not called by the integrator — only by callers that need a primary-image
    // copy (cell-list binning, initialization, display).
    inline void wrap(double& x, double& y) const {
        x -= Lx_ * std::floor(x / Lx_);
        y -= Ly_ * std::floor(y / Ly_);
    }

private:
    double Lx_ = 0.0;
    double Ly_ = 0.0;
};
