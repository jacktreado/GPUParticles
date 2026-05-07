#pragma once

#include <cstddef>
#include <vector>

// =============================================================================
// System
// -----------------------------------------------------------------------------
// Holds the state of all N particles in a Structure-of-Arrays (SoA) layout:
//
//     positions:    x_[0..N)  , y_[0..N)
//     forces:      fx_[0..N)  , fy_[0..N)
//     orientations: theta_[0..N)           (active Brownian angle, radians)
//     velocities:  vx_[0..N)  , vy_[0..N)  (displacement/dt, set by Integrator)
//     anchors:     ax_[0..N)  , ay_[0..N)  (per-particle spring anchor positions)
//
// All same-quantity values are contiguous in memory, so a future GPU port can
// hand a raw pointer (xData(), yData(), ...) directly to a CUDA kernel and get
// coalesced loads/stores for free.
//
// The class deliberately exposes:
//   - whole-array getters (const and non-const) for kernels / bulk operations,
//   - per-index getters/setters for unit tests and easy scripting,
//   - raw-pointer accessors for the eventual CUDA migration.
// =============================================================================
class System {
public:
    System() = default;
    explicit System(std::size_t N);

    // ---- Lifecycle -----------------------------------------------------------
    void resize(std::size_t N);
    void zeroForces();
    void zeroVelocities();

    // ---- Scalar properties ---------------------------------------------------
    std::size_t getNumParticles() const { return N_; }

    double getSigma() const { return sigma_; }
    void   setSigma(double s) { sigma_ = s; }

    // ---- Whole-array access (const) -- prefer these in performance code -----
    const std::vector<double>& getPositionsX()   const { return x_;     }
    const std::vector<double>& getPositionsY()   const { return y_;     }
    const std::vector<double>& getForcesX()      const { return fx_;    }
    const std::vector<double>& getForcesY()      const { return fy_;    }
    const std::vector<double>& getOrientations() const { return theta_; }
    const std::vector<double>& getVelocitiesX()  const { return vx_;    }
    const std::vector<double>& getVelocitiesY()  const { return vy_;    }
    const std::vector<double>& getAnchorsX()     const { return ax_;    }
    const std::vector<double>& getAnchorsY()     const { return ay_;    }

    // ---- Whole-array access (mutable) ---------------------------------------
    std::vector<double>& getPositionsX()   { return x_;     }
    std::vector<double>& getPositionsY()   { return y_;     }
    std::vector<double>& getForcesX()      { return fx_;    }
    std::vector<double>& getForcesY()      { return fy_;    }
    std::vector<double>& getOrientations() { return theta_; }
    std::vector<double>& getVelocitiesX()  { return vx_;    }
    std::vector<double>& getVelocitiesY()  { return vy_;    }
    std::vector<double>& getAnchorsX()     { return ax_;    }
    std::vector<double>& getAnchorsY()     { return ay_;    }

    // ---- Per-particle accessors (handy for tests / debugging) ---------------
    double getX    (std::size_t i) const { return x_[i];     }
    double getY    (std::size_t i) const { return y_[i];     }
    double getFx   (std::size_t i) const { return fx_[i];    }
    double getFy   (std::size_t i) const { return fy_[i];    }
    double getTheta(std::size_t i) const { return theta_[i]; }
    double getVx   (std::size_t i) const { return vx_[i];    }
    double getVy   (std::size_t i) const { return vy_[i];    }
    double getAx   (std::size_t i) const { return ax_[i];    }
    double getAy   (std::size_t i) const { return ay_[i];    }

    void setX    (std::size_t i, double v) { x_[i]     = v; }
    void setY    (std::size_t i, double v) { y_[i]     = v; }
    void setFx   (std::size_t i, double v) { fx_[i]    = v; }
    void setFy   (std::size_t i, double v) { fy_[i]    = v; }
    void setTheta(std::size_t i, double v) { theta_[i] = v; }
    void setVx   (std::size_t i, double v) { vx_[i]    = v; }
    void setVy   (std::size_t i, double v) { vy_[i]    = v; }
    void setAx   (std::size_t i, double v) { ax_[i]    = v; }
    void setAy   (std::size_t i, double v) { ay_[i]    = v; }

    void setPosition(std::size_t i, double xv, double yv) { x_[i]  = xv; y_[i]  = yv; }
    void setForce   (std::size_t i, double fx, double fy) { fx_[i] = fx; fy_[i] = fy; }
    void setAnchor  (std::size_t i, double xv, double yv) { ax_[i] = xv; ay_[i] = yv; }

    // ---- Raw pointer access (the GPU on-ramp) -------------------------------
    // Once we add a CUDA backend, these pointers can be replaced by device
    // pointers and the rest of the code is largely untouched.
    double*       xData()         { return x_.data();     }
    double*       yData()         { return y_.data();     }
    double*       fxData()        { return fx_.data();    }
    double*       fyData()        { return fy_.data();    }
    double*       thetaData()     { return theta_.data(); }
    double*       vxData()        { return vx_.data();    }
    double*       vyData()        { return vy_.data();    }
    double*       axData()        { return ax_.data();    }
    double*       ayData()        { return ay_.data();    }
    const double* xData()         const { return x_.data();     }
    const double* yData()         const { return y_.data();     }
    const double* fxData()        const { return fx_.data();    }
    const double* fyData()        const { return fy_.data();    }
    const double* thetaData()     const { return theta_.data(); }
    const double* vxData()        const { return vx_.data();    }
    const double* vyData()        const { return vy_.data();    }
    const double* axData()        const { return ax_.data();    }
    const double* ayData()        const { return ay_.data();    }

private:
    std::size_t N_ = 0;
    double sigma_ = 1.0;          // particle diameter

    // SoA storage. Layout chosen so that traversing one component for all
    // particles produces unit-stride memory access (cache-friendly on CPU,
    // coalesced on GPU).
    std::vector<double> x_;
    std::vector<double> y_;
    std::vector<double> fx_;
    std::vector<double> fy_;
    std::vector<double> theta_;   // orientation angle (radians), for active Brownian motion
    std::vector<double> vx_;      // instantaneous velocity x = Δx/dt (set by Integrator)
    std::vector<double> vy_;      // instantaneous velocity y = Δy/dt
    std::vector<double> ax_;      // anchor x position (spring tether)
    std::vector<double> ay_;      // anchor y position
};
