#include "ForceCalculator.hpp"

#include "Box.hpp"
#include "CellList.hpp"
#include "System.hpp"

#include <cmath>
#include <cstdint>

// =============================================================================
// Pair-potential primitives
// -----------------------------------------------------------------------------
// Each pair of (force, energy) functions is stateless: every quantity they
// need is passed in explicitly. The force function returns f_over_r2 = F/r so
// the caller writes (fx, fy) -= f_over_r2 * (dx, dy).
// =============================================================================

void ForceCalculator::wcaForce(double r2, double epsilon, double /*sigma*/,
                               double sigma2, double& f_over_r2) {
    // F(r) = 24 eps / r * sr6 * (2 sr6 - 1), so F/r = 24 eps / r^2 * sr6 * (2 sr6 - 1).
    const double sr2 = sigma2 / r2;
    const double sr6 = sr2 * sr2 * sr2;
    f_over_r2 = 24.0 * epsilon * sr6 * (2.0 * sr6 - 1.0) / r2;
}

double ForceCalculator::wcaEnergy(double r2, double epsilon, double /*sigma*/,
                                  double sigma2) {
    // U(r) = 4 eps [(sigma/r)^12 - (sigma/r)^6] + eps     for r < r_cut
    const double sr2  = sigma2 / r2;
    const double sr6  = sr2 * sr2 * sr2;
    const double sr12 = sr6 * sr6;
    return 4.0 * epsilon * (sr12 - sr6) + epsilon;
}

void ForceCalculator::softSphereForce(double r2, double epsilon, double sigma,
                                      double /*sigma2*/, double& f_over_r2) {
    // U(r)   = (eps/2) * (1 - r/sigma)^2          for r < sigma
    // F(r)   = -dU/dr = (eps/sigma) * (1 - r/sigma)
    // F/r    = (eps/sigma) * (1 - r/sigma) / r
    const double r         = std::sqrt(r2);
    const double inv_sigma = 1.0 / sigma;
    const double s         = 1.0 - r * inv_sigma;
    f_over_r2 = epsilon * inv_sigma * s / r;
}

double ForceCalculator::softSphereEnergy(double r2, double epsilon, double sigma,
                                         double /*sigma2*/) {
    const double r = std::sqrt(r2);
    const double s = 1.0 - r / sigma;
    return 0.5 * epsilon * s * s;
}

double ForceCalculator::f0ForOverlap(PotentialType type, double delta,
                                     double epsilon, double sigma) {
    if (delta <= 0.0) return 0.0;             // unphysical / singular
    const double r      = delta * sigma;
    const double r2     = r * r;
    const double sigma2 = sigma * sigma;

    // Cutoff differs per potential; outside it the force is identically zero,
    // which means the active drive needed to maintain contact is zero too —
    // i.e. delta beyond the cutoff is the passive limit, f0 = 0.
    const double r_cut = (type == PotentialType::WCA)
                             ? std::pow(2.0, 1.0 / 6.0) * sigma
                             : sigma;
    if (r >= r_cut) return 0.0;

    double f_over_r2 = 0.0;
    switch (type) {
        case PotentialType::WCA:
            wcaForce(r2, epsilon, sigma, sigma2, f_over_r2);
            break;
        case PotentialType::SoftSphere:
            softSphereForce(r2, epsilon, sigma, sigma2, f_over_r2);
            break;
    }
    // The pair fns return F/r (so callers can write force -= (F/r) * dx).
    // Multiply by r to recover the radial-force magnitude that the active
    // self-propulsion must match.
    return f_over_r2 * r;
}

// =============================================================================
// Construction / configuration
// =============================================================================

ForceCalculator::ForceCalculator(double eps, double sig, PotentialType type)
    : epsilon_(eps), sigma_(sig), potential_type_(type) {
    installPotentialFns();
    recomputeCachedConstants();
}

void ForceCalculator::installPotentialFns() {
    switch (potential_type_) {
        case PotentialType::WCA:
            pair_force_fn_  = &ForceCalculator::wcaForce;
            pair_energy_fn_ = &ForceCalculator::wcaEnergy;
            r_cut_factor_   = std::pow(2.0, 1.0 / 6.0);
            break;
        case PotentialType::SoftSphere:
            pair_force_fn_  = &ForceCalculator::softSphereForce;
            pair_energy_fn_ = &ForceCalculator::softSphereEnergy;
            r_cut_factor_   = 1.0;
            break;
    }
}

void ForceCalculator::recomputeCachedConstants() {
    r_cut_  = r_cut_factor_ * sigma_;
    r_cut2_ = r_cut_ * r_cut_;
    sigma2_ = sigma_ * sigma_;
}

void ForceCalculator::setEpsilon(double eps) {
    epsilon_ = eps;
}

void ForceCalculator::setSigma(double sig) {
    sigma_ = sig;
    recomputeCachedConstants();
}

void ForceCalculator::setPotentialType(PotentialType t) {
    potential_type_ = t;
    installPotentialFns();
    recomputeCachedConstants();
}

// =============================================================================
// Force / energy kernels
// =============================================================================

void ForceCalculator::compute(System& sys, const Box& box) const {
    sys.zeroForces();

    const std::size_t N = sys.getNumParticles();
    const double* __restrict__ x  = sys.xData();
    const double* __restrict__ y  = sys.yData();
    double*       __restrict__ fx = sys.fxData();
    double*       __restrict__ fy = sys.fyData();

    const PairForceFn pair_fn = pair_force_fn_;
    const double      eps     = epsilon_;
    const double      sig     = sigma_;
    const double      sig2    = sigma2_;
    const double      rc2     = r_cut2_;

    // ---------------------------------------------------------------------
    // Outer loop: over particles i. Each iteration is INDEPENDENT — it only
    // reads x[*]/y[*] and writes fx[i]/fy[i]. This is the loop that becomes
    // one CUDA thread per i in the GPU port.
    //
    // Inner loop: O(N) brute force. The pair function is called through a
    // function pointer so the same kernel handles every potential.
    // ---------------------------------------------------------------------
    for (std::size_t i = 0; i < N; ++i) {
        const double xi = x[i];
        const double yi = y[i];
        double fxi = 0.0;
        double fyi = 0.0;

        for (std::size_t j = 0; j < N; ++j) {
            if (j == i) continue;

            double dx = x[j] - xi;
            double dy = y[j] - yi;
            box.minimumImage(dx, dy);

            const double r2 = dx * dx + dy * dy;
            if (r2 >= rc2 || r2 == 0.0) continue;

            double f_over_r2;
            pair_fn(r2, eps, sig, sig2, f_over_r2);

            fxi -= f_over_r2 * dx;
            fyi -= f_over_r2 * dy;
        }

        fx[i] = fxi;
        fy[i] = fyi;
    }
}

void ForceCalculator::compute(System& sys, const Box& box, const CellList& cl) const {
    sys.zeroForces();

    const std::size_t N = sys.getNumParticles();
    const double* __restrict__ x  = sys.xData();
    const double* __restrict__ y  = sys.yData();
    double*       __restrict__ fx = sys.fxData();
    double*       __restrict__ fy = sys.fyData();

    const std::int32_t* __restrict__ nlist       = cl.nlistData();
    const std::int32_t* __restrict__ nlist_start = cl.nlistStartData();

    const PairForceFn pair_fn = pair_force_fn_;
    const double      eps     = epsilon_;
    const double      sig     = sigma_;
    const double      sig2    = sigma2_;
    const double      rc2     = r_cut2_;

    for (std::size_t i = 0; i < N; ++i) {
        const double xi = x[i];
        const double yi = y[i];
        double fxi = 0.0;
        double fyi = 0.0;

        const std::int32_t kbeg = nlist_start[i];
        const std::int32_t kend = nlist_start[i + 1];
        for (std::int32_t k = kbeg; k < kend; ++k) {
            const std::int32_t j = nlist[k];

            double dx = x[j] - xi;
            double dy = y[j] - yi;
            box.minimumImage(dx, dy);

            const double r2 = dx * dx + dy * dy;
            if (r2 >= rc2 || r2 == 0.0) continue;

            double f_over_r2;
            pair_fn(r2, eps, sig, sig2, f_over_r2);

            fxi -= f_over_r2 * dx;
            fyi -= f_over_r2 * dy;
        }

        fx[i] = fxi;
        fy[i] = fyi;
    }
}

double ForceCalculator::computeEnergy(const System& sys, const Box& box) const {
    const std::size_t N = sys.getNumParticles();
    const double* x = sys.xData();
    const double* y = sys.yData();

    const PairEnergyFn pair_fn = pair_energy_fn_;
    const double       eps     = epsilon_;
    const double       sig     = sigma_;
    const double       sig2    = sigma2_;
    const double       rc2     = r_cut2_;

    double U = 0.0;
    // Pair-once ordering j > i for energy. We *don't* do this in compute()
    // for the GPU-readiness reasons explained in the header, but for energy
    // (a single scalar reduction) it's harmless and saves 2x.
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i + 1; j < N; ++j) {
            double dx = x[j] - x[i];
            double dy = y[j] - y[i];
            box.minimumImage(dx, dy);

            const double r2 = dx * dx + dy * dy;
            if (r2 >= rc2 || r2 == 0.0) continue;

            U += pair_fn(r2, eps, sig, sig2);
        }
    }
    return U;
}
