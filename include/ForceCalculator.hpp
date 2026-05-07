#pragma once

#include <cmath>

class System;
class Box;
class CellList;

// =============================================================================
// PotentialType
// -----------------------------------------------------------------------------
// Tag for the pluggable pair potential. Each enumerator maps to a pair of
// static functions on ForceCalculator (force + energy) and a fixed cutoff
// factor (cutoff = factor * sigma).
//
//   WCA         purely-repulsive Lennard-Jones, r_cut = 2^(1/6) * sigma
//                 U(r) = 4 eps [(sigma/r)^12 - (sigma/r)^6] + eps
//   SoftSphere  harmonic soft sphere,           r_cut = sigma
//                 U(r) = (eps/2) * (1 - r/sigma)^2     for r < sigma
// =============================================================================
enum class PotentialType {
    WCA,
    SoftSphere
};

// =============================================================================
// ForceCalculator
// -----------------------------------------------------------------------------
// Computes pair forces using a runtime-selectable pair potential. The choice
// is stored as a pair of function pointers (one for force, one for energy)
// installed when the potential is set; the inner force loop dispatches
// through these pointers. Adding a new potential = write two static functions
// + register a case in installPotentialFns().
//
// IMPORTANT IMPLEMENTATION NOTE (GPU readiness):
// The compute() method computes each particle's force *independently* from
// scratch. We DO NOT use Newton's third law (F_ji = -F_ij) to halve the work,
// because that requires accumulating into two output slots per pair, which
// either (a) makes the loop sequential, or (b) requires atomic adds on GPU.
// Production GPU MD codes (HOOMD, LAMMPS-KOKKOS) use exactly this 2x-redundant
// pattern for the same reason — it's worth the extra arithmetic to keep every
// thread's writes fully independent.
// =============================================================================
class ForceCalculator {
public:
    // Pair-potential primitives. Stateless: every quantity they need is
    // passed in. f_over_r2 is F_radial(r) / r so the caller writes
    // (fx, fy) -= f_over_r2 * (dx, dy) and gets the correct vector force.
    using PairForceFn  = void  (*)(double r2, double epsilon, double sigma,
                                   double sigma2, double& f_over_r2);
    using PairEnergyFn = double(*)(double r2, double epsilon, double sigma,
                                   double sigma2);

    ForceCalculator() = default;
    ForceCalculator(double epsilon, double sigma,
                    PotentialType type = PotentialType::WCA);

    // Compute forces on all particles, brute-force O(N^2). Resets forces first.
    // Kept as both (a) the verification reference for the cell-list overload
    // and (b) the small-box fallback when CellList::useBruteForce() is true.
    void compute(System& sys, const Box& box) const;

    // Compute forces using a prebuilt Verlet neighbor list (CSR layout in cl).
    // Resets forces first. The caller is responsible for ensuring the list is
    // up to date — see CellList::needsRebuild() / CellList::rebuild().
    void compute(System& sys, const Box& box, const CellList& cl) const;

    // Total potential energy (each pair counted once). O(N^2); only called
    // every output_every steps so routing it through the Verlet list buys
    // nothing at this call frequency.
    double computeEnergy(const System& sys, const Box& box) const;

    // ---- Getters ------------------------------------------------------------
    double        getEpsilon()       const { return epsilon_; }
    double        getSigma()         const { return sigma_;   }
    double        getCutoff()        const { return r_cut_;   }
    double        getCutoffSquared() const { return r_cut2_;  }
    PotentialType getPotentialType() const { return potential_type_; }

    // ---- Setters (re-derive cached cutoff) ----------------------------------
    void setEpsilon(double eps);
    void setSigma  (double sig);
    void setPotentialType(PotentialType t);

    // ---- Pair primitives (exposed for unit testing) -------------------------
    static void   wcaForce       (double r2, double epsilon, double sigma,
                                  double sigma2, double& f_over_r2);
    static double wcaEnergy      (double r2, double epsilon, double sigma,
                                  double sigma2);
    static void   softSphereForce(double r2, double epsilon, double sigma,
                                  double sigma2, double& f_over_r2);
    static double softSphereEnergy(double r2, double epsilon, double sigma,
                                   double sigma2);

    // Active force needed to hold a steady-state pair contact at separation
    // r* = delta * sigma, from the force balance F_pot(r*) = f0 (two ABPs
    // pressing into each other with persistent self-propulsion). Returns 0
    // when delta is at or beyond the cutoff (passive limit).
    //
    // Closed forms with epsilon = sigma = 1 (the conventional units):
    //   WCA         f0 = 24 * delta^-7 * (2 delta^-6 - 1)   for 0 < delta < 2^(1/6)
    //   SoftSphere  f0 = 1 - delta                          for 0 < delta < 1
    //
    // See README "Typical pair overlap" for the derivation.
    static double f0ForOverlap(PotentialType type, double delta,
                               double epsilon, double sigma);

private:
    double epsilon_ = 1.0;
    double sigma_   = 1.0;

    // The default-constructed object is a WCA potential at sigma=1, epsilon=1
    // but with r_cut_ left at 0 (recomputeCachedConstants() is only called by
    // the parametric ctor / setSigma / setPotentialType). The default cutoff
    // factor matches the default function pointers so that a later setSigma()
    // produces the WCA cutoff without needing to call setPotentialType first.
    PotentialType potential_type_ = PotentialType::WCA;
    double        r_cut_factor_   = std::pow(2.0, 1.0 / 6.0);
    PairForceFn   pair_force_fn_  = &ForceCalculator::wcaForce;
    PairEnergyFn  pair_energy_fn_ = &ForceCalculator::wcaEnergy;

    // Cached, derived quantities — recomputed when sigma or potential changes.
    double r_cut_  = 0.0;
    double r_cut2_ = 0.0;
    double sigma2_ = 0.0;

    void recomputeCachedConstants();
    void installPotentialFns();
};
