#pragma once

#include <cstdint>
#include <random>

// =============================================================================
// RandomGenerator
// -----------------------------------------------------------------------------
// Thin wrapper around std::mt19937_64 + std::normal_distribution + uniform.
// The wrapper buys us:
//   - one consistent seeding interface,
//   - the option to swap in a different engine (Philox, etc.) without
//     changing call sites,
//   - a clean "RandomGenerator" type to thread through the integrator.
//
// GPU NOTE: a host-side Mersenne Twister cannot be used inside a CUDA kernel;
// when we port, this class will be replaced (or supplemented) by a
// counter-based per-particle generator such as Philox 4x32-10 (cuRAND), which
// gives independent streams without storing per-particle state.
// =============================================================================
class RandomGenerator {
public:
    explicit RandomGenerator(std::uint64_t seed = 12345);

    // Standard normal N(0, 1).
    double gaussian();

    // Uniform on [0, 1).
    double uniform();

    // Uniform on [a, b).
    double uniform(double a, double b);

    // ---- Seeding ------------------------------------------------------------
    void          seed(std::uint64_t s);
    std::uint64_t getSeed() const { return seed_; }

private:
    std::uint64_t seed_;
    std::mt19937_64 engine_;
    std::normal_distribution<double>      normal_;
    std::uniform_real_distribution<double> uniform_;
};
