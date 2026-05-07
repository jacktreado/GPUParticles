#include "Initializer.hpp"

#include "Box.hpp"
#include "RandomGenerator.hpp"
#include "System.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace {
inline double two_pi() { return 2.0 * std::acos(-1.0); }
}

void Initializer::placeOnLattice(System& sys, const Box& box) {
    const std::size_t N = sys.getNumParticles();
    if (N == 0) return;

    // Smallest n_per_side such that n_per_side^2 >= N. Excess sites are
    // simply left empty (no particle placed there).
    const std::size_t n_per_side = static_cast<std::size_t>(
        std::ceil(std::sqrt(static_cast<double>(N))));

    const double dx = box.getLx() / static_cast<double>(n_per_side);
    const double dy = box.getLy() / static_cast<double>(n_per_side);

    std::size_t placed = 0;
    for (std::size_t iy = 0; iy < n_per_side && placed < N; ++iy) {
        for (std::size_t ix = 0; ix < n_per_side && placed < N; ++ix) {
            sys.setPosition(placed,
                            (static_cast<double>(ix) + 0.5) * dx,
                            (static_cast<double>(iy) + 0.5) * dy);
            ++placed;
        }
    }
}

void Initializer::placeRandomly(System& sys, const Box& box,
                                RandomGenerator& rng,
                                double min_sep,
                                int max_attempts) {
    const std::size_t N = sys.getNumParticles();
    const double min_sep2 = min_sep * min_sep;

    for (std::size_t i = 0; i < N; ++i) {
        bool accepted = false;
        for (int attempt = 0; attempt < max_attempts; ++attempt) {
            const double xc = rng.uniform(0.0, box.getLx());
            const double yc = rng.uniform(0.0, box.getLy());

            bool ok = true;
            for (std::size_t j = 0; j < i; ++j) {
                double ddx = xc - sys.getX(j);
                double ddy = yc - sys.getY(j);
                box.minimumImage(ddx, ddy);
                if (ddx * ddx + ddy * ddy < min_sep2) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                sys.setPosition(i, xc, yc);
                accepted = true;
                break;
            }
        }
        if (!accepted) {
            throw std::runtime_error(
                "Initializer::placeRandomly: failed to place particle " +
                std::to_string(i) +
                " — phi may be too high. Use init_mode=\"lattice\" instead.");
        }
    }
}

void Initializer::randomizeOrientations(System& sys, RandomGenerator& rng) {
    const std::size_t N = sys.getNumParticles();
    for (std::size_t i = 0; i < N; ++i) {
        sys.setTheta(i, rng.uniform(0.0, two_pi()));
    }
}

void Initializer::placeAnchorsAtParticles(System& sys) {
    const std::size_t N = sys.getNumParticles();
    for (std::size_t i = 0; i < N; ++i) {
        sys.setAnchor(i, sys.getX(i), sys.getY(i));
    }
}
