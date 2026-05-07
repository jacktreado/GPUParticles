#include "System.hpp"

#include <algorithm>

System::System(std::size_t N) {
    resize(N);
}

void System::resize(std::size_t N) {
    N_ = N;
    x_.assign(N, 0.0);
    y_.assign(N, 0.0);
    fx_.assign(N, 0.0);
    fy_.assign(N, 0.0);
    theta_.assign(N, 0.0);
    vx_.assign(N, 0.0);
    vy_.assign(N, 0.0);
    ax_.assign(N, 0.0);
    ay_.assign(N, 0.0);
}

void System::zeroForces() {
    std::fill(fx_.begin(), fx_.end(), 0.0);
    std::fill(fy_.begin(), fy_.end(), 0.0);
}

void System::zeroVelocities() {
    std::fill(vx_.begin(), vx_.end(), 0.0);
    std::fill(vy_.begin(), vy_.end(), 0.0);
}
