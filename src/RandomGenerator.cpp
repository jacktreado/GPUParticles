#include "RandomGenerator.hpp"

RandomGenerator::RandomGenerator(std::uint64_t s)
    : seed_(s), engine_(s), normal_(0.0, 1.0), uniform_(0.0, 1.0) {}

double RandomGenerator::gaussian() {
    return normal_(engine_);
}

double RandomGenerator::uniform() {
    return uniform_(engine_);
}

double RandomGenerator::uniform(double a, double b) {
    return a + (b - a) * uniform_(engine_);
}

void RandomGenerator::seed(std::uint64_t s) {
    seed_ = s;
    engine_.seed(s);
    // Reset the distributions: some implementations cache state.
    normal_.reset();
    uniform_.reset();
}
