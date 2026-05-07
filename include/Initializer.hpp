#pragma once

class System;
class Box;
class RandomGenerator;

// =============================================================================
// Initializer
// -----------------------------------------------------------------------------
// Static methods that fill a System with an initial particle configuration.
// Two strategies are provided; choose based on packing fraction:
//
//   placeOnLattice : square lattice. Robust up to phi ~= pi/4 ~ 0.785, where
//                    nearest-neighbor spacing equals sigma. Above that the
//                    initial overlaps will produce huge forces — switch to a
//                    hexagonal lattice (a worthwhile extension) or to a
//                    compression protocol.
//
//   placeRandomly  : rejection-sampled random placement with a minimum-
//                    separation constraint. Fast and works only for low phi
//                    (roughly < 0.4). Throws if it cannot place a particle
//                    after `max_attempts` tries.
// =============================================================================
class Initializer {
public:
    // Place N particles on a square lattice that fills the box.
    static void placeOnLattice(System& sys, const Box& box);

    // Place N particles at random with no two centers closer than min_sep.
    static void placeRandomly(System& sys, const Box& box,
                              RandomGenerator& rng,
                              double min_sep,
                              int max_attempts_per_particle = 10000);

    // Assign each particle a uniformly random orientation in [0, 2*pi).
    // Call after placeOnLattice / placeRandomly to set the initial ABP directions.
    static void randomizeOrientations(System& sys, RandomGenerator& rng);

    // Initialize each anchor to coincide with its particle.  This zero-stretch
    // initial condition means the spring contributes no force at t=0.
    static void placeAnchorsAtParticles(System& sys);
};
