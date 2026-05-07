# Brownian Sim — 2D WCA particles under overdamped Langevin dynamics

A modular C++17 simulation engine for 2D particles with purely repulsive WCA
interactions, integrated with a Brownian-dynamics (overdamped Langevin)
scheme. Designed from the start to be ported to GPU.

The state of the system is laid out as a **Structure of Arrays (SoA)** so that
when we add CUDA later, we hand raw pointers to kernels and get coalesced
memory access for free.

---

## Physics

- **Interaction (WCA):**
  $$U(r) = 4\varepsilon\left[\left(\tfrac{\sigma}{r}\right)^{12} - \left(\tfrac{\sigma}{r}\right)^6\right] + \varepsilon, \quad r < 2^{1/6}\sigma$$
  and `U(r) = 0` beyond the cutoff. Force is continuous at `r_cut`.

- **Dynamics (overdamped Langevin / Brownian dynamics):**
  $$\dot{r}_i = \frac{D}{k_BT}\,F_i + \sqrt{2D}\,\eta_i(t),
    \quad \langle\eta_i(t)\eta_j(t')\rangle = \delta_{ij}\,\delta(t-t')$$
  Integrated with Euler-Maruyama:
  $$r_i(t+\Delta t) = r_i(t) + \tfrac{D}{k_BT}\,F_i\,\Delta t + \sqrt{2D\,\Delta t}\,\xi_i,
    \quad \xi_i \sim \mathcal{N}(0, 1)$$
  Fluctuation-dissipation is enforced by the noise amplitude `√(2D·dt)` being
  locked to `D`.

- **2D periodic box.** Side length is derived from the packing fraction:
  $$\varphi = \frac{N\pi\sigma^2}{4L^2}\;\;\Rightarrow\;\; L = \sqrt{\frac{N\pi\sigma^2}{4\varphi}}$$

- **Units.** We take `kT = 1`, `σ = 1`, `ε = 1` by default. `D` is then the
  diffusion coefficient in units of `σ²·(ε/kT)⁻¹` ≡ `σ²/τ` where `τ` is the
  natural time unit. `dt` is in units of `τ`.

---

## Typical pair overlap in steady state

When two active Brownian particles swim head-on into each other and remain in
contact long enough that orientational diffusion is negligible (i.e. the
persistence time `τ_θ` exceeds the pair-relaxation time), each particle
satisfies the overdamped equation of motion

$$\gamma \dot{r}_i \;=\; f_0\,\hat{e}_i \;-\; \nabla_{r_i} U.$$

In steady state, `\dot{r}_i = 0`, so the active drive is balanced by the
repulsive pair force:

$$f_0 \;=\; F_\text{pot}(r^*),$$

where `r*` is the steady-state center-to-center separation. We define the
**typical overlap** as the dimensionless separation

$$\delta \;\equiv\; \frac{r^*}{\sigma}.$$

(δ < 1 means compression below the hard-core diameter; δ ≥ cutoff means no
contact and the active drive does no work.) Inverting the force balance for
each potential:

### Soft sphere (harmonic)

$$U(r) = \tfrac{\varepsilon}{2}\!\left(1 - \tfrac{r}{\sigma}\right)^{\!2},
  \quad F(r) = \tfrac{\varepsilon}{\sigma}\!\left(1 - \tfrac{r}{\sigma}\right) \quad (r < \sigma).$$

Force balance gives the closed form

$$\boxed{\;f_0 \;=\; \frac{\varepsilon}{\sigma}\,(1 - \delta), \qquad
  \delta \;=\; 1 - \frac{f_0\,\sigma}{\varepsilon}.\;}$$

Linear: a 1% compression below contact (δ = 0.99) needs `f_0 = 0.01\,\varepsilon/\sigma`.

### WCA

$$U(r) = 4\varepsilon\!\left[\!\left(\tfrac{\sigma}{r}\right)^{12} - \left(\tfrac{\sigma}{r}\right)^{6}\!\right] + \varepsilon, \quad
  F(r) = \frac{24\varepsilon}{r}\!\left(\tfrac{\sigma}{r}\right)^{6}\!\left[2\!\left(\tfrac{\sigma}{r}\right)^{6} - 1\right].$$

Force balance gives the closed form

$$\boxed{\;f_0 \;=\; \frac{24\varepsilon}{\sigma}\,\delta^{-7}\!\left(2\,\delta^{-6} - 1\right), \quad 0 < \delta < 2^{1/6}.\;}$$

There is no closed form for `\delta(f_0)`, but the relation is monotone on
`(0, 2^{1/6})` and trivial to invert numerically. WCA is far stiffer than the
harmonic soft sphere: at contact (δ = 1) already `f_0 = 24\,\varepsilon/\sigma`.

### Validity of the force-balance picture

The relation `f_0 = F_\text{pot}(r^*)` assumes:

1. **Persistent drive.** Reorientation time `τ_θ` is longer than the pair
   relaxation time `τ_\text{pair} \sim \gamma\sigma/F'(r^*)`. With `γ = 0.1`,
   `ε = σ = 1`, and `τ_θ = 1`, both potentials satisfy
   `τ_\text{pair} \ll τ_θ`.
2. **Athermal limit.** `kT \ll \varepsilon` so thermal fluctuations don't
   dominate the contact distance. For finite `kT` the formulas above give the
   *typical* (most-probable) overlap; the distribution broadens with `kT`.
3. **Pairwise contact.** Three-body crowding shifts the effective δ; the
   formula is exact for a head-on pair, a useful estimate at low `φ`.

### Programmatic access

The static helper `ForceCalculator::f0ForOverlap(type, delta, eps, sigma)`
returns `f_0` for either potential, falling back to `0` when `δ` is at or
beyond the cutoff (passive limit). The GUI uses it to drive the active force
from the `δ` slider directly.

---

## Building

Requires:

- A C++17 compiler (g++ ≥ 7, clang++ ≥ 6, MSVC ≥ 19.14)
- CMake ≥ 3.16
- Boost (headers only; for `boost::numeric::odeint` adaptive integrator)
- HDF5 with C++ bindings (for trajectory output)
- Internet access on first build (to fetch `nlohmann/json`, GoogleTest, and —
  if the GUI is enabled — GLFW + Dear ImGui), or system-installed equivalents

On macOS:
```bash
brew install boost hdf5 cmake
```

### Default build (CLI + GUI)

The interactive GUI is built by default. It pulls in **GLFW** and **Dear ImGui**
via CMake `FetchContent` on first configure.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This produces:
- `build/sim` — the CLI driver (`sim run <input.json>`, `sim info <input.json>`)
- `build/gui/sim_gui` — the interactive ImGui-based viewer

### CLI-only build (no GUI dependencies)

If you don't have a working OpenGL / GLFW environment, or simply don't need the
viewer, disable the GUI:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_GUI=OFF
cmake --build build -j
```

Only `build/sim` is produced; nothing under `gui/` is built and GLFW + ImGui are
not fetched.

### Running the GUI

```bash
./build/gui/sim_gui
```

The GUI exposes three dimensionless control parameters with `γ = 0.1` and
`ε = σ = τ_θ = 1` fixed: the typical pair overlap `δ = r*/σ` (which sets `f₀`
by force balance — see [Typical pair overlap](#typical-pair-overlap-in-steady-state)),
the friction ratio `γ/γ_a`, and the relaxation-time ratio `τ_a/τ_θ` with
`τ_a = γ_a/k_a`. The pair potential (WCA or soft sphere) is a dropdown.
Number of particles `N`, packing fraction `φ`, `kT`, and `dt` are sliders.
The simulation runs in memory — no trajectory file is written. Particles are
drawn as filled disks, optionally with orientation arrows; anchors (when
`k_a > 0`) are smaller disks connected to their particles by springs
(toggleable).

## Running

```bash
# Inspect a config without running anything
./build/sim info examples/input.json

# Run the simulation
./build/sim run  examples/input.json
```

The trajectory is written in extended-XYZ format (loadable in OVITO and VMD).

---

## Testing

Tests use [Google Test](https://github.com/google/googletest) (fetched automatically by CMake on first build).

**Build and run all tests:**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

**Run with verbose pass/fail output:**

```bash
ctest --test-dir build --output-on-failure
```

**Run a single test suite by name:**

```bash
./build/test/bd_tests --gtest_filter="CellListTest.*"
```

**Run a single test case:**

```bash
./build/test/bd_tests --gtest_filter="ForceCalculatorTest.ForceAtSigmaSeparation"
```

There are 129 tests across seven files in `test/`:

| File | What it covers |
|---|---|
| `test_box.cpp` | `Box::minimumImage`, `Box::wrap`, construction, round-trip consistency |
| `test_system.cpp` | SoA storage, `resize`, `zeroForces`, raw-pointer / vector-data consistency |
| `test_force_calculator.cpp` | WCA cutoff, exact force at σ, zero force beyond cutoff, PBC pairs, `computeEnergy`, cell-list vs brute-force force equivalence (tolerance 1e-12) |
| `test_cell_list.cpp` | Grid geometry, small-box fallback, `needsRebuild` lifecycle, PBC-aware drift check, complete neighbor-list correctness vs brute force, cross-boundary neighbor detection |
| `test_integrator.cpp` | Cached constants, setter invalidation, drift magnitude, positions stay wrapped, MSD ≈ 4Dt statistical check |
| `test_config.cpp` | All field defaults, all overrides, every validation error that should throw |
| `test_integration.cpp` | Lattice/random init geometry, min-sep enforcement, zero energy on sparse lattice, repulsive relaxation, cell-list trajectory identical to brute-force over 50 steps, `TrajectoryWriter` XYZ format |

---

## Project layout

```
brownian-sim/
├── CMakeLists.txt
├── examples/
│   └── input.json              # minimal example with all required + optional fields
├── include/                    # public headers — one per module
│   ├── System.hpp              # SoA particle state + getters/setters
│   ├── Box.hpp                 # 2D periodic box, minimum-image, wrap
│   ├── ForceCalculator.hpp     # WCA forces, GPU-shaped pair loop
│   ├── Integrator.hpp          # Brownian-dynamics Euler-Maruyama step
│   ├── RandomGenerator.hpp     # std::mt19937_64 wrapper (will swap for Philox)
│   ├── Initializer.hpp         # lattice / random initial configurations
│   ├── Config.hpp              # JSON-driven parameters
│   └── TrajectoryWriter.hpp    # XYZ trajectory output
├── src/                        # corresponding .cpp implementations
└── main.cpp                    # subcommand dispatcher: run / info / help
```

Each `.hpp` header keeps its dependencies in forward declarations where
possible, so adding modules later (cell lists, observables, alternative
integrators) doesn't trigger sweeping rebuilds.

---

## Required JSON fields

| Field | Type    | Meaning                                             |
|-------|---------|-----------------------------------------------------|
| `N`   | int     | number of particles                                 |
| `D`   | double  | single-particle diffusion coefficient (σ²/τ units)  |
| `phi` | double  | 2D packing fraction, must be in (0, 1)              |

## Optional JSON fields (with defaults)

| Field          | Default          | Meaning                              |
|----------------|------------------|--------------------------------------|
| `sigma`        | 1.0              | particle diameter                    |
| `epsilon`      | 1.0              | WCA energy scale                     |
| `kT`           | 1.0              | thermal energy                       |
| `dt`           | 1e-4             | Brownian-dynamics timestep           |
| `n_steps`      | 100000           | total number of integration steps    |
| `output_every` | 1000             | trajectory write frequency           |
| `output_file`  | `trajectory.xyz` | trajectory output path               |
| `init_mode`    | `"lattice"`      | `"lattice"` or `"random"`            |
| `seed`         | 12345            | RNG seed                             |
| `r_skin`       | 0.2              | Verlet neighbor-list skin layer (σ)  |
| `verlet_check_every` | 1          | steps between Verlet rebuild checks  |

---

## Design choices that matter for the GPU port

1. **SoA storage** — `x_`, `y_`, `fx_`, `fy_` are separate `std::vector<double>`s,
   not a `std::vector<Particle>`. Future replacement: `thrust::device_vector<double>`
   or raw `cudaMalloc` pointers, with the rest of the code unchanged.

2. **No Newton's-third-law accumulation in the force kernel.** Each particle's
   force is built by iterating over all others and writing only to its own
   slot. This wastes 2× CPU work but means every iteration of the outer loop
   is fully independent — exactly what a CUDA kernel needs.

3. **Cached scalar constants** (`r_cut2_`, `mobility_dt_`, `noise_amplitude_`)
   so the inner loop has no branches besides the cutoff test, no `sqrt`, and
   no division beyond `1/r²`.

4. **`__restrict__` annotations** on the hot pointers: lets the compiler assume
   no aliasing, which is important both for vectorization on CPU and for the
   eventual `__restrict__` discipline in CUDA.

5. **Two-pass integrator step** (update positions, then wrap). On GPU we'll
   keep the same shape — the wrap pass becomes a separate light kernel, which
   is fine because it's bandwidth-bound and trivial.

6. **Single host-side RNG today, comment-flagged for replacement.** The clean
   interface makes it a one-file change to switch to a per-particle Philox
   stream when we move to CUDA.

---

## What's intentionally missing (will be added later)

- **Cell / neighbor lists** — implemented. `CellList` provides a sort-based
  cell list with a Verlet skin layer. Each pipeline stage (compute cell IDs →
  sort → find boundaries → build neighbor list) is in its own method, ready
  for a near-mechanical translation to CUDA kernels.
- **Higher-order BD integrators** (Heyes, Stratonovich correctors, etc.).
- **Hexagonal lattice initializer** for `φ > π/4`.
- **Observables** (mean-squared displacement, radial distribution, pressure
  via the virial). These belong in their own module so they stay decoupled
  from the integrator.
- **Multi-GPU / MPI** — the SoA layout is designed so domain decomposition
  remains an option later.


## To - Do

- [x] Update integrator to be better + adaptive (`cash-karp54` on my mind)
- [x] Change output to `.h5` format
- [x] Write a visualization python class
- [x] Add activity (active brownian, flocking)
- [x] Add viscoelasticity
- [x] Add `AdaptiveIntegrator` to GUI
  - Need to incorporate `CellList` into integrator
  - Make `AdaptiveIntegrator` work on all drift forces, not just WCA
  - Make a universal `strang_step` function (or something) that uses strang splitting and automatically uses `boost` to take a step with all drift forces, included non-bonded interactions
- [x] Fix soft sphere problems in GUI, they are too slow compared to WCA for some reason
  - all comes down to $\delta$, need to sort + fix
- [x] Fix `AdaptiveIntegrator` sliders in GUI, they dont slide properly, just jump between min and max
- Port to GPUs, figure out how to test (multi-threaded or many params at once?)
