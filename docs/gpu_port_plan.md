# GPU port plan

A pre-implementation design for adding a CUDA backend to the GPUParticles Brownian dynamics simulator. **No code yet** — every `cuda/` path mentioned below is hypothetical and exists only in this document. The goal is to ship a plan you can evaluate end-to-end before any line of CUDA is written.

## Goals (the five questions this doc answers)

1. **What parts of the simulation base + psweeps would be affected?** — [§3 Affected surfaces](#3-affected-surfaces).
2. **What would have to change for GPU?** — [§5 Per-component change list](#5-per-component-change-list) and [§6 Schema + psweep changes](#6-schema--psweep-changes).
3. **How to test (locally and on the cluster)?** — [§7 Testing strategy](#7-testing-strategy).
4. **How to verify correctness once on the cluster?** — [§8 Cluster verification checklist](#8-cluster-verification-checklist).
5. **How to make the GPU code totally unpolluting to the existing CPU code?** — [§4 Isolation strategy](#4-isolation-strategy).

## Non-goals (v1)

- Multi-GPU.
- HIP / Apple Metal / SYCL — CUDA only.
- Half-precision (fp16) anywhere.
- AdaptiveStrang on GPU — the Boost.Odeint Cash-Karp 5(4) controller behind [include/AdaptiveIntegrator.hpp:163-166](../include/AdaptiveIntegrator.hpp#L163) stays CPU-only.
- On-device `CorrelationAccumulator` or `ContactDurationAccumulator` — host-side for v1.
- Bit-identical CPU/GPU output at non-zero kT — the RNG stream is structurally different (see [§5.5](#55-randomgenerator--rewrite-gpu-only)).

Status legend used below: **[v1]** = in scope, **[v2]** = deferred to a later release, **[oos]** = out of scope.

---

## 1. Preamble

The CPU code was written with a GPU port in mind. The SoA storage, the no-Newton-3rd-law force loop, the four kernel-shaped CellList stages, and the pImpl boundary on AdaptiveIntegrator all exist *specifically* to make the eventual CUDA translation a structural overlay, not a rewrite. This doc treats that decision as the foundation and builds the GPU plan as a sibling subtree that bolts on cleanly.

The hard external constraint: **the GPU code must be totally independent of the CPU code**. Concretely:

- A macOS dev box with no CUDA toolkit installed must continue to build and test the CPU `sim` binary exactly as it does today.
- `git diff main -- src/ include/ main.cpp` must be near-empty after the port (modulo a small `Config` change for the new `backend` field).
- No `#include <cuda*>`, no `#include <thrust*>`, no `#ifdef CUDA`, no `__device__` keyword anywhere under [include/](../include/) or [src/](../src/).

[§4](#4-isolation-strategy) makes this enforceable.

---

## 2. Architecture overview

### 2.1 Current call graph

```
main.cpp
  ├─ runEulerMaruyama   ─┐
  ├─ runHeun            ─┼─→ ForceCalculator  ─→ pair primitives (WCA, SoftSphere)
  └─ runAdaptive        ─┤   CellList         (four stage methods)
                         │   Integrator / HeunIntegrator / AdaptiveIntegrator
                         │   RandomGenerator  (std::mt19937_64)
                         │   TrajectoryWriter (HDF5)
                         │   CorrelationAccumulator
                         └─→ ContactDurationAccumulator
       backed by:        System (SoA), Box, Config, Initializer
```

### 2.2 Source inventory

Library [CMakeLists.txt:40-54](../CMakeLists.txt#L40) — 13 .cpp files under [src/](../src/), 14 headers under [include/](../include/). Binary `sim` is a thin `main.cpp` wrapper [CMakeLists.txt:74-76](../CMakeLists.txt#L74). Tests link `brownian_lib` + GoogleTest 1.15.2.

| Header | One-line role |
|---|---|
| [System.hpp](../include/System.hpp) | SoA particle state (x, y, fx, fy, theta, vx, vy, ax, ay) |
| [Box.hpp](../include/Box.hpp) | Periodic box, minimumImage, wrap |
| [ForceCalculator.hpp](../include/ForceCalculator.hpp) | Pair forces (WCA / SoftSphere), brute-force + Verlet overloads |
| [CellList.hpp](../include/CellList.hpp) | Sort-based cell list + CSR Verlet list, four pipeline stages |
| [Integrator.hpp](../include/Integrator.hpp) | Euler-Maruyama overdamped Langevin |
| [HeunIntegrator.hpp](../include/HeunIntegrator.hpp) | Heun predictor-corrector |
| [AdaptiveIntegrator.hpp](../include/AdaptiveIntegrator.hpp) | Cash-Karp 5(4) inner + Strang macro step (pImpl) |
| [AdaptiveMacroIntegrator.hpp](../include/AdaptiveMacroIntegrator.hpp) | Adaptive macro Δt wrapper (CPU-only, v2 territory) |
| [RandomGenerator.hpp](../include/RandomGenerator.hpp) | `std::mt19937_64` + N(0,1) + uniform |
| [Initializer.hpp](../include/Initializer.hpp) | Lattice / random initial placement (host, one-shot) |
| [Config.hpp](../include/Config.hpp) | JSON parser, derived parameters |
| [TrajectoryWriter.hpp](../include/TrajectoryWriter.hpp) | HDF5 frame writer |
| [CorrelationAccumulator.hpp](../include/CorrelationAccumulator.hpp) | Ring-buffer time correlators |
| [ContactDurationAccumulator.hpp](../include/ContactDurationAccumulator.hpp) | Streaming pair-contact statistics |

### 2.3 Properties that make the port tractable

- **SoA storage with raw-pointer accessors.** [include/System.hpp:89-109](../include/System.hpp#L89) exposes `xData()`, `yData()`, ... — direct cudaMemcpy targets. The comment at [include/System.hpp:17-19](../include/System.hpp#L17) explicitly documents this is "for the eventual CUDA migration".
- **No Newton's 3rd law trick in the force loop.** [include/ForceCalculator.hpp:35-43](../include/ForceCalculator.hpp#L35) explains that each particle's force is computed independently from scratch — every kernel thread writes only its own `(fx[i], fy[i])`, no atomics.
- **CellList already decomposed into four kernel-shaped stages.** [include/CellList.hpp:21-44](../include/CellList.hpp#L21) lays out `computeCellIds → sortByCellId → findCellBoundaries → buildVerletList` with the comment "each one becomes a single CUDA kernel in the GPU port".
- **CellList bookkeeping is flat `int32` arrays.** [include/CellList.hpp:65-72](../include/CellList.hpp#L65) — never nested vectors; raw-pointer accessors at [include/CellList.hpp:104-112](../include/CellList.hpp#L104).
- **AdaptiveIntegrator hides Boost behind pImpl.** [include/AdaptiveIntegrator.hpp:82-86](../include/AdaptiveIntegrator.hpp#L82) explicitly says "a CUDA port replaces only AdaptiveIntegrator.cpp's Impl".
- **Single library + thin binary.** [CMakeLists.txt:40-76](../CMakeLists.txt#L40) — the engine is `brownian_lib`, the binary is `main.cpp`. Adding a parallel `brownian_lib_cuda` + `gpu_main.cpp` does not perturb anything.

### 2.4 Properties hostile to the port

- **`std::mt19937_64` per-thread state.** [include/RandomGenerator.hpp:38-43](../include/RandomGenerator.hpp#L38). Cannot live in a kernel. The header itself anticipates this at [include/RandomGenerator.hpp:16-19](../include/RandomGenerator.hpp#L16) — replacement is Philox 4x32-10 via cuRAND's device API.
- **`std::sort` in `CellList::sortByCellId`.** Stage 2 of the CellList pipeline. Replacement: `thrust::sort_by_key`.
- **`std::unordered_map` in ContactDurationAccumulator.** Host-only data structure for streaming pair contact statistics. Stays on host in v1 (see [§5.11](#511-contactdurationaccumulator--stays-on-host-v1)).
- **`Boost.Odeint` Cash-Karp inside AdaptiveIntegrator's `Impl`.** No CUDA equivalent without a substantial rewrite. v2.

---

## 3. Affected surfaces

### 3.1 C++ engine

| Existing file | Status under the port |
|---|---|
| [include/System.hpp](../include/System.hpp) | **Unchanged.** A new `SystemDevice` mirror struct lives in `cuda/include/`, built from a host `System` via the existing raw-pointer accessors. |
| [include/Box.hpp](../include/Box.hpp) | **Unchanged.** A new `BoxDevice.cuh` re-emits the same `minimumImage` / `wrap` math as `__host__ __device__` inlines in the cuda subtree. |
| [include/ForceCalculator.hpp](../include/ForceCalculator.hpp) | **Unchanged.** A peer class `ForceCalculatorCuda` in `cuda/include/` mirrors the public interface. |
| [include/CellList.hpp](../include/CellList.hpp) | **Unchanged.** Peer class `CellListCuda` mirrors the four-stage pipeline. |
| [include/Integrator.hpp](../include/Integrator.hpp), [include/HeunIntegrator.hpp](../include/HeunIntegrator.hpp) | **Unchanged.** Peers `IntegratorCuda`, `HeunIntegratorCuda`. |
| [include/AdaptiveIntegrator.hpp](../include/AdaptiveIntegrator.hpp), [include/AdaptiveMacroIntegrator.hpp](../include/AdaptiveMacroIntegrator.hpp) | **Unchanged.** **[v2]** — no GPU port in v1; `sim_gpu` errors out clearly if asked. |
| [include/RandomGenerator.hpp](../include/RandomGenerator.hpp) | **Unchanged.** GPU code uses Philox; never instantiates `RandomGenerator`. |
| [include/TrajectoryWriter.hpp](../include/TrajectoryWriter.hpp) | **Unchanged.** GPU runner pulls state to host before each write. |
| [include/CorrelationAccumulator.hpp](../include/CorrelationAccumulator.hpp), [include/ContactDurationAccumulator.hpp](../include/ContactDurationAccumulator.hpp) | **Unchanged.** Stay on host, fed by D2H at sample cadence. |
| [include/Initializer.hpp](../include/Initializer.hpp) | **Unchanged.** Host-only one-shot init, then `pushFrom` to device. |
| [include/Config.hpp](../include/Config.hpp), [src/Config.cpp](../src/Config.cpp) | **Small change [v1].** Add a `Backend` enum and a `Backend backend = Backend::CPU` field. The only host-side header change in the port. |
| [main.cpp](../main.cpp) | **Unchanged.** Refuses to run if `cfg.backend == Backend::CUDA` with a helpful "use sim_gpu" error. |

### 3.2 Build system

- [CMakeLists.txt](../CMakeLists.txt) gets **one** new block: an `option(BUILD_CUDA "Build the CUDA backend" OFF)` plus a guarded `add_subdirectory(cuda)`. Follows the exact pattern already used for `BUILD_GUI` at [CMakeLists.txt:97-100](../CMakeLists.txt#L97).
- `brownian_lib`'s source list at [CMakeLists.txt:40-54](../CMakeLists.txt#L40) **stays exactly as today**. Zero `.cu` files in it.
- The default build (`BUILD_CUDA=OFF`) never calls `enable_language(CUDA)`, never touches `find_package(CUDAToolkit)`.

### 3.3 Tests

- [test/CMakeLists.txt](../test/CMakeLists.txt) — **unchanged**. Existing `bd_tests` binary stays pure CPU.
- New `cuda/test/CMakeLists.txt` declares a separate `bd_tests_cuda` binary that links `brownian_lib_cuda` plus `brownian_lib` (for reference oracles).
- pytest integration tests under [tests/](../tests/) — **unchanged**. They run on the analysis pipeline, not the simulator.

### 3.4 psweep + cluster

[psweep.py](../psweep.py) touches four places, each one localized:

- `DEFAULT_BINARY` at [psweep.py:60](../psweep.py#L60).
- Schema validation at [psweep.py:73-118](../psweep.py#L73): add `backend` to `NON_SWEEPABLE`, add `{"cpu","cuda"}` to `ALLOWED_VALUES`, add `"backend"` to `STRING_FIELDS`.
- Build step at [psweep.py:373-393](../psweep.py#L373): pass `-DBUILD_CUDA=ON` when the sweep runs on GPU; build the `sim_gpu` target.
- SLURM template at [psweep.py:460-498](../psweep.py#L460): inject GPU resource flags + `module load cuda/<version>`.

[examples/input.json](../examples/input.json) gets one new field, `"backend"`, defaulting to `"cpu"`. Existing JSONs remain valid unchanged.

### 3.5 Out of scope (not touched)

- GUI ([gui/CMakeLists.txt](../gui/CMakeLists.txt))
- Python analysis pipeline ([python/](../python/))
- pytest tests ([tests/](../tests/))
- Existing docs ([docs/GPUParticles.md](GPUParticles.md), [docs/active_langevin_splitting.md](active_langevin_splitting.md), [docs/adaptive_macro_step.md](adaptive_macro_step.md), [docs/analysis_pipeline.md](analysis_pipeline.md))

---

## 4. Isolation strategy

**Recommendation:** CUDA lives in a new top-level `cuda/` subtree, builds a separate static library `brownian_lib_cuda` and a separate binary `sim_gpu`, both gated by a CMake option `-DBUILD_CUDA=ON` (off by default).

### 4.1 Why this beats every alternative

| Alternative | Pollutes CPU? | Failure mode against the constraint |
|---|---|---|
| `#ifdef CUDA` inside existing .cpp / .hpp files | Yes | Violates the user's hard constraint by definition. |
| Abstract `IBackend` virtual interface in `brownian_lib` | Yes (subtly) | Adds a vtable into the hot path; the interface header has to compile on macOS, which forces the CPU build to know GPU types exist; introduces an ABI dependency between the two backends. |
| Runtime dispatch in a single `sim` binary on a `--backend` flag | Yes | The binary must link both libs, so CMake config must locate nvcc unconditionally — breaks the macOS build the moment CUDA isn't installed. |
| Separate `.so`, dlopen at runtime | Partially | The CPU binary has to optionally find the shared library at startup. Gains nothing over a separate binary, adds an extra moving piece for cluster operations. |
| **Separate library + separate binary + CMake gate** | **No** | Recommended. |

When `BUILD_CUDA=OFF`, CMake never touches the cuda subtree. `brownian_lib`, `sim`, and `bd_tests` see exactly today's build graph.

### 4.2 The `cuda/` directory layout

```
cuda/
    CMakeLists.txt              # the ONLY project file that calls enable_language(CUDA)
    include/
        SystemDevice.cuh        # device-side SoA pointer view of System
        BoxDevice.cuh           # __host__ __device__ minimumImage / wrap
        ForceCalculatorCuda.cuh
        CellListCuda.cuh
        IntegratorCuda.cuh
        HeunIntegratorCuda.cuh
        RandomCuda.cuh          # Philox helpers
    src/
        ForceCalculatorCuda.cu
        CellListCuda.cu
        IntegratorCuda.cu
        HeunIntegratorCuda.cu
        kernels/force_kernels.cu
        kernels/celllist_kernels.cu
        kernels/integrator_kernels.cu
    gpu_main.cpp                # mirrors main.cpp; builds the `sim_gpu` target
    test/
        CMakeLists.txt
        test_force_calculator_cuda.cpp
        test_cell_list_cuda.cpp
        test_integrator_cuda.cpp
        test_heun_integrator_cuda.cpp
        test_rng_cuda.cpp
        test_box_device.cpp
        test_system_device.cpp
        test_equivalence.cpp    # end-to-end CPU-vs-GPU regression
```

Justification for each top-level slot is in [§5](#5-per-component-change-list) below.

### 4.3 The unpollution contract

These five rules are the contract. They are mechanically checkable from CI and from a `grep`.

1. No `#include <cuda*>` or `#include <thrust*>` anywhere under [include/](../include/) or [src/](../src/).
2. No `#ifdef CUDA` or `#ifdef BUILD_CUDA` anywhere under [include/](../include/) or [src/](../src/). No `__device__`, no `__global__`, no `__host__` either.
3. `brownian_lib`'s source list at [CMakeLists.txt:40-54](../CMakeLists.txt#L40) stays exactly as today.
4. `git diff main -- src/ include/ main.cpp` is empty modulo the `Config` change ([§5.13](#513-config--host-only-small-change-v1)).
5. `cmake -S . -B build -DBUILD_CUDA=OFF` on a machine without nvcc succeeds and produces a working `sim` and `bd_tests`.

The cluster verification checklist ([§8.1](#81-build-sanity)) makes rules 1, 2, 3, and 5 mechanical: a `grep` plus an `ldd` is enough to prove the contract holds.

### 4.4 How `System` is shared without polluting it

- `System` stays a pure C++ host class. No header edit.
- A new `SystemDevice` struct (defined only in `cuda/include/SystemDevice.cuh`) holds device-side pointers: `double *x, *y, *fx, *fy, *theta, *vx, *vy, *ax, *ay`.
- Constructed from a host `System` by allocating device buffers (one `cudaMalloc` each, sized from `sys.getNumParticles()`).
- Sync routines `pullTo(System&)` (D2H) and `pushFrom(const System&)` (H2D) are the *only* CPU↔GPU bridge. They use the existing raw-pointer accessors at [include/System.hpp:89-109](../include/System.hpp#L89).

### 4.5 How `Box` is shared

`Box` is value-typed and trivially copyable. The GPU subtree gets a `BoxDevice.cuh` that re-emits the math from [include/Box.hpp](../include/Box.hpp) (specifically `minimumImage` and `wrap`) as `__host__ __device__` inlines. Bit-identical math (`nearbyint`, `floor`). Box values are passed by value into kernels — no separate device-allocated copy is needed.

The duplication is one source-file of inline math (~40 lines). The alternative (annotating [include/Box.hpp](../include/Box.hpp) with `__host__ __device__`) would violate rule 2. The duplication is worth it.

### 4.6 What lives where (file-by-file table)

| File | CPU build sees it? | GPU build sees it? |
|---|---|---|
| `include/System.hpp` | yes | yes (for `pushFrom` / `pullTo` on host) |
| `include/Box.hpp` | yes | yes (host-side construction) |
| `include/ForceCalculator.hpp` | yes | yes (test oracle in `test_equivalence`) |
| `include/CellList.hpp` | yes | yes (test oracle) |
| `include/Integrator.hpp`, `HeunIntegrator.hpp` | yes | yes (test oracle) |
| `include/AdaptiveIntegrator.hpp`, `AdaptiveMacroIntegrator.hpp` | yes | yes (still compiles; just never instantiated in GPU runs) |
| `include/RandomGenerator.hpp` | yes | yes (test oracle; CPU side of equivalence tests) |
| `include/TrajectoryWriter.hpp`, `CorrelationAccumulator.hpp`, `ContactDurationAccumulator.hpp` | yes | yes (called from `gpu_main.cpp` host side) |
| `include/Config.hpp` | yes | yes |
| `src/*.cpp` | yes | yes (linked into `brownian_lib`, which `brownian_lib_cuda` depends on) |
| `main.cpp` | yes (produces `sim`) | yes (still compiles into `sim` even with `BUILD_CUDA=ON`) |
| `cuda/include/*.cuh` | **no** | yes |
| `cuda/src/*.cu` | **no** | yes |
| `cuda/gpu_main.cpp` | **no** | yes (produces `sim_gpu`) |
| `cuda/test/*` | **no** | yes (produces `bd_tests_cuda`) |

The "no" rows are what makes the contract self-enforcing: the CMake gate at `BUILD_CUDA=OFF` cuts every `cuda/` entry out of the build graph.

---

## 5. Per-component change list

Each sub-section uses the same template:

> **CPU file** | **GPU file** | **Classification** | **Kernel sketch** | **Data movement** | **Validation hook**

The four classifications:

- **rewrite** — entirely new code, no CPU analog.
- **device-mirror** — a thin device-side mirror of a CPU class (state-only).
- **port-as-kernel** — CPU loop → one or more CUDA kernels with the same semantics.
- **stays-on-host** — runs only in host code; fed by D2H copies.

### 5.1 `System` — device-mirror [v1]

| Field | Value |
|---|---|
| CPU | [include/System.hpp](../include/System.hpp), [src/System.cpp](../src/System.cpp) |
| GPU | `cuda/include/SystemDevice.cuh` |
| Class | device-mirror |

Layout: nine `double*` device pointers (one per SoA component in [include/System.hpp:118-126](../include/System.hpp#L118)), sized from the host `System::getNumParticles()`. One `cudaMalloc` per array at construction; matching `cudaFree` in the destructor.

Bridge: `pushFrom(const System&)` → 9 H2D `cudaMemcpy` calls. `pullTo(System&)` → 9 D2H copies. Used at initialization, at each output write, and at each correlation/contact-sample tick.

Precision: stays `double` for v1, matching [include/System.hpp:118](../include/System.hpp#L118). `kT` can be as low as `1e-8` per [examples/input.json:9](../examples/input.json#L9); the noise-amplitude `sqrt(2*kT/gamma*dt)` benefits measurably from fp64 at these magnitudes. fp32 deferred (see [§9](#9-open-decisions--deferred-items)).

**Validation:** `test_system_device.cpp` — round-trip a randomized `System` through `pushFrom` then `pullTo`, assert element-wise equality.

### 5.2 `Box` — value-typed, copy by value [v1]

| Field | Value |
|---|---|
| CPU | [include/Box.hpp](../include/Box.hpp) |
| GPU | `cuda/include/BoxDevice.cuh` (inline math re-emitted) |
| Class | port-as-kernel (math duplicated) |

The two methods used in kernels — `minimumImage` and `wrap` — are re-emitted as `__host__ __device__` inlines. Box parameters (`Lx`, `Ly`, `invLx`, `invLy`) are passed into kernels by value as a small POD struct.

**Validation:** `test_box_device.cpp` — fuzz 10000 random (dx, dy) pairs through both CPU and GPU implementations; assert agreement within `1e-15`.

### 5.3 `ForceCalculator` — port-as-kernel [v1]

| Field | Value |
|---|---|
| CPU | [include/ForceCalculator.hpp](../include/ForceCalculator.hpp), [src/ForceCalculator.cpp](../src/ForceCalculator.cpp) |
| GPU | `cuda/include/ForceCalculatorCuda.cuh`, `cuda/src/ForceCalculatorCuda.cu`, `cuda/src/kernels/force_kernels.cu` |
| Class | port-as-kernel |

**Kernel sketch.** One thread per particle `i`. Thread walks its CSR row of the Verlet list (`nlist_start_[i]..nlist_start_[i+1]`, per the CSR layout documented at [include/CellList.hpp:159-161](../include/CellList.hpp#L159)), reads neighbor `j` positions, applies `BoxDevice::minimumImage`, computes the pair force, and accumulates into local `fx`, `fy`. Final write to `fx[i]`, `fy[i]`. **No atomics needed** — independence is guaranteed by the no-Newton-3rd-law decision at [include/ForceCalculator.hpp:35-43](../include/ForceCalculator.hpp#L35).

**Potential dispatch.** The CPU code uses runtime function pointers ([include/ForceCalculator.hpp:49-52](../include/ForceCalculator.hpp#L49)). On GPU, dispatch via a kernel template parameter on `PotentialType` (compile-time selection). Two kernel instantiations: WCA, SoftSphere. This is faster than `__device__` function pointers and lets `nvcc` inline the pair primitive into the inner loop.

**Brute-force overload.** [include/ForceCalculator.hpp:61](../include/ForceCalculator.hpp#L61) — port as a second kernel for the small-box fallback that triggers in [include/CellList.hpp:56-63](../include/CellList.hpp#L56). Same one-thread-per-particle pattern, inner loop over all `j != i`.

**`computeEnergy`.** Called once per `output_dt` ([main.cpp:298](../main.cpp#L298) — see source for exact line). For v1, do this on host after `pullTo` — it exercises the existing reference implementation and avoids writing a parallel reduction kernel that runs once per output. Cost: one extra D2H + an O(N²) host pass at output cadence (rare).

**Block size.** Decide empirically. Start at 128; profile in [§8.4](#84-performance-ceiling-check).

**Data movement.** Inputs (positions) and outputs (forces) are device-resident. No H2D / D2H in the step loop.

**Validation:** `test_force_calculator_cuda.cpp` — fixed deterministic position layout, N=64, both potentials; assert per-particle force agrees with CPU within `1e-12`.

### 5.4 `CellList` — port-as-kernel (four sub-kernels) [v1]

| Field | Value |
|---|---|
| CPU | [include/CellList.hpp](../include/CellList.hpp), [src/CellList.cpp](../src/CellList.cpp) |
| GPU | `cuda/include/CellListCuda.cuh`, `cuda/src/CellListCuda.cu`, `cuda/src/kernels/celllist_kernels.cu` |
| Class | port-as-kernel |

The doc at [include/CellList.hpp:21-44](../include/CellList.hpp#L21) is the kernel design — port it verbatim.

| Stage | CPU method | GPU kernel | Atomic? |
|---|---|---|---|
| 1 | `computeCellIds` | One thread per particle, writes `cell_id_[i]` | no |
| 2 | `sortByCellId` (std::sort) | `thrust::sort_by_key(cell_id_, sorted_particle_ids_)` | n/a |
| 3 | `findCellBoundaries` | One thread per sorted slot, writes `cell_start_[c]`, `cell_end_[c]` | no |
| 4a | `buildVerletList` count | One thread per particle, walks 3×3 stencil, counts | no |
| 4b | exclusive scan | `thrust::exclusive_scan` over per-particle counts | n/a |
| 4c | `buildVerletList` fill | One thread per particle, walks 3×3 stencil, fills `nlist_` | no |

**Rebuild trigger.** `needsRebuild` ([include/CellList.hpp:88-94](../include/CellList.hpp#L88)) becomes `thrust::any_of` over a per-particle drift predicate. Algebra unchanged.

**Snapshot.** `captureSnapshot` is two D2D copies of the position arrays into device-side `x_snapshot_`, `y_snapshot_`.

**Brute-force fallback.** When `useBruteForce()` is true ([include/CellList.hpp:102](../include/CellList.hpp#L102)), rebuild is a no-op and the caller invokes the brute-force force kernel from [§5.3](#53-forcecalculator--port-as-kernel-v1). Same dispatch logic as CPU — no GPU-specific decision.

**Data movement.** All CellList arrays stay device-resident. Pointers exposed via `cuda/include/CellListCuda.cuh` accessors for kernel use; never copied to host except for diagnostics.

**Validation:** `test_cell_list_cuda.cpp` — build a small system (N=128), run `rebuild` on both CPU and GPU, assert:

- `cell_id_` equal element-wise;
- `cell_start_`, `cell_end_` equal element-wise;
- `nlist_start_` equal element-wise;
- `nlist_` rows equal *as sets* (neighbor order within a row may differ — that's allowed by the algorithm).

### 5.5 `RandomGenerator` — rewrite (GPU-only) [v1]

| Field | Value |
|---|---|
| CPU | [include/RandomGenerator.hpp](../include/RandomGenerator.hpp), [src/RandomGenerator.cpp](../src/RandomGenerator.cpp) — unchanged |
| GPU | `cuda/include/RandomCuda.cuh` |
| Class | rewrite (no port — fundamentally different algorithm) |

**Algorithm.** Counter-based Philox 4x32-10 via cuRAND's device API. Stateless: given `(seed, step_index, particle_index, draw_subindex)` the algorithm returns the same uniformly distributed `uint32` deterministically. Box-Muller transforms two uniforms into two `N(0,1)` samples in the kernel.

**Why Philox.** A per-particle `curandState` would require storing 192+ bytes per particle and a `curand_init` pass before the first step. Philox is stateless: same `(seed, counter)` ⇒ same number, no state allocation, no init pass. Heun-style integrators that need the same noise across two sub-steps (see [§5.7](#57-heunintegrator--port-as-kernel-v1)) get bit-identical noise simply by reusing the counter.

**Counter scheme.**

```
uniform = philox(seed, step, particle_index, draw_index)
```

Up to ~5 draws per particle per step (translational x, translational y, rotational theta, anchor x, anchor y). `draw_index ∈ [0, 5)`.

**Critical caveat — CALL OUT IN README AND IN PSWEEP LOGS:**

> The GPU and CPU backends do **not** produce bit-identical trajectories for the same `seed`, because the RNG stream is structurally different (Mersenne Twister sequential draws vs. Philox per-particle parallel draws). Bit-reproducibility is preserved *within a backend* (CPU/CPU same-seed and GPU/GPU same-seed both match exactly) but **not across backends**. Equivalence is validated statistically — see [§7](#7-testing-strategy).

**Validation:** `test_rng_cuda.cpp` — draw 10⁶ samples, assert sample mean ∈ ±3σ of 0, sample variance ∈ ±3σ of 1, χ² goodness-of-fit on a binned histogram.

### 5.6 `Integrator` (Euler-Maruyama) — port-as-kernel [v1]

| Field | Value |
|---|---|
| CPU | [include/Integrator.hpp](../include/Integrator.hpp), [src/Integrator.cpp](../src/Integrator.cpp) |
| GPU | `cuda/include/IntegratorCuda.cuh`, `cuda/src/IntegratorCuda.cu`, `cuda/src/kernels/integrator_kernels.cu` |
| Class | port-as-kernel |

**Kernel sketch.** One thread per particle. Reads `(x, y, theta, vx, vy, ax, ay, fx, fy)`. Draws 5 Gaussians via Philox. Applies the EM update — translational drift + active drift + spring drift, capped to `max_drift_`, then translational noise; rotational diffusion update on theta; anchor update on `(ax, ay)`. Finally `BoxDevice::wrap` on position.

**Cached constants** ([include/Integrator.hpp:132-136](../include/Integrator.hpp#L132)) — `mobility_dt_`, `noise_amplitude_`, `rot_noise_amplitude_`, `anchor_mobility_dt_`, `anchor_noise_amplitude_` — are computed on host and bundled into an `IntegratorParams` POD struct passed to the kernel as a scalar argument.

**`max_drift_` cap** ([include/Integrator.hpp:127](../include/Integrator.hpp#L127)) — per-particle clamp inside the kernel, applied to the combined deterministic drift before noise is added. Matches the CPU semantics documented at [include/Integrator.hpp:44-49](../include/Integrator.hpp#L44).

**`kickStep`** ([include/Integrator.hpp:84-85](../include/Integrator.hpp#L84)) — also ported as a kernel; needed by the Heun corrector.

**Data movement.** None per step. The `IntegratorParams` POD is recomputed on host whenever a setter changes a parameter (`setKBT`, `setTimestep`, `setFriction`, `setAnchorFriction`, `setRotationalDiffusion`) and passed in by value.

**Validation:**

- `test_integrator_cuda` (deterministic): `kT = 0`, `D_r = 0`, one `step()`. Final state matches CPU within `1e-12`.
- `test_integrator_cuda_with_noise` (stochastic): ensemble of 10000 single-step runs from rest at `kT > 0`. Compare sample moments `⟨x²⟩`, `⟨y²⟩` to analytic `2*D*dt` within ±3σ.

### 5.7 `HeunIntegrator` — port-as-kernel [v1]

| Field | Value |
|---|---|
| CPU | [include/HeunIntegrator.hpp](../include/HeunIntegrator.hpp), [src/HeunIntegrator.cpp](../src/HeunIntegrator.cpp) |
| GPU | `cuda/include/HeunIntegratorCuda.cuh`, `cuda/src/HeunIntegratorCuda.cu` |
| Class | port-as-kernel |

Two-stage predictor-corrector:

1. **Predictor.** Compute forces at `(x, y)`. Half-step kernel with noise draws indexed by `(seed, step, particle, draw)` and `predictor_phase = 0`.
2. **Corrector.** Compute forces at predicted `(x', y')`. Full-step kernel with **the same noise draws** (re-derive from the same counter, just re-call Philox with the same indices). Heun is exact in the noise when both sub-steps use the same realization of the Wiener increment.

Force evaluation between predictor and corrector reuses the kernel from [§5.3](#53-forcecalculator--port-as-kernel-v1). CellList drift check happens once per macro step on entry.

**Why this is free with Philox.** A stateful generator would require either (a) saving generator state after the predictor and restoring it before the corrector, or (b) storing the noise samples in a temporary device array. Philox's counter-based addressing means we just call it twice with the same counter — same numbers, no extra memory traffic.

**Validation:** Same two patterns as Euler-Maruyama ([§5.6](#56-integrator-euler-maruyama--port-as-kernel-v1)).

### 5.8 `AdaptiveIntegrator` — deferred [v2]

| Field | Value |
|---|---|
| CPU | [include/AdaptiveIntegrator.hpp](../include/AdaptiveIntegrator.hpp), [src/AdaptiveIntegrator.cpp](../src/AdaptiveIntegrator.cpp) |
| GPU | none in v1 |
| Class | deferred |

The Boost.Odeint Cash-Karp 5(4) stepper inside `AdaptiveIntegrator::Impl` is non-trivial to port (custom adaptive controller, PI step-size logic, per-component error estimate reduction across particles). The pImpl boundary at [include/AdaptiveIntegrator.hpp:163-166](../include/AdaptiveIntegrator.hpp#L163) means a future v2 port replaces `Impl` without touching the public API. Same applies to `AdaptiveMacroIntegrator`.

`gpu_main.cpp` checks `cfg.integrator` early and returns a clear error if it's `adaptive_strang`:

```
Error: integrator 'adaptive_strang' is not supported on the CUDA backend
       in this release. Use 'euler_maruyama' or 'heun', or run on CPU
       via backend='cpu'.
```

### 5.9 `TrajectoryWriter` — stays on host [v1]

| Field | Value |
|---|---|
| CPU | [include/TrajectoryWriter.hpp](../include/TrajectoryWriter.hpp), [src/TrajectoryWriter.cpp](../src/TrajectoryWriter.cpp) |
| GPU | n/a — same code |
| Class | stays-on-host |

Every `output_dt`, `gpu_main.cpp` calls `SystemDevice::pullTo(host_sys)` then passes `host_sys` to the existing `TrajectoryWriter::writeFrame()` untouched.

The energy diagnostic in `main.cpp`'s output path needs forces on host; the GPU runner computes that on host after the pull (one O(N²) call per output, rare).

**Cost.** Per write: 9 × N × 8 bytes D2H. For N=10⁶ that's 72 MB. At `output_dt = 1.0` and `dt = 1e-3` (typical settings from [examples/input.json](../examples/input.json)) writes happen every 1000 steps — utterly negligible against the step loop's throughput.

**No HDF5 / h5cpp on device. Period.**

### 5.10 `CorrelationAccumulator` — stays on host [v1]

| Field | Value |
|---|---|
| CPU | [include/CorrelationAccumulator.hpp](../include/CorrelationAccumulator.hpp), [src/CorrelationAccumulator.cpp](../src/CorrelationAccumulator.cpp) |
| GPU | n/a — same code |
| Class | stays-on-host |

Same pattern: D2H copy at each `corr_dt`, then call the existing host-side `sample()`.

**Recommendation.** Configure correlation cadence so `corr_dt >= output_dt` whenever possible — that way the D2H needed for `TrajectoryWriter` doubles as the D2H for `CorrelationAccumulator`. If high-cadence correlation is required (`corr_dt << output_dt`), defer to a v2 on-device accumulator. v1 will work correctly with any cadence; it just pays a D2H per sample.

### 5.11 `ContactDurationAccumulator` — stays on host [v1]

| Field | Value |
|---|---|
| CPU | [include/ContactDurationAccumulator.hpp](../include/ContactDurationAccumulator.hpp), [src/ContactDurationAccumulator.cpp](../src/ContactDurationAccumulator.cpp) |
| GPU | n/a — same code |
| Class | stays-on-host |

The accumulator owns a private CellList that it rebuilds per sample (so contact detection sees the current geometry). On GPU this becomes the dominant non-step host cost when `compute_contact_durations=true` because the per-sample CellList rebuild is host-side O(N) at minimum.

**Mitigation [v1, recommended in the doc but not the code]:** if the sample cadence matches the step cadence (very common), reuse the same device cell list from the step loop — pull `cell_id_`, `cell_start_`, `cell_end_`, `nlist_start_`, `nlist_` back to host once per sample and avoid rebuilding the host accumulator's private cell list. Bookkeeping arrays for N=10⁶ are ~24 MB total — comparable cost to the trajectory D2H.

**[v2]:** Port the whole accumulator on-device. The `std::unordered_map<uint64_t, ContactEntry>` keyed on pair-id would have to become a sorted device array of `(pair_id, entry)` keyed on `thrust::sort_by_key`. Non-trivial; deferred.

**v1 guidance documented in the doc and surfaced as a warning at `sim_gpu` startup when `compute_contact_durations=true && N > 10^5`:**

```
Warning: compute_contact_durations adds significant host cost at large N
         on the CUDA backend. Consider disabling for N > 10^5 or running
         contact-duration analysis as a post-hoc pass on the trajectory.
```

### 5.12 `Initializer` — host-only [v1]

| Field | Value |
|---|---|
| CPU | [include/Initializer.hpp](../include/Initializer.hpp), [src/Initializer.cpp](../src/Initializer.cpp) |
| GPU | n/a |
| Class | stays-on-host |

Initial placement runs once before the step loop. Host code. After init, `SystemDevice::pushFrom(host_sys)` copies state to device. No change needed.

### 5.13 `Config` — host-only, small change [v1]

| Field | Value |
|---|---|
| CPU | [include/Config.hpp](../include/Config.hpp), [src/Config.cpp](../src/Config.cpp) |
| GPU | n/a — same code |
| Class | small extension |

Add a `Backend` enum and a `backend` field. **This is the only change to anything under [include/](../include/) or [src/](../src/) in v1.**

```cpp
enum class Backend { CPU, CUDA };
struct Config {
    // ... existing fields ...
    Backend backend = Backend::CPU;
};
```

`Config::fromFile()` reads the optional `"backend"` JSON field (default `"cpu"`). Allowed values `{"cpu", "cuda"}`. `Config::toJson()` emits it.

`main.cpp` rejects `Backend::CUDA` with `"Error: this binary (sim) was built without CUDA support. Use sim_gpu, or set backend=cpu."`. `gpu_main.cpp` rejects `Backend::CPU` with the symmetric message.

---

## 6. Schema + psweep changes

### 6.1 JSON schema additions

Exactly one new field, `"backend"`. Default `"cpu"`. Allowed values `{"cpu", "cuda"}`. Forbidden as a sweep variable.

Before:

```jsonc
{
    "N":   1000,
    "phi": 0.4,
    "integrator": "heun",
    "max_drift":  0.1,
    ...
}
```

After:

```jsonc
{
    "N":   1000,
    "phi": 0.4,
    "integrator": "heun",
    "backend":    "cpu",
    "max_drift":  0.1,
    ...
}
```

Existing JSONs remain valid (default is `"cpu"`).

### 6.2 `psweep.py` changes

**1. Allowed override + non-sweepable.** [psweep.py:73-118](../psweep.py#L73):

- Add to `NON_SWEEPABLE`: `"backend": "a hardware-backend choice, not a physical parameter"`.
- Add to `ALLOWED_VALUES`: `"backend": {"cpu", "cuda"}`.
- Add to `STRING_FIELDS`: `"backend"`.

**2. Binary selection.** [psweep.py:60](../psweep.py#L60) `DEFAULT_BINARY` becomes a function:

```python
def binary_for(backend):
    return os.path.join(BUILD_DIR, "sim_gpu" if backend == "cuda" else "sim")
```

The backend is read from the resolved JSON for each sweep (one backend per sweep — mixing is unsupported in v1).

**3. Build step.** [psweep.py:373-393](../psweep.py#L373):

- When the sweep is CUDA, pass `-DBUILD_CUDA=ON` to cmake and build the `sim_gpu` target.
- The CPU `sim` target is still built unconditionally (cheap, and useful for fallback verification).

**4. SLURM template.** [psweep.py:460-498](../psweep.py#L460):

When `backend == "cuda"`, inject:

```bash
#SBATCH --partition=<gpu-partition>     # required, no default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2               # or whatever the GPU partition requires
module load cuda/<version>              # configurable via --cuda-module flag
```

Add a new psweep CLI flag `--cuda-module <name>` (default: `"cuda/12.4"`, configurable per cluster).

**5. v1 restriction documented.** Mixing backends within a single sweep is unsupported in v1. The error is raised at `parse_overrides`: if `backend` is supplied, it must be a single value, not a sweep target. This matches the existing `NON_SWEEPABLE` enforcement pattern at [psweep.py:281-291](../psweep.py#L281).

### 6.3 Worked-example diff (SLURM body)

```bash
# Before (CPU)
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
build/sim run input.json

# After (CUDA)
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
module load cuda/12.4
build/sim_gpu run input.json
```

---

## 7. Testing strategy

Three-layer pyramid. Each layer must pass before merging.

### 7.1 Layer 1 — Per-component CPU-vs-GPU unit tests

GoogleTest binaries under `cuda/test/`, built only when `BUILD_CUDA=ON`. Each test builds a tiny system (N ≤ 128, fixed seed), runs the CPU reference, runs the GPU port, compares output.

| Test | What it pins down | Tolerance |
|---|---|---|
| `test_box_device.cpp` | `minimumImage`, `wrap` math bit-identical | `< 1e-15` |
| `test_system_device.cpp` | `pushFrom`/`pullTo` round-trip | exact |
| `test_rng_cuda.cpp` | Philox N(0,1) statistics | ±3σ |
| `test_force_calculator_cuda.cpp` | One `compute()` call, deterministic positions, both potentials, both brute-force and Verlet | `< 1e-12` element-wise on forces |
| `test_cell_list_cuda.cpp` | All four pipeline outputs (cell_id, cell_start, cell_end, nlist_start, nlist) | element-wise on cells; set-equality on rows |
| `test_integrator_cuda.cpp` (det) | One EM step at kT=0 | `< 1e-12` element-wise |
| `test_integrator_cuda.cpp` (stoch) | Ensemble of single-step runs, sample variance vs analytic 2Ddt | ±3σ |
| `test_heun_integrator_cuda.cpp` | Same two flavors as EM | same |

### 7.2 Layer 2 — End-to-end small-system regression

A single GoogleTest file, `cuda/test/test_equivalence.cpp`, that runs an honest-to-goodness multi-step simulation.

**Sub-test A — deterministic equivalence.** N=128, fixed seed, kT=0, both EM and Heun, 1000 steps. Final positions agree with CPU within `1e-10`.

**Sub-test B — stochastic equivalence (statistical).** N=128, fixed seed, kT>0, run for `t_end` long enough that the system equilibrates. Compare:

- `⟨U/N⟩` averaged over the trajectory: GPU vs CPU within 1% relative.
- Time-averaged g(r): GPU vs CPU within 1% per bin (over bins with at least 100 counts).
- Tolerance justified by CLT on the number of independent samples.

**This file is the gate.** No GPU port merges with a red `test_equivalence`.

### 7.3 Layer 3 — Cluster end-to-end via paired psweeps

Run the same sweep twice, once on CPU and once on GPU, then compare via the existing analysis pipeline ([python/analysis/](../python/analysis/)). Catalog of comparisons in [§8.3](#83-output-equivalence-vs-cpu).

### 7.4 Local-development reality

The Mac dev box cannot validate the GPU port. Be explicit:

- `BUILD_CUDA=OFF` is the default; macOS builds and runs all CPU tests exactly as today via `ctest --test-dir build -V`.
- All GPU iteration happens on the cluster (interactive `salloc` for tight loops, batch jobs for full runs).
- A small `cuda/ci/run_tests.sh` script (sketched in [§8.6](#86-final-go-no-go)) wraps "configure + build + ctest" for a one-line CI invocation.

### 7.5 Recommended addition: golden-output regression

No golden-output / regression-test fixtures exist today. **Before the GPU merges, introduce one** under `tests/golden/`:

- A small `.h5` produced by the CPU `sim` at a frozen seed and frozen input.
- A pytest in [tests/](../tests/) that runs a fresh `sim` (or `sim_gpu`) with the same input and asserts the resulting `.h5` matches the golden within statistical tolerance (same metrics as Layer 2, sub-test B).

This is the single biggest risk-reduction lever in the port — gives a "is the physics still right" signal that doesn't depend on any specific code path. The plan does **not** depend on golden-output existing, but it's the strongest signal you can build before the merge.

---

## 8. Cluster verification checklist

Numbered, runnable, top-to-bottom. Each item is a single command or a single inspection. The user (or whoever ships the port) executes this list end-to-end before declaring the GPU backend production-ready.

### 8.1 Build sanity

1. **Configure with CUDA on the cluster login (or compile) node:**
   ```
   cmake -S . -B build_gpu -DBUILD_CUDA=ON -DCMAKE_BUILD_TYPE=Release
   ```
2. **Build the GPU binary:**
   ```
   cmake --build build_gpu -j --target sim_gpu
   ```
   Expect: `build_gpu/sim_gpu` exists and is executable.
3. **Build the CPU binary in the same tree:**
   ```
   cmake --build build_gpu -j --target sim
   ```
   Expect: `build_gpu/sim` exists. Proves the CPU build still works inside a CUDA-enabled configuration.
4. **Confirm the GPU binary links CUDA runtime:**
   ```
   ldd build_gpu/sim_gpu | grep -E 'cudart|cuda'
   ```
   Expect: at least one matching line.
5. **Confirm the CPU binary is unpolluted:**
   ```
   ldd build_gpu/sim | grep -E 'cudart|cuda'
   ```
   Expect: **no output**. This is the mechanical proof of rule 1 of the [unpollution contract](#43-the-unpollution-contract).
6. **Confirm a CUDA-free machine still builds CPU:**
   ```
   cmake -S . -B build_cpu_only -DCMAKE_BUILD_TYPE=Release   # BUILD_CUDA defaults OFF
   cmake --build build_cpu_only -j
   ctest --test-dir build_cpu_only -V
   ```
   On macOS or any node without nvcc. Expect: green.
7. **Grep proof of rules 1 and 2:**
   ```
   ! grep -RE '#include[[:space:]]*<(cuda|thrust)' include/ src/ main.cpp
   ! grep -RE '#ifdef[[:space:]]+(CUDA|BUILD_CUDA)' include/ src/ main.cpp
   ! grep -RE '__(device|global|host)__' include/ src/ main.cpp
   ```
   Each command should exit with status 1 (no matches). Make this a one-liner in CI.

### 8.2 Launch sanity

8. **Allocate a single GPU interactively:**
   ```
   salloc --gres=gpu:1 -t 30:00
   ```
9. **Run a tiny job:** copy [examples/input.json](../examples/input.json) to a scratch dir, edit to set `"N": 1024`, `"t_end": 0.1`, `"output_dt": 0.01`, `"backend": "cuda"`. Then:
   ```
   build_gpu/sim_gpu run input.json
   ```
   Expect: exit 0; `trajectory_*.h5` written.
10. **Inspect the trajectory:**
    ```
    h5ls -r trajectory_*.h5
    ```
    Expect: 11 frames (t_end / output_dt + 1), each with the expected datasets and shapes.

### 8.3 Output equivalence vs CPU

11. **Run the same config on CPU:** flip `"backend"` to `"cpu"`, rename `output_file`, run `build_gpu/sim run input.json`.
12. **Compare in a Python notebook** (the analysis pipeline at [python/analysis/](../python/analysis/) is the right tool; sketch the cells):
    - **Per-frame `⟨U/N⟩`:** GPU vs CPU within **5%** relative (statistical — different RNG stream).
    - **Final-frame g(r):** GPU vs CPU within **1% per bin** after smoothing (bins with ≥100 counts).
    - **MSD slope (diffusion coefficient):** within **5%**.
    - **For Pe > 0, rotational correlation time:** within **5%**.
13. **Acceptance.** All four within tolerance → port is correct. Any one failing by >3× the tolerance → trace via the Layer 1 unit tests in [§7.1](#71-layer-1--per-component-cpu-vs-gpu-unit-tests).

### 8.4 Performance ceiling check

14. **Scaling sweep.** Run a fixed `t_end` at N ∈ {10³, 10⁴, 10⁵, 10⁶} on CPU and GPU. Wall-clock per step vs N. Expectation:
    - GPU break-even around N ~ 10⁴.
    - GPU dominant by 10–100× for N ≥ 10⁵.
    - GPU sub-linear in N (cell-list amortization).
15. **Profile a representative run:**
    ```
    nsys profile -o gpu_profile build_gpu/sim_gpu run input.json
    nsys stats gpu_profile.nsys-rep
    ```
    Confirm:
    - Force kernel is the dominant cost (>60% of GPU time).
    - **No H2D / D2H in the inner step loop** — only at `output_dt` and `corr_dt` ticks.
    - CellList rebuilds occur at the expected cadence (compare against `cl_rebuilds` reported by `sim_gpu` at exit).

### 8.5 Memory budget

16. **Sanity-check VRAM for target N.** For N = 10⁶:
    - 9 × N × 8 B SoA state on device = 72 MB.
    - CellList bookkeeping (`cell_id_`, `sorted_particle_ids_`, `cell_start_`, `cell_end_`, `nlist_start_`, 2× snapshots) ≈ 6 × N × 4 B + 2 × N × 8 B ≈ 40 MB.
    - Verlet list `nlist_` at mean ~12 neighbors ≈ 12 × N × 4 B ≈ 48 MB.
    - Total ≈ ~160 MB. A40 (48 GB) and A100 (40/80 GB) have room to spare.
17. **Confirm actual VRAM via `nvidia-smi`** during a real run.

### 8.6 Final go / no-go

18. ☐ Build sanity (1–7) all green.
19. ☐ Layer 1 tests all green (`ctest --test-dir build_gpu -L cuda -V`).
20. ☐ Layer 2 `test_equivalence` green.
21. ☐ Cluster launch sanity (8–10) green.
22. ☐ Output equivalence vs CPU (11–13) within tolerance for at least one test config.
23. ☐ Performance scaling (14–15) shows expected GPU dominance for large N.
24. ☐ VRAM budget verified for target N.
25. ☐ Recommended golden-output regression in place (or explicitly waived).

Sign off on all eight before declaring the port production-ready. Items 24 and 25 are the most likely to slip — call them out at planning time.

---

## 9. Open decisions / deferred items

Each item is **question → recommendation → rationale → cost of deferral**.

1. **Precision: double or float?**
   - **Recommend:** double for v1.
   - **Rationale:** matches CPU exactly except for the RNG stream; small-`kT` regimes (e.g. `1e-8` in [examples/input.json](../examples/input.json)) benefit from fp64 noise amplitude precision.
   - **Cost of deferral:** ~2× device memory and ~2× kernel time on hardware with fast fp32. Acceptable for v1.

2. **AdaptiveStrang on GPU?**
   - **Recommend:** v2.
   - **Rationale:** porting Boost.Odeint's Cash-Karp 5(4) plus PI controller is a large independent effort. The pImpl boundary at [include/AdaptiveIntegrator.hpp:163-166](../include/AdaptiveIntegrator.hpp#L163) preserves the API for a future swap.
   - **Cost of deferral:** users running adaptive sweeps stay on CPU. Workaround documented in the `sim_gpu` error message.

3. **Contact-duration accumulator on device?**
   - **Recommend:** v2.
   - **Rationale:** the `std::unordered_map` keyed pair-history is a substantial port (sort-based device dictionary). Host fallback works for v1 with the mitigation in [§5.11](#511-contactdurationaccumulator--stays-on-host-v1).
   - **Cost of deferral:** host CPU dominates the step loop at very high N when `compute_contact_durations=true`. Document and warn.

4. **Correlation accumulator on device?**
   - **Recommend:** v2.
   - **Cost of deferral:** D2H per `corr_dt`. Mitigate by aligning `corr_dt` to `output_dt`.

5. **Multi-GPU?**
   - **[oos]** in v1 and v2.
   - The isolation pattern in [§4](#4-isolation-strategy) doesn't preclude it (per-simulation can become per-device with no architectural change), but no work is planned.

6. **HIP / Apple Metal / SYCL?**
   - **[oos].** The `cuda/` shape extends — a `metal/` subtree next to it would mirror the same layout, with its own `sim_metal` binary and CMake gate.

7. **`thrust` vs hand-rolled scan / sort?**
   - **Recommend:** use thrust. Already a transitive dependency of any CUDA build; saves implementation time on stage 2 of the cell list and on `exclusive_scan` in stage 4.

8. **Small-box brute-force kernel fallback?**
   - **Recommend:** yes — port the brute-force kernel ([§5.3](#53-forcecalculator--port-as-kernel-v1)). Same dispatch decision the CPU makes ([include/CellList.hpp:56-63](../include/CellList.hpp#L56)).

9. **Cluster module-load story for HDF5 / Boost / nlohmann_json / CUDA?**
   - **Recommend:** document the cluster module loads in a section of the cluster README (separate doc, not this one). Standard pattern: `module load hdf5 boost cuda/12.4`. The cmake configuration auto-detects via `find_package` per [CMakeLists.txt:17-33](../CMakeLists.txt#L17), no code change needed.

10. **Mixed-backend sweeps in psweep?**
    - **Recommend:** unsupported in v1. The user runs two separate sweeps (CPU and CUDA) and compares them via the analysis pipeline. Mixing inside one sweep adds complexity for ~no benefit (the analysis pipeline is already device-agnostic on HDF5 output).

---

## 10. Implementation checklist

A linear PR sequence, smallest blast-radius first. Each item is a deliverable; nothing depends on a later item.

1. **JSON `backend` field.** Add to [include/Config.hpp](../include/Config.hpp), [src/Config.cpp](../src/Config.cpp), [examples/input.json](../examples/input.json), [psweep.py](../psweep.py). Pure CPU change. Gated only on existing CPU tests still passing. (~1 day.)

2. **CMake skeleton for `cuda/`.** Add `option(BUILD_CUDA OFF)` and a guarded `add_subdirectory(cuda)` to [CMakeLists.txt](../CMakeLists.txt). Create `cuda/CMakeLists.txt` with `enable_language(CUDA)`, declare empty `brownian_lib_cuda` target. No `.cu` files yet. Verifies CPU build is byte-identical with `BUILD_CUDA=OFF`. (~1 day.)

3. **`SystemDevice` + `BoxDevice` + first green GPU test.** Implement `cuda/include/SystemDevice.cuh`, `cuda/include/BoxDevice.cuh`, a no-op `cuda/src/kernels/integrator_kernels.cu` to exercise the toolchain. Add `cuda/test/CMakeLists.txt`, `test_box_device.cpp`, `test_system_device.cpp`. **Milestone:** first green CUDA test runs on the cluster. (~2 days.)

4. **`ForceCalculatorCuda`** + `test_force_calculator_cuda.cpp`. Both potentials, both brute-force and Verlet overloads. (~3 days.)

5. **`CellListCuda`** (all four stages) + `test_cell_list_cuda.cpp`. (~4 days.)

6. **`RandomCuda` (Philox)** + `test_rng_cuda.cpp`. (~2 days.)

7. **`IntegratorCuda`** (Euler-Maruyama) + `test_integrator_cuda.cpp` (deterministic + stochastic). (~3 days.)

8. **`HeunIntegratorCuda`** + `test_heun_integrator_cuda.cpp`. (~3 days.)

9. **`gpu_main.cpp`** + `sim_gpu` target + `test_equivalence.cpp`. Wire `TrajectoryWriter`, `CorrelationAccumulator`, `ContactDurationAccumulator` via D2H at sample/output cadence. **Milestone:** end-to-end small-system regression passes. (~3 days.)

10. **`psweep.py` GPU plumbing.** Binary selection, build target switch, SLURM template injection, `--cuda-module` flag. (~1 day.)

11. **Cluster verification checklist** from [§8](#8-cluster-verification-checklist). Run end-to-end; patch anything that fails; record results. Add the recommended golden-output regression from [§7.5](#75-recommended-addition-golden-output-regression) in the same PR or a follow-up. (~3 days.)

Rough total: ~25 days of focused work, plus the cluster verification tail. Treat as a rough sequence indicator, not a commitment.
