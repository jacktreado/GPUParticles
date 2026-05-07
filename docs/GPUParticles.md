# GPU-Accelerated Multiparticle Simulations in C++ — A Research Reference

A curated guide to the state of the art in GPU-accelerated Discrete Element Method (DEM) and Molecular Dynamics (MD), the algorithms behind fast neighbor / contact detection, and the C++ libraries that make CUDA bearable.

> **How to use this document:** Skim the overview first, then jump into whichever section matches your immediate need. Every named tool/paper has a working link. The recommendations at the end will help us collaborate more effectively.

---

## Table of Contents

1. [Why GPUs Win for Particle Simulations](#1-why-gpus-win-for-particle-simulations)
2. [State-of-the-Art Open-Source Codes](#2-state-of-the-art-open-source-codes)
3. [Algorithms for Particle–Particle Distance Detection on GPUs](#3-algorithms-for-particleparticle-distance-detection-on-gpus)
4. [Making CUDA C++ Less Painful: Modern Libraries & Abstractions](#4-making-cuda-c-less-painful-modern-libraries--abstractions)
5. [Performance Patterns Every GPU Particle Code Lives or Dies By](#5-performance-patterns-every-gpu-particle-code-lives-or-dies-by)
6. [Annotated Code Snippets](#6-annotated-code-snippets)
7. [Recommended Learning Path](#7-recommended-learning-path)
8. [What I Need From You to Help You Better](#8-what-i-need-from-you-to-help-you-better)

---

## 1. Why GPUs Win for Particle Simulations

Particle methods (DEM, MD, SPH, etc.) are dominated by two costs:

- **Force / contact computation** — pair interactions evaluated for every nearby particle pair. This is *embarrassingly parallel* across particles.
- **Neighbor search** — finding *who is near whom* every few timesteps. This is the hard part because it involves irregular memory access and dynamic data structures.

GPUs are spectacularly good at the first and *can* be very good at the second if you respect their architecture (lots of threads, narrow cache, fast shared memory, hostile to branchy code). A typical DEM timestep is `O(10^-6 – 10^-5) s` so simulations need millions of steps; even modest per-step speedups compound enormously.

For context, codes like **DEM-Engine** can run 150 million elements on two NVIDIA A100s with roughly linear scaling, and **HOOMD-blue** has demonstrated 108-million-particle simulations across 3,375 GPUs. A single modern GPU often beats 64–128 CPU cores on Lennard-Jones-style benchmarks.

---

## 2. State-of-the-Art Open-Source Codes

These are the codes worth reading source for. They each represent different design philosophies.

### 2.1 HOOMD-blue (University of Michigan, Glotzer group)

- **Repo:** <https://github.com/glotzerlab/hoomd-blue>
- **Paper:** <https://arxiv.org/abs/1308.5587>
- **Strong-scaling paper:** <https://arxiv.org/abs/1412.3387>
- **Faceted-DEM extension:** <https://arxiv.org/abs/1607.02427>

Built **GPU-first** from 2007 — every step lives on the GPU, no host round-trips per timestep. Python frontend, C++/CUDA backend. Good place to study a well-engineered cell-list neighbor search and BVH alternative.

### 2.2 LAMMPS (Sandia) — KOKKOS package

- **Repo:** <https://github.com/lammps/lammps>
- **LAMMPS-KOKKOS paper (2025):** <https://arxiv.org/abs/2508.13523>

LAMMPS has two GPU strategies side-by-side:
- The older **GPU package** offloads only the force kernel; everything else runs on host. Simple, but bottlenecked by PCIe traffic each step.
- The newer **KOKKOS package** keeps state on the device. Performance-portable across NVIDIA / AMD / Intel GPUs through Kokkos. This is the design you should study if you want one codebase running on Frontier, El Capitan, *and* Aurora.

### 2.3 Project Chrono — DEM-Engine (UW Madison, Negrut group)

- **Repo:** <https://github.com/projectchrono/chrono> (DEM-Engine submodule)
- **Paper:** <https://arxiv.org/abs/2311.04648>
- **Clump-DEM paper:** <https://arxiv.org/abs/2307.03445>

Genuinely interesting design choices for DEM specifically:
- **Dual-GPU architecture** with two asynchronous CUDA streams.
- A **delayed contact detection** algorithm that decouples contact discovery from force evaluation, so the two run on separate streams.
- **JIT compilation** of user-defined contact force models (you write a custom force law in C++ at runtime).
- **Compressed custom data types** (bit-packed) to maximize bandwidth.

If your interest is granular materials specifically, this is the closest thing to a reference design at the moment.

### 2.4 MUSEN (Hamburg University of Technology)

- **Repo:** <https://github.com/msolids/musen>
- **SoftwareX paper:** <https://www.sciencedirect.com/science/article/pii/S2352711020303319>

C++17 + CUDA + Qt GUI. Designed to run on a workstation, not a cluster. Modular contact-model architecture makes it a good educational base; the GUI lowers the barrier to "just run a simulation."

### 2.5 BlazeDEM3D-GPU

- **Repo:** <https://github.com/ElsevierSoftwareX/SOFTX-D-15-00085>
- **Paper:** <https://www.sciencedirect.com/science/article/pii/S235271101630005X>

Notable for native polyhedral particle support (not just sphere clumps), millions of polyhedra on a single workstation GPU. Worth studying for non-spherical contact detection.

### 2.6 GROMACS

- **Repo:** <https://gitlab.com/gromacs/gromacs>
- **SYCL vs CUDA paper:** <https://arxiv.org/abs/2406.10362>

Mainly biomolecular MD. Now supports CUDA, OpenCL, and SYCL backends. Useful as a case study of how a long-lived code transitions to vendor-portable GPU programming.

### 2.7 NVIDIA Warp (Python, not C++ — but worth your time)

- **Repo:** <https://github.com/NVIDIA/warp>
- **Docs:** <https://nvidia.github.io/warp/>
- **Intro blog:** <https://developer.nvidia.com/blog/creating-differentiable-graphics-and-physics-simulation-in-python-with-nvidia-warp/>

Warp lets you write GPU kernels as decorated Python functions that JIT-compile to CUDA. It offers built-in **hash grids** for neighbor search and **automatic differentiation** of simulation code (huge for ML-coupled workflows). It's not C++, but the concepts (hash grids, kernel launches) translate directly, and it's a *very* low-friction way to prototype an algorithm before porting to CUDA C++.

---

## 3. Algorithms for Particle–Particle Distance Detection on GPUs

This is the heart of your question. The naive `O(N²)` approach is fine up to a few thousand particles; past that, you need spatial acceleration. Below are the dominant approaches in roughly increasing order of cleverness.

### 3.1 Cell Linked Lists (CLL) — the workhorse

Partition the simulation domain into a uniform grid where each cell side is at least `r_cut` (the interaction cutoff). Then every potential neighbor of a particle in cell `c` lies in `c` or its 26 neighbors (in 3D). Build complexity: `O(N)`. Search cost per particle: `O(particles_in_27_cells)`.

The classic GPU build uses one thread per particle plus `atomicAdd` to insert into per-cell counters (Simon Green's NVIDIA particle SDK demo, 2007/2010, set the template):
- <https://developer.download.nvidia.com/compute/DevZone/C/html_x64/5_Simulations/particles/doc/particles.pdf>

**Weakness:** if `r_cut` varies between particle types (size-disparate systems), the grid wastes huge volumes — only ~16% of the explored cell volume actually lies within `r_cut`.

### 3.2 Stenciled Cell Lists

Reduce wasted distance checks by precomputing a **stencil** of which cell offsets must be examined for each cutoff radius. Especially helpful with mixed-size particles. See Howard et al. (2016): <https://www.sciencedirect.com/science/article/abs/pii/S0010465516300182>.

### 3.3 Verlet Neighbor Lists with Skin Distance

Build an explicit list of neighbor pairs within `r_cut + r_skin`. Use it for many timesteps until particles have moved more than `r_skin/2`, then rebuild. On GPU, two flavors compete:
- **Thread-per-atom**: one thread iterates that atom's neighbor list. Good when neighbor counts are low.
- **Block-per-atom (warp-per-atom)**: a warp cooperates on one atom's neighbors. Better at high coordination.

LAMMPS's GPU package documents this trade-off carefully:
- <https://arxiv.org/abs/1009.4330>

### 3.4 Hybrid Neighbor + Cell Lists

Combine both: a cell list as scaffolding, a Verlet list as the actual force-loop input. Reduces the spurious distance checks of pure cell lists while keeping the cheap rebuilds. ~10% wins reported on top of well-tuned standalone implementations:
- <https://dl.acm.org/doi/10.1145/2506583.2506649>

### 3.5 Spatial Hashing

Instead of an explicit cell array, hash particle positions into buckets. Memory-efficient for sparse domains. The classic Le Grand stencil-hash design appears in many SPH/cloth simulators:
- <https://min-tang.github.io/home/PSCC/files/pscc.pdf>

NVIDIA Warp's `wp.HashGrid` is a polished, ready-to-use implementation of this idea.

### 3.6 Linear Bounding Volume Hierarchies (LBVH)

Tree-based, borrowed from ray tracing. Each particle is wrapped in an AABB; the tree is built top-down via Morton-code sorting. Traversal is O(log N) per query. **LBVH beats cell lists when interaction radii vary widely** (colloidal systems, multi-scale astrophysics, cloth):
- Howard et al. 2016: <https://www.sciencedirect.com/science/article/abs/pii/S0010465516300182>

### 3.7 Quantized BVH (QBVH)

Howard et al. (2019) compressed the BVH nodes to low precision to make traversal cache-friendly. Roughly **2–4× faster than uniform-grid cell lists** across common MD benchmarks:
- Paper: <https://arxiv.org/abs/1901.08088>

This is currently the recommendation for new GPU MD codes that don't have specialized requirements.

### 3.8 Space-Filling Curve (SFC) Compressed Lists

Recent (2026) work uses Morton/Hilbert curves as a data layout, supporting *per-particle* cutoff radii, integrating cleanly with octree domain decomposition for distributed simulations:
- <https://arxiv.org/abs/2602.19873> (preprint, "GPU-Native Compressed Neighbor Lists with a Space-Filling-Curve Data Layout")

### 3.9 Delayed Contact Detection (Chrono DEM-Engine)

Specific to DEM. Splits the simulation into two asynchronous GPU streams — one detects contacts, one computes forces — using a *deliberately stale* contact list with a safety buffer. The two streams overlap, hiding latency. This is described in <https://arxiv.org/abs/2311.04648>.

### 3.10 Random-Batch Neighbor Lists

For very large MD systems, fully constructing the neighbor list is itself expensive. Random-batch methods (RBMD package) sample stochastic subsets of pairs, achieving linear complexity for long-range interactions. Trade-off: introduces controlled noise. See <https://www.researchgate.net/publication/222648138_Improved_neighbor_list_algorithm_in_molecular_simulations_using_cell_decomposition_and_data_sorting_method> for context and follow citations forward.

### Summary — which to pick?

| System type                          | Best default                  |
|--------------------------------------|-------------------------------|
| Monodisperse spheres, dense          | Stenciled cell list           |
| Polydisperse / size-disparate        | Quantized BVH                 |
| Cloth / soft-body / triangle meshes  | Spatial hashing + BVH hybrid  |
| Granular DEM with custom force laws  | Cell list + delayed detection |
| Multi-scale astrophysics             | SFC-compressed neighbor list  |

---

## 4. Making CUDA C++ Less Painful: Modern Libraries & Abstractions

You're right that raw CUDA is rough — manual memory management, error codes everywhere, no RAII, vendor lock-in. The C++ ecosystem has multiple answers, ranging from "thin wrappers" to "full abstraction layers."

### 4.1 CCCL — CUDA Core Compute Libraries (NVIDIA, the modern default)

- **Repo:** <https://github.com/NVIDIA/cccl>

CCCL bundles three previously-separate projects:

- **Thrust** — high-level parallel algorithms (`sort`, `reduce`, `scan`, `transform`) over `device_vector`. Looks like the C++ STL but runs on GPU. Has CUDA, OpenMP, and TBB backends — same code can target multicore CPUs.
- **CUB** — lower-level, CUDA-specific cooperative primitives (block-wide sort, warp-wide reductions). Use this when Thrust isn't fast enough.
- **libcudacxx** — a CUDA-compatible implementation of the C++ Standard Library (`<atomic>`, `<chrono>`, `<barrier>`, etc.) usable in `__device__` code.

This is the *first* place to look when you'd otherwise write a custom kernel for a standard parallel pattern.

### 4.2 cuda-api-wrappers (Eyal Roz)

- **Repo:** <https://github.com/eyalroz/cuda-api-wrappers>

Header-only, thin RAII wrappers around CUDA Runtime + Driver APIs. No magic, no caches, no behind-your-back behavior. You get streams, events, kernels, and contexts as proper C++ objects with destructors. Great middle ground between "raw CUDA" and "Thrust."

### 4.3 cudawrappers (Netherlands eScience Center)

- **Repo:** <https://github.com/nlesc-recruit/cudawrappers>
- **Docs:** <https://cudawrappers.readthedocs.io>

Similar philosophy to cuda-api-wrappers but with broader coverage of cuFFT, NVRTC (runtime kernel compilation), and HIP. Header-only, CMake-friendly.

### 4.4 RAPIDS Memory Manager (RMM)

- **Repo:** <https://github.com/rapidsai/rmm>
- **Blog:** <https://developer.nvidia.com/blog/fast-flexible-allocation-for-cuda-with-rapids-memory-manager/>

A pluggable memory-allocator framework. `cudaMalloc` is slow; pool allocators dramatically reduce per-step allocation cost in particle codes that resize neighbor lists. Provides RAII `cuda_stream`, `device_buffer`, etc.

### 4.5 Kokkos (Sandia) — performance-portable C++

- **Repo:** <https://github.com/kokkos/kokkos>
- **Comparative paper (2024):** <https://arxiv.org/abs/2402.08950>

A C++ template library that compiles the same source to CUDA, HIP (AMD), SYCL (Intel), OpenMP, HPX, or plain pthreads. You write `Kokkos::parallel_for` with a lambda; Kokkos picks the right execution space. The `View<T*>` type abstracts memory layout (AoS vs SoA, row- vs column-major) so the same code can be optimal on different hardware. **This is the abstraction LAMMPS chose for its modern GPU port.**

Trade-off: another dependency, smaller community than raw CUDA, learning curve for views and execution policies.

### 4.6 SYCL / oneAPI

- **Spec:** <https://www.khronos.org/sycl/>

A Khronos open standard for single-source C++ heterogeneous programming. Intel's main horse (DPC++/oneAPI). Implementations: Intel oneAPI, AdaptiveCpp (formerly hipSYCL), Codeplay ComputeCpp. Increasingly viable on NVIDIA hardware too. GROMACS now ships a SYCL backend.

### 4.7 RAJA (Lawrence Livermore)

- **Repo:** <https://github.com/LLNL/RAJA>

LLNL's answer in the same space as Kokkos. Less type-system-heavy; centered on parallel-for execution policies. Often paired with **Umpire** (memory) and **CHAI** (managed array) for a full stack.

### 4.8 alpaka (HZDR)

- **Repo:** <https://github.com/alpaka-group/alpaka>

C++ header-only abstraction at a lower level than Kokkos — closer in shape to CUDA/HIP itself, so easier porting from existing CUDA code. Backends: CUDA, HIP, SYCL, OpenMP, TBB.

### 4.9 ArrayFire

- **Site:** <https://arrayfire.com/>

A higher-level "MATLAB-feeling" array library with CUDA / OpenCL / CPU backends. Less common in particle simulation, but worth knowing.

### Quick decision tree

- **Want a single-vendor solution, willing to commit to NVIDIA?** → Raw CUDA + CCCL + cuda-api-wrappers + RMM.
- **Want vendor portability for HPC?** → Kokkos (mature, widely used) or RAJA.
- **Want a future-proof open standard?** → SYCL via AdaptiveCpp.
- **Want maximum prototyping speed and ML interop?** → NVIDIA Warp (Python).

---

## 5. Performance Patterns Every GPU Particle Code Lives or Dies By

These are the rules every code in section 2 obeys. If you violate them, no clever algorithm will save you.

### 5.1 Structure of Arrays (SoA), not Array of Structures (AoS)

```cpp
// BAD on GPU — each thread reads strided memory
struct Particle { float x, y, z, vx, vy, vz; };
Particle particles[N];

// GOOD on GPU — coalesced access
struct Particles {
    float* x;  float* y;  float* z;
    float* vx; float* vy; float* vz;
};
```

When 32 threads in a warp each load `particles[i].x`, AoS produces a strided access pattern; SoA produces a single contiguous 128-byte memory transaction. This routinely costs 3–8× performance on its own. Fast N-body code in NVIDIA's own tutorial uses `float4` precisely so memory accesses coalesce: <https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda>.

### 5.2 Avoid Atomic Contention; Compute Forces Twice

When two particles interact, the symmetric force `F_ij = -F_ji` means each particle's force gets a contribution. Naively you'd compute it once and `atomicAdd` to both — but heavy atomic contention kills GPU performance. HOOMD-blue, LAMMPS, NAMD, and OxDNA all **compute pair forces twice** (once from each particle's perspective) to skip atomics entirely. This is a real, intentional design choice:
- See <https://ar5iv.labs.arxiv.org/html/1401.4350>

When atomics are unavoidable, modern Kepler+ hardware makes them cheap enough that they often beat alternatives. Use warp shuffles (`__shfl_*`) to reduce within a warp before going atomic:
- <https://developer.nvidia.com/blog/voting-and-shuffling-optimize-atomic-operations/>

### 5.3 Sort Particles Spatially (Z-order / Morton) Periodically

After a few hundred timesteps, particle indices and spatial positions decorrelate; cache locality drops. Sorting particles by their Morton code (or cell index) every N steps restores spatial locality and speeds *every* subsequent operation. HOOMD-blue and LAMMPS-KOKKOS both do this.

### 5.4 Keep Everything on the Device

PCIe transfers are death. Don't copy positions back to the host every step. The whole reason the KOKKOS package in LAMMPS outperforms the older GPU package is that KOKKOS keeps state on the device while GPU package round-trips per timestep.

### 5.5 Persistent Kernels and Stream Overlap

Launch one big kernel that does many particles' work, and overlap independent kernels on separate CUDA streams. Chrono DEM-Engine's two-stream design is the canonical example.

### 5.6 Mind Your Floating Point

DEM with stiff contacts may need `double` for stability; LJ MD usually thrives in `float` or mixed precision. Modern NVIDIA GPUs (H100, GH200) have ~34 TFLOPS FP64, but consumer GPUs (RTX series) are crippled in double precision — design for your target hardware.

### 5.7 Profile Religiously

`nvprof` is deprecated; use **NVIDIA Nsight Compute** (kernel-level) and **Nsight Systems** (timeline-level). Don't guess what's slow — measure.

---

## 6. Annotated Code Snippets

These are minimal illustrative examples, not production code.

### 6.1 The "Hello World" of CUDA particle simulation: Euler integration

```cpp
// integrate.cu  — one CUDA thread per particle
#include <cuda_runtime.h>

__global__ void integrate(float4* pos, float4* vel,
                          const float4* force, float dt, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Load (coalesced because float4 is 16-byte aligned)
    float4 p = pos[i];
    float4 v = vel[i];
    float4 f = force[i];

    // Velocity-Verlet style half step
    v.x += f.x * dt;  v.y += f.y * dt;  v.z += f.z * dt;
    p.x += v.x * dt;  p.y += v.y * dt;  p.z += v.z * dt;

    pos[i] = p;
    vel[i] = v;
}

void launch_integrate(float4* d_pos, float4* d_vel,
                      const float4* d_force, float dt, int N)
{
    int block = 256;
    int grid  = (N + block - 1) / block;
    integrate<<<grid, block>>>(d_pos, d_vel, d_force, dt, N);
}
```

Notes: `float4` because the `w` component packs mass (or charge, or radius) and the alignment guarantees coalesced loads. Three floats `(x, y, z)` would *not* coalesce as well.

### 6.2 Building a cell list with atomicAdd (sketch)

```cpp
// Step 1 — count particles per cell
__global__ void count_cells(const float4* pos, int* cell_count,
                            float3 origin, float cell_size,
                            int3 grid_dim, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int cx = (int)((pos[i].x - origin.x) / cell_size);
    int cy = (int)((pos[i].y - origin.y) / cell_size);
    int cz = (int)((pos[i].z - origin.z) / cell_size);
    int c  = (cz * grid_dim.y + cy) * grid_dim.x + cx;

    atomicAdd(&cell_count[c], 1);
}

// Step 2 — exclusive prefix-sum cell_count -> cell_start (use Thrust or CUB)
//           thrust::exclusive_scan(cell_count, cell_count+ncells, cell_start);

// Step 3 — scatter particle indices into per-cell buckets
__global__ void fill_cells(const float4* pos, const int* cell_start,
                           int* cell_count_running, int* particle_in_cell,
                           float3 origin, float cell_size,
                           int3 grid_dim, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int cx = (int)((pos[i].x - origin.x) / cell_size);
    int cy = (int)((pos[i].y - origin.y) / cell_size);
    int cz = (int)((pos[i].z - origin.z) / cell_size);
    int c  = (cz * grid_dim.y + cy) * grid_dim.x + cx;

    int slot = atomicAdd(&cell_count_running[c], 1);
    particle_in_cell[cell_start[c] + slot] = i;
}
```

This is the classic scatter pattern — Simon Green's NVIDIA tutorial uses essentially this structure. In production you'd use **Thrust** to do the exclusive scan and a sort-based variant of the same idea to avoid the second atomic pass entirely.

### 6.3 Same thing in Thrust + CUB style (much shorter)

```cpp
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

// Sort particles by cell index — net effect: particles in the same cell
// are contiguous in memory, and we can find each cell's start with
// a single lower_bound.
thrust::device_vector<int>    cell_idx(N);
thrust::device_vector<float4> pos(N);

// 1. compute cell index for every particle (kernel or thrust::transform)
// 2. sort positions by cell_idx
thrust::sort_by_key(cell_idx.begin(), cell_idx.end(), pos.begin());
// 3. find start of each cell in O(log N) per cell using lower_bound
```

This pattern — **sort by cell, then index by binary search** — is what HOOMD-blue and many production codes actually do, because it has zero atomics and great memory access patterns.

### 6.4 Skeleton of a Lennard-Jones force kernel using a neighbor list

```cpp
__global__ void lj_forces(const float4* __restrict__ pos,
                          float4*       __restrict__ force,
                          const int*    __restrict__ nlist,    // packed
                          const int*    __restrict__ nlist_start,
                          int N, float r_cut2, float eps, float sig2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 pi = pos[i];
    float fx = 0.f, fy = 0.f, fz = 0.f;

    int begin = nlist_start[i];
    int end   = nlist_start[i + 1];

    for (int k = begin; k < end; ++k) {
        int j = nlist[k];
        float4 pj = pos[j];

        float dx = pj.x - pi.x;
        float dy = pj.y - pi.y;
        float dz = pj.z - pi.z;
        float r2 = dx*dx + dy*dy + dz*dz;

        if (r2 < r_cut2 && r2 > 0.f) {
            float inv_r2  = 1.f / r2;
            float sr2     = sig2 * inv_r2;
            float sr6     = sr2 * sr2 * sr2;
            float f_over_r = 24.f * eps * sr6 * (2.f * sr6 - 1.f) * inv_r2;
            fx += f_over_r * dx;
            fy += f_over_r * dy;
            fz += f_over_r * dz;
        }
    }
    // Two-pass design (each pair seen twice) — no atomics needed.
    force[i] = make_float4(fx, fy, fz, 0.f);
}
```

Note `__restrict__` (lets the compiler assume no aliasing) and the use of compressed neighbor lists (a single packed array indexed by `nlist_start`).

### 6.5 The same problem in NVIDIA Warp (Python, but generates CUDA)

```python
import warp as wp

@wp.kernel
def lj_forces(pos: wp.array(dtype=wp.vec3),
              force: wp.array(dtype=wp.vec3),
              grid: wp.uint64,        # hash grid handle
              r_cut: float, eps: float, sig: float):
    i = wp.tid()
    pi = pos[i]
    f  = wp.vec3(0.0, 0.0, 0.0)

    neighbors = wp.hash_grid_query(grid, pi, r_cut)
    for j in neighbors:
        if j == i: continue
        d  = pos[j] - pi
        r2 = wp.dot(d, d)
        if r2 < r_cut * r_cut and r2 > 0.0:
            inv_r2 = 1.0 / r2
            sr6    = (sig * sig * inv_r2) ** 3.0
            mag    = 24.0 * eps * sr6 * (2.0 * sr6 - 1.0) * inv_r2
            f      = f + mag * d
    force[i] = f
```

Same algorithm, dramatically less ceremony. Useful for prototyping; you can profile and convert hot paths to CUDA C++ later.

---

## 7. Recommended Learning Path

If you're starting from "comfortable C++ developer, new to CUDA":

1. **Read NVIDIA's Mark Harris blog series** on the NVIDIA developer blog — coalescing, shared memory, reductions. Short, high signal.
2. **Walk through the original NVIDIA particle SDK example** (Simon Green) — it's old but the algorithm structure is still canonical: <https://developer.download.nvidia.com/compute/DevZone/C/html_x64/5_Simulations/particles/doc/particles.pdf>
3. **Read the HOOMD-blue overview paper** to see what a top-tier GPU MD code's *architecture* looks like: <https://arxiv.org/abs/1308.5587>
4. **Read the Howard et al. quantized-BVH paper** to understand modern neighbor search: <https://arxiv.org/abs/1901.08088>
5. **Read the Chrono DEM-Engine paper** if your interest is granular DEM specifically: <https://arxiv.org/abs/2311.04648>
6. **Build a toy problem in Warp first** (an hour or two, in Python) to validate algorithm choices before you commit to CUDA C++.
7. **Then build the real thing** in CUDA C++ + CCCL + cuda-api-wrappers, or jump straight to Kokkos if you want vendor portability.

Bonus reading on the abstraction-layer landscape:
- <https://arxiv.org/abs/2402.08950> — solid empirical comparison of CUDA, HIP, Kokkos, RAJA, OpenMP, OpenACC, and SYCL.

---

## 8. What I Need From You to Help You Better

To collaborate on building this with you, the more of the following I know, the more useful I can be:

**About the problem:**
- DEM, MD, SPH, or something else? Each implies different defaults (DEM cares about contact stiffness, MD cares about long-range Coulomb, etc.)
- Particle count: thousands, millions, hundreds of millions? This changes every algorithmic choice.
- Particle shapes: spheres, sphere clumps, polyhedra, faceted, anisotropic?
- Force model: Lennard-Jones, Hertzian contact, Hooke + friction, custom? Custom force laws steer the design *a lot*.
- Boundary conditions: periodic, walls, complex meshed geometry?
- Time scale: a single relaxation, or millions of steps? (DEM is almost always the latter.)
- Single precision OK, or do you need double?

**About the hardware:**
- NVIDIA only, or do you need AMD / Intel portability? (This decides CUDA-vs-Kokkos.)
- Single GPU, single node multi-GPU, or multi-node cluster?
- Specific GPU model(s) you're targeting? FP64 throughput varies wildly across consumer/datacenter cards.

**About you:**
- Existing CUDA experience level?
- Existing C++ comfort: C++11, C++17, C++20, templates, modern idioms?
- Build-system preferences: CMake, Bazel, Make, something else?
- Are you writing this from scratch, extending an existing code, or replacing a CPU implementation you already have?

**About scope:**
- Is this a research prototype, a production tool, or a teaching project?
- Do you need visualization (live or offline)?
- Will you couple this to ML / differentiable physics? (If so, NVIDIA Warp becomes very attractive.)
- What's your timeline?

The single most useful thing you could do is point me at a small example problem you'd like to solve — even a synthetic one — so we can ground the design choices in something concrete. From there we can sketch a project structure, pick the right neighbor-search algorithm for your regime, and iterate.

---

## Appendix: Quick Reference Links

| Resource | Link |
|---|---|
| HOOMD-blue | <https://github.com/glotzerlab/hoomd-blue> |
| LAMMPS | <https://github.com/lammps/lammps> |
| Project Chrono | <https://github.com/projectchrono/chrono> |
| MUSEN | <https://github.com/msolids/musen> |
| BlazeDEM3D-GPU | <https://github.com/ElsevierSoftwareX/SOFTX-D-15-00085> |
| GROMACS | <https://gitlab.com/gromacs/gromacs> |
| NVIDIA Warp | <https://github.com/NVIDIA/warp> |
| CCCL (Thrust/CUB/libcudacxx) | <https://github.com/NVIDIA/cccl> |
| cuda-api-wrappers | <https://github.com/eyalroz/cuda-api-wrappers> |
| cudawrappers | <https://github.com/nlesc-recruit/cudawrappers> |
| RAPIDS Memory Manager | <https://github.com/rapidsai/rmm> |
| Kokkos | <https://github.com/kokkos/kokkos> |
| RAJA | <https://github.com/LLNL/RAJA> |
| alpaka | <https://github.com/alpaka-group/alpaka> |
| AdaptiveCpp (SYCL) | <https://github.com/AdaptiveCpp/AdaptiveCpp> |
| NVIDIA Developer Blog | <https://developer.nvidia.com/blog/> |
| GPU Gems N-body chapter | <https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda> |