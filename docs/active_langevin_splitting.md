# Adaptive Integration of Active Langevin Dynamics via Operator Splitting

This document explains how to combine an **adaptive deterministic integrator** (such as `boost::numeric::odeint::runge_kutta_cash_karp54`) with the **stochastic updates** required for a Langevin-thermostatted active matter simulation. The trick is operator splitting, which lets each subsystem be integrated by the method best suited to it.

---

## 1. Problem Setup

Consider $N$ self-propelled particles in $d$ dimensions. Each particle has a position $\mathbf{r}_i$, a velocity $\mathbf{v}_i$, and a body-frame orientation unit vector $\hat{\mathbf{e}}_i$ (in 2D this is a single angle $\theta_i$; in 3D a unit vector on $S^2$ or a quaternion). The equations of motion are

$$
\begin{aligned}
d\mathbf{r}_i &= \mathbf{v}_i\, dt, \\
m\, d\mathbf{v}_i &= \underbrace{\mathbf{F}_i^{\text{int}}\!\left(\{\mathbf{r}_j\}\right) dt}_{\text{pair forces (e.g. LJ)}} \;+\; \underbrace{f_a\, \hat{\mathbf{e}}_i\, dt}_{\text{active force}} \;\underbrace{-\,\gamma\, \mathbf{v}_i\, dt + \sqrt{2 \gamma k_B T}\, d\mathbf{W}_i^t}_{\text{Langevin thermostat}}, \\
d\hat{\mathbf{e}}_i &= \sqrt{2 D_r}\;\, d\mathbf{W}_i^r \times \hat{\mathbf{e}}_i \qquad\text{(Stratonovich, } |\hat{\mathbf{e}}_i| = 1 \text{ preserved).}
\end{aligned}
$$

In two dimensions the orientation update collapses to a scalar Brownian motion of the angle:

$$
d\theta_i = \sqrt{2 D_r}\, dW_i^r .
$$

The pair force $\mathbf{F}_i^{\text{int}}$ can be very stiff (Lennard–Jones is $\sim r^{-13}$ near contact), which is precisely why you want an *adaptive* high-order integrator such as Cash–Karp 5(4) for the deterministic part.

> **Overdamped variant.** If you drop inertia, the same machinery applies with $m\dot{\mathbf{v}} \to 0$ and a single equation for $\mathbf{r}_i$:
> $$d\mathbf{r}_i = \mu\big[\mathbf{F}_i^{\text{int}} + f_a\,\hat{\mathbf{e}}_i\big]dt + \sqrt{2 D_t}\, d\mathbf{W}_i^t .$$
> The splitting strategy below carries over with the velocity-update step replaced by translational noise on positions.

---

## 2. Why You Cannot Simply Plug a Stochastic RHS into Cash–Karp

It is tempting to write the right-hand side as $\dot{x} = f(x) + g(x)\xi(t)$, where $\xi$ is white noise scaled like $\sqrt{1/\Delta t}$, and hand it to an adaptive ODE stepper. **This does not work**, for three independent reasons:

1. **Wiener increments scale as $\sqrt{\Delta t}$, not $\Delta t$.** A Cash–Karp 5(4) embedded pair estimates the local error by comparing two solutions of order $4$ and $5$. Both expansions assume the right-hand side is smooth in time on the step. White noise is nowhere differentiable; the embedded "error" picks up $\mathcal{O}(\sqrt{\Delta t})$ contributions from the noise that have nothing to do with discretisation error. The PI controller will then shrink $\Delta t$ to zero.

2. **Brownian paths must be sampled consistently across step rejections.** Adaptive ODE controllers freely retry steps with different $\Delta t$. Naively redrawing the noise on a rejected step destroys the underlying Wiener process and biases the statistics. A correct adaptive SDE solver requires "rejection sampling with memory" (Rackauckas & Nie 2017) or a no-skip dyadic rule (Foster et al. 2023) — both nontrivial bookkeeping.

3. **Higher-order Runge–Kutta tableaux do not lift to higher strong order on SDEs.** The Itô–Taylor expansion contains iterated stochastic integrals (Lévy areas) that classical RK weights do not encode. Cash–Karp's order-5 accuracy is wasted on the noise term — the best you can hope for is strong order $1/2$ (Euler–Maruyama) or $1$ (Milstein with diagonal noise).

The clean solution is **operator splitting**: keep Cash–Karp doing what it does well — adapt its way through the stiff *deterministic* flow — and integrate each stochastic piece with a method that is *exact* (or analytically correct) for that subsystem.

---

## 3. Operator Splitting — Generator Decomposition

Write the Kolmogorov backward generator of the SDE as a sum:

$$
\mathcal{L} = \mathcal{L}_D + \mathcal{L}_O + \mathcal{L}_R ,
$$

where

$$
\begin{aligned}
\mathcal{L}_D &= \sum_i \Bigg[\mathbf{v}_i\!\cdot\!\nabla_{\mathbf{r}_i} + \frac{1}{m}\!\left(\mathbf{F}_i^{\text{int}} + f_a\,\hat{\mathbf{e}}_i\right)\!\cdot\!\nabla_{\mathbf{v}_i}\Bigg] && \text{(deterministic flow, } \hat{\mathbf{e}} \text{ frozen)}\\
\mathcal{L}_O &= \sum_i \Bigg[-\frac{\gamma}{m}\,\mathbf{v}_i\!\cdot\!\nabla_{\mathbf{v}_i} + \frac{\gamma k_B T}{m^2}\,\nabla_{\mathbf{v}_i}^2\Bigg] && \text{(Ornstein–Uhlenbeck thermostat)}\\
\mathcal{L}_R &= D_r \sum_i \Delta_{\hat{\mathbf{e}}_i} && \text{(rotational diffusion on } S^{d-1}\text{)}
\end{aligned}
$$

Crucially, **each of the three sub-generators corresponds to an SDE that can be integrated exactly or near-exactly:**

| Piece | Sub-equation | Integrator | Why exact |
|-------|-------------|-----------|-----------|
| $\mathcal{L}_D$ | $\dot{\mathbf{r}} = \mathbf{v}$, $m\dot{\mathbf{v}} = \mathbf{F}^{\text{int}} + f_a\hat{\mathbf{e}}$ with $\hat{\mathbf{e}}$ frozen | **Cash–Karp 5(4) adaptive** | high-order ODE |
| $\mathcal{L}_O$ | $m\,d\mathbf{v} = -\gamma\mathbf{v}\,dt + \sqrt{2\gamma k_B T}\,d\mathbf{W}^t$ | **Exact OU update** | linear SDE, closed form |
| $\mathcal{L}_R$ | $d\hat{\mathbf{e}} = \sqrt{2D_r}\,d\mathbf{W}^r \times \hat{\mathbf{e}}$ | **Exact / geometric rotation** | rotation by Gaussian angle |

This is the cleanest possible factorisation: the deterministic and stiff piece goes to your favourite high-order adaptive ODE solver, and the two purely-stochastic pieces are advanced by closed-form propagators that introduce zero numerical error on their own subsystem.

### 3.1 Exact subflow propagators

**Ornstein–Uhlenbeck on velocities (Langevin O step).** For each particle,

$$
\mathbf{v}(t+\tau) = e^{-\gamma\tau/m}\,\mathbf{v}(t) + \sqrt{\frac{k_B T}{m}\!\left(1 - e^{-2\gamma\tau/m}\right)}\;\boldsymbol{\eta} ,
$$

where $\boldsymbol{\eta} \sim \mathcal{N}(0, I_d)$. This is *exact in distribution* for any $\tau > 0$; there is no time-step error in this piece.

**Rotational diffusion (R step).** In two dimensions the angle is just a Brownian motion:

$$
\theta(t+\tau) = \theta(t) + \sqrt{2 D_r \tau}\;\eta_r ,\qquad \eta_r \sim \mathcal{N}(0,1) .
$$

In three dimensions, the recommended approach (Höfling & Straube 2024) is a **geometric rotation**: draw a Gaussian rotation vector $\boldsymbol{\phi} \sim \mathcal{N}(0, 2 D_r \tau\, I_3)$ and apply

$$
\hat{\mathbf{e}}(t+\tau) = R(\boldsymbol{\phi})\,\hat{\mathbf{e}}(t) ,
$$

where $R(\boldsymbol{\phi})$ is the rotation matrix corresponding to axis $\boldsymbol{\phi}/|\boldsymbol{\phi}|$ and angle $|\boldsymbol{\phi}|$ (Rodrigues' formula). This *exactly* preserves $|\hat{\mathbf{e}}| = 1$ and is weakly first-order accurate; an even better weakly-exact variant uses a tabulated propagator on the sphere.

**Deterministic flow (D step).** With $\hat{\mathbf{e}}$ frozen and the thermostat noise switched off, the equations $\dot{\mathbf{r}} = \mathbf{v}$, $m\dot{\mathbf{v}} = \mathbf{F}^{\text{int}}(\mathbf{r}) + f_a\hat{\mathbf{e}}$ are an ordinary Hamiltonian-with-constant-bias ODE and are perfectly suited to Cash–Karp 5(4). Inside this substep, `integrate_adaptive` from Boost will take many internal steps with PI-controlled error.

### 3.2 Lie–Trotter and Strang composition

Composing the subflows over a macro-step $\Delta t$ gives a propagator approximation. Two canonical choices:

**Lie–Trotter (1st order weak):**

$$
\Phi^{\Delta t} \;\approx\; \Phi_R^{\Delta t}\,\circ\,\Phi_O^{\Delta t}\,\circ\,\Phi_D^{\Delta t}
$$

**Strang (2nd order weak, palindromic):**

$$
\boxed{\;\Phi^{\Delta t} \;\approx\; \Phi_O^{\Delta t/2}\,\circ\,\Phi_R^{\Delta t/2}\,\circ\,\Phi_D^{\Delta t}\,\circ\,\Phi_R^{\Delta t/2}\,\circ\,\Phi_O^{\Delta t/2}\;}
$$

The Strang version recovers second-order weak accuracy in $\Delta t$ even though the subgenerators do not commute (commutator errors are pushed to $\mathcal{O}(\Delta t^3)$ per step, $\mathcal{O}(\Delta t^2)$ globally). Empirically, Strang-style sandwiches with the noise on the outside (or with the noise sandwiched in the middle, BAOAB-style) give *much* better long-time configurational sampling than first-order Lie–Trotter — the "BAOAB superconvergence" phenomenon discovered by Leimkuhler & Matthews, where configurational averages converge with an error prefactor that is dramatically smaller than the formal weak order would suggest.

### 3.3 Convergence summary

For an SDE with smooth coefficients and additive noise:

| Scheme | Strong order | Weak order | Configurational sampling error |
|--------|--------------|-----------|--------------------------------|
| Lie–Trotter (any ordering) | $1/2$ | $1$ | $\mathcal{O}(\Delta t)$ |
| Strang (any palindromic ordering) | $1$ | $2$ | $\mathcal{O}(\Delta t^2)$ |
| BAOAB-style (with exact O) | $1$ | $2$ | $\mathcal{O}(\Delta t^4)$ on harmonic part |

Note that the cash_karp54 integration of the deterministic flow contributes essentially zero error compared to these splitting errors — its local error per macro-step is $\mathcal{O}(\text{tol})$ where tol is the controller tolerance, so as long as the deterministic tolerance is well below $\Delta t^2$, the splitting error dominates and the recommended Strang ordering achieves its full second-order weak accuracy.

---

## 4. Recommended Algorithm

Putting it all together, the recommended scheme for a macro-step $\Delta t$ is:

**Algorithm: Strang(O‑R‑D‑R‑O) with Cash–Karp 5(4) adaptive D-substep**

```
Inputs at time t:  r_i, v_i, ê_i  (i = 1..N)
Macro step:        Δt
Tolerances:        atol, rtol (for Cash–Karp inner)

1.  Half OU step:
    For each i, draw η_i ~ N(0, I_d):
        v_i  ←  exp(-γ Δt / 2m) v_i + sqrt((kT/m)(1 - exp(-γ Δt / m))) η_i

2.  Half rotational diffusion:
    For each i, draw a Gaussian rotation φ_i with var = D_r Δt:
        ê_i  ←  R(φ_i) ê_i        (in 2D: θ_i ← θ_i + sqrt(D_r Δt) η_r)

3.  Full deterministic flow (orientations frozen, thermostat off):
    Use Boost::odeint integrate_adaptive with make_controlled(atol, rtol,
        runge_kutta_cash_karp54<state>):
        d r_i / dt = v_i
        d v_i / dt = (F_i^int(r) + f_a ê_i) / m
    Integrate from t to t + Δt; the controlled stepper takes adaptive
    internal substeps to meet (atol, rtol).

4.  Half rotational diffusion:
        ê_i  ←  R(φ'_i) ê_i        (fresh draw with var = D_r Δt)

5.  Half OU step:
        v_i  ←  exp(-γ Δt / 2m) v_i + sqrt((kT/m)(1 - exp(-γ Δt / m))) η'_i
        (fresh draw η'_i)

t  ←  t + Δt
```

A few notes:

- **Each random draw is independent.** The half-steps in 1 and 5 use *independent* noise; same for 2 and 4. (This is important — using the same draw twice would change the invariant measure.) The "half-step" refers to the time interval, but the noise variance follows the *exact* OU/Brownian formula for that interval, so each Gaussian has variance proportional to $1 - e^{-\gamma\Delta t/m}$ for OU and $D_r\Delta t$ for orientation, *not* halved.

- **Cash–Karp does its own adaptation.** The macro $\Delta t$ controls the stochastic accuracy and the splitting error; the inner Cash–Karp controls the deterministic accuracy. The two controls are decoupled.

- **Why O on the outside?** This BAOAB-cousin ordering tends to give the smallest configurational sampling bias. If you instead want the smallest momentum-distribution bias, swap to `R-O-D-O-R`. The differences are subtle and matter mostly when you push $\Delta t$ to be large.

### 4.1 Choosing the macro step $\Delta t$

The macro step is bounded above by the fastest stochastic timescale you want to resolve:

$$
\Delta t \ll \min\!\left(\tau_R,\;\tau_\gamma\right) \;=\; \min\!\left(\frac{1}{D_r},\;\frac{m}{\gamma}\right) .
$$

Choosing $\Delta t \approx 10^{-2}\,\min(\tau_R, \tau_\gamma)$ is typical and gives splitting errors well below $1\%$. The Cash–Karp tolerances then govern only the deterministic accuracy, which can be tightened independently when interactions are stiff.

### 4.2 Optional: adapting $\Delta t$ too

If you genuinely need an adaptive *macro* step (e.g. activity ramps up, or the system passes through a stiff transient), the cleanest way is a PI controller driven by a stochastic monitor — for example, the trial-step / half-step comparison of two macro Strang steps versus one. Doing this rigorously requires the no-skip / dyadic-time machinery of Foster et al. 2023 to avoid Lévy-area bias. In practice almost everyone uses a fixed macro $\Delta t$ and lets the Cash–Karp inner control absorb the deterministic stiffness; that's a good default.

---

## 5. C++ Reference Implementation (Boost.Odeint)

Below is a self-contained sketch in 2D (positions, velocities, scalar angle). The translation to 3D requires only swapping the `R` step for a Rodrigues rotation. The example uses `std::vector<double>` for the state and assumes a pair-force routine you provide.

```cpp
// active_langevin.cpp -- Strang(O-R-D-R-O) splitting with Cash-Karp inner ODE
#include <boost/numeric/odeint.hpp>
#include <random>
#include <vector>
#include <cmath>

namespace odeint = boost::numeric::odeint;
using state_type = std::vector<double>;   // size 4N: [r1x,r1y,v1x,v1y, r2x,r2y,v2x,v2y, ...]

// ----- Simulation parameters -----
struct Params {
    std::size_t N;          // number of particles
    double m;               // mass
    double gamma;           // friction
    double kT;              // k_B T
    double f_a;             // active force magnitude
    double D_r;             // rotational diffusion coefficient
    // ... LJ parameters, box size, etc.
};

// ----- Pair forces (user-supplied, e.g. truncated Lennard-Jones) -----
void compute_pair_forces(const state_type& x, std::vector<double>& F, const Params& p);

// ----- The deterministic right-hand side: orientations are frozen here -----
// state x stores positions and velocities only; orientations are a separate vector
// captured by the functor. The thermostat is OFF in this RHS by design.
class DeterministicRHS {
public:
    DeterministicRHS(const std::vector<double>& theta_frozen, const Params& p)
        : theta_(theta_frozen), p_(p), F_(2 * p.N) {}

    void operator()(const state_type& x, state_type& dxdt, double /*t*/) {
        compute_pair_forces(x, F_, p_);
        for (std::size_t i = 0; i < p_.N; ++i) {
            const double cx = std::cos(theta_[i]);
            const double cy = std::sin(theta_[i]);
            // dr/dt = v
            dxdt[4*i + 0] = x[4*i + 2];
            dxdt[4*i + 1] = x[4*i + 3];
            // m dv/dt = F_int + f_a ê     (no thermostat term here!)
            dxdt[4*i + 2] = (F_[2*i + 0] + p_.f_a * cx) / p_.m;
            dxdt[4*i + 3] = (F_[2*i + 1] + p_.f_a * cy) / p_.m;
        }
    }
private:
    const std::vector<double>& theta_;
    Params p_;
    std::vector<double> F_;
};

// ----- Exact OU half-step on velocities -----
void ou_step(state_type& x, double tau, const Params& p, std::mt19937_64& rng)
{
    const double a = std::exp(-p.gamma * tau / p.m);
    const double sigma = std::sqrt(p.kT / p.m * (1.0 - a*a));
    std::normal_distribution<double> N01(0.0, 1.0);
    for (std::size_t i = 0; i < p.N; ++i) {
        x[4*i + 2] = a * x[4*i + 2] + sigma * N01(rng);
        x[4*i + 3] = a * x[4*i + 3] + sigma * N01(rng);
    }
}

// ----- Rotational diffusion in 2D -----
void rot_step_2d(std::vector<double>& theta, double tau, const Params& p,
                 std::mt19937_64& rng)
{
    const double sigma = std::sqrt(2.0 * p.D_r * tau);
    std::normal_distribution<double> N01(0.0, 1.0);
    for (std::size_t i = 0; i < p.N; ++i)
        theta[i] += sigma * N01(rng);
}

// ----- 3D Rodrigues rotation step (use this instead in 3D) -----
// void rot_step_3d(std::vector<std::array<double,3>>& e, double tau,
//                  const Params& p, std::mt19937_64& rng) {
//     const double s = std::sqrt(p.D_r * tau);
//     std::normal_distribution<double> N01(0.0, 1.0);
//     for (auto& ei : e) {
//         double phi[3] = { s*N01(rng), s*N01(rng), s*N01(rng) };
//         double a = std::sqrt(phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2]);
//         if (a < 1e-12) continue;
//         double k[3] = { phi[0]/a, phi[1]/a, phi[2]/a };
//         double c = std::cos(a), s_ = std::sin(a);
//         double dot = k[0]*ei[0] + k[1]*ei[1] + k[2]*ei[2];
//         double cr[3] = { k[1]*ei[2]-k[2]*ei[1],
//                          k[2]*ei[0]-k[0]*ei[2],
//                          k[0]*ei[1]-k[1]*ei[0] };
//         for (int d=0; d<3; ++d)
//             ei[d] = ei[d]*c + cr[d]*s_ + k[d]*dot*(1.0-c);
//     }
// }

// ----- One macro step of the Strang(O-R-D-R-O) integrator -----
template <class Stepper>
void strang_step(Stepper& stepper,
                 state_type& x, std::vector<double>& theta,
                 double t, double dt,
                 const Params& p, std::mt19937_64& rng)
{
    // 1. half OU
    ou_step(x, 0.5 * dt, p, rng);

    // 2. half rotational diffusion
    rot_step_2d(theta, 0.5 * dt, p, rng);

    // 3. full deterministic flow with adaptive Cash-Karp inner stepper
    DeterministicRHS rhs(theta, p);
    odeint::integrate_adaptive(stepper, rhs, x, t, t + dt, dt * 0.1);
    //                        ^                              ^
    //                  controlled stepper          initial inner step guess

    // 4. half rotational diffusion (independent draws)
    rot_step_2d(theta, 0.5 * dt, p, rng);

    // 5. half OU
    ou_step(x, 0.5 * dt, p, rng);
}

// ----- Main -----
int main()
{
    Params p;
    p.N     = 1024;
    p.m     = 1.0;
    p.gamma = 1.0;
    p.kT    = 1.0;
    p.f_a   = 5.0;
    p.D_r   = 0.1;
    // ... initialise positions, velocities, orientations ...

    state_type x(4 * p.N);
    std::vector<double> theta(p.N);
    std::mt19937_64 rng(12345);

    // Build the Cash-Karp 5(4) controlled stepper. Tolerances apply to the
    // inner adaptive substepping inside each macro step.
    auto stepper = odeint::make_controlled(
        1.0e-8,                                            // abs_tol
        1.0e-6,                                            // rel_tol
        odeint::runge_kutta_cash_karp54<state_type>()      // base method
    );

    const double dt    = 1.0e-3;     // macro Strang step
    const double T_end = 100.0;
    double t = 0.0;
    while (t < T_end) {
        strang_step(stepper, x, theta, t, dt, p, rng);
        t += dt;
        // ... observables, neighbour-list updates, etc. ...
    }
    return 0;
}
```

### 5.1 What the controlled stepper is doing inside step 3

Inside the call `integrate_adaptive(stepper, rhs, x, t, t+dt, dt*0.1)`, Boost will:

1. Try a Cash–Karp 5(4) substep of size $\delta t \approx \Delta t/10$.
2. Use the embedded 4th-order solution to estimate the local error.
3. Compare to `atol + rtol * |x|` componentwise; either accept and grow $\delta t$, or shrink and retry.
4. Repeat until $t + \Delta t$ is reached, possibly with one final partial substep.

So the deterministic flow gets the full benefit of adaptive 5th-order integration *every macro step*, transparently to the outer splitting.

### 5.2 A subtlety: `integrate_adaptive` vs. neighbour lists

If you use a Verlet/cell list, be aware that `integrate_adaptive` may evaluate the force functor at intermediate states whose particle separations briefly violate the cell-list skin. The cleanest fix is:

- Recompute / verify the neighbour list at the **start of each macro step** (after the O-R-O preamble has shifted velocities and orientations only — positions are unchanged), or
- Use a "skin" generous enough that the cumulative position drift over one macro step plus the four CK substeps stays inside it.

Cash–Karp does not modify positions during its embedded error estimate (it modifies the trial state, but that is local), so a skin sized for a single macro step is enough.

---

## 6. Practical Recommendations

1. **Start with the Strang $O\!-\!R\!-\!D\!-\!R\!-\!O$ ordering.** It is second-order weak in $\Delta t$, simple to implement, and has excellent equilibrium-sampling behaviour.

2. **Set the Cash–Karp tolerances tight** — say `atol = 1e-8, rtol = 1e-6` — and forget about them. The deterministic flow inside one macro step is benign, and the inner adaptive cost is paid for by stable behaviour through close LJ encounters.

3. **Pick the macro $\Delta t$ from the slowest of the fast stochastic timescales:** $\Delta t \approx 0.01\min(m/\gamma,\,1/D_r)$ is a reliable starting point. Check convergence by halving $\Delta t$ and confirming observables agree to within their statistical errors.

4. **Verify equipartition.** A correctly-implemented O step gives $\langle m v_x^2 \rangle = k_B T$ to machine precision in equilibrium; this is a powerful sanity check.

5. **Verify orientational autocorrelation.** $\langle \hat{\mathbf{e}}(t)\!\cdot\!\hat{\mathbf{e}}(0)\rangle$ should decay as $\exp(-(d-1)D_r t)$ in $d$ dimensions; this directly tests the R step.

6. **Don't be tempted by stochastic Cash–Karp variants.** Embedded SRK schemes (Rackauckas & Nie 2017) exist and give true adaptive control of the *whole* SDE, but they are far more complex than the splitting approach and do not let you reuse a battle-tested ODE library. The splitting approach is simpler, modular, and gives you adaptive control where you need it (the stiff deterministic forces) and exactness where you can have it (the linear OU and the rotational diffusion).

7. **Overdamped active matter.** If you drop inertia, the algorithm simplifies to a Strang $T\!-\!R\!-\!D\!-\!R\!-\!T$ where $T$ is a translational-noise half-step on positions and $D$ is the deterministic position-flow $\dot{\mathbf{r}} = \mu(\mathbf{F}^{\text{int}} + f_a\hat{\mathbf{e}})$. Cash–Karp again handles the stiff $D$ piece adaptively.

---

## 7. References

- B. Leimkuhler & C. Matthews, *Robust and efficient configurational molecular sampling via Langevin dynamics*, J. Chem. Phys. **138**, 174102 (2013). — The original BAOAB paper.
- B. Leimkuhler & C. Matthews, *Molecular Dynamics: With Deterministic and Stochastic Numerical Methods*, Springer (2015). — The textbook treatment of splitting schemes for Langevin dynamics, including the [ABO]-notation.
- A. Telatovich & X. Li, *The strong convergence of operator-splitting methods for the Langevin dynamics model*, arXiv:1706.04237 (2017). — Strong convergence orders for Lie–Trotter and Strang on additive-noise Langevin SDEs.
- F. Höfling & A. V. Straube, *Langevin equations and a geometric integration scheme for the overdamped limit of rotational Brownian motion*, arXiv:2403.04501 (2024). — Geometric (norm-preserving) integration of orientational diffusion.
- C. Rackauckas & Q. Nie, *Adaptive methods for stochastic differential equations via natural embeddings and rejection sampling with memory*, Discrete Contin. Dyn. Syst. B **22** (2017). — How adaptive SDE solvers can be done rigorously, if you ever need to step away from splitting.
- J. Foster, G. dos Reis, & C. Strange, *High-order splitting methods for SDEs satisfying a commutativity condition*, arXiv:2210.17543 (2022).
- A. Leroy, B. Leimkuhler, J. Latz, & D. J. Higham, *Adaptive stepsize algorithms for Langevin dynamics*, SIAM J. Sci. Comput. (2025); arXiv:2403.11993. — Invariant-measure-preserving adaptive timestep control if you want to go beyond a fixed macro $\Delta t$.
- C. Bechinger et al., *Active particles in complex and crowded environments*, Rev. Mod. Phys. **88**, 045006 (2016). — The standard active-matter reference.
- Boost.Numeric.Odeint documentation, in particular the `runge_kutta_cash_karp54`, `make_controlled`, and `integrate_adaptive` interfaces: https://www.boost.org/doc/libs/release/libs/numeric/odeint/.
