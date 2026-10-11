# freezeframe

**Dark-matter freeze-out beyond kinetic equilibrium.** A dependency-free Rust crate that solves the
Boltzmann equation for the full dark-matter phase-space density `f_χ(x, q)` for any user-supplied
particle model, together with the coupled moment equations for `Y(x)` and `y(x)` and the standard
number-density equation for `Y(x)`. The physics follows

> T. Binder, T. Bringmann, M. Gustafsson, A. Hryczuk, *Early kinetic decoupling of dark matter: when
> the standard way of calculating the thermal relic density fails*, Phys. Rev. D 96, 115010 (2017),
> [arXiv:1706.07433](https://arxiv.org/abs/1706.07433), Sec. II.

## What it computes

| `Method`                      | evolves                    | output                                   | typical cost |
|-------------------------------|----------------------------|------------------------------------------|--------------|
| `Method::NumberDensity`       | `Y = n/s`                  | `Y(x)`, `Ωh²`                            | 0.15 s       |
| `Method::coupled()`           | `Y`, `y = m T_χ / s^{2/3}` | `Y(x)`, `y(x)`, `T_χ/T`, `Ωh²`           | 0.4 s        |
| `Method::phase_space()`       | `f(x, q)`, `q = p/T`       | `f(x, q)`, `Y(x)`, `y(x)`, `Ωh²`         | 3–7 s        |

Timings are for one model point on one core (Apple-silicon laptop), `x = m/T ∈ [10, 10⁶]`. With
`.threads(n)` the phase-space solver rebuilds its annihilation kernel in parallel (≈2.5× faster with
4–8 threads, bit-identical results).

## Quick start

```toml
[dependencies]
freezeframe = { path = "path/to/freezeframe" }
```

```rust
use freezeframe::prelude::*;

fn main() -> freezeframe::Result<()> {
    // 0.47 GeV Dirac fermion, pseudoscalar mediator at 1 GeV coupled to e and μ
    let g = 5.56e-5;
    let model = SimplifiedModel::builder(Mediator::Pseudoscalar)
        .dm_mass(0.47)
        .mediator_mass(1.0)
        .dm_coupling(g)
        .sm_coupling(SmFermion::Electron, g)
        .sm_coupling(SmFermion::Muon, g)
        .build()?;

    // relic density only
    let omega_std = relic_density(&model, Method::NumberDensity)?;

    // phase-space density, Y(x), y(x) and Ωh²
    let sol = Solver::new(&model)
        .trace_log(10.0, 1e3, 200)          // Y, y at 201 log-spaced x
        .snapshots([20.0, 50.0, 100.0])     // f(q) at these x
        .threads(4)
        .solve(Method::phase_space())?;

    println!("Ωh²: standard {omega_std:.3}, full phase space {:.3}", sol.omega_h2());
    for s in sol.snapshots() {
        // s.q[i] = p/T,  s.f[i] = f_χ(x, q_i)
    }
    Ok(())
}
```

`cargo run --release --example quickstart` walks through all three outputs for a model defined from
scratch.

## Supplying a model

Everything model-dependent enters through the trait `DarkMatterModel`:

```rust
pub trait DarkMatterModel {
    fn mass(&self) -> f64;                                   // m_χ [GeV]
    fn dof(&self) -> f64;                                    // internal dof g_χ of one particle
    fn kind(&self) -> ParticleKind { SelfConjugate }         // or Distinct (Dirac-like, no asymmetry)
    fn sigma_v(&self, s: f64) -> f64;                        // σ·v_cms(s) [GeV⁻²]
    fn momentum_transfer_rate(&self, temperature: f64) -> f64; // γ(T) [GeV]
    fn resonances(&self) -> Vec<Resonance> { vec![] }        // optional: s-channel poles (mass, width)
    fn thresholds(&self) -> Vec<f64> { vec![] }              // optional: kinks of σv in sqrt(s)
}
```

Conventions (identical to the paper):

* `sigma_v(s)` is `σ(χχ̄ → SM)` times the Møller velocity in the CM frame, `v_cms = 2 sqrt(1 − 4m²/s)`,
  averaged over initial internal degrees of freedom and summed over final states. Then
  `F'(s̃) = s̃ σv_cms = (2s̃ − 1) σv_lab` with `s̃ = s/4m²`. Helpers `model::cms_velocity` and
  `model::lab_velocity` convert.
* `momentum_transfer_rate(T)` is the `γ(T)` of the Fokker–Planck operator
  `C_el = (E/2) γ ∇_p·[E T ∇_p + p] f` (semi-relativistic form, App. B of the paper).
* `ParticleKind::Distinct`: `χ` and `χ̄` share one distribution, `Ωh²` counts both.
* Declaring narrow resonances is optional but strongly recommended: they place quadrature and
  interpolation nodes; they do not change the physics.

`ModelFn::new(mass, dof, |s| ..., |t| ...)` builds a model from two closures.

### Computing γ(T) from a matrix element

`freezeframe::scattering` evaluates Eqs. (6–7) of the paper,

```
γ(T) = 1/(48π³ g_χ m³ T) Σ_f ∫ dω g_f(1 ∓ g_f) k⁴⟨|M|²⟩_t,
k⁴⟨|M|²⟩_t = (1/8) ∫_{−4k_cm²}^0 dt (−t) |M|²(s, t),   s = m² + 2ωm + m_f²,  k_cm² = k²m²/s,
```

from an elastic matrix element summed over all spins and colors:

```rust
use freezeframe::scattering::{RateTable, Scatterer};
let msq = |s: f64, t: f64| /* |M|²(χ e → χ e) */;
let gamma = |t: f64| Scatterer::fermion_pair(0.000511, msq).momentum_transfer_rate(m_chi, g_chi, t);
// optional: tabulate once (ln γ vs ln T, pieces split at discontinuities such as T_QCD)
let table = RateTable::new(m_chi * 1e-9, 2.0 * m_chi, 40, &[], gamma)?;
```

### Bundled models (`freezeframe::models`)

* **`ScalarSinglet`**: the Higgs-portal real scalar of Sec. III of the paper, with
  `QuarkScattering::{Maximal, Minimal}` (scenarios A and B) and the HDECAY Higgs width.
* **`SimplifiedModel`**: a Dirac fermion with an s-channel `Mediator::{Scalar, Pseudoscalar, Vector,
  AxialVector}` coupled to any set of SM fermions (`SmFermion`). Annihilation cross sections,
  scattering matrix elements and mediator widths are listed in the module documentation. They were
  verified against explicit Dirac-trace numerics (`scripts/check_matrix_elements.py`).

| mediator     | `χχ̄ → ℓℓ̄`                     | `χℓ → χℓ`                    |
|--------------|-------------------------------|------------------------------|
| scalar       | p-wave                        | unsuppressed (SI)            |
| pseudoscalar | s-wave                        | suppressed, `|M|² ∝ t²`      |
| vector       | s-wave                        | unsuppressed (SI)            |
| axial-vector | p-wave + `m_ℓ²` s-wave        | unsuppressed (SD)            |

## Working with the solution

```rust
let sol = Solver::new(&model)
    .x_range(10.0, 1e6)          // default; DM in full equilibrium at x_start
    .rtol(1e-4)                  // default 1e-4 (fBE), 1e-8 (nBE, cBE)
    .trace_at([20.0, 50.0])      // or .trace_log(x0, x1, n)
    .snapshots([20.0, 50.0])     // phase-space method only
    .cosmology(&my_cosmology)    // default: Drees, Hajkarim & Schmitz (2015)
    .solve(Method::PhaseSpace(PhaseSpaceGrid { cells: 300, q_max: None }))?;

sol.omega_h2();                 // Ωh²
sol.final_abundance();          // Y(x_end) of one species
sol.final_temp_param();         // y(x_end)
sol.trace();                    // &[TracePoint { x, abundance, abundance_eq, temp_param, temp_param_eq }]
sol.snapshots();                // &[Snapshot { x, q, f }]
sol.decoupling();               // first x with 10% deviation of Y (chemical) and y (kinetic)
sol.write_trace(file)?;         // TSV
sol.write_snapshots(file)?;     // TSV
```

To fix a coupling by the relic density with any method:

```rust
let fit = find_coupling(
    |g| relic_density(&make_model(g)?, Method::phase_space()),
    1e-3,                                     // initial guess
    &CouplingSearch { target: 0.12, ..Default::default() },
)?;
```

## Demonstration: four models

`scripts/run_scans.sh` (≈20 min on 16 cores) runs `examples/relic_scan.rs` for each model, and
`scripts/make_figures.py` makes the figures. At each mass the coupling is fixed by the standard
(nBE) relic density. The coupled and full solutions are evaluated at that coupling, and the coupling
is then refitted with each of them.

1. **Scalar Singlet** (benchmark, scenario B, `Ωh² = 0.1188` as in the paper). Reproduces Figs. 1 and 3
   of arXiv:1706.07433. Peak `Ω_cBE/Ω_std = 6.46` at 58 GeV and `Ω_fBE/Ω_std = 9.07` at 58.5 GeV
   (paper ≈ 6.5 and ≈ 9). Minimum 0.32 (cBE) and 0.215 (fBE) at 62.55 GeV (paper ≈ 0.3).
   `T_kd(45 GeV) = 0.25 GeV`.
2. **Pseudoscalar mediator**: s-wave annihilation, momentum-suppressed scattering (`|M|² ∝ t²`).
3. **Vector mediator**: s-wave annihilation, unsuppressed scattering.
4. **Axial-vector mediator**: p-wave annihilation into light leptons, spin-dependent scattering.

Models 2–4 are sub-GeV Dirac fermions with `m_med = 1 GeV`, coupled to e and μ with a universal
`g = g_χ = g_ℓ` and `Ωh² = 0.12`. The scalar mediator is included in the scan for comparison
(p-wave, like the axial-vector).

Results of the mediator scan (`m_χ/m_med = 0.25 … 0.6`):

* Far from the pole (`m_χ/m_med ≤ 0.35`, `≥ 0.51`) kinetic decoupling happens long after freeze-out
  (`x_kd ≈ 90–1100` vs `x_cd ≈ 18`). All three treatments agree to ≲ 2% (pseudoscalar/vector above
  threshold: within 10% for `m_χ/m_med ≥ 0.51`).
* Below the pole the required couplings are tiny (`g ~ 10⁻⁵–10⁻³`). Kinetic decoupling then coincides
  with chemical decoupling, DM cools as `T_χ ∝ T²`, and the high-velocity tail that reaches the
  resonance is depleted. The full solution gives `Ω_fBE/Ω_std` up to 55 (pseudoscalar and vector at
  `m_χ/m_med = 0.42`) and ≈ 36 for all four mediators at 0.44. The relic coupling rises by up to
  ×11. The coupled (Y, y) equations capture only about half of the effect there
  (`Ω_cBE/Ω_std ≈ 19–24` for the s-wave mediators).
* For `0.44 ≤ m_χ/m_med < 0.5` the four mediators agree to ≲ 1%, because the narrow resonance
  dominates. Only `M` and `m_χ` matter there.
* Immediately above threshold (`m_χ/m_med = 0.5005`) the s-wave mediators give `Ω_fBE/Ω_std ≈ 0.57–0.6`,
  since cold DM sits on the pole. The p-wave ones stay within 2% because `σv → 0` at threshold.
* Just below threshold (`m_χ/m_med = 0.4995`) the resonance removes slow particles. The distribution
  becomes strongly non-thermal (`T_χ > T` transiently), and the coupled equations overestimate
  `Ω` by 40% relative to the full solution.

Physics caveats for these benchmarks: DM self-scattering (which could re-thermalize `f` among DM
particles), mediator production, and constraints on such couplings are not included.

## Numerical methods

* **Comoving momentum grid.** The phase-space equation is solved in `q̂ = q / r(x)` with
  `r = [h_eff(T)/h_eff(T_start)]^{1/3}`, i.e. `q̂ ∝ p a`. Because `x r'/r = −g̃`, the Liouville term
  `g̃ q ∂_q f` of Eq. (BEps) cancels exactly and `x ∂ₓ f|_q̂ = C[f]`. The collision terms are evaluated on the
  instantaneous physical cells `q = r q̂`. An upwinded `g̃` term would be first-order accurate and
  diffusive, which shows up in `y(x)` after the QCD and e⁺e⁻ entropy releases.
* **Finite volumes** with piecewise-constant `f`. The grid is uniform up to `q_max = 60`. For larger `q_max`
  it is uniform below `q ≈ 30` and sinh-stretched above. By default `q_max = max(60, 10 σ_q)`, where
  `σ_q² = x T_χ/T` at `x_end` comes from a preliminary coupled-equation solve. This keeps
  late-decoupling distributions, with width `√x` in `q` until kinetic decoupling, on the grid.
* **Fokker–Planck operator** in conservative, exponentially fitted (Chang–Cooper/Scharfetter–Gummel)
  form: `e^{−E/T}` is the exact discrete zero-flux state and number is conserved to round-off.
* **Annihilation kernel** in rapidity variables. `q̃²⟨vσ⟩^θ dq̃ = x³/(uE)[F(s̃₊) − F(s̃₋)] dẼ`
  depends only on `η ± η̃`. Tabulating `Φ(z) = F(cosh²(z/2))` and its first two antiderivatives on a
  resonance-adapted grid gives cell integrals that are exact in the Breit–Wigner factor, so widths
  `Γ/M ~ 10⁻¹¹` are handled without sub-sampling.
* **Time stepping**: L-stable Rosenbrock 2(3) (Shampine–Reichelt, as in MATLAB `ode23s`) with an
  analytic dense Jacobian for the phase-space system.
* **Thermal averages** `⟨σv⟩`, `⟨σv⟩₂` (Eqs. 14, 30) by resonance-adapted quadrature, tabulated in
  `ln x` with power-law extrapolation.

## Validation

`cargo test --release` (35 tests, ~25 s) covers, among others:

* quadrature, Bessel functions (vs SciPy), splines, LU, the Rosenbrock integrator (Robertson problem);
* `⟨σv⟩ = ⟨σv⟩₂ = 1` for `σv_lab = 1` at `x = 1 … 10⁶`; `w(x = 10) = 0.826`;
* a narrow resonance (`Γ/M = 6·10⁻⁸`) against the narrow-width limit to 10⁻⁵, a pole below
  threshold against a direct integral, and the tabulated `Φ` of a pole against its closed form;
* the generic `γ(T)` integral against the closed form of Eq. (29) for the Scalar Singlet;
* simplified-model s-wave limits and pseudoscalar momentum suppression;
* number conservation without annihilation (10⁻¹⁵ for `γ/H ≤ 10⁶`);
* the tight-coupling limit (`γ = 10⁶ H`): fBE, cBE and nBE agree to < 0.3%;
* free streaming: `y` from the fBE and the cBE agree to 1%, `x_kd ≈ 11.5`;
* Scalar Singlet regression against the earlier reproduction of the paper's figures.

Convergence of the phase-space solution (cells N = 150 / 300 / 600 at fixed `rtol = 1e-4`):

| benchmark                                   | Ωh² (N=150/300/600)            | y(x_end)                    |
|---------------------------------------------|--------------------------------|-----------------------------|
| Scalar Singlet, m = 62.55 GeV, λ_GG         | 0.025680 / 0.025682 / 0.025682 | 11.899 / 11.901 / 11.901    |
| Scalar Singlet, m = 57 GeV, λ_GG            | 0.79732 / 0.79751 / 0.79757    | 4.4830 / 4.4842 / 4.4844    |
| pseudoscalar, m = 0.47, M = 1 GeV, g_GG     | 1.2924 / 1.2935 / 1.2938       | 2.8242 / 2.8267 / 2.8273    |
| axial-vector, m = 0.50, M = 1 GeV, g_GG     | 0.025044 / 0.025044 / 0.025044 | 129.98 / 129.98 / 129.98    |

## Limitations

* DM obeys Maxwell–Boltzmann statistics and is assumed to be in full equilibrium at `x_start`
  (default 10). Bath particles have Fermi–Dirac/Bose–Einstein statistics in `γ(T)`.
* `γ(T)` is the leading term in `T/m`; its momentum dependence is neglected (as in the paper).
* Only `2 → 2` annihilation into the SM is included (no `χχ̄ → mediator pairs`, no Sommerfeld
  enhancement, no bound states). An s-channel mediator is assumed to be in equilibrium with the bath.
* In `SimplifiedModel`, quarks are free partons (scattering above `T_QCD` only); below a few GeV this
  is not a reliable description of hadronic final states.
* A resonance placed *exactly* at threshold (`M = 2m`, where `Γ(M → χχ̄) = 0`) is ill-conditioned;
  stay a few × 10⁻⁴ away in `m/M`.
* For extremely strong scattering (`γ/H ≳ 10¹²` at `x_start`) round-off in the stiff linear solves
  limits number conservation to ~10⁻⁵.

## Layout

```
src/
  lib.rs            crate docs, re-exports, prelude
  model.rs          DarkMatterModel trait, ModelFn, Resonance, ParticleKind
  solver/           Solver builder, Method, Solution; nBE, cBE and fBE systems
  kernel.rs         rapidity-space annihilation kernel (Φ, Φ₁, Φ₂ tables)
  thermal.rs        ⟨σv⟩, ⟨σv⟩₂ tables; Maxwell–Boltzmann moments
  scattering.rs     γ(T) from |M|²(s, t); RateTable
  cosmology.rs      h_eff, g_eff, g̃, H, s; Ωh² normalization
  relic.rs          coupling search
  ode.rs            Rosenbrock integrator
  math/             quadrature, Bessel K_ν, splines, LU
  models/           ScalarSinglet, SimplifiedModel, SmFermion
data/               degrees of freedom (Drees et al. 2015), Higgs width (HDECAY) — both from DRAKE v1.0
examples/           quickstart, relic_scan, convergence, validate_scalar_singlet
runs/               scan results (scan_*.tsv), traces and phase-space snapshots
figures/            demonstration figures
scripts/            matrix-element checks, scan driver, plotting
```

## Conventions and inputs

* `Ωh² = m Y s₀ / ρ_c` with `s₀` from the same `h_eff` table at `T_CMB = 2.7255 K` and
  `ρ_c/h² = 1.05367 × 10⁻⁵ GeV cm⁻³`. This is 0.4% above the Gondolo–Gelmini constant quoted in
  Eq. (15) of the paper; the coupled and full solutions are unaffected.
* Planck target `Ωh² = 0.120` (`constants::OMEGA_DM_H2_PLANCK`); the paper used 0.1188.
* SM inputs: `m_h = 125.09 GeV`, `v = 246.2 GeV`, `Γ_h = 4.042 MeV`, DRAKE fermion masses.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
