# Project: DSMC (arXiv:2611.25812) in freezeframe

Exploration of what freezeframe should take from

> F. Kahlhoefer, A. C. Wesener, *Direct simulation Monte Carlo for dark matter production and
> phase-space evolution*, arXiv:2611.25812 (v1, 19 Nov 2026), TTP26-043.

This folder holds the analysis (this file) and the work plan ([PLAN.md](PLAN.md)). Nothing in `src/`
changes on this branch.

Provenance: the paper was read from the arXiv PDF; equation numbers below are the paper's. Several
display equations were garbled in text extraction, so any formula taken from the paper must be
checked against the typeset PDF before it is implemented. Statements about freezeframe cite the
files at `bd74c1e`. Anything not read directly is marked *(inferred)*.

## 1. The paper in one page

**Method.** Direct simulation Monte Carlo (Bird's DSMC, as used for neutrino decoupling in
Ovchynnikov & Syvolap, arXiv:2409.07378): `N` unweighted simulation particles in a homogeneous
volume `V(t)`, each a sample of `f(p, t)`. No momentum grid, no thermal averages. One time step is

1. **collision step**: No-Time-Counter (NTC) scheme generalized from 2→2 to n→m. Draw
   `Poisson(C(N, n) P^max_n)` candidate n-plets, accept each with `P_{i1…in} / P^max_n`, where
   `P_{2→2} = σv Δt/Δx³` (3.5), `P_{2→3} = σ_{2→3}v Δt/Δx³` (3.7),
   `P_{3→2} = I_32 Δt / (8E₁E₂E₃ (Δx³)²)` (3.9–3.10). Several processes per step: total candidate
   count is Poisson in the sum, each candidate picks its process with weight ∝ its share.
2. **transport step**: redshift all momenta and the bath temperature, expand `V`.

`Δt = 0.01 min(H⁻¹, Γ_max⁻¹)`. Two-body final states isotropic in the CM frame, three-body by RAMBO
(taken from Hazma). Decays are not NTC: every `ϕ` decays with `P = (m/E) Γ Δt` (3.18).

**Variance reduction.** Production from the bath (`ψψ → ϕϕ`) samples bath pairs from
`g_eq(p) ∝ p² e^{−p/T}` with an analytic pair normalization (3.11–3.15) and accepts with
`σv/(σv)^max`. For 2→2 among the simulated species, pairs are chosen with probability ∝ `1/E`
using `S_n = Σ_k E_k^{−n}` (3.22–3.23), so the majorant is on `U₂₂ = 4E₁E₂ σv`, not on `σv`; 3→2
likewise with triplets ∝ `1/(E₁E₂E₃)` (3.26–3.28).

**Early times.** The equilibrium yield falls by orders of magnitude, which no affordable `N`
resolves. The simulation starts where the nBE yield deviates from `Y_eq` by 10% and is initialized
with Maxwell–Boltzmann at the bath temperature.

**Physics covered.** Toy model of real scalars: bath `ψ`, DM-like `ϕ` freezing out via
`ψψ ↔ ϕϕ` (constant `|M|²`, no `ψϕ` scattering), `ϕ → ρρ` in and out of equilibrium (bimodal `ρ`
spectrum), `ρρ → ρρ` self-scattering, `2ρ ↔ 3ρ` (cannibal). Classical statistics only.

**Validation and results.**

| test | outcome as stated |
|---|---|
| freeze-out, `|M|² = 3.4×10⁻¹¹`, fig. 4 | DSMC ≈ cBE in `Y` and `T_ϕ`; nBE off (no elastic scattering). Small `T` difference attributed to missing `ϕϕ` self-scattering |
| pure decay, `Γ = 10⁻¹⁴ GeV`, fig. 5 | matches analytic `Y`, `f` (3.30–3.31) |
| 2↔3 relaxation, `N = 8×10⁴`, figs. 6–7 | matches energy-conservation equilibrium; without 2→2, `f` is non-MB at intermediate times |
| benchmark (`m_ϕ = 1 TeV`, `m_ρ = 10 keV`, `|M|² = 2.3×10⁻⁸`, `Γ = 10⁻¹⁷ GeV`, `T₀ = 10 m_ϕ`) | `Ωh² = 0.12` without number-changing processes |
| `2ρ→3ρ`, `|M|²` = 6×10⁻¹², 2×10⁻¹¹, 4×10⁻¹¹ GeV⁻² | ×2 in `|M|²` changes the final abundance by ~×10 |
| convergence, `N_end ≈ 4.5×10⁴, 1.8×10⁵, 4.05×10⁵`, 10 runs each | percent-level precision at the largest `N`; spread larger than Poisson (correlations from decays and 2↔3) |

**What it does not have.** No comparison with an fBE solution (only nBE, cBE and analytic limits),
no timings, no released code (only an acknowledgment of the 2409.07378 code), no resonances, no
elastic scattering on the bath, no quantum statistics, constant matrix elements only. It is
explicitly a proof of concept.

## 2. freezeframe today (what the mapping is against)

* `DarkMatterModel` (`src/model.rs`) exposes exactly `σv_cms(s)` and `γ(T)`; DM is one MB species,
  self-conjugate or `Distinct`. `fprime(model, s̃) = s̃ σv_cms` is the quantity every collision
  integral uses.
* `Method::{NumberDensity, Coupled, PhaseSpace}` (`src/solver/mod.rs`) all run through the
  Rosenbrock integrator (`src/ode.rs`, `OdeSystem` + `Observer`). `Recorder` produces the trace,
  snapshots and `Decoupling`, whose chemical marker is the same 10% criterion the paper uses to start
  DSMC.
* The fBE (`src/solver/systems.rs`, `PhaseSpace`) evolves `f` on a **comoving** grid
  `q̂ = q/r(x)`, `r = [h_eff(T)/h_eff(T_start)]^{1/3}`, so the Liouville term vanishes; annihilation
  uses the rapidity-space `PhiTable` (`src/kernel.rs`), exact in the Breit–Wigner factor; elastic
  scattering is the Chang–Cooper Fokker–Planck operator with zero-flux state `e^{−E/T}`.
* `GridMoments::moments` computes `y` from `⟨p²/E⟩/3`, the same temperature estimator as the
  paper's (3.29).
* `ThermalAverages` (`src/thermal.rs`) integrates `F'(s̃)` against the `K₁` kernel on panels refined
  at resonances and thresholds (`math::peak_edges`).
* `Cosmology::from_rows` accepts any `(T, h_eff, g_eff, g̃)` table, so the paper's constant
  `g* = 3.36`, `g*S = 3.91` cosmology is expressible without code changes.
* The crate has no dependencies (`Cargo.toml`, `src/math/mod.rs`) and no random number generator.
* README limitations that the paper speaks to: no DM self-scattering, no `χχ → mediators`, MB
  statistics, single species.

A smoke run on `bd74c1e` (not committed): `ModelFn` with `σv_cms = |M|²/(16π s)`, `|M|² = 2.3×10⁻⁸`,
`m = 1 TeV`, `γ = 0`, constant `g*` cosmology, `x_end = 10⁵`. All three methods run from
`x_start = 0.1` (fBE ≈ 10 s). Freeze-out is at `x ≈ 7.8`, so the default `x_start = 10` is too
late for this model (nBE `Y` off by 6.6%). `Y_fBE = 1.2827×10⁻⁵` vs `Y_cBE = 1.2821×10⁻⁵`,
`y_fBE/y_cBE − 1 = 1.2%`. Converted to `ρ` (`Y_ρ = 2Y_ϕ`, `m_ρ = 10 keV`) this is `Ωh² ≈ 0.070`
against the paper's 0.12 *(inferred: the σv normalization for identical scalars in the initial and
final state has not been checked against (3.16); a factor ≈1.7 is open)*.

## 3. Candidate results and where they land

### A. Reproduce the paper's freeze-out test with nBE/cBE/fBE

The paper compares DSMC only with nBE and cBE. freezeframe can supply the missing fBE curve for
fig. 4 with no library changes: `ModelFn` (σv from the constant `|M|²`, `γ = 0`),
`Cosmology::from_rows` (constant `g*`), `Solver::x_range(0.1, …)`, `trace_log`, `snapshots`.

* **Touches**: a new `examples/dsmc_freezeout.rs`; nothing in `src/`.
* **Value**: high. First fBE reference for the paper's test; checks freezeframe in a regime it has
  not been run in (constant `g*`, `x_start < 10`, `γ = 0` freeze-out); gives the target data for B.
* **Effort**: small (a day). The σv normalization from (3.16) is the only real work.
* **Risk**: low. Digitizing fig. 4 is approximate; agreement is only meaningful at the few-% level.

### B. A DSMC method for freezeframe's own physics (`Method::Dsmc`)

An independent stochastic solver of the same equation the fBE solves (2→2 annihilation +
Fokker–Planck on the bath), as a cross-check of the fBE in the regime where it matters most (near
resonances, where the README reports `Ω_fBE/Ω_std` up to 55 and cBE capturing half the effect).
Design that fits the crate *(inferred, not in the paper unless noted)*:

* **State**: comoving momenta `q̂_k` of `N` particles in a comoving volume, so the paper's transport
  step is the identity, as in `PhaseSpace`, and `Y ∝ N` exactly (entropy conservation). Physical
  `q = r(x) q̂` from `GridMoments::scale`.
* **Annihilation**: NTC with the paper's `1/E` pair selection (3.22–3.23). In crate variables
  `σ v_Møl = m² F'(s̃)/(E₁E₂)` *(inferred, from `v_Møl = s v_cms/(4E₁E₂)`)*, so the majorant is on
  `F'(s̃)` alone and the acceptance is `F'(s̃_ij)/F'_max`, with `cos θ_ij` sampled per candidate.
* **Production from the bath**: inject `Poisson(V n_eq² ⟨σv⟩ Δt / S)` pairs per step and draw each
  pair's `s̃` by inverse CDF of `F'(s̃) × (K₁ kernel)` on the panels `thermal.rs` already builds
  (resonance-refined), then the pair's CM boost from its thermal distribution. Resonance-robust,
  unlike the paper's accept/reject on `σv/(σv)^max` (3.15).
* **Elastic scattering on the bath**: a Langevin (Itô) step equivalent to
  `C_el = (E/2) γ ∇_p·[E T ∇_p + p] f`, with an exponential (Ornstein–Uhlenbeck-type) update so
  that stiff coupling (`γ/H` up to 10⁶ in `tests/physics.rs`) does not force `Δt ≲ 1/γ`. The paper
  has no bath scattering; this is new.
* **Start**: at the cBE chemical-decoupling marker (`Decoupling::chemical`, the paper's 10%
  criterion), initialized MB at the cBE `T_χ` (paper: bath `T`), so early kinetic decoupling is
  respected.
* **Output**: the existing `Solution` (trace, `Decoupling`, `Ωh²`); snapshots as histograms on
  `PhaseSpaceGrid::edges`, directly comparable with fBE snapshots. Lives in `src/solver/dsmc.rs` to
  reuse the private `Recorder`.
* **RNG**: an in-crate `math::rng` (xoshiro256** seeded by SplitMix64) to keep the crate
  dependency-free; a `seed` on the method for reproducibility.

* **Touches**: `src/solver/mod.rs` (`Method`, dispatch), new `src/solver/dsmc.rs`,
  `src/math/rng.rs`, `src/thermal.rs` (expose the `s̃` panels for sampling), `tests/physics.rs`.
* **Value**: medium. An independent check of the fBE's large effects; noise of a few ×10⁻³ at
  `N ~ 10⁶` cannot test the fBE's 10⁻⁴ convergence, but the effects of interest are O(1)–O(10).
  *(Noise estimate inferred from the paper's percent level at `N ≈ 4×10⁵` and `1/√N` scaling.)*
  It is also the substrate for C.
* **Effort**: medium–large (1–2 weeks).
* **Risks**: (i) narrow resonances make `F'_max` huge and NTC acceptance ~`Γ/M`; needs
  stratification in `s̃` or a tighter piecewise majorant, and is the main open design question;
  (ii) statistical error at late times when `N` has shrunk by the annihilated fraction (splitting,
  which the paper defers to future work); (iii) Itô drift correction and the discrete stationary
  state of the Langevin step must be verified to reproduce `e^{−E/T}`.

### C. DM self-scattering

Listed under "Physics caveats" in the README for the mediator benchmarks; the paper shows it as a
2→2 NTC process among simulated particles and attributes its fig. 4 temperature offset to it.

* **Touches**: `DarkMatterModel` (an optional `self_scattering(s, t)` or `σ_self(s)` with a default
  of zero), `src/solver/dsmc.rs`.
* **Value**: medium. Answers whether the README caveat changes `Ωh²` in the resonance window, where
  the high-velocity tail matters. A deterministic fBE term (a three-momentum kernel, or a BGK
  relaxation) is only worth building if DSMC shows a significant effect *(inferred)*.
* **Effort**: medium once B exists; small relative to a deterministic kernel.
* **Risk**: low numerically; the physics question is which `σ_self` to assume for the
  simplified models.

### D. Decays, freeze-in, number-changing (3→2, cannibal), multiple species

The bulk of the paper's results. freezeframe models one MB species with 2→2 annihilation into the
bath; these need several species, decay widths and `n → m` amplitudes in the model interface.

* **Touches**: `DarkMatterModel` (new species and processes), every solver.
* **Value**: low for the crate's current purpose (early kinetic decoupling in thermal freeze-out).
* **Effort**: large. Not recommended now; keep as an option once B and C exist.

## 4. Recommendation

1. **A** first: cheap, produces the fBE line the paper lacks, and fixes the conventions.
2. **B** next, restricted to smooth `σv` until the resonance majorant is solved; validate against
   the analytic and fBE limits already in `tests/physics.rs`, then against the README's convergence
   table points.
3. **C** on top of B, to settle the self-scattering caveat.
4. **D** deferred.

Task breakdown: [PLAN.md](PLAN.md).

## Caveats

* Paper formulas come from a text extraction with garbled equations; check every one against the
  typeset PDF.
* The ×1.7 gap between the smoke run and the paper's `Ωh² = 0.12` is unexplained (inferred to be
  a σv convention for identical particles); A resolves it.
* B's design (Langevin bath scattering, `s̃`-space inverse-CDF production, start at the cBE
  chemical-decoupling marker) is my proposal, not the paper's.
* The paper reports no timings; DSMC cost relative to the 3–7 s fBE is unknown until B exists.
* The smoke run used rustc 1.97 with `--ignore-rust-version` (the crate declares 1.98).
