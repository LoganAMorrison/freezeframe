# Plan: DSMC (arXiv:2611.25812) in freezeframe

Analysis and rationale: [README.md](README.md). Each task is self-contained: an agent should be able
to pick it up with this file, the README and the paper.

Ground rules for every task:

* Simple, obvious code that fits the surrounding crate: no new dependencies, no new abstractions
  unless two call sites need them, iterators over index loops where it reads better, `Result` errors
  via `crate::error::Error`.
* Every physics claim gets a test against an analytic limit or an existing solver.
* `cargo test --release` and `cargo clippy --all-targets` pass; README "Validation" updated with any
  new test.
* Formulas from the paper are checked against the typeset PDF, not a text extraction.

## Phase A: reference curves for the paper's freeze-out test

**A1. Fix the conventions.** Derive `σv_cms(s)` for `ϕϕ → ψψ` with constant `|M|²` in the
crate's convention (`model.rs`: averaged over initial dof, summed over final states, self-conjugate
`ϕ`), and compare with the paper's (3.16). Resolve the factor ≈1.7 between the smoke run
(`Ωh² ≈ 0.070`) and the paper's 0.12 for the Sec. 4 benchmark.
*Done when* a short derivation is in this folder (`conventions.md`) and the benchmark `Ωh²` agrees
with 0.12 to the precision of the paper's `g*` and `ρ_c` choices, or the remaining difference is
explained.

**A2. Example `examples/dsmc_freezeout.rs`.** Paper's Sec. 3.4 freeze-out test
(`|M|² = 3.4×10⁻¹¹`) and the Sec. 4 `ϕ` freeze-out (`|M|² = 2.3×10⁻⁸`): constant-`g*` cosmology via
`Cosmology::from_rows`, `γ = 0`, `x_range(0.1, 1e5)`, nBE/cBE/fBE traces (`Y`, `T_ϕ/T`) and fBE
snapshots at a few `x` written as TSV next to the existing `runs/` output.
*Done when* nBE and cBE reproduce fig. 4 by eye, and the fBE curve and its deviation from cBE are
reported in this folder (`results-a.md`) with a figure from `scripts/make_figures.py` or a sibling
script.

**A3. Regression test.** One fast `tests/physics.rs` case for `x_start < 10` with `γ = 0` (fBE `y`
vs cBE `y` to ~2%, as the smoke run gave 1.2%), so the regime stays supported.

## Phase B: `Method::Dsmc`

**B1. RNG.** `src/math/rng.rs`: xoshiro256** seeded by SplitMix64, with `f64` in `[0, 1)`,
standard normal, exponential and Poisson (inversion for small mean, PTRS or normal approximation
for large). Tests: moments and a χ² on a histogram. No public API beyond `math`.

**B2. Skeleton and free streaming.** `Method::Dsmc(DsmcSettings { particles, seed })` in
`src/solver/mod.rs`; `src/solver/dsmc.rs` with comoving `q̂` per particle, fixed `Δln x`, the
existing `Recorder` for trace/`Decoupling`, histogram snapshots on `PhaseSpaceGrid::edges`.
Initialization: MB at `x_start`. With no interactions `q̂` is constant, so `y(x)` must match the
fBE free-streaming test (`tests/physics.rs`, `x_kd ≈ 11.5`) within its noise.

**B3. Bath scattering (Langevin).** Derive the Itô SDE whose Fokker–Planck equation is
`C_el = (E/2) γ ∇_p·[E T ∇_p + p] f` (drift including the noise-induced term), and an exponential
update that is exact for the linearized problem so `γ/H ≫ 1` is stable at `Δln x ~ 10⁻²`.
Tests: stationary `T_χ = T` within noise for `γ = 10⁶ H`; relaxation rate against the cBE `y`
equation in the non-relativistic limit.
*Done when* DSMC `y(x)` agrees with fBE `y(x)` for the Scalar Singlet with annihilation switched off.

**B4. Annihilation.** NTC with `1/E` pair selection (paper 3.22–3.23) and acceptance
`F'(s̃_ij)/F'_max`, `cos θ_ij` sampled per candidate, `σv_Møl = m² F'(s̃)/(E₁E₂)` (verify).
`F'_max` from the `ThermalAverages` panels over the `s̃` range the current ensemble can reach.
Smooth `σv` only in this task. Test: `γ = 0`, pure annihilation from an over-abundant start against
fBE.

**B5. Production from the bath.** Inject `Poisson(V n_eq² ⟨σv⟩ Δt / S)` pairs per step, `S` the
identical-particle factor fixed in A1. Sample `s̃` by inverse CDF of the `⟨σv⟩` integrand on the
`thermal.rs` panels (expose them `pub(crate)`), then the pair's CM rapidity from its thermal
distribution, then isotropic decay. Test: detailed balance, the ensemble stays at `Y_eq`, `T` when
started there with strong annihilation.

**B6. Start at chemical decoupling.** Solve cBE first; start DSMC at `Decoupling::chemical`
initialized MB at the cBE `T_χ`; choose the comoving volume so that `N` particles represent `Y` at
that point. Tests: tight-coupling limit vs nBE (`tests/physics.rs` analogue) and the Scalar Singlet
regression point (`m = 57 GeV`) vs fBE within `3/√N_end`.

**B7. Resonances.** Measure NTC acceptance near the pole (pseudoscalar, `m/M = 0.47`, README
convergence table). If it is unusable, stratify candidate pairs in `s̃` (or use a piecewise majorant
on the `peak_edges` panels). *Done when* the four README convergence-table points agree with fBE
within statistical error, with timings recorded.

**B8. Scan.** DSMC column in `examples/relic_scan.rs` for a subset of masses; figure of
`Ω_DSMC/Ω_fBE − 1` with error bars across the resonance window.

## Phase C: DM self-scattering

**C1. Interface.** Optional `DarkMatterModel::self_scattering_msq(s, t)` (default zero); document
the identical-particle convention.

**C2. DSMC 2→2 among DM.** NTC with `1/E` selection (paper 3.19–3.23), isotropic or
`t`-distributed final state. Tests: energy and number conservation per collision; relaxation of a
non-thermal `f` to MB at fixed energy.

**C3. Physics question.** Rerun B8 with a self-scattering cross section motivated by the mediator
couplings; report whether the README caveat changes `Ωh²` in the window `0.42 ≤ m/M ≤ 0.5`.
Decide on a deterministic fBE term only if it does.

## Phase D: deferred

Decays, freeze-in, `2↔3` and multi-species. Revisit after C.

## Open questions for Logan

1. Is the goal an independent check of the fBE (B, C), or extending the physics toward the paper's
   scenarios (D)? The order above assumes the former.
2. Should `Method::Dsmc` be public API in the crate, or live behind a cargo feature / in an example
   while it is experimental?
3. Keep the crate dependency-free by writing the RNG in-crate (as planned), or accept `rand` as a
   dependency?
4. Self-scattering: which `σ_self` should the simplified models assume (derived from `g_χ` and the
   mediator, or a free parameter)?
5. Do you want to ask the authors for their DSMC code or fig. 4 data for a direct comparison?
