//! Boltzmann solvers and their common front end, [`Solver`].
//!
//! Three levels of description are available (Sec. II of arXiv:1706.07433):
//!
//! | [`Method`]                 | unknowns                 | assumes kinetic equilibrium |
//! |----------------------------|--------------------------|-----------------------------|
//! | [`Method::NumberDensity`]  | `Y(x)`                   | yes (Gondolo-Gelmini)        |
//! | [`Method::Coupled`]        | `Y(x)`, `y(x)`           | Maxwell-Boltzmann shape at `T_χ ≠ T` |
//! | [`Method::PhaseSpace`]     | `f(x, q)`, hence `Y`, `y` | no                          |
//!
//! Here `x = m/T`, `q = p/T`, `Y = n/s` and `y = m T_χ / s^{2/3}`.

mod systems;

use std::io::{self, Write};
use std::ops::ControlFlow;

use crate::cosmology::Cosmology;
use crate::error::{Error, Result};
use crate::model::{DarkMatterModel, ParticleKind};
use crate::ode::{integrate, Observer, OdeOptions, OdeStats};
use crate::thermal::ThermalAverages;

pub(crate) use systems::equilibrium;
use systems::{Coupled, GridMoments, NumberDensity, PhaseSpace};

/// Scattering term in the equation for `y` (coupled method only).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScatteringTerm {
    /// Semi-relativistic operator, Eq. 32 (`γ w(T_χ) (T - T_χ)`).
    #[default]
    SemiRelativistic,
    /// Non-relativistic Fokker-Planck operator, Eq. 5 (`γ (T - T_χ)`).
    NonRelativistic,
}

/// Momentum grid of the full phase-space solver.
///
/// Cells are uniform in `q = p/T` up to `q_max = 60`. For larger `q_max` the
/// spacing is uniform for `q ≲ 30` and grows exponentially beyond
/// (`q = 30 sinh(A u)`, `u` uniform), so that early times stay resolved
/// while the late-time distribution of a kinetically coupled species,
/// whose width grows as `σ_q = sqrt(x T_χ/T)`, still fits on the grid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhaseSpaceGrid {
    /// Number of finite-volume cells.
    pub cells: usize,
    /// Upper end of the grid in `q = p/T`. `None` (default) chooses
    /// `q_max = max(60, 10 σ_q)` from the final `T_χ` of a preliminary
    /// solution of the coupled equations.
    pub q_max: Option<f64>,
}

impl Default for PhaseSpaceGrid {
    /// 300 cells, automatic `q_max` (relic density converged to ~0.1%).
    fn default() -> Self {
        Self {
            cells: 300,
            q_max: None,
        }
    }
}

impl PhaseSpaceGrid {
    /// Grid with a fixed upper end.
    pub fn fixed(cells: usize, q_max: f64) -> Self {
        Self {
            cells,
            q_max: Some(q_max),
        }
    }

    /// Cell edges `0 = q_0 < q_1 < … < q_N = q_max`.
    pub fn edges(cells: usize, q_max: f64) -> Vec<f64> {
        const Q_UNIFORM: f64 = 60.0;
        const Q_LINEAR: f64 = 30.0;
        let n = cells as f64;
        if q_max <= Q_UNIFORM {
            (0..=cells).map(|k| q_max * k as f64 / n).collect()
        } else {
            let a = (q_max / Q_LINEAR).asinh();
            let mut e: Vec<f64> = (0..=cells)
                .map(|k| Q_LINEAR * (a * k as f64 / n).sinh())
                .collect();
            e[cells] = q_max;
            e
        }
    }
}

/// Which Boltzmann hierarchy to solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Method {
    /// Standard number-density equation; computes `Y(x)` only.
    NumberDensity,
    /// Coupled equations for `Y(x)` and `y(x)`.
    Coupled(ScatteringTerm),
    /// Full phase-space density `f(x, q)` on the given grid.
    PhaseSpace(PhaseSpaceGrid),
}

impl Method {
    /// [`Method::Coupled`] with the semi-relativistic scattering term.
    pub fn coupled() -> Self {
        Self::Coupled(ScatteringTerm::SemiRelativistic)
    }

    /// [`Method::PhaseSpace`] on the default grid.
    pub fn phase_space() -> Self {
        Self::PhaseSpace(PhaseSpaceGrid::default())
    }

    /// Short label: `"nBE"`, `"cBE"`, `"cBE-nr"` or `"fBE"`.
    pub fn label(&self) -> &'static str {
        match self {
            Self::NumberDensity => "nBE",
            Self::Coupled(ScatteringTerm::SemiRelativistic) => "cBE",
            Self::Coupled(ScatteringTerm::NonRelativistic) => "cBE-nr",
            Self::PhaseSpace(_) => "fBE",
        }
    }

    fn default_rtol(&self) -> f64 {
        match self {
            Self::PhaseSpace(_) => 1e-4,
            _ => 1e-8,
        }
    }
}

/// `Y`, `y` and their equilibrium values at one `x`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TracePoint {
    /// `x = m/T`.
    pub x: f64,
    /// Abundance `Y = n_χ/s` (one species; see [`ParticleKind`]).
    pub abundance: f64,
    /// Equilibrium abundance `Y_eq`.
    pub abundance_eq: f64,
    /// Temperature parameter `y = m T_χ / s^{2/3}`. For
    /// [`Method::NumberDensity`] this equals `y_eq` by assumption.
    pub temp_param: f64,
    /// Equilibrium value `y_eq = m T / s^{2/3}`.
    pub temp_param_eq: f64,
}

impl TracePoint {
    /// `T_χ / T = y / y_eq`.
    pub fn temperature_ratio(&self) -> f64 {
        self.temp_param / self.temp_param_eq
    }
}

/// Phase-space density at one `x` ([`Method::PhaseSpace`] only).
#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    /// `x = m/T`.
    pub x: f64,
    /// Cell centres in `q = p/T` at this `x`. (The grid is comoving: it
    /// shrinks in `q` by `[h_eff(T)/h_eff(T_start)]^{1/3}` as entropy is
    /// transferred to the photon bath.)
    pub q: Vec<f64>,
    /// `f(q_i)`.
    pub f: Vec<f64>,
}

/// Decoupling markers: first `x` where `Y` (chemical) or `y` (kinetic)
/// deviates by more than 10% from its equilibrium value, linearly
/// interpolated in `ln x` between accepted steps.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Decoupling {
    /// Chemical decoupling, `|Y - Y_eq| = 0.1 Y_eq`.
    pub chemical: Option<f64>,
    /// Kinetic decoupling, `|y - y_eq| = 0.1 y_eq` (not for `NumberDensity`).
    pub kinetic: Option<f64>,
}

/// Result of a [`Solver::solve`] call.
#[derive(Clone, Debug)]
pub struct Solution {
    method: Method,
    mass: f64,
    kind: ParticleKind,
    x_end: f64,
    abundance_end: f64,
    temp_param_end: f64,
    omega_h2: f64,
    trace: Vec<TracePoint>,
    snapshots: Vec<Snapshot>,
    grid: Option<Vec<f64>>,
    edges: Option<Vec<f64>>,
    decoupling: Decoupling,
    stats: OdeStats,
}

impl Solution {
    /// Method that produced this solution.
    pub fn method(&self) -> Method {
        self.method
    }
    /// Dark-matter mass \[GeV\].
    pub fn mass(&self) -> f64 {
        self.mass
    }
    /// Final `x`.
    pub fn x_end(&self) -> f64 {
        self.x_end
    }
    /// `Y(x_end)` of one species.
    pub fn final_abundance(&self) -> f64 {
        self.abundance_end
    }
    /// `y(x_end)`.
    pub fn final_temp_param(&self) -> f64 {
        self.temp_param_end
    }
    /// Relic density `Ω h²` (both species for [`ParticleKind::Distinct`]).
    pub fn omega_h2(&self) -> f64 {
        self.omega_h2
    }
    /// Particle kind of the model.
    pub fn kind(&self) -> ParticleKind {
        self.kind
    }
    /// Requested trace points.
    pub fn trace(&self) -> &[TracePoint] {
        &self.trace
    }
    /// Requested phase-space snapshots.
    pub fn snapshots(&self) -> &[Snapshot] {
        &self.snapshots
    }
    /// Comoving cell centres `q̂_i` (equal to `q = p/T` at `x_start`).
    pub fn momentum_grid(&self) -> Option<&[f64]> {
        self.grid.as_deref()
    }
    /// Comoving cell edges (`len = cells + 1`).
    pub fn cell_edges(&self) -> Option<&[f64]> {
        self.edges.as_deref()
    }
    /// Decoupling markers.
    pub fn decoupling(&self) -> Decoupling {
        self.decoupling
    }
    /// Integrator work counters.
    pub fn stats(&self) -> OdeStats {
        self.stats
    }

    /// Write the trace as tab-separated columns `x Y Y_eq y y_eq`.
    pub fn write_trace<W: Write>(&self, mut w: W) -> io::Result<()> {
        writeln!(
            w,
            "# method={} m={:e} GeV  Omega_h2={:.6e}",
            self.method.label(),
            self.mass,
            self.omega_h2
        )?;
        writeln!(w, "# x\tY\tY_eq\ty\ty_eq")?;
        for p in &self.trace {
            writeln!(
                w,
                "{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}",
                p.x, p.abundance, p.abundance_eq, p.temp_param, p.temp_param_eq
            )?;
        }
        Ok(())
    }

    /// Write the snapshots as column pairs `q(x_k) f(x_k, q)`.
    pub fn write_snapshots<W: Write>(&self, mut w: W) -> io::Result<()> {
        let Some(n) = self.snapshots.first().map(|s| s.q.len()) else {
            return Ok(());
        };
        let header: Vec<String> = self
            .snapshots
            .iter()
            .map(|s| format!("q(x={0})\tf(x={0})", s.x))
            .collect();
        writeln!(w, "# {}", header.join("\t"))?;
        for i in 0..n {
            let cols: Vec<String> = self
                .snapshots
                .iter()
                .map(|s| format!("{:.8e}\t{:.8e}", s.q[i], s.f[i]))
                .collect();
            writeln!(w, "{}", cols.join("\t"))?;
        }
        Ok(())
    }
}

/// Configures and runs a Boltzmann solver for one model.
///
/// ```no_run
/// use dmphase::prelude::*;
/// # fn main() -> dmphase::Result<()> {
/// let model = ScalarSinglet::new(57.0, 1.6e-3, QuarkScattering::Minimal)?;
/// let sol = Solver::new(&model)
///     .trace_log(10.0, 1e3, 200)
///     .snapshots([16.0, 20.0, 25.0, 50.0])
///     .solve(Method::phase_space())?;
/// println!("Ωh² = {:.4}", sol.omega_h2());
/// # Ok(()) }
/// ```
#[derive(Debug)]
pub struct Solver<'a, M: ?Sized> {
    model: &'a M,
    cosmology: &'a Cosmology,
    x_start: f64,
    x_end: f64,
    rtol: Option<f64>,
    trace_x: Vec<f64>,
    snapshot_x: Vec<f64>,
    threads: usize,
}

impl<M: ?Sized> Clone for Solver<'_, M> {
    fn clone(&self) -> Self {
        Self {
            model: self.model,
            cosmology: self.cosmology,
            x_start: self.x_start,
            x_end: self.x_end,
            rtol: self.rtol,
            trace_x: self.trace_x.clone(),
            snapshot_x: self.snapshot_x.clone(),
            threads: self.threads,
        }
    }
}

impl<'a, M: DarkMatterModel + ?Sized> Solver<'a, M> {
    /// Solver with the standard SM cosmology, `x ∈ [10, 10⁶]`.
    pub fn new(model: &'a M) -> Self {
        Self {
            model,
            cosmology: Cosmology::standard(),
            x_start: 10.0,
            x_end: 1e6,
            rtol: None,
            trace_x: Vec::new(),
            snapshot_x: Vec::new(),
            threads: 1,
        }
    }

    /// Use a different thermal history.
    pub fn cosmology(mut self, c: &'a Cosmology) -> Self {
        self.cosmology = c;
        self
    }

    /// Integration range in `x = m/T`. DM is assumed to be in full
    /// equilibrium at `x_start`. The relic density needs `x_end ≳ 10⁵–10⁶`,
    /// in particular near thresholds or resonances.
    pub fn x_range(mut self, x_start: f64, x_end: f64) -> Self {
        self.x_start = x_start;
        self.x_end = x_end;
        self
    }

    /// Relative tolerance of the integrator (default: 1e-8 for the moment
    /// equations, 1e-4 for the phase-space equation).
    pub fn rtol(mut self, rtol: f64) -> Self {
        self.rtol = Some(rtol);
        self
    }

    /// Worker threads used to rebuild the annihilation kernel of
    /// [`Method::PhaseSpace`] (default 1). Results do not depend on this
    /// setting. Keep it at 1 when running many solves in parallel.
    pub fn threads(mut self, threads: usize) -> Self {
        self.threads = threads.max(1);
        self
    }

    /// Record `Y`, `y` at these values of `x`.
    pub fn trace_at(mut self, xs: impl IntoIterator<Item = f64>) -> Self {
        self.trace_x.extend(xs);
        self
    }

    /// Record `n + 1` log-spaced trace points on `[x0, x1]`.
    pub fn trace_log(self, x0: f64, x1: f64, n: usize) -> Self {
        let (l0, l1) = (x0.ln(), x1.ln());
        let n = n.max(1);
        self.trace_at((0..=n).map(move |i| (l0 + (l1 - l0) * i as f64 / n as f64).exp()))
    }

    /// Store `f(q)` at these values of `x` ([`Method::PhaseSpace`] only).
    pub fn snapshots(mut self, xs: impl IntoIterator<Item = f64>) -> Self {
        self.snapshot_x.extend(xs);
        self
    }

    fn validate(&self, method: &Method) -> Result<()> {
        let m = self.model.mass();
        let g = self.model.dof();
        if !(m > 0.0 && m.is_finite()) {
            return Err(Error::InvalidModel(format!(
                "mass must be positive, got {m}"
            )));
        }
        if g.is_nan() || g <= 0.0 {
            return Err(Error::InvalidModel(format!(
                "degrees of freedom must be positive, got {g}"
            )));
        }
        if !(self.x_start > 0.0 && self.x_end > self.x_start) {
            return Err(Error::InvalidInput(format!(
                "bad x range [{}, {}]",
                self.x_start, self.x_end
            )));
        }
        if let Method::PhaseSpace(grid) = method {
            if grid.cells < 10 || grid.q_max.is_some_and(|q| q.is_nan() || q <= 0.0) {
                return Err(Error::InvalidInput(format!(
                    "bad phase-space grid {grid:?}"
                )));
            }
        }
        for t in [m / self.x_start, m / self.x_end] {
            let gam = self.model.momentum_transfer_rate(t);
            if !(gam.is_finite() && gam >= 0.0) {
                return Err(Error::InvalidModel(format!("γ(T = {t:e}) = {gam}")));
            }
        }
        Ok(())
    }

    /// `q_max = max(60, 10 σ_q)` with `σ_q² = x T_χ/T` at `x_end` from the
    /// coupled equations: the asymptotic comoving width of `f` in `q`.
    fn auto_q_max(&self) -> Result<f64> {
        let probe = Solver::new(self.model)
            .cosmology(self.cosmology)
            .x_range(self.x_start, self.x_end);
        let c = probe.solve(Method::coupled())?;
        let (_, y_eq) = equilibrium(
            self.model.mass(),
            self.model.dof(),
            self.cosmology,
            self.x_end,
        );
        let sigma2 = self.x_end * c.final_temp_param() / y_eq;
        Ok((10.0 * sigma2.sqrt()).clamp(60.0, 1e4))
    }

    /// Solve with the chosen method.
    pub fn solve(&self, method: Method) -> Result<Solution> {
        self.validate(&method)?;
        let m = self.model.mass();
        let dof = self.model.dof();
        let cosmo = self.cosmology;
        let (x0, x1) = (self.x_start, self.x_end);
        let eq = |x: f64| equilibrium(m, dof, cosmo, x);
        let rtol = self.rtol.unwrap_or_else(|| method.default_rtol());

        let mut trace_x: Vec<f64> = self
            .trace_x
            .iter()
            .copied()
            .filter(|&x| x >= x0 && x <= x1)
            .collect();
        trace_x.sort_by(f64::total_cmp);
        trace_x.dedup();
        let mut snap_x: Vec<f64> = match method {
            Method::PhaseSpace(_) => self
                .snapshot_x
                .iter()
                .copied()
                .filter(|&x| x >= x0 && x <= x1)
                .collect(),
            _ => Vec::new(),
        };
        snap_x.sort_by(f64::total_cmp);
        snap_x.dedup();
        let mut outputs: Vec<f64> = trace_x.iter().chain(&snap_x).copied().collect();
        outputs.sort_by(f64::total_cmp);
        outputs.dedup();

        let (ye0, yse0) = eq(x0);
        let (_, stats, rec, grid, moments_end) = match method {
            Method::NumberDensity => {
                let th = ThermalAverages::new(self.model, 0.5 * x0, 2.0 * x1, false)?;
                let mut sys = NumberDensity {
                    mass: m,
                    dof,
                    cosmo,
                    averages: &th,
                };
                let mut rec = Recorder::new(
                    &trace_x,
                    &snap_x,
                    |x, v: &[f64]| (v[0], eq(x).1),
                    &eq,
                    false,
                );
                let opts = OdeOptions {
                    rtol,
                    h0: 1e-6,
                    max_step_fraction: 0.02,
                    ..Default::default()
                };
                let (y, st) = integrate(&mut sys, x0, &[ye0], x1, &outputs, &opts, &mut rec)?;
                let end = (y[0], eq(x1).1);
                (y, st, rec.finish(), None, end)
            }
            Method::Coupled(scattering) => {
                let th =
                    ThermalAverages::new(self.model, (0.1 * x0).min(1.0), 1e10_f64.max(x1), true)?;
                let mut sys = Coupled {
                    model: self.model,
                    dof,
                    cosmo,
                    averages: &th,
                    scattering,
                };
                let mut rec =
                    Recorder::new(&trace_x, &snap_x, |_, v: &[f64]| (v[0], v[1]), &eq, true);
                let opts = OdeOptions {
                    rtol,
                    h0: 1e-6,
                    max_step_fraction: 0.005,
                    ..Default::default()
                };
                let (y, st) = integrate(&mut sys, x0, &[ye0, yse0], x1, &outputs, &opts, &mut rec)?;
                let end = (y[0], y[1]);
                (y, st, rec.finish(), None, end)
            }
            Method::PhaseSpace(grid) => {
                let q_max = match grid.q_max {
                    Some(q) => q,
                    None => self.auto_q_max()?,
                };
                let edges = PhaseSpaceGrid::edges(grid.cells, q_max);
                let mut sys = PhaseSpace::new(self.model, cosmo, edges, x0, self.threads);
                let gm = GridMoments {
                    mass: m,
                    dof,
                    cosmo,
                    q_hat: sys.q_hat.clone(),
                    vol_hat: sys.vol_hat.clone(),
                    heff_start: sys.heff_start(),
                };
                let f0 = sys.equilibrium_distribution(x0);
                let mut rec = Recorder::new(
                    &trace_x,
                    &snap_x,
                    |x, f: &[f64]| gm.moments(x, f),
                    &eq,
                    true,
                );
                let scale = |x: f64| gm.scale(x);
                rec.grid_scale = Some(&scale);
                rec.grid_hat = &gm.q_hat;
                let opts = OdeOptions {
                    rtol,
                    atol_scaled: 1e-7,
                    h0: 1e-5,
                    max_step_fraction: 0.05,
                    ..Default::default()
                };
                let (f, st) = integrate(&mut sys, x0, &f0, x1, &outputs, &opts, &mut rec)?;
                let end = gm.moments(x1, &f);
                let rec = rec.finish();
                (
                    f,
                    st,
                    rec,
                    Some((gm.q_hat.clone(), sys.qe_hat.clone())),
                    end,
                )
            }
        };
        let (abundance_end, temp_param_end) = moments_end;
        if !(abundance_end.is_finite() && abundance_end > 0.0) {
            return Err(Error::InvalidModel(format!(
                "final abundance is {abundance_end}"
            )));
        }
        let kind = self.model.kind();
        Ok(Solution {
            method,
            mass: m,
            kind,
            x_end: x1,
            abundance_end,
            temp_param_end,
            omega_h2: cosmo.omega_h2(m, kind.species() * abundance_end),
            trace: rec.trace,
            snapshots: rec.snapshots,
            grid: grid.as_ref().map(|g| g.0.clone()),
            edges: grid.map(|g| g.1),
            decoupling: rec.decoupling,
            stats,
        })
    }
}

/// Relic density `Ω h²` of `model` with the default solver settings.
pub fn relic_density<M: DarkMatterModel + ?Sized>(model: &M, method: Method) -> Result<f64> {
    Solver::new(model).solve(method).map(|s| s.omega_h2())
}

/// Collects trace points, snapshots and decoupling markers.
struct Recorder<'r, Mo, Eq> {
    trace_x: &'r [f64],
    snap_x: &'r [f64],
    moments: Mo,
    eq: &'r Eq,
    track_kinetic: bool,
    prev: Option<(f64, f64, f64)>,
    trace: Vec<TracePoint>,
    snapshots: Vec<Snapshot>,
    decoupling: Decoupling,
    /// comoving centres and `r(x)` for phase-space snapshots
    grid_scale: Option<&'r dyn Fn(f64) -> f64>,
    grid_hat: &'r [f64],
}

struct Recorded {
    trace: Vec<TracePoint>,
    snapshots: Vec<Snapshot>,
    decoupling: Decoupling,
}

impl<'r, Mo, Eq> Recorder<'r, Mo, Eq>
where
    Mo: Fn(f64, &[f64]) -> (f64, f64),
    Eq: Fn(f64) -> (f64, f64),
{
    fn new(
        trace_x: &'r [f64],
        snap_x: &'r [f64],
        moments: Mo,
        eq: &'r Eq,
        track_kinetic: bool,
    ) -> Self {
        Self {
            trace_x,
            snap_x,
            moments,
            eq,
            track_kinetic,
            prev: None,
            trace: Vec::with_capacity(trace_x.len()),
            snapshots: Vec::with_capacity(snap_x.len()),
            decoupling: Decoupling::default(),
            grid_scale: None,
            grid_hat: &[],
        }
    }

    fn finish(self) -> Recorded {
        Recorded {
            trace: self.trace,
            snapshots: self.snapshots,
            decoupling: self.decoupling,
        }
    }
}

impl<Mo, Eq> Observer for Recorder<'_, Mo, Eq>
where
    Mo: Fn(f64, &[f64]) -> (f64, f64),
    Eq: Fn(f64) -> (f64, f64),
{
    fn output(&mut self, x: f64, state: &[f64]) {
        if self.trace_x.binary_search_by(|v| v.total_cmp(&x)).is_ok() {
            let (yy, ys) = (self.moments)(x, state);
            let (ye, yse) = (self.eq)(x);
            self.trace.push(TracePoint {
                x,
                abundance: yy,
                abundance_eq: ye,
                temp_param: ys,
                temp_param_eq: yse,
            });
        }
        if self.snap_x.binary_search_by(|v| v.total_cmp(&x)).is_ok() {
            let r = self.grid_scale.map_or(1.0, |g| g(x));
            let q = self.grid_hat.iter().map(|qh| r * qh).collect();
            self.snapshots.push(Snapshot {
                x,
                q,
                f: state.to_vec(),
            });
        }
    }

    fn step(&mut self, x: f64, state: &[f64]) -> ControlFlow<()> {
        let d = &mut self.decoupling;
        if d.chemical.is_some() && (d.kinetic.is_some() || !self.track_kinetic) {
            return ControlFlow::Continue(());
        }
        let (yy, ys) = (self.moments)(x, state);
        let (ye, yse) = (self.eq)(x);
        let (d1, d2) = ((yy - ye).abs() / ye, (ys - yse).abs() / yse);
        if let Some((xp, d1p, d2p)) = self.prev {
            let cross = |a: f64, b: f64| {
                let s = ((0.1 - a) / (b - a)).clamp(0.0, 1.0);
                (xp.ln() + s * (x.ln() - xp.ln())).exp()
            };
            if d.chemical.is_none() && d1 > 0.1 {
                d.chemical = Some(cross(d1p, d1));
            }
            if self.track_kinetic && d.kinetic.is_none() && d2 > 0.1 {
                d.kinetic = Some(cross(d2p, d2));
            }
        }
        self.prev = Some((x, d1, d2));
        ControlFlow::Continue(())
    }
}
