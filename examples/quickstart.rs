//! Minimal tour of the API: define a model, then compute
//!   1. the relic density only,
//!   2. Y(x) [and y(x)] from the moment equations,
//!   3. the full phase-space density f(x, q).
//!
//! cargo run --release --example quickstart

use std::f64::consts::PI;
use std::fs::File;

use freezeframe::prelude::*;
use freezeframe::scattering::Scatterer;

/// A self-conjugate scalar of mass `m` annihilating through a heavy
/// s-channel scalar of mass `big_m` and width `width` into electrons, with
/// t-channel scattering on e±. Any type implementing `DarkMatterModel` works.
struct ToyResonance {
    m: f64,
    big_m: f64,
    width: f64,
    c: f64,
}

impl DarkMatterModel for ToyResonance {
    fn mass(&self) -> f64 {
        self.m
    }
    fn dof(&self) -> f64 {
        1.0
    }
    /// σ v_cms(s) [GeV⁻²]
    fn sigma_v(&self, s: f64) -> f64 {
        let d = (s - self.big_m.powi(2)).powi(2) + (self.big_m * self.width).powi(2);
        self.c * self.c * s / (8.0 * PI * d)
    }
    /// γ(T) [GeV] from |M|²(s, t) for χ e → χ e, summed over spins
    fn momentum_transfer_rate(&self, t: f64) -> f64 {
        let (c, big_m, me) = (self.c, self.big_m, 0.000511);
        let msq = move |_s: f64, tt: f64| {
            2.0 * c * c * (4.0 * me * me - tt) / (tt - big_m * big_m).powi(2)
        };
        Scatterer::fermion_pair(me, msq).momentum_transfer_rate(self.m, 1.0, t)
    }
    fn resonances(&self) -> Vec<Resonance> {
        vec![Resonance {
            mass: self.big_m,
            width: self.width,
        }]
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = ToyResonance {
        m: 0.1,
        big_m: 0.19,
        width: 1e-6,
        c: 7.1e-6,
    };

    // 1. relic density only
    for method in [
        Method::NumberDensity,
        Method::coupled(),
        Method::phase_space(),
    ] {
        println!(
            "{:>4}: Ωh² = {:.4e}",
            method.label(),
            relic_density(&model, method)?
        );
    }

    // 2. Y(x) and y(x) along the way
    let cbe = Solver::new(&model)
        .trace_log(10.0, 1e3, 100)
        .solve(Method::coupled())?;
    let d = cbe.decoupling();
    println!(
        "cBE: x_cd = {:.1}, x_kd = {:.1}",
        d.chemical.unwrap_or(f64::NAN),
        d.kinetic.unwrap_or(f64::NAN)
    );
    cbe.write_trace(File::create("quickstart_trace_cbe.tsv")?)?;

    // 3. full phase-space density at selected x
    let fbe = Solver::new(&model)
        .threads(4)
        .trace_log(10.0, 1e3, 100)
        .snapshots([15.0, 30.0, 100.0])
        .solve(Method::phase_space())?;
    for snap in fbe.snapshots() {
        let (i, f) = snap
            .f
            .iter()
            .enumerate()
            .max_by(|a, b| (snap.q[a.0].powi(2) * a.1).total_cmp(&(snap.q[b.0].powi(2) * b.1)))
            .unwrap();
        println!(
            "fBE x = {:>5}: q²f peaks at q = {:.2} (f = {f:.3e})",
            snap.x, snap.q[i]
        );
    }
    println!(
        "fBE: Y_∞ = {:.4e}, T_χ/T(x=1e3) = {:.3e}",
        fbe.final_abundance(),
        fbe.trace().last().unwrap().temperature_ratio()
    );
    fbe.write_snapshots(File::create("quickstart_snapshots.tsv")?)?;
    Ok(())
}
