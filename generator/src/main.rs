use std::f64::consts::PI;
use freezeframe::math::bessel_k_scaled;
use freezeframe::prelude::*;
use freezeframe::thermal::thermal_averages;
use freezeframe::{relic_density, Method};

type R = std::result::Result<(), Box<dyn std::error::Error>>;

fn axial(m: f64, big_m: f64, gx: f64, gf: f64, w: f64) -> SimplifiedModel {
    SimplifiedModel::builder(Mediator::AxialVector)
        .dm_mass(m).mediator_mass(big_m).dm_coupling(gx)
        .sm_coupling(SmFermion::Electron, gf).sm_coupling(SmFermion::Muon, gf)
        .extra_width(w).build().unwrap()
}

fn omegas(label: &str, model: &dyn DarkMatterModel) -> R {
    let n = relic_density(model, Method::NumberDensity)?;
    let c = relic_density(model, Method::coupled())?;
    let f = relic_density(model, Method::phase_space())?;
    println!("{label}\tnBE {n:.6e}\tcBE {c:.6e}\tfBE {f:.6e}");
    Ok(())
}

fn main() -> R {
    println!("## A1 axial sigma_v at s = M^2 = 4m^2 (m=0.5, M=1, g_chi=0.3, g_l=0.01)");
    for w in [0.0, 1e-3, 0.1] {
        println!("extra_width {w}\tsigma_v {:.6e}", axial(0.5, 1.0, 0.3, 0.01, w).sigma_v(1.0));
    }
    println!("## A2 axial sigma_v(s) [GeV^-2] near the pole (m=0.4, M=1, g_chi=1, g_l=0.1, extra_width=0.2)");
    let a = axial(0.4, 1.0, 1.0, 0.1, 0.2);
    for s in [0.9, 0.98, 1.0, 1.02, 1.1] {
        println!("s {s}\tsigma_v {:.6e}", a.sigma_v(s));
    }
    println!("## A3 axial relic density, broad mediator at M = 2m (m=0.5, M=1, g_chi=1, g_l=1e-3)");
    for w in [0.0, 0.01, 0.1] {
        omegas(&format!("extra_width {w}"), &axial(0.5, 1.0, 1.0, 1e-3, w))?;
    }

    println!("## B1 pion benchmark: thermal average / narrow-width limit");
    let (mp, gamma) = (0.1349768_f64, 7.8e-9);
    let m = 0.99 * mp / 2.0;
    let c = (1.6e-4 / 137.035999_f64).powi(2) * m * m / (4.0 * PI.powi(3));
    let sigma = move |s: f64| c * s * s / ((s - mp * mp).powi(2) + (mp * gamma).powi(2));
    let pion = ModelFn::new(m, 2.0, sigma, |_| 0.0).with_resonance(mp, gamma);
    let (sr, ws) = (mp * mp / (4.0 * m * m), mp * gamma / (4.0 * m * m));
    for x in [20.0_f64, 100.0] {
        let a = 2.0 * x * sr.sqrt();
        let k2 = bessel_k_scaled(2.0, x);
        let damp = (2.0 * x - a).exp() / (k2 * k2);
        let area = c * sr.powi(3) * PI / ws;
        let nwa1 = area * damp * 2.0 * x * (sr - 1.0).sqrt() * bessel_k_scaled(1.0, a);
        let (s1, _) = thermal_averages(&pion, x, false);
        println!("x {x}\t<sv>/NWA {:.8}", s1 / nwa1);
    }

    println!("## B2 Scalar Singlet benchmarks (Minimal quark scattering, couplings from runs/scan_ss.tsv)");
    for (m, l) in [(57.0, 2.474826e-3), (62.55, 8.882341e-4)] {
        omegas(&format!("m {m}"), &ScalarSinglet::new(m, l, QuarkScattering::Minimal)?)?;
    }
    println!("## B3 axial scan points near resonance (M=1, universal coupling from runs/scan_axial.tsv)");
    for (m, g) in [(0.495, 6.858078e-5)] {
        let mdl = SimplifiedModel::builder(Mediator::AxialVector).dm_mass(m).mediator_mass(1.0)
            .dm_coupling(g).sm_coupling(SmFermion::Electron, g).sm_coupling(SmFermion::Muon, g).build()?;
        println!("width {:.4e}", mdl.mediator_width());
        omegas(&format!("m {m}"), &mdl)?;
    }
    println!("## B4 nBE Omega at every committed scan point (coupling g_nBE from runs/scan_*.tsv)");
    for name in ["ss", "scalar", "pseudoscalar", "vector", "axial"] {
        let text = std::fs::read_to_string(format!("/Users/logan.morrison/dev/freezeframe/runs/scan_{name}.tsv"))?;
        for line in text.lines().filter(|l| !l.starts_with('#')) {
            let c: Vec<f64> = line.split('\t').take(2).map(|v| v.parse().unwrap()).collect();
            let (m, g) = (c[0], c[1]);
            let mdl: Box<dyn DarkMatterModel> = match name {
                "ss" => Box::new(ScalarSinglet::new(m, g, QuarkScattering::Minimal)?),
                _ => {
                    let med = match name { "scalar" => Mediator::Scalar, "pseudoscalar" => Mediator::Pseudoscalar,
                        "vector" => Mediator::Vector, _ => Mediator::AxialVector };
                    Box::new(SimplifiedModel::builder(med).dm_mass(m).mediator_mass(1.0).dm_coupling(g)
                        .sm_coupling(SmFermion::Electron, g).sm_coupling(SmFermion::Muon, g).build()?)
                }
            };
            println!("scan\t{name}\t{m}\t{:.9e}", relic_density(&*mdl, Method::NumberDensity)?);
        }
    }
    Ok(())
}
