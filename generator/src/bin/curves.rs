use std::f64::consts::PI;
use freezeframe::math::bessel_k_scaled;
use freezeframe::prelude::*;
use freezeframe::thermal::thermal_averages;

fn main() {
    // Axial-vector sigma_v above threshold with M = 2m (m = 0.5, M = 1, g_chi = 0.3, g_l = 0.01).
    for w in [1e-3, 0.1] {
        let mdl = SimplifiedModel::builder(Mediator::AxialVector)
            .dm_mass(0.5).mediator_mass(1.0).dm_coupling(0.3)
            .sm_coupling(SmFermion::Electron, 0.01).sm_coupling(SmFermion::Muon, 0.01)
            .extra_width(w).build().unwrap();
        for k in 0..=120 {
            let u = 10f64.powf(-8.0 + 7.0 * k as f64 / 120.0);
            println!("axial\t{w}\t{u:.6e}\t{:.9e}", mdl.sigma_v(1.0 + u));
        }
    }
    // pi0-like pole: <sigma v> divided by the narrow-width limit.
    let (mp, gamma) = (0.1349768_f64, 7.8e-9);
    let m = 0.99 * mp / 2.0;
    let c = (1.6e-4 / 137.035999_f64).powi(2) * m * m / (4.0 * PI.powi(3));
    let sigma = move |s: f64| c * s * s / ((s - mp * mp).powi(2) + (mp * gamma).powi(2));
    let pion = ModelFn::new(m, 2.0, sigma, |_| 0.0).with_resonance(mp, gamma);
    let (sr, ws) = (mp * mp / (4.0 * m * m), mp * gamma / (4.0 * m * m));
    for k in 0..=60 {
        let x = 10f64.powf(1.0 + 2.0 * k as f64 / 60.0);
        let a = 2.0 * x * sr.sqrt();
        let k2 = bessel_k_scaled(2.0, x);
        let nwa = c * sr.powi(3) * PI / ws * (2.0 * x - a).exp() / (k2 * k2)
            * 2.0 * x * (sr - 1.0).sqrt() * bessel_k_scaled(1.0, a);
        println!("pion\t{x:.6e}\t{:.10}", thermal_averages(&pion, x, false).0 / nwa);
    }
}
