//! Physical limits that the three solvers must reproduce.

use freezeframe::prelude::*;

/// `σ v_cms = a + b β²` with `β² = 1 − 4m²/s`: s- plus p-wave.
fn s_plus_p_wave(m: f64, a: f64, b: f64) -> impl Fn(f64) -> f64 {
    move |s: f64| a + b * (1.0 - 4.0 * m * m / s)
}

#[test]
fn number_is_conserved_without_annihilation() {
    let m = 50.0;
    let cosmo = Cosmology::standard();
    // γ = 10⁶ H; for γ/H ≳ 10¹² round-off in the stiff solves starts to show (~1e-5)
    let model = ModelFn::new(m, 1.0, |_| 0.0, |t| 1e6 * cosmo.hubble(t));
    let sol = Solver::new(&model)
        .x_range(10.0, 1e4)
        .trace_at([10.0])
        .solve(Method::phase_space())
        .unwrap();
    let y0 = sol.trace()[0].abundance;
    assert!(
        (sol.final_abundance() / y0 - 1.0).abs() < 1e-9,
        "{} vs {y0}",
        sol.final_abundance()
    );
}

#[test]
fn tight_coupling_reduces_to_standard_equation() {
    let m = 100.0;
    let cosmo = Cosmology::standard();
    // γ = 10⁶ H keeps DM in kinetic equilibrium throughout
    let model = ModelFn::new(m, 1.0, s_plus_p_wave(m, 1e-9, 2e-8), |t| {
        1e6 * cosmo.hubble(t)
    });
    let nbe = relic_density(&model, Method::NumberDensity).unwrap();
    let cbe = relic_density(&model, Method::coupled()).unwrap();
    let fbe = relic_density(&model, Method::phase_space()).unwrap();
    assert!((cbe / nbe - 1.0).abs() < 2e-3, "cBE {cbe} vs nBE {nbe}");
    assert!((fbe / nbe - 1.0).abs() < 3e-3, "fBE {fbe} vs nBE {nbe}");
}

#[test]
fn free_streaming_temperature_agrees_between_methods() {
    let m = 100.0;
    // negligible annihilation: otherwise χχ̄ ↔ SM keeps f thermal until freeze-out
    let model = ModelFn::new(m, 1.0, s_plus_p_wave(m, 1e-16, 0.0), |_| 0.0);
    let s = Solver::new(&model).x_range(10.0, 1e4);
    let cbe = s.solve(Method::coupled()).unwrap();
    let fbe = s.solve(Method::phase_space()).unwrap();
    let r = fbe.final_temp_param() / cbe.final_temp_param();
    assert!((r - 1.0).abs() < 1e-2, "y_fBE / y_cBE = {r}");
    // with no scattering T_chi/T ∝ (x0/x)^w with w(x≈10) ≈ 0.83: 10% deviation at x ≈ 11.5
    for sol in [&cbe, &fbe] {
        let xkd = sol.decoupling().kinetic.unwrap();
        assert!(xkd > 11.0 && xkd < 12.5, "x_kd = {xkd}");
    }
}

#[test]
fn scalar_singlet_regression() {
    // m_S = 57 GeV at the Gondolo-Gelmini coupling (scenario B), cf. Fig. 4 of arXiv:1706.07433
    let model = ScalarSinglet::new(57.0, 2.46861325e-3, QuarkScattering::Minimal).unwrap();
    let s = Solver::new(&model).threads(4);
    let nbe = s.solve(Method::NumberDensity).unwrap();
    let cbe = s.solve(Method::coupled()).unwrap();
    let fbe = s.solve(Method::phase_space()).unwrap();
    assert!((nbe.final_abundance() / 7.5693e-12 - 1.0).abs() < 1e-3);
    assert!((cbe.final_abundance() / 3.9220e-11 - 1.0).abs() < 1e-3);
    assert!((fbe.final_abundance() / 5.0605e-11 - 1.0).abs() < 2e-3);
    assert_eq!(fbe.snapshots().len(), 0);
}

#[test]
fn trait_objects_and_coupling_search() {
    let base = ScalarSinglet::new(45.0, 0.17, QuarkScattering::Minimal).unwrap();
    let search = CouplingSearch {
        target: 0.1188,
        ..Default::default()
    };
    let sol = find_coupling(
        |lam| {
            let model: Box<dyn DarkMatterModel> = Box::new(base.with_coupling(lam));
            relic_density(&model, Method::NumberDensity)
        },
        0.1,
        &search,
    )
    .unwrap();
    // earlier reproduction: λ_GG(45 GeV) = 0.1752 with the 0.4% different Ω normalization
    assert!(
        (sol.coupling / 0.1752 - 1.0).abs() < 1e-2,
        "{}",
        sol.coupling
    );
    assert!((sol.omega_h2 / 0.1188 - 1.0).abs() < 1e-3);
}
