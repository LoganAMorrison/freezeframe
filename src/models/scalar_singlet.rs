//! Scalar Singlet ("Higgs portal") dark matter, Sec. III of arXiv:1706.07433.
//!
//! ```text
//! L ⊃ −(λ_S/2) S² |H|²
//! σv_cms = 2 λ_S² v₀² / sqrt(s) · Γ_{h→SM}(sqrt s) / [(s − m_h²)² + m_h² Γ_h²]       (Eqs. 26–27)
//! |M|²(S f → S f) = 2 N_c λ_S² m_f² (4m_f² − t) / (t − m_h²)²                          (Eq. 29)
//! ```

use std::f64::consts::PI;
use std::sync::OnceLock;

use super::SmFermion;
use crate::constants::{M_HIGGS, T_QCD, V_HIGGS};
use crate::error::{Error, Result};
use crate::math::CubicSpline;
use crate::model::{DarkMatterModel, Resonance};
use crate::scattering::{RateTable, Scatterer};

/// SM Higgs width at `m_h` used in the propagator \[GeV\] (paper value).
pub const HIGGS_WIDTH_SM: f64 = 4.042e-3;

static HIGGS_WIDTH_TABLE: &str = include_str!("../../data/higgs_width_hdecay.dat");

/// SM width of a Higgs-like scalar of mass `mass` (HDECAY table shipped with
/// DRAKE), rescaled so that `Γ(m_h) = HIGGS_WIDTH_SM`.
fn higgs_width_sm(mass: f64) -> f64 {
    static TABLE: OnceLock<(CubicSpline, f64)> = OnceLock::new();
    let (spl, scale) = TABLE.get_or_init(|| {
        let (xs, ys): (Vec<f64>, Vec<f64>) = HIGGS_WIDTH_TABLE
            .lines()
            .filter(|l| !l.trim_start().starts_with('#'))
            .filter_map(|l| {
                let v: Vec<f64> = l
                    .split_whitespace()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                (v.len() == 2).then(|| (v[0], v[1].ln()))
            })
            .unzip();
        let spl = CubicSpline::new(xs, ys).expect("bundled Higgs width table is valid");
        let scale = HIGGS_WIDTH_SM / spl.eval(M_HIGGS).exp();
        (spl, scale)
    });
    scale * spl.eval(mass).exp()
}

/// Treatment of scattering on quarks around the QCD transition (Sec. III.A).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuarkScattering {
    /// Scenario A: all quarks are free scatterers down to `T_c = 154 MeV`.
    Maximal,
    /// Scenario B: only u, d, s and only above `4 T_c`.
    Minimal,
}

impl QuarkScattering {
    fn active(self, f: SmFermion, t: f64) -> bool {
        match (f.is_quark(), self) {
            (false, _) => true,
            (true, Self::Maximal) => t > T_QCD,
            (true, Self::Minimal) => f.is_light_quark() && t > 4.0 * T_QCD,
        }
    }

    fn switch_temperature(self) -> f64 {
        match self {
            Self::Maximal => T_QCD,
            Self::Minimal => 4.0 * T_QCD,
        }
    }
}

/// `γ(T)/λ_S²` from the generic scattering integral.
fn unit_rate(mass: f64, scenario: QuarkScattering, t: f64) -> f64 {
    SmFermion::ALL
        .iter()
        .filter(|f| scenario.active(**f, t))
        .map(|&f| {
            let (mf, nc) = (f.mass(), f.colors());
            let msq = move |_s: f64, tt: f64| {
                2.0 * nc * mf * mf * (4.0 * mf * mf - tt) / (tt - M_HIGGS * M_HIGGS).powi(2)
            };
            Scatterer::fermion_pair(mf, msq).momentum_transfer_rate(mass, 1.0, t)
        })
        .sum()
}

/// Real scalar singlet coupled through the Higgs portal.
#[derive(Clone, Debug)]
pub struct ScalarSinglet {
    mass: f64,
    lambda: f64,
    scenario: QuarkScattering,
    higgs_total_width: f64,
    unit_gamma: RateTable,
}

impl ScalarSinglet {
    /// Model with mass `mass` \[GeV\], portal coupling `lambda` and a quark
    /// scattering scenario.
    pub fn new(mass: f64, lambda: f64, scenario: QuarkScattering) -> Result<Self> {
        if mass.is_nan() || mass <= 0.0 {
            return Err(Error::InvalidInput(format!(
                "mass must be positive, got {mass}"
            )));
        }
        let unit_gamma = RateTable::new(
            mass * 1e-9,
            2.0 * mass,
            40,
            &[scenario.switch_temperature()],
            |t| unit_rate(mass, scenario, t),
        )?;
        Ok(Self::assemble(mass, lambda, scenario, unit_gamma))
    }

    fn assemble(mass: f64, lambda: f64, scenario: QuarkScattering, unit_gamma: RateTable) -> Self {
        let invisible = if 2.0 * mass < M_HIGGS {
            lambda * lambda * V_HIGGS * V_HIGGS / (32.0 * PI * M_HIGGS)
                * (1.0 - 4.0 * mass * mass / (M_HIGGS * M_HIGGS)).sqrt()
        } else {
            0.0
        };
        Self {
            mass,
            lambda,
            scenario,
            higgs_total_width: HIGGS_WIDTH_SM + invisible,
            unit_gamma,
        }
    }

    /// Same mass and scenario with a different coupling (reuses the
    /// tabulated scattering rate).
    pub fn with_coupling(&self, lambda: f64) -> Self {
        Self::assemble(self.mass, lambda, self.scenario, self.unit_gamma.clone())
    }

    /// Portal coupling `λ_S`.
    pub fn coupling(&self) -> f64 {
        self.lambda
    }

    /// Total Higgs width including `h → SS` \[GeV\].
    pub fn higgs_total_width(&self) -> f64 {
        self.higgs_total_width
    }
}

impl DarkMatterModel for ScalarSinglet {
    fn mass(&self) -> f64 {
        self.mass
    }
    fn dof(&self) -> f64 {
        1.0
    }
    fn sigma_v(&self, s: f64) -> f64 {
        let rs = s.sqrt();
        let mh2 = M_HIGGS * M_HIGGS;
        let d = (s - mh2).powi(2) + mh2 * self.higgs_total_width.powi(2);
        2.0 * self.lambda.powi(2) * V_HIGGS * V_HIGGS / rs * higgs_width_sm(rs) / d
    }
    fn momentum_transfer_rate(&self, t: f64) -> f64 {
        self.lambda.powi(2) * self.unit_gamma.eval(t)
    }
    fn resonances(&self) -> Vec<Resonance> {
        vec![Resonance {
            mass: M_HIGGS,
            width: self.higgs_total_width,
        }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The generic t-integral reproduces the closed form of Eq. (29).
    #[test]
    fn gamma_matches_closed_form() {
        let (m, t) = (60.0, 3.0);
        let f = SmFermion::Bottom;
        let (mf, nc) = (f.mass(), f.colors());
        let generic = Scatterer::fermion_pair(mf, move |_s, tt: f64| {
            2.0 * nc * mf * mf * (4.0 * mf * mf - tt) / (tt - M_HIGGS * M_HIGGS).powi(2)
        })
        .momentum_transfer_rate(m, 1.0, t);
        // closed form of the t-integral: mh² A(r) + 4 mf² B(r), r = 4k_cm²/mh²
        let gl = crate::math::gauss_legendre(24);
        let kmax = ((mf + 60.0 * t).powi(2) - mf * mf).sqrt();
        let edges: Vec<f64> = (0..=6).map(|p| kmax * (p as f64 / 6.0).powi(2)).collect();
        let closed = gl.integrate_panels(&edges, |k| {
            let w = (k * k + mf * mf).sqrt();
            let gg = 0.25 / (0.5 * w / t).cosh().powi(2);
            let kcm2 = k * k * m * m / (m * m + 2.0 * w * m + mf * mf);
            let r = 4.0 * kcm2 / (M_HIGGS * M_HIGGS);
            let l = r.ln_1p();
            let (a, b) = (r * (2.0 + r) / (1.0 + r) - 2.0 * l, l - r / (1.0 + r));
            (k / w) * gg * (M_HIGGS * M_HIGGS * a + 4.0 * mf * mf * b)
        }) * nc
            * mf
            * mf
            / (96.0 * PI.powi(3) * m.powi(3) * t);
        assert!(
            (generic / closed - 1.0).abs() < 1e-6,
            "{generic} vs {closed}"
        );
    }
}
