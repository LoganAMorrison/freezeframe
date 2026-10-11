//! Thermal history of the Standard-Model plasma: effective degrees of
//! freedom, Hubble rate and entropy density in a radiation-dominated universe.

use std::f64::consts::PI;
use std::sync::OnceLock;

use crate::constants::{HBAR_C_GEV_CM, M_PLANCK, RHO_CRIT_OVER_H2, T_CMB};
use crate::error::{Error, Result};
use crate::math::CubicSpline;

static DREES_TABLE: &str = include_str!("../data/dof_drees_etal.dat");

/// One row of a degrees-of-freedom table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DofRow {
    /// Photon temperature \[GeV\].
    pub temperature: f64,
    /// Entropy degrees of freedom `h_eff`.
    pub heff: f64,
    /// Energy degrees of freedom `g_eff`.
    pub geff: f64,
    /// `g̃ = (1/3) d ln h_eff / d ln T`.
    pub gtilde: f64,
}

/// Plasma quantities at a given temperature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plasma {
    /// Photon temperature `T` \[GeV\].
    pub temperature: f64,
    /// Entropy degrees of freedom `h_eff(T)`.
    pub heff: f64,
    /// Energy degrees of freedom `g_eff(T)`.
    pub geff: f64,
    /// `g̃(T) = (1/3) d ln h_eff / d ln T`.
    pub gtilde: f64,
    /// Hubble rate `H` \[GeV\].
    pub hubble: f64,
    /// `H̃ = H / (1 + g̃)` \[GeV\], the rate appearing in `dx/dt = H̃ x`.
    pub htilde: f64,
    /// Entropy density `s = (2π²/45) h_eff T³` \[GeV³\].
    pub entropy_density: f64,
}

/// Effective degrees of freedom of the plasma as smooth functions of `T`.
///
/// [`Cosmology::standard`] is the lattice-QCD based tabulation of Drees,
/// Hajkarim & Schmitz, JCAP 1506 (2015) 025, as distributed with DRAKE v1.0.
/// Custom thermal histories can be supplied with [`Cosmology::from_rows`].
/// Outside the tabulated range all quantities are held constant.
#[derive(Clone, Debug)]
pub struct Cosmology {
    ln_heff: CubicSpline,
    ln_geff: CubicSpline,
    gtilde: CubicSpline,
}

impl Cosmology {
    /// Standard-Model thermal history (Drees et al. 2015).
    pub fn standard() -> &'static Cosmology {
        static STD: OnceLock<Cosmology> = OnceLock::new();
        STD.get_or_init(|| {
            let rows = parse_drees(DREES_TABLE);
            Cosmology::from_rows(&rows).expect("bundled degrees-of-freedom table is valid")
        })
    }

    /// Build from tabulated rows (any order, at least three distinct
    /// temperatures).
    pub fn from_rows(rows: &[DofRow]) -> Result<Self> {
        let mut rows = rows.to_vec();
        if rows
            .iter()
            .any(|r| !(r.temperature > 0.0 && r.heff > 0.0 && r.geff > 0.0))
        {
            return Err(Error::InvalidInput(
                "degrees-of-freedom rows must be positive".into(),
            ));
        }
        rows.sort_by(|a, b| a.temperature.total_cmp(&b.temperature));
        rows.dedup_by(|a, b| a.temperature == b.temperature);
        let lt: Vec<f64> = rows.iter().map(|r| r.temperature.ln()).collect();
        let col = |f: fn(&DofRow) -> f64| rows.iter().map(f).collect::<Vec<_>>();
        Ok(Self {
            ln_heff: CubicSpline::new(lt.clone(), col(|r| r.heff.ln()))?,
            ln_geff: CubicSpline::new(lt.clone(), col(|r| r.geff.ln()))?,
            gtilde: CubicSpline::new(lt, col(|r| r.gtilde))?,
        })
    }

    /// `h_eff(T)`.
    pub fn heff(&self, t: f64) -> f64 {
        self.ln_heff.eval(t.ln()).exp()
    }

    /// `g_eff(T)`.
    pub fn geff(&self, t: f64) -> f64 {
        self.ln_geff.eval(t.ln()).exp()
    }

    /// `g̃(T)`.
    pub fn gtilde(&self, t: f64) -> f64 {
        self.gtilde.eval(t.ln())
    }

    /// Hubble rate `H = sqrt(4π³ g_eff / 45) T² / M_Pl` \[GeV\].
    pub fn hubble(&self, t: f64) -> f64 {
        (4.0 * PI.powi(3) * self.geff(t) / 45.0).sqrt() * t * t / M_PLANCK
    }

    /// Entropy density `s(T)` \[GeV³\].
    pub fn entropy_density(&self, t: f64) -> f64 {
        2.0 * PI * PI / 45.0 * self.heff(t) * t.powi(3)
    }

    /// All plasma quantities at temperature `t` \[GeV\].
    pub fn plasma(&self, t: f64) -> Plasma {
        let (heff, geff, gtilde) = (self.heff(t), self.geff(t), self.gtilde(t));
        let hubble = (4.0 * PI.powi(3) * geff / 45.0).sqrt() * t * t / M_PLANCK;
        Plasma {
            temperature: t,
            heff,
            geff,
            gtilde,
            hubble,
            htilde: hubble / (1.0 + gtilde),
            entropy_density: 2.0 * PI * PI / 45.0 * heff * t.powi(3),
        }
    }

    /// Present-day entropy density \[cm⁻³\], with `h_eff` of this thermal history
    /// at `T_CMB`, so that `Y s_0` is the comoving number density
    /// consistently with how `Y = n/s` was computed.
    pub fn entropy_today(&self) -> f64 {
        self.entropy_density(T_CMB) / HBAR_C_GEV_CM.powi(3)
    }

    /// `Ω h²` of a species of mass `mass` \[GeV\] with asymptotic yield
    /// `yield_` (`n/s`, all particles *and* antiparticles included).
    pub fn omega_h2(&self, mass: f64, yield_: f64) -> f64 {
        mass * yield_ * self.entropy_today() / RHO_CRIT_OVER_H2
    }
}

/// Parse the four-column Drees et al. table `T, h_eff, sqrt(g_eff), sqrt(g*)`,
/// where `1 + g̃ = sqrt(g*) sqrt(g_eff) / h_eff`.
fn parse_drees(text: &str) -> Vec<DofRow> {
    text.lines()
        .filter_map(|line| {
            let v: Vec<f64> = line
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            (v.len() == 4).then(|| DofRow {
                temperature: v[0],
                heff: v[1],
                geff: v[2] * v[2],
                gtilde: v[3] * v[2] / v[1] - 1.0,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_limits() {
        let c = Cosmology::standard();
        assert!((c.heff(1e3) - 106.8).abs() < 0.2);
        assert!((c.heff(1e-6) - 3.94).abs() < 0.02);
        assert!((c.geff(1e-6) - 3.38).abs() < 0.02);
        // g~ peaks at the QCD transition
        assert!(c.gtilde(0.15) > 0.3 && c.gtilde(10.0) < 0.05);
    }

    #[test]
    fn present_entropy_density() {
        // PDG: s0 = 2891.2 cm^-3 with h_eff = 43/11; the Drees table has
        // h_eff = 3.939 at low T.
        let s0 = Cosmology::standard().entropy_today();
        assert!(
            (s0 / (2891.2 * 3.9387 / (43.0 / 11.0)) - 1.0).abs() < 2e-3,
            "{s0}"
        );
    }
}
