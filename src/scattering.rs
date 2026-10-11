//! Momentum-transfer rate `γ(T)` from elastic scattering `χ f → χ f` on a
//! thermal bath (Eqs. 6–7 of arXiv:1706.07433):
//!
//! ```text
//! γ(T) = 1/(48π³ g_χ m_χ³ T) Σ_f ∫ dω g_f(1 ∓ g_f) k⁴ ⟨|M|²⟩_t ,
//! k⁴ ⟨|M|²⟩_t = (1/8) ∫_{−4k_cm²}^0 dt (−t) |M|²(s, t) ,
//! s = m_χ² + 2 ω m_χ + m_f² ,   k_cm² = k² m_χ² / s ,
//! ```
//!
//! where `|M|²` is summed over the internal degrees of freedom of *all* initial
//! and final states (the `1/g_χ` average is in the prefactor) and `g_f` is the
//! Fermi–Dirac or Bose–Einstein occupation of the scatterer. The expression
//! is the leading term in `T/m_χ`.

use std::f64::consts::PI;

use crate::error::Result;
use crate::math::{gauss_legendre, CubicSpline, Extrapolation};

/// Quantum statistics of a scattering partner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Statistics {
    /// Fermi–Dirac, `g(1 − g)` Pauli blocking.
    Fermion,
    /// Bose–Einstein, `g(1 + g)` enhancement.
    Boson,
}

/// A species of the thermal bath that dark matter scatters on.
#[derive(Clone, Debug)]
pub struct Scatterer<F> {
    /// Mass \[GeV\].
    pub mass: f64,
    /// Quantum statistics.
    pub statistics: Statistics,
    /// Number of identical copies, e.g. 2 for a fermion and its antiparticle
    /// with equal matrix elements.
    pub multiplicity: f64,
    /// `|M|²(s, t)`, summed over all internal degrees of freedom of initial
    /// and final states.
    pub msq: F,
}

impl<F: Fn(f64, f64) -> f64> Scatterer<F> {
    /// A fermion together with its antiparticle (`multiplicity = 2`).
    pub fn fermion_pair(mass: f64, msq: F) -> Self {
        Self {
            mass,
            statistics: Statistics::Fermion,
            multiplicity: 2.0,
            msq,
        }
    }

    /// A single bosonic species (`multiplicity = 1`).
    pub fn boson(mass: f64, msq: F) -> Self {
        Self {
            mass,
            statistics: Statistics::Boson,
            multiplicity: 1.0,
            msq,
        }
    }

    /// Contribution of this species to `γ(T)` \[GeV\].
    pub fn momentum_transfer_rate(&self, dm_mass: f64, dm_dof: f64, temperature: f64) -> f64 {
        momentum_transfer_rate(dm_mass, dm_dof, temperature, self)
    }
}

/// `∫_{−4k_cm²}^0 dt (−t) |M|²(s, t)`, with geometric panels towards `t = 0`
/// so that light t-channel mediators are resolved.
pub fn t_moment(s: f64, k_cm2: f64, msq: impl Fn(f64, f64) -> f64) -> f64 {
    let gl = gauss_legendre(8);
    let u_max = 4.0 * k_cm2;
    let mut total = 0.0;
    let mut hi = u_max;
    for _ in 0..14 {
        let lo = 0.25 * hi;
        total += gl.integrate(lo, hi, |u| u * msq(s, -u));
        hi = lo;
    }
    total + gl.integrate(0.0, hi, |u| u * msq(s, -u))
}

/// Contribution of one bath species to the momentum-transfer rate `γ(T)` \[GeV\].
pub fn momentum_transfer_rate<F: Fn(f64, f64) -> f64>(
    dm_mass: f64,
    dm_dof: f64,
    temperature: f64,
    sc: &Scatterer<F>,
) -> f64 {
    let (m, mf, t) = (dm_mass, sc.mass, temperature);
    let gl = gauss_legendre(24);
    let k_max = ((mf + 60.0 * t).powi(2) - mf * mf).sqrt();
    let occupancy = |w: f64| {
        let z = 0.5 * w / t;
        match sc.statistics {
            Statistics::Fermion => 0.25 / z.cosh().powi(2),
            Statistics::Boson => 0.25 / z.sinh().powi(2),
        }
    };
    let integrand = |k: f64| {
        if k <= 0.0 {
            return 0.0;
        }
        let w = (k * k + mf * mf).sqrt();
        let s = m * m + 2.0 * w * m + mf * mf;
        let k_cm2 = k * k * m * m / s;
        (k / w) * occupancy(w) * t_moment(s, k_cm2, &sc.msq)
    };
    const PANELS: usize = 6;
    let edges: Vec<f64> = (0..=PANELS)
        .map(|p| k_max * (p as f64 / PANELS as f64).powi(2))
        .collect();
    let integral = gl.integrate_panels(&edges, integrand);
    sc.multiplicity * integral / (384.0 * PI.powi(3) * dm_dof * m.powi(3) * t)
}

/// Tabulation of a non-negative rate `r(T)` as `ln r` against `ln T` on
/// separate pieces between user-specified discontinuities, e.g. the switch-off
/// of quark scattering at the QCD transition.
#[derive(Clone, Debug)]
pub struct RateTable {
    breaks: Vec<f64>,
    pieces: Vec<Option<CubicSpline>>,
}

impl RateTable {
    /// Tabulate `rate` on `[t_min, t_max]` with `per_decade` nodes per decade.
    /// `discontinuities` are temperatures where `rate` may jump.
    pub fn new(
        t_min: f64,
        t_max: f64,
        per_decade: usize,
        discontinuities: &[f64],
        rate: impl Fn(f64) -> f64,
    ) -> Result<Self> {
        let mut breaks: Vec<f64> = discontinuities
            .iter()
            .copied()
            .filter(|&b| b > t_min && b < t_max)
            .collect();
        breaks.sort_by(f64::total_cmp);
        breaks.dedup();
        let mut edges = vec![t_min];
        edges.extend(&breaks);
        edges.push(t_max);
        let mut pieces = Vec::with_capacity(edges.len() - 1);
        for w in edges.windows(2) {
            let (l0, l1) = (w[0].ln(), w[1].ln());
            let n =
                (((l1 - l0) / std::f64::consts::LN_10 * per_decade as f64).ceil() as usize).max(4);
            let eps = 1e-9 * (l1 - l0);
            let lt: Vec<f64> = (0..=n)
                .map(|i| l0 + (l1 - l0) * i as f64 / n as f64)
                .collect();
            let vals: Vec<f64> = lt
                .iter()
                .enumerate()
                .map(|(i, &l)| {
                    // evaluate strictly inside the piece, never on a jump
                    let l = if i == 0 {
                        l + eps
                    } else if i == n {
                        l - eps
                    } else {
                        l
                    };
                    rate(l.exp())
                })
                .collect();
            pieces.push(if vals.iter().all(|&v| v <= 0.0) {
                None
            } else {
                let ln: Vec<f64> = vals.iter().map(|v| v.max(1e-300).ln()).collect();
                Some(CubicSpline::new(lt, ln)?.with_extrapolation(Extrapolation::Linear))
            });
        }
        Ok(Self { breaks, pieces })
    }

    /// Interpolated rate at temperature `t`.
    pub fn eval(&self, t: f64) -> f64 {
        let k = self.breaks.partition_point(|&b| b <= t);
        match &self.pieces[k] {
            None => 0.0,
            Some(s) => {
                let v = s.eval(t.ln());
                if v < -690.0 {
                    0.0
                } else {
                    v.exp()
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Contact interaction |M|² = c: ⟨|M|²⟩_t = c k_cm⁴ / k⁴ and for a
    /// non-relativistic heavy scatterer the rate is analytic.
    #[test]
    fn t_moment_constant() {
        let v = t_moment(10.0, 0.3, |_, _| 2.0);
        assert!((v - 2.0 * 0.5 * (4.0_f64 * 0.3).powi(2)).abs() < 1e-12);
    }

    /// Massless fermions with constant |M|² = c: to leading order in T/m,
    /// k⁴⟨|M|²⟩_t = c k⁴ and ∫ dk k⁴ g(1−g) = (7π⁴/30) T⁵.
    #[test]
    fn massless_fermion_contact() {
        let (m, t, c) = (1e4, 1.0, 1.0);
        let got = Scatterer::fermion_pair(0.0, |_, _| c).momentum_transfer_rate(m, 1.0, t);
        let lead = 2.0 * 8.0 * c * t.powi(4) * (7.0 * PI.powi(4) / 30.0)
            / (384.0 * PI.powi(3) * m.powi(3));
        assert!((got / lead - 1.0).abs() < 1e-2, "{got} vs {lead}");
    }

    #[test]
    fn rate_table_with_jump() {
        let f = |t: f64| {
            if t > 1.0 {
                3.0 * t.powi(4)
            } else {
                0.5 * t.powi(5)
            }
        };
        let tab = RateTable::new(1e-3, 1e3, 20, &[1.0], f).unwrap();
        for t in [2e-3, 0.3, 0.999, 1.001, 50.0] {
            assert!((tab.eval(t) / f(t) - 1.0).abs() < 1e-6, "T = {t}");
        }
        // power-law extrapolation below the table
        assert!((tab.eval(1e-4) / f(1e-4) - 1.0).abs() < 1e-6);
    }
}
