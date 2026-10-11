//! Dirac-fermion dark matter `χ` with an s-channel mediator coupled to SM
//! fermions `f` (the standard "simplified models"):
//!
//! | [`Mediator`]    | interaction                                      |
//! |-----------------|--------------------------------------------------|
//! | `Scalar`        | `−φ (g_χ χ̄χ + g_f f̄f)`                         |
//! | `Pseudoscalar`  | `−i a (g_χ χ̄γ⁵χ + g_f f̄γ⁵f)`                   |
//! | `Vector`        | `V_μ (g_χ χ̄γ^μχ + g_f f̄γ^μf)`                  |
//! | `AxialVector`   | `A_μ (g_χ χ̄γ^μγ⁵χ + g_f f̄γ^μγ⁵f)`              |
//!
//! Annihilation `χχ̄ → f f̄` (spin averaged, angle integrated, `D = s − M² + iMΓ`):
//!
//! ```text
//! σv_cms = N_c g_χ² g_f² β_f /(8π s) · [ A(s)/|D|² + C ]
//!   A_S = (s − 4m²)(s − 4m_f²)             (p-wave)
//!   A_P = s²                               (s-wave)
//!   A_V = (4/3)(s + 2m²)(s + 2m_f²)        (s-wave)
//!   A_A = (4/3)(s − 4m²)(s − 4m_f²)        (p-wave, spin-1 part of the exchange)
//!   C_A = 16 m² m_f² / M⁴                  (helicity-suppressed s-wave, spin-0 part)
//! ```
//!
//! The contact term `C` is nonzero only for the axial-vector, whose current is
//! not conserved. The spin-0 projection of the Proca propagator,
//! `(−g^{μν} + q^μ q^ν/M²)/(s − M²) = −(g^{μν} − q^μ q^ν/s)/(s − M²) + q^μ q^ν/(s M²)`,
//! has no pole, so the Breit–Wigner width enters only the transverse part,
//! `Π^{μν} = −(g^{μν} − q^μ q^ν/s)/D + q^μ q^ν/(s M²)`. The two parts do not
//! interfere. This is the tree-level prescription and not a loop-consistent
//! resummation of the width.
//!
//! Elastic scattering `χ f → χ f` (summed over all spins, `Σ = m² + m_f²`,
//! `u = 2Σ − s − t`):
//!
//! ```text
//! |M|²_S = 4 N_c g_χ² g_f² (4m² − t)(4m_f² − t) / (t − M²)²
//! |M|²_P = 4 N_c g_χ² g_f² t² / (t − M²)²                         (momentum suppressed)
//! |M|²_V = 8 N_c g_χ² g_f² [(s−Σ)² + (u−Σ)² + 2tΣ] / (t − M²)²
//! |M|²_A = 8 N_c g_χ² g_f² [(s−Σ)² + (u−Σ)² − 2tΣ + 8m²m_f²(1 + (1 − t/M²)²)] / (t − M²)²
//! ```
//!
//! Mediator partial widths into a fermion pair (`β = sqrt(1 − 4m_f²/M²)`):
//! `Γ_S = N_c g² M β³/(8π)`, `Γ_P = N_c g² M β/(8π)`,
//! `Γ_V = N_c g² M β (1 + 2m_f²/M²)/(12π)`, `Γ_A = N_c g² M β³/(12π)`.
//!
//! These expressions were checked against explicit Dirac-trace numerics
//! (`scripts/check_matrix_elements.py`). Quarks are treated as free partons
//! (scattering only above `T_QCD`); below ~2 GeV this is not a reliable
//! description of hadronic final states.

use std::f64::consts::PI;

use super::SmFermion;
use crate::constants::T_QCD;
use crate::error::{Error, Result};
use crate::model::{DarkMatterModel, ParticleKind, Resonance};
use crate::scattering::{RateTable, Scatterer};

/// Lorentz structure of the mediator couplings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mediator {
    /// CP-even scalar.
    Scalar,
    /// CP-odd scalar.
    Pseudoscalar,
    /// Vector boson with vector couplings.
    Vector,
    /// Vector boson with axial-vector couplings.
    AxialVector,
}

impl Mediator {
    /// Short name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Pseudoscalar => "pseudoscalar",
            Self::Vector => "vector",
            Self::AxialVector => "axial-vector",
        }
    }

    /// Spin-averaged, angle-integrated `χχ̄ → f f̄` numerators for unit
    /// couplings and colors, as the pair `(A, C)` of the module docs: `A` is
    /// divided by `|D|²` and `C` is a width-independent contact term.
    fn annihilation_numerators(self, s: f64, m: f64, mf: f64, big_m: f64) -> (f64, f64) {
        let (m2, mf2) = (m * m, mf * mf);
        match self {
            Self::Scalar => ((s - 4.0 * m2) * (s - 4.0 * mf2), 0.0),
            Self::Pseudoscalar => (s * s, 0.0),
            Self::Vector => (4.0 / 3.0 * (s + 2.0 * m2) * (s + 2.0 * mf2), 0.0),
            Self::AxialVector => (
                4.0 / 3.0 * (s - 4.0 * m2) * (s - 4.0 * mf2),
                16.0 * m2 * mf2 / big_m.powi(4),
            ),
        }
    }

    /// Spin-summed `χ f → χ f` matrix element times `(t − M²)²`, unit
    /// couplings and colors.
    fn scattering_numerator(self, s: f64, t: f64, m: f64, mf: f64, big_m: f64) -> f64 {
        let (m2, mf2) = (m * m, mf * mf);
        let sig = m2 + mf2;
        let u = 2.0 * sig - s - t;
        match self {
            Self::Scalar => 4.0 * (4.0 * m2 - t) * (4.0 * mf2 - t),
            Self::Pseudoscalar => 4.0 * t * t,
            Self::Vector => 8.0 * ((s - sig).powi(2) + (u - sig).powi(2) + 2.0 * t * sig),
            Self::AxialVector => {
                8.0 * ((s - sig).powi(2) + (u - sig).powi(2) - 2.0 * t * sig
                    + 8.0 * m2 * mf2 * (1.0 + (1.0 - t / (big_m * big_m)).powi(2)))
            }
        }
    }

    /// Partial width into a fermion pair of mass `mf` for unit coupling
    /// and color.
    pub fn partial_width(self, big_m: f64, mf: f64) -> f64 {
        if big_m <= 2.0 * mf {
            return 0.0;
        }
        let b = (1.0 - 4.0 * mf * mf / (big_m * big_m)).sqrt();
        match self {
            Self::Scalar => big_m * b.powi(3) / (8.0 * PI),
            Self::Pseudoscalar => big_m * b / (8.0 * PI),
            Self::Vector => big_m * b * (1.0 + 2.0 * mf * mf / (big_m * big_m)) / (12.0 * PI),
            Self::AxialVector => big_m * b.powi(3) / (12.0 * PI),
        }
    }
}

/// Dirac dark matter with an s-channel mediator; see the module docs.
///
/// Build with [`SimplifiedModel::builder`].
#[derive(Clone, Debug)]
pub struct SimplifiedModel {
    mediator: Mediator,
    m_dm: f64,
    m_med: f64,
    g_dm: f64,
    sm: Vec<(SmFermion, f64)>,
    extra_width: f64,
    width: f64,
    gamma: RateTable,
}

/// Builder for [`SimplifiedModel`].
#[derive(Clone, Debug)]
pub struct SimplifiedModelBuilder {
    mediator: Mediator,
    m_dm: Option<f64>,
    m_med: Option<f64>,
    g_dm: f64,
    sm: Vec<(SmFermion, f64)>,
    extra_width: f64,
}

impl SimplifiedModelBuilder {
    /// Dark-matter mass \[GeV\].
    pub fn dm_mass(mut self, m: f64) -> Self {
        self.m_dm = Some(m);
        self
    }
    /// Mediator mass \[GeV\].
    pub fn mediator_mass(mut self, m: f64) -> Self {
        self.m_med = Some(m);
        self
    }
    /// Mediator coupling to dark matter, `g_χ`.
    pub fn dm_coupling(mut self, g: f64) -> Self {
        self.g_dm = g;
        self
    }
    /// Mediator coupling to one SM fermion (replaces an earlier value).
    pub fn sm_coupling(mut self, f: SmFermion, g: f64) -> Self {
        self.sm.retain(|(h, _)| *h != f);
        self.sm.push((f, g));
        self
    }
    /// The same coupling to e, μ and τ.
    pub fn lepton_coupling(self, g: f64) -> Self {
        SmFermion::LEPTONS
            .into_iter()
            .fold(self, |b, f| b.sm_coupling(f, g))
    }
    /// Additional mediator width \[GeV\] from channels not modelled here.
    pub fn extra_width(mut self, w: f64) -> Self {
        self.extra_width = w;
        self
    }

    /// Validate and assemble (tabulates `γ(T)`, ~0.1 s).
    pub fn build(self) -> Result<SimplifiedModel> {
        let m = self
            .m_dm
            .ok_or_else(|| Error::InvalidInput("dark-matter mass not set".into()))?;
        let big_m = self
            .m_med
            .ok_or_else(|| Error::InvalidInput("mediator mass not set".into()))?;
        if !(m > 0.0 && big_m > 0.0) || self.extra_width < 0.0 {
            return Err(Error::InvalidInput(
                "masses must be positive and extra width non-negative".into(),
            ));
        }
        if self.sm.is_empty() {
            return Err(Error::InvalidInput("no SM couplings specified".into()));
        }
        let med = self.mediator;
        let width = self.extra_width
            + self.g_dm.powi(2) * med.partial_width(big_m, m)
            + self
                .sm
                .iter()
                .map(|&(f, g)| f.colors() * g * g * med.partial_width(big_m, f.mass()))
                .sum::<f64>();
        let sm = self.sm;
        let g_dm = self.g_dm;
        let has_quarks = sm.iter().any(|(f, _)| f.is_quark());
        let breaks: &[f64] = if has_quarks { &[T_QCD] } else { &[] };
        let gamma = RateTable::new(m * 1e-9, 2.0 * m, 40, breaks, |t| {
            sm.iter()
                .filter(|(f, _)| !f.is_quark() || t > T_QCD)
                .map(|&(f, g)| {
                    let (mf, c) = (f.mass(), f.colors() * (g * g_dm).powi(2));
                    let msq = move |s: f64, tt: f64| {
                        c * med.scattering_numerator(s, tt, m, mf, big_m)
                            / (tt - big_m * big_m).powi(2)
                    };
                    Scatterer::fermion_pair(mf, msq).momentum_transfer_rate(m, 2.0, t)
                })
                .sum()
        })?;
        let extra_width = self.extra_width;
        Ok(SimplifiedModel {
            mediator: med,
            m_dm: m,
            m_med: big_m,
            g_dm,
            sm,
            extra_width,
            width,
            gamma,
        })
    }
}

impl SimplifiedModel {
    /// Start building a model with the given mediator type.
    pub fn builder(mediator: Mediator) -> SimplifiedModelBuilder {
        SimplifiedModelBuilder {
            mediator,
            m_dm: None,
            m_med: None,
            g_dm: 0.0,
            sm: Vec::new(),
            extra_width: 0.0,
        }
    }

    /// Builder pre-filled with this model's parameters, e.g. to change a
    /// coupling.
    pub fn to_builder(&self) -> SimplifiedModelBuilder {
        SimplifiedModelBuilder {
            mediator: self.mediator,
            m_dm: Some(self.m_dm),
            m_med: Some(self.m_med),
            g_dm: self.g_dm,
            sm: self.sm.clone(),
            extra_width: self.extra_width,
        }
    }

    /// Mediator type.
    pub fn mediator(&self) -> Mediator {
        self.mediator
    }
    /// Mediator mass \[GeV\].
    pub fn mediator_mass(&self) -> f64 {
        self.m_med
    }
    /// Total mediator width \[GeV\].
    pub fn mediator_width(&self) -> f64 {
        self.width
    }
    /// `g_χ`.
    pub fn dm_coupling(&self) -> f64 {
        self.g_dm
    }
    /// SM couplings.
    pub fn sm_couplings(&self) -> &[(SmFermion, f64)] {
        &self.sm
    }
}

impl DarkMatterModel for SimplifiedModel {
    fn mass(&self) -> f64 {
        self.m_dm
    }
    fn dof(&self) -> f64 {
        2.0
    }
    fn kind(&self) -> ParticleKind {
        ParticleKind::Distinct
    }
    fn sigma_v(&self, s: f64) -> f64 {
        let (m, big_m) = (self.m_dm, self.m_med);
        let d2 = (s - big_m * big_m).powi(2) + (big_m * self.width).powi(2);
        let pref = (self.g_dm * self.g_dm) / (8.0 * PI * s);
        self.sm
            .iter()
            .filter(|(f, _)| s > 4.0 * f.mass().powi(2))
            .map(|&(f, g)| {
                let mf = f.mass();
                let beta = (1.0 - 4.0 * mf * mf / s).sqrt();
                let (resonant, contact) = self.mediator.annihilation_numerators(s, m, mf, big_m);
                f.colors() * g * g * beta * (resonant / d2 + contact)
            })
            .sum::<f64>()
            * pref
    }
    fn momentum_transfer_rate(&self, t: f64) -> f64 {
        self.gamma.eval(t)
    }
    fn resonances(&self) -> Vec<Resonance> {
        vec![Resonance {
            mass: self.m_med,
            width: self.width,
        }]
    }
    fn thresholds(&self) -> Vec<f64> {
        self.sm
            .iter()
            .map(|(f, _)| 2.0 * f.mass())
            .filter(|&r| r > 2.0 * self.m_dm)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(med: Mediator) -> SimplifiedModel {
        SimplifiedModel::builder(med)
            .dm_mass(0.1)
            .mediator_mass(0.5)
            .dm_coupling(0.3)
            .sm_coupling(SmFermion::Electron, 0.01)
            .sm_coupling(SmFermion::Muon, 0.01)
            .build()
            .unwrap()
    }

    fn axial_model(m: f64, big_m: f64, gx: f64, gf: f64, extra_width: f64) -> SimplifiedModel {
        SimplifiedModel::builder(Mediator::AxialVector)
            .dm_mass(m)
            .mediator_mass(big_m)
            .dm_coupling(gx)
            .sm_coupling(SmFermion::Electron, gf)
            .sm_coupling(SmFermion::Muon, gf)
            .extra_width(extra_width)
            .build()
            .unwrap()
    }

    /// Non-relativistic limits: σv(v→0) for the s-wave channels.
    #[test]
    fn s_wave_limits() {
        let (m, big_m, gx, gf) = (0.1_f64, 0.5_f64, 0.3_f64, 0.01_f64);
        let s0 = 4.0 * m * m * (1.0 + 1e-10);
        let mods = [Mediator::Pseudoscalar, Mediator::Vector];
        for med in mods {
            let mdl = model(med);
            let d2 = (4.0 * m * m - big_m * big_m).powi(2) + (big_m * mdl.mediator_width()).powi(2);
            let want: f64 = [SmFermion::Electron, SmFermion::Muon]
                .iter()
                .filter(|f| f.mass() < m)
                .map(|f| {
                    let mf = f.mass();
                    let bf = (1.0 - mf * mf / (m * m)).sqrt();
                    match med {
                        Mediator::Pseudoscalar => gx * gx * gf * gf * m * m * bf / (2.0 * PI * d2),
                        _ => gx * gx * gf * gf * (2.0 * m * m + mf * mf) * bf / (2.0 * PI * d2),
                    }
                })
                .sum();
            assert!((mdl.sigma_v(s0) / want - 1.0).abs() < 1e-6, "{med:?}");
        }
        // p-wave: vanishes at threshold
        assert!(model(Mediator::Scalar).sigma_v(s0) < 1e-9 * model(Mediator::Vector).sigma_v(s0));
    }

    /// Axial-vector exchange at `M = 2m`, with both leptons open: at `s = M²`
    /// the transverse part vanishes and the cross section is the contact term.
    #[test]
    fn axial_contact_at_resonant_threshold() {
        let (m, gx, gf) = (0.5_f64, 0.3_f64, 0.01_f64);
        let big_m = 2.0 * m;
        let mdl = axial_model(m, big_m, gx, gf, 0.0);
        let s = big_m * big_m;
        let want: f64 = [SmFermion::Electron, SmFermion::Muon]
            .iter()
            .map(|f| {
                let mf = f.mass();
                let beta = (1.0 - 4.0 * mf * mf / s).sqrt();
                f.colors() * gx * gx * gf * gf * beta * 16.0 * m * m * mf * mf
                    / (8.0 * PI * s * big_m.powi(4))
            })
            .sum();
        let got = mdl.sigma_v(s);
        assert!(
            got > 0.0 && (got / want - 1.0).abs() < 1e-12,
            "{got} vs {want}"
        );
    }

    /// The contact term does not depend on the mediator width.
    #[test]
    fn axial_contact_is_width_independent() {
        let (m, big_m) = (0.5, 1.0);
        let s = big_m * big_m;
        let base = axial_model(m, big_m, 0.3, 0.01, 0.0).sigma_v(s);
        for w in [1e-3, 0.1] {
            let got = axial_model(m, big_m, 0.3, 0.01, w).sigma_v(s);
            assert!((got / base - 1.0).abs() < 1e-12, "extra width {w}");
        }
    }

    /// For a massless final state the axial-vector exchange is purely
    /// transverse: the contact term vanishes and `A` is the vector-like
    /// `(4/3) s (s − 4m²)`.
    #[test]
    fn axial_contact_vanishes_for_massless_fermion() {
        let (s, m, big_m) = (3.0, 0.5, 1.0);
        let (resonant, contact) = Mediator::AxialVector.annihilation_numerators(s, m, 0.0, big_m);
        assert_eq!(contact, 0.0);
        assert!((resonant / (4.0 / 3.0 * s * (s - 4.0 * m * m)) - 1.0).abs() < 1e-15);
    }

    /// For `Γ → 0` away from the pole the split propagator reduces to the
    /// Proca propagator, `[A + 16 m² m_f² (1 − s/M²)²]/(s − M²)²`.
    #[test]
    fn axial_zero_width_limit_is_proca() {
        let (m, big_m, g, s) = (0.5_f64, 1.0_f64, 1e-4_f64, 3.0_f64);
        let mdl = axial_model(m, big_m, g, g, 0.0);
        let want: f64 = [SmFermion::Electron, SmFermion::Muon]
            .iter()
            .map(|f| {
                let (m2, mf2) = (m * m, f.mass().powi(2));
                let beta = (1.0 - 4.0 * mf2 / s).sqrt();
                let numerator = 4.0 / 3.0 * (s - 4.0 * m2) * (s - 4.0 * mf2)
                    + 16.0 * m2 * mf2 * (1.0 - s / (big_m * big_m)).powi(2);
                g.powi(4) * beta * numerator / (8.0 * PI * s * (s - big_m * big_m).powi(2))
            })
            .sum();
        assert!((mdl.sigma_v(s) / want - 1.0).abs() < 1e-9);
    }

    /// Pseudoscalar scattering is suppressed by ~|t|/(4m²) ∝ (T/m)² relative
    /// to scalar exchange (up to O(T/m) corrections from k_cm² = k²m²/s).
    #[test]
    fn pseudoscalar_scattering_suppressed() {
        let (s, p) = (model(Mediator::Scalar), model(Mediator::Pseudoscalar));
        let ratio = |t: f64| p.momentum_transfer_rate(t) / s.momentum_transfer_rate(t);
        let (r1, r2) = (ratio(0.005), ratio(0.0025));
        assert!(r1 < 0.1, "ratio {r1}");
        assert!(r1 / r2 > 2.5 && r1 / r2 < 4.5, "scaling {}", r1 / r2);
    }
}
