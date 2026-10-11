//! The interface between a particle-physics model and the Boltzmann solvers.
//!
//! The solvers need exactly two model-dependent functions (see Sec. II of
//! arXiv:1706.07433):
//!
//! * the annihilation cross section times Møller velocity in the CM frame,
//!   `σv(s)`, which fixes both the collision integral of the full
//!   phase-space equation and the thermal averages of the moment equations;
//! * the momentum-transfer rate `γ(T)` of elastic scattering on the plasma,
//!   which sets the strength of the Fokker–Planck operator.
//!
//! Implement [`DarkMatterModel`] for your own type, or wrap two closures with
//! [`ModelFn`]. Helpers for computing `γ(T)` from a scattering matrix element
//! live in [`crate::scattering`].

/// Whether dark matter is its own antiparticle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ParticleKind {
    /// Real scalar, Majorana fermion, ...: one species, `Ω ∝ m Y`.
    #[default]
    SelfConjugate,
    /// Complex scalar, Dirac fermion, ... with no asymmetry: `χ` and `χ̄` are
    /// evolved as one distribution (`f_χ = f_χ̄`); `Ω ∝ 2 m Y`.
    Distinct,
}

impl ParticleKind {
    /// Number of species that contribute to the relic density.
    pub fn species(self) -> f64 {
        match self {
            Self::SelfConjugate => 1.0,
            Self::Distinct => 2.0,
        }
    }
}

/// An s-channel resonance in `χχ̄ → SM`. Only used to place quadrature and
/// interpolation nodes; it does not change the physics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Resonance {
    /// Mass of the s-channel state \[GeV\].
    pub mass: f64,
    /// Total width \[GeV\] (as used in the Breit–Wigner propagator).
    pub width: f64,
}

/// Model-dependent input of the Boltzmann equations.
///
/// Conventions (identical to arXiv:1706.07433):
///
/// * `σv(s)` is `σ(χχ̄→SM) · v_cms` with the Møller velocity in the CM frame,
///   `v_cms = 2 sqrt(1 − 4m²/s)`, averaged over the internal degrees of
///   freedom of the initial state and summed over final states. For a
///   distinct (Dirac-like) particle it is the `χχ̄` cross section.
/// * `γ(T)` is the momentum-transfer rate of the Fokker–Planck operator,
///   `C_el = (E/2) γ ∇_p·[E T ∇_p + p] f`.
///
/// All energies in GeV.
pub trait DarkMatterModel {
    /// Dark-matter mass `m_χ` \[GeV\].
    fn mass(&self) -> f64;

    /// Internal degrees of freedom `g_χ` of one particle (spin, colour, ...).
    fn dof(&self) -> f64;

    /// Self-conjugate or distinct particle–antiparticle pair.
    fn kind(&self) -> ParticleKind {
        ParticleKind::SelfConjugate
    }

    /// `σ v_cms` \[GeV⁻²\] at centre-of-mass energy squared `s > 4 m_χ²`.
    fn sigma_v(&self, s: f64) -> f64;

    /// Momentum-transfer rate `γ(T)` \[GeV\] at photon temperature `T` \[GeV\].
    fn momentum_transfer_rate(&self, temperature: f64) -> f64;

    /// Narrow s-channel resonances (optional, improves accuracy).
    fn resonances(&self) -> Vec<Resonance> {
        Vec::new()
    }

    /// Values of `sqrt(s)` \[GeV\] where `σv` has a kink, e.g. thresholds of new
    /// final states (optional).
    fn thresholds(&self) -> Vec<f64> {
        Vec::new()
    }
}

impl<M: DarkMatterModel + ?Sized> DarkMatterModel for &M {
    fn mass(&self) -> f64 {
        (**self).mass()
    }
    fn dof(&self) -> f64 {
        (**self).dof()
    }
    fn kind(&self) -> ParticleKind {
        (**self).kind()
    }
    fn sigma_v(&self, s: f64) -> f64 {
        (**self).sigma_v(s)
    }
    fn momentum_transfer_rate(&self, t: f64) -> f64 {
        (**self).momentum_transfer_rate(t)
    }
    fn resonances(&self) -> Vec<Resonance> {
        (**self).resonances()
    }
    fn thresholds(&self) -> Vec<f64> {
        (**self).thresholds()
    }
}

impl<M: DarkMatterModel + ?Sized> DarkMatterModel for Box<M> {
    fn mass(&self) -> f64 {
        (**self).mass()
    }
    fn dof(&self) -> f64 {
        (**self).dof()
    }
    fn kind(&self) -> ParticleKind {
        (**self).kind()
    }
    fn sigma_v(&self, s: f64) -> f64 {
        (**self).sigma_v(s)
    }
    fn momentum_transfer_rate(&self, t: f64) -> f64 {
        (**self).momentum_transfer_rate(t)
    }
    fn resonances(&self) -> Vec<Resonance> {
        (**self).resonances()
    }
    fn thresholds(&self) -> Vec<f64> {
        (**self).thresholds()
    }
}

/// Møller velocity in the CM frame, `v_cms = 2 sqrt(1 − 4m²/s)`.
pub fn cms_velocity(s: f64, mass: f64) -> f64 {
    2.0 * (1.0 - 4.0 * mass * mass / s).max(0.0).sqrt()
}

/// Lab-frame (rest frame of one particle) relative velocity,
/// `v_lab = sqrt(s(s − 4m²)) / (s − 2m²)`.
pub fn lab_velocity(s: f64, mass: f64) -> f64 {
    let m2 = mass * mass;
    (s * (s - 4.0 * m2)).max(0.0).sqrt() / (s - 2.0 * m2)
}

/// A model assembled from closures, for quick experiments.
///
/// ```
/// use dmphase::prelude::*;
/// // p-wave annihilation σv = b v_cms²/4 with a constant momentum-transfer rate
/// let m = 100.0;
/// let model = ModelFn::new(m, 1.0, move |s| 1e-8 * (1.0 - 4.0 * m * m / s), |t| 1e-12 * t.powi(6));
/// assert_eq!(model.mass(), 100.0);
/// ```
#[derive(Clone)]
pub struct ModelFn<S, G> {
    mass: f64,
    dof: f64,
    kind: ParticleKind,
    sigma_v: S,
    gamma: G,
    resonances: Vec<Resonance>,
    thresholds: Vec<f64>,
}

impl<S, G> ModelFn<S, G>
where
    S: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
{
    /// `sigma_v(s)` \[GeV⁻²\] and `gamma(T)` \[GeV\] as closures.
    pub fn new(mass: f64, dof: f64, sigma_v: S, gamma: G) -> Self {
        Self {
            mass,
            dof,
            kind: ParticleKind::SelfConjugate,
            sigma_v,
            gamma,
            resonances: vec![],
            thresholds: vec![],
        }
    }

    /// Set the particle kind (default: self-conjugate).
    pub fn with_kind(mut self, kind: ParticleKind) -> Self {
        self.kind = kind;
        self
    }

    /// Declare a narrow resonance.
    pub fn with_resonance(mut self, mass: f64, width: f64) -> Self {
        self.resonances.push(Resonance { mass, width });
        self
    }

    /// Declare a threshold in `sqrt(s)`.
    pub fn with_threshold(mut self, sqrt_s: f64) -> Self {
        self.thresholds.push(sqrt_s);
        self
    }
}

impl<S, G> DarkMatterModel for ModelFn<S, G>
where
    S: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
{
    fn mass(&self) -> f64 {
        self.mass
    }
    fn dof(&self) -> f64 {
        self.dof
    }
    fn kind(&self) -> ParticleKind {
        self.kind
    }
    fn sigma_v(&self, s: f64) -> f64 {
        (self.sigma_v)(s)
    }
    fn momentum_transfer_rate(&self, t: f64) -> f64 {
        (self.gamma)(t)
    }
    fn resonances(&self) -> Vec<Resonance> {
        self.resonances.clone()
    }
    fn thresholds(&self) -> Vec<f64> {
        self.thresholds.clone()
    }
}

/// `F'(s̃) = s̃ σv_cms(4m² s̃) = (2s̃ − 1) σv_lab`, with `s̃ = s/(4m²)`.
/// This is the combination that enters every collision integral.
pub(crate) fn fprime<M: DarkMatterModel + ?Sized>(model: &M, st: f64) -> f64 {
    let m = model.mass();
    st * model.sigma_v(4.0 * m * m * st)
}

/// Resonances as `(s̃_R, half-width in s̃)` pairs, keeping only physical ones.
pub(crate) fn resonances_st<M: DarkMatterModel + ?Sized>(model: &M) -> Vec<(f64, f64)> {
    let m2 = 4.0 * model.mass().powi(2);
    model
        .resonances()
        .into_iter()
        .filter(|r| r.mass > 0.0 && r.width > 0.0)
        .map(|r| (r.mass * r.mass / m2, r.mass * r.width / m2))
        .collect()
}

/// Thresholds as `s̃` values above 1.
pub(crate) fn thresholds_st<M: DarkMatterModel + ?Sized>(model: &M) -> Vec<f64> {
    let m2 = 4.0 * model.mass().powi(2);
    model
        .thresholds()
        .into_iter()
        .map(|r| r * r / m2)
        .filter(|&s| s > 1.0)
        .collect()
}
