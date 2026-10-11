//! # dmphase
//!
//! Dark-matter freeze-out beyond the assumption of kinetic equilibrium.
//!
//! The crate solves, for a user-supplied particle model, the Boltzmann
//! equation for the dark-matter phase-space density `f_χ(t, p)` in an
//! expanding, radiation-dominated universe, following Binder, Bringmann,
//! Gustafsson & Hryczuk, *Early kinetic decoupling of dark matter: when the
//! standard way of calculating the thermal relic density fails*,
//! [Phys. Rev. D 96, 115010 (2017)](https://arxiv.org/abs/1706.07433).
//!
//! In the variables `x = m_χ/T` and `q = p/T` the equation reads
//! (Sec. II.C of the paper)
//!
//! ```text
//! x ∂ₓf = g̃ q ∂_q f + (g_χ/(2π²)) m³/(H̃ x) ∫ dq̃ q̃² ⟨vσ⟩^θ (f_eq f̃_eq − f f̃)
//!         + γ/(2H̃) (1/q²) ∂_q [q² (x_q ∂_q + q) f]/x ,       x_q = sqrt(x² + q²)
//! ```
//!
//! with a semi-relativistic Fokker–Planck operator for elastic scattering.
//! Three levels of approximation are offered through [`Method`]:
//!
//! * [`Method::PhaseSpace`] — the full equation for `f(x, q)`; yields the
//!   phase-space density, `Y(x)`, `y(x)` and `Ω h²`;
//! * [`Method::Coupled`] — coupled moment equations for the abundance
//!   `Y = n/s` and the temperature parameter `y = m T_χ / s^{2/3}`;
//! * [`Method::NumberDensity`] — the standard Gondolo–Gelmini equation for
//!   `Y(x)` alone.
//!
//! ## Supplying a model
//!
//! Everything model dependent enters through two functions, collected in
//! the [`DarkMatterModel`] trait: the annihilation cross section `σv_cms(s)`
//! and the momentum-transfer rate `γ(T)`. [`scattering`] computes `γ(T)`
//! from an elastic matrix element `|M|²(s, t)`. Ready-made models are in
//! [`models`].
//!
//! ```no_run
//! use dmphase::prelude::*;
//!
//! # fn main() -> dmphase::Result<()> {
//! // 100 MeV Dirac fermion with a vector mediator near the s-channel pole
//! let model = SimplifiedModel::builder(Mediator::Vector)
//!     .dm_mass(0.1)
//!     .mediator_mass(0.205)
//!     .dm_coupling(1e-3)
//!     .sm_coupling(SmFermion::Electron, 1e-3)
//!     .build()?;
//! let nbe = relic_density(&model, Method::NumberDensity)?;
//! let fbe = Solver::new(&model).solve(Method::phase_space())?;
//! println!("Ωh²: nBE {nbe:.3}, fBE {:.3}", fbe.omega_h2());
//! # Ok(()) }
//! ```
//!
//! All quantities are in natural units with energies in GeV.

#![warn(missing_docs)]

pub mod constants;
pub mod cosmology;
pub mod error;
mod kernel;
pub mod math;
pub mod model;
pub mod models;
pub mod ode;
pub mod relic;
pub mod scattering;
pub mod solver;
pub mod thermal;

pub use cosmology::{Cosmology, Plasma};
pub use error::{Error, Result};
pub use model::{DarkMatterModel, ModelFn, ParticleKind, Resonance};
pub use relic::{find_coupling, CouplingSearch, CouplingSolution};
pub use solver::{
    relic_density, Decoupling, Method, PhaseSpaceGrid, ScatteringTerm, Snapshot, Solution, Solver,
    TracePoint,
};

/// Convenient glob import: `use dmphase::prelude::*;`.
///
/// The crate's `Result` alias is deliberately *not* re-exported here so that
/// it does not shadow `std::result::Result`; use `dmphase::Result`.
pub mod prelude {
    pub use crate::cosmology::Cosmology;
    pub use crate::error::Error;
    pub use crate::model::{DarkMatterModel, ModelFn, ParticleKind, Resonance};
    pub use crate::models::{Mediator, QuarkScattering, ScalarSinglet, SimplifiedModel, SmFermion};
    pub use crate::relic::{find_coupling, CouplingSearch};
    pub use crate::scattering::{Scatterer, Statistics};
    pub use crate::solver::{
        relic_density, Method, PhaseSpaceGrid, ScatteringTerm, Solution, Solver,
    };
}
