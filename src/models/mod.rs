//! Reference implementations of [`DarkMatterModel`](crate::DarkMatterModel).
//!
//! * [`ScalarSinglet`] — Higgs-portal real scalar, the benchmark of
//!   arXiv:1706.07433 (Sec. III);
//! * [`SimplifiedModel`] — Dirac fermion with an s-channel scalar,
//!   pseudoscalar, vector or axial-vector mediator coupled to SM fermions.

mod scalar_singlet;
mod simplified;
mod sm;

pub use scalar_singlet::{QuarkScattering, ScalarSinglet};
pub use simplified::{Mediator, SimplifiedModel, SimplifiedModelBuilder};
pub use sm::SmFermion;
