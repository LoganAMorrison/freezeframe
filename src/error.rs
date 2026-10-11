//! Error type shared by the whole crate.

use std::fmt;

/// Everything that can go wrong while setting up or integrating a Boltzmann
/// equation.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// A user-supplied parameter is out of its domain.
    InvalidInput(String),
    /// The model returned a non-finite or negative quantity.
    InvalidModel(String),
    /// The adaptive step size fell below the round-off floor.
    StepSizeUnderflow {
        /// Value of `x = m/T` where the integrator gave up.
        x: f64,
    },
    /// The integrator exceeded its step budget.
    TooManySteps {
        /// Value of `x = m/T` where the integrator gave up.
        x: f64,
    },
    /// An iterative search (e.g. for the relic coupling) did not converge.
    NoConvergence(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(m) => write!(f, "invalid input: {m}"),
            Self::InvalidModel(m) => write!(f, "invalid model: {m}"),
            Self::StepSizeUnderflow { x } => write!(f, "ODE step size underflow at x = {x:.6e}"),
            Self::TooManySteps { x } => write!(f, "ODE step budget exhausted at x = {x:.6e}"),
            Self::NoConvergence(m) => write!(f, "no convergence: {m}"),
        }
    }
}

impl std::error::Error for Error {}

/// `Result` alias with [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
