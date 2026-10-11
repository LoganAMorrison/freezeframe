//! Fixing a coupling by the observed relic density.

use crate::constants::OMEGA_DM_H2_PLANCK;
use crate::error::{Error, Result};

/// Settings of [`find_coupling`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CouplingSearch {
    /// Target `Ω h²`.
    pub target: f64,
    /// Stop when `|Ω/target − 1| < rel_tol`.
    pub rel_tol: f64,
    /// Maximal number of `Ω` evaluations.
    pub max_evals: usize,
    /// Assumed `d ln Ω / d ln g` for the first step (−2 for `σ ∝ g²`, −4 for
    /// `σ ∝ g⁴`).
    pub initial_slope: f64,
}

impl Default for CouplingSearch {
    fn default() -> Self {
        Self {
            target: OMEGA_DM_H2_PLANCK,
            rel_tol: 1e-3,
            max_evals: 40,
            initial_slope: -2.0,
        }
    }
}

/// Outcome of [`find_coupling`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CouplingSolution {
    /// Coupling that reproduces the target.
    pub coupling: f64,
    /// `Ω h²` at that coupling.
    pub omega_h2: f64,
    /// Number of `Ω` evaluations used.
    pub evaluations: usize,
}

/// Find `g > 0` with `Ω h²(g) = target` by a secant iteration in
/// `(ln g, ln Ω)`, starting from `guess`. `omega_h2` maps a coupling to the
/// relic density, typically by building a model and calling
/// [`Solver::solve`](crate::Solver::solve). `Ω` must decrease monotonically
/// with `g` near the solution.
///
/// ```
/// # use freezeframe::relic::{find_coupling, CouplingSearch};
/// // toy: Ω = 0.12 (g / 0.3)^-2
/// let sol = find_coupling(|g| Ok(0.12 * (g / 0.3_f64).powi(-2)), 1.0, &CouplingSearch::default()).unwrap();
/// assert!((sol.coupling / 0.3 - 1.0).abs() < 1e-3);
/// ```
pub fn find_coupling(
    mut omega_h2: impl FnMut(f64) -> Result<f64>,
    guess: f64,
    search: &CouplingSearch,
) -> Result<CouplingSolution> {
    if !(guess > 0.0 && search.target > 0.0) {
        return Err(Error::InvalidInput(
            "coupling guess and target must be positive".into(),
        ));
    }
    let ln_target = search.target.ln();
    let mut eval = |lg: f64| -> Result<f64> {
        let om = omega_h2(lg.exp())?;
        if om.is_finite() && om > 0.0 {
            Ok(om.ln())
        } else {
            Err(Error::NoConvergence(format!(
                "Ω h² = {om} at coupling {:e}",
                lg.exp()
            )))
        }
    };
    let mut l0 = guess.ln();
    let mut f0 = eval(l0)?;
    let mut evaluations = 1;
    let converged = |f: f64| (f - ln_target).exp_m1().abs() < search.rel_tol;
    if converged(f0) {
        return Ok(CouplingSolution {
            coupling: guess,
            omega_h2: f0.exp(),
            evaluations,
        });
    }
    let mut l1 = l0 + ((ln_target - f0) / search.initial_slope).clamp(-3.0, 3.0);
    while evaluations < search.max_evals {
        let f1 = eval(l1)?;
        evaluations += 1;
        if converged(f1) {
            return Ok(CouplingSolution {
                coupling: l1.exp(),
                omega_h2: f1.exp(),
                evaluations,
            });
        }
        let slope = if (f1 - f0).abs() > 1e-14 && (l1 - l0).abs() > 1e-14 {
            (f1 - f0) / (l1 - l0)
        } else {
            search.initial_slope
        };
        // guard against flat or wrong-sign secants (e.g. across a resonance)
        let slope = if slope < -0.05 {
            slope
        } else {
            search.initial_slope
        };
        let l2 = l1 + ((ln_target - f1) / slope).clamp(-3.0, 3.0);
        (l0, f0, l1) = (l1, f1, l2);
    }
    Err(Error::NoConvergence(format!(
        "relic coupling not found after {} evaluations (last g = {:e})",
        search.max_evals,
        l1.exp()
    )))
}
