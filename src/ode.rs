//! L-stable Rosenbrock integrator for stiff systems.
//!
//! This is the second-order method with embedded third-order error estimate of
//! Shampine & Reichelt (SIAM J. Sci. Comput. 18, 1997), the method behind
//! MATLAB's `ode23s` that the authors of arXiv:1706.07433 used. One LU
//! factorization of `I − h d J` per step, FSAL, and a continuous extension for
//! dense output.

use std::ops::ControlFlow;

use crate::error::{Error, Result};
use crate::math::{DenseMatrix, LuFactors};

/// A system of first-order ODEs `dy/dx = F(x, y)`.
pub trait OdeSystem {
    /// Number of unknowns.
    fn dim(&self) -> usize;
    /// Evaluate `F(x, y)` into `dydx`.
    fn rhs(&mut self, x: f64, y: &[f64], dydx: &mut [f64]);
    /// Jacobian `J_ik = ∂F_i/∂y_k`. The default uses forward differences.
    fn jacobian(&mut self, x: f64, y: &[f64], jac: &mut DenseMatrix) {
        finite_difference_jacobian(self, x, y, jac);
    }
}

/// Forward-difference Jacobian; adequate for small systems.
pub fn finite_difference_jacobian<S: OdeSystem + ?Sized>(
    sys: &mut S,
    x: f64,
    y: &[f64],
    jac: &mut DenseMatrix,
) {
    let n = y.len();
    let mut f0 = vec![0.0; n];
    let mut f1 = vec![0.0; n];
    sys.rhs(x, y, &mut f0);
    let mut yp = y.to_vec();
    for k in 0..n {
        let dy = 1e-7 * y[k].abs().max(1e-300);
        yp[k] = y[k] + dy;
        sys.rhs(x, &yp, &mut f1);
        for i in 0..n {
            jac[(i, k)] = (f1[i] - f0[i]) / dy;
        }
        yp[k] = y[k];
    }
}

/// Receives the solution while it is being integrated.
pub trait Observer {
    /// Called at each requested output abscissa (dense output).
    fn output(&mut self, _x: f64, _y: &[f64]) {}
    /// Called after every accepted step; return `Break` to stop early.
    fn step(&mut self, _x: f64, _y: &[f64]) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
}

/// Observer that ignores everything.
impl Observer for () {}

/// Step-size control parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct OdeOptions {
    /// Relative tolerance (max-norm).
    pub rtol: f64,
    /// Absolute tolerance floor.
    pub atol: f64,
    /// Absolute tolerance relative to the largest component, `atol_i =
    /// max(atol, atol_scaled · max_k |y_k|)`. Useful for distribution
    /// functions whose tails are exponentially small.
    pub atol_scaled: f64,
    /// Initial step.
    pub h0: f64,
    /// Maximal step as a fraction of `|x|`.
    pub max_step_fraction: f64,
    /// Step budget.
    pub max_steps: usize,
}

impl Default for OdeOptions {
    fn default() -> Self {
        Self {
            rtol: 1e-6,
            atol: 1e-300,
            atol_scaled: 0.0,
            h0: 1e-5,
            max_step_fraction: 0.05,
            max_steps: 2_000_000,
        }
    }
}

/// Work counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OdeStats {
    /// Accepted steps.
    pub steps: usize,
    /// Rejected steps.
    pub rejected: usize,
    /// Right-hand-side evaluations.
    pub rhs_evals: usize,
    /// Jacobian evaluations.
    pub jacobian_evals: usize,
    /// LU factorizations.
    pub factorizations: usize,
}

/// Integrate `sys` from `x0` to `x_end`. `outputs` must be sorted; the
/// observer receives the interpolated solution at each of them.
pub fn integrate<S: OdeSystem + ?Sized, O: Observer + ?Sized>(
    sys: &mut S,
    x0: f64,
    y0: &[f64],
    x_end: f64,
    outputs: &[f64],
    opts: &OdeOptions,
    observer: &mut O,
) -> Result<(Vec<f64>, OdeStats)> {
    let n = sys.dim();
    if y0.len() != n {
        return Err(Error::InvalidInput(format!(
            "initial state has {} entries, system has {n}",
            y0.len()
        )));
    }
    if x_end.is_nan() || x_end <= x0 {
        return Err(Error::InvalidInput(format!(
            "x_end = {x_end} must exceed x0 = {x0}"
        )));
    }
    let d = 1.0 / (2.0 + std::f64::consts::SQRT_2);
    let e32 = 6.0 + std::f64::consts::SQRT_2;
    let mut stats = OdeStats::default();
    let mut x = x0;
    let mut y = y0.to_vec();
    let mut f0 = vec![0.0; n];
    sys.rhs(x, &y, &mut f0);
    stats.rhs_evals += 1;

    let mut jac = DenseMatrix::zeros(n);
    let mut lu = LuFactors::new(n);
    let [mut dfdx, mut ft, mut k1, mut k2, mut k3, mut f1, mut f2, mut ytmp, mut ynew, mut yout] =
        std::array::from_fn(|_| vec![0.0; n]);

    let mut next_out = outputs.partition_point(|&xo| xo < x0);
    while next_out < outputs.len() && outputs[next_out] == x0 {
        observer.output(x0, &y);
        next_out += 1;
    }

    let mut h = opts.h0.min(x_end - x0);
    while x < x_end {
        if stats.steps >= opts.max_steps {
            return Err(Error::TooManySteps { x });
        }
        h = h
            .min(opts.max_step_fraction * x.abs().max(f64::MIN_POSITIVE))
            .min(x_end - x);

        sys.jacobian(x, &y, &mut jac);
        stats.jacobian_evals += 1;
        let dx = 1e-7 * x.abs().max(1e-7);
        sys.rhs(x + dx, &y, &mut ft);
        stats.rhs_evals += 1;
        for i in 0..n {
            dfdx[i] = (ft[i] - f0[i]) / dx;
        }

        loop {
            {
                let w = lu.matrix_mut();
                for i in 0..n {
                    for (wk, jk) in w.row_mut(i).iter_mut().zip(jac.row(i)) {
                        *wk = -h * d * jk;
                    }
                    w[(i, i)] += 1.0;
                }
            }
            stats.factorizations += 1;
            if !lu.factorize() {
                h *= 0.25;
                continue;
            }
            for i in 0..n {
                k1[i] = f0[i] + h * d * dfdx[i];
            }
            lu.solve(&mut k1);
            for i in 0..n {
                ytmp[i] = y[i] + 0.5 * h * k1[i];
            }
            sys.rhs(x + 0.5 * h, &ytmp, &mut f1);
            for i in 0..n {
                k2[i] = f1[i] - k1[i];
            }
            lu.solve(&mut k2);
            for i in 0..n {
                k2[i] += k1[i];
                ynew[i] = y[i] + h * k2[i];
            }
            sys.rhs(x + h, &ynew, &mut f2);
            stats.rhs_evals += 2;
            for i in 0..n {
                k3[i] = f2[i] - e32 * (k2[i] - f1[i]) - 2.0 * (k1[i] - f0[i]) + h * d * dfdx[i];
            }
            lu.solve(&mut k3);

            let ymax = y.iter().chain(&ynew).fold(0.0_f64, |a, v| a.max(v.abs()));
            let atol = opts.atol.max(opts.atol_scaled * ymax);
            let err = (0..n)
                .map(|i| {
                    let e = (h / 6.0 * (k1[i] - 2.0 * k2[i] + k3[i])).abs();
                    e / (atol + opts.rtol * y[i].abs().max(ynew[i].abs()))
                })
                .fold(0.0_f64, f64::max);

            if !err.is_finite() || err > 1.0 {
                stats.rejected += 1;
                h *= if err.is_finite() {
                    (0.8 * err.powf(-1.0 / 3.0)).max(0.1)
                } else {
                    0.1
                };
                if h < 1e-14 * x.abs() {
                    return Err(Error::StepSizeUnderflow { x });
                }
                continue;
            }

            while next_out < outputs.len() && outputs[next_out] <= x + h {
                let s = (outputs[next_out] - x) / h;
                let c1 = s * (1.0 - s) / (1.0 - 2.0 * d);
                let c2 = s * (s - 2.0 * d) / (1.0 - 2.0 * d);
                for i in 0..n {
                    yout[i] = y[i] + h * (c1 * k1[i] + c2 * k2[i]);
                }
                observer.output(outputs[next_out], &yout);
                next_out += 1;
            }
            x += h;
            y.copy_from_slice(&ynew);
            f0.copy_from_slice(&f2);
            stats.steps += 1;
            h *= if err > 0.0 {
                (0.8 * err.powf(-1.0 / 3.0)).min(4.0)
            } else {
                4.0
            };
            if observer.step(x, &y).is_break() {
                return Ok((y, stats));
            }
            break;
        }
    }
    Ok((y, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Robertson's stiff chemical kinetics problem.
    struct Robertson;
    impl OdeSystem for Robertson {
        fn dim(&self) -> usize {
            3
        }
        fn rhs(&mut self, _x: f64, y: &[f64], f: &mut [f64]) {
            f[0] = -0.04 * y[0] + 1e4 * y[1] * y[2];
            f[2] = 3e7 * y[1] * y[1];
            f[1] = -f[0] - f[2];
        }
    }

    #[test]
    fn robertson_reference() {
        let opts = OdeOptions {
            rtol: 1e-7,
            atol: 1e-12,
            h0: 1e-6,
            max_step_fraction: 1.0,
            ..Default::default()
        };
        let (y, _) = integrate(
            &mut Robertson,
            0.0,
            &[1.0, 0.0, 0.0],
            40.0,
            &[],
            &opts,
            &mut (),
        )
        .unwrap();
        // reference (Hairer & Wanner): y1(40) = 0.7158270687
        assert!((y[0] / 0.715_827_068_7 - 1.0).abs() < 1e-4, "{y:?}");
        assert!((y.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn dense_output_of_exponential() {
        struct Decay;
        impl OdeSystem for Decay {
            fn dim(&self) -> usize {
                1
            }
            fn rhs(&mut self, _x: f64, y: &[f64], f: &mut [f64]) {
                f[0] = -y[0];
            }
        }
        struct Collect(Vec<(f64, f64)>);
        impl Observer for Collect {
            fn output(&mut self, x: f64, y: &[f64]) {
                self.0.push((x, y[0]));
            }
        }
        let mut c = Collect(Vec::new());
        let outs = [0.5, 1.0, 2.0, 3.0];
        let opts = OdeOptions {
            rtol: 1e-8,
            max_step_fraction: 1.0,
            ..Default::default()
        };
        integrate(&mut Decay, 0.0, &[1.0], 3.0, &outs, &opts, &mut c).unwrap();
        assert_eq!(c.0.len(), 4);
        for (x, v) in c.0 {
            assert!((v - (-x).exp()).abs() < 1e-6, "x = {x}");
        }
    }
}
