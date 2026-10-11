use crate::error::{Error, Result};

/// Behaviour of a [`CubicSpline`] outside its tabulated range.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Extrapolation {
    /// Return the end value.
    #[default]
    Clamp,
    /// Continue linearly with the end slope.
    Linear,
}

/// Natural cubic spline through `(x_i, y_i)` on a strictly increasing grid.
#[derive(Clone, Debug)]
pub struct CubicSpline {
    x: Vec<f64>,
    y: Vec<f64>,
    y2: Vec<f64>,
    extrapolation: Extrapolation,
}

impl CubicSpline {
    /// Build the spline. Fails for fewer than three points, mismatched
    /// lengths, non-finite values or a grid that is not strictly increasing.
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Result<Self> {
        let n = x.len();
        if n < 3 || y.len() != n {
            return Err(Error::InvalidInput(format!(
                "spline needs >= 3 points of equal length (got {} and {})",
                n,
                y.len()
            )));
        }
        if x.windows(2)
            .any(|w| w[1].partial_cmp(&w[0]) != Some(std::cmp::Ordering::Greater))
            || x.iter().chain(&y).any(|v| !v.is_finite())
        {
            return Err(Error::InvalidInput(
                "spline grid must be finite and strictly increasing".into(),
            ));
        }
        let mut y2 = vec![0.0; n];
        let mut u = vec![0.0; n];
        for i in 1..n - 1 {
            let sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
            let p = sig * y2[i - 1] + 2.0;
            y2[i] = (sig - 1.0) / p;
            let dd = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
            u[i] = (6.0 * dd / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p;
        }
        for k in (0..n - 1).rev() {
            y2[k] = y2[k] * y2[k + 1] + u[k];
        }
        Ok(Self {
            x,
            y,
            y2,
            extrapolation: Extrapolation::Clamp,
        })
    }

    /// Select the extrapolation policy.
    pub fn with_extrapolation(mut self, e: Extrapolation) -> Self {
        self.extrapolation = e;
        self
    }

    /// Tabulated domain `(x_first, x_last)`.
    pub fn domain(&self) -> (f64, f64) {
        (self.x[0], self.x[self.x.len() - 1])
    }

    fn segment(&self, xv: f64) -> usize {
        self.x
            .partition_point(|&xi| xi <= xv)
            .clamp(1, self.x.len() - 1)
            - 1
    }

    fn eval_in(&self, k: usize, xv: f64) -> f64 {
        let h = self.x[k + 1] - self.x[k];
        let a = (self.x[k + 1] - xv) / h;
        let b = 1.0 - a;
        a * self.y[k]
            + b * self.y[k + 1]
            + ((a * a * a - a) * self.y2[k] + (b * b * b - b) * self.y2[k + 1]) * h * h / 6.0
    }

    fn slope_in(&self, k: usize, xv: f64) -> f64 {
        let h = self.x[k + 1] - self.x[k];
        let a = (self.x[k + 1] - xv) / h;
        let b = 1.0 - a;
        (self.y[k + 1] - self.y[k]) / h
            + ((1.0 - 3.0 * a * a) * self.y2[k] + (3.0 * b * b - 1.0) * self.y2[k + 1]) * h / 6.0
    }

    /// Spline value at `xv`.
    pub fn eval(&self, xv: f64) -> f64 {
        let n = self.x.len();
        let (lo, hi) = (self.x[0], self.x[n - 1]);
        match self.extrapolation {
            Extrapolation::Clamp if xv <= lo => self.y[0],
            Extrapolation::Clamp if xv >= hi => self.y[n - 1],
            Extrapolation::Linear if xv < lo => self.y[0] + (xv - lo) * self.slope_in(0, lo),
            Extrapolation::Linear if xv > hi => {
                self.y[n - 1] + (xv - hi) * self.slope_in(n - 2, hi)
            }
            _ => self.eval_in(self.segment(xv), xv),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reproduces_smooth_function() {
        let x: Vec<f64> = (0..=200).map(|i| i as f64 * 0.05).collect();
        let y: Vec<f64> = x.iter().map(|v| v.sin()).collect();
        let s = CubicSpline::new(x, y).unwrap();
        for v in [0.37, 3.3, 7.77] {
            assert!((s.eval(v) - f64::sin(v)).abs() < 1e-6);
        }
    }

    #[test]
    fn extrapolation_policies() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 2.0, 4.0, 6.0];
        let s = CubicSpline::new(x, y).unwrap();
        assert_eq!(s.eval(10.0), 6.0);
        let s = s.with_extrapolation(Extrapolation::Linear);
        assert!((s.eval(10.0) - 20.0).abs() < 1e-12);
        assert!((s.eval(-1.0) + 2.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_bad_grids() {
        assert!(CubicSpline::new(vec![0.0, 1.0], vec![0.0, 1.0]).is_err());
        assert!(CubicSpline::new(vec![0.0, 1.0, 1.0], vec![0.0, 1.0, 2.0]).is_err());
    }
}
