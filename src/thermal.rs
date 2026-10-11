//! Thermal averages for the moment equations.
//!
//! With `s̃ = s/(4m²)` and `F'(s̃) = s̃ σv_cms` (see [`crate::model`]):
//!
//! ```text
//! ⟨σv⟩(x)   = ∫ ds̃ F'(s̃) 2x sqrt(s̃−1) K₁(2x sqrt s̃) / K₂(x)²                    (Eq. 14)
//! ⟨σv⟩₂(x)  = ∫ ds̃ F'(s̃) 4s̃x³/(3K₂(x)²) ∫₁^∞ dε e^{−aε}
//!             [ε sqrt((s̃−1)(ε²−1)) − atanh(sqrt((s̃−1)(ε²−1))/(sqrt(s̃) ε))/sqrt(s̃)],  a = 2x sqrt s̃   (Eq. 30)
//! ```
//!
//! The `s̃` integral is done in `v` with `v² = 2x(sqrt s̃ − 1)`, on panels
//! that are refined (atan-spaced) around declared resonances and split at
//! declared thresholds.

use crate::error::{Error, Result};
use crate::math::{bessel_k_scaled, gauss_legendre, CubicSpline, Extrapolation};
use crate::model::{fprime, resonances_st, thresholds_st, DarkMatterModel};

/// Upper limit of `v² = 2x(sqrt s̃ − 1)`; the integrand is `∝ e^{−v²}`.
const V2_MAX: f64 = 60.0;

fn breakpoints(resonances: &[(f64, f64)], thresholds: &[f64], x: f64) -> Vec<f64> {
    let v_max = V2_MAX.sqrt();
    let v_of_st = |st: f64| (2.0 * x * (st.sqrt() - 1.0)).sqrt();
    let st_hi = (1.0 + V2_MAX / (2.0 * x)).powi(2);
    let mut bp: Vec<f64> = (0..=48).map(|i| v_max * i as f64 / 48.0).collect();
    bp.extend((1..40).map(|k| v_max / 48.0 * 0.7_f64.powi(k)));
    for &(sr, w) in resonances {
        if sr > 1.0 - 50.0 * w && sr < st_hi + 50.0 * w {
            let th0 = ((1.0 - sr) / w).atan();
            let th1 = ((st_hi - sr) / w).atan();
            const N: usize = 160;
            bp.extend(
                (0..=N)
                    .map(|k| sr + w * (th0 + (th1 - th0) * k as f64 / N as f64).tan())
                    .filter(|&st| st > 1.0 && st < st_hi)
                    .map(v_of_st),
            );
        }
    }
    bp.extend(
        thresholds
            .iter()
            .filter(|&&st| st < st_hi)
            .map(|&st| v_of_st(st)),
    );
    bp.retain(|v| v.is_finite() && (0.0..=v_max).contains(v));
    bp.sort_by(f64::total_cmp);
    bp.dedup_by(|a, b| (*a - *b).abs() < 1e-14 * (1.0 + b.abs()));
    bp
}

/// `L(s̃, a) = ∫_0^∞ dw e^{−w} atanh(B/A)` with `ε = 1 + w/a`.
fn log_term(st: f64, a: f64) -> f64 {
    let gl = gauss_legendre(16);
    let (sq, sm1) = (st.sqrt(), st - 1.0);
    let f = |r: f64| {
        let w = r * r;
        let e = 1.0 + w / a;
        let e2m1 = (w / a) * (2.0 + w / a);
        2.0 * r * (-w).exp() * ((sm1 * e2m1).sqrt() / (sq * e)).atanh()
    };
    gl.integrate_panels(&[0.0, 1.5, 3.0, 50_f64.sqrt()], f)
}

/// `(⟨σv⟩(x), ⟨σv⟩₂(x))` \[GeV⁻²\] by direct quadrature. Pass
/// `second_moment = false` to skip the (more expensive) `⟨σv⟩₂`.
pub fn thermal_averages<M: DarkMatterModel + ?Sized>(
    model: &M,
    x: f64,
    second_moment: bool,
) -> (f64, f64) {
    let res = resonances_st(model);
    let thr = thresholds_st(model);
    averages_with(model, &res, &thr, x, second_moment)
}

fn averages_with<M: DarkMatterModel + ?Sized>(
    model: &M,
    res: &[(f64, f64)],
    thr: &[f64],
    x: f64,
    second_moment: bool,
) -> (f64, f64) {
    let bp = breakpoints(res, thr, x);
    let gl = gauss_legendre(6);
    let k2 = bessel_k_scaled(2.0, x);
    let (mut sv, mut sv2) = (0.0, 0.0);
    for w in bp.windows(2) {
        for (v, wt) in gl.points(w[0], w[1]) {
            let t = 1.0 + v * v / (2.0 * x); // sqrt(s~)
            let st = t * t;
            let jac = wt * fprime(model, st) * (-v * v).exp() * 2.0 * t * v / x;
            let sm1 = (st - 1.0).max(0.0);
            let a = 2.0 * x * t;
            // e^{-2x(t-1)} = e^{-v²} is factored out of both kernels
            sv += jac * 2.0 * x * sm1.sqrt() * bessel_k_scaled(1.0, a) / (k2 * k2);
            if second_moment {
                let br = sm1.sqrt() * bessel_k_scaled(2.0, a) / a - log_term(st, a) / (t * a);
                sv2 += jac * 4.0 * st * x.powi(3) / (3.0 * k2 * k2) * br;
            }
        }
    }
    (sv, sv2)
}

/// `⟨σv⟩(x)` and `⟨σv⟩₂(x)` tabulated as `ln⟨σv⟩` against `ln x`, with
/// power-law extrapolation outside the table.
#[derive(Clone, Debug)]
pub struct ThermalAverages {
    sv: CubicSpline,
    sv2: Option<CubicSpline>,
}

impl ThermalAverages {
    /// Tabulate on `[x_min, x_max]` with 40 nodes per decade.
    pub fn new<M: DarkMatterModel + ?Sized>(
        model: &M,
        x_min: f64,
        x_max: f64,
        second_moment: bool,
    ) -> Result<Self> {
        if !(x_min > 0.0 && x_max > x_min) {
            return Err(Error::InvalidInput(format!(
                "bad x range [{x_min}, {x_max}]"
            )));
        }
        let res = resonances_st(model);
        let thr = thresholds_st(model);
        let (l0, l1) = (x_min.ln(), x_max.ln());
        let n = ((l1 - l0) / std::f64::consts::LN_10 * 40.0).ceil().max(8.0) as usize;
        let mut lx = Vec::with_capacity(n + 1);
        let mut a = Vec::with_capacity(n + 1);
        let mut b = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let l = l0 + (l1 - l0) * i as f64 / n as f64;
            let (s1, s2) = averages_with(model, &res, &thr, l.exp(), second_moment);
            if !(s1.is_finite() && s1 >= 0.0) || (second_moment && !(s2.is_finite() && s2 >= 0.0)) {
                return Err(Error::InvalidModel(format!(
                    "thermal average not finite/non-negative at x = {:.4e} (⟨σv⟩ = {s1}, ⟨σv⟩₂ = {s2})",
                    l.exp()
                )));
            }
            lx.push(l);
            a.push(s1.max(1e-300).ln());
            b.push(s2.max(1e-300).ln());
        }
        let sv = CubicSpline::new(lx.clone(), a)?.with_extrapolation(Extrapolation::Linear);
        let sv2 = if second_moment {
            Some(CubicSpline::new(lx, b)?.with_extrapolation(Extrapolation::Linear))
        } else {
            None
        };
        Ok(Self { sv, sv2 })
    }

    /// `⟨σv⟩(x)` \[GeV⁻²\].
    pub fn sigma_v(&self, x: f64) -> f64 {
        self.sv.eval(x.ln()).exp()
    }

    /// `⟨σv⟩₂(x)` \[GeV⁻²\]. Panics if the table was built without it.
    pub fn sigma_v2(&self, x: f64) -> f64 {
        self.sv2
            .as_ref()
            .expect("table built without the second moment")
            .eval(x.ln())
            .exp()
    }
}

/// For a Maxwell–Boltzmann distribution `e^{−E/T_χ}` with `x_χ = m/T_χ`:
/// `⟨p⁴/E³⟩/(3T_χ)`, which equals `2(1 − w)` in the notation of Eq. 32.
pub fn mb_w34(x_chi: f64) -> f64 {
    let gl = gauss_legendre(16);
    let edges = [0.0, 1.0, 2.0, 3.5, 5.5, 62_f64.sqrt()];
    let (mut num, mut den) = (0.0, 0.0);
    for w in edges.windows(2) {
        for (v, wt) in gl.points(w[0], w[1]) {
            // kinetic energy k = v², momentum t = sqrt(k² + 2kx), t² dt = t E dk
            let k = v * v;
            let t = (k * (k + 2.0 * x_chi)).sqrt();
            let e = k + x_chi;
            let weight = wt * t * e * (-k).exp() * 2.0 * v;
            den += weight;
            num += weight * t.powi(4) / e.powi(3);
        }
    }
    num / den / 3.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{cms_velocity, lab_velocity, ModelFn};

    /// σv_lab = 1 ⇒ ⟨σv⟩ = ⟨σv⟩₂ = 1 exactly, at any x.
    #[test]
    fn unit_lab_cross_section() {
        let m = 50.0;
        let model = ModelFn::new(
            m,
            1.0,
            move |s: f64| {
                let st = s / (4.0 * m * m);
                (2.0 * st - 1.0) / st
            },
            |_| 0.0,
        );
        for x in [1.0, 10.0, 1e3, 1e6] {
            let (a, b) = thermal_averages(&model, x, true);
            assert!((a - 1.0).abs() < 1e-9, "x = {x}: {a}");
            assert!((b - 1.0).abs() < 1e-7, "x = {x}: {b}");
        }
    }

    #[test]
    fn velocity_conventions() {
        let (m, s) = (3.0, 50.0);
        assert!(
            (cms_velocity(s, m) * s - 2.0 * lab_velocity(s, m) * (s - 2.0 * m * m)).abs() < 1e-12
        );
    }

    #[test]
    fn mb_limits() {
        // non-relativistic: <p⁴/E³>/(3T) → 5 T/m ... ; w = 1 - w34/2 → 1
        assert!(mb_w34(1e6) < 1e-5);
        // w(x=10) = 0.826 (paper quotes ≈ 0.8)
        assert!(((1.0 - 0.5 * mb_w34(10.0)) - 0.826).abs() < 1e-3);
    }
}
