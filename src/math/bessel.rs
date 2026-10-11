use std::f64::consts::PI;

/// Exponentially scaled modified Bessel function of the second kind,
/// `e^z K_ν(z)`, for real `ν >= 0` and `z > 0`.
///
/// For `z >= 25` the Hankel asymptotic series is used (truncated at its
/// smallest term, relative accuracy ~1e-16). Below, the integral
/// representation `e^z K_ν(z) = ∫_0^∞ exp[-z(cosh t − 1)] cosh(νt) dt` is
/// evaluated with the trapezoidal rule, which converges geometrically for this
/// analytic, rapidly decaying integrand.
pub fn bessel_k_scaled(nu: f64, z: f64) -> f64 {
    debug_assert!(z > 0.0, "bessel_k_scaled requires z > 0");
    if z >= 25.0 {
        let mu = 4.0 * nu * nu;
        let (mut term, mut sum) = (1.0_f64, 1.0_f64);
        for k in 1..200 {
            let kf = k as f64;
            let next = term * (mu - (2.0 * kf - 1.0).powi(2)) / (8.0 * kf * z);
            if next.abs() > term.abs() {
                break;
            }
            term = next;
            sum += term;
            if term.abs() < 1e-17 * sum.abs() {
                break;
            }
        }
        (PI / (2.0 * z)).sqrt() * sum
    } else {
        // Step 0.05 is far inside the geometric-convergence regime
        // (error ~ exp(-2π d / h) with strip half-width d ~ 1).
        let t_max = (1.0 + 70.0 / z).acosh() + 1.0;
        let n = (t_max / 0.05).ceil() as usize;
        let h = t_max / n as f64;
        let tail: f64 = (1..=n)
            .map(|i| {
                let t = i as f64 * h;
                (-z * (t.cosh() - 1.0)).exp() * (nu * t).cosh()
            })
            .sum();
        (0.5 + tail) * h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_values() {
        // scipy.special.kve
        let cases = [
            (1.0, 0.1, 10.890182683049696),
            (2.0, 1.0, 4.416770052333411),
            (1.0, 10.0, 0.41076657059578875),
            (2.0, 30.0, 0.24333027424143452),
            (2.0, 0.01, 20200.49838567655),
            (1.0, 24.9, 0.25490238558081707),
        ];
        for (nu, z, want) in cases {
            let got = bessel_k_scaled(nu, z);
            assert!(
                (got / want - 1.0).abs() < 1e-10,
                "K_{nu}({z}): {got} vs {want}"
            );
        }
    }

    #[test]
    fn half_integer_closed_form() {
        // e^z K_{1/2}(z) = sqrt(pi / 2z)
        for z in [0.3, 4.0, 24.9, 25.1, 400.0] {
            let want = (PI / (2.0 * z)).sqrt();
            assert!((bessel_k_scaled(0.5, z) / want - 1.0).abs() < 1e-12);
        }
    }
}
