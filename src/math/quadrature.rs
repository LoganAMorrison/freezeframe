use std::f64::consts::PI;
use std::sync::OnceLock;

/// Gauss–Legendre nodes and weights on `[-1, 1]`.
#[derive(Clone, Debug)]
pub struct GaussLegendre {
    nodes: Vec<f64>,
    weights: Vec<f64>,
}

impl GaussLegendre {
    /// Rule with `n >= 1` points (Newton iteration on the Legendre polynomials).
    pub fn new(n: usize) -> Self {
        assert!(n >= 1, "Gauss-Legendre rule needs at least one node");
        let mut nodes = vec![0.0; n];
        let mut weights = vec![0.0; n];
        for i in 0..n.div_ceil(2) {
            let mut z = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
            let mut dp;
            loop {
                let (mut p1, mut p2) = (1.0_f64, 0.0_f64);
                for j in 0..n {
                    let p3 = p2;
                    p2 = p1;
                    p1 = ((2 * j + 1) as f64 * z * p2 - j as f64 * p3) / (j + 1) as f64;
                }
                dp = n as f64 * (z * p1 - p2) / (z * z - 1.0);
                let dz = p1 / dp;
                z -= dz;
                if dz.abs() < 1e-15 {
                    break;
                }
            }
            nodes[i] = -z;
            nodes[n - 1 - i] = z;
            let w = 2.0 / ((1.0 - z * z) * dp * dp);
            weights[i] = w;
            weights[n - 1 - i] = w;
        }
        Self { nodes, weights }
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Always false (a rule has at least one node).
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Iterator over `(abscissa, weight)` pairs mapped to `[a, b]`.
    pub fn points(&self, a: f64, b: f64) -> impl Iterator<Item = (f64, f64)> + '_ {
        let (c, d) = (0.5 * (a + b), 0.5 * (b - a));
        self.nodes
            .iter()
            .zip(&self.weights)
            .map(move |(&x, &w)| (c + d * x, w * d))
    }

    /// `∫_a^b f(x) dx`.
    pub fn integrate(&self, a: f64, b: f64, mut f: impl FnMut(f64) -> f64) -> f64 {
        self.points(a, b).map(|(x, w)| w * f(x)).sum()
    }

    /// Composite rule over consecutive panels `edges[k]..edges[k+1]`.
    pub fn integrate_panels(&self, edges: &[f64], mut f: impl FnMut(f64) -> f64) -> f64 {
        edges
            .windows(2)
            .map(|e| self.integrate(e[0], e[1], &mut f))
            .sum()
    }
}

/// Shared, lazily built rule with `n <= 64` nodes.
pub fn gauss_legendre(n: usize) -> &'static GaussLegendre {
    const MAX: usize = 64;
    static RULES: [OnceLock<GaussLegendre>; MAX + 1] = [const { OnceLock::new() }; MAX + 1];
    assert!(
        (1..=MAX).contains(&n),
        "cached Gauss-Legendre rules exist for 1..=64 nodes"
    );
    RULES[n].get_or_init(|| GaussLegendre::new(n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integrates_polynomials_exactly() {
        let gl = gauss_legendre(6);
        // exact for degree <= 11
        let v = gl.integrate(-0.3, 2.1, |x| x.powi(11) - 3.0 * x.powi(4));
        let exact = |x: f64| x.powi(12) / 12.0 - 0.6 * x.powi(5);
        assert!((v - (exact(2.1) - exact(-0.3))).abs() < 1e-11 * exact(2.1).abs());
    }

    #[test]
    fn weights_sum_to_two() {
        for n in [1, 2, 7, 24, 64] {
            let s: f64 = gauss_legendre(n).points(-1.0, 1.0).map(|(_, w)| w).sum();
            assert!((s - 2.0).abs() < 1e-13, "n = {n}");
        }
    }
}
