//! Annihilation kernel of the full phase-space equation.
//!
//! With rapidities `p = m sinh η` the angle-averaged kernel of Eq. (BEps)
//! becomes `q̃² ⟨vσ⟩^θ dq̃ = x³/(u E) [F(s̃₊) − F(s̃₋)] dẼ`, where
//! `F(s̃) = ∫_1^s̃ F'`, `s̃± = cosh²((η ± η̃)/2)`. The kernel thus depends on
//! `η ± η̃` only. We tabulate
//!
//! ```text
//! Φ(z) = F(cosh²(z/2)),   Φ₁ = ∫_0^z Φ,   Φ₂ = ∫_0^z Φ₁
//! ```
//!
//! with quintic Hermite interpolation on a grid that is refined
//! (atan-spaced) around every declared resonance. The integral of the kernel
//! over a momentum cell of the piecewise-constant `f` is then exact in the
//! sharp Breit–Wigner factor and second order in the smooth Jacobian `sinh η̃`.

use crate::math::gauss_legendre;
use crate::model::{fprime, resonances_st, thresholds_st, DarkMatterModel};

#[derive(Clone, Debug)]
pub(crate) struct PhiTable {
    z: Vec<f64>,
    /// per node: [dΦ/dz, Φ, Φ₁, Φ₂]
    v: Vec<[f64; 4]>,
    bucket: Vec<u32>,
    inv_db: f64,
    z_max: f64,
}

impl PhiTable {
    pub(crate) fn new<M: DarkMatterModel + ?Sized>(model: &M, z_max: f64) -> Self {
        let mut nodes = vec![0.0];
        let mut z = 1e-11;
        while z < 0.05 {
            nodes.push(z);
            z *= 1.06;
        }
        const DZ: f64 = 0.002;
        let n_uniform = (z_max / DZ).ceil() as usize + 1;
        nodes.extend((1..=n_uniform).map(|k| k as f64 * DZ));
        let z_top = n_uniform as f64 * DZ;
        let z_of_st = |st: f64| 2.0 * st.sqrt().acosh();
        for (sr, w) in resonances_st(model) {
            if sr < 1.0 - 50.0 * w {
                continue;
            }
            // Breit-Wigner half-width in z: w / (ds~/dz) = w / (sinh(z)/2) for a
            // pole well above threshold, crossing over to ~sqrt(w) at s~_R -> 1.
            let zr = z_of_st(sr.max(1.0));
            let dzr = w / (0.5 * zr.sinh()).max(0.5 * w.sqrt());
            let th0 = (-zr / dzr).atan();
            let th1 = ((z_top - zr) / dzr).atan();
            const N_RES: usize = 1200;
            nodes.extend(
                (1..N_RES)
                    .map(|k| zr + dzr * (th0 + (th1 - th0) * k as f64 / N_RES as f64).tan())
                    .filter(|&zz| zz > 0.0 && zz < z_top),
            );
        }
        nodes.extend(
            thresholds_st(model)
                .into_iter()
                .map(z_of_st)
                .filter(|&zz| zz < z_top),
        );
        nodes.sort_by(f64::total_cmp);
        nodes.dedup_by(|a, b| (*a - *b).abs() < 1e-13 * (1.0 + b.abs()));

        let dphi = |zz: f64| {
            let c = (0.5 * zz).cosh();
            fprime(model, c * c) * 0.5 * zz.sinh()
        };
        let gl = gauss_legendre(8);
        let n = nodes.len();
        let mut v = vec![[0.0_f64; 4]; n];
        v[0] = [dphi(0.0), 0.0, 0.0, 0.0];
        for k in 0..n - 1 {
            let (a, b) = (nodes[k], nodes[k + 1]);
            let h = b - a;
            let (mut i0, mut i1, mut i2) = (0.0, 0.0, 0.0);
            for (zz, wt) in gl.points(a, b) {
                let f = wt * dphi(zz);
                let r = b - zz;
                i0 += f;
                i1 += f * r;
                i2 += f * 0.5 * r * r;
            }
            let [_, p, p1, p2] = v[k];
            v[k + 1] = [
                dphi(b),
                p + i0,
                p1 + h * p + i1,
                p2 + h * p1 + 0.5 * h * h * p + i2,
            ];
        }
        const N_BUCKETS: usize = 400_000;
        let inv_db = N_BUCKETS as f64 / z_top;
        let mut bucket = vec![0_u32; N_BUCKETS + 2];
        let mut k = 0;
        for (b, slot) in bucket.iter_mut().enumerate() {
            let zb = b as f64 / inv_db;
            while k + 1 < n && nodes[k + 1] <= zb {
                k += 1;
            }
            *slot = k as u32;
        }
        Self {
            z: nodes,
            v,
            bucket,
            inv_db,
            z_max: z_top,
        }
    }

    pub(crate) fn z_max(&self) -> f64 {
        self.z_max
    }

    fn interval(&self, z: f64) -> usize {
        let b = ((z * self.inv_db) as usize).min(self.bucket.len() - 2);
        let mut lo = self.bucket[b] as usize;
        let mut hi = (self.bucket[b + 1] as usize + 1).min(self.z.len() - 1);
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.z[mid] > z {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        lo.min(self.z.len() - 2)
    }

    /// `(Φ₁(z), Φ₂(z))` for real `z` (Φ₁ odd, Φ₂ even).
    #[inline]
    pub(crate) fn eval12(&self, z: f64) -> (f64, f64) {
        let az = z.abs();
        debug_assert!(az <= self.z_max, "z = {az} beyond table {}", self.z_max);
        let k = self.interval(az);
        let (a, b) = (self.z[k], self.z[k + 1]);
        let h = b - a;
        let t = (az - a) / h;
        let (t2, t3) = (t * t, t * t * t);
        let (t4, t5) = (t3 * t, t3 * t2);
        let h0 = 1.0 - 10.0 * t3 + 15.0 * t4 - 6.0 * t5;
        let h1 = t - 6.0 * t3 + 8.0 * t4 - 3.0 * t5;
        let h2 = 0.5 * (t2 - 3.0 * t3 + 3.0 * t4 - t5);
        let h3 = 1.0 - h0;
        let h4 = -4.0 * t3 + 7.0 * t4 - 3.0 * t5;
        let h5 = 0.5 * (t3 - 2.0 * t4 + t5);
        let [f0a, pa, p1a, p2a] = self.v[k];
        let [f0b, pb, p1b, p2b] = self.v[k + 1];
        let hh = h * h;
        let p1 = p1a * h0 + h * pa * h1 + hh * f0a * h2 + p1b * h3 + h * pb * h4 + hh * f0b * h5;
        let p2 = p2a * h0 + h * p1a * h1 + hh * pa * h2 + p2b * h3 + h * p1b * h4 + hh * pb * h5;
        (if z < 0.0 { -p1 } else { p1 }, p2)
    }
}
