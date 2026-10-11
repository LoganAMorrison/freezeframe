//! The ODE systems: standard number-density equation, coupled moment
//! equations and the discretized full phase-space equation.

use std::f64::consts::PI;

use crate::cosmology::Cosmology;
use crate::kernel::PhiTable;
use crate::math::{bessel_k_scaled, DenseMatrix};
use crate::model::DarkMatterModel;
use crate::ode::OdeSystem;
use crate::thermal::{mb_w34, ThermalAverages};

use super::ScatteringTerm;

/// Equilibrium values `(Y_eq, y_eq)` at `x = m/T` (Maxwell–Boltzmann).
pub(crate) fn equilibrium(mass: f64, dof: f64, cosmo: &Cosmology, x: f64) -> (f64, f64) {
    let t = mass / x;
    let s = cosmo.entropy_density(t);
    let n_eq = dof * mass.powi(3) * bessel_k_scaled(2.0, x) * (-x).exp() / (2.0 * PI * PI * x);
    (n_eq / s, mass * t / s.powf(2.0 / 3.0))
}

// --------------------------------------------------------------- number density

/// `dY/dx = (s⟨σv⟩ / (x H̃)) (Y_eq² − Y²)`  (Eq. 13).
pub(crate) struct NumberDensity<'a> {
    pub mass: f64,
    pub dof: f64,
    pub cosmo: &'a Cosmology,
    pub averages: &'a ThermalAverages,
}

impl NumberDensity<'_> {
    fn coefficients(&self, x: f64) -> (f64, f64) {
        let p = self.cosmo.plasma(self.mass / x);
        let (y_eq, _) = equilibrium(self.mass, self.dof, self.cosmo, x);
        (
            p.entropy_density / (x * p.htilde) * self.averages.sigma_v(x),
            y_eq,
        )
    }
}

impl OdeSystem for NumberDensity<'_> {
    fn dim(&self) -> usize {
        1
    }
    fn rhs(&mut self, x: f64, y: &[f64], dy: &mut [f64]) {
        let (lambda, y_eq) = self.coefficients(x);
        dy[0] = lambda * (y_eq * y_eq - y[0] * y[0]);
    }
    fn jacobian(&mut self, x: f64, y: &[f64], jac: &mut DenseMatrix) {
        let (lambda, _) = self.coefficients(x);
        jac[(0, 0)] = -2.0 * lambda * y[0];
    }
}

// --------------------------------------------------------------- coupled (Y, y)

/// Coupled equations for `Y` and `y = m T_χ / s^{2/3}` (Eqs. 21–22), with
/// the semi-relativistic scattering term of Eq. 32 by default.
pub(crate) struct Coupled<'a, M: ?Sized> {
    pub model: &'a M,
    pub dof: f64,
    pub cosmo: &'a Cosmology,
    pub averages: &'a ThermalAverages,
    pub scattering: ScatteringTerm,
}

impl<M: DarkMatterModel + ?Sized> OdeSystem for Coupled<'_, M> {
    fn dim(&self) -> usize {
        2
    }
    fn rhs(&mut self, x: f64, v: &[f64], dy: &mut [f64]) {
        let m = self.model.mass();
        let (yy, ys) = (v[0], v[1]);
        let t = m / x;
        let p = self.cosmo.plasma(t);
        let (ye, yse) = equilibrium(m, self.dof, self.cosmo, x);
        let x_chi = x * yse / ys;
        let lam = p.entropy_density / (x * p.htilde);
        let th = self.averages;
        let (sv, sv2, svc, sv2c) = (
            th.sigma_v(x),
            th.sigma_v2(x),
            th.sigma_v(x_chi),
            th.sigma_v2(x_chi),
        );
        let w34 = mb_w34(x_chi);
        let wfac = match self.scattering {
            ScatteringTerm::SemiRelativistic => 1.0 - 0.5 * w34,
            ScatteringTerm::NonRelativistic => 1.0,
        };
        let gamma = self.model.momentum_transfer_rate(t);
        dy[0] = lam * (ye * ye * sv - yy * yy * svc);
        let rate = gamma / (x * p.htilde) * wfac * (yse / ys - 1.0)
            + lam * yy * (svc - sv2c)
            + lam * ye * ye / yy * (yse / ys * sv2 - sv)
            + (1.0 + p.gtilde) / x * w34;
        dy[1] = ys * rate;
    }
}

// --------------------------------------------------------------- full phase space

/// Finite-volume discretisation of Eq. (BEps), piecewise-constant `f`.
///
/// The momentum variable is the *comoving* `q̂ = q / r(x)` with
/// `r = [h_eff(T)/h_eff(T_start)]^{1/3}`, i.e. `q̂ ∝ p a`. Since
/// `x r'/r = −g̃`, the Liouville term `g̃ q ∂_q f` of Eq. (BEps) is absorbed
/// exactly, `x ∂ₓ f|_q̂ = C[f]`, and the collision terms are evaluated on the
/// instantaneous physical cells `q = r q̂`:
///
/// * Fokker–Planck term in conservative, exponentially fitted
///   (Chang–Cooper / Scharfetter–Gummel) form, so that `e^{−E/T}` is the
///   exact discrete zero-flux state and number is conserved to round-off;
/// * annihilation from cell integrals of the rapidity-space kernel.
pub(crate) struct PhaseSpace<'a, M: ?Sized> {
    model: &'a M,
    cosmo: &'a Cosmology,
    dof: f64,
    n: usize,
    /// comoving cell centres, edges and volumes `∫ q̂² dq̂`
    pub(crate) q_hat: Vec<f64>,
    pub(crate) qe_hat: Vec<f64>,
    pub(crate) vol_hat: Vec<f64>,
    heff_start: f64,
    /// physical (`r q̂`) quantities at `cache_x`
    q: Vec<f64>,
    qe: Vec<f64>,
    vol: Vec<f64>,
    phi: PhiTable,
    cache_x: f64,
    kern: DenseMatrix,
    feq: Vec<f64>,
    req: Vec<f64>,
    cfp: Vec<f64>,
    ap: Vec<f64>,
    am: Vec<f64>,
    gfp: f64,
    xq: Vec<f64>,
    xqe: Vec<f64>,
    etae: Vec<f64>,
    sj: Vec<f64>,
    cj: Vec<f64>,
    he: Vec<f64>,
    threads: usize,
}

impl<'a, M: DarkMatterModel + ?Sized> PhaseSpace<'a, M> {
    pub(crate) fn heff_start(&self) -> f64 {
        self.heff_start
    }

    pub(crate) fn new(
        model: &'a M,
        cosmo: &'a Cosmology,
        qe_hat: Vec<f64>,
        x_min: f64,
        threads: usize,
    ) -> Self {
        let n = qe_hat.len() - 1;
        let q_max = qe_hat[n];
        let q_hat: Vec<f64> = qe_hat.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();
        let vol_hat: Vec<f64> = qe_hat
            .windows(2)
            .map(|w| (w[1].powi(3) - w[0].powi(3)) / 3.0)
            .collect();
        // h_eff only decreases with time, so r <= 1 and q_max bounds the physical grid
        let heff_start = cosmo.heff(model.mass() / x_min);
        let z_max = 2.0 * (q_max / x_min).asinh() * 1.001 + 1e-3;
        let z = || vec![0.0; n];
        let ze = || vec![0.0; n + 1];
        Self {
            model,
            cosmo,
            dof: model.dof(),
            n,
            phi: PhiTable::new(model, z_max),
            q: q_hat.clone(),
            qe: qe_hat.clone(),
            vol: vol_hat.clone(),
            q_hat,
            qe_hat,
            vol_hat,
            heff_start,
            cache_x: f64::NAN,
            kern: DenseMatrix::zeros(n),
            feq: z(),
            req: z(),
            cfp: ze(),
            ap: ze(),
            am: ze(),
            gfp: 0.0,
            xq: z(),
            xqe: ze(),
            etae: ze(),
            sj: z(),
            cj: z(),
            he: z(),
            threads,
        }
    }

    /// `f_eq = e^{−E/T}` on the cell centres (at `x_start`, where `r = 1`).
    pub(crate) fn equilibrium_distribution(&self, x: f64) -> Vec<f64> {
        self.q_hat
            .iter()
            .map(|&q| (-(x * x + q * q).sqrt()).exp())
            .collect()
    }

    /// Recompute all `x`-dependent coefficients (cached on `x`).
    fn prepare(&mut self, x: f64) {
        if x == self.cache_x {
            return;
        }
        debug_assert!(2.0 * (self.qe[self.n] / x).asinh() <= self.phi.z_max());
        let n = self.n;
        let m = self.model.mass();
        let p = self.cosmo.plasma(m / x);
        let r = (p.heff / self.heff_start).cbrt();
        for (dst, src) in self.q.iter_mut().zip(&self.q_hat) {
            *dst = r * src;
        }
        for (dst, src) in self.qe.iter_mut().zip(&self.qe_hat) {
            *dst = r * src;
        }
        for (dst, src) in self.vol.iter_mut().zip(&self.vol_hat) {
            *dst = r * r * r * src;
        }
        self.gfp = self.model.momentum_transfer_rate(m / x) / (2.0 * p.htilde * x);
        let pref = m.powi(3) * self.dof / (2.0 * PI * PI * p.htilde * x);
        for i in 0..n {
            self.xq[i] = (x * x + self.q[i] * self.q[i]).sqrt();
            self.feq[i] = (-self.xq[i]).exp();
        }
        for k in 0..=n {
            self.xqe[k] = (x * x + self.qe[k] * self.qe[k]).sqrt();
            self.etae[k] = (self.qe[k] / x).asinh();
        }
        self.cfp[0] = 0.0;
        self.cfp[n] = 0.0;
        for k in 1..n {
            self.cfp[k] = self.qe[k] * self.qe[k] * self.xqe[k] / (self.q[k] - self.q[k - 1]);
            self.ap[k] = (self.xq[k] - self.xqe[k]).exp();
            self.am[k] = (self.xq[k - 1] - self.xqe[k]).exp();
        }
        for j in 0..n {
            let c = 0.5 * (self.etae[j] + self.etae[j + 1]);
            self.sj[j] = c.sinh();
            self.cj[j] = c.cosh();
            self.he[j] = self.etae[j + 1] - self.etae[j];
        }
        let ctx = KernelRows {
            phi: &self.phi,
            q: &self.q,
            xq: &self.xq,
            etae: &self.etae,
            sj: &self.sj,
            cj: &self.cj,
            he: &self.he,
            feq: &self.feq,
            pref,
            x,
        };
        let threads = self.threads.clamp(1, n);
        if threads == 1 {
            ctx.fill(0, self.kern.as_mut_slice(), &mut self.req);
        } else {
            let rows_per = n.div_ceil(threads);
            std::thread::scope(|scope| {
                let kern_chunks = self.kern.as_mut_slice().chunks_mut(rows_per * n);
                let req_chunks = self.req.chunks_mut(rows_per);
                for (c, (kc, rc)) in kern_chunks.zip(req_chunks).enumerate() {
                    let ctx = &ctx;
                    scope.spawn(move || ctx.fill(c * rows_per, kc, rc));
                }
            });
        }
        self.cache_x = x;
    }
}

impl<M: DarkMatterModel + ?Sized> OdeSystem for PhaseSpace<'_, M> {
    fn dim(&self) -> usize {
        self.n
    }

    fn rhs(&mut self, x: f64, f: &[f64], df: &mut [f64]) {
        self.prepare(x);
        let n = self.n;
        for i in 0..n {
            let rate: f64 = self.kern.row(i).iter().zip(f).map(|(k, fj)| k * fj).sum();
            df[i] = self.feq[i] * self.req[i] - f[i] * rate;
        }
        let mut j_left = 0.0;
        for i in 0..n {
            let j_right = if i + 1 < n {
                self.cfp[i + 1] * (f[i + 1] * self.ap[i + 1] - f[i] * self.am[i + 1])
            } else {
                0.0
            };
            df[i] += self.gfp * (j_right - j_left) / self.vol[i];
            j_left = j_right;
        }
    }

    fn jacobian(&mut self, x: f64, f: &[f64], jac: &mut DenseMatrix) {
        self.prepare(x);
        let n = self.n;
        for i in 0..n {
            let krow = self.kern.row(i);
            let rate: f64 = krow.iter().zip(f).map(|(k, fj)| k * fj).sum();
            let fi = f[i];
            let jrow = jac.row_mut(i);
            for (jk, kk) in jrow.iter_mut().zip(krow) {
                *jk = -fi * kk;
            }
            jrow[i] -= rate;
        }
        for i in 0..n {
            let v = self.vol[i];
            if i + 1 < n {
                jac[(i, i + 1)] += self.gfp * self.cfp[i + 1] * self.ap[i + 1] / v;
                jac[(i, i)] -= self.gfp * self.cfp[i + 1] * self.am[i + 1] / v;
            }
            if i >= 1 {
                jac[(i, i)] -= self.gfp * self.cfp[i] * self.ap[i] / v;
                jac[(i, i - 1)] += self.gfp * self.cfp[i] * self.am[i] / v;
            }
        }
    }
}

/// Read-only data needed to fill rows of the annihilation kernel.
struct KernelRows<'k> {
    phi: &'k PhiTable,
    q: &'k [f64],
    xq: &'k [f64],
    etae: &'k [f64],
    sj: &'k [f64],
    cj: &'k [f64],
    he: &'k [f64],
    feq: &'k [f64],
    pref: f64,
    x: f64,
}

impl KernelRows<'_> {
    /// Fill kernel rows `first..first + req.len()` (row-major `kern`) and
    /// the equilibrium rates `req_i = Σ_j K_ij f_eq,j`.
    fn fill(&self, first: usize, kern: &mut [f64], req: &mut [f64]) {
        let n = self.q.len();
        let mut p1p = vec![0.0; n + 1];
        let mut p2p = vec![0.0; n + 1];
        let mut p1m = vec![0.0; n + 1];
        let mut p2m = vec![0.0; n + 1];
        for (r, (row, req_i)) in kern.chunks_exact_mut(n).zip(req.iter_mut()).enumerate() {
            let i = first + r;
            let u = self.q[i] / self.x;
            let eta = u.asinh();
            let e = self.xq[i] / self.x;
            for k in 0..=n {
                (p1p[k], p2p[k]) = self.phi.eval12(eta + self.etae[k]);
                (p1m[k], p2m[k]) = self.phi.eval12(eta - self.etae[k]);
            }
            let pr = self.pref / (u * e);
            let mut acc = 0.0;
            for (j, kij_out) in row.iter_mut().enumerate() {
                let hj = self.he[j];
                let m0p = p1p[j + 1] - p1p[j];
                let m1p = 0.5 * hj * (p1p[j + 1] + p1p[j]) - (p2p[j + 1] - p2p[j]);
                let m0m = p1m[j] - p1m[j + 1];
                let m1m = -0.5 * hj * (p1m[j + 1] + p1m[j]) + (p2m[j] - p2m[j + 1]);
                let kij = pr * (self.sj[j] * (m0p - m0m) + self.cj[j] * (m1p - m1m));
                *kij_out = kij;
                acc += kij * self.feq[j];
            }
            *req_i = acc;
        }
    }
}

/// `(Y, y)` from a discretised distribution on the comoving grid (Eqs. 16, 19).
#[derive(Clone, Debug)]
pub(crate) struct GridMoments<'a> {
    pub mass: f64,
    pub dof: f64,
    pub cosmo: &'a Cosmology,
    pub q_hat: Vec<f64>,
    pub vol_hat: Vec<f64>,
    pub heff_start: f64,
}

impl GridMoments<'_> {
    /// `r(x) = [h_eff(T)/h_eff(T_start)]^{1/3}`: physical `q = r q̂`.
    pub(crate) fn scale(&self, x: f64) -> f64 {
        (self.cosmo.heff(self.mass / x) / self.heff_start).cbrt()
    }

    pub(crate) fn moments(&self, x: f64, f: &[f64]) -> (f64, f64) {
        let r = self.scale(x);
        let (mut n0, mut n2) = (0.0, 0.0);
        for ((&qh, &v), &fi) in self.q_hat.iter().zip(&self.vol_hat).zip(f) {
            let q = r * qh;
            n0 += v * fi;
            n2 += v * fi * q * q / (x * x + q * q).sqrt();
        }
        // n = g T³ r³ ∫q̂² f dq̂ /(2π²) and s ∝ h_eff T³ = h_start r³ T³
        let yield_ = 45.0 * self.dof / (4.0 * PI.powi(4) * self.heff_start) * n0;
        let (_, y_eq) = equilibrium(self.mass, self.dof, self.cosmo, x);
        (yield_, y_eq * n2 / (3.0 * n0))
    }
}
