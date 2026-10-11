//! Self-contained numerical building blocks: Gauss–Legendre quadrature,
//! exponentially scaled Bessel functions, cubic splines and dense LU
//! factorization. The crate deliberately has no external dependencies.

mod bessel;
mod linalg;
mod quadrature;
mod spline;

pub use bessel::bessel_k_scaled;
pub use linalg::{DenseMatrix, LuFactors};
pub use quadrature::{gauss_legendre, GaussLegendre};
pub use spline::{CubicSpline, Extrapolation};
