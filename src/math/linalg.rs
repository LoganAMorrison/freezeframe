use std::ops::{Index, IndexMut};

/// Dense, row-major square matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseMatrix {
    n: usize,
    data: Vec<f64>,
}

impl DenseMatrix {
    /// `n × n` zero matrix.
    pub fn zeros(n: usize) -> Self {
        Self {
            n,
            data: vec![0.0; n * n],
        }
    }

    /// Dimension `n`.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Set every entry to zero.
    pub fn fill_zero(&mut self) {
        self.data.fill(0.0);
    }

    /// Row `i` as a slice.
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.n..(i + 1) * self.n]
    }

    /// Row `i` as a mutable slice.
    pub fn row_mut(&mut self, i: usize) -> &mut [f64] {
        &mut self.data[i * self.n..(i + 1) * self.n]
    }

    /// Raw row-major storage.
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Raw row-major storage, mutable.
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// `y = A x`.
    pub fn mul_vec(&self, x: &[f64], y: &mut [f64]) {
        for (yi, row) in y.iter_mut().zip(self.data.chunks_exact(self.n)) {
            *yi = row.iter().zip(x).map(|(a, b)| a * b).sum();
        }
    }
}

impl Index<(usize, usize)> for DenseMatrix {
    type Output = f64;
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.data[i * self.n + j]
    }
}

impl IndexMut<(usize, usize)> for DenseMatrix {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        &mut self.data[i * self.n + j]
    }
}

/// LU factorisation with partial pivoting, `P A = L U`, stored in place.
#[derive(Clone, Debug)]
pub struct LuFactors {
    lu: DenseMatrix,
    pivots: Vec<usize>,
}

impl LuFactors {
    /// Workspace for `n × n` systems.
    pub fn new(n: usize) -> Self {
        Self {
            lu: DenseMatrix::zeros(n),
            pivots: vec![0; n],
        }
    }

    /// Mutable access to the matrix that the next [`factorize`](Self::factorize)
    /// call decomposes.
    pub fn matrix_mut(&mut self) -> &mut DenseMatrix {
        &mut self.lu
    }

    /// Factorise the matrix currently held. Returns `false` when it is
    /// numerically singular.
    pub fn factorize(&mut self) -> bool {
        let n = self.lu.n;
        let a = &mut self.lu.data;
        for k in 0..n {
            let (p, amax) = (k..n)
                .map(|i| (i, a[i * n + k].abs()))
                .fold((k, -1.0), |best, c| if c.1 > best.1 { c } else { best });
            if amax == 0.0 || !amax.is_finite() {
                return false;
            }
            self.pivots[k] = p;
            if p != k {
                for j in 0..n {
                    a.swap(k * n + j, p * n + j);
                }
            }
            let (top, bottom) = a.split_at_mut((k + 1) * n);
            let row_k = &top[k * n..(k + 1) * n];
            let akk = row_k[k];
            for row in bottom.chunks_exact_mut(n) {
                let l = row[k] / akk;
                row[k] = l;
                if l != 0.0 {
                    for (r, &u) in row[k + 1..].iter_mut().zip(&row_k[k + 1..]) {
                        *r -= l * u;
                    }
                }
            }
        }
        true
    }

    /// Solve `A x = b` in place (`b` becomes `x`).
    pub fn solve(&self, b: &mut [f64]) {
        let n = self.lu.n;
        for (k, &p) in self.pivots.iter().enumerate() {
            if p != k {
                b.swap(k, p);
            }
        }
        for i in 0..n {
            let row = self.lu.row(i);
            let s: f64 = row[..i].iter().zip(&b[..i]).map(|(l, x)| l * x).sum();
            b[i] -= s;
        }
        for i in (0..n).rev() {
            let row = self.lu.row(i);
            let s: f64 = row[i + 1..]
                .iter()
                .zip(&b[i + 1..])
                .map(|(u, x)| u * x)
                .sum();
            b[i] = (b[i] - s) / row[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_random_system() {
        let n = 7;
        let mut a = DenseMatrix::zeros(n);
        let mut seed = 12345_u64;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = rnd();
            }
        }
        let x: Vec<f64> = (0..n).map(|i| i as f64 - 2.5).collect();
        let mut b = vec![0.0; n];
        a.mul_vec(&x, &mut b);
        let mut lu = LuFactors::new(n);
        *lu.matrix_mut() = a;
        assert!(lu.factorize());
        lu.solve(&mut b);
        for (u, v) in b.iter().zip(&x) {
            assert!((u - v).abs() < 1e-12);
        }
    }

    #[test]
    fn detects_singular() {
        let mut lu = LuFactors::new(2);
        lu.matrix_mut()[(0, 0)] = 1.0;
        lu.matrix_mut()[(1, 0)] = 2.0;
        assert!(!lu.factorize());
    }
}
