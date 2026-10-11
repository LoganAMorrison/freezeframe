//! Cross-check against the earlier reproduction of arXiv:1706.07433 (scenario B).
use std::time::Instant;

use dmphase::prelude::*;

fn main() -> dmphase::Result<()> {
    // (mass, λ_GG) pairs from the Fig. 3 scan
    let points = [
        (45.0, 1.75197334e-1),
        (57.0, 2.46861325e-3),
        (58.5, 1.26501997e-3),
        (62.5, 2.00140148e-4),
        (62.55, 8.86406212e-4),
    ];
    println!("m\tlambda\tY0_nBE\tY0_cBE\tY0_fBE\tOmega_nBE\tt_build\tt_nBE\tt_cBE\tt_fBE");
    for (m, lam) in points {
        let t0 = Instant::now();
        let model = ScalarSinglet::new(m, lam, QuarkScattering::Minimal)?;
        let tb = t0.elapsed().as_secs_f64();
        let mut row = vec![];
        let mut times = vec![];
        let mut om = 0.0;
        for method in [
            Method::NumberDensity,
            Method::coupled(),
            Method::phase_space(),
        ] {
            let t = Instant::now();
            let sol = Solver::new(&model).solve(method)?;
            times.push(t.elapsed().as_secs_f64());
            row.push(sol.final_abundance());
            if method == Method::NumberDensity {
                om = sol.omega_h2();
            }
        }
        println!(
            "{m}\t{lam:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{om:.5}\t{tb:.2}\t{:.2}\t{:.2}\t{:.2}",
            row[0], row[1], row[2], times[0], times[1], times[2]
        );
    }
    Ok(())
}
