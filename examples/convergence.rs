//! Grid/tolerance convergence of the phase-space solution.
//! cargo run --release --example convergence -- <mediator> <M> <m> <g>
use dmphase::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let med = match a[0].as_str() {
        "scalar" => Mediator::Scalar,
        "pseudoscalar" => Mediator::Pseudoscalar,
        "vector" => Mediator::Vector,
        _ => Mediator::AxialVector,
    };
    let (big_m, m, g): (f64, f64, f64) = (a[1].parse()?, a[2].parse()?, a[3].parse()?);
    let model: Box<dyn DarkMatterModel> = if a[0] == "ss" {
        Box::new(ScalarSinglet::new(m, g, QuarkScattering::Minimal)?)
    } else {
        Box::new(
            SimplifiedModel::builder(med)
                .dm_mass(m)
                .mediator_mass(big_m)
                .dm_coupling(g)
                .sm_coupling(SmFermion::Electron, g)
                .sm_coupling(SmFermion::Muon, g)
                .build()?,
        )
    };
    let base = Solver::new(&model).threads(4);
    println!("variant\tOmega\ty_end\tsteps");
    for (label, grid, rtol) in [
        (
            "N150",
            PhaseSpaceGrid {
                cells: 150,
                q_max: None,
            },
            1e-4,
        ),
        ("N300", PhaseSpaceGrid::default(), 1e-4),
        (
            "N600",
            PhaseSpaceGrid {
                cells: 600,
                q_max: None,
            },
            1e-4,
        ),
        ("N300_rtol1e-5", PhaseSpaceGrid::default(), 1e-5),
        ("N300_qmax120", PhaseSpaceGrid::fixed(300, 120.0), 1e-4),
    ] {
        let sol = base.clone().rtol(rtol).solve(Method::PhaseSpace(grid))?;
        println!(
            "{label}\t{:.6e}\t{:.6e}\t{}",
            sol.omega_h2(),
            sol.final_temp_param(),
            sol.stats().steps
        );
    }
    Ok(())
}
