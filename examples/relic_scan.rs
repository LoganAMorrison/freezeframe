//! Relic-density scans.
//!
//! For each DM mass the coupling is fixed by the target Ωh² in the standard
//! treatment (nBE); the coupled (cBE) and full phase-space (fBE) solutions are
//! then computed at the same coupling. With `--find-fbe` the coupling is also
//! re-fitted with the cBE and the fBE.
//!
//! Models:
//! * `ss`: Scalar Singlet, scenario B, coupling λ_S, target 0.1188 (as in arXiv:1706.07433);
//! * `scalar|pseudoscalar|vector|axial`: Dirac DM, mediator mass M, universal coupling
//!   g = g_χ = g_e = g_μ, target 0.12.
//!
//! cargo run --release --example relic_scan -- <model> <M> m1,m2,.. out.tsv [--find-fbe] [--trace DIR]

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use dmphase::prelude::*;
use dmphase::relic::CouplingSolution;
use dmphase::Result;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Family {
    ScalarSinglet,
    Simplified(Mediator),
}

impl Family {
    fn name(self) -> &'static str {
        match self {
            Self::ScalarSinglet => "ss",
            Self::Simplified(m) => m.name(),
        }
    }
    fn target(self) -> f64 {
        match self {
            Self::ScalarSinglet => 0.1188,
            Self::Simplified(_) => 0.12,
        }
    }
    fn guess(self) -> f64 {
        match self {
            Self::ScalarSinglet => 1e-2,
            Self::Simplified(_) => 1e-2,
        }
    }
}

fn build(fam: Family, m: f64, big_m: f64, g: f64) -> Result<Box<dyn DarkMatterModel>> {
    Ok(match fam {
        Family::ScalarSinglet => Box::new(ScalarSinglet::new(m, g, QuarkScattering::Minimal)?),
        Family::Simplified(med) => Box::new(
            SimplifiedModel::builder(med)
                .dm_mass(m)
                .mediator_mass(big_m)
                .dm_coupling(g)
                .sm_coupling(SmFermion::Electron, g)
                .sm_coupling(SmFermion::Muon, g)
                .build()?,
        ),
    })
}

fn width(model: &dyn DarkMatterModel) -> f64 {
    model.resonances().first().map_or(f64::NAN, |r| r.width)
}

fn fit(fam: Family, m: f64, big_m: f64, method: Method, guess: f64) -> Result<CouplingSolution> {
    let slope = if fam == Family::ScalarSinglet {
        -2.0
    } else {
        -3.0
    };
    let search = CouplingSearch {
        initial_slope: slope,
        rel_tol: 5e-4,
        target: fam.target(),
        ..Default::default()
    };
    find_coupling(
        |g| {
            Ok(Solver::new(&build(fam, m, big_m, g)?)
                .solve(method)?
                .omega_h2())
        },
        guess,
        &search,
    )
}

struct Row {
    m: f64,
    line: String,
}

type AnyResult<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn run_point(
    fam: Family,
    m: f64,
    big_m: f64,
    find_fbe: bool,
    trace_dir: Option<&str>,
) -> AnyResult<String> {
    let t0 = Instant::now();
    let gg = fit(fam, m, big_m, Method::NumberDensity, fam.guess())?;
    let model = build(fam, m, big_m, gg.coupling)?;
    let mk = |method: Method| {
        let mut s = Solver::new(&model).trace_log(10.0, 1e3, 200);
        if let Method::PhaseSpace(_) = method {
            s = s.snapshots([12.0, 16.0, 20.0, 25.0, 30.0, 50.0, 100.0, 300.0, 1e3]);
        }
        s.solve(method)
    };
    let nbe = mk(Method::NumberDensity)?;
    let cbe = mk(Method::coupled())?;
    let fbe = mk(Method::phase_space())?;
    if let Some(dir) = trace_dir {
        for (tag, sol) in [("nbe", &nbe), ("cbe", &cbe), ("fbe", &fbe)] {
            let f = File::create(format!("{dir}/trace_{}_{tag}_m{m:.5}.tsv", fam.name()))?;
            sol.write_trace(BufWriter::new(f))?;
        }
        let f = File::create(format!("{dir}/snap_{}_m{m:.5}.tsv", fam.name()))?;
        fbe.write_snapshots(BufWriter::new(f))?;
    }
    let (g_fbe, om_fbe_fit) = if find_fbe {
        let s = fit(fam, m, big_m, Method::phase_space(), gg.coupling)?;
        (s.coupling, s.omega_h2)
    } else {
        (f64::NAN, f64::NAN)
    };
    let g_cbe = if find_fbe {
        fit(fam, m, big_m, Method::coupled(), gg.coupling)?.coupling
    } else {
        f64::NAN
    };
    let dc = cbe.decoupling();
    let df = fbe.decoupling();
    let opt = |v: Option<f64>| v.unwrap_or(f64::NAN);
    Ok(format!(
        "{m:.6}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.4e}\t{:.4e}\t{:.4e}\t{:.4e}\t{:.4e}\t{:.1}",
        gg.coupling,
        width(&*model),
        nbe.omega_h2(),
        cbe.omega_h2(),
        fbe.omega_h2(),
        g_cbe,
        g_fbe,
        om_fbe_fit,
        fbe.final_temp_param() / cbe.final_temp_param(),
        cbe.final_abundance() / nbe.final_abundance(),
        opt(dc.chemical),
        opt(dc.kinetic),
        opt(df.chemical),
        opt(df.kinetic),
        fbe.final_abundance(),
        t0.elapsed().as_secs_f64()
    ))
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let fam = match args.first().map(String::as_str) {
        Some("ss") => Family::ScalarSinglet,
        Some("scalar") => Family::Simplified(Mediator::Scalar),
        Some("pseudoscalar") => Family::Simplified(Mediator::Pseudoscalar),
        Some("vector") => Family::Simplified(Mediator::Vector),
        Some("axial") | Some("axial-vector") => Family::Simplified(Mediator::AxialVector),
        _ => return Err("usage: relic_scan <ss|scalar|pseudoscalar|vector|axial> M m1,m2,.. out.tsv [--find-fbe] [--trace DIR]".into()),
    };
    let big_m: f64 = args[1].parse()?;
    let masses: Vec<f64> = args[2]
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    let out = &args[3];
    let find_fbe = args.iter().any(|a| a == "--find-fbe");
    let trace_dir = args
        .iter()
        .position(|a| a == "--trace")
        .map(|i| args[i + 1].clone());
    if let Some(d) = &trace_dir {
        std::fs::create_dir_all(d)?;
    }
    let workers = std::thread::available_parallelism().map_or(4, |n| n.get());
    let next = AtomicUsize::new(0);
    let rows = Mutex::new(Vec::<Row>::new());
    std::thread::scope(|s| {
        for _ in 0..workers.min(masses.len()) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(&m) = masses.get(i) else { break };
                let line = run_point(fam, m, big_m, find_fbe, trace_dir.as_deref())
                    .unwrap_or_else(|e| format!("{m:.6}\t# error: {e}"));
                eprintln!("{} {line}", fam.name());
                rows.lock().unwrap().push(Row { m, line });
            });
        }
    });
    let mut rows = rows.into_inner()?;
    rows.sort_by(|a, b| a.m.total_cmp(&b.m));
    let mut w = BufWriter::new(File::create(out)?);
    writeln!(
        w,
        "# model={} M={big_m} GeV target Omega h^2 = {}",
        fam.name(),
        fam.target()
    )?;
    writeln!(w, "# m\tg_nBE\tGamma_res\tOmega_nBE\tOmega_cBE\tOmega_fBE\tg_cBE\tg_fBE\tOmega_fBE_fit\ty_fBE/y_cBE\tY_cBE/Y_nBE\tx_cd_cBE\tx_kd_cBE\tx_cd_fBE\tx_kd_fBE\tY_fBE\tseconds")?;
    for r in rows {
        writeln!(w, "{}", r.line)?;
    }
    Ok(())
}
