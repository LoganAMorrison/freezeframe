# Evidence for PR #1

These are the before/after figures for https://github.com/LoganAMorrison/freezeframe/pull/1. The branch only hosts them and is not meant to be merged.

`generator/` is a small crate with a path dependency on a freezeframe checkout. It was built once against `main` (`602095f`) and once against the PR head (`90e5f81`).

- `cargo run --release` prints the tables in the PR comment. Its `scan` lines, joined across the two builds, give `data/scan2.tsv` (family, m, Ωh² on main, Ωh² on the PR, relative shift).
- `cargo run --release --bin curves` prints `data/curves_<build>.tsv`.

`python plot.py data .` renders the three PNGs from those files with matplotlib.
