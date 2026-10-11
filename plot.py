"""Before/after figures for the axial-width and resonance-quadrature fixes."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

data, out = Path(sys.argv[1]), Path(sys.argv[2])  # data: curves_*.tsv and scan2.tsv
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BRANCH, MAIN = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "text.usetex": False, "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "DejaVu Sans"], "mathtext.fontset": "dejavusans",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titlecolor": INK, "legend.frameon": False, "legend.labelcolor": INK2,
})


def rows(name, tag):
    return [l.split("\t")[1:] for l in (data / name).read_text().splitlines() if l.startswith(tag)]


# 1. Axial-vector sigma_v above threshold at M = 2m
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True, constrained_layout=True)
for ax, w in zip(axs, ["0.001", "0.1"]):
    for build, col, lab in [("main", MAIN, "main"), ("branch", BRANCH, "this PR")]:
        pts = [(float(u), float(v)) for ww, u, v in rows(f"curves_{build}.tsv", "axial") if ww == w]
        pts = [(u, v) for u, v in pts if v > 0]
        ax.plot(*zip(*pts), color=col, lw=2, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"extra width {float(w):g} GeV", loc="left")
    ax.set_xlabel(r"$s/4m_\chi^2 - 1$")
axs[0].set_ylabel(r"$\sigma v$  [GeV$^{-2}$]")
axs[1].legend(loc="lower right")
fig.suptitle("Axial-vector σv just above threshold with M = 2m: main falls to zero, this PR keeps the contact term",
             color=INK, fontsize=11, x=0.01, ha="left")
fig.savefig(out / "axial_threshold.png", dpi=200)

# 2. pi0-like pole: thermal average relative to the narrow-width limit
fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
for build, col, lab in [("main", MAIN, "main"), ("branch", BRANCH, "this PR")]:
    pts = [(float(x), 100 * (float(r) - 1)) for x, r in rows(f"curves_{build}.tsv", "pion") if float(x) <= 320]
    ax.plot(*zip(*pts), color=col, lw=2, marker="o", ms=4, label=lab)
    ax.annotate(lab, pts[-1], xytext=(6, 0), textcoords="offset points", va="center", color=INK2)
ax.axhline(0, color=BASE, lw=0.8)
ax.set_xscale("log"); ax.set_xlim(9, 420)
ax.set_xlabel(r"$x = m_\chi/T$"); ax.set_ylabel(r"$\langle\sigma v\rangle / \mathrm{NWA} - 1$  [%]")
ax.set_title("π⁰-like pole (Γ/M = 6·10⁻⁸): thermal average relative to the narrow-width limit", loc="left")
ax.legend(loc="center right", bbox_to_anchor=(1, 0.6))
fig.savefig(out / "pion_thermal_average.png", dpi=200)

# 3. Committed scans: nBE relic-density shift, small multiples per mediator
scan = [l.split("\t") for l in (data / "scan2.tsv").read_text().splitlines()]
fig, axs = plt.subplots(2, 2, figsize=(9, 5.4), sharex=True, sharey=True, constrained_layout=True)
for ax, fam in zip(axs.flat, ["scalar", "pseudoscalar", "vector", "axial"]):
    pts = sorted((float(m), 100 * float(r)) for f, m, _, _, r in scan if f == fam)
    ax.axvline(0.5, color=BASE, lw=0.8)
    ax.axhline(0, color=BASE, lw=0.8)
    ax.plot(*zip(*pts), color=BRANCH, lw=2, marker="o", ms=4)
    ax.set_title(fam, loc="left")
for ax in axs[1]:
    ax.set_xlabel(r"$m_\chi$  [GeV]   (M = 1 GeV)")
for ax in axs[:, 0]:
    ax.set_ylabel(r"$\Delta\Omega h^2/\Omega h^2$  [%]")
axs[0, 0].annotate("M = 2m", (0.5, -0.12), xytext=(-4, 0), ha="right", textcoords="offset points", color=MUTED, fontsize=9)
fig.suptitle("Shift in nBE Ωh² at every committed scan point (this PR vs main, same coupling)",
             color=INK, fontsize=11, x=0.01, ha="left")
fig.savefig(out / "scan_relic_shift.png", dpi=200)
