"""Demonstration figures for freezeframe (reads runs/ produced by scripts/run_scans.sh).

python scripts/make_figures.py [RUNS_DIR] [OUT_DIR]
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("figures")
OUT.mkdir(exist_ok=True)

COLS = ["m", "g_nBE", "Gamma", "Om_n", "Om_c", "Om_f", "g_c", "g_f", "Om_f_fit", "y_ratio", "Y_ratio",
        "xcd_c", "xkd_c", "xcd_f", "xkd_f", "Y_f", "sec"]


def load_scan(name):
    a = np.loadtxt(RUNS / f"scan_{name}.tsv", comments="#")
    return {k: a[:, i] for i, k in enumerate(COLS)}


def load_trace(fam, method, m):
    return np.loadtxt(RUNS / "traces" / f"trace_{fam}_{method}_m{m:.5f}.tsv", comments="#")


def load_snaps(fam, m):
    path = RUNS / "traces" / f"snap_{fam}_m{m:.5f}.tsv"
    head = path.read_text().splitlines()[0].lstrip("# ").split("\t")
    xs = [float(h[h.index("=") + 1:-1]) for h in head[1::2]]
    a = np.loadtxt(path, comments="#")
    return {x: (a[:, 2 * k], a[:, 2 * k + 1]) for k, x in enumerate(xs)}


# ----------------------------------------------------------------- colors
METHOD = {"nBE": "#7f7f7f", "cBE": "#4C72B0", "fBE": "#C44E52"}
MED = {  # mediator -> (label, color, marker)
    "pseudoscalar": ("pseudoscalar", "#4C72B0", "o"),
    "vector": ("vector", "#55A868", "s"),
    "axial": ("axial-vector", "#C44E52", "^"),
    "scalar": ("scalar", "#8172B3", "D"),
}
FAMNAME = {"pseudoscalar": "pseudoscalar", "vector": "vector", "axial": "axial-vector", "scalar": "scalar", "ss": "ss"}


def style(ax):
    ax.tick_params(direction="out", length=3, width=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ================================================================ Figure 1: Scalar Singlet validation
ss = load_scan("ss")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)
a1.axhline(1.0, color="0.75", lw=0.6, zorder=0)
a1.plot(ss["m"], ss["Om_c"] / ss["Om_n"], color=METHOD["cBE"], lw=1.3, label="coupled (Y, y)")
a1.plot(ss["m"], ss["Om_f"] / ss["Om_n"], color=METHOD["fBE"], lw=1.0, marker="o", ms=3, label="full phase space")
a1.set_yscale("log")
a1.set_yticks([0.2, 0.5, 1, 2, 5, 10])
a1.set_yticklabels(["0.2", "0.5", "1", "2", "5", "10"])
a1.set_xlabel(r"$m_S$ [GeV]")
a1.set_ylabel(r"$\Omega h^2 \,/\, \Omega h^2_{\rm standard}$")
a1.set_title("Relic density at the standard-treatment coupling", loc="left")
a1.legend(frameon=False, loc="upper left")
a1.margins(x=0.02)
style(a1)

a2.plot(ss["m"], ss["g_nBE"], color=METHOD["nBE"], lw=1.3, label="standard (nBE)")
a2.plot(ss["m"], ss["g_c"], color=METHOD["cBE"], lw=1.3, label="coupled (Y, y)")
a2.plot(ss["m"], ss["g_f"], color=METHOD["fBE"], lw=1.0, marker="o", ms=3, label="full phase space")
a2.set_yscale("log")
a2.set_xlabel(r"$m_S$ [GeV]")
a2.set_ylabel(r"$\lambda_S$ for $\Omega h^2 = 0.1188$")
a2.set_title("Portal coupling required by the relic density", loc="left")
a2.legend(frameon=False, loc="lower left")
a2.margins(x=0.02)
style(a2)
for ax, l in ((a1, "a"), (a2, "b")):
    ax.text(-0.13, 1.06, l, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")
fig.savefig(OUT / "fig1_scalar_singlet_validation.png", dpi=300)
plt.close(fig)

# ================================================================ Figure 2: sub-GeV mediator scan
scans = {k: load_scan(k) for k in MED}
fig, axs = plt.subplots(3, 1, figsize=(4.6, 6.6), sharex=True, constrained_layout=True,
                        gridspec_kw={"height_ratios": [1.25, 1, 0.9]})
for k, (lab, col, mk) in MED.items():
    d = scans[k]
    r = d["m"]  # M = 1 GeV, so m/M = m
    axs[0].plot(r, d["Om_f"] / d["Om_n"], color=col, marker=mk, ms=3, lw=1.0, label=lab, zorder=3)
    axs[0].plot(r, d["Om_c"] / d["Om_n"], color=col, lw=0.8, ls="--", zorder=2)
    axs[1].plot(r, d["g_f"] / d["g_nBE"], color=col, marker=mk, ms=3, lw=1.0, zorder=3)
    axs[2].plot(r, d["xkd_f"], color=col, marker=mk, ms=3, lw=1.0, zorder=3)
xcd = np.mean([scans[k]["xcd_f"] for k in MED], axis=0)
axs[2].plot(scans["vector"]["m"], xcd, color="0.35", lw=1.0, ls=":", zorder=2)
axs[2].text(0.255, 23, "chemical decoupling", color="0.35", fontsize=6, va="bottom")
for ax in axs:
    ax.axvline(0.5, color="0.8", lw=0.6, zorder=0)
    style(ax)
axs[0].axhline(1, color="0.75", lw=0.6, zorder=0)
axs[1].axhline(1, color="0.75", lw=0.6, zorder=0)
axs[0].set_yscale("log")
axs[0].set_yticks([0.5, 1, 2, 5, 10, 20, 50])
axs[0].set_yticklabels(["0.5", "1", "2", "5", "10", "20", "50"])
axs[0].set_ylabel(r"$\Omega h^2\,/\,\Omega h^2_{\rm standard}$")
axs[0].set_title(r"Kinetic decoupling raises $\Omega h^2$ up to $55\times$ below the pole", loc="left")
handles = [Line2D([], [], color=c, marker=mk, ms=3, lw=1.0, label=l) for (l, c, mk) in MED.values()]
handles += [Line2D([], [], color="0.3", lw=1.0, label="full phase space"),
            Line2D([], [], color="0.3", lw=0.8, ls="--", label="coupled (Y, y)")]
axs[0].legend(handles=handles, frameon=False, loc="upper left", ncol=1, fontsize=6)
axs[0].text(0.507, 30, r"$m_\chi = m_{\rm med}/2$" + "\n(pole at threshold)", fontsize=6, color="0.5", va="center")
axs[1].set_yscale("log")
axs[1].set_yticks([1, 2, 5, 10])
axs[1].set_yticklabels(["1", "2", "5", "10"])
axs[1].set_ylabel(r"$g_{\rm full}\,/\,g_{\rm standard}$")
axs[1].set_title(r"Relic coupling $g=g_\chi=g_\ell$ shifts by up to $11\times$", loc="left")
axs[2].set_yscale("log")
axs[2].set_ylabel(r"$x_{\rm kd} = m_\chi/T_{\rm kd}$")
axs[2].set_title("Near the pole, kinetic decoupling coincides with freeze-out", loc="left")
axs[2].set_xlabel(r"$m_\chi / m_{\rm med}$   ($m_{\rm med}$ = 1 GeV)")
axs[0].margins(x=0.02)
for ax, l in zip(axs, "abc"):
    ax.text(-0.16, 1.04, l, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")
fig.savefig(OUT / "fig2_mediator_resonance_scan.png", dpi=300)
plt.close(fig)

# ================================================================ Figure 3: phase space, Y, T_chi
BENCH = [  # (family, mass, title)
    ("ss", 57.0, r"Scalar Singlet, $m_S=57$ GeV"),
    ("pseudoscalar", 0.44, r"pseudoscalar, $m_\chi=0.44\,m_{\rm med}$"),
    ("axial", 0.42, r"axial-vector, $m_\chi=0.42\,m_{\rm med}$"),
    ("vector", 0.4995, r"vector, $m_\chi=0.4995\,m_{\rm med}$"),
]
XSNAP = [20.0, 50.0, 300.0]
SEQ = plt.get_cmap("viridis")(np.linspace(0.05, 0.75, len(XSNAP)))
fig, axs = plt.subplots(3, 4, figsize=(7.2, 5.6), constrained_layout=True)
for j, (fam, m, title) in enumerate(BENCH):
    fn = FAMNAME[fam]
    snaps = load_snaps(fn, m)
    tf = load_trace(fn, "fbe", m)
    tc = load_trace(fn, "cbe", m)
    tn = load_trace(fn, "nbe", m)
    ax = axs[0, j]
    for x, col in zip(XSNAP, SEQ):
        q, f = snaps[x]
        w = q**2 * f
        norm = np.trapezoid(w, q)
        ax.plot(q, w / norm, color=col, lw=1.2, label=f"x = {x:g}")
        # Maxwell-Boltzmann shape at the same T_chi (the coupled-equation ansatz)
        tr = np.interp(np.log(x), np.log(tf[:, 0]), tf[:, 3] / tf[:, 4])
        mb = q**2 * np.exp(-(np.sqrt(x * x + q * q) - x) / tr)
        ax.plot(q, mb / np.trapezoid(mb, q), color=col, lw=0.8, ls="--")
    ax.set_xlim(0, 30 if fam != "ss" else 25)
    ax.set_title(title, loc="left", fontsize=7)
    style(ax)
    ax = axs[1, j]
    ax.plot(tn[:, 0], tn[:, 2], color="0.6", lw=0.8, ls=":", label=r"$Y_{\rm eq}$")
    for lab, t in (("nBE", tn), ("cBE", tc), ("fBE", tf)):
        ax.plot(t[:, 0], t[:, 1], color=METHOD[lab], lw=1.2 if lab != "nBE" else 1.0, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    yend = min(tn[-1, 1], tc[-1, 1], tf[-1, 1])
    ax.set_ylim(yend / 5, tn[0, 1] * 2)
    ax.set_xlim(10, 1e3)
    style(ax)
    ax = axs[2, j]
    ax.plot(tc[:, 0], tc[:, 3] / tc[:, 4], color=METHOD["cBE"], lw=1.2, label="cBE")
    ax.plot(tf[:, 0], tf[:, 3] / tf[:, 4], color=METHOD["fBE"], lw=1.2, label="fBE")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10, 1e3)
    ax.set_xlabel(r"$x = m_\chi/T$")
    style(ax)
axs[0, 0].set_ylabel(r"$q^2 f(q)$, unit area")
axs[1, 0].set_ylabel(r"$Y = n_\chi/s$")
axs[2, 0].set_ylabel(r"$T_\chi/T$")
for j in range(4):
    axs[0, j].set_xlabel(r"$q = p/T$")
axs[0, 0].legend(frameon=False, fontsize=6, loc="upper right",
                 handles=[Line2D([], [], color=c, lw=1.2, label=f"x = {x:g}") for x, c in zip(XSNAP, SEQ)]
                 + [Line2D([], [], color="0.3", lw=0.8, ls="--", label=r"MB at $T_\chi$")])
axs[1, 0].legend(frameon=False, fontsize=6, loc="upper right",
                 handles=[Line2D([], [], color=METHOD["nBE"], lw=1.0, label="standard (nBE)"),
                          Line2D([], [], color=METHOD["cBE"], lw=1.2, label="coupled (cBE)"),
                          Line2D([], [], color=METHOD["fBE"], lw=1.2, label="full (fBE)"),
                          Line2D([], [], color="0.6", lw=0.8, ls=":", label="equilibrium")])
for ax, l in zip(axs[:, 0], "abc"):
    ax.text(-0.42, 1.02, l, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")
fig.savefig(OUT / "fig3_phase_space_benchmarks.png", dpi=300)
plt.close(fig)
print("figures written to", OUT)
