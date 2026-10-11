"""Check the closed-form simplified-mediator matrix elements used in
src/models/simplified.rs against explicit Dirac-trace numerics.

Dirac DM chi (mass m), SM fermion f (mass mf), mediator mass M.
Couplings set to 1, N_c = 1. Run: python scripts/check_matrix_elements.py
"""
import numpy as np

I2, Z = np.eye(2), np.zeros((2, 2))
sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]], complex)
g0 = np.block([[I2, Z], [Z, -I2]]).astype(complex)
G = [g0] + [np.block([[Z, s], [-s, Z]]) for s in (sx, sy, sz)]
g5 = 1j * G[0] @ G[1] @ G[2] @ G[3]
eta = np.diag([1, -1, -1, -1.0])
E4 = np.eye(4)


def sl(p):
    return sum(eta[m, m] * p[m] * G[m] for m in range(4))


def dot(a, b):
    return a @ eta @ b


def mom(m, pvec):
    return np.array([np.sqrt(m * m + pvec @ pvec), *pvec])


def vert(kind):
    return {"S": [E4], "P": [1j * g5], "V": G, "A": [G[m] @ g5 for m in range(4)]}[kind]


def bar(Gm):
    return g0 @ Gm.conj().T @ g0


def msq(kind, chi_proj, f_proj, q, M):
    """|M|^2 numerator: L_chi^{mu a} P_{mu nu} P*_{a b} L_f^{nu b}."""
    V = vert(kind)
    n = len(V)

    def L(Po, Pi):
        return np.array([[np.trace(Po @ V[a] @ Pi @ bar(V[b])) for b in range(n)] for a in range(n)])

    Lc, Lf = L(*chi_proj), L(*f_proj)
    if n == 1:
        return (Lc[0, 0] * Lf[0, 0]).real
    ql = eta @ q
    P = -eta + np.outer(ql, ql) / M**2
    return np.einsum("ma,mn,ab,nb->", Lc, P, P.conj(), Lf).real


m, mf, M = 1.3, 0.4, 2.1
Sig = m * m + mf * mf
rng = np.random.default_rng(1)
worst = 0.0

# t-channel scattering chi(p1) f(k1) -> chi(p2) f(k2), summed over all spins
for kind in "SPVA":
    pc, th = 0.9, rng.uniform(0, np.pi)
    n1, n2 = np.array([0, 0, 1.0]), np.array([np.sin(th), 0, np.cos(th)])
    p1, k1, p2, k2 = mom(m, pc * n1), mom(mf, -pc * n1), mom(m, pc * n2), mom(mf, -pc * n2)
    s, t, u = dot(p1 + k1, p1 + k1), dot(p1 - p2, p1 - p2), dot(p1 - k2, p1 - k2)
    num = msq(kind, (sl(p2) + m * E4, sl(p1) + m * E4), (sl(k2) + mf * E4, sl(k1) + mf * E4), p1 - p2, M)
    closed = {
        "S": 4 * (4 * m * m - t) * (4 * mf * mf - t),
        "P": 4 * t * t,
        "V": 8 * ((s - Sig) ** 2 + (u - Sig) ** 2 + 2 * t * Sig),
        "A": 8 * ((s - Sig) ** 2 + (u - Sig) ** 2 - 2 * t * Sig + 8 * m * m * mf * mf * (1 + (1 - t / M**2) ** 2)),
    }[kind]
    worst = max(worst, abs(num / closed - 1))
    print(f"scattering {kind}: trace {num:.12g}  closed form {closed:.12g}")

# s-channel annihilation, spin averaged and angle averaged (propagator stripped)
s = 7.9
E = np.sqrt(s) / 2
pc, kc = np.sqrt(E * E - m * m), np.sqrt(E * E - mf * mf)
xs, ws = np.polynomial.legendre.leggauss(12)
for kind in "SPVA":
    acc = 0.0
    for c, w in zip(xs, ws):
        n1, n2 = np.array([0, 0, 1.0]), np.array([np.sqrt(1 - c * c), 0, c])
        p1, p2, k1, k2 = mom(m, pc * n1), mom(m, -pc * n1), mom(mf, kc * n2), mom(mf, -kc * n2)
        acc += w / 2 * msq(kind, (sl(p2) - m * E4, sl(p1) + m * E4), (sl(k1) + mf * E4, sl(k2) - mf * E4), p1 + p2, M)
    num = acc / 4
    closed = {
        "S": (s - 4 * m * m) * (s - 4 * mf * mf),
        "P": s * s,
        "V": 4 / 3 * (s + 2 * m * m) * (s + 2 * mf * mf),
        "A": 4 / 3 * (s - 4 * m * m) * (s - 4 * mf * mf) + 16 * m * m * mf * mf * (1 - s / M**2) ** 2,
    }[kind]
    worst = max(worst, abs(num / closed - 1))
    print(f"annihilation {kind}: trace {num:.12g}  closed form {closed:.12g}")

# mediator partial widths into f fbar
Mm = 3.0
kc = np.sqrt(Mm * Mm / 4 - mf * mf)
b = 2 * kc / Mm
k1, k2 = mom(mf, np.array([0, 0, kc])), mom(mf, np.array([0, 0, -kc]))
for kind in "SPVA":
    V = vert(kind)
    if len(V) == 1:
        Ms, npol = np.trace((sl(k1) + mf * E4) @ V[0] @ (sl(k2) - mf * E4) @ bar(V[0])).real, 1
    else:
        L = np.array([[np.trace((sl(k1) + mf * E4) @ V[a] @ (sl(k2) - mf * E4) @ bar(V[c])) for c in range(4)] for a in range(4)])
        Pl = eta @ (k1 + k2)
        Ms, npol = np.einsum("ac,ac->", L, -eta + np.outer(Pl, Pl) / Mm**2).real, 3
    num = kc / (8 * np.pi * Mm * Mm) * Ms / npol
    closed = {
        "S": Mm * b**3 / (8 * np.pi),
        "P": Mm * b / (8 * np.pi),
        "V": Mm * b * (1 + 2 * mf * mf / Mm**2) / (12 * np.pi),
        "A": Mm * b**3 / (12 * np.pi),
    }[kind]
    worst = max(worst, abs(num / closed - 1))
    print(f"width {kind}: trace {num:.12g}  closed form {closed:.12g}")

print(f"max relative deviation: {worst:.2e}")
assert worst < 1e-12
