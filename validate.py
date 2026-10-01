"""
validate.py - verify the solver against exact solutions before trusting it.

Tests: (1) sphere, (2) prolate spheroid L/D = 6, (3) Rankine body,
       (4) convergence with panel count.
Saves figures to results/.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pfhull import (spheroid, spheroid_exact_cp, rankine_body,
                    rankine_exact_cp, solve_panel)

os.makedirs("results", exist_ok=True)
os.makedirs("data", exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def rms_error(x_num, cp_num, x_ex, cp_ex, trim=0.0, L=None):
    """RMS error of numerical Cp vs exact Cp (interpolated to numerical points).
    trim = fraction of body length excluded at each end (end-panel effects)."""
    L = L if L is not None else x_num.max() - x_num.min()
    m = (x_num > x_num.min() + trim * L) & (x_num < x_num.max() - trim * L)
    cp_ref = np.interp(x_num[m], x_ex, cp_ex)
    return np.sqrt(np.mean((cp_num[m] - cp_ref) ** 2))


def test_sphere(N=100):
    R0 = 1.0
    x, r = spheroid(R0, R0, N)
    s = solve_panel(x, r)
    xe, cpe = spheroid_exact_cp(R0, R0)
    err = rms_error(s["x"], s["Cp"], xe, cpe, trim=0.02)
    print(f"Sphere      N={N}: Cp_min num={s['Cp'].min():.4f}  exact=-1.2500  "
          f"RMS err={err:.4e}  cond={s['cond']:.1e}")
    return s, (xe, cpe)


def test_spheroid(N=100, a=3.0, b=0.5):
    x, r = spheroid(a, b, N)
    s = solve_panel(x, r)
    xe, cpe = spheroid_exact_cp(a, b)
    err = rms_error(s["x"], s["Cp"], xe, cpe, trim=0.02)
    print(f"Spheroid L/D={2*a/(2*b):.0f} N={N}: Cp_min num={s['Cp'].min():.4f}  "
          f"exact={cpe.min():.4f}  RMS err={err:.4e}  cond={s['cond']:.1e}")
    return s, (xe, cpe)


def test_rankine(N=100, M=1.0):
    x, r, xs = rankine_body(a=1.0, U=1.0, M=M, n=N)
    s = solve_panel(x, r)
    cp_ex = rankine_exact_cp(s["x"], s["r"], a=1.0, U=1.0, M=M)
    err = np.sqrt(np.mean((s["Cp"][2:-2] - cp_ex[2:-2]) ** 2))
    print(f"Rankine     N={N}: half-length={xs:.3f}  max radius={r.max():.3f}  "
          f"Cp_min num={s['Cp'].min():.4f}  exact={cp_ex.min():.4f}  "
          f"RMS err={err:.4e}")
    return s, cp_ex, (x, r)


def convergence():
    Ns = [20, 40, 80, 160, 320]
    e_sph, e_spd = [], []
    for N in Ns:
        x, r = spheroid(1.0, 1.0, N)
        s = solve_panel(x, r)
        xe, cpe = spheroid_exact_cp(1.0, 1.0)
        e_sph.append(rms_error(s["x"], s["Cp"], xe, cpe, trim=0.02))
        x, r = spheroid(3.0, 0.5, N)
        s = solve_panel(x, r)
        xe, cpe = spheroid_exact_cp(3.0, 0.5)
        e_spd.append(rms_error(s["x"], s["Cp"], xe, cpe, trim=0.02))
    return Ns, e_sph, e_spd


if __name__ == "__main__":
    print("=== Validation of the ring-source panel solver ===")
    s1, ex1 = test_sphere()
    s2, ex2 = test_spheroid()
    s3, cp3, (xb, rb) = test_rankine()
    Ns, e_sph, e_spd = convergence()
    print("\nConvergence (RMS Cp error, ends trimmed 2%):")
    for N, a, b in zip(Ns, e_sph, e_spd):
        print(f"  N={N:4d}   sphere={a:.3e}   spheroid={b:.3e}")

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].plot(ex1[0], ex1[1], "k-", label="Exact")
    ax[0, 0].plot(s1["x"], s1["Cp"], "r.", ms=4, label="Solver")
    ax[0, 0].set(title="Sphere", xlabel="x / R", ylabel="Cp")
    ax[0, 1].plot(ex2[0] / 3.0, ex2[1], "k-", label="Exact")
    ax[0, 1].plot(s2["x"] / 3.0, s2["Cp"], "r.", ms=4, label="Solver")
    ax[0, 1].set(title="Prolate spheroid, L/D = 6", xlabel="x / a", ylabel="Cp")
    xs = xb.max()
    ax[1, 0].plot(s3["x"] / xs, cp3, "k-", label="Exact")
    ax[1, 0].plot(s3["x"] / xs, s3["Cp"], "r.", ms=4, label="Solver")
    ax[1, 0].set(title="Rankine body", xlabel="x / x_stag", ylabel="Cp")
    ax[1, 1].loglog(Ns, e_sph, "o-", label="Sphere")
    ax[1, 1].loglog(Ns, e_spd, "s-", label="Spheroid L/D=6")
    ax[1, 1].set(title="Convergence", xlabel="Number of panels N",
                 ylabel="RMS Cp error")
    for a_ in ax.ravel():
        a_.legend()
    for a_ in ax.ravel()[:3]:
        a_.invert_yaxis()
    plt.tight_layout()
    plt.savefig("results/validation.png", dpi=200)
    print("\nSaved results/validation.png")
