"""
run_hull.py - analyse a torpedo / AUV hull (Myring profile).

Edit the CONFIG block, run:  python run_hull.py
Outputs: results/hull_cp.png, results/hull_summary.txt, data/potential_cp.csv
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pfhull import myring, solve_panel, drag_estimate, cavitation_inception

# ------------------------------ CONFIG --------------------------------
L = 1.0            # hull length [m]  (use the SAME geometry as your Fluent case)
D = L / 7.0        # max diameter [m]
A_NOSE = 0.20      # nose length [m]
C_TAIL = 0.30      # tail length [m]
N_NOSE = 2.0       # nose shape exponent (2 = elliptical)
THETA = 20.0       # tail half-angle [deg]
U = 5.0            # speed [m/s]
DEPTH = 10.0       # depth for cavitation estimate [m]
RHO, NU = 1025.0, 1.19e-6      # seawater ~15 C
N_PANELS = 200
# ----------------------------------------------------------------------

os.makedirs("results", exist_ok=True)
os.makedirs("data", exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

x, r = myring(L=L, D=D, a=A_NOSE, c=C_TAIL, n_nose=N_NOSE,
              theta_deg=THETA, n=N_PANELS)
sol = solve_panel(x, r, U=U)

Cp = sol["Cp"]
i_min = int(np.argmin(Cp))
cav = cavitation_inception(Cp[i_min], depth=DEPTH, rho=RHO)
drag = drag_estimate(x, r, U, rho=RHO, nu=NU)

lines = [
    f"Hull: L={L} m, D={D:.4f} m (L/D={L/D:.1f}), Myring nose={A_NOSE}, tail={C_TAIL}, "
    f"theta={THETA} deg",
    f"Panels: {N_PANELS}   matrix condition number: {sol['cond']:.2f}",
    f"Net source (should be ~0): {sol['net_source']:.2e}",
    f"Max residual normal velocity / U: {np.max(np.abs(sol['Vn']))/U:.2e}",
    f"Cp_min = {Cp[i_min]:.4f} at x/L = {sol['x'][i_min]/L:.3f}",
    f"Cavitation inception (first order, depth {DEPTH} m): sigma_i = {cav['sigma_i']:.4f}, "
    f"V_inception = {cav['V_inception']:.1f} m/s",
    f"Re_L = {drag['Re']:.3e}   Cf(ITTC-57) = {drag['Cf']:.5f}   "
    f"form factor = {drag['form_factor']:.3f}",
    f"Wetted area = {drag['S_wet']:.4f} m^2",
    f"Drag estimate at U={U} m/s: F = {drag['F']:.2f} N   "
    f"CD(wetted) = {drag['CD_wetted']:.5f}   CD(frontal) = {drag['CD_frontal']:.5f}",
    "Pressure (form) drag from potential flow = 0 (d'Alembert); "
    "drag above is empirical friction + form factor.",
]
print("\n".join(lines))
with open("results/hull_summary.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

# Cp table for later comparison with Fluent
np.savetxt("data/potential_cp.csv",
           np.column_stack([sol["x"] / L, sol["Cp"]]),
           delimiter=",", header="x_over_L,Cp", comments="")

# ------------------------------ PLOTS ---------------------------------
fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                       gridspec_kw={"height_ratios": [1, 2]})
ax[0].fill_between(x / L, -r / L, r / L, color="0.8", edgecolor="k")
ax[0].set(ylabel="r / L", title=f"Myring hull, L/D = {L/D:.1f}")
ax[0].set_aspect("equal", adjustable="datalim")
ax[1].plot(sol["x"] / L, Cp, "b-", lw=1.8, label="Potential flow (ring panels)")
ax[1].plot(sol["x"][i_min] / L, Cp[i_min], "ro",
           label=f"Cp,min = {Cp[i_min]:.3f}")
ax[1].invert_yaxis()
ax[1].set(xlabel="x / L", ylabel="Cp")
ax[1].legend()
plt.tight_layout()
plt.savefig("results/hull_cp.png", dpi=200)
print("Saved results/hull_cp.png, results/hull_summary.txt, data/potential_cp.csv")
