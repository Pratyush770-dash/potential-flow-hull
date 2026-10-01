"""
pfhull.py - Potential-flow solver for axisymmetric bodies (torpedo / AUV hulls)

Method : Axisymmetric ring-source panel method (Hess-Smith type) in uniform flow.
         The hull meridian is split into N straight panels; each panel is a ring
         of constant source density. Ring-induced velocities use complete elliptic
         integrals (scipy.special.ellipk / ellipe). Zero-normal-velocity condition
         is enforced at panel midpoints; solution is a well-conditioned
         second-kind integral equation.
Outputs: surface velocity, Cp distribution, Cp_min, cavitation inception
         estimate, and a friction + form-factor drag estimate.

NOTE: potential flow gives ZERO pressure drag (d'Alembert). Drag here is an
empirical estimate: ITTC-57 friction line x Hoerner form factor.

Body coordinates: x along the axis (nose -> tail), r = radius. Flow is +x.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import ellipk, ellipe


# ----------------------------------------------------------------------
# 1. GEOMETRY
# ----------------------------------------------------------------------
def cosine_spacing(x0, x1, n):
    """n+1 points from x0 to x1, clustered at both ends."""
    k = np.arange(n + 1)
    return x0 + 0.5 * (x1 - x0) * (1.0 - np.cos(np.pi * k / n))


def spheroid(a=3.0, b=0.5, n=100):
    """Prolate spheroid, semi-axes a (axial) and b (radial). Sphere if a == b.
    Nose at x = 0. Parametrised by angle so panels are well distributed."""
    phi = np.linspace(0.0, np.pi, n + 1)
    x = a * (1.0 - np.cos(phi))
    r = b * np.sin(phi)
    r[0] = r[-1] = 0.0
    return x, r


def myring(L=1.0, D=1.0 / 7.0, a=0.2, c=0.3, n_nose=2.0, theta_deg=20.0, n=120):
    """Myring hull profile: nose (length a) + cylinder (length b) + tail (length c).
    b = L - a - c. theta_deg = tail half-angle."""
    b = L - a - c
    if b < 0:
        raise ValueError("a + c must be less than L")
    th = np.radians(theta_deg)
    x = cosine_spacing(0.0, L, n)
    r = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi <= a:                                   # nose
            r[i] = 0.5 * D * max(1.0 - ((xi - a) / a) ** 2, 0.0) ** (1.0 / n_nose)
        elif xi <= a + b:                             # parallel mid-body
            r[i] = 0.5 * D
        else:                                         # tail
            s = xi - a - b
            r[i] = (0.5 * D
                    - (3.0 * D / (2.0 * c ** 2) - np.tan(th) / c) * s ** 2
                    + (D / c ** 3 - np.tan(th) / c ** 2) * s ** 3)
    r = np.maximum(r, 0.0)
    r[0] = r[-1] = 0.0
    return x, r


def rankine_body(a=1.0, U=1.0, M=1.0, n=120):
    """Axisymmetric Rankine body: uniform flow + point source at x=-a and equal
    sink at x=+a. M = m/(4*pi). Returns nose->tail coordinates (x, r) plus
    the half-length xs. Body is the psi = 0 stream surface."""
    # stagnation points on the axis (x > a): u(x) = 0
    f_axis = lambda x: U + M * (1.0 / (x + a) ** 2 - 1.0 / (x - a) ** 2)
    xs = brentq(f_axis, a * (1.0 + 1e-6), 1e3 * a)

    def psi(x, r):
        R1 = np.hypot(x + a, r)
        R2 = np.hypot(x - a, r)
        return 0.5 * U * r ** 2 - M * ((x + a) / R1 - (x - a) / R2)

    x = -xs + 2.0 * xs * 0.5 * (1.0 - np.cos(np.pi * np.arange(n + 1) / n))
    r = np.zeros_like(x)
    for i in range(1, n):
        r_lo, r_hi = 1e-4 * a, 50.0 * xs
        r[i] = brentq(lambda rr: psi(x[i], rr), r_lo, r_hi)
    return x, r, xs


def rankine_exact_cp(x, r, a=1.0, U=1.0, M=1.0):
    """Exact Cp at points (x, r) for the Rankine flow (analytical velocity)."""
    R1 = np.hypot(x + a, r)
    R2 = np.hypot(x - a, r)
    u = U + M * (x + a) / R1 ** 3 - M * (x - a) / R2 ** 3
    v = M * r / R1 ** 3 - M * r / R2 ** 3
    return 1.0 - (u ** 2 + v ** 2) / U ** 2


def spheroid_exact_cp(a, b, n_pts=400):
    """Exact Cp for a prolate spheroid (or sphere) in axial flow.
    Surface speed = (1 + k) * U * t_x, with k the axial added-mass coefficient.
    Returns x, Cp arrays (nose at x = 0)."""
    phi = np.linspace(1e-6, np.pi - 1e-6, n_pts)
    x = a * (1.0 - np.cos(phi))
    tx = a * np.sin(phi) / np.hypot(a * np.sin(phi), b * np.cos(phi))
    if abs(a - b) < 1e-12:
        k = 0.5
    else:
        e = np.sqrt(1.0 - (b / a) ** 2)
        alpha0 = 2.0 * (1.0 - e ** 2) / e ** 3 * (0.5 * np.log((1 + e) / (1 - e)) - e)
        k = alpha0 / (2.0 - alpha0)
    return x, 1.0 - ((1.0 + k) * tx) ** 2


# ----------------------------------------------------------------------
# 2. SOLVER  (axisymmetric ring-source panel method)
# ----------------------------------------------------------------------
def _ring_kernels(A, r, rho):
    """Integrals over the azimuth of a unit ring source (radius rho) seen from a
    field point at axial offset A and radius r:
        Iu = int A/R^3 dphi ,  Ir = int (r - rho cos(phi))/R^3 dphi
    Closed forms via complete elliptic integrals."""
    S2 = A ** 2 + (r + rho) ** 2
    S = np.sqrt(S2)
    m = np.clip(4.0 * r * rho / S2, 0.0, 1.0 - 1e-15)
    D2 = A ** 2 + (r - rho) ** 2
    K, E = ellipk(m), ellipe(m)
    Iu = 4.0 * A * E / (S * D2)
    Ir = 4.0 / S ** 3 * ((r + rho) * E / (1.0 - m)
                         - S2 / (2.0 * r) * (E / (1.0 - m) - K))
    return Iu, Ir


def solve_panel(x, r, U=1.0, n_sub=8):
    """Solve potential flow past a body of revolution.

    Parameters
    ----------
    x, r  : node coordinates from nose to tail (r[0] = r[-1] = 0)
    U     : free-stream speed
    n_sub : Gauss points per panel used to integrate the ring sources

    Returns dict: control points (x, r), tangential speed Vt, Cp,
    source strengths sigma, residual normal velocity Vn, net source.
    """
    x = np.asarray(x, float)
    r = np.asarray(r, float)
    N = len(x) - 1

    dx, dr = np.diff(x), np.diff(r)
    ds = np.hypot(dx, dr)
    tx, tr = dx / ds, dr / ds              # unit tangent (nose -> tail)
    nx, nr = -tr, tx                       # outward unit normal
    xc, rc = 0.5 * (x[:-1] + x[1:]), 0.5 * (r[:-1] + r[1:])

    # Gauss points on every source panel
    g, w = np.polynomial.legendre.leggauss(n_sub)
    xs = xc[:, None] + 0.5 * dx[:, None] * g[None, :]          # (N, n_sub)
    rs = rc[:, None] + 0.5 * dr[:, None] * g[None, :]
    wt = 0.5 * ds[:, None] * w[None, :]                        # arc-length weights

    # velocity at control point i due to unit source density on panel j
    Aoff = xc[:, None, None] - xs[None, :, :]                  # (N, N, n_sub)
    Iu, Ir = _ring_kernels(Aoff, rc[:, None, None], rs[None, :, :])
    fac = wt[None, :, :] * rs[None, :, :] / (4.0 * np.pi)
    u_ind = np.sum(fac * Iu, axis=2)
    v_ind = np.sum(fac * Ir, axis=2)

    # normal-velocity influence matrix (+ 1/2 jump term on the diagonal)
    A = u_ind * nx[:, None] + v_ind * nr[:, None] + 0.5 * np.eye(N)
    sigma = np.linalg.solve(A, -U * nx)

    u = U + u_ind @ sigma
    v = v_ind @ sigma
    Vt = u * tx + v * tr                   # surface speed
    Cp = 1.0 - (Vt / U) ** 2
    Vn = u * nx + v * nr + 0.5 * sigma     # residual normal velocity (~0)

    net = float(np.sum(sigma * 2.0 * np.pi * rc * ds))
    return dict(x=xc, r=rc, ds=ds, Vt=Vt, Vn=Vn, Cp=Cp, sigma=sigma,
                cond=np.linalg.cond(A), net_source=net)


# ----------------------------------------------------------------------
# 3. DRAG AND CAVITATION ESTIMATES
# ----------------------------------------------------------------------
def wetted_area(x, r):
    """Surface area of the body of revolution (trapezoidal on the panels)."""
    ds = np.hypot(np.diff(x), np.diff(r))
    rm = 0.5 * (r[:-1] + r[1:])
    return float(np.sum(2.0 * np.pi * rm * ds))


def drag_estimate(x, r, U, rho=1025.0, nu=1.19e-6):
    """Friction (ITTC-57) x Hoerner form factor. Defaults: seawater ~15 C.
    Returns dict of Re, Cf, form factor, drag force and drag coefficients."""
    L = x[-1] - x[0]
    D = 2.0 * np.max(r)
    S = wetted_area(x, r)
    Re = U * L / nu
    Cf = 0.075 / (np.log10(Re) - 2.0) ** 2
    ff = 1.0 + 1.5 * (D / L) ** 1.5 + 7.0 * (D / L) ** 3
    q_dyn = 0.5 * rho * U ** 2
    F = q_dyn * S * Cf * ff
    A_front = np.pi * D ** 2 / 4.0
    return dict(Re=Re, Cf=Cf, form_factor=ff, S_wet=S, F=F,
                CD_wetted=F / (q_dyn * S), CD_frontal=F / (q_dyn * A_front))


def cavitation_inception(Cp_min, depth=10.0, rho=1025.0, p_atm=101325.0,
                         p_vap=1700.0, g=9.81):
    """First-order inception estimate: sigma_i ~ -Cp_min.
    Returns inception number and inception speed at the given depth (m)."""
    p_inf = p_atm + rho * g * depth
    sigma_i = -Cp_min
    V_i = np.sqrt(2.0 * (p_inf - p_vap) / (rho * sigma_i)) if sigma_i > 0 else np.inf
    return dict(sigma_i=sigma_i, V_inception=V_i, p_inf=p_inf)
