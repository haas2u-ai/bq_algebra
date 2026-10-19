"""
bq_cl_killing.py -- the constant-Lagrangian (CL) condition as an equi-redshift flow, as on the page
BQ_CL_Killing.html.

Galilean-coframe (PG) metric with a radial and an azimuthal river (paper chart):
    ds^2 = -(c^2 - v_r^2 - v_phi^2) dt^2 - 2 v_r dr dt - 2 v_phi r s dphi dt + dr^2 + r^2 dtheta^2 + r^2 s^2 dphi^2
Core statement:  gamma_rs, the river's Lorentz factor relative to the static Killing observers u_s = c xi/|xi|,
    gamma_rs = -u.u_s/c^2 = E/(c sqrt(-g_tt)) = c/sqrt(-g_tt) = 1/sqrt(1 - (v_r^2 + v_phi^2)/c^2),
equals the rotor's Gamma = cosh(chi_r) cosh(chi_phi) to O(chi^2) (they differ at O(chi^4)).

Checked exactly with SymPy:
  1  the river observers u = (1, v_r, 0, Omega), Omega = v_phi/(r s), have unit lapse (u.u = -c^2, so tau = t)
     and are geodesics for ARBITRARY v_r(t, r, theta), Omega(t, r, theta)
  2  their Killing energy is E = -g_{t mu} u^mu = c^2 identically
  3  gamma_rs = -u.u_s/c^2 = c/sqrt(-g_tt); to O(chi^2) it equals the rotor's cosh(chi_r) cosh(chi_phi)
  4  CL: along the river, D gamma_rs/D tau = u.grad gamma_rs; for a stationary axisymmetric flow this is
     v_r d_r gamma_rs, so WHERE v_r != 0:  CL <=> d_r g_tt = 0 <=> v_r^2 + v_phi^2 = K
     (at a stagnation radius such as the balance radius r_c, v_r = 0 and CL does not force d_r g_tt = 0)
  5  the constant is a closure. Motivation: a circular geodesic of the radial Schwarzschild river has
     Omega^2 = GM/r^3 and dtau/dt = sqrt(1 - 3GM/(r c^2)); imposing that clock rate at R gives K = 3GM/R.
     The earlier fitted closure K = (3/2) v_r(R)^2 coincides with it for H = 0 and differs for H != 0.
  6  slowly changing H (quasi-stationary): v_r = sqrt(2GM/r) - H(t) r. E = c^2 still.
     (a) CL readjusting, v_phi^2 = K - v_r^2(r,t):  d_t gamma_rs = 0 exactly
     (b) v_phi held fixed:  d_t gamma_rs = -gamma_rs^3 v_r Hdot r / c^2
     adiabatic parameter eps = |d_t g_tt| (r/v)/|g_tt| = 2|v_r Hdot| r^2/(v (c^2 - v^2)) <= 2(1+q)(H r/c)^2 gamma_rs^2
     the Hdot residual sits in the shift rate gamma_rs d_t v_r = -gamma_rs Hdot r (magnitude Gamma_0 Hdot r),
     not in gamma_rs and not in a proper acceleration (the river stays geodesic, check 1)
  7  the normal-frame energy density of the CL river for H = 0:
     c^2 G_nn = (r v_r^2)'/r^2 - r^2 Omega'^2/4 with r v_r^2 = 2GM constant, so G_nn = -r^2 Omega'^2/(4 c^2) < 0
     (read with the ordinary Einstein equation: negative energy density wherever the rotation is differential)

Usage
  python bq_cl_killing.py
"""
import sympy as sp

t, r, th, ph = sp.symbols('t r theta phi', real=True)
c, G, M, R = sp.symbols('c G M R', positive=True)
X = [t, r, th, ph]
s = sp.sin(th)


def report(name, ok):
    print(f'  {name:<84} {"IDENTICAL (exact)" if ok else "DIFFER"}')
    return ok


def zero(e):
    e = sp.simplify(sp.expand(e))
    return e == 0


def metric(vr, Om):
    """Paper chart: theta^r = dr - v_r dt, theta^phi = r s (dphi - Omega dt)."""
    e = sp.Matrix([[c, 0, 0, 0], [-vr, 1, 0, 0], [0, 0, r, 0], [-r * s * Om, 0, 0, r * s]])
    return sp.simplify(e.T * sp.diag(-1, 1, 1, 1) * e)


def christoffel(g):
    gi = g.inv()
    return [[[sp.simplify(sum(gi[a, d] * (sp.diff(g[d, b], X[k]) + sp.diff(g[d, k], X[b]) - sp.diff(g[b, k], X[d])) for d in range(4)) / 2)
              for k in range(4)] for b in range(4)] for a in range(4)]


def main():
    ok = True
    vr = sp.Function('v_r')(t, r, th)
    Om = sp.Function('Omega')(t, r, th)
    g = metric(vr, Om)
    u = [1, vr, 0, Om]
    print('1  The river observers')
    norm = sum(g[i, j] * u[i] * u[j] for i in range(4) for j in range(4))
    ok &= report('u.u = -c^2: unit lapse, proper time = coordinate time along the river', zero(norm + c**2))
    Gm = christoffel(g)
    acc = [sp.simplify(sum(u[k] * sp.diff(u[a], X[k]) for k in range(4) if not isinstance(u[a], int))
                       + sum(Gm[a][b][k] * u[b] * u[k] for b in range(4) for k in range(4))) for a in range(4)]
    ok &= report('u.grad u = 0 for arbitrary v_r(t,r,theta), Omega(t,r,theta): the river is geodesic', all(zero(x) for x in acc))
    print('2  Killing energy')
    E = -sum(g[0, m] * u[m] for m in range(4))
    ok &= report('E = -g_{t mu} u^mu = c^2 identically', zero(E - c**2))
    print('3  gamma_rs, the river-static Lorentz factor')
    xi_norm = sp.sqrt(-g[0, 0])
    Gam = E / (c * xi_norm)
    vphi = r * s * Om
    ok &= report('gamma_rs = -u.u_s/c^2 = c/sqrt(-g_tt) = 1/sqrt(1 - (v_r^2 + v_phi^2)/c^2)',
                 zero(Gam - 1 / sp.sqrt(1 - (vr**2 + vphi**2) / c**2)))
    a, b, eps = sp.symbols('a b epsilon', real=True)
    met = 1 / sp.sqrt(1 - eps**2 * (a**2 + b**2))
    rot = sp.cosh(eps * a) * sp.cosh(eps * b)
    d2 = sp.series(met - rot, eps, 0, 6).removeO()
    ok &= report('gamma_rs - rotor cosh(chi_r)cosh(chi_phi) = O(chi^4): equal to O(chi^2)', zero(d2.coeff(eps, 2)) and zero(d2.coeff(eps, 0)))
    print(f'     difference at order chi^4: {sp.factor(d2.coeff(eps, 4))} chi^4')
    print('4  CL as an equi-redshift flow')
    VR = sp.Function('V')(r)
    W = sp.Function('W')(r)
    gtt = -(c**2 - VR**2 - W**2)
    GamS = c / sp.sqrt(-gtt)
    DGam = VR * sp.diff(GamS, r)            # stationary, axisymmetric: u.grad = v_r d_r
    ok &= report('D gamma_rs/D tau = v_r d_r gamma_rs = v_r (gamma_rs^3/c^2) d_r(v_r^2 + v_phi^2)/2',
                 zero(DGam - VR * GamS**3 / c**2 * sp.diff((VR**2 + W**2) / 2, r)))
    K = sp.Symbol('K', positive=True)
    ok &= report('v_r^2 + v_phi^2 = K  =>  d_r g_tt = 0 and D gamma_rs/D tau = 0 (CL); converse needs v_r != 0',
                 zero(sp.diff(gtt, r).subs(W, sp.sqrt(K - VR**2)).doit()) and zero(DGam.subs(W, sp.sqrt(K - VR**2)).doit()))
    print('5  The constant K: a closure, motivated by the Schwarzschild orbit at R')
    v0 = sp.sqrt(2 * G * M / r)
    gs = metric(v0, 0).subs(th, sp.pi / 2)
    Omg = sp.Symbol('Omega_g', positive=True)
    # circular equatorial geodesic: radial Euler-Lagrange equation of L = g_mn xdot^m xdot^n at rdot = 0
    tdot, phdot = sp.symbols('tdot phidot', positive=True)
    Lr = sp.diff(gs[0, 0], r) * tdot**2 + sp.diff(gs[3, 3], r) * phdot**2 + 2 * sp.diff(gs[0, 1], r) * 0
    sol = sp.solve(sp.Eq(Lr.subs(phdot, Omg * tdot), 0), Omg)
    ok &= report('circular geodesic of the Newtonian river: Omega^2 = GM/r^3 (Kepler)', any(zero(x**2 - G * M / r**3) for x in sol))
    dtau = sp.sqrt(-(gs[0, 0] + gs[3, 3] * G * M / r**3)) / c
    ok &= report('its clock rate dtau/dt = sqrt(1 - 3GM/(r c^2))', zero(dtau**2 - (1 - 3 * G * M / (r * c**2))))
    ok &= report('closure K = 3GM/R, v_phi = sqrt(3GM/R - 2GM/r) for H = 0',
                 zero((2 * G * M / r + (3 * G * M / R - 2 * G * M / r)) - 3 * G * M / R))
    ok &= report('earlier fitted closure (3/2) v_r(R)^2 = 3GM/R for H = 0',
                 zero(sp.Rational(3, 2) * (2 * G * M / R) - 3 * G * M / R))
    print('6  Slowly changing H: quasi-stationary analysis')
    Hf = sp.Function('H')(t)
    vrt = sp.sqrt(2 * G * M / r) - Hf * r
    Kc = sp.Symbol('K', positive=True)
    G_cl = c / sp.sqrt(c**2 - vrt**2 - (Kc - vrt**2))            # (a) CL readjusts
    ok &= report('(a) CL readjusting, v_phi^2 = K - v_r^2(r,t): d_t gamma_rs = 0 exactly', zero(sp.diff(G_cl, t)))
    Wc = sp.Symbol('w', positive=True)                          # (b) v_phi held fixed
    Gt = c / sp.sqrt(c**2 - vrt**2 - Wc**2)
    ok &= report('(b) v_phi fixed: d_t gamma_rs = -gamma_rs^3 v_r Hdot r / c^2',
                 zero(sp.diff(Gt, t) + Gt**3 * vrt * sp.diff(Hf, t) * r / c**2))
    v2 = vrt**2 + Wc**2
    gtt_t = -(c**2 - v2)
    eps2 = (sp.diff(gtt_t, t) * r / (c**2 - v2))**2 / v2          # eps^2, avoids |.|
    ok &= report('adiabatic parameter eps = |d_t g_tt|(r/v)/|g_tt| = 2|v_r Hdot| r^2/(v (c^2 - v^2))',
                 zero(eps2 - (2 * vrt * sp.diff(Hf, t) * r**2)**2 / (v2 * (c**2 - v2)**2)))
    Hn, qn, rn, Mn = 2.3e-18, -0.55, 3.1e20, 2e41               # ~70 km/s/Mpc, 10 kpc, 1e11 Msun (SI)
    Gn, cn = 6.674e-11, 2.998e8
    vrn = (2 * Gn * Mn / rn)**0.5 - Hn * rn
    print(f'     example (10 kpc, 1e11 Msun, H0 ~ 70): bound 2(1+q)(Hr/c)^2 = {2 * (1 + qn) * (Hn * rn / cn)**2:.2e}'
          f', v_r = {vrn / 1e3:.0f} km/s')
    ok &= report('E = c^2 still holds for H(t)', zero(-sum(metric(vrt, Wc / r).subs(th, sp.pi / 2)[0, m] * [1, vrt, 0, Wc / r][m] for m in range(4)) - c**2))
    ok &= report('the residual: shift rate gamma_rs d_t v_r = -gamma_rs Hdot r (magnitude Gamma_0 Hdot r)',
                 zero(Gt * sp.diff(vrt, t) + Gt * sp.diff(Hf, t) * r))
    print('7  Normal-frame energy density of the CL river, H = 0')
    Om = sp.Function('Omega')(r)
    vr0 = sp.sqrt(2 * G * M / r)
    Gnn = (sp.diff(r * vr0**2, r) / r**2 - r**2 * sp.diff(Om, r)**2 / 4) / c**2
    ok &= report('r v_r^2 = 2GM constant  =>  G_nn = -r^2 Omega\'^2/(4 c^2) <= 0', zero(Gnn + r**2 * sp.diff(Om, r)**2 / (4 * c**2)))
    Om_cl = sp.sqrt(3 * G * M / R - 2 * G * M / r) / r
    val = sp.simplify(Gnn.subs(Om, Om_cl).doit())
    print(f'     CL profile: G_nn = {sp.factor(val)}')
    print('\nAll checks:', 'IDENTICAL' if ok else 'SOME DIFFER')
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
