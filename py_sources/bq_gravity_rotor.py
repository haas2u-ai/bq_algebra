"""
bq_gravity_rotor.py -- the BQ gravity rotor at the BQ-Pauli level, as on the page BQ_Gravity_Rotor.html.

Rapidity potentials (de Haas 2026, The Gravitational Rapidity Potential):
    psi_N = sqrt(2GM/r)/c,  psi_H = H r/c,  chi_r = psi_N - psi_H,  chi_phi = v_phi/c
feed the local rotor G = exp(chi sigma/2), which acts on the line element
    dR = c dt T^ + dr I^_r + r dtheta J^_theta + r s dphi K^_phi        (s = sin theta)
by the double-sided adjoint action, radial boost first:
    dR^G = G_phi^-1 (G_r^-1 dR G_r^-1) G_phi^-1 ,  G_r = exp(chi_r sigma_I/2),  G_phi = exp(chi_phi sigma_K/2).

Checked exactly with SymPy:
  1  the rotor: exp(chi sigma/2) = cosh(chi/2) 1^ + sinh(chi/2) sigma
  2  dR^G is a pure minquat with the stated channels (step 3 of the page)
  3  the exact quadratic is flat: (dR^G)^T dR^G = dR^T dR
  4  the single product G^-1 dR G^-1 with G = G_phi G_r leaves the minquat space (order chi_r chi_phi)
  5  the Galilean coframe: slow-flow power counting of the one-forms gives theta^0 = c(dt - v dr/c^2) ~ c dt and
       theta^r = dr - v_r dt (the Galilean limit t' = t, x' = x - vt of the boost):
       theta^0 = c dt, theta^r = dr - v_r dt, theta^theta = r dtheta, theta^phi = r s dphi - v_phi dt.
       In the quadratic the dropped term becomes 2 v dr dt, the same size as the kept gravitational terms; keeping it,
       the cross terms cancel and the result is flat. The gravitational content is in the choice of a Galilean coframe
       with a Lorentzian quadratic; the script shows the cancellation and the term the Galilean coframe drops.
  6  the coframe quadratic (theta)^T theta = -ds^2 1^ and g_munu = eta_ab e^a_mu e^b_nu, det g = -c^2 r^4 s^2
  7  comparisons: Schwarzschild (PG to Schwarzschild time), de Sitter and expanding FLRW (r = a chi),
     Schwarzschild-de Sitter (cross term -2 c^2 psi_N psi_H), rigid rotation (phi' = phi - Omega t removes the shift)
  8  the Einstein tensor of the prescribed metric in the river frame (equatorial plane, constant H), for general
     u(r) and Omega(r): Schwarzschild is vacuum, de Sitter is vacuum + Lambda, Newton + Hubble needs the source
     -3H sqrt(2GM)/(c^2 r^(3/2)) of the cross term, rigid rotation adds nothing, differential rotation adds
     -r^2 Omega'^2/4 to the energy density and the flux (r^4 Omega')'/(2 c r^3)
Both time orientations are checked: the paper chart (rotor rapidity chi, theta^r = dr - v dt, eqs. 46, 51) and the
ingoing/expanding chart (rotor rapidity -chi, theta^r = dr + v dt).

Usage
  python bq_gravity_rotor.py            all checks, printed step by step
"""
import sympy as sp
from bq_core import BASIS, SIGMA, decompose

c, G, M, H, r, th, Om = sp.symbols('c G M H r theta Omega', positive=True)
dt, dr, dth, dph, a, b, eps = sp.symbols('dt dr dtheta dphi a b epsilon', real=True)
ONE, T, I, J, K = BASIS['1'], BASIS['T'], BASIS['I'], BASIS['J'], BASIS['K']
sI, sK = SIGMA[0], SIGMA[2]
s = sp.sin(th)


def rotor(sig, x):
    return sp.cosh(x / 2) * ONE + sp.sinh(x / 2) * sig


def zero(e):
    e = sp.expand(sp.sympify(e).rewrite(sp.exp))
    return e == 0 or sp.simplify(e) == 0


def report(name, ok):
    print(f'  {name:<72} {"IDENTICAL (exact)" if ok else "DIFFER"}')
    return ok


def minquat(c0, c1, c2, c3):
    return c0 * T + c1 * I + c2 * J + c3 * K


def channels(Mx):
    d = decompose(Mx.applyfunc(sp.expand))
    return d['1'], d['T'], d['K'], d['s']


def main():
    ok = True
    dR = minquat(c * dt, dr, r * dth, r * s * dph)
    print('1  The rotor')
    ok &= report('exp(chi sigma_I/2) = cosh(chi/2) 1^ + sinh(chi/2) sigma_I = diag(e^(chi/2), e^(-chi/2))',
                 all(zero(e) for e in rotor(sI, a) - sp.diag(sp.exp(a / 2), sp.exp(-a / 2))))
    ok &= report('exp(chi sigma_K/2) = cosh(chi/2) 1^ + sinh(chi/2) sigma_K', all(zero(e) for e in rotor(sK, b) - (sp.Matrix(sp.exp(b / 2 * sK)))))

    for sgn, chart in ((1, 'paper chart: rotor rapidity chi'), (-1, 'ingoing/expanding chart: rotor rapidity -chi')):
        print(f'\n=== {chart} ===')
        Gri, Gpi = rotor(sI, -sgn * a), rotor(sK, -sgn * b)          # a = chi_r, b = chi_phi
        dRG = Gpi * (Gri * dR * Gri) * Gpi
        one, t0, k, sg = channels(dRG)
        Cr, Sr, Cp, Sp = sp.cosh(a), sgn * sp.sinh(a), sp.cosh(b), sgn * sp.sinh(b)
        T1 = Cr * c * dt - Sr * dr
        exact = [Cp * T1 - Sp * r * s * dph, Cr * dr - Sr * c * dt, r * dth, Cp * r * s * dph - Sp * T1]
        print('2  dR^G = G_phi^-1 (G_r^-1 dR G_r^-1) G_phi^-1')
        for nm, v in zip(['T^', 'I^_r', 'J^_theta', 'K^_phi'], exact):
            print(f'     {nm:<9}: {v}')
        ok &= report('matrices = these channels, and no 1^ or sigma channel (a pure minquat)',
                     all(zero(p - q) for p, q in zip([t0] + k, exact)) and zero(one) and all(zero(v) for v in sg))
        print('3  The exact quadratic')
        Q = minquat(-t0, *k) * dRG
        q1, qT, qK, qs = channels(Q)
        flat = c**2 * dt**2 - dr**2 - r**2 * dth**2 - r**2 * s**2 * dph**2
        ok &= report('(dR^G)^T dR^G = (c^2 dt^2 - dr^2 - r^2 dtheta^2 - r^2 s^2 dphi^2) 1^: flat, the rotor cancels',
                     zero(q1 - flat) and zero(qT) and all(zero(v) for v in qK + qs))
        if sgn == 1:
            print('4  The single product G^-1 dR G^-1 with G = G_phi G_r')
            Gi = rotor(sI, -a) * rotor(sK, -b)
            p1, _, _, ps = channels(Gi * dR * Gi)
            leak = [sp.simplify(sp.series(v.rewrite(sp.exp).subs({a: eps * a, b: eps * b}), eps, 0, 3).removeO().subs(eps, 1)) for v in [p1] + ps]
            print(f'     1^ and sigma channels to second order: {leak}')
            print('     -> nonzero at order chi_r chi_phi: the product of two boosts is not a pure boost, so the action is nested')
        print('5  The Galilean (PG) coframe: where gravity enters')
        lin = [sp.series(sp.sympify(v).subs({a: eps * a, b: eps * b}), eps, 0, 2).removeO().subs(eps, 1) for v in exact]
        quad_lin = sp.expand(-lin[0]**2 + lin[1]**2 + lin[2]**2 + lin[3]**2)
        flat_ds2 = -c**2 * dt**2 + dr**2 + r**2 * dth**2 + r**2 * s**2 * dph**2
        dev = sp.expand(sp.series(quad_lin.subs({a: eps * a, b: eps * b}), eps, 0, 2).removeO().subs(eps, 1) - flat_ds2)
        print(f'     first order of the exact image: theta^0 = {sp.expand(lin[0])}, theta^r = {sp.expand(lin[1])}')
        ok &= report('its quadratic equals the flat ds^2 to first order in chi: the cross terms cancel', zero(dev))
        vr, vp = c * a, c * b
        coframe = [c * dt, dr - sgn * vr * dt, r * dth, r * s * dph - sgn * vp * dt]
        dropped = sp.expand(lin[0] - coframe[0])
        print(f'     the Galilean coframe drops {dropped} from theta^0 (clock form c dt, unit lapse) and keeps the spatial legs')
        ok &= report(f'Galilean spatial legs = first-order exact legs: theta^r = dr {"-" if sgn == 1 else "+"} v_r dt, '
                     f'theta^phi = r s dphi {"-" if sgn == 1 else "+"} v_phi dt',
                     zero(sp.expand(lin[1] - coframe[1]).subs(b, 0)) and zero(sp.expand(lin[3] - coframe[3]).subs(a, 0)))
        quad_pg = sp.expand(-coframe[0]**2 + coframe[1]**2 + coframe[2]**2 + coframe[3]**2)
        print(f'     Lorentzian quadratic of the Galilean coframe minus flat: {sp.factor(quad_pg - flat_ds2)}  <- the gravitational content')
        print('6  The coframe quadratic and g_munu')
        th_ = minquat(*coframe)
        c1, cT, cK, cs = channels(minquat(-coframe[0], *coframe[1:]) * th_)
        e = sp.Matrix([[c, 0, 0, 0], [-sgn * vr, 1, 0, 0], [0, 0, r, 0], [-sgn * vp, 0, 0, r * s]])
        eta = sp.diag(-1, 1, 1, 1)
        g = sp.simplify(e.T * eta * e)
        dx = sp.Matrix([dt, dr, dth, dph])
        ds2 = sp.expand((dx.T * g * dx)[0])
        print('     g_munu =', g.tolist())
        ok &= report('(theta)^T theta = -ds^2 1^ with ds^2 = g_munu dx^mu dx^nu', zero(c1 + ds2) and zero(cT) and all(zero(v) for v in cK + cs))
        ok &= report('det g = -c^2 r^4 sin^2 theta (unit lapse)', zero(g.det() + c**2 * r**4 * s**2))
        print('7  Comparison with known metrics (radial part)')
        v = sp.Symbol('v', real=True)
        gr = sp.Matrix([[-(c**2 - v**2), -sgn * v], [-sgn * v, 1]])
        # dt = dT + f dr with f = -sgn v/(c^2 - v^2): the cross term drops out
        f = -sgn * v / (c**2 - v**2)
        Jm = sp.Matrix([[1, f], [0, 1]])
        gd = sp.simplify(Jm.T * gr * Jm)
        ok &= report('dT = dt + (sgn) v dr/(c^2 - v^2) diagonalises: -(c^2 - v^2) dT^2 + c^2 dr^2/(c^2 - v^2)',
                     zero(gd[0, 1]) and zero(gd[0, 0] + (c**2 - v**2)) and zero(gd[1, 1] - c**2 / (c**2 - v**2)))
        rs = 2 * G * M / c**2
        ok &= report('v = sqrt(2GM/r): g_tt = -c^2(1 - r_s/r), g_rr -> 1/(1 - r_s/r): Schwarzschild',
                     zero(gd[0, 0].subs(v, sp.sqrt(2 * G * M / r)) + c**2 * (1 - rs / r)) and zero(gd[1, 1].subs(v, sp.sqrt(2 * G * M / r)) - 1 / (1 - rs / r)))
        ok &= report('v = -Hr: g_tt = -c^2(1 - H^2 r^2/c^2): static de Sitter, Lambda = 3H^2/c^2',
                     zero(gd[0, 0].subs(v, -H * r) + c**2 * (1 - H**2 * r**2 / c**2)))
        vNH = sp.sqrt(2 * G * M / r) - H * r
        sds = -c**2 * (1 - rs / r - H**2 * r**2 / c**2)
        ok &= report('v = sqrt(2GM/r) - Hr: g_tt - g_tt(Schwarzschild-de Sitter) = -2c^2 psi_N psi_H = -2H sqrt(2GMr)',
                     zero(-(c**2 - vNH**2) - sds + 2 * H * sp.sqrt(2 * G * M * r)))
        # FLRW: theta^r = dr - sgn*v dt with v = -Hr; r = A(t) x, H = A'/A
        tt, x = sp.symbols('t x', positive=True)
        A = sp.Function('A', positive=True)(tt)
        rr = A * x
        Hs = sp.diff(A, tt) / A
        dx_ = sp.Symbol('dx', real=True)
        thr = (sp.diff(rr, tt) * dt + A * dx_) - sgn * (-Hs * rr) * dt
        flrw = zero(thr - A * dx_)
        if sgn == -1:
            ok &= report('r = a(t) chi, H = a\'/a: theta^r = dr - Hr dt = a dchi: expanding flat FLRW', flrw)
        else:
            print(f'     r = a(t) chi with H = a\'/a gives theta^r = a dchi + 2Hr dt (not FLRW): this chart is the time reverse;'
                  f' with H -> -H it is FLRW  ({"as expected" if not flrw else "unexpected"})')
        # rigid rotation: theta^phi = r s (dphi - sgn Omega dt); phi' = phi - sgn Omega t removes the shift
        dphi_p = sp.Symbol('dphi_p', real=True)
        thp = r * s * dph - sgn * Om * r * dt
        ok &= report('v_phi = Omega r (theta = pi/2): phi\' = phi - (sgn) Omega t gives theta^phi = r dphi\': flat (Langevin)',
                     zero(thp.subs({th: sp.pi / 2, dph: dphi_p + sgn * Om * dt}) - r * dphi_p))
    ok &= einstein_checks()
    print('\nAll checks:', 'IDENTICAL' if ok else 'SOME DIFFER')
    return ok


def einstein_tensor(g, X):
    gi = g.inv()
    n = 4
    Gam = [[[sp.simplify(sum(gi[i, d] * (sp.diff(g[d, j], X[k]) + sp.diff(g[d, k], X[j]) - sp.diff(g[j, k], X[d])) for d in range(n)) / 2)
             for k in range(n)] for j in range(n)] for i in range(n)]
    Ric = sp.zeros(n)
    for j in range(n):
        for k in range(n):
            Ric[j, k] = sp.simplify(sum(sp.diff(Gam[i][j][k], X[i]) - sp.diff(Gam[i][j][i], X[k])
                                        + sum(Gam[i][i][d] * Gam[d][j][k] - Gam[i][k][d] * Gam[d][j][i] for d in range(n)) for i in range(n)))
    R = sp.simplify(sum(gi[i, j] * Ric[i, j] for i in range(n) for j in range(n)))
    return sp.simplify(Ric - R * g / 2)


def einstein_checks():
    print('\n8  The Einstein tensor of the prescribed metric (river frame, equatorial plane, static: constant H)')
    t, ph = sp.symbols('t phi', real=True)
    X = [t, r, th, ph]
    u, W = sp.Function('u')(r), sp.Function('W')(r)          # theta^r = dr - u dt, theta^phi = r s dphi - s W dt, W = r Omega
    e = sp.Matrix([[c, 0, 0, 0], [-u, 1, 0, 0], [0, 0, r, 0], [-W * s, 0, 0, r * s]])
    g = sp.simplify(e.T * sp.diag(-1, 1, 1, 1) * e)
    G = einstein_tensor(g, X)
    ei = e.inv()
    Gh = sp.simplify((ei.T * G * ei).subs(th, sp.pi / 2))
    Om = sp.Function('Omega')(r)
    Gh = sp.simplify(Gh.subs(W, r * Om).doit())
    F = r * u**2
    Op = sp.diff(Om, r)
    J = sp.diff(r**4 * Op, r)
    expect = {(0, 0): (sp.diff(F, r) / r**2 - r**2 * Op**2 / 4) / c**2, (1, 1): -(sp.diff(F, r) / r**2 - r**2 * Op**2 / 4) / c**2,
              (2, 2): (-sp.diff(F, r, 2) / (2 * r) - r**2 * Op**2 / 4) / c**2, (3, 3): (-sp.diff(F, r, 2) / (2 * r) - 3 * r**2 * Op**2 / 4) / c**2,
              (0, 3): J / (2 * c * r**3), (1, 3): u * J / (2 * c**2 * r**3)}
    ok = True
    for (i, j), v in expect.items():
        ok &= zero(sp.expand(Gh[i, j] - v))
    others = [(i, j) for i in range(4) for j in range(i, 4) if (i, j) not in expect]
    ok &= all(zero(Gh[i, j]) for i, j in others)
    ok = report('c^2 G00 = -c^2 Grr = (r u^2)\'/r^2 - r^2 Om\'^2/4, c^2 Gthth = -(r u^2)\'\'/(2r) - r^2 Om\'^2/4, '
                'c^2 Gphph = -(r u^2)\'\'/(2r) - 3 r^2 Om\'^2/4, c G0ph = (r^4 Om\')\'/(2 r^3), c^2 Grph = u (r^4 Om\')\'/(2 r^3)', ok)
    sub = lambda uu, oo: {k: sp.simplify(v.subs({u: uu, Om: oo}).doit()) for k, v in expect.items()}
    sch = sub(sp.sqrt(2 * G_ * M_ / r), 0)
    ok &= report('u = sqrt(2GM/r): all components vanish (Schwarzschild is vacuum)', all(zero(v) for v in sch.values()))
    for sg_ in (1, -1):
        ds = sub(sg_ * (-H * r), 0)
        Lam = 3 * H**2 / c**2
        ok &= report(f'u = {"-" if sg_ == 1 else "+"}Hr: G = -Lambda eta, Lambda = 3H^2/c^2 (de Sitter, both charts)',
                     zero(ds[(0, 0)] - Lam) and zero(ds[(1, 1)] + Lam) and zero(ds[(2, 2)] + Lam) and zero(ds[(3, 3)] + Lam))
    nh = sub(sp.sqrt(2 * G_ * M_ / r) - H * r, 0)
    Lam = 3 * H**2 / c**2
    cross = -3 * H * sp.sqrt(2 * G_ * M_) / (c**2 * r**sp.Rational(3, 2))
    ok &= report('u = sqrt(2GM/r) - Hr: G00 - Lambda = -3H sqrt(2GM)/(c^2 r^(3/2)), Gthth + Lambda = -(1/4)(that): not vacuum + Lambda',
                 zero(nh[(0, 0)] - Lam - cross) and zero(nh[(1, 1)] + Lam + cross) and zero(nh[(2, 2)] + Lam + cross / 4))
    rig = sub(0, Om_c)
    ok &= report('rigid rotation, Omega constant: all components vanish', all(zero(v) for v in rig.values()))
    kep = sub(0, sp.sqrt(G_ * M_ / r**3))
    ok &= report('Keplerian Omega: G00 = -(9/16) GM/(c^2 r^3), c G0ph = -(9/8) sqrt(GM) r^(-5/2): not vacuum',
                 zero(kep[(0, 0)] + sp.Rational(9, 16) * G_ * M_ / (c**2 * r**3)) and zero(kep[(0, 3)] + sp.Rational(9, 8) * sp.sqrt(G_ * M_) / (c * r**sp.Rational(5, 2))))
    return ok


G_, M_ = G, M
Om_c = sp.Symbol('Omega_0', positive=True)


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
