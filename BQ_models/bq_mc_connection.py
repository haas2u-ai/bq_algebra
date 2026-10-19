"""
bq_mc_connection.py -- the Maurer-Cartan route to isorapidity at the BQ-Pauli level, as on the page
BQ_MC_Connection.html (step 10).

Following de Haas (2026), A Spin(1,3) Rapidity Theory for Galactic Rotation Curves (V5), sections 2-3:

    rapidities chi_r, chi_phi  ->  ordered rotor G = G_phi G_r,  G_i = cosh(chi_i/2) 1 + sinh(chi_i/2) sigma_i
    ->  U_g = G (c T) G^dagger  ->  Omega_tau = (DG/Dtau) G^-1  ->  triple product U_g Omega_tau

Orthonormal spherical legs in the order (I, J, K) = (r, theta, phi): sigma_r = sigma_I, sigma_phi = sigma_K,
and the Thomas-Wigner channel appears on J_theta (V5's K_theta).

Checked exactly with SymPy, by explicit 2x2 matrix products (written rationally in p = e^(chi_r/2), s = e^(chi_phi/2)):
  1  U_g = c Gamma T + c sinh(chi_r) I_r + c cosh(chi_r) sinh(chi_phi) K_phi, Gamma = cosh chi_r cosh chi_phi,
     and U_g^T U_g = c^2 1 (T the BQ time adjoint, -u0 T + u.K)
  2  G_phi sigma_r G_phi^-1 = cosh(chi_phi) sigma_r - sinh(chi_phi) J_theta   (V5 eq. 47)
  3  Omega_tau = 1/2 cosh(chi_phi) Dchi_r sigma_r + 1/2 Dchi_phi sigma_phi - 1/2 sinh(chi_phi) Dchi_r J_theta   (eq. 48)
  4  U_g Omega_tau: e_1 = 0, e_T = u.w_s, e_K = u0 w_s + u x w_K, e_s = u x w_s - u0 w_K   (eq. 52-53)
  5  e_T = (c/2) DGamma/Dtau EXACTLY in the rapidities (no slow-flow limit)
  6  Newtonian only (chi_phi = 0): e_T = (c/2) sinh(chi_r) Dchi_r, never zero for a moving radial river
  7  isorapidity: cosh chi_r cosh chi_phi = Gamma_0 along the flow (chi_phi = arcosh(Gamma_0/cosh chi_r)) gives
     e_T = 0 identically, while e_sigma (the sigma_theta channel) stays nonzero
  8  slow flow and closure: Gamma = 1 + (chi_r^2 + chi_phi^2)/2 + ..., virial chi_phi(R)^2 = chi_r(R)^2/2 gives
     v_L^2 = (3/2) v_r(R)^2 and v_phi^2(r) = (3/2)(sqrt(2GM/R) - HR)^2 - (sqrt(2GM/r) - Hr)^2   (V5 eq. 43);
     for H = 0, v_L^2 = 3GM/R

Usage
  python bq_mc_connection.py
"""
import sympy as sp

c = sp.Symbol('c', positive=True)
# exact rational form: p = exp(chi_r/2), s = exp(chi_phi/2); then every hyperbolic function is rational in p, s
P, Q = sp.symbols('p s', positive=True)
Dr, Dp = sp.symbols('Dchi_r Dchi_phi', real=True)
ch = lambda z: (z + 1 / z) / 2          # cosh(chi/2) for z = exp(chi/2); cosh(chi) = ch(z**2)
sh = lambda z: (z - 1 / z) / 2
i = sp.I
BASIS = {
    '1': sp.eye(2), 'T': i * sp.eye(2),
    'I_r': sp.Matrix([[i, 0], [0, -i]]), 'J_theta': sp.Matrix([[0, 1], [-1, 0]]), 'K_phi': sp.Matrix([[0, i], [i, 0]]),
    's_r': sp.Matrix([[1, 0], [0, -1]]), 's_theta': sp.Matrix([[0, -i], [i, 0]]), 's_phi': sp.Matrix([[0, 1], [1, 0]]),
}
KEYS = list(BASIS)


def report(name, ok):
    print(f'  {name:<86} {"IDENTICAL (exact)" if ok else "DIFFER"}')
    return ok


def zero(e):
    return sp.cancel(sp.together(sp.expand(e))) == 0


def dec(M):
    """Exact decomposition of a 2x2 complex matrix on the eight BQ basis matrices (real coefficients)."""
    p, s, u, l = (sp.expand(M[0, 0]), sp.expand(M[1, 1]), sp.expand(M[0, 1]), sp.expand(M[1, 0]))
    re, im = (lambda z: sp.cancel(sp.re(z))), (lambda z: sp.cancel(sp.im(z)))
    return {'1': re(p + s) / 2, 'T': im(p + s) / 2, 'I_r': im(p - s) / 2, 'J_theta': re(u - l) / 2, 'K_phi': im(u + l) / 2,
            's_r': re(p - s) / 2, 's_theta': im(l - u) / 2, 's_phi': re(u + l) / 2}


def build(d):
    return sum((d.get(k, 0) * BASIS[k] for k in KEYS), sp.zeros(2))


def boost(z, sig):
    return ch(z) * sp.eye(2) + sh(z) * sig


def main():
    ok = True
    Gr, Gp = boost(P, BASIS['s_r']), boost(Q, BASIS['s_phi'])
    G = Gp * Gr
    Cr, Sr, Cp, Sp = ch(P**2), sh(P**2), ch(Q**2), sh(Q**2)      # cosh/sinh of chi_r, chi_phi
    Gam = Cr * Cp

    print('1  The gravitational four-velocity U_g = G (cT) G^dagger')
    U = G * (c * BASIS['T']) * G.T                                 # G is real, so G^dagger = G^T
    dU = dec(U)
    ok &= report('U_g = c Gamma T + c sinh chi_r I_r + c cosh chi_r sinh chi_phi K_phi',
                 all(zero(dU[k] - v) for k, v in {'1': 0, 'T': c * Gam, 'I_r': c * Sr, 'J_theta': 0,
                                                  'K_phi': c * Cr * Sp, 's_r': 0, 's_theta': 0, 's_phi': 0}.items()))
    Ut = build({**dU, 'T': -dU['T']})
    ok &= report('U_g^T U_g = c^2 1  (BQ time adjoint)', all(zero(x) for x in (Ut * U - c**2 * sp.eye(2))))

    print('2  The non-commuting composition')
    adj = Gp * BASIS['s_r'] * Gp.inv()
    ok &= report('G_phi sigma_r G_phi^-1 = cosh chi_phi sigma_r - sinh chi_phi J_theta  (Thomas-Wigner)',
                 all(zero(x) for x in (adj - Cp * BASIS['s_r'] + Sp * BASIS['J_theta'])))

    print('3  The proper-time Maurer-Cartan connection Omega_tau = (DG/Dtau) G^-1')
    DG = sp.diff(G, P) * P / 2 * Dr + sp.diff(G, Q) * Q / 2 * Dp   # d/dchi = (z/2) d/dz
    Om = DG * G.inv()
    dO = dec(Om)
    wr, wp, wt = Cp * Dr / 2, Dp / 2, -Sp * Dr / 2
    ok &= report('Omega_tau = 1/2 cosh chi_phi Dchi_r s_r + 1/2 Dchi_phi s_phi - 1/2 sinh chi_phi Dchi_r J_theta',
                 all(zero(dO[k] - v) for k, v in {'1': 0, 'T': 0, 'I_r': 0, 'J_theta': wt, 'K_phi': 0,
                                                  's_r': wr, 's_theta': 0, 's_phi': wp}.items()))

    print('4  Transport: the triple product U_g Omega_tau')
    dE = dec(U * Om)
    u0, u = dU['T'], sp.Matrix([dU['I_r'], dU['J_theta'], dU['K_phi']])
    ws, wk = sp.Matrix([dO['s_r'], dO['s_theta'], dO['s_phi']]), sp.Matrix([dO['I_r'], dO['J_theta'], dO['K_phi']])
    eT, eK, eS = u.dot(ws), u0 * ws + u.cross(wk), u.cross(ws) - u0 * wk
    ok &= report('e_1 = 0', zero(dE['1']))
    ok &= report('e_T = u.w_sigma', zero(dE['T'] - eT))
    ok &= report('e_K = u0 w_sigma + u x w_K', all(zero(dE[k] - eK[n]) for n, k in enumerate(['I_r', 'J_theta', 'K_phi'])))
    ok &= report('e_sigma = u x w_sigma - u0 w_K', all(zero(dE[k] - eS[n]) for n, k in enumerate(['s_r', 's_theta', 's_phi'])))

    print('5  The time channel is the rate of Gamma')
    DGam = sp.diff(Gam, P) * P / 2 * Dr + sp.diff(Gam, Q) * Q / 2 * Dp
    ok &= report('[U_g Omega_tau]_T = (c/2) DGamma/Dtau, exact in the rapidities', zero(dE['T'] - c / 2 * DGam))

    print('6  Newtonian only')
    ok &= report('chi_phi = 0: e_T = (c/2) sinh chi_r Dchi_r (nonzero for a moving radial river)',
                 zero(dE['T'].subs({Q: 1, Dp: 0}) - c / 2 * Sr * Dr))

    print('7  Isorapidity: cosh chi_r cosh chi_phi = Gamma_0 along the flow')
    Dp_iso = -Sr * Cp / (Cr * Sp) * Dr                            # implicit derivative of Gamma = Gamma_0
    ok &= report('DGamma = 0 fixes Dchi_phi = -tanh chi_r / tanh chi_phi Dchi_r, and then e_T = 0 identically',
                 zero(DGam.subs(Dp, Dp_iso)) and zero(dE['T'].subs(Dp, Dp_iso)))
    es_claim = c * Dr * Cp / 2 * (2 * Cr * Sp + Sr**2 / (Cr * Sp))
    ok &= report('e_sigma,theta left over: (c/2) cosh chi_phi Dchi_r (2 cosh chi_r sinh chi_phi + sinh^2 chi_r/(cosh chi_r sinh chi_phi)) != 0',
                 zero(dE['s_theta'].subs(Dp, Dp_iso) - es_claim))
    r = sp.Symbol('r', positive=True)

    print('8  Slow flow and the virial closure at R')
    eps = sp.Symbol('epsilon', positive=True)
    a, b = sp.symbols('a b', real=True)
    ser = sp.series(sp.cosh(eps * a) * sp.cosh(eps * b), eps, 0, 4).removeO()
    ok &= report('Gamma = 1 + (chi_r^2 + chi_phi^2)/2 + O(chi^4): isorapidity -> v_r^2 + v_phi^2 = v_L^2',
                 zero(ser - 1 - eps**2 * (a**2 + b**2) / 2))
    G_, M, H, R = sp.symbols('G M H R', positive=True)
    vr = lambda x: sp.sqrt(2 * G_ * M / x) - H * x
    vL2 = vr(R)**2 + vr(R)**2 / 2
    ok &= report('virial v_phi(R)^2 = v_r(R)^2/2 gives v_L^2 = (3/2) v_r(R)^2', zero(vL2 - sp.Rational(3, 2) * vr(R)**2))
    ok &= report('v_phi^2(r) = (3/2)(sqrt(2GM/R) - HR)^2 - (sqrt(2GM/r) - Hr)^2   (V5 eq. 43)',
                 zero((vL2 - vr(r)**2) - (sp.Rational(3, 2) * (sp.sqrt(2 * G_ * M / R) - H * R)**2 - (sp.sqrt(2 * G_ * M / r) - H * r)**2)))
    ok &= report('H = 0: v_L^2 = 3GM/R, flat limit sqrt(3) v_K(R)', zero(vL2.subs(H, 0) - 3 * G_ * M / R))
    print(f'     balance radius (v_r = 0): r_c = {sp.solve(sp.Eq(vr(r), 0), r)}')

    print('\nAll checks:', 'IDENTICAL' if ok else 'SOME DIFFER')
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
