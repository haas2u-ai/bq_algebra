"""
bq_kinematic_route.py -- the kinematic route to isorapidity at the BQ-Pauli level, as on the page
BQ_Kinematic_Route.html (step 11).

Following de Haas (2026), A Spin(1,3) Rapidity Theory for Galactic Rotation Curves (V5), sections 2.2-2.5:

    rapidities chi_r, chi_phi  ->  ordered rotor G = G_phi G_r  ->  U_g = G (c T) G^dagger
    ->  convective four-acceleration A_g = DU_g/Dtau,   D/Dtau = Gamma d_t + u^r d_r  (u^0/c = Gamma)

Orthonormal legs in the order (I, J, K) = (r, theta, phi); the BQ time adjoint is A^T = -a0 T + a.K.

Checked exactly with SymPy, by explicit 2x2 matrix products (rational in p = e^(chi_r/2), s = e^(chi_phi/2)):
  1  U_g = c Gamma T + c sinh chi_r I_r + c cosh chi_r sinh chi_phi K_phi and U_g^T U_g = c^2 1
  2  A_g = DU_g/Dtau is a minquat: a0 T + a_r I_r + a_phi K_phi, with
       a0 = c DGamma/Dtau,  a_r = c cosh chi_r Dchi_r,  a_phi = c (sinh chi_r sinh chi_phi Dchi_r + cosh chi_r cosh chi_phi Dchi_phi)
     (stationary: Dchi = u^r chi', which gives V5 eqs. 14-16)
  3  orthogonality: U_g^T A_g has no 1 channel, u0 a0 - u_r a_r - u_phi a_phi = 0 (V5 eq. 17);
     its remaining channels are the bilinear (u x a).K + (u0 a - a0 u).sigma
  4  power: c a0 = c^2 DGamma/Dtau = (1/m) DE/Dtau with E/m = c^2 Gamma (V5 eq. 24)
  5  the two branches: a0 = 0 <=> u_r a_r + u_phi a_phi = 0 (transverse only); a0 != 0 => longitudinal part
     Newtonian only (chi_phi = 0): a0 = c sinh chi_r Dchi_r, all of the spatial acceleration is along the flow
  6  isorapidity cosh chi_r cosh chi_phi = Gamma_0: a0 = 0 and the spatial acceleration is purely transverse, nonzero
  7  bridge to step 10: A_g = Omega_tau U_g + U_g Omega_tau^dagger exactly, Omega_tau = (DG/Dtau) G^-1,
     so a0 = 2 e_T = 2 [U_g Omega_tau]_T: the kinematic and Maurer-Cartan routes are one statement
  8  the rotation of the orthonormal legs adds (-u_phi^2/r, u_r u_phi/r) to (a_r, a_phi): transverse, no change to a0

Usage
  python bq_kinematic_route.py
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
    D = lambda X: sp.diff(X, P) * P / 2 * Dr + sp.diff(X, Q) * Q / 2 * Dp     # D/Dtau through the rapidities

    print('1  The gravitational four-velocity')
    U = G * (c * BASIS['T']) * G.T
    dU = dec(U)
    ok &= report('U_g = G(cT)G^dagger = c Gamma T + c sinh chi_r I_r + c cosh chi_r sinh chi_phi K_phi',
                 all(zero(dU[k] - v) for k, v in {'1': 0, 'T': c * Gam, 'I_r': c * Sr, 'J_theta': 0,
                                                  'K_phi': c * Cr * Sp, 's_r': 0, 's_theta': 0, 's_phi': 0}.items()))
    adj = lambda d: build({**d, 'T': -d['T']})
    ok &= report('U_g^T U_g = c^2 1', all(zero(x) for x in (adj(dU) * U - c**2 * sp.eye(2))))

    print('2  The convective four-acceleration A_g = DU_g/Dtau')
    A = D(U)
    dA = dec(A)
    a0, ar, ap = c * D(Gam), c * Cr * Dr, c * (Sr * Sp * Dr + Cr * Cp * Dp)
    ok &= report('A_g is a minquat: a0 T + a_r I_r + a_phi K_phi, no other channels',
                 all(zero(dA[k]) for k in ('1', 'J_theta', 's_r', 's_theta', 's_phi')))
    ok &= report('a0 = c DGamma/Dtau  (V5 eq. 14)', zero(dA['T'] - a0))
    ok &= report('a_r = c cosh chi_r Dchi_r,  a_phi = c(sinh chi_r sinh chi_phi Dchi_r + cosh chi_r cosh chi_phi Dchi_phi)',
                 zero(dA['I_r'] - ar) and zero(dA['K_phi'] - ap))

    print('3  Orthogonality')
    UA = dec(adj(dU) * A)
    ok &= report('[U_g^T A_g]_1 = u0 a0 - u_r a_r - u_phi a_phi = 0  (V5 eq. 17)', zero(UA['1']))
    u0, u = dU['T'], sp.Matrix([dU['I_r'], dU['J_theta'], dU['K_phi']])
    av = sp.Matrix([dA['I_r'], dA['J_theta'], dA['K_phi']])
    ux, sg = u.cross(av), u0 * av - dA['T'] * u
    ok &= report('U_g^T A_g = (u x a).K + (u0 a - a0 u).sigma  (the bilinear channels)',
                 zero(UA['T']) and all(zero(UA[k] - ux[n]) for n, k in enumerate(['I_r', 'J_theta', 'K_phi']))
                 and all(zero(UA[k] - sg[n]) for n, k in enumerate(['s_r', 's_theta', 's_phi'])))

    print('4  Power')
    ok &= report('c a0 = c^2 DGamma/Dtau = (1/m) DE/Dtau, E/m = c u0 = c^2 Gamma', zero(c * dA['T'] - D(c * dU['T'])))

    print('5  The two branches')
    ok &= report('u_r a_r + u_phi a_phi = u0 a0: the longitudinal part is fixed by a0', zero(u.dot(av) - u0 * dA['T']))
    ok &= report('Newtonian only: a0 = c sinh chi_r Dchi_r and u.a = u0 a0 (all along the flow)',
                 zero(dA['T'].subs({Q: 1, Dp: 0}) - c * Sr * Dr))

    print('6  Isorapidity')
    Dp_iso = -Sr * Cp / (Cr * Sp) * Dr
    ok &= report('cosh chi_r cosh chi_phi = Gamma_0: a0 = 0 and u_r a_r + u_phi a_phi = 0 (purely transverse)',
                 zero(dA['T'].subs(Dp, Dp_iso)) and zero(u.dot(av).subs(Dp, Dp_iso)))
    at2 = sp.cancel((av.dot(av)).subs(Dp, Dp_iso))
    ok &= report('then a_phi = -c sinh chi_r Dchi_r / sinh chi_phi and the transverse |a|^2 = c^2 Dchi_r^2 (cosh^2 chi_r + sinh^2 chi_r/sinh^2 chi_phi)',
                 zero(dA['K_phi'].subs(Dp, Dp_iso) + c * Sr * Dr / Sp) and zero(at2 - c**2 * Dr**2 * (Cr**2 + Sr**2 / Sp**2)))

    print('7  Bridge to the Maurer-Cartan route (step 10)')
    Om = D(G) * G.inv()
    ok &= report('A_g = Omega_tau U_g + U_g Omega_tau^dagger, exactly', all(zero(x) for x in (Om * U + U * Om.T - A)))
    ok &= report('a0 = 2 e_T = 2 [U_g Omega_tau]_T', zero(dA['T'] - 2 * dec(U * Om)['T']))

    print('8  Rotation of the orthonormal legs')
    r = sp.Symbol('r', positive=True)
    ur, up = dU['I_r'], dU['K_phi']
    ok &= report('leg terms (-u_phi^2/r, +u_r u_phi/r) are transverse: u_r(-u_phi^2/r) + u_phi(u_r u_phi/r) = 0',
                 zero(ur * (-up**2 / r) + up * (ur * up / r)))

    print('\nAll checks:', 'IDENTICAL' if ok else 'SOME DIFFER')
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
