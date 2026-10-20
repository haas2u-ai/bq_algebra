"""
bq_dirac_kinematic_route.py -- the kinematic route to isorapidity at the Dirac level (Weyl representation), as on the
page BQ_Dirac_Kinematic_Route.html. The BQ-Pauli version is bq_kinematic_route.py (page BQ_Kinematic_Route.html).

Pauli level (de Haas 2026, A Spin(1,3) Rapidity Theory for Galactic Rotation Curves, sec. 2.2-2.5):
    G = G_phi G_r (G_r = exp(chi_r sigma_I/2), G_phi = exp(chi_phi sigma_K/2)),  U_g = G (c T) G^dagger,
    A_g = DU_g/Dtau,  Omega_tau = (DG/Dtau) G^-1,  A_g = Omega_tau U_g + U_g Omega_tau^dagger.
Dirac level: the active rotor (rest frame -> river) is the general boost rule diag(M, (M^dagger)^-1) with M = G,
    Q_g = diag(G, (G^dagger)^-1) = Q_g,phi Q_g,r,  Q_g,x = diag(G_x, G_x^-1) = exp(+(chi_x/2) alpha_x)
(the coordinate rotor of the gravity-rotor page with the opposite rapidities: active instead of passive).

Checked exactly with SymPy (rational in z = exp(chi_r/2), w = exp(chi_phi/2); D acts through Dchi_r, Dchi_phi):
  1  Q_g = diag(G, (G^dagger)^-1) = Q_g,phi Q_g,r, Q_g,x = cosh(chi_x/2) 1 + sinh(chi_x/2) alpha_x
  2  U/_g = Q_g (c beta_0) Q_g^-1 = [[0, U_g], [-U_g^T, 0]] with the Pauli U_g in its upper block;
     U/_g = c Gamma beta_0 + c sinh chi_r beta_r + c cosh chi_r sinh chi_phi beta_phi, Gamma = cosh chi_r cosh chi_phi;
     U/_g^2 = -c^2 1
  3  the Dirac Maurer-Cartan form Omega/_tau = (DQ_g) Q_g^-1 = diag(Omega_tau, -Omega_tau^dagger): an even element,
     bivector channels only (alpha and Sigma)
  4  one commutator: A/_g = DU/_g = [Omega/_tau, U/_g], whose upper block is the Pauli bridge Omega_tau U_g + U_g Omega_tau^dagger;
     A/_g = a0 beta_0 + a_r beta_r + a_phi beta_phi with a0 = c DGamma/Dtau, a_r = c cosh chi_r Dchi_r,
     a_phi = c (sinh chi_r sinh chi_phi Dchi_r + cosh chi_r cosh chi_phi Dchi_phi)
  5  orthogonality in one line: {U/_g, A/_g} = D(U/_g^2) = 0, i.e. u0 a0 - u_r a_r - u_phi a_phi = 0
  6  U/_g A/_g is then a pure bivector (no 1, vector, pseudovector or chi channel); its lower block is -U_g^T A_g, the
     Pauli bilinear, so the Pauli statement 'no 1 channel in U_g^T A_g' sits inside it
  7  isorapidity: with Dchi_phi = -(sinh chi_r cosh chi_phi)/(cosh chi_r sinh chi_phi) Dchi_r (DGamma = 0): a0 = 0,
     the spatial acceleration is orthogonal to the spatial velocity and nonzero
  8  power: c a0 = c^2 DGamma/Dtau, the rate of the composed Lorentz factor

Usage
  python bq_dirac_kinematic_route.py
"""
import sympy as sp
from bq_dirac_table import P, blk

Z2 = sp.zeros(2)


def report(name, ok):
    print(f'  {name:<96} {"IDENTICAL (exact)" if ok else "DIFFER"}')
    return ok


def zero(M):
    M = M if isinstance(M, sp.MatrixBase) else sp.Matrix([M])
    return all(sp.cancel(sp.together(sp.expand(x))) == 0 for x in M)


def main():
    ok = True
    c = sp.Symbol('c', positive=True)
    z, w = sp.symbols('z w', positive=True)
    Dr, Dp = sp.symbols('Dchi_r Dchi_phi', real=True)
    one4 = sp.eye(4)
    b0, bI, bJ, bK = [blk(Z2, P['T'], P['T'], Z2)] + [blk(Z2, P[k], -P[k], Z2) for k in 'IJK']
    aI, aK = b0 * bI, b0 * bK
    chi = -b0 * bI * bJ * bK
    names = ['1', 'b0', 'bI', 'bJ', 'bK', 'aI', 'aJ', 'aK', 'SI', 'SJ', 'SK', 'bt0', 'btI', 'btJ', 'btK', 'chi']
    mats = [one4, b0, bI, bJ, bK, aI, b0 * bJ, aK, bJ * bK, bK * bI, bI * bJ, chi * b0, chi * bI, chi * bJ, chi * bK, chi]
    coef = lambda X, M: sp.cancel(sp.together(sp.expand((X * M).trace() / (X * X).trace())))
    chans = lambda M: {n: coef(X, M) for n, X in zip(names, mats)}
    D = lambda M: (sp.diff(M, z) * z * Dr / 2 + sp.diff(M, w) * w * Dp / 2)     # D/Dtau through the rapidities

    cr, sr = (z + 1 / z) / 2, (z - 1 / z) / 2
    cp, spp = (w + 1 / w) / 2, (w - 1 / w) / 2
    Gr, Gp = cr * P['1'] + sr * P['sI'], cp * P['1'] + spp * P['sK']
    G = Gp * Gr
    Gd = G.H
    Q = blk(G, Z2, Z2, Gd.inv())
    Qi = Q.inv()
    C1, S1 = (z**2 + z**-2) / 2, (z**2 - z**-2) / 2
    C2, S2 = (w**2 + w**-2) / 2, (w**2 - w**-2) / 2
    Gam = C1 * C2

    print('1  The active Dirac rotor')
    Qr, Qp = blk(Gr, Z2, Z2, Gr.inv()), blk(Gp, Z2, Z2, Gp.inv())
    ok &= report('Q_g = diag(G, (G^dagger)^-1) = Q_g,phi Q_g,r, Q_g,x = cosh(chi_x/2) 1 + sinh(chi_x/2) alpha_x',
                 zero(Q - Qp * Qr) and zero(Qr - (cr * one4 + sr * aI)) and zero(Qp - (cp * one4 + spp * aK)))
    print('2  The four-velocity')
    U = Q * (c * b0) * Qi
    Up = G * (c * P['T']) * Gd
    UpT = (-c * Gam) * P['T'] + c * S1 * P['I'] + c * C1 * S2 * P['K']
    ok &= report('U/_g = [[0, U_g], [-U_g^T, 0]] with the Pauli U_g = G (c T) G^dagger', zero(U - blk(Z2, Up, -UpT, Z2)))
    ok &= report('U/_g = c Gamma beta_0 + c sinh chi_r beta_r + c cosh chi_r sinh chi_phi beta_phi, U/_g^2 = -c^2 1',
                 zero(U - (c * Gam * b0 + c * S1 * bI + c * C1 * S2 * bK)) and zero(U * U + c**2 * one4))
    print('3  The Dirac Maurer-Cartan form')
    Om = D(Q) * Qi
    Omp = D(G) * G.inv()
    ok &= report('Omega/_tau = (DQ_g) Q_g^-1 = diag(Omega_tau, -Omega_tau^dagger), Omega_tau = (DG) G^-1', zero(Om - blk(Omp, Z2, Z2, -Omp.H)))
    ch = chans(Om)
    ok &= report('Omega/_tau is even: only the alpha and Sigma channels', all(zero(ch[n]) for n in names if n not in ('aI', 'aJ', 'aK', 'SI', 'SJ', 'SK')))
    print('4  The acceleration as one commutator')
    A = D(U)
    ok &= report('A/_g = DU/_g = [Omega/_tau, U/_g]', zero(A - (Om * U - U * Om)))
    ok &= report('upper block = Omega_tau U_g + U_g Omega_tau^dagger (the Pauli bridge)', zero(A[0:2, 2:4] - (Omp * Up + Up * Omp.H)))
    a0 = c * (S1 * C2 * Dr + C1 * S2 * Dp)
    ar = c * C1 * Dr
    ap = c * (S1 * S2 * Dr + C1 * C2 * Dp)
    ok &= report('A/_g = a0 beta_0 + a_r beta_r + a_phi beta_phi, a0 = c DGamma/Dtau, a_r, a_phi as in V5 eqs. 14-16',
                 zero(A - (a0 * b0 + ar * bI + ap * bK)) and zero(a0 - c * D(sp.Matrix([Gam]))[0]))
    print('5  Orthogonality')
    ok &= report('{U/_g, A/_g} = D(U/_g^2) = 0:  u0 a0 - u_r a_r - u_phi a_phi = 0', zero(U * A + A * U))
    print('6  The product U/_g A/_g')
    pa = chans(U * A)
    ok &= report('U/_g A/_g is a pure bivector (alpha and Sigma only)', all(zero(pa[n]) for n in names if n not in ('aI', 'aJ', 'aK', 'SI', 'SJ', 'SK')))
    Ap = A[0:2, 2:4]
    ApT = -A[2:4, 0:2]
    ok &= report('its lower block is -U_g^T A_g, the Pauli bilinear (without a 1 channel)', zero((U * A)[2:4, 2:4] + UpT * Ap)
                 and zero(sp.Matrix([((UpT * Ap).trace()) / 2])))
    print('7  Isorapidity')
    Dp_iso = -(S1 * C2) / (C1 * S2) * Dr
    a0i, ari, api = [sp.cancel(sp.together(e.subs(Dp, Dp_iso))) for e in (a0, ar, ap)]
    ok &= report('DGamma = 0 gives a0 = 0', zero(a0i))
    ok &= report('then u_r a_r + u_phi a_phi = 0: the spatial acceleration is transverse', zero(c * S1 * ari + c * C1 * S2 * api))
    ok &= report('and nonzero for Dchi_r != 0 (a_r = c cosh chi_r Dchi_r)', sp.simplify(ari) != 0)
    print('8  Power')
    ok &= report('c a0 = c^2 DGamma/Dtau', zero(c * a0 - c**2 * D(sp.Matrix([Gam]))[0]))
    print('\nAll checks:', 'IDENTICAL' if ok else 'SOME DIFFER')
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
