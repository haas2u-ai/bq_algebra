"""
bq_dirac_mc_connection.py -- the Maurer-Cartan route to isorapidity and CL at the Dirac level (Weyl representation),
as on the page BQ_Dirac_MC_Connection.html. The BQ-Pauli version is bq_mc_connection.py (page BQ_MC_Connection.html).

Pauli level (de Haas 2026, A Spin(1,3) Rapidity Theory for Galactic Rotation Curves, sec. 3):
    G = G_phi G_r,  Omega_tau = (DG/Dtau) G^-1,  U_g Omega_tau = e_T T + e_K.K + e_sigma.sigma,  e_T = (c/2) DGamma/Dtau.
Dirac level (V5 sec. 4-5): the active rotor Q_g = diag(G, (G^dagger)^-1) = Q_g,phi Q_g,r, Q_g,x = exp(+(chi_x/2) alpha_x),
    Omega/_tau = (DQ_g/Dtau) Q_g^-1,  U/_g = Q_g (c beta_0) Q_g^-1,  the transport product U/_g Omega/_tau.
Legs: r, theta, phi = I, J, K; Sigma_theta = Sigma_J = beta_phi beta_r (cyclic); beta~_theta = chi beta_theta.
V5 writes Sigma_theta = alpha_r alpha_phi = beta_r beta_phi = -Sigma_J; its eqs. 71, 81, 82 agree with the results below
once that label is used.

Checked exactly with SymPy (rational in z = exp(chi_r/2), w = exp(chi_phi/2); D acts through Dchi_r, Dchi_phi):
  1  Q_g = ch ch 1 + sh ch alpha_r + ch sh alpha_phi + sh sh Sigma_theta  (V5 eq. 71, with its -Sigma_theta = +Sigma_J)
  2  Q_phi alpha_r Q_phi^-1 = cosh chi_phi alpha_r + sinh chi_phi Sigma_theta  (V5 eq. 81)
  3  Omega/_tau = 1/2 cosh chi_phi Dchi_r alpha_r + 1/2 Dchi_phi alpha_phi + 1/2 sinh chi_phi Dchi_r Sigma_theta  (V5 eq. 82),
     = diag(Omega_tau, -Omega_tau^dagger), the lift of the Pauli connection with sigma_i -> alpha_i, K_i -> -Sigma_i
  4  U/_g Omega/_tau is odd: only vector and pseudovector channels (V5 eqs. 86-89), with
        [ ]_beta0 = -(c/2) DGamma/Dtau, [ ]_beta_r = -(c/2) cosh chi_r Dchi_r,
        [ ]_beta_phi = -(c/2)(Gamma Dchi_phi + sinh chi_r sinh chi_phi Dchi_r),
        [ ]_beta~_theta = -(c/2) sinh chi_r Dchi_phi    <-- V5 eq. 89 has a + sign here (reported, not repaired)
  5  which Pauli product is lifted: the upper block of U/_g Omega/_tau is -U_g Omega_tau^dagger, so the Dirac product
     carries the channels f of U_g Omega_tau^dagger (vector -f_T, -f_K; pseudovector +f_1, +f_sigma). The minquat part of
     U_g Omega_tau^dagger is exactly (1/2) A_g. V5 eqs. 54-55 are these channels; the undaggered U_g Omega_tau has the same
     e_T but e_K,r = (c/2) cosh chi_r cosh 2chi_phi Dchi_r (reported: V5 eqs. 55a-56 belong to U_g Omega_tau^dagger)
  6  grade split: <U/ Omega/>_1 = -(1/2)[Omega/, U/] = -(1/2) A/_g and <U/ Omega/>_3 = (1/2){U/, Omega/}:
     the kinematic acceleration is the vector grade of the transport product, a_0 = -2 [U/ Omega/]_beta0 (Pauli: a_0 = 2 e_T)
  7  unconditional orthogonality: u0 X_0 - u_r X_r - u_phi X_phi = 0 for X = <U/ Omega/>_1;
     u_r X_r + u_phi X_phi = u0 X_0 = -(c^2 Gamma/2) DGamma/Dtau   <-- V5 eqs. 66 and 95 write c/(2 Gamma^2) (reported)
  8  isorapidity: with Dchi_phi = -(sinh chi_r cosh chi_phi)/(cosh chi_r sinh chi_phi) Dchi_r: [ ]_beta0 = 0, the vector part
     is transverse to the flow, the beta~_theta channel (normal to the disk) stays nonzero, and Omega/_tau != 0
  9  slow flow and closure: 2(Gamma - 1) = chi_r^2 + chi_phi^2 + O(chi^4); with chi_phi(R)^2 = chi_r(R)^2/2 the constant is
     v_L^2 = (3/2) v_r(R)^2 and v_phi^2(r) = (3/2) v_r(R)^2 - v_r(r)^2 (V5 eq. 43)

Usage
  python bq_dirac_mc_connection.py
"""
import sympy as sp
from bq_dirac_table import P, blk

Z2 = sp.zeros(2)


def report(name, ok):
    print(f'  {name:<100} {"IDENTICAL (exact)" if ok else "DIFFER"}')
    return ok


def zero(M):
    M = M if isinstance(M, sp.MatrixBase) else sp.Matrix([M])
    return all(sp.cancel(sp.together(sp.expand(x))) == 0 for x in M)


def pauli_dec(M):
    """Real coefficients on 1, T, I, J, K, sI, sJ, sK (the decomposition of the BQ-Pauli pages)."""
    re, im = lambda x: sp.re(sp.expand(x)), lambda x: sp.im(sp.expand(x))
    p, s, u, l = M[0, 0], M[1, 1], M[0, 1], M[1, 0]
    return {'1': (re(p) + re(s)) / 2, 'T': (im(p) + im(s)) / 2, 'I': (im(p) - im(s)) / 2, 'J': (re(u) - re(l)) / 2,
            'K': (im(u) + im(l)) / 2, 'sI': (re(p) - re(s)) / 2, 'sJ': (im(l) - im(u)) / 2, 'sK': (re(u) + re(l)) / 2}


def main():
    ok = True
    c = sp.Symbol('c', positive=True)
    z, w = sp.symbols('z w', positive=True)
    Dr, Dp = sp.symbols('Dchi_r Dchi_phi', real=True)
    one4 = sp.eye(4)
    b0, bI, bJ, bK = [blk(Z2, P['T'], P['T'], Z2)] + [blk(Z2, P[k], -P[k], Z2) for k in 'IJK']
    aI, aJ, aK = b0 * bI, b0 * bJ, b0 * bK
    SI, SJ, SK = bJ * bK, bK * bI, bI * bJ
    chi = -b0 * bI * bJ * bK
    names = ['1', 'b0', 'bI', 'bJ', 'bK', 'aI', 'aJ', 'aK', 'SI', 'SJ', 'SK', 'bt0', 'btI', 'btJ', 'btK', 'chi']
    mats = [one4, b0, bI, bJ, bK, aI, aJ, aK, SI, SJ, SK, chi * b0, chi * bI, chi * bJ, chi * bK, chi]
    coef = lambda X, M: sp.cancel(sp.together(sp.expand((X * M).trace() / (X * X).trace())))
    chans = lambda M: {n: coef(X, M) for n, X in zip(names, mats)}
    D = lambda M: (sp.diff(M, z) * z * Dr / 2 + sp.diff(M, w) * w * Dp / 2)     # D/Dtau through the rapidities

    cr, sr = (z + 1 / z) / 2, (z - 1 / z) / 2                  # cosh, sinh (chi_r/2)
    cp, spp = (w + 1 / w) / 2, (w - 1 / w) / 2                 # cosh, sinh (chi_phi/2)
    C1, S1 = (z**2 + z**-2) / 2, (z**2 - z**-2) / 2            # cosh, sinh chi_r
    C2, S2 = (w**2 + w**-2) / 2, (w**2 - w**-2) / 2            # cosh, sinh chi_phi
    Gam = C1 * C2
    DGam = D(sp.Matrix([Gam]))[0]
    Gr, Gp = cr * P['1'] + sr * P['sI'], cp * P['1'] + spp * P['sK']
    G = Gp * Gr
    Q = blk(G, Z2, Z2, G.H.inv())
    Qi = Q.inv()
    Qp = blk(Gp, Z2, Z2, Gp.inv())
    Sth = SJ                                                   # Sigma_theta, cyclic (r, theta, phi) = (I, J, K)

    print('1  The ordered rotor and its Thomas-Wigner part')
    ok &= report('Q_g = ch ch 1 + sh ch alpha_r + ch sh alpha_phi + sh sh Sigma_theta  (V5 eq. 71, Sigma_theta_V5 = -Sigma_J)',
                 zero(Q - (cr * cp * one4 + sr * cp * aI + cr * spp * aK + sr * spp * Sth)) and zero(Sth + aI * aK))
    ok &= report('Q_phi alpha_r Q_phi^-1 = cosh chi_phi alpha_r + sinh chi_phi Sigma_theta  (V5 eq. 81)',
                 zero(Qp * aI * Qp.inv() - (C2 * aI + S2 * Sth)))

    print('2  The Dirac Maurer-Cartan connection on the flow')
    Om = D(Q) * Qi
    ok &= report('Omega/_tau = 1/2 cosh chi_phi Dchi_r alpha_r + 1/2 Dchi_phi alpha_phi + 1/2 sinh chi_phi Dchi_r Sigma_theta  (V5 eq. 82)',
                 zero(Om - (C2 * Dr / 2 * aI + Dp / 2 * aK + S2 * Dr / 2 * Sth)))
    Omp = D(G) * G.inv()
    ok &= report('= diag(Omega_tau, -Omega_tau^dagger): the lift of the Pauli connection, sigma_i -> alpha_i, K_i -> -Sigma_i',
                 zero(Om - blk(Omp, Z2, Z2, -Omp.H)))
    dO = pauli_dec(Omp)
    lift = dO['sI'] * aI + dO['sJ'] * aJ + dO['sK'] * aK - dO['I'] * SI - dO['J'] * SJ - dO['K'] * SK
    ok &= report('  channel by channel: Pauli (omega_r, omega_phi, omega_theta) on (sigma_r, sigma_phi, J_theta) -> (alpha_r, alpha_phi, -Sigma_theta)',
                 zero(Om - lift) and all(zero(dO[k]) for k in ('1', 'T')))

    print('3  The transport product U/_g Omega/_tau')
    U = Q * (c * b0) * Qi
    ok &= report('U/_g = c Gamma beta_0 + c sinh chi_r beta_r + c cosh chi_r sinh chi_phi beta_phi', zero(U - (c * Gam * b0 + c * S1 * bI + c * C1 * S2 * bK)))
    PO = U * Om
    ch = chans(PO)
    X0, Xr, Xp, Yth = -c / 2 * DGam, -c / 2 * C1 * Dr, -c / 2 * (Gam * Dp + S1 * S2 * Dr), -c / 2 * S1 * Dp
    ok &= report('odd: only vector and pseudovector channels (no 1, alpha, Sigma, chi)',
                 all(zero(ch[n]) for n in ('1', 'aI', 'aJ', 'aK', 'SI', 'SJ', 'SK', 'chi')))
    ok &= report('[ ]_beta0 = -(c/2) DGamma/Dtau  (V5 eq. 91)', zero(ch['b0'] - X0))
    ok &= report('[ ]_beta_r = -(c/2) cosh chi_r Dchi_r,  [ ]_beta_phi = -(c/2)(Gamma Dchi_phi + sinh chi_r sinh chi_phi Dchi_r)  (V5 eqs. 87-88)',
                 zero(ch['bI'] - Xr) and zero(ch['bK'] - Xp) and zero(ch['bJ']))
    ok &= report('[ ]_beta~_theta = -(c/2) sinh chi_r Dchi_phi; all other pseudovector channels zero',
                 zero(ch['btJ'] - Yth) and zero(ch['bt0']) and zero(ch['btI']) and zero(ch['btK']))
    print('     V5 eq. 89 writes +(c/2) sinh chi_r Dchi_phi beta~_theta: the sign differs (reported, not repaired)')

    print('4  Which Pauli product the Dirac product lifts')
    Up = G * (c * P['T']) * G.H
    ok &= report('upper block of U/_g Omega/_tau = -U_g Omega_tau^dagger (since Omega/_tau = diag(Omega_tau, -Omega_tau^dagger))',
                 zero(PO[0:2, 2:4] + Up * Omp.H))
    f = pauli_dec(Up * Omp.H)
    liftF = (-f['T'] * b0 - f['I'] * bI - f['J'] * bJ - f['K'] * bK
             + f['1'] * chi * b0 + f['sI'] * chi * bI + f['sJ'] * chi * bJ + f['sK'] * chi * bK)
    ok &= report('U/_g Omega/_tau = -f_T beta_0 - f_K.beta + f_1 beta~_0 + f_sigma.beta~,  f = channels of U_g Omega_tau^dagger',
                 zero(PO - liftF))
    Ap = D(Up)
    ok &= report('minquat part of U_g Omega_tau^dagger = (1/2) A_g, f_1 = 0: the Pauli face of the grade split',
                 zero(f['1']) and zero((f['T'] * P['T'] + f['I'] * P['I'] + f['J'] * P['J'] + f['K'] * P['K']) - Ap / 2))
    ok &= report('so V5 eqs. 54-55: f_T = (c/2) DGamma, f_r = (c/2) cosh chi_r Dchi_r, f_phi = (c/2)(Gamma Dchi_phi + sinh chi_r sinh chi_phi Dchi_r)',
                 zero(f['T'] - c / 2 * DGam) and zero(f['I'] + Xr) and zero(f['K'] + Xp))
    ok &= report('f_sigma,theta = -(c/2) sinh chi_r Dchi_phi, the only sigma channel (V5 eq. 56 has +)',
                 zero(f['sJ'] - Yth) and zero(f['sI']) and zero(f['sK']))
    e = pauli_dec(Up * Omp)
    ok &= report('the undaggered U_g Omega_tau has the same time channel, e_T = (c/2) DGamma/Dtau (V5 eq. 59)', zero(e['T'] - c / 2 * DGam))
    ok &= report('but e_K,r = (c/2) cosh chi_r cosh 2chi_phi Dchi_r: its K and sigma channels are not V5 eqs. 55-56',
                 zero(e['I'] - c / 2 * C1 * (C2**2 + S2**2) * Dr) and not zero(e['I'] - f['I']))
    print('     V5 eqs. 55a-56 are the channels of U_g Omega_tau^dagger (up to the sigma_theta sign), not of U_g Omega_tau (reported)')

    print('5  The grade split: the kinematic route inside the transport product')
    A = D(U)
    v1 = ch['b0'] * b0 + ch['bI'] * bI + ch['bJ'] * bJ + ch['bK'] * bK
    v3 = PO - v1
    ok &= report('<U/ Omega/>_1 = -(1/2)[Omega/, U/] = -(1/2) A/_g  (A/_g = DU/_g/Dtau, the kinematic route)',
                 zero(v1 + (Om * U - U * Om) / 2) and zero(v1 + A / 2))
    ok &= report('<U/ Omega/>_3 = (1/2){U/, Omega/}', zero(v3 - (U * Om + Om * U) / 2))
    ok &= report('a_0 = -2 [U/ Omega/]_beta0 = c DGamma/Dtau  (Pauli: a_0 = 2 e_T)', zero(coef(b0, A) + 2 * ch['b0']) and zero(coef(b0, A) - c * DGam))

    print('6  Orthogonality, unconditional')
    u0, ur, up = c * Gam, c * S1, c * C1 * S2
    ok &= report('u0 X_0 - u_r X_r - u_phi X_phi = 0 for the vector grade X of U/ Omega/  (V5 eq. 93-94)', zero(u0 * X0 - ur * Xr - up * Xp))
    ok &= report('u_r X_r + u_phi X_phi = -(c^2 Gamma/2) DGamma/Dtau', zero(ur * Xr + up * Xp + c**2 * Gam / 2 * DGam))
    print('     V5 eqs. 66 and 95 write the factor as c/(2 Gamma^2); u0 = c Gamma gives c^2 Gamma/2 (reported, not repaired)')

    print('7  Isorapidity')
    Dp_iso = -(S1 * C2) / (C1 * S2) * Dr
    s = lambda e: sp.cancel(sp.together(e.subs(Dp, Dp_iso)))
    ok &= report('DGamma = 0 gives [ ]_beta0 = 0', zero(s(X0)))
    ok &= report('then u_r X_r + u_phi X_phi = 0: the vector grade is transverse to the flow', zero(s(ur * Xr + up * Xp)))
    ok &= report('the beta~_theta channel stays: -(c/2) sinh chi_r Dchi_phi != 0 for Dchi_r != 0', sp.simplify(s(Yth)) != 0)
    ok &= report('and Omega/_tau != 0: isorapidity is weaker than a flat connection', not zero(Om.subs(Dp, Dp_iso)))

    print('8  Slow flow and the virial closure')
    a, b, t, R_, vR = sp.symbols('chi_r chi_phi t R v_R', positive=True)
    ser = sp.series(sp.cosh(t * a) * sp.cosh(t * b), t, 0, 4).removeO().subs(t, 1)
    ok &= report('2(Gamma - 1) = chi_r^2 + chi_phi^2 + O(chi^4)', sp.simplify(2 * (ser - 1) - (a**2 + b**2)) == 0)
    vL2 = vR**2 + vR**2 / 2
    ok &= report('virial closure chi_phi(R)^2 = chi_r(R)^2/2 gives v_L^2 = (3/2) v_r(R)^2', sp.simplify(vL2 - sp.Rational(3, 2) * vR**2) == 0)
    G_, M_, H_, r_ = sp.symbols('G M H r', positive=True)
    vr = lambda x: sp.sqrt(2 * G_ * M_ / x) - H_ * x
    vphi2 = sp.Rational(3, 2) * vr(R_)**2 - vr(r_)**2
    ok &= report('v_phi^2(r) = (3/2)(sqrt(2GM/R) - HR)^2 - (sqrt(2GM/r) - Hr)^2 conserves v_r^2 + v_phi^2 (V5 eq. 43)',
                 sp.simplify(sp.diff(vphi2 + vr(r_)**2, r_)) == 0)
    print('\nAll checks:', 'IDENTICAL' if ok else 'SOME DIFFER')
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
