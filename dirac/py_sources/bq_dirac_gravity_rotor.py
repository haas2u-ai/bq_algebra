"""
bq_dirac_gravity_rotor.py -- the BQ gravity rotor lifted to the Dirac level (Weyl representation), as on the page
BQ_Dirac_Gravity_Rotor.html. The BQ-Pauli version is bq_gravity_rotor.py.

Pauli level: G_r = exp(chi_r sigma_I/2), G_phi = exp(chi_phi sigma_K/2), dR^G = G_phi^-1 (G_r^-1 dR G_r^-1) G_phi^-1,
             dR = c dt T^ + dr I^_r + r dtheta J^_theta + r s dphi K^_phi.
Dirac level: the same rotor, doubled exactly as the global boost of the Lorentz models,
    Q_g,r = diag(G_r^-1, G_r) = exp(-(chi_r/2) alpha_I) = exp((chi_r/2) sigma_I^W),
    Q_g,phi = diag(G_phi^-1, G_phi) = exp(-(chi_phi/2) alpha_K),
    dR/ = c dt beta_0 + dr beta_r + r dtheta beta_theta + r s dphi beta_phi = [[0, dR], [-dR^T, 0]],
    dR/_g = Q_g,phi (Q_g,r dR/ Q_g,r^-1) Q_g,phi^-1 = Q_g dR/ Q_g^-1,  Q_g = Q_g,phi Q_g,r  (the exact image).
Q_g,r = exp(-(1/2) chi_r beta_0 beta_r) as in BQ Dirac Gravity, Sec. 4.3. The Galilean limit of dR/_g is theta/
(the paper's dR_g of Sec. 4.1); dR/_g here is the exact image (the paper's A/^G, eq. 124).

Checked exactly with SymPy (rational in z = exp(chi_r/2), w = exp(chi_phi/2)); both time orientations (rotor rapidity
chi and -chi):
  1  the rotors: Q_g,r = diag(G_r^-1, G_r) = exp(-(chi_r/2) alpha_I) = cosh(chi_r/2) 1 - sinh(chi_r/2) alpha_I;
     Q_g,phi likewise with alpha_K; the nested action is the single rotor Q_g = Q_g,phi Q_g,r = diag(G_phi^-1 G_r^-1, G_phi G_r)
  2  dR/_g = [[0, dR^G], [-(dR^G)^T, 0]]: its upper block is the BQ-Pauli image G_phi^-1(G_r^-1 dR G_r^-1)G_phi^-1 and
     its lower block the time adjoint; dR/_g is a pure vector, with (gamma = cosh chi, v = c tanh chi)
     beta_0: g_phi[g_r(c dt - v_r dr/c) - (v_phi/c) r s dphi], beta_r: g_r(dr - v_r dt), beta_theta: r dtheta,
     beta_phi: g_phi[r s dphi - (v_phi/c) g_r(c dt - v_r dr/c)]
  3  the exact square is flat: dR/_g^2 = dR/^2 = (-c^2 dt^2 + dr^2 + r^2 dtheta^2 + r^2 s^2 dphi^2) 1
  4  the basis route with both rotors: beta^_a = Q_g beta_a Q_g^-1,
     beta^_0 = g_r g_phi (beta_0 - (v_phi/c) beta_phi) - g_r (v_r/c) beta_r,
     beta^_r = g_r beta_r - g_r (v_r/c) g_phi (beta_0 - (v_phi/c) beta_phi), beta^_theta = beta_theta,
     beta^_phi = g_phi (beta_phi - (v_phi/c) beta_0); {beta^_a, beta^_b} = 2 eta_ab 1 (eta = diag(-1, 1, 1, 1));
     dR/_g = c dt beta^_0 + dr beta^_r + r dtheta beta^_theta + r s dphi beta^_phi, the same matrix as the coordinate route.
     BQ Dirac Gravity eqs. 146-150 write beta~_0 = cosh psi beta_0 + sinh psi beta_r for Q_g = exp(-(1/2) psi beta_0 beta_r);
     in these matrices that rotor gives the minus sign, the plus sign belongs to rotor rapidity -psi (reported, not changed)
  5  the coframe as a Dirac vector: theta/ = c dt beta_0 + (dr - v_r dt) beta_r + r dtheta beta_theta + (r s dphi - v_phi dt) beta_phi;
     theta/ theta/ = ds^2 1 with ds^2 = -c^2 dt^2 + (dr - v_r dt)^2 + r^2 dtheta^2 + (r s dphi - v_phi dt)^2;
     in blocks theta/theta/ = [[-theta theta^T, 0], [0, -theta^T theta]], so the lower block is the Pauli quadratic -theta^T theta
  6  coordinate by coordinate: dR/_g = dR/_g,mu dx^mu with dR/_g,t = c beta^_0, dR/_g,r = beta^_r, dR/_g,theta = r beta^_theta,
     dR/_g,phi = r s beta^_phi, and {dR/_g,mu, dR/_g,nu} = 2 g0_mu_nu 1 with the flat g0 = diag(-c^2, 1, r^2, r^2 s^2);
     theta/ = theta/_mu dx^mu with theta/_mu = theta^a_mu beta_a and {theta/_mu, theta/_nu} = 2 g_mu_nu 1,
     g = PG with the azimuthal shift, det g = det g0 = -c^2 r^4 s^2 (BQ Dirac Gravity eqs. 173, 191-192)
  7  Dirac representation: S Q_g,r S^-1 = [[c 1, s sigma_I], [s sigma_I, c 1]] (c, s of chi_r/2), and
     the anticommutators, hence g_mu_nu, are the same in both representations

Usage
  python bq_dirac_gravity_rotor.py
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
    c, r, th = sp.symbols('c r theta', positive=True)
    dt, dr, dth, dph, vr, vp = sp.symbols('dt dr dtheta dphi v_r v_phi', real=True)
    z, w = sp.symbols('z w', positive=True)
    s = sp.sin(th)
    one4 = sp.eye(4)
    b = [blk(Z2, P['T'], P['T'], Z2)] + [blk(Z2, P[k], -P[k], Z2) for k in 'IJK']
    aI, aK = b[0] * b[1], b[0] * b[3]
    vec = lambda q: sum((q[k] * b[k] for k in range(4)), sp.zeros(4))
    pauli = lambda q: q[0] * P['T'] + q[1] * P['I'] + q[2] * P['J'] + q[3] * P['K']
    adjT = lambda q: pauli([-q[0], q[1], q[2], q[3]])
    S = blk(P['1'], P['1'], -P['1'], P['1']) / sp.sqrt(2)
    eta = sp.diag(-1, 1, 1, 1)

    for sgn, chart in ((1, 'paper chart: rotor rapidity chi'), (-1, 'ingoing / expanding chart: rotor rapidity -chi')):
        print(f'\n=== {chart} ===')
        Z, W = (z, w) if sgn == 1 else (1 / z, 1 / w)                  # exp(sgn chi/2)
        cr, sr = (Z + 1 / Z) / 2, (Z - 1 / Z) / 2                       # cosh, sinh of (sgn chi_r)/2
        cp, sp_ = (W + 1 / W) / 2, (W - 1 / W) / 2
        Gr, Gri = cr * P['1'] + sr * P['sI'], cr * P['1'] - sr * P['sI']
        Gp, Gpi = cp * P['1'] + sp_ * P['sK'], cp * P['1'] - sp_ * P['sK']
        Lr, Lri = blk(Gri, Z2, Z2, Gr), blk(Gr, Z2, Z2, Gri)
        Lp, Lpi = blk(Gpi, Z2, Z2, Gp), blk(Gp, Z2, Z2, Gpi)

        print('1  The rotors')
        ok &= report('Q_g,r = diag(G_r^-1, G_r) = cosh 1 - sinh alpha_I;  Q_g,phi = diag(G_phi^-1, G_phi) = cosh 1 - sinh alpha_K',
                     zero(Lr - (cr * one4 - sr * aI)) and zero(Lp - (cp * one4 - sp_ * aK)))
        Q, Qi = Lp * Lr, Lri * Lpi
        ok &= report('Q_g = Q_g,phi Q_g,r = diag(G_phi^-1 G_r^-1, G_phi G_r), Q_g^-1 = Q_g,r^-1 Q_g,phi^-1',
                     zero(Q - blk(Gpi * Gri, Z2, Z2, Gp * Gr)) and zero(Q * Qi - one4))

        print('2  The image of the line element')
        q = [c * dt, dr, r * dth, r * s * dph]
        dRs = vec(q)
        dRG4 = Lp * (Lr * dRs * Lri) * Lpi
        dRGp = Gpi * (Gri * pauli(q) * Gri) * Gpi
        C2, S2 = (Z**2 + Z**-2) / 2, (Z**2 - Z**-2) / 2                  # cosh, sinh of sgn chi_r
        Cp2, Sp2 = (W**2 + W**-2) / 2, (W**2 - W**-2) / 2
        T1 = C2 * q[0] - S2 * q[1]
        ex = [Cp2 * T1 - Sp2 * q[3], C2 * q[1] - S2 * q[0], q[2], Cp2 * q[3] - Sp2 * T1]
        ok &= report('nested Q_g,phi(Q_g,r dR/ Q_g,r^-1)Q_g,phi^-1 = Q_g dR/ Q_g^-1', zero(dRG4 - Q * dRs * Qi))
        ok &= report('dR/_g = [[0, dR^G], [-(dR^G)^T, 0]] with the BQ-Pauli image dR^G = G_phi^-1(G_r^-1 dR G_r^-1)G_phi^-1',
                     zero(dRG4 - blk(Z2, dRGp, -adjT_from(dRGp), Z2)))
        ok &= report('dR/_g = a pure vector with the gamma-form coordinates (= Pauli step 3)', zero(dRG4 - vec(ex)))
        print('3  The exact square')
        ok &= report('dR/_g^2 = dR/^2 = (-c^2 dt^2 + dr^2 + r^2 dtheta^2 + r^2 s^2 dphi^2) 1',
                     zero(dRG4 * dRG4 - (-(c * dt)**2 + dr**2 + (r * dth)**2 + (r * s * dph)**2) * one4))
        print('4  The basis route (both rotors)')
        bq = [Q * x * Qi for x in b]
        vR, vP = S2 / C2, Sp2 / Cp2                                       # v/c = tanh
        bq0 = C2 * Cp2 * (b[0] - vP * b[3]) - C2 * vR * b[1]
        bqr = C2 * b[1] - C2 * vR * Cp2 * (b[0] - vP * b[3])
        bqp = Cp2 * (b[3] - vP * b[0])
        ok &= report('beta^_0 = g_r g_phi(beta_0 - (v_phi/c)beta_phi) - g_r(v_r/c)beta_r, beta^_r, beta^_theta = beta_theta, beta^_phi = g_phi(beta_phi - (v_phi/c)beta_0)',
                     zero(bq[0] - bq0) and zero(bq[1] - bqr) and zero(bq[2] - b[2]) and zero(bq[3] - bqp))
        ok &= report('{beta^_a, beta^_b} = 2 eta_ab 1 and dR/_g = c dt beta^_0 + dr beta^_r + r dtheta beta^_theta + r s dphi beta^_phi',
                     all(zero(bq[i] * bq[j] + bq[j] * bq[i] - 2 * eta[i, j] * one4) for i in range(4) for j in range(4))
                     and zero(dRG4 - sum((q[k] * bq[k] for k in range(4)), sp.zeros(4))))
        print('   radial rotor alone')
        bh = [Lr * x * Lri for x in b]
        ok &= report('beta^_0 = C beta_0 - S beta_r, beta^_r = C beta_r - S beta_0, beta^_theta = beta_theta, beta^_phi = beta_phi',
                     zero(bh[0] - (C2 * b[0] - S2 * b[1])) and zero(bh[1] - (C2 * b[1] - S2 * b[0])) and zero(bh[2] - b[2]) and zero(bh[3] - b[3]))
        ok &= report('{beta^_a, beta^_b} = 2 eta_ab 1: the boosted basis is orthonormal',
                     all(zero(bh[i] * bh[j] + bh[j] * bh[i] - 2 * eta[i, j] * one4) for i in range(4) for j in range(4)))
        qr = [c * dt, dr, r * dth, r * s * dph]
        ok &= report('Q_g,r dR/ Q_g,r^-1 = dx^a (Q_g,r beta_a Q_g,r^-1)',
                     zero(Lr * dRs * Lri - sum((qr[k] * bh[k] for k in range(4)), sp.zeros(4))))
        if sgn == 1:
            psi = sp.Symbol('psi', real=True)
            Qp = (-(psi / 2) * b[0] * b[1]).exp().applyfunc(lambda x: sp.simplify(x.rewrite(sp.exp)))
            Qpi = ((psi / 2) * b[0] * b[1]).exp().applyfunc(lambda x: sp.simplify(x.rewrite(sp.exp)))
            got = (Qp * b[0] * Qpi).applyfunc(sp.simplify)
            minus = zero((got - (sp.cosh(psi) * b[0] - sp.sinh(psi) * b[1])).applyfunc(lambda x: x.rewrite(sp.exp)))
            plus = zero((got - (sp.cosh(psi) * b[0] + sp.sinh(psi) * b[1])).applyfunc(lambda x: x.rewrite(sp.exp)))
            ok &= report('Q_g = exp(-(1/2) psi beta_0 beta_r) gives Q_g beta_0 Q_g^-1 = cosh psi beta_0 - sinh psi beta_r (not +)', minus and not plus)
            print('     -> BQ Dirac Gravity eqs. 146-150 write + sinh psi beta_r; the + sign belongs to rotor rapidity -psi. Reported, not changed.')

        print('5  The coframe as a Dirac vector')
        m = -1 if sgn == 1 else 1                                        # theta^r = dr - v dt in the paper chart
        tha = [c * dt, dr + m * vr * dt, r * dth, r * s * dph + m * vp * dt]
        thv = vec(tha)
        ds2 = -(c * dt)**2 + tha[1]**2 + tha[2]**2 + tha[3]**2
        ok &= report('theta/ theta/ = ds^2 1,  ds^2 = -c^2 dt^2 + (dr -+ v_r dt)^2 + r^2 dtheta^2 + (r s dphi -+ v_phi dt)^2',
                     zero(thv * thv - ds2 * one4))
        thp = pauli(tha)
        ok &= report('in blocks theta/theta/ = [[-theta theta^T, 0], [0, -theta^T theta]], theta^T theta = -ds^2 1^ (the Pauli quadratic)',
                     zero(thv * thv - blk(-thp * adjT(tha), Z2, Z2, -adjT(tha) * thp)) and zero(adjT(tha) * thp + ds2 * P['1']))
        print('6  Coordinate by coordinate: exact and Galilean')
        sc = [c, 1, r, r * s]
        Rg = [sc[k] * bq[k] for k in range(4)]
        g0 = sp.diag(-c**2, 1, r**2, r**2 * s**2)
        ok &= report('dR/_g,mu = (c beta^_0, beta^_r, r beta^_theta, r s beta^_phi): {dR/_g,mu, dR/_g,nu} = 2 g0_mu_nu 1, g0 flat',
                     all(zero(Rg[i] * Rg[j] + Rg[j] * Rg[i] - 2 * g0[i, j] * one4) for i in range(4) for j in range(4))
                     and zero(dRG4 - (dt * Rg[0] + dr * Rg[1] + dth * Rg[2] + dph * Rg[3])))
        E = sp.Matrix([[c, 0, 0, 0], [m * vr, 1, 0, 0], [0, 0, r, 0], [m * vp, 0, 0, r * s]])     # theta^a_mu, mu = t, r, theta, phi
        Gs = [sum((E[a, mu] * b[a] for a in range(4)), sp.zeros(4)) for mu in range(4)]
        g = E.T * eta * E
        ok &= report('theta/_mu = theta^a_mu beta_a:  {theta/_mu, theta/_nu} = 2 g_mu_nu 1 for all mu, nu',
                     all(zero(Gs[i] * Gs[j] + Gs[j] * Gs[i] - 2 * g[i, j] * one4) for i in range(4) for j in range(4)))
        gPG = sp.Matrix([[-(c**2 - vr**2 - vp**2), m * vr, 0, m * vp * r * s], [m * vr, 1, 0, 0], [0, 0, r**2, 0], [m * vp * r * s, 0, 0, r**2 * s**2]])
        ok &= report('g_mu_nu = the PG metric with azimuthal shift (Pauli step 8), det g = -c^2 r^4 s^2',
                     zero(g - gPG) and zero(g.det() + c**2 * r**4 * s**2))
        print('7  Dirac representation')
        LD = S * Lr * S.T
        ok &= report('S Q_g,r S^-1 = [[c 1, s sigma_I], [s sigma_I, c 1]]', zero(LD - blk(cr * P['1'], sr * P['sI'], sr * P['sI'], cr * P['1'])))
        GD = [S * x * S.T for x in Gs]
        ok &= report('the anticommutators, and so g_mu_nu, are the same in the Dirac representation',
                     all(zero(GD[i] * GD[j] + GD[j] * GD[i] - 2 * g[i, j] * one4) for i in range(4) for j in range(4)))

    print('\nAll checks:', 'IDENTICAL' if ok else 'SOME DIFFER')
    return ok


def adjT_from(M):
    """BQ time adjoint of a 2x2 minquat matrix: flips the T^ channel (and, generally, T^ and sigma)."""
    tr = (M[0, 0] + M[1, 1]) / 2                     # = i a0 for a minquat
    return M - 2 * tr * sp.eye(2)


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
