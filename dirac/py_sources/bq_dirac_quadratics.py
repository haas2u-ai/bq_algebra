"""
bq_dirac_quadratics.py -- the quadratic products of a Dirac-level (BQ-Weyl) field, as on the page
BQ_Dirac_Field_Quadratics.html.

A field is an even element  F/ = f 1 + p.alpha + q.Sigma  (scalar plus two bivectors). Two quadratics follow:
  the field square      F/F/  = (f^2 + p^2 - q^2) 1 + 2f p.alpha + 2f q.Sigma - 2(p.q) chi
  the sandwich          F/a/F/ = [(f^2 - p^2 - q^2) a0 + 2 a.(p x q)] beta_0
                               + [(f^2 + p^2 + q^2) a - 2(a.p) p - 2(a.q) q - 2 a0 (p x q)].beta
                               - 2f (a.q) beta~_0 + 2f (a x p - a0 q).beta~
for any vector a/ = a0 beta_0 + a.beta. The general even product is
  F/G/ = (fg + p.m - q.n) 1 + (f m + g p - p x n - q x m).alpha + (f n + g q + p x m - q x n).Sigma - (p.n + q.m) chi
with G/ = g 1 + m.alpha + n.Sigma.

Checked exactly with SymPy (4x4 matrices built from the BQ-Pauli blocks, as in bq_dirac_table.py):
  1  the even product rule F/G/ (8 channels, nothing odd) and the square F/F/; a pure bivector squares into {1, chi}
  2  the sandwich rule F/a/F/ for a general vector a/; for f = 0 it is a pure vector (grade 1)
  3  block form, with the BQ-Pauli field F = -f 1^ + q.K^ + p.sigma^ and its time adjoint F^T:
       F/ = diag(-F^T, -F),  F/G/ = diag(F^T G^T, F G),  F/a/F/ = [[0, F^T A F], [-F A^T F^T, 0]],
     so the lower block of F/F/ is the Pauli square FF and the upper block of F/beta_0F/ is T^ F^T F
       [F/F/]_1 = [FF]_1^, [F/F/]_chi = -[FF]_T^, [F/F/]_alpha = -[FF]_sigma^, [F/F/]_Sigma = -[FF]_K^
       [F/b0F/]_b0 = [T^F^TF]_T^, _beta = _K^, _beta~0 = -_1^, _beta~ = -_sigma^
  4  electromagnetism, B/ = b1 1 + (E/c).alpha + B.Sigma:
       B/B/ = (E^2/c^2 - B^2) 1 - (2/c)(E.B) chi                      (Lorenz gauge, eq. 124)
       -B/beta_0B//(2 mu0) = u_EM beta_0 + (S/c).beta                 (eq. 150)
       -B/beta_iB//(2 mu0) = -(S_i/c) beta_0 + sigma_ji beta_j         (eqs. 153-154, Maxwell stress)
  5  fluid, H/ = sigma_E 1 - (Pi/c).alpha + Omega.Sigma: eqs. 296, 316 and 318 (with the beta~ channel -2 sigma_E Omega)
  6  Lorentz boost along I, Lambda = diag(U^-1, U), U = diag(e^(psi/2), e^(-psi/2)) (Weyl EM, eq. 163):
       Lambda F/ Lambda^-1 is even; (F/^L)^2 = Lambda F/^2 Lambda^-1, and its 1 and chi channels are invariant;
       F/^L a/^L F/^L = Lambda (F/a/F/) Lambda^-1 (covariant map);
       F/^L beta_0 F/^L differs from Lambda (F/beta_0F/) Lambda^-1 (the energy-flux reading belongs to a frame)
     Note: eq. 161 writes U = exp((psi/2) I^), which is diag(e^(i psi/2), e^(-i psi/2)); the boost matrix of eq. 163
     needs U = exp((psi/2) sigma_I). The script uses eq. 163 and reports the difference.
     The boosted EM field is the standard one for a frame moving with v = c tanh psi along I:
       E'_I = E_I, E'_J = g(E_J - vB_K), E'_K = g(E_K + vB_J),  B'_I = B_I, B'_J = g(B_J + vE_K/c^2), B'_K = g(B_K - vE_J/c^2)

Usage
  python bq_dirac_quadratics.py
"""
import sympy as sp
from bq_dirac_table import EL, P, blk, NAMES

c, mu0 = sp.symbols('c mu_0', positive=True)
Z = sp.Symbol('z', positive=True)                 # z = exp(psi/2): the boost is rational in z
BETA = [EL['b0'], EL['bI'], EL['bJ'], EL['bK']]
sy = lambda s: sp.symbols(s, real=True)


def report(name, ok):
    print(f'  {name:<86} {"IDENTICAL (exact)" if ok else "DIFFER"}')
    return ok


def zero(e):
    if isinstance(e, sp.MatrixBase):
        return all(zero(v) for v in e)
    return sp.cancel(sp.together(sp.expand(e))) == 0


def channels(M):
    return {n: sp.expand((EL[n] * M).trace() / (EL[n] * EL[n]).trace()) for n in NAMES}


def check(M, want):
    ch = channels(M)
    return all(zero(ch[n] - want.get(n, 0)) for n in NAMES)


def lin(cs, ks):
    M = sp.zeros(4)
    for x, k in zip(cs, ks):
        M += x * EL[k]
    return M


def even(f, p, q):
    return f * EL['one'] + lin(p, ['aI', 'aJ', 'aK']) + lin(q, ['SI', 'SJ', 'SK'])


def vec4(a):
    return lin(a, ['b0', 'bI', 'bJ', 'bK'])


def pauli(one, T, k, s):
    return one * P['1'] + T * P['T'] + sum((k[i] * P[x] for i, x in enumerate('IJK')), sp.zeros(2)) \
        + sum((s[i] * P[x] for i, x in enumerate(['sI', 'sJ', 'sK'])), sp.zeros(2))


def pdec(M):
    """exact decomposition of a 2x2 matrix on the eight BQ-Pauli elements (real coefficients)"""
    a, d, u, l = (sp.expand(M[0, 0]), sp.expand(M[1, 1]), sp.expand(M[0, 1]), sp.expand(M[1, 0]))
    re, im = sp.re, sp.im
    return {'1': re(a + d) / 2, 'T': im(a + d) / 2, 'I': im(a - d) / 2, 'J': re(u - l) / 2, 'K': im(u + l) / 2,
            'sI': re(a - d) / 2, 'sJ': im(l - u) / 2, 'sK': re(u + l) / 2}


def adjT(M):
    """BQ time adjoint of a 2x2 element: the complex conjugate of the biquaternion coefficients on (1^, I^, J^, K^),
    i.e. the T^ and sigma^ channels change sign. For a vector A^T = -a0 T^ + a.K^; for a field
    F^T = f 1^ + k.K^ - s.sigma^ (as in bq_quadratics.py and Weyl EM eq. 128)."""
    d = pdec(M)
    return M - 2 * (d['T'] * P['T'] + d['sI'] * P['sI'] + d['sJ'] * P['sJ'] + d['sK'] * P['sK'])


V = lambda x: sp.Matrix(x)
dot = lambda u, v: (V(u).T * V(v))[0]
cross = lambda u, v: list(V(u).cross(V(v)))


def sq_rule(f, p, q):
    return {'one': f**2 + dot(p, p) - dot(q, q), 'chi': -2 * dot(p, q),
            **{n: 2 * f * p[i] for i, n in enumerate(['aI', 'aJ', 'aK'])},
            **{n: 2 * f * q[i] for i, n in enumerate(['SI', 'SJ', 'SK'])}}


def sandwich_rule(f, p, q, a):
    a0, av = a[0], a[1:]
    pq = cross(p, q)
    r = {'b0': (f**2 - dot(p, p) - dot(q, q)) * a0 + 2 * dot(av, pq), 'bt0': -2 * f * dot(av, q)}
    for i, n in enumerate(['bI', 'bJ', 'bK']):
        r[n] = (f**2 + dot(p, p) + dot(q, q)) * av[i] - 2 * dot(av, p) * p[i] - 2 * dot(av, q) * q[i] - 2 * a0 * pq[i]
    axp = cross(av, p)
    for i, n in enumerate(['btI', 'btJ', 'btK']):
        r[n] = 2 * f * (axp[i] - a0 * q[i])
    return r


def main():
    ok = True
    f, g = sy('f g')
    p, q, m, n, a = sy('p1:4'), sy('q1:4'), sy('m1:4'), sy('n1:4'), sy('a0:4')
    F, G, A = even(f, p, q), even(g, m, n), vec4(a)

    print('1  The even product and the field square')
    want = {'one': f * g + dot(p, m) - dot(q, n), 'chi': -(dot(p, n) + dot(q, m))}
    al = [f * m[i] + g * p[i] - cross(p, n)[i] - cross(q, m)[i] for i in range(3)]
    sg = [f * n[i] + g * q[i] + cross(p, m)[i] - cross(q, n)[i] for i in range(3)]
    want.update({k: al[i] for i, k in enumerate(['aI', 'aJ', 'aK'])}); want.update({k: sg[i] for i, k in enumerate(['SI', 'SJ', 'SK'])})
    ok &= report('F/G/ = (fg + p.m - q.n) 1 + (fm + gp - pxn - qxm).alpha + (fn + gq + pxm - qxn).Sigma - (p.n + q.m) chi', check(F * G, want))
    ok &= report('F/F/ = (f^2 + p^2 - q^2) 1 + 2f p.alpha + 2f q.Sigma - 2(p.q) chi', check(F * F, sq_rule(f, p, q)))
    F0 = even(0, p, q)
    ok &= report('pure bivector (f = 0): F/F/ lies in {1, chi} only', check(F0 * F0, sq_rule(0, p, q)))

    print('2  The sandwich F/a/F/')
    ok &= report('F/a/F/ = vector [(f^2-p^2-q^2)a0 + 2a.(pxq), (f^2+p^2+q^2)a - 2(a.p)p - 2(a.q)q - 2a0 pxq]',
                 check(F * A * F, sandwich_rule(f, p, q, a)))
    ok &= report('            + pseudovector [-2f a.q, 2f(a x p - a0 q)]; for f = 0 a pure vector', check(F0 * A * F0, sandwich_rule(0, p, q, a)))

    print('3  The Pauli products inside: block form')
    Fp = pauli(-f, 0, q, p)                             # F = -f 1^ + q.K^ + p.sigma^
    Gp = pauli(-g, 0, n, m)
    Ap = pauli(0, a[0], a[1:], [0, 0, 0])               # A = a0 T^ + a.K^
    ok &= report('F/ = diag(-F^T, -F), F = -f 1^ + q.K^ + p.sigma^', zero(F - blk(-adjT(Fp), sp.zeros(2), sp.zeros(2), -Fp)))
    ok &= report('a/ = [[0, A], [-A^T, 0]], A = a0 T^ + a.K^', zero(A - blk(sp.zeros(2), Ap, -adjT(Ap), sp.zeros(2))))
    ok &= report('F/G/ = diag(F^T G^T, F G)', zero(F * G - blk(adjT(Fp) * adjT(Gp), sp.zeros(2), sp.zeros(2), Fp * Gp)))
    ok &= report('F/a/F/ = [[0, F^T A F], [-F A^T F^T, 0]]',
                 zero(F * A * F - blk(sp.zeros(2), adjT(Fp) * Ap * Fp, -Fp * adjT(Ap) * adjT(Fp), sp.zeros(2))))
    FF, ch2 = pdec(Fp * Fp), channels(F * F)
    ok &= report('[F/F/]_1 = [FF]_1^, _chi = -[FF]_T^, _alpha = -[FF]_sigma^, _Sigma = -[FF]_K^',
                 zero(ch2['one'] - FF['1']) and zero(ch2['chi'] + FF['T'])
                 and all(zero(ch2[w] + FF[s]) for w, s in zip(['aI', 'aJ', 'aK'], ['sI', 'sJ', 'sK']))
                 and all(zero(ch2[w] + FF[s]) for w, s in zip(['SI', 'SJ', 'SK'], 'IJK')))
    TFF, chs = pdec(P['T'] * adjT(Fp) * Fp), channels(F * EL['b0'] * F)
    ok &= report('upper block of F/b0F/ is T^F^TF: b0 = [.]_T^, beta = [.]_K^, beta~0 = -[.]_1^, beta~ = -[.]_sigma^',
                 zero((F * EL['b0'] * F)[0:2, 2:4] - P['T'] * adjT(Fp) * Fp) and zero(chs['b0'] - TFF['T']) and zero(chs['bt0'] + TFF['1'])
                 and all(zero(chs[w] - TFF[s]) for w, s in zip(['bI', 'bJ', 'bK'], 'IJK'))
                 and all(zero(chs[w] + TFF[s]) for w, s in zip(['btI', 'btJ', 'btK'], ['sI', 'sJ', 'sK'])))

    print('4  Electromagnetism: B/ = b1 1 + (E/c).alpha + B.Sigma')
    b1, E, B = sy('b1'), sy('E1:4'), sy('B1:4')
    Ec = [e / c for e in E]
    Bm, Bm0 = even(b1, Ec, B), even(0, Ec, B)
    ok &= report('B/B/ = (E^2/c^2 - B^2) 1 - (2/c)(E.B) chi  (Lorenz gauge, eq. 124)',
                 check(Bm0 * Bm0, {'one': dot(E, E) / c**2 - dot(B, B), 'chi': -2 * dot(E, B) / c}))
    ok &= report('with b1: + b1^2 1 + 2 b1 ((E/c).alpha + B.Sigma)', check(Bm * Bm, sq_rule(b1, Ec, B)))
    u = (dot(E, E) / c**2 + dot(B, B)) / (2 * mu0)
    S = [x / mu0 for x in cross(E, B)]
    T0 = -Bm0 * EL['b0'] * Bm0 / (2 * mu0)
    ok &= report('-B/b0B//(2mu0) = u_EM b0 + (S/c).beta, u_EM = (E^2/c^2 + B^2)/(2mu0), S = E x B/mu0  (eq. 150)',
                 check(T0, {'b0': u, 'bI': S[0] / c, 'bJ': S[1] / c, 'bK': S[2] / c}))
    sig = lambda i, j: (E[i] * E[j] / c**2 + B[i] * B[j] - sp.KroneckerDelta(i, j) * (dot(E, E) / c**2 + dot(B, B)) / 2) / mu0
    ok &= report('-B/b_iB//(2mu0) = -(S_i/c) b0 + sigma_ji b_j, the Maxwell stress  (eqs. 153-154)',
                 all(check(-Bm0 * BETA[i + 1] * Bm0 / (2 * mu0),
                           {'b0': -S[i] / c, **{k: sig(j, i) for j, k in enumerate(['bI', 'bJ', 'bK'])}}) for i in range(3)))
    tr = sum(channels(-Bm0 * BETA[mu] * Bm0 / (2 * mu0))[k] for mu, k in enumerate(['b0', 'bI', 'bJ', 'bK']))
    ok &= report('the diagonal of the array sums to zero: u_EM + sigma_II + sigma_JJ + sigma_KK = 0 (traceless)', zero(tr))

    print('5  Fluid: H/ = sigma_E 1 - (Pi/c).alpha + Omega.Sigma')
    sE, Pi, Om = sy('sigma_E'), sy('Pi1:4'), sy('Omega1:4')
    Pc = [-x / c for x in Pi]
    H, H0 = even(sE, Pc, Om), even(0, Pc, Om)
    ok &= report('H/0^2 = (Pi^2/c^2 - Omega^2) 1 + (2/c)(Omega.Pi) chi  (eq. 296)',
                 check(H0 * H0, {'one': dot(Pi, Pi) / c**2 - dot(Om, Om), 'chi': 2 * dot(Om, Pi) / c}))
    ok &= report('H/^2 = (sE^2 - Omega^2 + Pi^2/c^2) 1 + 2 sE Omega.Sigma - (2 sE/c) Pi.alpha + (2/c)(Omega.Pi) chi  (eq. 316)',
                 check(H * H, {'one': sE**2 - dot(Om, Om) + dot(Pi, Pi) / c**2, 'chi': 2 * dot(Om, Pi) / c,
                               **{k: 2 * sE * Om[i] for i, k in enumerate(['SI', 'SJ', 'SK'])},
                               **{k: -2 * sE * Pi[i] / c for i, k in enumerate(['aI', 'aJ', 'aK'])}}))
    PxO = cross(Pi, Om)
    ok &= report('H/b0H/ = (sE^2 - Omega^2 - Pi^2/c^2) b0 + (2/c)(Pi x Omega).beta - 2 sE Omega.beta~  (eq. 318)',
                 check(H * EL['b0'] * H, {'b0': sE**2 - dot(Om, Om) - dot(Pi, Pi) / c**2,
                                          **{k: 2 * PxO[i] / c for i, k in enumerate(['bI', 'bJ', 'bK'])},
                                          **{k: -2 * sE * Om[i] for i, k in enumerate(['btI', 'btJ', 'btK'])}}))

    print('6  Lorentz boost along I: Lambda = diag(U^-1, U), U = diag(z, 1/z), z = exp(psi/2)')
    U = sp.diag(Z, 1 / Z)
    L, Li = sp.diag(1 / Z, Z, Z, 1 / Z), sp.diag(Z, 1 / Z, 1 / Z, Z)
    ok &= report('Lambda = diag(U^-1, U) = diag(e^-psi/2, e^psi/2, e^psi/2, e^-psi/2)  (eq. 163)',
                 zero(L - blk(U.inv(), sp.zeros(2), sp.zeros(2), U)))
    psi = sp.Symbol('psi', real=True)
    ok &= report('U = exp((psi/2) sigma_I) = diag(e^psi/2, e^-psi/2)', zero((psi / 2 * P['sI']).exp() - sp.diag(sp.exp(psi / 2), sp.exp(-psi / 2))))
    print('     note: Weyl EM eq. 161 writes U = exp((psi/2) I^); that is', (psi / 2 * P['I']).exp().tolist(),
          '\n           a phase, not the boost of eq. 163 (reported, not repaired)')
    ok &= report('vectors: Lambda a/ Lambda^-1 has the Pauli block U^-1 A U^-1 (A^L of the Pauli level)',
                 zero((L * A * Li)[0:2, 2:4] - U.inv() * Ap * U.inv()))
    FL = L * F * Li
    ok &= report('Lambda F/ Lambda^-1 is again even (no beta, beta~ channels)',
                 all(zero(channels(FL)[k]) for k in ['b0', 'bI', 'bJ', 'bK', 'bt0', 'btI', 'btJ', 'btK']))
    chL, ch0 = channels(FL * FL), channels(F * F)
    ok &= report('(F/^L)^2 = Lambda F/^2 Lambda^-1; its 1 and chi channels are invariant',
                 zero(FL * FL - L * F * F * Li) and zero(chL['one'] - ch0['one']) and zero(chL['chi'] - ch0['chi']))
    ok &= report('covariance of the map: F/^L a/^L F/^L = Lambda (F/a/F/) Lambda^-1', zero(FL * (L * A * Li) * FL - L * F * A * F * Li))
    diff = FL * EL['b0'] * FL - L * F * EL['b0'] * F * Li
    ok &= report('fixed b0: F/^L b0 F/^L != Lambda (F/b0F/) Lambda^-1  (eq. 194: frame-resolved, not covariant)', not zero(diff))
    BL = channels(L * Bm0 * Li)
    gam, gv = (Z**2 + Z**-2) / 2, (Z**2 - Z**-2) / 2       # cosh psi, sinh psi = gamma v/c
    EL_ = [E[0], gam * E[1] - gv * c * B[2], gam * E[2] + gv * c * B[1]]
    BL_ = [B[0], gam * B[1] + gv * E[2] / c, gam * B[2] - gv * E[1] / c]
    ok &= report("boosted field: E' = (E_I, g(E_J - vB_K), g(E_K + vB_J)), B' = (B_I, g(B_J + vE_K/c^2), g(B_K - vE_J/c^2))",
                 all(zero(BL[k] - EL_[i] / c) for i, k in enumerate(['aI', 'aJ', 'aK']))
                 and all(zero(BL[k] - BL_[i]) for i, k in enumerate(['SI', 'SJ', 'SK'])))

    print('\nAll checks:', 'IDENTICAL' if ok else 'SOME DIFFER')
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
