"""
bq_dirac_lorentz.py -- Lorentz transformations of Dirac-level (BQ-Weyl) four-vectors, as on the page
BQ_Dirac_Lorentz_Transformations.html. Vectors only.

The Pauli boost along I acts from both sides, A^L = U^-1 A U^-1 with U = exp(psi sigma_I/2) = diag(e^psi/2, e^-psi/2),
and the time adjoint goes the other way, (A^L)^T = U A^T U. The Dirac vector holds both,
    a/ = [[0, A], [-A^T, 0]],
and one similarity with a block-diagonal matrix moves both blocks at once:
    Lambda = diag(U^-1, U) = exp(-psi alpha_I/2) = cosh(psi/2) 1 - sinh(psi/2) alpha_I,   a/^L = Lambda a/ Lambda^-1.
Lambda is the double of U: the upper block carries U^-1, the lower block U, and alpha_I = diag(sigma_I, -sigma_I)
carries the opposite signs.

Checked exactly with SymPy (written rationally in z = exp(psi/2)):
  1  the doubling: diag(X, Y) a/ diag(X, Y)^-1 = [[0, X A Y^-1], [-Y A^T X^-1, 0]]; X = U^-1, Y = U gives the two
     Pauli rules U^-1 A U^-1 and -U A^T U = -(A^L)^T at once
  2  Lambda = diag(U^-1, U) = exp(-psi alpha_I/2) = cosh(psi/2) 1 - sinh(psi/2) alpha_I = diag(e^-psi/2, e^psi/2, e^psi/2, e^-psi/2)
     (Weyl EM eq. 163), Lambda^-1 = exp(+psi alpha_I/2)
  3  a/^L has a0' = g(a0 - b a1), a1' = g(a1 - b a0), a2, a3 unchanged; Lambda^-1 a/^L Lambda = a/; a/a/ = -(a0^2 - a^2) 1 is
     invariant, and the scalar of a/b/ is invariant
  4  basis route: Lambda beta_mu Lambda^-1 = Lambda^2 beta_mu for mu = 0, I (alpha_I anticommutes with them) and = beta_mu for
     J, K (it commutes); Lambda b0 Lambda^-1 = cosh psi b0 - sinh psi bI, Lambda bI Lambda^-1 = cosh psi bI - sinh psi b0;
     the Dirac lemma beta_nu Lambda^nu_mu = Lambda beta_mu Lambda^-1, whose upper block is the Pauli lemma
     K_nu Lambda^nu_mu = U^-1 K_mu U^-1
  5  any direction n: Lambda_n = exp(-psi n.alpha/2) = cosh(psi/2) 1 - sinh(psi/2) n.alpha gives the standard boost
     a0' = g(a0 - b n.a), a' = a + (g - 1)(n.a) n - g b a0 n; its upper block U_n^-1 A U_n^-1 with the Hermitian
     U_n = exp(psi n.sigma/2)
  6  rotations: R = exp(-theta n.Sigma/2) = cos(theta/2) 1 - sin(theta/2) n.Sigma = diag(r, r), r = exp(theta n.K^/2) unitary;
     a0 unchanged, a rotated right-handedly by theta about n (Rodrigues); the Pauli rule is the similarity r A r^-1
  7  one rule for both: Lambda = diag(M, (M^dagger)^-1), with M = U^-1 (Hermitian) for a boost and M = r (unitary) for a rotation
  8  composition: Lambda(psi1) Lambda(psi2) = Lambda(psi1 + psi2) along I (velocities add as (b1 + b2)/(1 + b1 b2));
     an I-boost followed by a J-boost: Lambda_J Lambda_I = c1c2 1 - s1c2 alpha_I - c1s2 alpha_J - s1s2 Sigma_K
     (c_i = cosh(psi_i/2), s_i = sinh(psi_i/2)): the Sigma_K channel is the Thomas-Wigner rotation.
     It factors as a pure boost B times a right-handed rotation R_K(theta_W) = exp(-theta_W Sigma_K/2) about K,
     tan(theta_W/2) = tanh(psi1/2) tanh(psi2/2) (the rotation acts first: B R a/ R^-1 B^-1),
     and the combined boost has cosh eta = cosh psi1 cosh psi2

Usage
  python bq_dirac_lorentz.py
"""
import sympy as sp
from bq_dirac_table import EL, P, blk, NAMES

Z, Z1, Z2 = sp.symbols('z z1 z2', positive=True)          # z = exp(psi/2)
th = sp.Symbol('theta', real=True)
ch_ = lambda z: (z + 1 / z) / 2                             # cosh(psi/2) for z = exp(psi/2)
sh_ = lambda z: (z - 1 / z) / 2
BETA = [EL['b0'], EL['bI'], EL['bJ'], EL['bK']]
ALPHA = [EL['aI'], EL['aJ'], EL['aK']]
SIGMA = [EL['SI'], EL['SJ'], EL['SK']]
KP = [P['I'], P['J'], P['K']]
SP = [P['sI'], P['sJ'], P['sK']]
sy = lambda s: sp.symbols(s, real=True)


def report(name, ok):
    print(f'  {name:<88} {"IDENTICAL (exact)" if ok else "DIFFER"}')
    return ok


def zero(e):
    if isinstance(e, sp.MatrixBase):
        return all(zero(v) for v in e)
    return sp.simplify(sp.cancel(sp.together(sp.expand(e)))) == 0


def channels(M):
    return {n: sp.expand((EL[n] * M).trace() / (EL[n] * EL[n]).trace()) for n in NAMES}


def vec4(a):
    return sum((a[m] * BETA[m] for m in range(4)), sp.zeros(4))


def coords(M):
    c = channels(M)
    return [c['b0'], c['bI'], c['bJ'], c['bK']]


def adjT(M):
    """BQ time adjoint of a 2x2 element: the T^ and sigma^ channels change sign (complex conjugate coefficients)"""
    return P['sJ'] * M.conjugate() * P['sJ']


def main():
    ok = True
    a, b = sy('a0:4'), sy('b0:4')
    A, Bv = vec4(a), vec4(b)
    Ap = a[0] * P['T'] + sum((a[i + 1] * KP[i] for i in range(3)), sp.zeros(2))
    U, Ui = sp.diag(Z, 1 / Z), sp.diag(1 / Z, Z)
    cosh, sinh = ch_(Z**2), sh_(Z**2)                          # cosh psi = gamma, sinh psi = beta gamma

    print('1  The doubling: one similarity carries both Pauli rules')
    ok &= report('time adjoint: (a0 T^ + a.K^)^T = -a0 T^ + a.K^', zero(adjT(Ap) - (-a[0] * P['T'] + Ap - a[0] * P['T'])))
    ok &= report('a/ = [[0, A], [-A^T, 0]]', zero(A - blk(sp.zeros(2), Ap, -adjT(Ap), sp.zeros(2))))
    X, Y = sp.Matrix(2, 2, sy('x0:4')), sp.Matrix(2, 2, sy('y0:4'))
    D = blk(X, sp.zeros(2), sp.zeros(2), Y)
    ok &= report('diag(X, Y) a/ diag(X, Y)^-1 = [[0, X A Y^-1], [-Y A^T X^-1, 0]]',
                 zero(D * A * D.inv() - blk(sp.zeros(2), X * Ap * Y.inv(), -Y * adjT(Ap) * X.inv(), sp.zeros(2))))
    L, Li = blk(Ui, sp.zeros(2), sp.zeros(2), U), blk(U, sp.zeros(2), sp.zeros(2), Ui)
    AL = L * A * Li
    ApL = Ui * Ap * Ui
    ok &= report('X = U^-1, Y = U: upper block U^-1 A U^-1 = A^L, lower block -U A^T U = -(A^L)^T',
                 zero(AL[0:2, 2:4] - ApL) and zero(AL[2:4, 0:2] + U * adjT(Ap) * U) and zero(adjT(ApL) - U * adjT(Ap) * U))

    print('2  Lambda as a rotor of the Weyl algebra')
    psi = sp.Symbol('psi', real=True)
    Lexp = (-psi / 2 * EL['aI']).exp()
    ok &= report('exp(-psi alpha_I/2) = diag(e^-psi/2, e^psi/2, e^psi/2, e^-psi/2) = diag(U^-1, U)  (eq. 163)',
                 zero(Lexp - sp.diag(sp.exp(-psi / 2), sp.exp(psi / 2), sp.exp(psi / 2), sp.exp(-psi / 2))) and zero(L - Lexp.subs(psi, 2 * sp.log(Z))))
    ok &= report('Lambda = cosh(psi/2) 1 - sinh(psi/2) alpha_I, Lambda^-1 = cosh(psi/2) 1 + sinh(psi/2) alpha_I',
                 zero(L - (ch_(Z) * EL['one'] - sh_(Z) * EL['aI'])) and zero(Li - (ch_(Z) * EL['one'] + sh_(Z) * EL['aI'])))
    ok &= report('alpha_I = diag(sigma_I, -sigma_I), alpha_I^2 = 1', zero(EL['aI'] - blk(P['sI'], sp.zeros(2), sp.zeros(2), -P['sI'])) and zero(EL['aI']**2 - EL['one']))

    print('3  Boosting a vector')
    g_, gb = cosh, sinh
    ok &= report("a0' = g(a0 - b a1), a1' = g(a1 - b a0), a2' = a2, a3' = a3  (g = cosh psi, g b = sinh psi)",
                 all(zero(x - y) for x, y in zip(coords(AL), [g_ * a[0] - gb * a[1], g_ * a[1] - gb * a[0], a[2], a[3]])))
    ok &= report('AL is a pure vector (no other channels)', all(zero(v) for k, v in channels(AL).items() if k not in ('b0', 'bI', 'bJ', 'bK')))
    ok &= report('inverse: Lambda^-1 a/^L Lambda = a/', zero(Li * AL * L - A))
    ok &= report('a/a/ = -(a0^2 - a^2) 1, and (a/^L)^2 = a/a/', zero(A * A + (a[0]**2 - a[1]**2 - a[2]**2 - a[3]**2) * EL['one']) and zero(AL * AL - A * A))
    ok &= report('scalar of a/b/ invariant: [a/^L b/^L]_1 = [a/b/]_1 = -(a0 b0 - a.b)',
                 zero(channels(AL * (L * Bv * Li))['one'] - channels(A * Bv)['one']) and zero(channels(A * Bv)['one'] + a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]))

    print('4  Basis route and the Dirac lemma')
    ok &= report('alpha_I anticommutes with b0, bI and commutes with bJ, bK',
                 zero(EL['aI'] * EL['b0'] + EL['b0'] * EL['aI']) and zero(EL['aI'] * EL['bI'] + EL['bI'] * EL['aI'])
                 and zero(EL['aI'] * EL['bJ'] - EL['bJ'] * EL['aI']) and zero(EL['aI'] * EL['bK'] - EL['bK'] * EL['aI']))
    ok &= report('so Lambda b Lambda^-1 = Lambda^2 b for b0, bI (half rapidity -> full rapidity), = b for bJ, bK',
                 zero(L * EL['b0'] * Li - L * L * EL['b0']) and zero(L * EL['bI'] * Li - L * L * EL['bI'])
                 and zero(L * EL['bJ'] * Li - EL['bJ']) and zero(L * EL['bK'] * Li - EL['bK']))
    ok &= report('Lambda^2 = cosh psi 1 - sinh psi alpha_I, alpha_I b0 = bI, alpha_I bI = b0',
                 zero(L * L - (cosh * EL['one'] - sinh * EL['aI'])) and zero(EL['aI'] * EL['b0'] - EL['bI']) and zero(EL['aI'] * EL['bI'] - EL['b0']))
    ok &= report('Lambda b0 Lambda^-1 = cosh psi b0 - sinh psi bI, Lambda bI Lambda^-1 = cosh psi bI - sinh psi b0',
                 zero(L * EL['b0'] * Li - (cosh * EL['b0'] - sinh * EL['bI'])) and zero(L * EL['bI'] * Li - (cosh * EL['bI'] - sinh * EL['b0'])))
    Lam = sp.Matrix([[cosh, -sinh, 0, 0], [-sinh, cosh, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])   # coordinate boost a' = Lam a
    ok &= report('Dirac lemma: beta_nu Lambda^nu_mu = Lambda beta_mu Lambda^-1 (a similarity)',
                 all(zero(sum((Lam[nu, mu] * BETA[nu] for nu in range(4)), sp.zeros(4)) - L * BETA[mu] * Li) for mu in range(4)))
    KK = [P['T']] + KP
    ok &= report('its upper block is the Pauli lemma K_nu Lambda^nu_mu = U^-1 K_mu U^-1',
                 all(zero(sum((Lam[nu, mu] * KK[nu] for nu in range(4)), sp.zeros(2)) - Ui * KK[mu] * Ui) for mu in range(4)))
    ok &= report('and its lower block: -U K_mu^T U = -(U^-1 K_mu U^-1)^T = -(K_nu Lambda^nu_mu)^T, so the Dirac lemma follows',
                 all(zero((L * BETA[mu] * Li)[2:4, 0:2] + U * adjT(KK[mu]) * U) and zero(U * adjT(KK[mu]) * U - adjT(Ui * KK[mu] * Ui))
                     and zero(adjT(sum((Lam[nu, mu] * KK[nu] for nu in range(4)), sp.zeros(2))) - sum((Lam[nu, mu] * adjT(KK[nu]) for nu in range(4)), sp.zeros(2)))
                     for mu in range(4)))
    ok &= report('beta_mu = [[0, K_mu], [-K_mu^T, 0]] with K_0 = T^, K_i = K^_i', all(zero(BETA[mu] - blk(sp.zeros(2), KK[mu], -adjT(KK[mu]), sp.zeros(2))) for mu in range(4)))
    gl = sp.Symbol('lam', nonzero=True)
    ok &= report('a common factor cancels: diag(lam U^-1, lam U) gives the same boost (Lambda is fixed up to it)',
                 zero(blk(gl * Ui, sp.zeros(2), sp.zeros(2), gl * U) * A * blk(gl * Ui, sp.zeros(2), sp.zeros(2), gl * U).inv() - AL))

    print('5  Any direction n (here n = (2/3, 1/3, 2/3), exact)')
    n = [sp.Rational(2, 3), sp.Rational(1, 3), sp.Rational(2, 3)]
    nal = sum((n[i] * ALPHA[i] for i in range(3)), sp.zeros(4))
    Ln, Lni = ch_(Z) * EL['one'] - sh_(Z) * nal, ch_(Z) * EL['one'] + sh_(Z) * nal
    ok &= report('(n.alpha)^2 = 1, so exp(-psi n.alpha/2) = cosh(psi/2) 1 - sinh(psi/2) n.alpha; Lambda_n Lambda_n^-1 = 1',
                 zero(nal * nal - EL['one']) and zero(Ln * Lni - EL['one']))
    na = sum(n[i] * a[i + 1] for i in range(3))
    want = [cosh * a[0] - sinh * na] + [a[i + 1] + (cosh - 1) * na * n[i] - sinh * a[0] * n[i] for i in range(3)]
    ok &= report("a0' = g(a0 - b n.a), a' = a + (g - 1)(n.a) n - g b a0 n", all(zero(x - y) for x, y in zip(coords(Ln * A * Lni), want)))
    Un = ch_(Z) * P['1'] + sh_(Z) * sum((n[i] * SP[i] for i in range(3)), sp.zeros(2))
    ok &= report('upper block = U_n^-1 A U_n^-1, U_n = exp(psi n.sigma/2) Hermitian (U_n^dagger = U_n)',
                 zero((Ln * A * Lni)[0:2, 2:4] - Un.inv() * Ap * Un.inv()) and zero(Un.H - Un) and zero(Ln - blk(Un.inv(), sp.zeros(2), sp.zeros(2), Un)))

    print('6  Rotations')
    m = [sp.Rational(2, 7), sp.Rational(3, 7), sp.Rational(6, 7)]
    mS = sum((m[i] * SIGMA[i] for i in range(3)), sp.zeros(4))
    R = sp.cos(th / 2) * EL['one'] - sp.sin(th / 2) * mS
    Ri = sp.cos(th / 2) * EL['one'] + sp.sin(th / 2) * mS
    ok &= report('(m.Sigma)^2 = -1, so exp(-theta m.Sigma/2) = cos(theta/2) 1 - sin(theta/2) m.Sigma',
                 zero(mS * mS + EL['one']) and zero(R * Ri - EL['one']))
    rr = sp.cos(th / 2) * P['1'] + sp.sin(th / 2) * sum((m[i] * KP[i] for i in range(3)), sp.zeros(2))
    ok &= report('R = diag(r, r), r = cos(theta/2) 1^ + sin(theta/2) m.K^ = exp(theta m.K^/2), unitary',
                 zero(R - blk(rr, sp.zeros(2), sp.zeros(2), rr)) and zero(sp.simplify(rr.H * rr) - P['1']))
    av = sp.Matrix(a[1:])
    mv = sp.Matrix(m)
    rod = av * sp.cos(th) + mv.cross(av) * sp.sin(th) + mv * (mv.dot(av)) * (1 - sp.cos(th))
    ok &= report("a0' = a0, a' = a cos(theta) + (m x a) sin(theta) + m (m.a)(1 - cos(theta))  (right-handed, Rodrigues)",
                 all(zero(x - y) for x, y in zip(coords(R * A * Ri), [a[0]] + list(rod))))
    ok &= report('Pauli rule for a rotation: the similarity r A r^-1', zero((R * A * Ri)[0:2, 2:4] - rr * Ap * rr.inv()))

    print('7  One rule for boosts and rotations: Lambda = diag(M, (M^dagger)^-1)')
    ok &= report('boost: M = U_n^-1 (Hermitian), (M^dagger)^-1 = U_n', zero(Ln - blk(Un.inv(), sp.zeros(2), sp.zeros(2), (Un.inv().H).inv())))
    ok &= report('rotation: M = r (unitary), (M^dagger)^-1 = r', zero(sp.simplify(R - blk(rr, sp.zeros(2), sp.zeros(2), sp.simplify((rr.H).inv())))))

    print('8  Composition')
    L1 = ch_(Z1) * EL['one'] - sh_(Z1) * EL['aI']
    L2 = ch_(Z2) * EL['one'] - sh_(Z2) * EL['aI']
    L12 = ch_(Z1 * Z2) * EL['one'] - sh_(Z1 * Z2) * EL['aI']
    ok &= report('collinear: Lambda(psi1) Lambda(psi2) = Lambda(psi1 + psi2), rapidities add', zero(L1 * L2 - L12))
    b1, b2 = (Z1**4 - 1) / (Z1**4 + 1), (Z2**4 - 1) / (Z2**4 + 1)              # tanh psi_i
    ok &= report('velocity addition: tanh(psi1 + psi2) = (b1 + b2)/(1 + b1 b2)', zero(((Z1 * Z2)**4 - 1) / ((Z1 * Z2)**4 + 1) - (b1 + b2) / (1 + b1 * b2)))
    c1, s1, c2, s2 = ch_(Z1), sh_(Z1), ch_(Z2), sh_(Z2)
    LJ = c2 * EL['one'] - s2 * EL['aJ']
    LI = c1 * EL['one'] - s1 * EL['aI']
    Pm = LJ * LI
    cP = channels(Pm)
    ok &= report('I-boost then J-boost: Lambda_J Lambda_I = c1c2 1 - s1c2 alpha_I - c1s2 alpha_J - s1s2 Sigma_K',
                 zero(Pm - (c1 * c2 * EL['one'] - s1 * c2 * EL['aI'] - c1 * s2 * EL['aJ'] - s1 * s2 * EL['SK'])))
    N = sp.Symbol('N', positive=True)                                          # N = sqrt(c1^2c2^2 + s1^2s2^2)
    Rw_inv = (c1 * c2 / N) * EL['one'] + (s1 * s2 / N) * EL['SK']              # inverse of R_K = cos(t/2) 1 - sin(t/2) Sigma_K
    Bw = Pm * Rw_inv                                                           # pure boost part
    cB = channels(Bw)
    N2 = c1**2 * c2**2 + s1**2 * s2**2
    ok &= report('factor: Lambda_J Lambda_I = B R_K(theta_W), tan(theta_W/2) = tanh(psi1/2) tanh(psi2/2); B a pure boost',
                 all(zero(cB[k]) for k in ['SI', 'SJ', 'SK', 'chi', 'aK']) and zero((cB['one'] * N).subs(N, sp.sqrt(N2)) - N2))
    ok &= report('the combined boost: cosh eta = 2 cosh^2(eta/2) - 1 = cosh psi1 cosh psi2', zero(2 * N2 - 1 - ch_(Z1**2) * ch_(Z2**2)))

    print('\nAll checks:', 'IDENTICAL' if ok else 'SOME DIFFER')
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
