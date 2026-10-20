"""
bq_dirac_bilinear_lorentz.py -- the Lorentz boost of the Dirac-level bilinear C/ = A/B/, as on the page
BQ_Dirac_Bilinear_Lorentz_Transformation.html.

Boosting both vectors along I, a/^L = Lambda a/ Lambda^-1 with Lambda = diag(U^-1, U) = exp(-psi alpha_I/2), gives
    C/^L = (Lambda A/ Lambda^-1)(Lambda B/ Lambda^-1) = Lambda C/ Lambda^-1
and, read on the channels of C/ = s 1 + c_a.alpha + c_S.Sigma (c = cosh psi = gamma, sh = sinh psi = beta gamma):
    s, c_a,I, c_S,I unchanged
    c_a,J' = c c_a,J - sh c_S,K     c_S,K' = c c_S,K - sh c_a,J
    c_a,K' = c c_a,K + sh c_S,J     c_S,J' = c c_S,J + sh c_a,K

Checked exactly with SymPy (z = exp(psi/2), so cosh and sinh are rational):
  1  block form: C/ = A/B/ = diag(-A B^T, -A^T B); C/^L = Lambda C/ Lambda^-1 = diag(-U^-1(A B^T)U, -U(A^T B)U^-1);
     the lower block is minus the Pauli boost of the bilinear, (A^L)^T B^L = U (A^T B) U^-1
  2  the basis: Lambda X Lambda^-1 = X for X = 1, alpha_I, Sigma_I, chi (they commute with alpha_I); = Lambda^2 X for
     alpha_J, alpha_K, Sigma_J, Sigma_K (they anticommute), with
        Lambda alpha_J Lambda^-1 = c alpha_J - sh Sigma_K,  Lambda Sigma_K Lambda^-1 = c Sigma_K - sh alpha_J
        Lambda alpha_K Lambda^-1 = c alpha_K + sh Sigma_J,  Lambda Sigma_J Lambda^-1 = c Sigma_J + sh alpha_K
  3  the channel rule above, for a general even element (and the chi channel is invariant)
  4  the same rule from the boosted vectors: the bilinear of A/^L and B/^L
  5  invariants: s, c_a^2 - c_S^2 and c_a.c_S
  6  complex form: Sigma_i = -chi alpha_i, so the bivector is (c_a - chi c_S).alpha; with Cc = c_a - chi c_S the boost is a
     rotation about I through the imaginary angle chi psi: Cc_J' = c Cc_J - sh chi Cc_K, Cc_K' = c Cc_K + sh chi Cc_J
  7  Pauli correspondence: c_a = c_sigma, c_S = c_K of C = A^T B; the Pauli rule c_K,J' = g(c_K,J + b c_s,K) is the rule above

Usage
  python bq_dirac_bilinear_lorentz.py
"""
import sympy as sp
from bq_dirac_table import EL, P, blk, NAMES

Z = sp.Symbol('z', positive=True)
ch_ = lambda z: (z + 1 / z) / 2
sh_ = lambda z: (z - 1 / z) / 2
BETA = [EL['b0'], EL['bI'], EL['bJ'], EL['bK']]
KP = [P['I'], P['J'], P['K']]
SP = [P['sI'], P['sJ'], P['sK']]
sy = lambda s: sp.symbols(s, real=True)


def report(name, ok):
    print(f'  {name:<88} {"IDENTICAL (exact)" if ok else "DIFFER"}')
    return ok


def zero(e):
    if isinstance(e, sp.MatrixBase):
        return all(zero(v) for v in e)
    return sp.cancel(sp.together(sp.expand(e))) == 0


def channels(M):
    return {n: sp.expand((EL[n] * M).trace() / (EL[n] * EL[n]).trace()) for n in NAMES}


def lin(cs, ks):
    M = sp.zeros(4)
    for x, k in zip(cs, ks):
        M += x * EL[k]
    return M


def adjT(M):
    return P['sJ'] * M.conjugate() * P['sJ']


def pdec(M):
    a, d, u, l = (sp.expand(M[0, 0]), sp.expand(M[1, 1]), sp.expand(M[0, 1]), sp.expand(M[1, 0]))
    return {'1': sp.re(a + d) / 2, 'T': sp.im(a + d) / 2, 'I': sp.im(a - d) / 2, 'J': sp.re(u - l) / 2, 'K': sp.im(u + l) / 2,
            'sI': sp.re(a - d) / 2, 'sJ': sp.im(l - u) / 2, 'sK': sp.re(u + l) / 2}


def main():
    ok = True
    a, b = sy('a0:4'), sy('b0:4')
    A, B = lin(a, ['b0', 'bI', 'bJ', 'bK']), lin(b, ['b0', 'bI', 'bJ', 'bK'])
    Ap = a[0] * P['T'] + sum((a[i + 1] * KP[i] for i in range(3)), sp.zeros(2))
    Bp = b[0] * P['T'] + sum((b[i + 1] * KP[i] for i in range(3)), sp.zeros(2))
    U, Ui = sp.diag(Z, 1 / Z), sp.diag(1 / Z, Z)
    L, Li = blk(Ui, sp.zeros(2), sp.zeros(2), U), blk(U, sp.zeros(2), sp.zeros(2), Ui)
    c, sh = ch_(Z**2), sh_(Z**2)
    C = A * B

    print('1  Block form and the Pauli bilinear inside')
    ok &= report('C/ = A/B/ = diag(-A B^T, -A^T B)', zero(C - blk(-Ap * adjT(Bp), sp.zeros(2), sp.zeros(2), -adjT(Ap) * Bp)))
    CL = (L * A * Li) * (L * B * Li)
    ok &= report('(Lambda A/ Lambda^-1)(Lambda B/ Lambda^-1) = Lambda C/ Lambda^-1', zero(CL - L * C * Li))
    ApL, BpL = Ui * Ap * Ui, Ui * Bp * Ui
    ok &= report('lower block: -U (A^T B) U^-1 = -(A^L)^T B^L, the Pauli boost of the bilinear',
                 zero(CL[2:4, 2:4] + U * adjT(Ap) * Bp * Ui) and zero(U * adjT(Ap) * Bp * Ui - adjT(ApL) * BpL))
    ok &= report('upper block: -U^-1 (A B^T) U = -A^L (B^L)^T', zero(CL[0:2, 0:2] + Ui * Ap * adjT(Bp) * U) and zero(Ui * Ap * adjT(Bp) * U - ApL * adjT(BpL)))

    print('2  The basis elements of the even algebra')
    for k in ['one', 'aI', 'SI', 'chi']:
        ok &= report(f'Lambda {k} Lambda^-1 = {k}  (commutes with alpha_I)', zero(L * EL[k] * Li - EL[k]))
    for k in ['aJ', 'aK', 'SJ', 'SK']:
        ok &= report(f'{k} anticommutes with alpha_I, so Lambda {k} Lambda^-1 = Lambda^2 {k}',
                     zero(EL['aI'] * EL[k] + EL[k] * EL['aI']) and zero(L * EL[k] * Li - L * L * EL[k]))
    rules = {'aJ': c * EL['aJ'] - sh * EL['SK'], 'SK': c * EL['SK'] - sh * EL['aJ'],
             'aK': c * EL['aK'] + sh * EL['SJ'], 'SJ': c * EL['SJ'] + sh * EL['aK']}
    ok &= report('Lambda aJ = c aJ - sh SK, SK -> c SK - sh aJ, aK -> c aK + sh SJ, SJ -> c SJ + sh aK',
                 all(zero(L * EL[k] * Li - v) for k, v in rules.items()))
    ok &= report('via the table: alpha_I alpha_J = Sigma_K, alpha_I Sigma_K = alpha_J, alpha_I alpha_K = -Sigma_J, alpha_I Sigma_J = -alpha_K',
                 zero(EL['aI'] * EL['aJ'] - EL['SK']) and zero(EL['aI'] * EL['SK'] - EL['aJ']) and zero(EL['aI'] * EL['aK'] + EL['SJ']) and zero(EL['aI'] * EL['SJ'] + EL['aK']))

    print('3  The channel rule for a general even element')
    s, h = sy('s h')
    p, q = sy('p1:4'), sy('q1:4')
    G = s * EL['one'] + lin(p, ['aI', 'aJ', 'aK']) + lin(q, ['SI', 'SJ', 'SK']) + h * EL['chi']
    want = {'one': s, 'chi': h, 'aI': p[0], 'SI': q[0], 'aJ': c * p[1] - sh * q[2], 'SK': c * q[2] - sh * p[1],
            'aK': c * p[2] + sh * q[1], 'SJ': c * q[1] + sh * p[2]}
    chL = channels(L * G * Li)
    ok &= report("s, chi, p_I, q_I unchanged; p_J' = c p_J - sh q_K, q_K' = c q_K - sh p_J, p_K' = c p_K + sh q_J, q_J' = c q_J + sh p_K",
                 all(zero(chL[k] - v) for k, v in want.items()) and all(zero(chL[k]) for k in NAMES if k not in want))

    print('4  The same rule from the boosted vectors')
    ch0 = channels(C)
    s0, pa, qa = ch0['one'], [ch0['aI'], ch0['aJ'], ch0['aK']], [ch0['SI'], ch0['SJ'], ch0['SK']]
    w2 = {'one': s0, 'aI': pa[0], 'SI': qa[0], 'aJ': c * pa[1] - sh * qa[2], 'SK': c * qa[2] - sh * pa[1],
          'aK': c * pa[2] + sh * qa[1], 'SJ': c * qa[1] + sh * pa[2]}
    chCL = channels(CL)
    ok &= report('channels of A/^L B/^L = the rule applied to the channels of A/B/', all(zero(chCL[k] - v) for k, v in w2.items()))

    print('5  Invariants')
    pv, qv = sp.Matrix(p), sp.Matrix(q)
    pL = sp.Matrix([chL['aI'], chL['aJ'], chL['aK']])
    qL = sp.Matrix([chL['SI'], chL['SJ'], chL['SK']])
    ok &= report("s' = s, p'^2 - q'^2 = p^2 - q^2, p'.q' = p.q", zero(pL.dot(pL) - qL.dot(qL) - pv.dot(pv) + qv.dot(qv)) and zero(pL.dot(qL) - pv.dot(qv)))

    print('6  Complex form')
    ok &= report('Sigma_i = -chi alpha_i, chi^2 = -1, chi commutes with alpha_i',
                 all(zero(EL[sk] + EL['chi'] * EL[ak]) for sk, ak in zip(['SI', 'SJ', 'SK'], ['aI', 'aJ', 'aK']))
                 and zero(EL['chi'] * EL['chi'] + EL['one']) and all(zero(EL['chi'] * EL[k] - EL[k] * EL['chi']) for k in ['aI', 'aJ', 'aK']))
    X = EL['chi']
    Cc = [p[i] * EL['one'] - q[i] * X for i in range(3)]                     # complex components as matrices
    CcL = [chL['aI'] * EL['one'] - chL['SI'] * X, chL['aJ'] * EL['one'] - chL['SJ'] * X, chL['aK'] * EL['one'] - chL['SK'] * X]
    ok &= report("Cc = c_a - chi c_S: Cc_I' = Cc_I, Cc_J' = c Cc_J - sh chi Cc_K, Cc_K' = c Cc_K + sh chi Cc_J",
                 zero(CcL[0] - Cc[0]) and zero(CcL[1] - (c * Cc[1] - sh * X * Cc[2])) and zero(CcL[2] - (c * Cc[2] + sh * X * Cc[1])))

    print('7  Pauli correspondence')
    PC = pdec(adjT(Ap) * Bp)
    ok &= report('c_alpha = c_sigma and c_Sigma = c_K of the Pauli bilinear A^T B; s = -c_1',
                 zero(ch0['one'] + PC['1']) and all(zero(ch0[d] - PC[pk]) for d, pk in zip(['aI', 'aJ', 'aK'], ['sI', 'sJ', 'sK']))
                 and all(zero(ch0[d] - PC[pk]) for d, pk in zip(['SI', 'SJ', 'SK'], ['I', 'J', 'K'])))
    PL = pdec(U * adjT(Ap) * Bp * Ui)
    ok &= report("Pauli rule c_K,J' = g(c_K,J + b c_s,K), c_s,K' = g(c_s,K + b c_K,J), ... is the rule above",
                 zero(PL['J'] - (c * PC['J'] + sh * PC['sK'])) and zero(PL['sK'] - (c * PC['sK'] + sh * PC['J']))
                 and zero(PL['K'] - (c * PC['K'] - sh * PC['sJ'])) and zero(PL['sJ'] - (c * PC['sJ'] - sh * PC['K']))
                 and zero(chCL['SJ'] - PL['J']) and zero(chCL['aK'] - PL['sK']))

    print('\nAll checks:', 'IDENTICAL' if ok else 'SOME DIFFER')
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
