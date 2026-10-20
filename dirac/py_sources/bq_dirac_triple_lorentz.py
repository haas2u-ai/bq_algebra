"""
bq_dirac_triple_lorentz.py -- Lorentz covariance of the Dirac-level triple product E/ = D/(A/B/), as on the page
BQ_Dirac_Triple_Product_Covariance.html.

With a/^L = Lambda a/ Lambda^-1 for every vector, Lambda = diag(U^-1, U) = exp(-psi alpha_I/2):
    E/^L = D/^L (A/^L B/^L) = Lambda D/ Lambda^-1 Lambda C/ Lambda^-1 = Lambda E/ Lambda^-1
The triple is odd, E/ = e0 b0 + e.beta + r0 b~0 + r.beta~, and both sectors are boosted as four-vectors:
    e0' = g(e0 - b e_I), e_I' = g(e_I - b e0),   r0' = g(r0 - b r_I), r_I' = g(r_I - b r0),   J, K unchanged.
An equation D/C/ = X/ therefore keeps its form in every frame.

Checked exactly with SymPy (z = exp(psi/2)):
  1  block form E/ = [[0, -D(A^T B)], [D^T A B^T, 0]]; E/^L = (Lambda D/ Lambda^-1)(Lambda A/ Lambda^-1)(Lambda B/ Lambda^-1)
     = Lambda E/ Lambda^-1; its upper block -U^-1 (D A^T B) U^-1 is minus the Pauli rule E^L = U^-1 E U^-1
  2  the pseudovector basis: Lambda beta~_mu Lambda^-1 = beta~_nu Lambda^nu_mu, because chi commutes with Lambda
  3  the channel rule for a general odd element: both sectors as four-vectors
  4  invariants: with E/ = E/_v + E/_p (vector and pseudovector parts), E/_v^2 = -(e0^2 - e^2) 1, E/_p^2 = -(r0^2 - r^2) 1
     and the chi channel of E/_v E/_p is e0 r0 - e.r; all three are unchanged by the boost
  5  Pauli correspondence: [E/]_b0 = -e_T, [E/]_beta = -e_K, [E/]_b~0 = e_1, [E/]_beta~ = e_sigma (minquat sector -> vector,
     pauliquat sector -> pseudovector), and the Pauli sector rules follow
  6  the four-gradient: for a field C/(x) (linear in x with general coefficients, which covers every first derivative at
     a point), the boosted field C/'(x') = Lambda C/(x) Lambda^-1 and the derivatives taken in the new coordinates
     x' = Lambda x give d/' C/' = Lambda (d/ C/) Lambda^-1: the operator triples (Maxwell d/B/ = -mu0 J/, d/(J/A/),
     d/(U/G/), d/H/) are covariant
  7  Maxwell: a pseudovector sector that vanishes in one frame vanishes in all

Usage
  python bq_dirac_triple_lorentz.py
"""
import sympy as sp
from bq_dirac_table import EL, P, blk, NAMES

Z = sp.Symbol('z', positive=True)
ch_ = lambda z: (z + 1 / z) / 2
sh_ = lambda z: (z - 1 / z) / 2
BETA = [EL['b0'], EL['bI'], EL['bJ'], EL['bK']]
BT = [EL['bt0'], EL['btI'], EL['btJ'], EL['btK']]
KK = [P['T'], P['I'], P['J'], P['K']]
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


def lin(cs, Ms):
    M = sp.zeros(*Ms[0].shape)
    for x, X in zip(cs, Ms):
        M += x * X
    return M


def adjT(M):
    return P['sJ'] * M.conjugate() * P['sJ']


def pdec(M):
    a, d, u, l = (sp.expand(M[0, 0]), sp.expand(M[1, 1]), sp.expand(M[0, 1]), sp.expand(M[1, 0]))
    return {'1': sp.re(a + d) / 2, 'T': sp.im(a + d) / 2, 'I': sp.im(a - d) / 2, 'J': sp.re(u - l) / 2, 'K': sp.im(u + l) / 2,
            'sI': sp.re(a - d) / 2, 'sJ': sp.im(l - u) / 2, 'sK': sp.re(u + l) / 2}


def main():
    ok = True
    U, Ui = sp.diag(Z, 1 / Z), sp.diag(1 / Z, Z)
    L, Li = blk(Ui, sp.zeros(2), sp.zeros(2), U), blk(U, sp.zeros(2), sp.zeros(2), Ui)
    c, sh = ch_(Z**2), sh_(Z**2)
    Lam = sp.Matrix([[c, -sh, 0, 0], [-sh, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    d, a, b = sy('d0:4'), sy('a0:4'), sy('b0:4')
    Dm, Am, Bm = lin(d, BETA), lin(a, BETA), lin(b, BETA)
    pv = lambda x: lin(x, KK)
    Dp, Ap, Bp = pv(d), pv(a), pv(b)
    E = Dm * Am * Bm

    print('1  One similarity for the triple, and the Pauli rule inside')
    ok &= report('E/ = D/A/B/ = [[0, -D(A^T B)], [D^T A B^T, 0]]', zero(E - blk(sp.zeros(2), -Dp * adjT(Ap) * Bp, adjT(Dp) * Ap * adjT(Bp), sp.zeros(2))))
    EL_ = (L * Dm * Li) * (L * Am * Li) * (L * Bm * Li)
    ok &= report('(Lambda D/ Lambda^-1)(Lambda A/ Lambda^-1)(Lambda B/ Lambda^-1) = Lambda E/ Lambda^-1', zero(EL_ - L * E * Li))
    DL, AL, BL = Ui * Dp * Ui, Ui * Ap * Ui, Ui * Bp * Ui
    ok &= report('upper block -U^-1 (D A^T B) U^-1 = -D^L (A^L)^T B^L: the Pauli rule E^L = U^-1 E U^-1',
                 zero(EL_[0:2, 2:4] + Ui * Dp * adjT(Ap) * Bp * Ui) and zero(Ui * Dp * adjT(Ap) * Bp * Ui - DL * adjT(AL) * BL))

    print('2  The basis of the odd algebra')
    ok &= report('chi commutes with Lambda', zero(EL['chi'] * L - L * EL['chi']))
    ok &= report('Lambda beta_mu Lambda^-1 = beta_nu Lambda^nu_mu  (vector lemma)',
                 all(zero(L * BETA[m] * Li - lin([Lam[n, m] for n in range(4)], BETA)) for m in range(4)))
    ok &= report('Lambda beta~_mu Lambda^-1 = chi Lambda beta_mu Lambda^-1 = beta~_nu Lambda^nu_mu',
                 all(zero(L * BT[m] * Li - lin([Lam[n, m] for n in range(4)], BT)) for m in range(4)))

    print('3  The channel rule for a general odd element')
    e, r = sy('e0:4'), sy('r0:4')
    O = lin(e, BETA) + lin(r, BT)
    ch = channels(L * O * Li)
    want = {'b0': c * e[0] - sh * e[1], 'bI': c * e[1] - sh * e[0], 'bJ': e[2], 'bK': e[3],
            'bt0': c * r[0] - sh * r[1], 'btI': c * r[1] - sh * r[0], 'btJ': r[2], 'btK': r[3]}
    ok &= report("e0' = g(e0 - b e_I), e_I' = g(e_I - b e0), r0' = g(r0 - b r_I), r_I' = g(r_I - b r0), J, K unchanged",
                 all(zero(ch[k] - v) for k, v in want.items()) and all(zero(ch[k]) for k in NAMES if k not in want))

    print('4  Invariants')
    Ev, Ep = lin(e, BETA), lin(r, BT)
    m4 = lambda u, v: u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3]
    ok &= report('E/_v E/_v = -(e0^2 - e^2) 1, E/_p E/_p = -(r0^2 - r^2) 1, [E/_v E/_p]_chi = e0 r0 - e.r',
                 zero(Ev * Ev + m4(e, e) * EL['one']) and zero(Ep * Ep + m4(r, r) * EL['one']) and zero(channels(Ev * Ep)['chi'] - m4(e, r)))
    w = [ch['b0'], ch['bI'], ch['bJ'], ch['bK']]
    x = [ch['bt0'], ch['btI'], ch['btJ'], ch['btK']]
    ok &= report("e0^2 - e^2, r0^2 - r^2 and e0 r0 - e.r unchanged by the boost",
                 zero(m4(w, w) - m4(e, e)) and zero(m4(x, x) - m4(r, r)) and zero(m4(w, x) - m4(e, r)))

    print('5  Pauli correspondence')
    PE = pdec(Dp * adjT(Ap) * Bp)
    chE = channels(E)
    ok &= report('[E/]_b0 = -e_T, [E/]_beta = -e_K, [E/]_b~0 = e_1, [E/]_beta~ = e_sigma',
                 zero(chE['b0'] + PE['T']) and all(zero(chE[k] + PE[p]) for k, p in zip(['bI', 'bJ', 'bK'], 'IJK'))
                 and zero(chE['bt0'] - PE['1']) and all(zero(chE[k] - PE[p]) for k, p in zip(['btI', 'btJ', 'btK'], ['sI', 'sJ', 'sK'])))
    PEL = pdec(Ui * Dp * adjT(Ap) * Bp * Ui)
    ok &= report("so the Pauli sectors are four-vectors: e_T' = g(e_T - b e_K,I), e_1' = g(e_1 - b e_s,I), ...",
                 zero(PEL['T'] - (c * PE['T'] - sh * PE['I'])) and zero(PEL['I'] - (c * PE['I'] - sh * PE['T']))
                 and zero(PEL['1'] - (c * PE['1'] - sh * PE['sI'])) and zero(PEL['sI'] - (c * PE['sI'] - sh * PE['1'])))

    print('6  The four-gradient on a field')
    X = sy('x0:4')                                    # x0 = ct
    s0 = sy('s0'); kc = {k: sy(f'g_{k}_0:4') for k in ['one', 'aI', 'aJ', 'aK', 'SI', 'SJ', 'SK']}
    const = {k: sp.Symbol(f'c_{k}', real=True) for k in kc}
    field = lambda xx: sum(((const[k] + sum(kc[k][m] * xx[m] for m in range(4))) * EL[k] for k in kc), sp.zeros(4))
    dslash = lambda M, xx: -EL['b0'] * M.diff(xx[0]) + sum((BETA[i] * M.diff(xx[i]) for i in range(1, 4)), sp.zeros(4))
    Xp = sy('y0:4')                                   # new coordinates x' = Lambda x
    Xold = list(Lam.inv() * sp.Matrix(Xp))           # x = Lambda^-1 x'
    Fp = L * field(Xold) * Li                         # C/'(x') = Lambda C/(x(x')) Lambda^-1
    lhs = dslash(Fp, Xp)
    rhs = L * dslash(field(X), X) * Li
    ok &= report("d/' C/'(x') = Lambda (d/ C/) Lambda^-1 for a general field (first derivatives arbitrary)", zero(lhs - rhs))
    print('7  Maxwell in every frame')
    sub0 = {ri: 0 for ri in r}
    ok &= report("a vanishing pseudovector sector (r0, r) = 0 stays zero: the homogeneous equations hold in every frame",
                 all(zero(ch[k].subs(sub0)) for k in ['bt0', 'btI', 'btJ', 'btK']))
    Bf = sum(((const[k] + sum(kc[k][m] * X[m] for m in range(4))) * EL[k] for k in ['aI', 'aJ', 'aK', 'SI', 'SJ', 'SK']), sp.zeros(4))
    BfP = L * Bf.subs(dict(zip(X, Xold)), simultaneous=True) * Li
    chP = channels(dslash(BfP, Xp)); ch0 = channels(dslash(Bf, X))
    ok &= report("for a pure bivector field B/: the boosted field's d/'B/' has pseudovector part Lambda^nu_mu r^mu",
                 all(zero(chP[k] - sum(Lam[n, m] * ch0[kk] for m, kk in enumerate(['bt0', 'btI', 'btJ', 'btK'])))
                     for n, k in enumerate(['bt0', 'btI', 'btJ', 'btK'])))
    print('\nAll checks:', 'IDENTICAL' if ok else 'SOME DIFFER')
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
