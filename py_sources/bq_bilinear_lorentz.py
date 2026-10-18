"""
bq_bilinear_lorentz.py -- the Lorentz boost of the bilinear C = A^T B, as on the page
BQ_Bilinear_Lorentz_Transformation.html.

Boosting both factors along I^ (A^L = U^-1 A U^-1, (A^L)^T = U A^T U) gives
    C^L = (A^L)^T B^L = U (A^T B) U^-1 = U C U^-1 ,   U = exp(psi sigma_I/2)
and, read on the channels (gamma = cosh psi, beta gamma = sinh psi),
    c1, c_K,I, c_sigma,I unchanged
    c_K,J' = gamma(c_K,J + beta c_s,K)    c_s,K' = gamma(c_s,K + beta c_K,J)
    c_K,K' = gamma(c_K,K - beta c_s,J)    c_s,J' = gamma(c_s,J - beta c_K,K)

For every pair the script checks exactly (symbolic rapidity psi; a four-gradient acts on
everything to its right, and on the test field psi when it stands last):
  1  matrix route : (A^L)^T B^L as 2x2 matrices equals U C U^-1
  2  rule route   : the bilinear rule applied to the boosted coordinates equals the gamma formulas
  3  U C U^-1 decomposed on the basis equals the gamma formulas
  4  inverse U^-1 C^L U = C, and the time adjoint (C^L)^T = U^-1 C^T U  (C^T flips the sigma channel)
  5  invariants c1, c_s^2 - c_K^2 and c_K . c_s
  6  complex form: with Cc = c_s + i c_K, Cc^L = Lambda_C Cc,  Lambda_C = [[1,0,0],[0,g,i bg],[0,-i bg,g]]
--basis checks U X U^-1 for the eight basis elements by the multiplication table
(U = ch 1^ + sh sigma_I) and by matrices, and sigma_b (Lambda_C)^b_a = U sigma_a U^-1.

Usage
  python bq_bilinear_lorentz.py R P        one pair
  python bq_bilinear_lorentz.py --basis    the basis-element rules (steps 3-5 and 7 of the page)
  python bq_bilinear_lorentz.py --all      all 169 pairs and the basis rules
Vectors: R dR V U P D(=four-gradient) A J G F a b d
"""
import sys
import sympy as sp
from bq_core import (VECTORS, BASIS, ORDER, PRETTY, SIGMA, Op, bilinear_rule, decompose, left_multiply,
                     eval_matrix, psi, table_product, show, show_channels)

w = sp.Symbol('ψ', real=True)
U = sp.diag(sp.exp(w / 2), sp.exp(-w / 2))
Ui = sp.diag(sp.exp(-w / 2), sp.exp(w / 2))
g, bg = sp.cosh(w), sp.sinh(w)
ONE, KB = BASIS['1'], [BASIS['I'], BASIS['J'], BASIS['K']]


def z(e):
    e = sp.expand(sp.sympify(e).rewrite(sp.exp))
    return e == 0 or sp.simplify(e) == 0


def same(c1, c2):
    return z(c1['1'] - c2['1']) and z(c1['T'] - c2['T']) and \
        all(z(p - q) for p, q in zip(c1['K'] + c1['s'], c2['K'] + c2['s']))


def lin(a, x, b, y):
    """a x + b y for two components (operators combine into one operator)."""
    if isinstance(x, Op) or isinstance(y, Op):
        val = lambda q, e: q(e) if isinstance(q, Op) else q * e
        return Op(lambda e: a * val(x, e) + b * val(y, e), f'{a}*{x}+{b}*{y}')
    return a * x + b * y


def boost(comps, sgn=1):
    """Coordinates of A^L (sgn = +1) or A^{L^-1} (sgn = -1)."""
    c0, c1 = comps[0], comps[1]
    return [lin(g, c0, -sgn * bg, c1), lin(g, c1, -sgn * bg, c0)] + comps[2:]


def time_reversed(comps):
    c0 = comps[0]
    return [c0.scaled(-1) if isinstance(c0, Op) else -c0] + comps[1:]


def gamma_rule(c, sgn=1):
    """The channel form of U C U^-1 (sgn = +1) or U^-1 C U (sgn = -1)."""
    kI, kJ, kK = c['K']
    sI, sJ, sK = c['s']
    b = sgn * bg
    return {'1': c['1'], 'T': c['T'], 'K': [kI, g * kJ + b * sK, g * kK - b * sJ], 's': [sI, g * sJ - b * kK, g * sK + b * kJ]}


def chan_matrix(c):
    return c['1'] * ONE + c['T'] * BASIS['T'] + sum((c['K'][i] * KB[i] + c['s'][i] * SIGMA[i] for i in range(3)), sp.zeros(2, 2))


def adjoint(c):
    return {'1': c['1'], 'T': c['T'], 'K': c['K'], 's': [-v for v in c['s']]}


def report(name, ok):
    print(f'  {name:<64} {"IDENTICAL (exact)" if ok else "DIFFER"}')
    return ok


def check_pair(ka, kb, verbose=True):
    A, B = VECTORS[ka], VECTORS[kb]
    C = left_multiply(A.time_reversed(), eval_matrix(B.comps))
    c = decompose(C)
    AL, BL = boost(A.comps), boost(B.comps)
    CL_mat = decompose(left_multiply(time_reversed(AL), eval_matrix(BL)))
    CL_rule = bilinear_rule(AL, [q(psi) if isinstance(q, Op) else q for q in BL])
    UCU = decompose(U * C * Ui)
    gr = gamma_rule(c)
    ok = {}
    ok['matrix'] = same(CL_mat, UCU)
    ok['rule'] = same(CL_rule, gr)
    ok['gamma'] = same(UCU, gr)
    ok['inverse'] = same(decompose(Ui * (U * C * Ui) * U), c) and same(decompose(Ui * C * U), gamma_rule(c, -1))
    ok['adjoint'] = same(adjoint(UCU), decompose(Ui * chan_matrix(adjoint(c)) * U))
    inv = lambda q: (q['1'], sum(v**2 for v in q['s']) - sum(v**2 for v in q['K']), sum(p * q2 for p, q2 in zip(q['K'], q['s'])))
    ok['invariants'] = all(z(p - q) for p, q in zip(inv(UCU), inv(c)))
    Cc = [c['s'][i] + sp.I * c['K'][i] for i in range(3)]
    CcL = [UCU['s'][i] + sp.I * UCU['K'][i] for i in range(3)]
    Lam = sp.Matrix([[1, 0, 0], [0, g, sp.I * bg], [0, -sp.I * bg, g]])
    ok['complex'] = all(z(p - q) for p, q in zip(CcL, list(Lam * sp.Matrix(Cc))))
    if verbose:
        print(f'A = {A.name}: {A.what}\nB = {B.name}: {B.what}' + ('\n(an operator in B acts on the test field psi)' if B.is_op else ''))
        print(show_channels(c, f'C = {A.name}^T {B.name}'))
        print(show_channels({k: ([sp.simplify(v) for v in val] if isinstance(val, list) else sp.simplify(val)) for k, val in gr.items()},
                            f'C^L = U C U^-1   (cosh(ψ) = gamma, sinh(ψ) = beta gamma)'))
        report('(A^L)^T B^L by 2x2 matrices = U C U^-1', ok['matrix'])
        report('bilinear rule on the boosted coordinates = gamma formulas', ok['rule'])
        report('U C U^-1 decomposed = gamma formulas', ok['gamma'])
        report('U^-1 C^L U = C, and U^-1 C U = gamma formulas with -beta', ok['inverse'])
        report('(C^L)^T = U^-1 C^T U = (C^T)^{L^-1}', ok['adjoint'])
        report('c1, c_s^2 - c_K^2 and c_K.c_s unchanged', ok['invariants'])
        report('complex form: (c_s + i c_K)^L = Lambda_C (c_s + i c_K)', ok['complex'])
    return all(ok.values())


def check_basis(verbose=True):
    ch, sh = sp.symbols('ch sh', real=True)
    Uh, Uih = {'1': ch, 'sI': sh}, {'1': ch, 'sI': -sh}        # U = ch 1^ + sh sigma_I, basis kept explicit
    expect = {'1': {'1': 1}, 'T': {'T': 1}, 'I': {'I': 1}, 'sI': {'sI': 1},
              'J': {'J': g, 'sK': bg}, 'sK': {'sK': g, 'J': bg}, 'K': {'K': g, 'sJ': -bg}, 'sJ': {'sJ': g, 'K': -bg}}
    all_ok = True
    if verbose:
        print('U X U^-1 = (ch 1^ + sh sigma_I) X (ch 1^ - sh sigma_I), ch = cosh(psi/2), sh = sinh(psi/2)')
    for X in ['1', 'T', 'I', 'sI', 'J', 'sK', 'K', 'sJ']:
        out = table_product(table_product(Uh, {X: sp.Integer(1)}), Uih)
        res = {k: sp.factor(sp.expand(out[k])) for k in ORDER if sp.expand(out[k]) != 0}
        sub = {k: e.subs({ch: sp.cosh(w / 2), sh: sp.sinh(w / 2)}) for k, e in res.items()}
        ok_t = all(z(sub.get(k, 0) - expect[X].get(k, 0)) for k in ORDER)
        rhs = sum((cf * BASIS[k] for k, cf in expect[X].items()), sp.zeros(2, 2))
        ok_m = all(z(e) for e in (U * BASIS[X] * Ui - rhs))
        all_ok &= ok_t and ok_m
        if verbose:
            lhs = ' + '.join(f'({e}) {PRETTY[k]}' for k, e in res.items())
            tgt = ' + '.join(f'{e} {PRETTY[k]}' for k, e in expect[X].items()).replace('+ -', '- ')
            print(f'  U {PRETTY[X]} U^-1 = {lhs}\n      = {tgt}   table: {"IDENTICAL" if ok_t else "DIFFER"}, matrices: {"IDENTICAL" if ok_m else "DIFFER"}')
    Lam = sp.Matrix([[1, 0, 0], [0, g, sp.I * bg], [0, -sp.I * bg, g]])
    ok_c = all(all(z(e) for e in (U * SIGMA[b] * Ui - sum((Lam[a, b] * SIGMA[a] for a in range(3)), sp.zeros(2, 2)))) for b in range(3))
    all_ok &= ok_c
    if verbose:
        print(f'  sigma_b (Lambda_C)^b_a = U sigma_a U^-1 (Minkowski 1908):  {"IDENTICAL (exact)" if ok_c else "DIFFER"}')
    return all_ok


if __name__ == '__main__':
    args = sys.argv[1:]
    if args == ['--all']:
        keys = list(VECTORS)
        bad = [(a, b) for a in keys for b in keys if not check_pair(a, b, verbose=False)]
        okb = check_basis(verbose=False)
        print(f'{len(keys)**2} pairs checked (matrix route, rule route, gamma formulas, inverse, time adjoint, '
              f'invariants, complex form), {len(bad)} failures', bad or '')
        print(f'basis-element rules and Minkowski form: {"IDENTICAL" if okb else "DIFFER"}')
        sys.exit(0 if not bad and okb else 1)
    elif args == ['--basis']:
        sys.exit(0 if check_basis() else 1)
    elif len(args) == 2 and all(a in VECTORS for a in args):
        sys.exit(0 if check_pair(*args) else 1)
    else:
        print(__doc__)
        print('Vectors:', ', '.join(VECTORS))
