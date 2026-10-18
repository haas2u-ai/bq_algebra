"""
bq_triple_lorentz.py -- Lorentz covariance of the triple product E = D(A^T B), as on the page
BQ_Triple_Product_Covariance.html.

Boosting the three factors along I^ (D^L = U^-1 D U^-1, C^L = (A^L)^T B^L = U C U^-1) gives
    E^L = D^L C^L = U^-1 (D C) U^-1 = U^-1 E U^-1 ,   U = exp(psi sigma_I/2)
so E is boosted like a four-vector, and so are both of its sectors:
    minquat   (e_T, e_K):   e_T' = gamma(e_T - beta e_K,I),  e_K,I' = gamma(e_K,I - beta e_T)
    pauliquat (e_1, e_s):   e_1' = gamma(e_1 - beta e_s,I),  e_s,I' = gamma(e_s,I - beta e_1)
with the transverse components unchanged. An equation D C = X therefore keeps its form,
D^L C^L = X^L: Maxwell's equations dB = mu0 J and the Lorentz force J B = F are covariant.

For every triple the script checks exactly (symbolic rapidity psi; a four-gradient acts on
everything to its right, and on the test field psi when it stands last):
  1  D^L (A^L)^T B^L, multiplied as 2x2 matrices, equals U^-1 E U^-1
  2  U^-1 E U^-1 decomposed equals the gamma formulas of both sectors
  3  the inverse U (E^L) U = E
  4  the invariants e_T^2 - e_K^2, e_1^2 - e_s^2 and e_T e_1 - e_K . e_s
--basis checks U^-1 X U^-1 for all eight basis elements by the multiplication table
(U^-1 = ch 1^ - sh sigma_I) and by matrices: the LT lemma for both bases (T^, I^, J^, K^)
and (1^, sigma_I, sigma_J, sigma_K).
--maxwell checks with a general potential A(x) that the pauliquat sector of d(d^T A) vanishes
identically (the homogeneous Maxwell equations) and that its minquat sector is a four-vector.

Usage
  python bq_triple_lorentz.py D D A        one triple (here dB with B = d^T A)
  python bq_triple_lorentz.py J D A        the Lorentz-force product JB
  python bq_triple_lorentz.py --basis
  python bq_triple_lorentz.py --maxwell
  python bq_triple_lorentz.py --all        proof for all 2197 triples: the 8 general triples (each factor a
                                           general field vector or the four-gradient) and the papers' equations
  python bq_triple_lorentz.py --every      every one of the 2197 triples separately (over an hour)
Vectors: R dR V U P D(=four-gradient) A J G F a b d
"""
import sys
import sympy as sp
from bq_core import VECTORS, BASIS, ORDER, PRETTY, MINQUAT, SIGMA, Op, decompose, left_multiply, eval_matrix, table_product, show

w = sp.Symbol('ψ', real=True)
U = sp.diag(sp.exp(w / 2), sp.exp(-w / 2))
Ui = sp.diag(sp.exp(-w / 2), sp.exp(w / 2))
g, bg = sp.cosh(w), sp.sinh(w)


q = sp.Symbol('q', positive=True)                  # q = exp(psi/2): every expression becomes rational in q


def z(e):
    e = sp.expand(sp.sympify(e).rewrite(sp.exp).subs(w, 2 * sp.log(q)))
    return e == 0 or sp.cancel(sp.together(e)) == 0


def same(c1, c2):
    return z(c1['1'] - c2['1']) and z(c1['T'] - c2['T']) and \
        all(z(p - q) for p, q in zip(c1['K'] + c1['s'], c2['K'] + c2['s']))


def lin(a, x, b, y):
    if isinstance(x, Op) or isinstance(y, Op):
        val = lambda q, e: q(e) if isinstance(q, Op) else q * e
        return Op(lambda e: a * val(x, e) + b * val(y, e), 'boosted')
    return a * x + b * y


def boost(comps, sgn=1):
    c0, c1 = comps[0], comps[1]
    return [lin(g, c0, -sgn * bg, c1), lin(g, c1, -sgn * bg, c0)] + comps[2:]


def time_reversed(comps):
    c0 = comps[0]
    return [c0.scaled(-1) if isinstance(c0, Op) else -c0] + comps[1:]


def gamma_rule(e):
    """The four-vector boost on both sectors: (T, K_I) and (1, sigma_I)."""
    return {'T': g * e['T'] - bg * e['K'][0], 'K': [g * e['K'][0] - bg * e['T'], e['K'][1], e['K'][2]],
            '1': g * e['1'] - bg * e['s'][0], 's': [g * e['s'][0] - bg * e['1'], e['s'][1], e['s'][2]]}


def invariants(e):
    return (e['T']**2 - sum(v**2 for v in e['K']), e['1']**2 - sum(v**2 for v in e['s']),
            e['T'] * e['1'] - sum(p * q for p, q in zip(e['K'], e['s'])))


def report(name, ok):
    print(f'  {name:<66} {"IDENTICAL (exact)" if ok else "DIFFER"}')
    return ok


def show_e(e, title):
    return '\n'.join([title, f"  e_1     : {show(e['1'])}", f"  e_T     : {show(e['T'])}",
                      f"  e_K     : ({', '.join(show(v) for v in e['K'])})", f"  e_sigma : ({', '.join(show(v) for v in e['s'])})"])


def chan_matrix(e):
    return e['1'] * BASIS['1'] + e['T'] * BASIS['T'] + sum((e['K'][i] * MINQUAT[i + 1] + e['s'][i] * SIGMA[i] for i in range(3)), sp.zeros(2, 2))


def mzero(M):
    return all(z(v) for v in M)


def check_triple(kd, ka, kb, verbose=True):
    D, A, B = VECTORS[kd], VECTORS[ka], VECTORS[kb]
    E = left_multiply(D.comps, left_multiply(A.time_reversed(), eval_matrix(B.comps)))
    e = decompose(E)
    DL, AL, BL = boost(D.comps), boost(A.comps), boost(B.comps)
    EL_prod = left_multiply(DL, left_multiply(time_reversed(AL), eval_matrix(BL)))
    EL = Ui * E * Ui
    ok = {'product': mzero(EL_prod - EL), 'gamma': mzero(EL - chan_matrix(gamma_rule(e))),
          'inverse': mzero(U * EL * U - E),
          'invariants': all(z(p - q2) for p, q2 in zip(invariants(gamma_rule(e)), invariants(e)))}
    if verbose:
        print(f'D = {D.name}: {D.what}\nA = {A.name}: {A.what}\nB = {B.name}: {B.what}')
        print(show_e(e, f'E = {D.name}({A.name}^T {B.name})'))
        print('E^L = U^-1 E U^-1, read on the channels (gamma = cosh psi, beta gamma = sinh psi):\n'
              '  minquat   e_T\' = gamma(e_T - beta e_K,I),  e_K,I\' = gamma(e_K,I - beta e_T),  e_K,J, e_K,K unchanged\n'
              '  pauliquat e_1\' = gamma(e_1 - beta e_s,I),  e_s,I\' = gamma(e_s,I - beta e_1),  e_s,J, e_s,K unchanged')
        report('D^L (A^L)^T B^L by 2x2 matrices = U^-1 E U^-1', ok['product'])
        report('U^-1 E U^-1 = gamma formulas on the minquat and pauliquat sectors', ok['gamma'])
        report('U E^L U = E', ok['inverse'])
        report('e_T^2 - e_K^2, e_1^2 - e_s^2, e_T e_1 - e_K.e_s unchanged', ok['invariants'])
    return all(ok.values())


def check_basis(verbose=True):
    ch, sh = sp.symbols('ch sh', real=True)
    Uih = {'1': ch, 'sI': -sh}                               # U^-1 = ch 1^ - sh sigma_I
    expect = {'T': {'T': g, 'I': -bg}, 'I': {'I': g, 'T': -bg}, 'J': {'J': 1}, 'K': {'K': 1},
              '1': {'1': g, 'sI': -bg}, 'sI': {'sI': g, '1': -bg}, 'sJ': {'sJ': 1}, 'sK': {'sK': 1}}
    all_ok = True
    if verbose:
        print('U^-1 X U^-1 = (ch 1^ - sh sigma_I) X (ch 1^ - sh sigma_I), ch = cosh(psi/2), sh = sinh(psi/2)')
    for X in ['T', 'I', 'J', 'K', '1', 'sI', 'sJ', 'sK']:
        out = table_product(table_product(Uih, {X: sp.Integer(1)}), Uih)
        res = {k: sp.factor(sp.expand(out[k])) for k in ORDER if sp.expand(out[k]) != 0}
        sub = {k: e.subs({ch: sp.cosh(w / 2), sh: sp.sinh(w / 2)}) for k, e in res.items()}
        ok_t = all(z(sub.get(k, 0) - expect[X].get(k, 0)) for k in ORDER)
        rhs = sum((cf * BASIS[k] for k, cf in expect[X].items()), sp.zeros(2, 2))
        ok_m = all(z(v) for v in (Ui * BASIS[X] * Ui - rhs))
        all_ok &= ok_t and ok_m
        if verbose:
            lhs = ' + '.join(f'({e}) {PRETTY[k]}' for k, e in res.items())
            tgt = ' + '.join(f'{e} {PRETTY[k]}' for k, e in expect[X].items()).replace('+ -', '- ')
            print(f'  U^-1 {PRETTY[X]} U^-1 = {lhs}\n      = {tgt}   table: {"IDENTICAL" if ok_t else "DIFFER"}, matrices: {"IDENTICAL" if ok_m else "DIFFER"}')
    if verbose:
        print('Lemma for both bases: K^_mu Lambda^mu_nu = U^-1 K^_nu U^-1 and sigma_mu Lambda^mu_nu = U^-1 sigma_nu U^-1:',
              'IDENTICAL' if all_ok else 'DIFFER')
    return all_ok


def check_maxwell(verbose=True):
    """d(d^T A) for a general potential: pauliquat sector = 0 identically; minquat = mu0 J."""
    E = left_multiply(VECTORS['D'].comps, left_multiply(VECTORS['D'].time_reversed(), eval_matrix(VECTORS['A'].comps)))
    e = decompose(E)
    zero_p = z(e['1']) and all(z(v) for v in e['s'])
    if verbose:
        print(show_e(e, 'd B = d(d^T A) for a general potential A(t, x, y, z) (no gauge chosen)'))
        report('pauliquat sector -div B, (1/c)(curl E + d_t B) vanishes identically', zero_p)
        print('  minquat sector = mu0 J (with the gauge term in the Lorenz gauge set to zero); it is boosted as a four-vector by --all')
    return zero_p


if __name__ == '__main__':
    args = sys.argv[1:]
    if args in (['--all'], ['--every']):
        if args == ['--all']:
            # E is linear in the coordinates of each factor, and every physical four-vector is a special case
            # of a general field four-vector (d, a, b here) or is the four-gradient D. Checking the 8 combinations
            # with each slot either general or D therefore proves the identity for all 13^3 = 2197 triples.
            slots = [('d', 'D'), ('a', 'D'), ('b', 'D')]
            cases = [(x, y, w2) for x in slots[0] for y in slots[1] for w2 in slots[2]]
            cases += [('D', 'D', 'A'), ('J', 'D', 'A'), ('U', 'D', 'G'), ('D', 'D', 'G'), ('D', 'J', 'A'), ('D', 'U', 'G')]
            label = '8 general triples (each factor a general field vector or the four-gradient) + 6 equations of the papers'
        else:
            keys = list(VECTORS)
            cases = [(a, b, c) for a in keys for b in keys for c in keys]
            label = f'{len(cases)} triples'
        bad = []
        for n, t in enumerate(cases, 1):
            if not check_triple(*t, verbose=False):
                bad.append(t)
            if args == ['--every'] and n % 169 == 0:
                print(f'  {n}/{len(cases)} done, {len(bad)} failures', flush=True)
        okb, okm = check_basis(False), check_maxwell(False)
        print(f'{label} checked (boosted product, gamma formulas on both sectors, inverse, invariants): {len(bad)} failures', bad or '')
        print(f'basis rules and lemma for both bases: {"IDENTICAL" if okb else "DIFFER"};  homogeneous Maxwell identically zero: {"YES" if okm else "NO"}')
        sys.exit(0 if not bad and okb and okm else 1)
    elif args == ['--basis']:
        sys.exit(0 if check_basis() else 1)
    elif args == ['--maxwell']:
        sys.exit(0 if check_maxwell() else 1)
    elif len(args) == 3 and all(a in VECTORS for a in args):
        sys.exit(0 if check_triple(*args) else 1)
    else:
        print(__doc__)
        print('Vectors:', ', '.join(VECTORS))
