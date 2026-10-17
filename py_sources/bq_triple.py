"""
bq_triple.py -- the triple product E = D (A^T B) of three BQ four-vectors.

With C = A^T B = c1 1^ + cK.K^ + cs.sigma^ the triple rule is
  E = -(d.cK) 1^ + (d0 c1 + d.cs) T^ + (d c1 + d0 cs + d x cK).K^ + (d x cs - d0 cK).sigma^ ,
with D to the left of C, so a four-gradient D acts on the channels of C.
The script evaluates the rule, forms the explicit 2x2 matrix product
D (A^T B), decomposes it, and checks symbolically that both agree.

Usage
  python bq_triple.py D D A --c1zero     Maxwell: d(d^T A) in the Lorenz gauge
  python bq_triple.py J D A --c1zero     Lorentz force: J (d^T A)
  python bq_triple.py D J A              d(J^T A)
  python bq_triple.py --examples         the products from the papers
  python bq_triple.py --all              every triple, 2197 in all (about 6 minutes)
--c1zero sets the norm channel c1 of C to zero (Lorenz gauge for d^T A).
"""
import sys
import itertools
import sympy as sp
from bq_core import (VECTORS, bilinear_rule, triple_rule, decompose, left_multiply,
                     eval_matrix, right_values, compare, show_channels, One)

EXAMPLES = [('D', 'D', 'A', True), ('J', 'D', 'A', True), ('D', 'J', 'A', False),
            ('D', 'U', 'G', False), ('U', 'D', 'G', False), ('D', 'D', 'G', False), ('d', 'a', 'b', False)]


def triple(kd, ka, kb, c1zero=False, verbose=True):
    D, A, B = VECTORS[kd], VECTORS[ka], VECTORS[kb]
    C_rule = bilinear_rule(A.comps, right_values(B))
    Cm = left_multiply(A.time_reversed(), eval_matrix(B.comps))
    if c1zero:
        C_rule = dict(C_rule, **{'1': sp.Integer(0)})
        Cm = sp.expand(Cm - C_rule_norm(A, B) * One)
    rule = triple_rule(D.comps, C_rule)
    mat = decompose(left_multiply(D.comps, Cm))
    ok = compare(mat, rule)
    if verbose:
        print(f'D = {D.name}: {D.what}\nA = {A.name}: {A.what}\nB = {B.name}: {B.what}')
        print(show_channels({k: C_rule[k] for k in ('1', 'K', 's')}, f'C = {A.name}^T {B.name}' + ('  (c1 set to 0)' if c1zero else '')))
        print(show_channels(rule, f'E = {D.name}({A.name}^T {B.name})  (triple rule)'))
        print('matrix product and triple rule:', 'IDENTICAL (exact)' if ok else 'DIFFER')
    return ok


def C_rule_norm(A, B):
    return bilinear_rule(A.comps, right_values(B))['1']


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '--c1zero']
    c1z = '--c1zero' in sys.argv
    if args == ['--examples']:
        res = []
        for d, a, b, z in EXAMPLES:
            print('=' * 72); res.append(triple(d, a, b, z))
        print('=' * 72, f'\n{sum(res)} of {len(res)} examples verified')
        sys.exit(0 if all(res) else 1)
    elif args == ['--all']:
        keys = list(VECTORS)
        fails = [k for k in itertools.product(keys, repeat=3) if not triple(*k, verbose=False)]
        print(f'{len(keys)**3} triples checked, {len(fails)} failures', fails or '')
        sys.exit(1 if fails else 0)
    elif len(args) == 3:
        sys.exit(0 if triple(*args, c1zero=c1z) else 1)
    else:
        print(__doc__)
