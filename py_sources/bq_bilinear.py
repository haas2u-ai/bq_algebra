"""
bq_bilinear.py -- the bilinear product A^T B of two BQ four-vectors.

For a chosen pair the script checks the channel rule
    A^T B = (a0 b0 - a.b) 1^ + (a x b).K^ + (a0 b - a b0).sigma^
along two independent routes, keeping a to the left of b so that a
four-gradient in A acts on B:
  table route : expand (-a0 T^ + a.K^)(b0 T^ + b.K^) term by term and read
                each basis product from the BQ-Pauli multiplication table
                (16 products, no matrices);
  matrix route: form A^T and B as explicit 2x2 matrices, multiply them and
                decompose the result on 1^, T^, K^_i, sigma_i.
Both are compared exactly (symbolically) with the rule.

Usage
  python bq_bilinear.py R P            one pair, both routes
  python bq_bilinear.py R P --table    the same, with the 16 table products printed
  python bq_bilinear.py --all          verify every pair along both routes
  python bq_bilinear.py --list         show the available four-vectors
Vectors: R dR V U P D(=four-gradient) A J G F a b d
If B is the four-gradient, the product is evaluated on a test field psi.
"""
import sys
from bq_core import (VECTORS, bilinear_rule, bilinear_table, decompose, left_multiply, eval_matrix,
                     right_values, compare, show_channels, show_log)


def bilinear(ka, kb, verbose=True, table_log=False):
    A, B = VECTORS[ka], VECTORS[kb]
    rule = bilinear_rule(A.comps, right_values(B))
    log = [] if table_log else None
    tab = bilinear_table(A, B, log)
    mat = decompose(left_multiply(A.time_reversed(), eval_matrix(B.comps)))
    ok_t, ok_m = compare(tab, rule), compare(mat, rule)
    if verbose:
        print(f'A = {A.name}: {A.what}\nB = {B.name}: {B.what}' + ('\n(B is an operator: acting on psi)' if B.is_op else ''))
        if table_log:
            print(f'{A.name}^T {B.name} term by term (coordinate . basis element), basis products from the table:')
            print(show_log(log))
        print(show_channels(rule, f'{A.name}^T {B.name}  (channel rule)'))
        print('table route  and channel rule:', 'IDENTICAL (exact)' if ok_t else 'DIFFER')
        print('matrix route and channel rule:', 'IDENTICAL (exact)' if ok_m else 'DIFFER')
    return ok_t and ok_m


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '--table']
    show_table = '--table' in sys.argv
    if args == ['--list']:
        for k, v in VECTORS.items():
            print(f'{k:3s} {v.name:6s} {v.what}')
    elif args == ['--all']:
        keys = list(VECTORS)
        fails = [(a, b) for a in keys for b in keys if not bilinear(a, b, verbose=False)]
        print(f'{len(keys)**2} pairs checked along the table route and the matrix route, {len(fails)} failures', fails or '')
        sys.exit(1 if fails else 0)
    elif len(args) == 2:
        sys.exit(0 if bilinear(*args, table_log=show_table) else 1)
    else:
        print(__doc__)
