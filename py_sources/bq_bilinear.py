"""
bq_bilinear.py -- the bilinear product A^T B of two BQ four-vectors.

For a chosen pair the script
  1. evaluates the channel rule  A^T B = (a0 b0 - a.b) 1^ + (a x b).K^ + (a0 b - a b0).sigma^,
     keeping a to the left of b so a four-gradient in A acts on B;
  2. forms A^T and B as explicit 2x2 matrices, multiplies them and decomposes
     the result on 1^, T^, K^_i, sigma_i;
  3. checks symbolically that both give the same channels.

Usage
  python bq_bilinear.py R P        one pair
  python bq_bilinear.py --all      verify every pair of the list
  python bq_bilinear.py --list     show the available four-vectors
Vectors: R dR V U P D(=four-gradient) A J G F a b d
If B is the four-gradient, the product is evaluated on a test field psi.
"""
import sys
from bq_core import VECTORS, bilinear_rule, decompose, left_multiply, eval_matrix, right_values, compare, show_channels


def bilinear(ka, kb, verbose=True):
    A, B = VECTORS[ka], VECTORS[kb]
    rule = bilinear_rule(A.comps, right_values(B))
    mat = decompose(left_multiply(A.time_reversed(), eval_matrix(B.comps)))
    ok = compare(mat, rule)
    if verbose:
        print(f'A = {A.name}: {A.what}\nB = {B.name}: {B.what}' + ('\n(B is an operator: acting on psi)' if B.is_op else ''))
        print(show_channels(rule, f'{A.name}^T {B.name}  (channel rule)'))
        print('matrix product and channel rule:', 'IDENTICAL (exact)' if ok else 'DIFFER')
    return ok


if __name__ == '__main__':
    args = sys.argv[1:]
    if args == ['--list']:
        for k, v in VECTORS.items():
            print(f'{k:3s} {v.name:6s} {v.what}')
    elif args == ['--all']:
        keys = list(VECTORS)
        fails = [(a, b) for a in keys for b in keys if not bilinear(a, b, verbose=False)]
        print(f'{len(keys)**2} pairs checked, {len(fails)} failures', fails or '')
        sys.exit(1 if fails else 0)
    elif len(args) == 2:
        sys.exit(0 if bilinear(*args) else 1)
    else:
        print(__doc__)
