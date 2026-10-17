"""
bq_pauli_table.py -- the BQ-Pauli multiplication table, computed from the
explicit 2x2 matrices and checked against the table in the BQ-Pauli slides.

Usage
  python bq_pauli_table.py            full 8x8 table and the check
  python bq_pauli_table.py sI sJ      one product written out entry by entry
Element names: T I J K sI sJ sK 1
"""
import sys
import sympy as sp
from bq_core import BASIS, ORDER, PRETTY, SPACE_TIME, TABLE

# the table of slide 17 (row . column), shared with the table route in bq_core
PAPER = TABLE


def identify(M):
    """Return 'X' or '-X' with M = +/- basis element X."""
    for k in ORDER:
        if M == BASIS[k]:
            return k
        if M == -BASIS[k]:
            return '-' + k
    raise ValueError('not a signed basis element:\n' + str(M))


def worked(a, b):
    A, B = BASIS[a], BASIS[b]
    C = A * B
    print(f'{PRETTY[a]} . {PRETTY[b]}')
    print(f'  A = {A.tolist()}')
    print(f'  B = {B.tolist()}')
    for r in range(2):
        for s in range(2):
            t1, t2 = A[r, 0] * B[0, s], A[r, 1] * B[1, s]
            print(f'  C{r}{s} = A{r}0 B0{s} + A{r}1 B1{s} = ({A[r,0]})({B[0,s]}) + ({A[r,1]})({B[1,s]}) = {t1} + {t2} = {sp.simplify(t1 + t2)}')
    res = identify(C)
    rev = identify(B * A)
    rel = 'commute' if rev == res else ('anticommute' if rev.lstrip('-') == res.lstrip('-') else 'reversed product differs')
    sign, key = ('-', res[1:]) if res.startswith('-') else ('', res)
    print(f'  C = {C.tolist()} = {sign}{PRETTY[key]}   ({"space-time" if key in SPACE_TIME else "spin-norm"} basis)')
    print(f'  reversed: {PRETTY[b]} . {PRETTY[a]} = {rev}  -> the two {rel}')
    print(f'  paper   : {PAPER[a][ORDER.index(b)]}   ->', 'agrees' if PAPER[a][ORDER.index(b)] == res else 'DIFFERS')


def table():
    w = 6
    print('row . column'.ljust(w + 2) + ''.join(k.rjust(w) for k in ORDER))
    bad = 0
    for a in ORDER:
        row = []
        for j, b in enumerate(ORDER):
            res = identify(BASIS[a] * BASIS[b])
            bad += res != PAPER[a][j]
            row.append(res)
        print(a.ljust(w + 2) + ''.join(r.rjust(w) for r in row))
    print(f'\n64 products computed from the matrices; {64 - bad} agree with the paper, {bad} differ.')
    return bad


if __name__ == '__main__':
    if len(sys.argv) == 3:
        worked(sys.argv[1], sys.argv[2])
    else:
        sys.exit(1 if table() else 0)
