"""
bq_dirac_table.py -- the BQ-Dirac (BQ-Weyl, 4x4) multiplication table, as on the page
BQ_Dirac_Multiplication_Table.html.

Definitions (project conventions, Weyl representation), built from the BQ-Pauli blocks:
    beta_0 = [[0, T], [T, 0]],   beta_i = [[0, K_i], [-K_i, 0]]   (K_i = I, J, K)
    alpha_i = beta_0 beta_i,  Sigma_I = beta_J beta_K (cyclic),  chi = -beta_0 beta_I beta_J beta_K,
    beta~_mu = chi beta_mu,  and the 4x4 unit 1.
Only the four beta_mu are entered; the other twelve elements are computed.

Checked exactly with SymPy:
  1  the block (Pauli) content of all 16 elements, each block 0 or +-1 BQ-Pauli element
  2  beta_0^2 = -1, beta_i^2 = +1, beta_mu beta_nu + beta_nu beta_mu = 2 eta_mu_nu 1, eta = diag(-1,1,1,1)
  3  chi^2 = -1; chi anticommutes with beta_mu and beta~_mu, commutes with alpha_i and Sigma_i
  4  the 16 elements are orthogonal under tr(XY)
  5  all 256 products: block route (four blocks, each a sum of BQ-Pauli products) equals the 4x4 route,
     and each product is +-1 times exactly one basis element (no factor i)

Usage
  python bq_dirac_table.py              # all checks and the 16 x 16 table
  python bq_dirac_table.py bI bJ        # one product in detail: blocks, block terms, 4x4 matrices
Names: one b0 bI bJ bK aI aJ aK SI SJ SK bt0 btI btJ btK chi
"""
import sys
import sympy as sp

i = sp.I
P = {'1': sp.eye(2), 'T': i * sp.eye(2), 'I': sp.Matrix([[i, 0], [0, -i]]), 'J': sp.Matrix([[0, 1], [-1, 0]]),
     'K': sp.Matrix([[0, i], [i, 0]]), 'sI': sp.Matrix([[1, 0], [0, -1]]), 'sJ': sp.Matrix([[0, -i], [i, 0]]),
     'sK': sp.Matrix([[0, 1], [1, 0]])}
PN = {'1': '1^', 'T': 'T^', 'I': 'I^', 'J': 'J^', 'K': 'K^', 'sI': 'sigma_I', 'sJ': 'sigma_J', 'sK': 'sigma_K'}
Z2 = sp.zeros(2)


def pauli_of(M):
    """Block M as (sign, key) of one BQ-Pauli element, or None for the zero block."""
    if M == Z2:
        return None
    for k, X in P.items():
        for s in (1, -1):
            if M == s * X:
                return (s, k)
    raise ValueError(f'block is not +- one BQ-Pauli element: {M}')


def blk(a, b, c, d):
    return sp.Matrix(sp.BlockMatrix([[a, b], [c, d]]))


def blocks(M):
    return [[M[0:2, 0:2], M[0:2, 2:4]], [M[2:4, 0:2], M[2:4, 2:4]]]


b0 = blk(Z2, P['T'], P['T'], Z2)
bI, bJ, bK = (blk(Z2, P[k], -P[k], Z2) for k in 'IJK')
one = sp.eye(4)
chi = -b0 * bI * bJ * bK
EL = {'one': one, 'b0': b0, 'bI': bI, 'bJ': bJ, 'bK': bK,
      'aI': b0 * bI, 'aJ': b0 * bJ, 'aK': b0 * bK,
      'SI': bJ * bK, 'SJ': bK * bI, 'SK': bI * bJ,
      'bt0': chi * b0, 'btI': chi * bI, 'btJ': chi * bJ, 'btK': chi * bK, 'chi': chi}
NAMES = list(EL)


def ident(M):
    for n, X in EL.items():
        for s in (1, -1):
            if M == s * X:
                return (s, n)
    return None


def fmt_b(b):
    return '0' if b is None else ('-' if b[0] < 0 else '') + PN[b[1]]


def block_product(X, Y, log=None):
    """Block route: (XY)_ik = X_i0 Y_0k + X_i1 Y_1k, each term a BQ-Pauli product."""
    BX, BY = blocks(X), blocks(Y)
    out = [[None, None], [None, None]]
    for r in range(2):
        for c in range(2):
            S = BX[r][0] * BY[0][c] + BX[r][1] * BY[1][c]
            out[r][c] = S
            if log is not None:
                terms = []
                for m in range(2):
                    a, b = pauli_of(BX[r][m]), pauli_of(BY[m][c])
                    terms.append('0' if a is None or b is None else f'({fmt_b(a)})({fmt_b(b)}) = {fmt_b(pauli_of(BX[r][m] * BY[m][c]))}')
                log.append(f'  (XY)_{r}{c} = ' + '  +  '.join(terms) + f'   =>  {fmt_b(pauli_of(S))}')
    return blk(out[0][0], out[0][1], out[1][0], out[1][1])


def report(name, ok):
    print(f'  {name:<84} {"IDENTICAL (exact)" if ok else "DIFFER"}')
    return ok


def detail(a, b):
    X, Y = EL[a], EL[b]
    print(f'{a} . {b}')
    for n, M in ((a, X), (b, Y)):
        print(f'  {n} blocks: ' + str([[fmt_b(pauli_of(B)) for B in row] for row in blocks(M)]))
    log = []
    Bp = block_product(X, Y, log)
    print('Block route (BQ-Pauli products):')
    print('\n'.join(log))
    M = X * Y
    print('4x4 route:')
    sp.pprint(X); print('  times'); sp.pprint(Y); print('  equals'); sp.pprint(M)
    s, n = ident(M)
    print(f'Result: {"-" if s < 0 else "+"}{n};  block route equals 4x4 route: {Bp == M}')


def main():
    ok = True
    print('1  Block content of the 16 basis elements (BQ-Pauli entries)')
    for n, M in EL.items():
        print(f'  {n:<4} ' + str([[fmt_b(pauli_of(B)) for B in row] for row in blocks(M)]))
    print('2  Clifford relations of the beta_mu')
    B = [b0, bI, bJ, bK]
    eta = sp.diag(-1, 1, 1, 1)
    ok &= report('beta_mu beta_nu + beta_nu beta_mu = 2 eta_mu_nu 1, eta = diag(-1, 1, 1, 1)',
                 all(B[m] * B[n] + B[n] * B[m] == 2 * eta[m, n] * one for m in range(4) for n in range(4)))
    print('3  The pseudoscalar chi')
    ok &= report('chi^2 = -1', chi * chi == -one)
    ok &= report('chi anticommutes with beta_mu and beta~_mu', all(chi * EL[n] == -EL[n] * chi for n in NAMES[1:5] + NAMES[11:15]))
    ok &= report('chi commutes with alpha_i and Sigma_i', all(chi * EL[n] == EL[n] * chi for n in NAMES[5:11]))
    print('4  Orthogonality')
    ok &= report('tr(X Y) = 0 for distinct basis elements',
                 all(sp.simplify((EL[x] * EL[y]).trace()) == 0 for x in NAMES for y in NAMES if x != y))
    print('5  The 256 products')
    table, good = [], True
    for a in NAMES:
        row = []
        for b in NAMES:
            M = EL[a] * EL[b]
            r = ident(M)
            good &= (r is not None) and block_product(EL[a], EL[b]) == M
            row.append(('-' if r[0] < 0 else '+') + r[1])
        table.append(row)
    ok &= report('block route = 4x4 route, each product +- one basis element (no factor i)', good)
    print('\n       ' + ' '.join(n.ljust(5) for n in NAMES))
    for a, row in zip(NAMES, table):
        print(f'  {a:<4} ' + ' '.join(x.ljust(5) for x in row))
    print('\nAll checks:', 'IDENTICAL' if ok else 'SOME DIFFER')
    return ok


if __name__ == '__main__':
    if len(sys.argv) == 3:
        detail(sys.argv[1], sys.argv[2])
    else:
        raise SystemExit(0 if main() else 1)
