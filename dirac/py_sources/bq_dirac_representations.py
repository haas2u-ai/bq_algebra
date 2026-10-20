"""
bq_dirac_representations.py -- the BQ-Dirac basis in the Weyl and in the Dirac representation, and the matrix that
connects them, as on the page BQ_Dirac_Weyl_Dirac_Representations.html (S first, then the Weyl / Dirac switch).

Weyl representation (as in bq_dirac_table.py):   beta_0 = [[0, T^], [T^, 0]],    beta_i = [[0, K^_i], [-K^_i, 0]]
Dirac representation:                            beta_0 = [[T^, 0], [0, -T^]],   beta_i = [[0, K^_i], [-K^_i, 0]]
In both, the other twelve elements are products: alpha_i = beta_0 beta_i, Sigma_I = beta_J beta_K (cyclic),
chi = -beta_0 beta_I beta_J beta_K, beta~_mu = chi beta_mu.

Weyl to Dirac:  X_D = S X_W S^-1  with  S = (1/sqrt 2) [[1^, 1^], [-1^, 1^]],  S^-1 = S^T = (1/sqrt 2) [[1^, -1^], [1^, 1^]]

Checked exactly with SymPy:
  1  both sets of generators satisfy the Clifford relations beta_0^2 = -1, beta_i^2 = +1, distinct beta_mu anticommute
  2  S is real orthogonal (S S^T = 1) and S beta_mu^W S^-1 = beta_mu^D for the four generators
  3  hence S X_W S^-1 = X_D for all sixteen elements, each built independently in its own representation
  4  the 16 x 16 multiplication table is identical in both representations (X_D Y_D = S X_W Y_W S^-1)
  5  every Dirac element is again a block matrix with each block 0 or +- one BQ-Pauli element; the blocks are listed
  6  S is itself an element of the algebra: S = (1 - beta~_0^W)/sqrt 2 = exp(-(pi/4) beta~_0^W)
  7  what changes: in the Dirac representation beta_0, Sigma_i, beta~_i and 1 are block diagonal; the vector is
     a/_D = [[a0 T^, a.K^], [-a.K^, -a0 T^]]; the boost becomes Lambda_D = S Lambda_W S^-1 = [[c, s sigma_I], [s sigma_I, c]]
     (c = cosh(psi/2), s = sinh(psi/2)), the rotations stay diag(r, r)

Usage
  python bq_dirac_representations.py
"""
import sympy as sp
from bq_dirac_table import P, blk, pauli_of

Z2 = sp.zeros(2)
NAMES = ['one', 'b0', 'bI', 'bJ', 'bK', 'aI', 'aJ', 'aK', 'SI', 'SJ', 'SK', 'bt0', 'btI', 'btJ', 'btK', 'chi']


def report(name, ok):
    print(f'  {name:<88} {"IDENTICAL (exact)" if ok else "DIFFER"}')
    return ok


def zero(M):
    return all(sp.simplify(x) == 0 for x in M)


def build(b0):
    bI, bJ, bK = (blk(Z2, P[k], -P[k], Z2) for k in 'IJK')
    one = sp.eye(4)
    chi = -b0 * bI * bJ * bK
    return {'one': one, 'b0': b0, 'bI': bI, 'bJ': bJ, 'bK': bK, 'aI': b0 * bI, 'aJ': b0 * bJ, 'aK': b0 * bK,
            'SI': bJ * bK, 'SJ': bK * bI, 'SK': bI * bJ, 'bt0': chi * b0, 'btI': chi * bI, 'btJ': chi * bJ, 'btK': chi * bK, 'chi': chi}


def ident(M, EL):
    for n, X in EL.items():
        for s in (1, -1):
            if M == s * X:
                return (s, n)
    return None


def blocks_txt(M):
    out = []
    for r in (0, 2):
        for c in (0, 2):
            p = pauli_of(M[r:r + 2, c:c + 2])
            out.append('0' if p is None else ('-' if p[0] < 0 else '') + p[1])
    return f'[[{out[0]}, {out[1]}], [{out[2]}, {out[3]}]]'


def main():
    ok = True
    W = build(blk(Z2, P['T'], P['T'], Z2))
    D = build(blk(P['T'], Z2, Z2, -P['T']))

    print('1  Clifford relations in both representations')
    for nm, R in (('Weyl', W), ('Dirac', D)):
        B = [R['b0'], R['bI'], R['bJ'], R['bK']]
        eta = [-1, 1, 1, 1]
        ok &= report(f'{nm}: beta_mu beta_nu + beta_nu beta_mu = 2 eta_mu_nu 1, eta = diag(-1, 1, 1, 1)',
                     all(zero(B[m] * B[n] + B[n] * B[m] - (2 * eta[m] if m == n else 0) * sp.eye(4)) for m in range(4) for n in range(4)))

    print('2  The transformation matrix')
    S = blk(P['1'], P['1'], -P['1'], P['1']) / sp.sqrt(2)
    Si = S.T
    ok &= report('S = (1/sqrt 2)[[1, 1], [-1, 1]] is real orthogonal: S S^T = 1', zero(S * Si - sp.eye(4)))
    ok &= report('S beta_mu^W S^-1 = beta_mu^D for mu = 0, I, J, K', all(zero(S * W[k] * Si - D[k]) for k in ['b0', 'bI', 'bJ', 'bK']))

    print('3  All sixteen elements')
    ok &= report('S X_W S^-1 = X_D for all sixteen, each built in its own representation', all(zero(S * W[k] * Si - D[k]) for k in NAMES))

    print('4  The table')
    same = True
    for a in NAMES:
        for b in NAMES:
            same &= ident(W[a] * W[b], W) == ident(D[a] * D[b], D)
    ok &= report('the 256 products give the same signed basis element in both representations', same)

    print('5  Pauli block content of the Dirac basis')
    try:
        for k in NAMES:
            print(f'     {k:4s} Weyl {blocks_txt(W[k]):34s}  Dirac {blocks_txt(D[k])}')
        ok &= report('every Dirac element is a block matrix of 0 and +- one BQ-Pauli element', True)
    except ValueError:
        ok &= report('every Dirac element is a block matrix of 0 and +- one BQ-Pauli element', False)

    print('6  S inside the algebra')
    t = sp.Symbol('t', real=True)
    ok &= report('S = (1 - beta~_0^W)/sqrt 2 = cos(pi/4) 1 - sin(pi/4) beta~_0^W = exp(-(pi/4) beta~_0^W)',
                 zero(S - (sp.eye(4) - W['bt0']) / sp.sqrt(2)) and zero(W['bt0'] * W['bt0'] + sp.eye(4))
                 and zero((-(sp.pi / 4) * W['bt0']).exp() - S))

    print('7  What changes')
    a = sp.symbols('a0:4', real=True)
    AD = a[0] * D['b0'] + a[1] * D['bI'] + a[2] * D['bJ'] + a[3] * D['bK']
    aK = a[1] * P['I'] + a[2] * P['J'] + a[3] * P['K']
    ok &= report('a/_D = [[a0 T^, a.K^], [-a.K^, -a0 T^]]: time on the diagonal, space off the diagonal',
                 zero(AD - blk(a[0] * P['T'], aK, -aK, -a[0] * P['T'])))
    z = sp.Symbol('z', positive=True)
    c, s = (z + 1 / z) / 2, (z - 1 / z) / 2
    LW = blk(sp.diag(1 / z, z), Z2, Z2, sp.diag(z, 1 / z))
    LD = S * LW * Si
    ok &= report('Lambda_D = S Lambda_W S^-1 = cosh(psi/2) 1 - sinh(psi/2) alpha_I^D = [[c 1, s sigma_I], [s sigma_I, c 1]]',
                 zero(LD - (c * sp.eye(4) - s * D['aI'])) and zero(LD - blk(c * P['1'], s * P['sI'], s * P['sI'], c * P['1'])))
    th = sp.Symbol('theta', real=True)
    r = sp.cos(th / 2) * P['1'] + sp.sin(th / 2) * P['K']
    RW = blk(r, Z2, Z2, r)
    ok &= report('rotations diag(r, r) are the same in both representations', zero(S * RW * Si - RW))

    print('\nAll checks:', 'IDENTICAL' if ok else 'SOME DIFFER')
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
