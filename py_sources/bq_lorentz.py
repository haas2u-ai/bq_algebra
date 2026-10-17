"""
bq_lorentz.py -- Lorentz transformations of BQ four-vectors along I^, as on the page
BQ_Lorentz_Transformations.html.

  rotor        U = exp(psi sigma_I/2) = cosh(psi/2) 1^ + sinh(psi/2) sigma_I = diag(e^{psi/2}, e^{-psi/2})
  boost        A^L      = U^-1 A U^-1
  inverse      A^{L^-1} = U A U
  time adjoint (A^T)^L = U^-1 A^T U^-1 = (A^{L^-1})^T ,   (A^L)^T = U A^T U
with gamma = cosh(psi), beta gamma = sinh(psi).

Every statement is checked exactly (symbolic rapidity psi) for the chosen vector:
  1  the rotor: exponential series = cosh/sinh form = diagonal matrix
  2  A^L has the coordinates a0' = gamma(a0 - beta a1), a1' = gamma(a1 - beta a0), a2, a3 unchanged,
     and no 1^ or sigma^ channel
  3  U A^L U = A  (the inverse undoes the boost)
  4  (A^T)^L = (A^{L^-1})^T  and  (A^L)^T = U A^T U
  5  invariance (A^L)^T A^L = A^T A, and covariance (A^L)^T B^L = U (A^T B) U^-1 for every B
  6  light cone: the diagonal matrix entries i(a0 +- a1) are multiplied by e^{-+psi}
The derivation step 5 of the page (U^-1 = C 1^ - S sigma_I, C = cosh(psi/2), S = sinh(psi/2))
and the Lemma K^_mu Lambda^mu_nu = U^-1 K^_nu U^-1 are checked with --basis, by the
BQ-Pauli multiplication table (route 1) and by explicit matrices (route 2).

Usage
  python bq_lorentz.py R           all checks for R (vectors as in bq_bilinear.py --list)
  python bq_lorentz.py --basis     step 5 and the Lemma, term by term with the table
  python bq_lorentz.py --all       every vector and the basis checks
An operator vector (d) is evaluated on the test field psi(t,x,y,z): its coordinates are d_mu psi.
"""
import sys
import sympy as sp
from bq_core import (BASIS, ORDER, VECTORS, One, sI, decompose, eval_matrix, table_product,
                     as_channels, show, show_channels, PRETTY)

w = sp.Symbol('ψ', real=True)                      # rapidity
Cc, Ss = sp.cosh(w / 2), sp.sinh(w / 2)
U = sp.diag(sp.exp(w / 2), sp.exp(-w / 2))
Ui = sp.diag(sp.exp(-w / 2), sp.exp(w / 2))
MQ = [BASIS[k] for k in ('T', 'I', 'J', 'K')]


def z(e):
    e = sp.expand(sp.sympify(e).rewrite(sp.exp))
    return e == 0 or sp.simplify(e) == 0


def mzero(M):
    return all(z(e) for e in M)


def minquat(a):
    return sum((a[m] * MQ[m] for m in range(4)), sp.zeros(2, 2))


def coords(M):
    """Minquat coordinates (a0, a1, a2, a3) of a matrix, plus its spin-norm remainder."""
    ch = decompose(M)
    return [ch['T']] + ch['K'], [ch['1']] + ch['s']


def adjoint(M):
    """BQ time adjoint of a minquat: a0 -> -a0, the spatial part kept."""
    a, rest = coords(M)
    assert all(z(r) for r in rest), 'time adjoint taken of a non-minquat'
    return minquat([-a[0]] + a[1:])


def report(name, ok):
    print(f'  {name:<58} {"IDENTICAL (exact)" if ok else "DIFFER"}')
    return ok


def check_rotor(verbose=True):
    series = (sp.Rational(1, 2) * w * sI).exp()
    ok = mzero(series - U) and mzero(Cc * One + Ss * sI - U) and mzero(U * Ui - One)
    if verbose:
        print('The rotor U = exp(psi sigma_I/2)')
        report('exp(psi sigma_I/2) = cosh(psi/2) 1^ + sinh(psi/2) sigma_I = diag', ok)
    return ok


def check_vector(key, verbose=True):
    v = VECTORS[key]
    A = eval_matrix(v.comps)
    a, _ = coords(A)
    g, bg = sp.cosh(w), sp.sinh(w)
    AL = Ui * A * Ui
    aL, restL = coords(AL)
    expect = [g * a[0] - bg * a[1], g * a[1] - bg * a[0], a[2], a[3]]
    ok = {}
    ok['coords'] = all(z(aL[m] - expect[m]) for m in range(4)) and all(z(r) for r in restL)
    ok['inverse'] = mzero(U * AL * U - A)
    AT = adjoint(A)
    ok['adjL'] = mzero(Ui * AT * Ui - adjoint(U * A * U))
    ok['Ladj'] = mzero(adjoint(AL) - U * AT * U)
    ok['invariant'] = mzero(adjoint(AL) * AL - AT * A)
    cov = []
    for kb, vb in VECTORS.items():
        B = eval_matrix(vb.comps)
        cov.append(mzero(adjoint(AL) * (Ui * B * Ui) - U * (AT * B) * Ui))
    ok['covariant'] = all(cov)
    ok['lightcone'] = z(AL[0, 0] - sp.exp(-w) * A[0, 0]) and z(AL[1, 1] - sp.exp(w) * A[1, 1]) \
        and z(AL[0, 1] - A[0, 1]) and z(AL[1, 0] - A[1, 0])
    if verbose:
        print(f'{v.name} = {v.what}')
        print('  coordinates  : (' + ', '.join(show(q) for q in a) + ')')
        print("  boosted A^L  : a0' = cosh(psi) a0 - sinh(psi) a1 = " + show(expect[0]))
        print("                 a1' = cosh(psi) a1 - sinh(psi) a0 = " + show(expect[1]))
        print('                 a2, a3 unchanged;  cosh(psi) = gamma, sinh(psi) = beta gamma')
        report("A^L = U^-1 A U^-1 : a0', a1' by gamma, beta; a2, a3 unchanged", ok['coords'])
        report('U A^L U = A  (inverse boost)', ok['inverse'])
        report('(A^T)^L = U^-1 A^T U^-1 = (A^{L^-1})^T', ok['adjL'])
        report('(A^L)^T = U A^T U', ok['Ladj'])
        report('(A^L)^T A^L = A^T A  (invariant quadratic)', ok['invariant'])
        report(f'(A^L)^T B^L = U (A^T B) U^-1  for all {len(cov)} vectors B', ok['covariant'])
        report('diagonal entries times e^-psi, e^+psi; off-diagonal unchanged', ok['lightcone'])
    return all(ok.values())


def check_basis(verbose=True):
    """Step 5 of the page with the table, and the Lemma, for X = T^, I^, J^, K^."""
    C, S = sp.symbols('C S', real=True)
    Uinv = {'1': C, 'sI': -S}                       # U^-1 = C 1^ - S sigma_I, basis explicit
    g, bg = sp.cosh(w), sp.sinh(w)
    lam = {'T': {'T': g, 'I': -bg}, 'I': {'I': g, 'T': -bg}, 'J': {'J': 1}, 'K': {'K': 1}}
    all_ok = True
    if verbose:
        print('Step 5: (C 1^ - S sigma_I) X (C 1^ - S sigma_I), C = cosh(psi/2), S = sinh(psi/2)')
    for X in ('T', 'I', 'J', 'K'):
        left = table_product(Uinv, {X: sp.Integer(1)})
        out = table_product(left, Uinv)
        res = {k: sp.expand(out[k]) for k in ORDER if sp.expand(out[k]) != 0}
        sub = {k: sp.simplify(e.subs({C: Cc, S: Ss}).rewrite(sp.exp)) for k, e in res.items()}
        ok_t = all(z(sub.get(k, 0) - lam[X].get(k, 0)) for k in ORDER)
        mat = Ui * BASIS[X] * Ui
        rhs = sum((cf * BASIS[k] for k, cf in lam[X].items()), sp.zeros(2, 2))
        ok_m = mzero(mat - rhs)
        all_ok &= ok_t and ok_m
        if verbose:
            terms = ' + '.join(f'({e}) {PRETTY[k]}' for k, e in res.items())
            print(f'  U^-1 {PRETTY[X]} U^-1 = {terms}')
            target = ' + '.join(f'{e} {PRETTY[k]}' for k, e in lam[X].items()).replace('+ -', '- ')
            print(f'               = {target}')
            report('  table route (C^2 + S^2 = cosh psi, 2CS = sinh psi)', ok_t)
            report('  matrix route', ok_m)
    if verbose:
        print('Lemma: K^_mu Lambda^mu_nu = U^-1 K^_nu U^-1 holds for nu = T, I, J, K'
              if all_ok else 'Lemma: DIFFERS')
    return all_ok


if __name__ == '__main__':
    args = sys.argv[1:]
    if args == ['--all']:
        bad = [k for k in VECTORS if not check_vector(k, verbose=False)]
        okr, okb = check_rotor(False), check_basis(False)
        print(f'{len(VECTORS)} vectors checked (boost, inverse, adjoints, invariance, covariance with all '
              f'{len(VECTORS)} vectors, light cone), {len(bad)} failures', bad or '')
        print(f'rotor: {"IDENTICAL" if okr else "DIFFER"};  step 5 and Lemma: {"IDENTICAL" if okb else "DIFFER"}')
        sys.exit(0 if not bad and okr and okb else 1)
    elif args == ['--basis']:
        check_rotor()
        sys.exit(0 if check_basis() else 1)
    elif len(args) == 1 and args[0] in VECTORS:
        ok = check_rotor() & check_vector(args[0])
        sys.exit(0 if ok else 1)
    else:
        print(__doc__)
        print('Vectors:', ', '.join(VECTORS))
