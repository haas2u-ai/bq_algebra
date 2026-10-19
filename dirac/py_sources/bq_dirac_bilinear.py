"""
bq_dirac_bilinear.py -- the bilinear product of two Dirac-level four-vectors, as on the page
BQ_Dirac_Bilinear_Product_Explorer.html.

A Dirac-level four-vector has REAL coordinates on the Weyl vector basis (no time adjoint is used):
    A/ = A_mu beta^mu = a0 beta_0 + a1 beta_I + a2 beta_J + a3 beta_K
and the product of two of them lands in the scalar and bivector channels only:
    A/B/ = -(a0 b0 - a.b) 1 + (a0 b - a b0).alpha + (a x b).Sigma

Checked exactly with SymPy (4x4 matrices built from the BQ-Pauli blocks, as in bq_dirac_table.py):
  1  the channel rule for general coordinates; no beta, beta~ or chi channel
  2  block form: A/ = [[0, A], [-A^T, 0]] with A the Pauli minquat and A^T its BQ time adjoint,
     and A/B/ = [[-A B^T, 0], [0, -A^T B]]: the lower block is minus the Pauli bilinear A^T B,
     so the channels correspond as 1^ -> -1, K^ -> Sigma, sigma^ -> alpha
  3  all 144 ordered pairs of the named four-vectors (R, dR, V, U, P, d, A, J, G, F, a, b), with the operator
     order kept (a four-gradient acts on everything to its right; when it stands last the product acts on a
     test field psi): matrix route = channel rule
  4  results quoted in the papers:
       d/A/ = ((1/c^2) d_t phi + div A) 1 + (E/c).alpha + B.Sigma          (Weyl-level EM, eqs. 78-82)
       d/G/ = sigma_E 1 - (1/c) Pi.alpha + Omega.Sigma                     (eqs. 264-267)
       U/U/ = -c^2 1,  d/R/ = 4 1,  d/d/ = (laplacian - (1/c^2) d_t^2) 1

Usage
  python bq_dirac_bilinear.py
"""
import sympy as sp
from bq_dirac_table import EL, P, blk, NAMES

t, x, y, z = X = sp.symbols('t x y z', real=True)
c = sp.Symbol('c', positive=True)
psi = sp.Function('psi')(t, x, y, z)
Z2 = sp.zeros(2)
BETA = [EL['b0'], EL['bI'], EL['bJ'], EL['bK']]


def report(name, ok):
    print(f'  {name:<84} {"IDENTICAL (exact)" if ok else "DIFFER"}')
    return ok


def zero(e):
    if isinstance(e, sp.MatrixBase):
        return all(zero(x) for x in e)
    return sp.simplify(sp.expand(e)) == 0


def channels(M):
    return {n: sp.simplify((EL[n] * M).trace() / (EL[n] * EL[n]).trace()) for n in NAMES}


def vec4(comps):
    return sum((comps[m] * BETA[m] for m in range(4)), sp.zeros(4))


# ---------- named four-vectors: plain components, or operators (callables) ----------
f4 = lambda n: sp.Function(n)(t, x, y, z)
gam = sp.Symbol('gamma', positive=True)
sy = lambda s: sp.symbols(s, real=True)
OPS = [lambda F: -sp.diff(F, t) / c, lambda F: sp.diff(F, x), lambda F: sp.diff(F, y), lambda F: sp.diff(F, z)]
VECS = {
    'R': [c * t, x, y, z],
    'dR': [c * sy('dt'), *sy('dx dy dz')],
    'V': [c, *sy('v1 v2 v3')],
    'U': [gam * c, *[gam * v for v in sy('v1 v2 v3')]],
    'P': [sy('E') / c, *sy('p1 p2 p3')],
    'd': 'op',
    'A': [f4('phi') / c, f4('A1'), f4('A2'), f4('A3')],
    'J': [c * f4('rho'), f4('J1'), f4('J2'), f4('J3')],
    'G': [f4('epsilon') / c, f4('g1'), f4('g2'), f4('g3')],
    'F': [f4('Pw') / c, f4('F1'), f4('F2'), f4('F3')],
    'a': list(sy('a0 a1 a2 a3')),
    'b': list(sy('b0 b1 b2 b3')),
}


def apply(left, m, right_comp):
    """left component m acting on (or multiplying) one right component"""
    if left == 'op':
        return OPS[m](right_comp)
    return left[m] * right_comp


def right_comps(right):
    return [OPS[n](psi) for n in range(4)] if right == 'op' else right


def pair(Ln, Rn):
    L, R = VECS[Ln], right_comps(VECS[Rn])
    ap = lambda m, n: apply(L, m, R[n])
    # matrix route: sum over the 16 products beta_m beta_n
    M = sum((ap(m, n) * BETA[m] * BETA[n] for m in range(4) for n in range(4)), sp.zeros(4))
    # channel rule
    rule = {'one': -(ap(0, 0) - ap(1, 1) - ap(2, 2) - ap(3, 3)),
            'aI': ap(0, 1) - ap(1, 0), 'aJ': ap(0, 2) - ap(2, 0), 'aK': ap(0, 3) - ap(3, 0),
            'SI': ap(2, 3) - ap(3, 2), 'SJ': ap(3, 1) - ap(1, 3), 'SK': ap(1, 2) - ap(2, 1)}
    return M, rule


def main():
    ok = True
    a, b = sy('a0 a1 a2 a3'), sy('b0 b1 b2 b3')
    print('1  The channel rule, general coordinates')
    M = sp.expand(vec4(a) * vec4(b))
    ch = channels(M)
    rule = {'one': -(a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]),
            'aI': a[0] * b[1] - a[1] * b[0], 'aJ': a[0] * b[2] - a[2] * b[0], 'aK': a[0] * b[3] - a[3] * b[0],
            'SI': a[2] * b[3] - a[3] * b[2], 'SJ': a[3] * b[1] - a[1] * b[3], 'SK': a[1] * b[2] - a[2] * b[1]}
    ok &= report('A/B/ = -(a0b0 - a.b) 1 + (a0 b - a b0).alpha + (a x b).Sigma', all(zero(ch[n] - rule[n]) for n in rule))
    ok &= report('no beta, beta~ or chi channel', all(zero(ch[n]) for n in NAMES if n not in rule))
    print('2  Block form and the Pauli bilinear inside')
    pv = lambda v, s=1: s * v[0] * P['T'] + v[1] * P['I'] + v[2] * P['J'] + v[3] * P['K']
    Ap, ApT, Bp, BpT = pv(a), pv(a, -1), pv(b), pv(b, -1)
    ok &= report('A/ = [[0, A], [-A^T, 0]], A the Pauli minquat, A^T = -a0 T^ + a.K^', sp.expand(vec4(a) - blk(Z2, Ap, -ApT, Z2)) == sp.zeros(4))
    ok &= report('A/B/ = [[-A B^T, 0], [0, -A^T B]]: lower block = minus the Pauli bilinear', sp.expand(M - blk(-Ap * BpT, Z2, Z2, -ApT * Bp)) == sp.zeros(4))
    print('3  All 144 ordered pairs of named four-vectors, operator order kept')
    good = True
    for Ln in VECS:
        for Rn in VECS:
            M, rule = pair(Ln, Rn)
            ch = channels(sp.expand(M))
            good &= all(zero(ch[n] - rule[n]) for n in rule) and all(zero(ch[n]) for n in NAMES if n not in rule)
    ok &= report('matrix route (16 products beta_m beta_n) = channel rule, every pair', good)
    print('4  Results quoted in the papers')
    phi, A1, A2, A3 = f4('phi'), f4('A1'), f4('A2'), f4('A3')
    Av = sp.Matrix([A1, A2, A3]); grad = lambda f: sp.Matrix([sp.diff(f, v) for v in (x, y, z)])
    curl = lambda F: sp.Matrix([sp.diff(F[2], y) - sp.diff(F[1], z), sp.diff(F[0], z) - sp.diff(F[2], x), sp.diff(F[1], x) - sp.diff(F[0], y)])
    div = lambda F: sp.diff(F[0], x) + sp.diff(F[1], y) + sp.diff(F[2], z)
    E = -grad(phi) - sp.diff(Av, t); Bf = curl(Av)
    M, _ = pair('d', 'A'); ch = channels(sp.expand(M))
    ok &= report('d/A/ = ((1/c^2) d_t phi + div A) 1 + (E/c).alpha + B.Sigma',
                 zero(ch['one'] - (sp.diff(phi, t) / c**2 + div(Av)))
                 and all(zero(ch[n] - E[i] / c) for i, n in enumerate(['aI', 'aJ', 'aK']))
                 and all(zero(ch[n] - Bf[i]) for i, n in enumerate(['SI', 'SJ', 'SK'])))
    eps, g = f4('epsilon'), sp.Matrix([f4('g1'), f4('g2'), f4('g3')])
    Pi = sp.diff(g, t) + grad(eps); Om = curl(g); sE = sp.diff(eps, t) / c**2 + div(g)
    M, _ = pair('d', 'G'); ch = channels(sp.expand(M))
    ok &= report('d/G/ = sigma_E 1 - (1/c) Pi.alpha + Omega.Sigma, Pi = d_t g + grad eps',
                 zero(ch['one'] - sE) and all(zero(ch[n] + Pi[i] / c) for i, n in enumerate(['aI', 'aJ', 'aK']))
                 and all(zero(ch[n] - Om[i]) for i, n in enumerate(['SI', 'SJ', 'SK'])))
    v = sy('v1 v2 v3')
    M, _ = pair('U', 'U')
    ok &= report('U/U/ = -c^2 1 with gamma^2 (1 - v^2/c^2) = 1',
                 zero((M + c**2 * sp.eye(4)).subs(gam, 1 / sp.sqrt(1 - (v[0]**2 + v[1]**2 + v[2]**2) / c**2))))
    M, _ = pair('d', 'R')
    ok &= report('d/R/ = 4 1', zero(sp.Matrix(M) - 4 * sp.eye(4)))
    M, _ = pair('d', 'd')
    lap = sum(sp.diff(psi, w, 2) for w in (x, y, z)) - sp.diff(psi, t, 2) / c**2
    ok &= report("d/d/ psi = (laplacian - (1/c^2) d_t^2) psi 1, the d'Alembert operator", zero(sp.Matrix(M) - lap * sp.eye(4)))
    print('\nAll checks:', 'IDENTICAL' if ok else 'SOME DIFFER')
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
