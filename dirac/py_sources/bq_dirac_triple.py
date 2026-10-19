"""
bq_dirac_triple.py -- the triple product of three Dirac-level four-vectors, as on the page
BQ_Dirac_Triple_Product_Explorer.html.

With C/ = A/B/ = s 1 + c_alpha.alpha + c_Sigma.Sigma (an even element), the triple product is odd:
    E/ = D/C/ = (d0 s - d.c_alpha) beta_0 + (d s - d0 c_alpha - d x c_Sigma).beta
              - (d.c_Sigma) beta~_0 + (d x c_alpha - d0 c_Sigma).beta~

Checked exactly with SymPy (4x4 matrices built from the BQ-Pauli blocks, as in bq_dirac_table.py):
  1  the triple rule for a general vector D/ times a general even element C/; no 1, alpha, Sigma or chi channel
  2  the same rule for D/(A/B/) with general coordinates
  3  the four-gradient acting on a general even FIELD C/(t, x, y, z): the rule with d0 = -(1/c) d_t, d = grad
     (this covers d/(J/A/), d/(U/G/), d/H/ and every other operator triple)
  4  block form: D/(A/B/) = [[0, -D(A^T B)], [D^T A B^T, 0]]; the upper-right block is minus the BQ-Pauli
     triple product E = D(A^T B), so [E/]_beta0 = -e_T, [E/]_beta = -e_K, [E/]_beta~0 = e_1, [E/]_beta~ = e_sigma
  5  Maxwell: d/B/ with B/ = (E/c).alpha + B.Sigma (Lorenz gauge) gives
       beta_0: -(1/c) div E,  beta: -(curl B - (1/c^2) d_t E),  beta~_0: -div B,  beta~: (1/c)(curl E + d_t B)
     (Weyl-level EM, eqs. 92-93; with the Maxwell equations d/B/ = -mu0 J/, eq. 98)
  6  Lorentz force: J/B/ = -F/_L + pseudo, F/_L = (J.E/c) beta_0 + (rho E + J x B).beta,
       pseudo = -(J.B) beta~_0 + ((1/c) J x E - c rho B).beta~                    (eqs. 102-113)

Usage
  python bq_dirac_triple.py
"""
import sympy as sp
from bq_dirac_table import EL, P, blk, NAMES

t, x, y, z = sp.symbols('t x y z', real=True)
c = sp.Symbol('c', positive=True)
BETA = [EL['b0'], EL['bI'], EL['bJ'], EL['bK']]
ODD = ['b0', 'bI', 'bJ', 'bK', 'bt0', 'btI', 'btJ', 'btK']
sy = lambda s: sp.symbols(s, real=True)
f4 = lambda n: sp.Function(n)(t, x, y, z)


def report(name, ok):
    print(f'  {name:<84} {"IDENTICAL (exact)" if ok else "DIFFER"}')
    return ok


def zero(e):
    if isinstance(e, sp.MatrixBase):
        return all(zero(v) for v in e)
    return sp.simplify(sp.expand(e)) == 0


def channels(M):
    return {n: sp.expand((EL[n] * M).trace() / (EL[n] * EL[n]).trace()) for n in NAMES}


def vec4(v):
    return sum((v[m] * BETA[m] for m in range(4)), sp.zeros(4))


def even(s, ca, cs):
    return s * EL['one'] + sum((ca[i] * EL[n] for i, n in enumerate(['aI', 'aJ', 'aK'])), sp.zeros(4)) \
        + sum((cs[i] * EL[n] for i, n in enumerate(['SI', 'SJ', 'SK'])), sp.zeros(4))


def rule(d0, dv, s, ca, cs, ap=lambda a, b: a * b):
    """triple rule; ap(a, b) is 'a acting on b' (plain product, or an operator applied to a field)"""
    dot = lambda u, v: sum(ap(u[i], v[i]) for i in range(3))
    cr = lambda u, v: [ap(u[1], v[2]) - ap(u[2], v[1]), ap(u[2], v[0]) - ap(u[0], v[2]), ap(u[0], v[1]) - ap(u[1], v[0])]
    r = {'b0': ap(d0, s) - dot(dv, ca), 'bt0': -dot(dv, cs)}
    for i, n in enumerate(['bI', 'bJ', 'bK']):
        r[n] = ap(dv[i], s) - ap(d0, ca[i]) - cr(dv, cs)[i]
    for i, n in enumerate(['btI', 'btJ', 'btK']):
        r[n] = cr(dv, ca)[i] - ap(d0, cs[i])
    return r


def check(M, r):
    ch = channels(M)
    return all(zero(ch[n] - r[n]) for n in r) and all(zero(ch[n]) for n in NAMES if n not in r)


def main():
    ok = True
    d, a, b = sy('d0:4'), sy('a0:4'), sy('b0:4')
    s, ca, cs = sy('s'), sy('cA1:4'), sy('cS1:4')
    print('1  General vector times general even element')
    ok &= report('D/C/ = (d0 s - d.c_a) b0 + (d s - d0 c_a - d x c_S).b - (d.c_S) b~0 + (d x c_a - d0 c_S).b~',
                 check(sp.expand(vec4(d) * even(s, ca, cs)), rule(d[0], d[1:], s, ca, cs)))
    print('2  D/(A/B/) with general coordinates')
    sAB = -(a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3])
    cAB = [a[0] * b[i] - a[i] * b[0] for i in (1, 2, 3)]
    cSB = [a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1]]
    ok &= report('triple rule with s = -(a0b0 - a.b), c_alpha = a0 b - a b0, c_Sigma = a x b; only odd channels',
                 check(sp.expand(vec4(d) * vec4(a) * vec4(b)), rule(d[0], d[1:], sAB, cAB, cSB)))
    print('3  The four-gradient acting on a general even field')
    S, CA, CS = f4('s'), [f4('cA1'), f4('cA2'), f4('cA3')], [f4('cS1'), f4('cS2'), f4('cS3')]
    OPS = [lambda F: -sp.diff(F, t) / c, lambda F: sp.diff(F, x), lambda F: sp.diff(F, y), lambda F: sp.diff(F, z)]
    comps = [S] + CA + CS
    els = ['one', 'aI', 'aJ', 'aK', 'SI', 'SJ', 'SK']
    M = sum((OPS[m](comps[k]) * BETA[m] * EL[els[k]] for m in range(4) for k in range(7)), sp.zeros(4))
    tag = ['d0', 'd1', 'd2', 'd3']
    ap = lambda op, F: OPS[tag.index(op)](F)
    ok &= report('d/C/ = rule with d0 = -(1/c) d_t and d = grad, acting on s, c_alpha, c_Sigma',
                 check(sp.expand(M), rule('d0', ['d1', 'd2', 'd3'], S, CA, CS, ap)))
    print('4  Block form: the Pauli triple product inside')
    pv = lambda v, sg=1: sg * v[0] * P['T'] + v[1] * P['I'] + v[2] * P['J'] + v[3] * P['K']
    E = sp.expand(vec4(d) * vec4(a) * vec4(b))
    Ep = sp.expand(pv(d) * pv(a, -1) * pv(b))
    ok &= report('upper-right block of D/(A/B/) = -D(A^T B), minus the BQ-Pauli triple product', zero(E[0:2, 2:4] + Ep))
    ok &= report('lower-left block = D^T A B^T; diagonal blocks zero',
                 zero(E[2:4, 0:2] - pv(d, -1) * pv(a) * pv(b, -1)) and zero(E[0:2, 0:2]) and zero(E[2:4, 2:4]))
    print('5  Maxwell: d/B/ in the Lorenz gauge')
    phi, Av = f4('phi'), sp.Matrix([f4('A1'), f4('A2'), f4('A3')])
    grad = lambda f: sp.Matrix([sp.diff(f, v) for v in (x, y, z)])
    curl = lambda F: sp.Matrix([sp.diff(F[2], y) - sp.diff(F[1], z), sp.diff(F[0], z) - sp.diff(F[2], x), sp.diff(F[1], x) - sp.diff(F[0], y)])
    div = lambda F: sp.diff(F[0], x) + sp.diff(F[1], y) + sp.diff(F[2], z)
    Ef, Bf = -grad(phi) - sp.diff(Av, t), curl(Av)
    M = sum((OPS[m](Ef[k] / c) * BETA[m] * EL[['aI', 'aJ', 'aK'][k]] + OPS[m](Bf[k]) * BETA[m] * EL[['SI', 'SJ', 'SK'][k]]
             for m in range(4) for k in range(3)), sp.zeros(4))
    ch = channels(sp.expand(M))
    want = {'b0': -div(Ef) / c, 'bt0': -div(Bf)}
    want.update({n: -(curl(Bf)[i] - sp.diff(Ef[i], t) / c**2) for i, n in enumerate(['bI', 'bJ', 'bK'])})
    want.update({n: (curl(Ef)[i] + sp.diff(Bf[i], t)) / c for i, n in enumerate(['btI', 'btJ', 'btK'])})
    ok &= report('d/B/: b0 -(1/c) div E, b -(curl B - (1/c^2) d_t E), b~0 -div B, b~ (1/c)(curl E + d_t B)',
                 all(zero(ch[n] - want[n]) for n in want) and all(zero(ch[n]) for n in NAMES if n not in want))
    ok &= report('pseudovector sector vanishes identically for B = curl A, E = -grad phi - d_t A',
                 all(zero(ch[n]) for n in ['bt0', 'btI', 'btJ', 'btK']))
    print('6  Lorentz force: J/B/')
    rho, Jv = sy('rho'), sp.Matrix(sy('J1 J2 J3'))
    Ev, Bv = sp.Matrix(sy('E1 E2 E3')), sp.Matrix(sy('B1 B2 B3'))
    ch = channels(sp.expand(vec4([c * rho, *Jv]) * even(0, list(Ev / c), list(Bv))))
    FL0, FL = Jv.dot(Ev) / c, rho * Ev + Jv.cross(Bv)
    ok &= report('vector part = -F/_L: b0 -(J.E)/c, b -(rho E + J x B)',
                 zero(ch['b0'] + FL0) and all(zero(ch[n] + FL[i]) for i, n in enumerate(['bI', 'bJ', 'bK'])))
    pseudo = Jv.cross(Ev) / c - c * rho * Bv
    ok &= report('pseudovector part: b~0 -(J.B), b~ (1/c) J x E - c rho B',
                 zero(ch['bt0'] + Jv.dot(Bv)) and all(zero(ch[n] - pseudo[i]) for i, n in enumerate(['btI', 'btJ', 'btK'])))
    print('\nAll checks:', 'IDENTICAL' if ok else 'SOME DIFFER')
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
