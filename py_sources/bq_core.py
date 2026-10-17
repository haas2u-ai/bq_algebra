"""
bq_core.py -- shared definitions for the BQ-Pauli reproduction scripts.

Everything here follows the BQ-Pauli definitions of E.P.J. de Haas:

    1^ = [[1,0],[0,1]]      T^ = i 1^ = [[i,0],[0,i]]
    I^ = [[i,0],[0,-i]]     J^ = [[0,1],[-1,0]]      K^ = [[0,i],[i,0]]
    sI = [[1,0],[0,-1]]     sJ = [[0,-i],[i,0]]      sK = [[0,1],[1,0]]

with K^_i = T^ sigma^_i.  A physical four-vector has four REAL coordinates,
A = a0 T^ + a1 I^ + a2 J^ + a3 K^, and the BQ time-reversal operation is
A^T = -a0 T^ + a.K^  (not the matrix transpose).

Products are ordinary 2x2 matrix products, computed exactly with SymPy.
A four-gradient is an operator: in every product the left factor stands to
the left, so its derivatives act on everything to its right.  When the
right-most factor is itself an operator, the product is evaluated on a test
field psi(t, x, y, z).

The scripts verify each channel rule by forming the explicit matrix product,
decomposing it on the eight basis elements, and simplifying the difference
with the rule to zero (exact symbolic verification).
"""
from __future__ import annotations
import re
import sympy as sp

i = sp.I

# ---------------------------------------------------------------------------
# 1. The eight basis matrices
# ---------------------------------------------------------------------------
One = sp.Matrix([[1, 0], [0, 1]])
T = sp.Matrix([[i, 0], [0, i]])
I_ = sp.Matrix([[i, 0], [0, -i]])
J_ = sp.Matrix([[0, 1], [-1, 0]])
K_ = sp.Matrix([[0, i], [i, 0]])
sI = sp.Matrix([[1, 0], [0, -1]])
sJ = sp.Matrix([[0, -i], [i, 0]])
sK = sp.Matrix([[0, 1], [1, 0]])

BASIS = {'T': T, 'I': I_, 'J': J_, 'K': K_, 'sI': sI, 'sJ': sJ, 'sK': sK, '1': One}
ORDER = ['T', 'I', 'J', 'K', 'sI', 'sJ', 'sK', '1']
PRETTY = {'T': 'T^', 'I': 'I^', 'J': 'J^', 'K': 'K^', 'sI': 'sigma_I', 'sJ': 'sigma_J', 'sK': 'sigma_K', '1': '1^'}
SPACE_TIME = {'T', 'I', 'J', 'K'}          # minquat basis; '1' and sigma_i form the spin-norm basis
MINQUAT = [T, I_, J_, K_]                  # K^mu = (T^, I^, J^, K^)
SIGMA = [sI, sJ, sK]

# ---------------------------------------------------------------------------
# 2. Coordinates, fields and the physical four-vectors of the BQ papers
# ---------------------------------------------------------------------------
t, x, y, z = sp.symbols('t x y z', real=True)
X4 = (t, x, y, z)
c = sp.symbols('c', positive=True)


def field(name):
    """A real field of (t, x, y, z)."""
    return sp.Function(name, real=True)(*X4)


psi = field('psi')                        # test field for right-most operators


class Op:
    """A linear differential operator acting on a SymPy expression."""
    def __init__(self, fn, label):
        self.fn, self.label = fn, label

    def __call__(self, e):
        return self.fn(e)

    def scaled(self, k):
        return Op(lambda e, f=self.fn, k=k: k * f(e), f'{k}*{self.label}')


def apply(a, b):
    """Product of a left component a with a right expression b."""
    return a(b) if isinstance(a, Op) else a * b


class Vec:
    def __init__(self, key, name, comps, what):
        self.key, self.name, self.comps, self.what = key, name, list(comps), what

    @property
    def is_op(self):
        return any(isinstance(q, Op) for q in self.comps)

    def time_reversed(self):
        a0 = self.comps[0]
        return [a0.scaled(-1) if isinstance(a0, Op) else -a0] + self.comps[1:]


def _vf(prefix):
    return [field(prefix + s) for s in ('x', 'y', 'z')]


vx, vy, vz = _vf('v')
# gamma is kept as a positive field in the algebra (the identities hold for any gamma);
# its value 1/sqrt(1 - v^2/c^2) is substituted only when results are printed.
gamma = sp.Function('gamma', positive=True)(*X4)
GAMMA_VALUE = 1 / sp.sqrt(1 - (vx**2 + vy**2 + vz**2) / c**2)
E, px, py, pz = sp.symbols('E p_x p_y p_z', real=True)
dt, dx, dy, dz = sp.symbols('dt dx dy dz', real=True)

VECTORS = {
    'R':  Vec('R', 'R', [c * t, x, y, z], 'space-time position (ct, r)'),
    'dR': Vec('dR', 'dR', [c * dt, dx, dy, dz], 'displacement (c dt, dr)'),
    'V':  Vec('V', 'V', [c, vx, vy, vz], 'coordinate velocity (c, v)'),
    'U':  Vec('U', 'U', [gamma * c, gamma * vx, gamma * vy, gamma * vz], 'proper velocity (gamma c, gamma v)'),
    'P':  Vec('P', 'P', [E / c, px, py, pz], 'energy-momentum (E/c, p)'),
    'D':  Vec('D', 'd', [Op(lambda e: -sp.diff(e, t) / c, '-(1/c)d_t'),
                          Op(lambda e: sp.diff(e, x), 'd_x'),
                          Op(lambda e: sp.diff(e, y), 'd_y'),
                          Op(lambda e: sp.diff(e, z), 'd_z')], 'four-gradient (-(1/c) d_t, nabla)'),
    'A':  Vec('A', 'A', [field('phi') / c] + _vf('A'), 'electromagnetic four-potential (phi/c, A)'),
    'J':  Vec('J', 'J', [c * field('rho')] + _vf('J'), 'four-current density (c rho, J)'),
    'G':  Vec('G', 'G', [field('epsilon') / c] + _vf('g'), 'fluid energy-momentum density (epsilon/c, g)'),
    'F':  Vec('F', 'F', [field('Pw') / c] + _vf('F'), 'force-power density (Pw/c, F)'),
    'a':  Vec('a', 'A_gen', [field('a0'), field('a1'), field('a2'), field('a3')], 'general four-vector (a0, a)'),
    'b':  Vec('b', 'B_gen', [field('b0'), field('b1'), field('b2'), field('b3')], 'general four-vector (b0, b)'),
    'd':  Vec('d', 'D_gen', [field('d0'), field('d1'), field('d2'), field('d3')], 'general four-vector (d0, d)'),
}

# ---------------------------------------------------------------------------
# 3. Explicit matrix products with operator support
# ---------------------------------------------------------------------------
def op_matrix(comps):
    """2x2 matrix whose entries are lists of (coefficient, component) pairs."""
    M = [[[], []], [[], []]]
    for q, B in zip(comps, MINQUAT):
        for r in range(2):
            for s in range(2):
                if B[r, s] != 0:
                    M[r][s].append((B[r, s], q))
    return M


def eval_matrix(comps):
    """Matrix of a vector with plain (non-operator) components; operators act on psi."""
    vals = [q(psi) if isinstance(q, Op) else q for q in comps]
    return sp.Matrix(2, 2, lambda r, s: sum(v * B[r, s] for v, B in zip(vals, MINQUAT)))


def left_multiply(comps, M):
    """(sum_mu comp_mu K^mu) * M, with operator components acting on M's entries."""
    L = op_matrix(comps)
    return sp.Matrix(2, 2, lambda r, s: sp.expand(sum(
        coef * apply(q, M[k, s]) for k in range(2) for coef, q in L[r][k])))


def decompose(M):
    """Real coefficients of M on 1^, T^, K^_i, sigma_i (exact)."""
    half = sp.Rational(1, 2)
    z0 = sp.expand(half * M.trace())
    zs = [sp.expand(half * (S * M).trace()) for S in SIGMA]
    # derivatives of real fields are real: swap them for real placeholders while taking Re/Im
    ders = set().union(*[e.atoms(sp.Derivative) for e in [z0] + zs])
    rep = {d: sp.Dummy(real=True) for d in ders}
    back = {v: k for k, v in rep.items()}
    re = lambda e: sp.expand(sp.re(e.xreplace(rep)).xreplace(back))
    im = lambda e: sp.expand(sp.im(e.xreplace(rep)).xreplace(back))
    return {'1': re(z0), 'T': im(z0), 'K': [im(w) for w in zs], 's': [re(w) for w in zs]}


def zero(e):
    e = sp.expand(e)
    return e == 0 or sp.simplify(e) == 0


# ---------------------------------------------------------------------------
# 4. The channel rules, evaluated with the operator order kept (a before b)
# ---------------------------------------------------------------------------
def right_values(vec):
    """Components of the right-most factor; an operator acts on psi."""
    return [q(psi) if isinstance(q, Op) else q for q in vec.comps]


def bilinear_rule(a, b):
    """A^T B = (a0 b0 - a.b) 1^ + (a x b).K^ + (a0 b - a b0).sigma^ ; a may hold operators."""
    a0, av = a[0], a[1:]
    b0, bv = b[0], b[1:]
    m = apply
    c1 = m(a0, b0) - sum(m(av[k], bv[k]) for k in range(3))
    cK = [m(av[1], bv[2]) - m(av[2], bv[1]), m(av[2], bv[0]) - m(av[0], bv[2]), m(av[0], bv[1]) - m(av[1], bv[0])]
    cS = [m(a0, bv[k]) - m(av[k], b0) for k in range(3)]
    return {'1': c1, 'T': sp.Integer(0), 'K': cK, 's': cS}


def triple_rule(d, C):
    """E = D C = -d.cK 1^ + (d0 c1 + d.cs) T^ + (d c1 + d0 cs + d x cK).K^ + (d x cs - d0 cK).sigma^"""
    m = apply
    d0, dv = d[0], d[1:]
    c1, cK, cS = C['1'], C['K'], C['s']
    cross = lambda u, w: [m(u[1], w[2]) - m(u[2], w[1]), m(u[2], w[0]) - m(u[0], w[2]), m(u[0], w[1]) - m(u[1], w[0])]
    dxK, dxS = cross(dv, cK), cross(dv, cS)
    return {'1': -sum(m(dv[k], cK[k]) for k in range(3)),
            'T': m(d0, c1) + sum(m(dv[k], cS[k]) for k in range(3)),
            'K': [m(dv[k], c1) + m(d0, cS[k]) + dxK[k] for k in range(3)],
            's': [dxS[k] - m(d0, cK[k]) for k in range(3)]}


def compare(ch_matrix, ch_rule):
    """True when every channel of the matrix product equals the rule exactly."""
    pairs = [(ch_matrix['1'], ch_rule['1']), (ch_matrix['T'], ch_rule['T'])]
    pairs += list(zip(ch_matrix['K'], ch_rule['K'])) + list(zip(ch_matrix['s'], ch_rule['s']))
    return all(zero(p - q) for p, q in pairs)


# ---------------------------------------------------------------------------
# 5. Readable output
# ---------------------------------------------------------------------------
def show(e):
    s = str(sp.simplify(sp.sympify(e).subs(gamma, GAMMA_VALUE)))
    s = s.replace('(t, x, y, z)', '')
    def der(m):
        f, args = m.group(1), m.group(2)
        out = []
        for v, n in re.findall(r'\((\w), (\d+)\)|(\w)', args) and [(p[0] or p[2], p[1] or '1') for p in re.findall(r'\((\w), (\d+)\)|(\w)', args)]:
            out.append(f'd_{v}' + (f'^{n}' if n != '1' else ''))
        return ' '.join(out) + ' ' + f
    s = re.sub(r'Derivative\((\w+), ((?:\(\w, \d+\)|\w)(?:, (?:\(\w, \d+\)|\w))*)\)', der, s)
    return s


def show_channels(ch, title):
    lines = [title]
    lines.append(f"  1^      : {show(ch['1'])}")
    if 'T' in ch:
        lines.append(f"  T^      : {show(ch['T'])}")
    for lab, key in (('K^', 'K'), ('sigma^', 's')):
        lines.append(f"  {lab:<8}: ({', '.join(show(v) for v in ch[key])})")
    return '\n'.join(lines)
