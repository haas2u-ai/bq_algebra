"""
bq_quadratics.py -- the field quadratics FF and F^T F of a BQ field
F = f 1^ + k.K^ + s.sigma^ , with the time adjoint F^T = f 1^ + k.K^ - s.sigma^.

General rule for two such fields F = (f, k, s), G = (g, m, t):
  FG = (fg - k.m + s.t) 1^ + (k.t + s.m) T^
       + (f m + g k + k x m - s x t).K^ + (f t + g s + k x t + s x m).sigma^
Each product is checked exactly (symbolically) along three routes that must agree:
  rule route  : the formula above;
  table route : term by term with the BQ-Pauli multiplication table (49 products);
  matrix route: explicit 2x2 matrices, multiplied and decomposed.
For the electromagnetic and fluid fields the result is also compared with the
equations of de Haas (2026), BQ electromagnetism and fluid dynamics.

Usage
  python bq_quadratics.py B            EM field B = d^T A in the Lorenz gauge (f = 0)
  python bq_quadratics.py B --full     the same with the gauge term B0 kept
  python bq_quadratics.py H            fluid field H = d^T G
  python bq_quadratics.py --all        every field, with and without f, all routes
Fields: B (EM), H (fluid), M (R^T P), JA (J^T A), UG (U^T G), gen (general)
"""
import sys
import sympy as sp
from bq_core import BASIS, SIGMA, decompose, table_product, as_channels, compare, show_channels, c

ONE, T, KB = BASIS['1'], BASIS['T'], [BASIS['I'], BASIS['J'], BASIS['K']]


def vec(name):
    return sp.Matrix(sp.symbols(f'{name}_x {name}_y {name}_z', real=True))


B0, sE, S_act, L_lag, L_fl, f_gen = sp.symbols('B_0 sigma_E S L 𝓛 f', real=True)
Bv, Ev, Om, Pi, Lv, Mv, Tv, Sv, TK, TS, kv, sv = (vec(n) for n in ('B', 'E', 'Omega', 'Pi', 'L', 'M', 'T', 'S', 'T_K', 'T_s', 'k', 's'))

# F = (f, k, s): the three channels; f is dropped with f0 = True
FIELDS = {
    'B':   ('EM field B = d^T A',          B0,      Bv,  Ev / c),
    'H':   ('fluid field H = d^T G',       -sE,     Om,  -Pi / c),
    'M':   ('R^T P (action, L, M)',         -S_act,  Lv,  Mv),
    'JA':  ('J^T A (L, T, S)',              L_lag,   Tv,  Sv),
    'UG':  ('U^T G (Lagrangian, T_K, T_s)', L_fl,    TK,  TS),
    'gen': ('general field (f, k, s)',      f_gen,   kv,  sv),
}


def rule(F, G):
    (f, k, s), (g, m, t) = F, G
    return {'1': f*g - k.dot(m) + s.dot(t), 'T': k.dot(t) + s.dot(m),
            'K': list(f*m + g*k + k.cross(m) - s.cross(t)), 's': list(f*t + g*s + k.cross(t) + s.cross(m))}


def expansion(F):
    f, k, s = F
    return {'1': f, 'I': k[0], 'J': k[1], 'K': k[2], 'sI': s[0], 'sJ': s[1], 'sK': s[2]}


def matrix(F):
    f, k, s = F
    return f*ONE + sum((k[i]*KB[i] + s[i]*SIGMA[i] for i in range(3)), sp.zeros(2, 2))


def adjoint(F):
    return (F[0], F[1], -F[2])


def paper_checks(key, F, FF, FTF):
    """Compare with the published forms (de Haas 2026, BQ EM and fluid dynamics)."""
    out = []
    f, k, s = F
    if key == 'B':
        B2, E2, EB, ExB = Bv.dot(Bv), Ev.dot(Ev), Ev.dot(Bv), Ev.cross(Bv)
        ok1 = sp.simplify(FF['1'] - (f**2 - (B2 - E2/c**2))) == 0 and sp.simplify(FF['T'] - 2*EB/c) == 0
        ok2 = sp.simplify(FTF['1'] - (f**2 - (B2 + E2/c**2))) == 0 and all(sp.simplify(FTF['s'][i] + 2*ExB[i]/c) == 0 for i in range(3))
        out += [('BB = -(B^2 - E^2/c^2) 1^ + (2/c)(E.B) T^      (eq. 95, with B0 = 0)', ok1),
                ('B^T B = -(B^2 + E^2/c^2) 1^ - (2/c)(E x B).sigma^ (eq. 102, with B0 = 0)', ok2)]
    if key == 'H':
        O2, P2, PxO = Om.dot(Om), Pi.dot(Pi), Pi.cross(Om)
        ok = (sp.simplify(FTF['1'] - (sE**2 - O2 - P2/c**2)) == 0 if f != 0 else sp.simplify(FTF['1'] - (-O2 - P2/c**2)) == 0) \
            and all(sp.simplify(FTF['K'][i] - (-2*sE*Om[i] if f != 0 else 0)) == 0 for i in range(3)) \
            and all(sp.simplify(FTF['s'][i] - 2*PxO[i]/c) == 0 for i in range(3))
        out.append(('H^T H = (sE^2 - Om^2 - Pi^2/c^2) 1^ - 2 sE Om.K^ + (2/c)(Pi x Om).sigma^ (eq. 278)', ok))
    return out


def quadratics(key, f0=False, verbose=True):
    name, f, k, s = FIELDS[key]
    F = (sp.Integer(0) if f0 else f, k, s)
    Ft = adjoint(F)
    ok_all = True
    res = {}
    for label, L in (('FF', F), ('F^T F', Ft)):
        r = rule(L, F)
        tab = as_channels(table_product(expansion(L), expansion(F)))
        mat = decompose(matrix(L) * matrix(F))
        ok_t, ok_m = compare(tab, r), compare(mat, r)
        ok_all &= ok_t and ok_m
        res[label] = r
        if verbose:
            print(show_channels(r, f'{label}  for {name}' + ('  (f = 0)' if f0 else '')))
            print(f'  table route  and rule: {"IDENTICAL (exact)" if ok_t else "DIFFER"}')
            print(f'  matrix route and rule: {"IDENTICAL (exact)" if ok_m else "DIFFER"}')
    checks = paper_checks(key, F, res['FF'], res['F^T F']) if (key == 'H' or f0) else []
    for txt, ok in checks:
        ok_all &= ok
        if verbose:
            print(f'  paper: {txt}: {"AGREES" if ok else "DIFFERS"}')
    return ok_all


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '--full']
    full = '--full' in sys.argv
    if args == ['--all']:
        res = [(k, z, quadratics(k, z, verbose=False)) for k in FIELDS for z in (True, False)]
        bad = [(k, z) for k, z, ok in res if not ok]
        print(f'{len(res)} field cases checked (FF and F^T F, rule, table and matrix routes, paper forms), {len(bad)} failures', bad or '')
        sys.exit(1 if bad else 0)
    elif len(args) == 1 and args[0] in FIELDS:
        f0 = (args[0] == 'B' and not full)
        sys.exit(0 if quadratics(args[0], f0) else 1)
    else:
        print(__doc__)
