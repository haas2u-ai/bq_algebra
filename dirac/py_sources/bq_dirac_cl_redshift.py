"""
bq_dirac_cl_redshift.py -- CL as an equi-redshift flow at the Dirac level (Weyl representation), as on the page
BQ_Dirac_CL_Equi_Redshift.html. The BQ-Pauli version is the page BQ_CL_Killing.html.

Galilean coframe with a radial and an azimuthal river (paper chart, c kept):
    theta/_t = c beta_0 - v_r beta_r - v_phi beta_phi,  theta/_r = beta_r, ...,  g_tt = -(c^2 - v_r^2 - v_phi^2)
    river observer       u/   = c beta_0                       (theta^r = theta^phi = 0, unit lapse)
    static observer      u/_s = (c/|xi|) theta/_t,  |xi| = sqrt(-g_tt)   (along the Killing vector xi = d_t)
Exact rotor level (flat, the nested rotor of the gravity-rotor page): beta^_0 = Q_g beta_0 Q_g^-1, Q_g = Q_g,phi Q_g,r.

Checked exactly with SymPy:
  1  u/ u/ = u/_s u/_s = -c^2 1: both are unit timelike vectors
  2  the observer product: u/_s u/ = -c^2 gamma_rs (1 - (v_r/c) alpha_r - (v_phi/c) alpha_phi), a scalar plus a boost
     bivector only (no Sigma, beta, beta~ or chi channel); with (n.alpha)^2 = 1 it is -c^2 exp(-chi_rs n.alpha),
     cosh chi_rs = gamma_rs = c/sqrt(-g_tt), tanh chi_rs = sqrt(v_r^2 + v_phi^2)/c
  3  gamma_rs = -(1/(2c^2)) <{u/, u/_s}>: the Pauli-page inner product as an anticommutator
  4  Killing energy: -(1/2){u/, theta/_t} = c^2 1, i.e. E = c^2 for every river element
  5  rotor level: beta^_0 = Q_g beta_0 Q_g^-1 is the exact image of theta/_t / c, the static observer's time direction, and
     beta_0 the river's, so beta^_0 beta_0 corresponds to u/_s u/:
     beta^_0 beta_0 = -(Gamma 1 - sinh(chi_r) alpha_r - cosh(chi_r) sinh(chi_phi) alpha_phi), again only
     a scalar and a boost bivector, with Gamma = cosh chi_r cosh chi_phi = Gamma_rotor; -(1/2)<{beta^_0, beta_0}> = Gamma_rotor;
     the rotor Q_g itself has a Sigma_J (r-phi rotation) part, sinh(chi_r/2) sinh(chi_phi/2), which drops out of beta^_0 beta_0
  6  rotor versus metric: with v = c chi, gamma_rs - Gamma_rotor = (2 chi_r^4 + 3 chi_r^2 chi_phi^2 + 2 chi_phi^4)/6 + O(chi^6):
     relativistic composition of the two rapidities (rotor) versus the Galilean sum of the two river velocities (metric)
  7  CL: along a stationary, axisymmetric flow D/Dtau = v_r d_r, and v_r d_r gamma_rs = v_r gamma_rs^3 d_r(v_r^2 + v_phi^2)/(2c^2):
     where v_r != 0, constant gamma_rs <=> v_r^2 + v_phi^2 = K (the scalar part of u/_s u/ constant along the flow)

Usage
  python bq_dirac_cl_redshift.py
"""
import sympy as sp
from bq_dirac_table import P, blk

Z2 = sp.zeros(2)


def report(name, ok):
    print(f'  {name:<96} {"IDENTICAL (exact)" if ok else "DIFFER"}')
    return ok


def zero(M):
    M = M if isinstance(M, sp.MatrixBase) else sp.Matrix([M])
    return all(sp.simplify(sp.cancel(sp.together(sp.expand(x)))) == 0 for x in M)


def main():
    ok = True
    c = sp.Symbol('c', positive=True)
    vr, vp = sp.symbols('v_r v_phi', real=True)
    one4 = sp.eye(4)
    b0, bI, bJ, bK = [blk(Z2, P['T'], P['T'], Z2)] + [blk(Z2, P[k], -P[k], Z2) for k in 'IJK']
    aI, aK = b0 * bI, b0 * bK
    SI, SJ, SK = bJ * bK, bK * bI, bI * bJ
    chi = -b0 * bI * bJ * bK
    basis = {'1': one4, 'b0': b0, 'bI': bI, 'bJ': bJ, 'bK': bK, 'aI': aI, 'aJ': b0 * bJ, 'aK': aK, 'SI': SI, 'SJ': SJ, 'SK': SK,
             'bt0': chi * b0, 'btI': chi * bI, 'btJ': chi * bJ, 'btK': chi * bK, 'chi': chi}
    coef = lambda X, M: sp.simplify((X * M).trace() / (X * X).trace())

    xi = sp.sqrt(c**2 - vr**2 - vp**2)
    g = c / xi
    tht = c * b0 - vr * bI - vp * bK                    # beta_r = beta_I, beta_phi = beta_K
    u, us = c * b0, (c / xi) * tht
    print('1  Two unit timelike vectors')
    ok &= report('u/ u/ = -c^2 1,  u/_s u/_s = -c^2 1', zero(u * u + c**2 * one4) and zero(us * us + c**2 * one4))
    print('2  The observer product')
    pr = us * u
    ok &= report('u/_s u/ = -c^2 gamma_rs (1 - (v_r/c) alpha_r - (v_phi/c) alpha_phi)', zero(pr + c**2 * g * (one4 - (vr / c) * aI - (vp / c) * aK)))
    ok &= report('reversed order u/ u/_s: the same scalar part, the bivector part with opposite sign', zero(u * us + c**2 * g * (one4 + (vr / c) * aI + (vp / c) * aK)))
    rest = [k for k in basis if k not in ('1', 'aI', 'aK')]
    ok &= report('only the 1 and boost-bivector channels: no Sigma, beta, beta~ or chi', all(coef(basis[k], pr) == 0 for k in rest))
    nI, nK = sp.symbols('n_I n_K', real=True)
    na = nI * aI + nK * aK
    ok &= report('(n.alpha)^2 = (n_I^2 + n_K^2) 1: so exp(chi n.alpha) = cosh chi 1 + sinh chi n.alpha for |n| = 1',
                 zero(na * na - (nI**2 + nK**2) * one4))
    V = sp.sqrt(vr**2 + vp**2)
    ok &= report('cosh chi_rs = gamma_rs, sinh chi_rs = gamma_rs |v|/c: cosh^2 - sinh^2 = 1 (tanh chi_rs = |v|/c)',
                 zero(g**2 - (g * V / c)**2 - 1))
    print('3  gamma_rs as an anticommutator')
    ok &= report('-(1/(2c^2)) {u/, u/_s} = gamma_rs 1 = c/sqrt(-g_tt) 1', zero(-(u * us + us * u) / (2 * c**2) - g * one4))
    print('4  Killing energy')
    ok &= report('-(1/2){u/, theta/_t} = c^2 1: E = c^2', zero(-(u * tht + tht * u) / 2 - c**2 * one4))

    print('5  The rotor level')
    z, w = sp.symbols('z w', positive=True)          # z = exp(chi_r/2), w = exp(chi_phi/2)
    cr, sr = (z + 1 / z) / 2, (z - 1 / z) / 2
    cp, spp = (w + 1 / w) / 2, (w - 1 / w) / 2
    Gr, Gri = cr * P['1'] + sr * P['sI'], cr * P['1'] - sr * P['sI']
    Gp, Gpi = cp * P['1'] + spp * P['sK'], cp * P['1'] - spp * P['sK']
    Q, Qi = blk(Gpi, Z2, Z2, Gp) * blk(Gri, Z2, Z2, Gr), blk(Gr, Z2, Z2, Gri) * blk(Gp, Z2, Z2, Gpi)
    C1, S1 = (z**2 + z**-2) / 2, (z**2 - z**-2) / 2      # cosh, sinh chi_r
    C2, S2 = (w**2 + w**-2) / 2, (w**2 - w**-2) / 2      # cosh, sinh chi_phi
    bh0 = Q * b0 * Qi
    prr = bh0 * b0
    Gam = C1 * C2
    ok &= report('beta^_0 beta_0 = -(Gamma 1 - sinh chi_r alpha_r - cosh chi_r sinh chi_phi alpha_phi), Gamma = cosh chi_r cosh chi_phi',
                 zero(prr + (Gam * one4 - C1 * S2 * aK - S1 * aI)))
    ok &= report('only the 1 and boost-bivector channels', all(coef(basis[k], prr) == 0 for k in rest))
    ok &= report('-(1/2){beta^_0, beta_0} = Gamma_rotor 1', zero(-(bh0 * b0 + b0 * bh0) / 2 - Gam * one4))
    sig = [coef(basis[k], Q) for k in ('SI', 'SJ', 'SK')]
    ok &= report('Q_g has a Sigma_J part sinh(chi_r/2) sinh(chi_phi/2) (rotation in the r-phi plane, the Wigner-type rotation)',
                 zero(sp.Matrix([sig[0], sig[2], sig[1] - sr * spp])))

    print('6  Rotor versus metric')
    a, bb = sp.symbols('chi_r chi_phi', positive=True)
    gm = 1 / sp.sqrt(1 - a**2 - bb**2)
    gr = sp.cosh(a) * sp.cosh(bb)
    t = sp.Symbol('t')
    ser = sp.series((gm - gr).subs({a: t * a, bb: t * bb}), t, 0, 6).removeO()
    ok &= report('gamma_rs - Gamma_rotor = (2 chi_r^4 + 3 chi_r^2 chi_phi^2 + 2 chi_phi^4)/6 + O(chi^6)  (v = c chi)',
                 sp.simplify(ser.subs(t, 1) - (2 * a**4 + 3 * a**2 * bb**2 + 2 * bb**4) / 6) == 0)

    print('7  CL along the flow')
    r = sp.Symbol('r', positive=True)
    Vr, Vp = sp.Function('v_r')(r), sp.Function('v_phi')(r)
    gR = c / sp.sqrt(c**2 - Vr**2 - Vp**2)
    ok &= report('v_r d_r gamma_rs = v_r gamma_rs^3 d_r(v_r^2 + v_phi^2)/(2c^2)',
                 sp.simplify(Vr * sp.diff(gR, r) - Vr * gR**3 * sp.diff(Vr**2 + Vp**2, r) / (2 * c**2)) == 0)
    print('\nAll checks:', 'IDENTICAL' if ok else 'SOME DIFFER')
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
