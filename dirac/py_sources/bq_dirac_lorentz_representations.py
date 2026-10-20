"""
bq_dirac_lorentz_representations.py -- the Lorentz boost in the Weyl and in the Dirac representation, connected by
the Weyl-to-Dirac matrix S, as on the page BQ_Dirac_Lorentz_Weyl_and_Dirac.html.

Weyl:   beta_0 = [[0, T^], [T^, 0]],   a/_W = [[0, A], [-A^T, 0]],                 Lambda_W = diag(U^-1, U)
Dirac:  beta_0 = [[T^, 0], [0, -T^]],  a/_D = [[a0 T^, a.K^], [-a.K^, -a0 T^]]      (beta_i the same in both)
S = (1/sqrt 2)[[1^, 1^], [-1^, 1^]],  X_D = S X_W S^-1,  U = exp(psi sigma_I/2) = cosh(psi/2) 1^ + sinh(psi/2) sigma_I

Checked exactly with SymPy (rational in z = exp(psi/2), so c = cosh(psi/2) = (z + 1/z)/2, s = sinh(psi/2) = (z - 1/z)/2):
  1  Lambda_D = S Lambda_W S^-1 = (1/2)[[U^-1 + U, U - U^-1], [U - U^-1, U^-1 + U]] = [[c 1^, s sigma_I], [s sigma_I, c 1^]]
  2  rotor form in both: Lambda_W = c 1 - s alpha_I^W,  Lambda_D = c 1 - s alpha_I^D,  alpha_I^D = S alpha_I^W S^-1
  3  the square commutes: Lambda_D a/_D Lambda_D^-1 = S (Lambda_W a/_W Lambda_W^-1) S^-1
  4  the coordinates are the same: a/_W^L and a/_D^L both have a0' = g(a0 - b a1), a1' = g(a1 - b a0), a2, a3;
     in the Dirac blocks: a/_D^L = [[a0' T^, a'.K^], [-a'.K^, -a0' T^]], with cosh psi = c^2 + s^2 = g and
     sinh psi = 2cs = b g; the inter-block terms use sigma_I I^ = T^ and sigma_I T^ = I^
  5  the invariant a/a/ = -(a0^2 - a^2) 1 is the same in both representations and under the boost
  6  rotations: S diag(r, r) S^-1 = diag(r, r)
  7  spinors, Psi^L = Lambda Psi (BQ Dirac Gravity eqs. 103-104): Weyl keeps the two blocks apart,
     [xi; eta] -> [U^-1 xi; U eta]; Dirac mixes them, [phi; 0] -> [c phi; s sigma_I phi] and [0; eta] -> [s sigma_I eta; c eta];
     Psi_D = S Psi_W is carried along: Lambda_D (S Psi_W) = S (Lambda_W Psi_W)
  8  intensities: Lambda is Hermitian in both, Psi^dagger Psi -> Psi^dagger Lambda^2 Psi with
     Lambda_W^2 = diag(e^-psi, e^psi, e^psi, e^-psi) (the Doppler factors) and
     Lambda_D^2 = [[cosh psi 1^, sinh psi sigma_I], [sinh psi sigma_I, cosh psi 1^]] = cosh psi 1 - sinh psi alpha_I^D
     (BQ Dirac Gravity eq. 107 writes + alpha_I sinh psi; with alpha_I^D = [[0, -sigma_I], [-sigma_I, 0]] of From Weyl to
     Dirac eq. 14 the sign is minus -- reported, not changed; eq. 107 agrees if its alpha_I is the standard Dirac
     alpha = [[0, sigma_I], [sigma_I, 0]] = sigma_I^D below)
  9  exponential form, the same in both: with the 4x4 lifts of sigma_I
     sigma_I^W = [[-sigma_I, 0], [0, sigma_I]] = -alpha_I^W = diag(-1, 1, 1, -1),
     sigma_I^D = S sigma_I^W S^-1 = [[0, sigma_I], [sigma_I, 0]] = -alpha_I^D,
     (sigma_I^W)^2 = (sigma_I^D)^2 = 1, so exp((psi/2) sigma_I^X) = cosh(psi/2) 1 + sinh(psi/2) sigma_I^X = Lambda_X, exactly as
     the Pauli U = exp((psi/2) sigma_I) = cosh(psi/2) 1^ + sinh(psi/2) sigma_I; S exp(X) S^-1 = exp(S X S^-1); both have
     the eigenvalues e^(psi/2), e^(psi/2), e^(-psi/2), e^(-psi/2)

Usage
  python bq_dirac_lorentz_representations.py
"""
import sympy as sp
from bq_dirac_table import P, blk

Z2 = sp.zeros(2)


def report(name, ok):
    print(f'  {name:<96} {"IDENTICAL (exact)" if ok else "DIFFER"}')
    return ok


def zero(M):
    return all(sp.cancel(sp.together(sp.expand(x))) == 0 for x in M)


def main():
    ok = True
    z = sp.Symbol('z', positive=True)
    c, s = (z + 1 / z) / 2, (z - 1 / z) / 2
    a = sp.symbols('a0:4', real=True)
    g, gb = (z**2 + z**-2) / 2, (z**2 - z**-2) / 2          # gamma = cosh psi, beta gamma = sinh psi

    bW = [blk(Z2, P['T'], P['T'], Z2)] + [blk(Z2, P[k], -P[k], Z2) for k in 'IJK']
    bD = [blk(P['T'], Z2, Z2, -P['T'])] + bW[1:]
    S = blk(P['1'], P['1'], -P['1'], P['1']) / sp.sqrt(2)
    Si = S.T
    one4 = sp.eye(4)
    U = c * P['1'] + s * P['sI']
    Ui = c * P['1'] - s * P['sI']
    LW, LWi = blk(Ui, Z2, Z2, U), blk(U, Z2, Z2, Ui)
    LD, LDi = S * LW * Si, S * LWi * Si
    aW, aD = bW[0] * bW[1], bD[0] * bD[1]

    print('1  Lambda_D from Lambda_W')
    ok &= report('U = exp(psi sigma_I/2) = diag(z, 1/z): the Pauli rotor', zero(U - sp.diag(z, 1 / z)))
    ok &= report('S Lambda_W S^-1 = (1/2)[[U^-1 + U, U - U^-1], [U - U^-1, U^-1 + U]]',
                 zero(LD - blk(Ui + U, U - Ui, U - Ui, Ui + U) / 2))
    ok &= report('              = [[c 1, s sigma_I], [s sigma_I, c 1]]', zero(LD - blk(c * P['1'], s * P['sI'], s * P['sI'], c * P['1'])))
    print('2  Rotor form')
    ok &= report('Lambda_W = c 1 - s alpha_I^W, Lambda_D = c 1 - s alpha_I^D, alpha_I^D = S alpha_I^W S^-1',
                 zero(LW - (c * one4 - s * aW)) and zero(LD - (c * one4 - s * aD)) and zero(S * aW * Si - aD))
    print('3  The square commutes')
    AW = sum((a[k] * bW[k] for k in range(4)), sp.zeros(4))
    AD = sum((a[k] * bD[k] for k in range(4)), sp.zeros(4))
    ALW, ALD = LW * AW * LWi, LD * AD * LDi
    ok &= report('S a/_W S^-1 = a/_D (same coordinates)', zero(S * AW * Si - AD))
    ok &= report('Lambda_D a/_D Lambda_D^-1 = S (Lambda_W a/_W Lambda_W^-1) S^-1', zero(ALD - S * ALW * Si))
    print('4  The same coordinates')
    ap = [g * a[0] - gb * a[1], g * a[1] - gb * a[0], a[2], a[3]]
    ok &= report("Weyl:  a/_W^L = sum a'_mu beta_mu^W,  a0' = g(a0 - b a1), a1' = g(a1 - b a0)", zero(ALW - sum((ap[k] * bW[k] for k in range(4)), sp.zeros(4))))
    ok &= report("Dirac: a/_D^L = sum a'_mu beta_mu^D, the same a'_mu", zero(ALD - sum((ap[k] * bD[k] for k in range(4)), sp.zeros(4))))
    aK = lambda v: v[1] * P['I'] + v[2] * P['J'] + v[3] * P['K']
    ok &= report("a/_D^L = [[a0' T, a'.K], [-a'.K, -a0' T]]", zero(ALD - blk(ap[0] * P['T'], aK(ap), -aK(ap), -ap[0] * P['T'])))
    ok &= report('c^2 + s^2 = cosh psi = g, 2cs = sinh psi = b g; sigma_I I = T, sigma_I T = I',
                 sp.cancel(c**2 + s**2 - g) == 0 and sp.cancel(2 * c * s - gb) == 0
                 and zero(P['sI'] * P['I'] - P['T']) and zero(P['sI'] * P['T'] - P['I']))
    print('5  The invariant')
    q = -(a[0]**2 - a[1]**2 - a[2]**2 - a[3]**2)
    ok &= report('a/a/ = -(a0^2 - a^2) 1 in both, before and after the boost',
                 all(zero(M * M - q * one4) for M in (AW, AD, ALW, ALD)))
    print('6  Rotations')
    th = sp.Symbol('theta', real=True)
    r = sp.cos(th / 2) * P['1'] + sp.sin(th / 2) * P['K']
    ok &= report('S diag(r, r) S^-1 = diag(r, r)', zero(S * blk(r, Z2, Z2, r) * Si - blk(r, Z2, Z2, r)))
    print('7  Spinors, Psi^L = Lambda Psi')
    x = sp.Matrix(sp.symbols('x1:3')); y = sp.Matrix(sp.symbols('y1:3'))
    ok &= report('Weyl: [xi; eta] -> [U^-1 xi; U eta], the blocks stay apart',
                 zero(LW * sp.Matrix.vstack(x, y) - sp.Matrix.vstack(Ui * x, U * y)))
    ok &= report('Dirac: [phi; 0] -> [c phi; s sigma_I phi],  [0; eta] -> [s sigma_I eta; c eta]',
                 zero(LD * sp.Matrix.vstack(x, sp.zeros(2, 1)) - sp.Matrix.vstack(c * x, s * P['sI'] * x))
                 and zero(LD * sp.Matrix.vstack(sp.zeros(2, 1), y) - sp.Matrix.vstack(s * P['sI'] * y, c * y)))
    ok &= report('Lambda_D (S Psi_W) = S (Lambda_W Psi_W)', zero(LD * S * sp.Matrix.vstack(x, y) - S * LW * sp.Matrix.vstack(x, y)))
    print('8  Intensities')
    ok &= report('Lambda_W, Lambda_D Hermitian; Lambda_W^2 = diag(e^-psi, e^psi, e^psi, e^-psi)',
                 zero(LW - LW.H) and zero(LD - LD.H) and zero(LW * LW - sp.diag(z**-2, z**2, z**2, z**-2)))
    ok &= report('Lambda_D^2 = [[cosh psi 1, sinh psi sigma_I], [sinh psi sigma_I, cosh psi 1]] = cosh psi 1 - sinh psi alpha_I^D',
                 zero(LD * LD - blk(g * P['1'], gb * P['sI'], gb * P['sI'], g * P['1'])) and zero(LD * LD - (g * one4 - gb * aD)))
    print('9  Exponential form')
    psi = sp.Symbol('psi', real=True)
    sW, sD = blk(-P['sI'], Z2, Z2, P['sI']), blk(Z2, P['sI'], P['sI'], Z2)
    ok &= report('sigma_I^W = -alpha_I^W = diag(-1, 1, 1, -1), sigma_I^D = S sigma_I^W S^-1 = [[0, sigma_I], [sigma_I, 0]] = -alpha_I^D',
                 zero(sW + aW) and zero(sW - sp.diag(-1, 1, 1, -1)) and zero(S * sW * Si - sD) and zero(sD + aD))
    ok &= report('(sigma_I^W)^2 = (sigma_I^D)^2 = 1, as sigma_I^2 = 1', zero(sW * sW - one4) and zero(sD * sD - one4) and zero(P['sI'] ** 2 - P['1']))
    E = lambda M: ((psi / 2) * M).exp().applyfunc(lambda x: sp.simplify(x.rewrite(sp.exp)))
    zp = sp.exp(psi / 2)
    sub = lambda M: M.subs(z, zp).applyfunc(lambda x: sp.simplify(x.rewrite(sp.exp)))
    ok &= report('exp((psi/2) sigma_I^W) = Lambda_W,  exp((psi/2) sigma_I^D) = Lambda_D  (matrix exponential)',
                 zero(E(sW) - sub(LW)) and zero(E(sD) - sub(LD)) and zero(E(P['sI']) - sub(U)))
    ok &= report('= cosh(psi/2) 1 + sinh(psi/2) sigma_I^X in both', zero(LW - (c * one4 + s * sW)) and zero(LD - (c * one4 + s * sD)))
    ok &= report('the same eigenvalues e^(psi/2) (twice), e^(-psi/2) (twice) in both',
                 LW.eigenvals() == {z: 2, 1 / z: 2} and {sp.simplify(k): v for k, v in LD.eigenvals().items()} == {z: 2, 1 / z: 2})

    print('\nAll checks:', 'IDENTICAL' if ok else 'SOME DIFFER')
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
