# BQ-Pauli algebra: Python reproduction scripts

Independent Python/SymPy implementation of the nine interactive models in this
repository (the web pages run their own JavaScript). Every channel rule is checked
**exactly** (symbolically) along two independent routes:

* **table route** – expand the product term by term, e.g. (−a₀T̂ + a·**K̂**)(b₀T̂ + b·**K̂**),
  and read each basis product from the BQ-Pauli multiplication table, as the web pages do
  (16 products for AᵀB, 28 for D(AᵀB));
* **matrix route** – form the explicit 2×2 complex matrix product and decompose it on the
  eight basis elements 1̂, T̂, Î, Ĵ, K̂, σ̂_I, σ̂_J, σ̂_K.

`bq_pauli_table.py` closes the loop by checking the table itself against the matrices.

The scripts follow the four themes of the models.

### 1 · Algebra only

The basis and its 64 products.

| Script | Reproduces | Example |
|---|---|---|
| `bq_core.py` | basis, multiplication table, four-vectors, both routes and the channel rules | (imported) |
| `bq_pauli_table.py` | the 8×8 multiplication table, checked against the paper | `python bq_pauli_table.py sI sJ` |

### 2 · Vectors and their multiplication

Four-vectors on the space-time basis and their products.

| Script | Reproduces | Example |
|---|---|---|
| `bq_bilinear.py` | the channels of AᵀB, both routes | `python bq_bilinear.py R P --table` |
| `bq_triple.py` | the four channels of D(AᵀB), both routes | `python bq_triple.py J D A --c1zero --table` |
| `bq_quadratics.py` | FF and FᵀF of a field F = f1̂ + k·**K̂** + s·σ̂ (EM, fluid, RᵀP, JᵀA, UᵀG, general): rule, table and matrix routes, and the paper's BB, BᵀB, HᵀH | `python bq_quadratics.py B` |

### 3 · Lorentz transformations: a global rapidity

One constant rapidity ψ between two inertial frames.

| Script | Reproduces | Example |
|---|---|---|
| `bq_lorentz.py` | A^L = U⁻¹AU⁻¹ with U = e^{ψσ̂_I/2}: coordinates, inverse, time adjoints, invariance, covariance, light cone; step 5 and the Lemma by table and matrices | `python bq_lorentz.py R` |
| `bq_bilinear_lorentz.py` | C^L = (A^L)ᵀB^L = UCU⁻¹ for all 169 pairs: matrix and rule routes, γ-formulas for the channels, inverse, time adjoint, invariants, Minkowski's complex form; UX̂U⁻¹ for the eight basis elements by table and matrices | `python bq_bilinear_lorentz.py D A` |
| `bq_triple_lorentz.py` | E^L = D^L(A^L)ᵀB^L = U⁻¹EU⁻¹ for all 2197 triples: γ-formulas on the minquat and the pauliquat sector, inverse, invariants; U⁻¹X̂U⁻¹ for the eight basis elements (the LT lemma for both bases); the homogeneous Maxwell equations vanish identically | `python bq_triple_lorentz.py J D A` |

### 4 · Gravity: a local rapidity

A rapidity field χ(r, t) feeding a local rotor.

| Script | Reproduces | Example |
|---|---|---|
| `bq_gravity_rotor.py` | the gravity rotor G = e^{χσ̂/2} fed by ψ_N, ψ_H, χ_φ: dR^G = G⁻¹dRG⁻¹ (nested for two boosts), the flat exact quadratic (and flat consistent truncation), the Galilean (PG) coframe and the term it drops, g_μν, Schwarzschild, de Sitter/FLRW, Schwarzschild–de Sitter and rigid-rotation comparisons in both time orientations, and the Einstein tensor for general u(r), Ω(r) | `python bq_gravity_rotor.py` |
| `bq_cl_killing.py` | CL in the Galilean-coframe metric: river observers geodesic for any shift, Killing energy E = c², γ_rs = −u·u_s/c² = c/√(−g_tt) (= rotor Γ = cosh χ_r cosh χ_φ + O(χ⁴)), CL ⇔ ∂_r g_tt = 0 ⇔ v_r² + v_φ² = K where v_r ≠ 0, the closures K = 3GM/R and (3/2)v_r(R)², quasi-stationary H (adiabatic parameter, both evolution prescriptions, the Ḣ shift-rate residual), and G_n̂n̂ = −r²Ω′²/(4c²) < 0 for the CL disk at H = 0 | `python bq_cl_killing.py` |

## Install and run

```
git clone https://github.com/haas2u-ai/bq_algebra
cd bq_algebra/py_sources
pip install -r requirements.txt        # SymPy only

# 1 · algebra only
python bq_pauli_table.py               # 64 products, all checked against the paper

# 2 · vectors and their multiplication
python bq_bilinear.py R P              # one bilinear, both routes checked exactly
python bq_bilinear.py R P --table      # the same, with the 16 table products printed
python bq_bilinear.py --all            # all 169 pairs (a few seconds)
python bq_triple.py J D A --c1zero --table   # Lorentz force, products printed
python bq_triple.py --examples         # Maxwell, Lorentz force, fluid products
python bq_triple.py --all              # all 2197 triples (about 6 minutes)
python bq_quadratics.py B              # BB and BᵀB in the Lorenz gauge, against the paper
python bq_quadratics.py H              # fluid field HᵀH
python bq_quadratics.py --all          # all fields, with and without the 1̂ channel

# 3 · global rapidity: Lorentz transformations
python bq_lorentz.py --basis           # rotor, step 5 with U⁻¹ = C1̂ − Sσ̂_I, and the Lemma
python bq_lorentz.py R                 # all boost checks for R
python bq_lorentz.py --all             # every vector (a few seconds)
python bq_bilinear_lorentz.py D A      # boost of the EM field B = ∂ᵀA, all checks
python bq_bilinear_lorentz.py --basis  # U X U⁻¹ for the basis, U = ch 1̂ + sh σ̂_I, and Minkowski's Λ_C
python bq_bilinear_lorentz.py --all    # all 169 pairs (about a minute)
python bq_triple_lorentz.py D D A      # ∂B = μ₀J: covariance of the Maxwell equations
python bq_triple_lorentz.py J D A      # JB = F: covariance of the Lorentz force
python bq_triple_lorentz.py --basis    # U⁻¹X̂U⁻¹ for all eight basis elements, by table and matrices
python bq_triple_lorentz.py --maxwell  # pauliquat sector of ∂(∂ᵀA) vanishes identically
python bq_triple_lorentz.py --all      # proof for all 2197 triples via the 8 general triples (30 s)
python bq_triple_lorentz.py --every    # each of the 2197 triples separately (over an hour)

# 4 · local rapidity: gravity
python bq_gravity_rotor.py             # gravity rotor: rapidity -> dR^G -> Galilean coframe -> g_μν -> G_μν (about 20 s)
python bq_cl_killing.py                # CL as an equi-redshift flow (a few seconds)
```

Four-vectors (`python bq_bilinear.py --list`): `R` position (ct, r), `dR` displacement,
`V` coordinate velocity, `U` proper velocity (γc, γv), `P` energy-momentum (E/c, p),
`D` four-gradient (−(1/c)∂ₜ, ∇), `A` four-potential (φ/c, A), `J` four-current (cρ, J),
`G` fluid energy-momentum density (ε/c, g), `F` force-power density, and general
vectors `a`, `b`, `d`.

## Conventions

* Coordinates are real; the imaginary unit sits in T̂ = i1̂, and K̂_i = T̂σ̂_i.
* Aᵀ = −a₀T̂ + a·K̂ is the BQ time-reversal operation, not the matrix transpose.
* In every product the left factor stands to the left, so a four-gradient acts on
  everything to its right. When the right-most factor is the four-gradient, the
  product is evaluated on a test field ψ(t, x, y, z).
* `--table` prints every term-by-term product with its table entry, e.g. `(-c*t) T^ . (E/c) T^  ->  T^T^ = -1^`.
* `--c1zero` sets the norm channel of AᵀB to zero (the Lorenz gauge for ∂ᵀA).
* Field quadratics: Fᵀ = f1̂ + k·**K̂** − s·σ̂ (the time adjoint of a bilinear flips only the σ̂ channel).
  `bq_quadratics.py B` uses the Lorenz gauge B₀ = 0; `--full` keeps B₀.
* Lorentz transformation: boost along Î with rapidity ψ, γ = cosh ψ, βγ = sinh ψ;
  U⁻¹ = C1̂ − Sσ̂_I with C = cosh(ψ/2), S = sinh(ψ/2), the unit basis 1̂ kept explicit.
* Bilinear boost: C^L = UCU⁻¹ (a similarity, not U⁻¹CU⁻¹); 1̂, T̂, Î, σ̂_I are invariant, (Ĵ, σ̂_K) and (K̂, σ̂_J) mix.
  The complex form uses 𝒞 = c_σ + i c_K, from K̂_i = T̂σ̂_i = iσ̂_i.
* Triple boost: E^L = U⁻¹EU⁻¹ (vector-like); the minquat (e_T, e_K) and the pauliquat (e_1, e_σ) sectors are both four-vectors.
* Gravity rotor: G = e^{χσ̂/2} with a rapidity field χ(r, t); dR^G = G⁻¹dRG⁻¹, and for a radial plus an azimuthal
  boost the nested G_φ⁻¹(G_r⁻¹dRG_r⁻¹)G_φ⁻¹. With rotor rapidity χ the coframe is θ^r = dr − v dt (paper chart);
  with −χ it is dr + v dt (ingoing river, expanding FLRW).
* γ is kept as a positive field during the check and replaced by 1/√(1 − v²/c²) when printed.

Results are printed in terms of the potentials and fields; for example `bq_triple.py D D A --c1zero`
prints ∂B in terms of φ and A, which becomes −∇·B, (1/c)∇·E, ∇×B − (1/c²)∂ₜE and
(1/c)(∇×E + ∂ₜB) with B = ∇×A and E = −∂ₜA − ∇φ.

The definitions in the BQ papers remain authoritative; these scripts are a reproducibility aid.
E.P.J. de Haas.
