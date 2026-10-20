# BQ algebra at the Dirac level: Python reproduction scripts

Independent Python/SymPy implementation of the Dirac-level models in `../models/`
(the web pages run their own JavaScript). The 4×4 basis of the Weyl representation is
built from the BQ-Pauli blocks, and every product is checked **exactly** along two routes:

* **block route**: the four 2×2 blocks of a product, each a sum of BQ-Pauli products;
* **4×4 route**: the ordinary 4×4 complex matrix product, matched against the sixteen
  ± basis matrices.

The scripts are self-contained and do not import the BQ-Pauli scripts in `../../py_sources/`;
`bq_dirac_bilinear.py` imports the basis from `bq_dirac_table.py` in this folder.

### 1 · The basis

| Script | Reproduces | Example |
|---|---|---|
| `bq_dirac_table.py` | the 16×16 BQ-Dirac multiplication table: β₀ = [[0,T̂],[T̂,0]], β_i = [[0,K̂_i],[−K̂_i,0]], the other twelve elements computed (α_i, Σ_i, χ, β̃_μ); the Pauli block content of all sixteen; Clifford relations; χ² = −𝟙 and its (anti)commutation; trace orthogonality; all 256 products by the block route and the 4×4 route, each ± one basis element | `python bq_dirac_table.py bI bJ` |
| `bq_dirac_representations.py` | the same basis in the Dirac representation, β₀ = [[T̂,0],[0,−T̂]] with the β_i unchanged and the other twelve built as products; the matrix S = (1/√2)[[1̂,1̂],[−1̂,1̂]] (real orthogonal) with X_D = S X_W S⁻¹ for all sixteen; the identical 16×16 table; the Pauli blocks of both bases; S = (𝟙 − β̃₀)/√2 = e^(−(π/4)β̃₀); the vector [[a₀T̂, a·K̂], [−a·K̂, −a₀T̂]] and the boost [[c 1̂, s σ̂_I], [s σ̂_I, c 1̂]] in the Dirac representation | `python bq_dirac_representations.py` |

### 2 · Vectors and their products

| Script | Reproduces | Example |
|---|---|---|
| `bq_dirac_bilinear.py` | the bilinear A/B/ of two Dirac-level four-vectors A/ = A_μβ^μ: the channel rule −(a₀b₀ − a·b)𝟙 + (a₀b − ab₀)·α + (a×b)·Σ with no odd or χ channels; the block form A/ = [[0,A],[−Aᵀ,0]] and A/B/ = [[−ABᵀ,0],[0,−AᵀB]] (Pauli correspondence 1̂ → −𝟙, K̂ → Σ, σ̂ → α); all 144 pairs of named vectors with the operator order kept; ∂/A/, ∂/G/, U/U/, ∂/R/ and ∂/∂/ as in the papers | `python bq_dirac_bilinear.py` |
| `bq_dirac_triple.py` | the triple product D/(A/B/): the rule (d₀s − d·c_α)β₀ + (ds − d₀c_α − d×c_Σ)·β − (d·c_Σ)β̃₀ + (d×c_α − d₀c_Σ)·β̃ with only odd channels; the four-gradient on a general even field; the block form [[0, −D(AᵀB)], [DᵀABᵀ, 0]] (Pauli correspondence [E/]_β₀ = −e_T, [E/]_β = −e_K, [E/]_β̃₀ = e₁, [E/]_β̃ = e_σ); Maxwell ∂/B/ by grade and the Lorentz force J/B/ = −F/_L + pseudovector part, as in the Weyl-level EM paper | `python bq_dirac_triple.py` |
| `bq_dirac_quadratics.py` | the quadratics of a field F/ = f1 + p·α + q·Σ: the even product rule and the square (f² + p² − q²)1 + 2fp·α + 2fq·Σ − 2(p·q)χ; the sandwich F/a/F/ (vector, plus a pseudovector part when f ≠ 0); the blocks F/ = diag(−Fᵀ, −F), F/F/ = diag(FᵀFᵀ, FF), F/a/F/ = [[0, FᵀAF], [−FAᵀFᵀ, 0]]; the EM invariants (eq. 124), energy–Poynting (eq. 150) and Maxwell stress (eqs. 153–154); the fluid eqs. 296, 316 and 318; a boost Λ = diag(U⁻¹, U): invariant square, covariant map, frame-resolved β₀ | `python bq_dirac_quadratics.py` |

### 3 · Lorentz transformations

| Script | Reproduces | Example |
|---|---|---|
| `bq_dirac_lorentz.py` | the boost of a Dirac-level vector by one similarity, a/ᴸ = Λa/Λ⁻¹: diag(X, Y) acting on a/ = [[0,A],[−Aᵀ,0]] gives [[0, XAY⁻¹], [−YAᵀX⁻¹, 0]], so X = U⁻¹, Y = U carries both Pauli rules U⁻¹AU⁻¹ and (Aᴸ)ᵀ = UAᵀU (up to a common factor that cancels); Λ = diag(U⁻¹, U) = e^(−ψα_I/2) = cosh(ψ/2)𝟙 − sinh(ψ/2)α_I (Weyl EM eq. 163); the γ-formulas, the invariant a/a/; the basis route Λβ_μΛ⁻¹ = Λ²β_μ and the Dirac lemma β_νΛ^ν_μ = Λβ_μΛ⁻¹, proved block by block from the Pauli lemma (upper block) and its time adjoint (lower block); boosts along any n with the Hermitian U_n; rotations e^(−θm·Σ/2) = diag(r, r); Λ = diag(M, (M†)⁻¹); collinear rapidity addition and the Thomas–Wigner rotation of two perpendicular boosts, tan(θ_W/2) = tanh(ψ₁/2) tanh(ψ₂/2) | `python bq_dirac_lorentz.py` |
| `bq_dirac_bilinear_lorentz.py` | the boost of the bilinear C/ = A/B/: (ΛA/Λ⁻¹)(ΛB/Λ⁻¹) = ΛC/Λ⁻¹ = diag(−U⁻¹(ABᵀ)U, −U(AᵀB)U⁻¹), whose lower block is the Pauli rule UCU⁻¹; 𝟙, α_I, Σ_I, χ commute with α_I and stay, α_J, α_K, Σ_J, Σ_K anticommute and pick up Λ²; the channel rule c′_α,J = γ(c_α,J − βc_Σ,K), c′_Σ,K = γ(c_Σ,K − βc_α,J), c′_α,K = γ(c_α,K + βc_Σ,J), c′_Σ,J = γ(c_Σ,J + βc_α,K); invariants s, c_α² − c_Σ², c_α·c_Σ; the complex form 𝒞 = c_α − χc_Σ; the Pauli correspondence c_α = c_σ, c_Σ = c_K | `python bq_dirac_bilinear_lorentz.py` |
| `bq_dirac_triple_lorentz.py` | the covariance of the triple product E/ = D/(A/B/): E/^L = ΛE/Λ⁻¹ with the upper block −U⁻¹(DAᵀB)U⁻¹, the Pauli rule E^L = U⁻¹EU⁻¹; Λβ̃_μΛ⁻¹ = β̃_νΛ^ν_μ because χ commutes with Λ, so the vector (e₀, e) and pseudovector (r₀, r) sectors are both four-vectors; the invariants e₀² − e², r₀² − r² and e₀r₀ − e·r (the χ channel of E/_vE/_p); the Pauli sector correspondence; the four-gradient on a general field, ∂/′C/′(x′) = Λ(∂/C/)Λ⁻¹, so Maxwell and the fluid equations hold in every frame | `python bq_dirac_triple_lorentz.py` |
| `bq_dirac_lorentz_representations.py` | the boost in the Weyl and the Dirac representation: Λ_D = SΛ_WS⁻¹ = ½[[U⁻¹+U, U−U⁻¹],[U−U⁻¹, U⁻¹+U]] = [[c 1̂, s σ̂_I],[s σ̂_I, c 1̂]] = c𝟙 − sα_I^D; the commuting square Λ_D a/_D Λ_D⁻¹ = S(Λ_W a/_W Λ_W⁻¹)S⁻¹; the same boosted coordinates in both; the Dirac blocks [[a₀′T̂, a′·K̂],[−a′·K̂, −a₀′T̂]]; rotations unchanged; spinors Ψ^L = ΛΨ and Λ²; the exponential form Λ = e^((ψ/2)σ_I) in both, with σ_I^W = diag(−σ̂_I, σ̂_I), σ_I^D = Sσ_I^WS⁻¹ = [[0, σ̂_I],[σ̂_I, 0]] and (σ_I)² = 𝟙; the sign of BQ Dirac Gravity eq. 107 reported | `python bq_dirac_lorentz_representations.py` |

## Install and run

```
git clone https://github.com/haas2u-ai/bq_algebra
cd bq_algebra/dirac/py_sources
pip install -r requirements.txt        # SymPy only

# 1 · the basis
python bq_dirac_table.py               # all checks and the 16 x 16 table (a few seconds)
python bq_dirac_table.py bI bJ         # one product: Pauli blocks, block terms, 4x4 matrices
python bq_dirac_representations.py     # the Dirac representation and the Weyl-to-Dirac matrix S (seconds)

# 2 · vectors and their products
python bq_dirac_bilinear.py            # channel rule, Pauli block form, 144 pairs, paper results (about a minute)
python bq_dirac_triple.py              # triple rule, operator triples, Pauli block form, Maxwell, Lorentz force (seconds)
python bq_dirac_quadratics.py          # field square, sandwich, Pauli blocks, EM and fluid quadratics, boost (seconds)

# 3 · Lorentz transformations
python bq_dirac_lorentz.py             # doubling of U, boosts, Dirac lemma, any direction, rotations, Thomas-Wigner (about 15 s)
python bq_dirac_bilinear_lorentz.py    # boost of the bilinear, channel rule, invariants, complex form, Pauli rule (seconds)
python bq_dirac_triple_lorentz.py      # covariance of the triple, both sectors, invariants, four-gradient on a field (seconds)
python bq_dirac_lorentz_representations.py   # the boost in Weyl and Dirac, Lambda_D = S Lambda_W S^-1 (seconds)
```

Element names: `one b0 bI bJ bK aI aJ aK SI SJ SK bt0 btI btJ btK chi`.

## Conventions

* BQ-Pauli blocks: 1̂, T̂ = i1̂, Î, Ĵ, K̂, σ̂_I, σ̂_J, σ̂_K, with K̂_i = T̂σ̂_i.
* Weyl representation: β₀ = [[0,T̂],[T̂,0]], β_i = [[0,K̂_i],[−K̂_i,0]]; β₀² = −𝟙, β_i² = +𝟙.
* α_i = β₀β_i (boost bivectors), Σ_I = β_Jβ_K cyclic (spin bivectors),
  χ = −β₀β_Iβ_Jβ_K (pseudoscalar), β̃_μ = χβ_μ (pseudovectors).
* No Standard-Model gamma-matrix conventions and no General-Relativity index raising.

The definitions in the BQ papers remain authoritative; these scripts are a reproducibility aid.
E.P.J. de Haas.
