# BQ algebra at the Dirac level: Python reproduction scripts

Independent Python/SymPy implementation of the Dirac-level models in `../models/`
(the web pages run their own JavaScript). The 4×4 basis of the Weyl representation is
built from the BQ-Pauli blocks, and every product is checked **exactly** along two routes:

* **block route**: the four 2×2 blocks of a product, each a sum of BQ-Pauli products;
* **4×4 route**: the ordinary 4×4 complex matrix product, matched against the sixteen
  ± basis matrices.

The scripts are self-contained and do not import the BQ-Pauli scripts in `../../py_sources/`.

### 1 · The basis

| Script | Reproduces | Example |
|---|---|---|
| `bq_dirac_table.py` | the 16×16 BQ-Dirac multiplication table: β₀ = [[0,T̂],[T̂,0]], β_i = [[0,K̂_i],[−K̂_i,0]], the other twelve elements computed (α_i, Σ_i, χ, β̃_μ); the Pauli block content of all sixteen; Clifford relations; χ² = −𝟙 and its (anti)commutation; trace orthogonality; all 256 products by the block route and the 4×4 route, each ± one basis element | `python bq_dirac_table.py bI bJ` |

## Install and run

```
git clone https://github.com/haas2u-ai/bq_algebra
cd bq_algebra/dirac/py_sources
pip install -r requirements.txt        # SymPy only

# 1 · the basis
python bq_dirac_table.py               # all checks and the 16 x 16 table (a few seconds)
python bq_dirac_table.py bI bJ         # one product: Pauli blocks, block terms, 4x4 matrices
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
