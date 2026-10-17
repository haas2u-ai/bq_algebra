# BQ-Pauli algebra: Python reproduction scripts

Independent Python/SymPy implementation of the three interactive models in this
repository (the web pages run their own JavaScript). Every product is formed as an
explicit 2×2 complex matrix product, decomposed on the eight basis elements
1̂, T̂, Î, Ĵ, K̂, σ̂_I, σ̂_J, σ̂_K, and compared **exactly** (symbolically) with the
stated channel rule.

| Script | Reproduces | Example |
|---|---|---|
| `bq_pauli_table.py` | the 8×8 multiplication table, checked against the paper | `python bq_pauli_table.py sI sJ` |
| `bq_bilinear.py` | the channels of AᵀB | `python bq_bilinear.py R P` |
| `bq_triple.py` | the four channels of D(AᵀB) | `python bq_triple.py D D A --c1zero` |
| `bq_core.py` | shared basis, four-vectors, matrix products and channel rules | (imported) |

## Install and run

```
git clone https://github.com/haas2u-ai/bq_algebra
cd bq_algebra/py_sources
pip install -r requirements.txt        # SymPy only
python bq_pauli_table.py               # 64 products, all checked against the paper
python bq_bilinear.py R P              # one bilinear, with its exact check
python bq_bilinear.py --all            # all 169 pairs (a few seconds)
python bq_triple.py --examples         # Maxwell, Lorentz force, fluid products
python bq_triple.py --all              # all 2197 triples (about 6 minutes)
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
* `--c1zero` sets the norm channel of AᵀB to zero (the Lorenz gauge for ∂ᵀA).
* γ is kept as a positive field during the check and replaced by 1/√(1 − v²/c²) when printed.

Results are printed in terms of the potentials and fields; for example `bq_triple.py D D A --c1zero`
prints ∂B in terms of φ and A, which becomes −∇·B, (1/c)∇·E, ∇×B − (1/c²)∂ₜE and
(1/c)(∇×E + ∂ₜB) with B = ∇×A and E = −∂ₜA − ∇φ.

The definitions in the BQ papers remain authoritative; these scripts are a reproducibility aid.
E.P.J. de Haas.
