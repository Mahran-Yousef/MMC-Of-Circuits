# Mismatch cost of Boolean circuits: experiments and figures

Code for every numerical figure in *A strictly positive lower bound on the thermodynamic cost of
running a Boolean circuit* (Yadav, Yousef, Wolpert). One command regenerates all figures, the data
behind them, and their LaTeX captions, and can verify every number against an independent
brute-force calculation.

```bash
pip install -r requirements.txt
python run_all.py --verify          # all figures + full verification (~2 min)
python run_all.py --only fig07      # one figure
python -m pytest tests              # quick test suite
```

## Outputs (`output/`)

| Paper figure | File (name used in the LaTeX source) | Module |
|---|---|---|
| Fig. 4 | `ripple_layer_vs_gate` | `figures/fig04_layer_vs_gate.py` |
| Fig. 6 | `saturating_max4` | `figures/fig06_and_tree_bound.py` |
| Fig. 7, option A (one prior per gate type, straight lines) | `ripple_vs_kogg2` (+ `_vs_n`) | `figures/fig07_rca_vs_kogge_stone.py` |
| Fig. 7, option B (matched priors, concave Kogge–Stone) | `ripple_vs_kogg2_matched` (+ `_vs_n`) | same |
| Fig. 8 | `mix_binary_tree_3` | `figures/fig08_heterogeneous.py` |
| Fig. 9 | `mix_binary_tree5` | `figures/fig09_and_tree_priors.py` |
| Fig. 10 | `mixed_tree_nonlinear` | `figures/fig10_mixed_tree.py` |
| Fig. 11 | `L1distance` | `figures/fig11_single_gates.py` |
| Fig. 12 | `L1distance2` | `figures/fig12_four_layer_or_prior.py` |

Each figure is written as `.pdf` and `.png` to `output/figures/`. `output/data/<figure>.csv` holds
every plotted value. `output/captions.tex` holds the captions; any number quoted in a caption is
read from that CSV. `output/verification.txt` holds the verification report (with `--verify`).

## Layout

```
run_all.py            entry point
mmc/                  the library
  circuits.py         circuit DAGs; ripple-carry, Kogge-Stone, trees, the 4-layer AND/OR circuit
  ensemble.py         exact joint distribution p^0 of all nodes (Eq. 13) and entropies
  priors.py           uniform, per-gate, type-basis, matched, logic-based and joint layer priors
  mismatch.py         Eq. (26) gate-by-gate, Eq. (33) layer-by-layer, per-step forms (Eqs. 20, C31)
  bruteforce.py       Eq. (B5) on the full joint distribution (independent check)
  verify.py           verification suite
figures/              one module per figure: configs(), plot(), caption()
tests/                pytest
```

## Model and equations

The circuit is run repeatedly. At the start of a run the inputs are re-sampled from p_in, while
every gate keeps its value from the previous run. The gates are then updated one at a time, in
topological order (gate-by-gate), or one layer at a time (layer-by-layer). Just before a gate
fires, it still holds its previous-run value, which is independent of its freshly updated parents
(Eq. 21).

`mismatch.py` evaluates, exactly, by enumerating every input string:

```
Eq. (26)  MC = D(p_in||q_in) + sum_g I(X_g; X_pa(g))  +  sum_g [ D(p_g p_pa || q_{g,pa}) - D(p_pa || q_pa) ]
                               (after the update)          (before the update; = sum_pa p(pa) D(p_g || q_{g|pa}))
Eq. (33)  the same with each gate g replaced by the joint state of a layer l
```

`bruteforce.py` does not use these formulas. It evolves the joint distribution over every node
configuration through one run and evaluates MC_t = D(p^t || q) - D(p^{t+1} || q') at every step
(Eq. B5), with q = q_{g|pa} q_pa q_rest (Eq. B3) and q' = π_g q_pa q_rest (Eq. B4). The arbitrary
factors q_pa, q_rest and q_nin are drawn at random, which also tests App. A's claim that the
result does not depend on them. Brute force is exponential in circuit size, so it is used on
circuits of up to about 22 nodes. Every equation, and every plotted point small enough, agrees
with it to about 1e-14 bits.

All costs are in bits (base-2 logarithms).

## Priors used in each figure

| Figure | Inputs | Priors |
|---|---|---|
| 4 | uniform | gate-by-gate: q_{g\|pa} = p_g per gate; layer-by-layer: joint q_{l\|pa(l)} = p_l per layer |
| 6 | all-ones and uniform | uniform for every gate and input |
| 7A | uniform | one prior per gate type: AND 1/4, OR 3/4, XOR 1/2 |
| 7B | uniform | q_{g\|pa} = p_g per gate (each circuit's minimum MMC; total = Σ_g S(X_g)) |
| 8 | uniform | homogeneous: all uniform; heterogeneous: q_OR(1) = 0.01, AND uniform |
| 9 | uniform | Agree: q_{g\|pa} = p_g; Uniform; Disagree: 0.97 on the output opposite to AND |
| 10 | uniform | q_{g\|pa}(x_g = x AND y) = 1 − 1e-12, else 1e-12; trees of CONST0, AND, NOR, NAND |
| 11 | uniform | single AND/OR/XOR gate, q_g(1) from 0.01 to 0.99 |
| 12 | uniform | AND uniform, OR prior q_g(1) from 0.01 to 0.99 |

**Matched prior.** p_g is the distribution of the gate's output under the input distribution. It
is also the distribution of the value the gate holds when it fires, so this prior zeroes the KL
bracket of Eq. (26) for every gate. The cost then reduces to the mutual-information term,
Σ_g S(X_g). 1e-12 of the uniform distribution is mixed in so every prior has full support (App. A).

**Layer priors.** With the product of its gates' priors, a layer costs exactly what its gates cost
one by one: gates in a layer never read each other. Layer-by-layer execution is strictly cheaper
only with a joint layer prior. With matched priors, the saving is the summed within-layer total
correlation Σ_l [Σ_{g∈l} S(X_g) − S(X_l)].

## Circuits

* **Ripple-carry adder:** n full adders built from 2-input gates (x = a⊕b, g = a·b, s = x⊕c,
  t = x·c, c_out = g+t); carry-in tied to 0; |C| = 5n.
* **Kogge–Stone adder:** p_i = a_i⊕b_i and g_i = a_i·b_i, a Kogge–Stone prefix network computing
  G and P at every node, s_0 = p_0⊕c_0 (c_0 tied to 0) and s_i = p_i⊕G_{i−1};
  |C| = 9, 18, 27, 39, ... for n = 2, 3, 4, 5, ...
* **Fig. 8:** balanced binary tree whose first gate on every level is OR and all others AND.
* **Fig. 12:** 3 inputs; each of 4 layers computes AND(n0,n1), AND(n1,n2), OR(n2,n0) of the
  previous layer's nodes. The paper does not specify this wiring; replace `four_layer_and_or` in
  `mmc/circuits.py` if yours differs.

The adders and the Fig. 12 wiring were chosen to reproduce the circuit sizes of the original figures.
