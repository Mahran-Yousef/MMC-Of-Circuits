"""Fig. 7: ripple-carry vs Kogge-Stone adders computing ADD^n, gate-by-gate, uniform inputs.

Two prior settings:
  "type"     option A: one fixed prior per gate type, shared by both circuits (a basis with
             priors, Sec. IV): q_AND(1)=1/4, q_OR(1)=3/4, q_XOR(1)=1/2. Both families are ~linear in
             circuit size, with different slopes.          -> ripple_vs_kogg2[_vs_n]
  "matched"  option B: every gate's prior equals the distribution of the value it holds when it
             fires, q_{g|pa} = p_g, i.e. the minimum MMC each circuit can reach; MMC = sum_g S(X_g).
             The Kogge-Stone curve is concave.            -> ripple_vs_kogg2_matched[_vs_n]
"""

import numpy as np

from mmc import kogge_stone_adder, matched_priors, ripple_carry_adder, type_basis_priors

from .common import Config, finish, new_figure, select

ID = "fig07"
TITLE = "Ripple-carry vs Kogge-Stone (type-basis priors and matched priors)"
NS = range(2, 10)
FAMILIES = {"Ripple-Carry": ripple_carry_adder, "Kogge-Stone": kogge_stone_adder}
PRIORS = {"type": lambda E: type_basis_priors(E.C), "matched": lambda E: matched_priors(E)}


def configs():
    return [Config(dict(family=f, n=n, prior=p), lambda b=build, n=n: b(n), PRIORS[p])
            for p in PRIORS for f, build in FAMILIES.items() for n in NS]


def plot(rows, save):
    for prior, suffix in (("type", ""), ("matched", "_matched")):
        for key, xlabel, tail in (("gates", "Circuit Size", ""), ("n", "Input Size n (bits per operand)", "_vs_n")):
            fig, ax = new_figure()
            for fam, style in (("Ripple-Carry", "o-"), ("Kogge-Stone", "s-")):
                R = select(rows, family=fam, prior=prior)
                ax.plot([r[key] for r in R], [r["mmc"] for r in R], style, label=fam)
            finish(ax, xlabel, "Mismatch Cost")
            save(fig, f"ripple_vs_kogg2{suffix}{tail}")


def _common_intro():
    return rf"""Mismatch cost of two circuit families implementing $\mathrm{{ADD}}^n$, which takes
two integers in the standard $n$-bit representation and returns their sum in the
standard $(n+1)$-bit representation: the ripple-carry adder ($|C_n| = 5n$) and the
Kogge--Stone parallel-prefix adder ($|C_n| = 9, 18, 27, 39, \dots$), for
$n = {NS[0]},\dots,{NS[-1]}$, uniformly random inputs, and gate-by-gate execution [Eq.~(26)]."""


def caption(rows):
    def at(fam, prior, n):
        return select(rows, family=fam, prior=prior, n=n)[0]
    n_max = NS[-1]
    slope = {f: np.polyfit([r["gates"] for r in select(rows, family=f, prior="type")],
                           [r["mmc"] for r in select(rows, family=f, prior="type")], 1)[0] for f in FAMILIES}
    per_gate = {f: [r["mmc"] / r["gates"] for r in select(rows, family=f, prior="matched")] for f in FAMILIES}
    rca_t, ks_t = at("Ripple-Carry", "type", n_max)["mmc"], at("Kogge-Stone", "type", n_max)["mmc"]
    rca_m, ks_m = at("Ripple-Carry", "matched", n_max)["mmc"], at("Kogge-Stone", "matched", n_max)["mmc"]
    option_a = rf"""% Option A: one prior per gate type (ripple_vs_kogg2.pdf)
\caption{{{_common_intro()} Both
families use the same basis with priors: every gate of a given type has the prior given
by that type's output distribution on independent uniform inputs,
$q_{{\mathrm{{AND}}}}(1) = 1/4$, $q_{{\mathrm{{OR}}}}(1) = 3/4$, $q_{{\mathrm{{XOR}}}}(1) = 1/2$.
Both families scale linearly with circuit size but with different slopes: about
{slope["Ripple-Carry"]:.2f} bits per gate for the ripple-carry adder and {slope["Kogge-Stone"]:.2f} bits per gate for the
Kogge--Stone adder, whose deep prefix levels contain many nearly deterministic gates. At
equal input size, the ripple-carry adder is cheaper (e.g.\ {rca_t:.1f} versus {ks_t:.1f} bits at
$n = {n_max}$) because it has far fewer gates. With uniform priors, every gate would
contribute exactly one bit and both families would lie on the line
$\mathcal{{MC}}(C_n) = |C_n|$. Mismatch cost in bits.}}"""
    option_b = rf"""% Option B: priors matched to each circuit (ripple_vs_kogg2_matched.pdf)
\caption{{{_common_intro()} Each
circuit uses the priors that minimize its own mismatch cost for this input
distribution, $q_{{g|\mathrm{{pa}}(g)}} = p_g$, so that the total mismatch cost equals
$\sum_g S(X_g)$. The mismatch cost per gate rises with $n$ for the ripple-carry adder
(from {per_gate["Ripple-Carry"][0]:.2f} to {per_gate["Ripple-Carry"][-1]:.2f} bits) and falls for the Kogge--Stone adder (from {per_gate["Kogge-Stone"][0]:.2f} to {per_gate["Kogge-Stone"][-1]:.2f} bits),
so the two families separate when plotted against circuit size. The Kogge--Stone curve
is concave because its group-propagate signals over $k$ bits equal 1 with probability
$2^{{-k}}$: gates in deep prefix levels are nearly deterministic and contribute little. At
equal input size, the ripple-carry adder is cheaper (e.g.\ {rca_m:.1f} versus {ks_m:.1f} bits at
$n = {n_max}$) because it has far fewer gates. With uniform priors, every gate would
contribute exactly one bit and both families would lie on the line
$\mathcal{{MC}}(C_n) = |C_n|$. Mismatch cost in bits.}}"""
    companion = r"""% Optional companion (ripple_vs_kogg2_vs_n.pdf or ripple_vs_kogg2_matched_vs_n.pdf)
\caption{The data of Fig.~7 plotted against the input size $n$ (bits per operand). The
ripple-carry adder has the lower mismatch cost for every $n \ge 3$.}"""
    return "\n\n".join([option_a, option_b, companion])
