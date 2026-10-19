"""Fig. 9: AND binary trees under three gate priors, uniform inputs, gate-by-gate.

  Agree     q_{g|pa} = p_g: agrees with the state the gate holds just before it updates
            (its previous-run output, independent of its new parents) -> MMC = sum_g S(X_g)
  Uniform   q_{g|pa} = 1/2 -> exactly one bit per gate
  Disagree  probability 0.97 on the output opposite to an ideal AND gate given its inputs
"""

from mmc import binary_tree, logic_priors, matched_priors, uniform_priors

from .common import Config, finish, new_figure, select

ID = "fig09"
TITLE = "AND binary trees under agreeing, uniform and disagreeing priors"
N_INPUTS = (2, 4, 8, 16)
P_WRONG = 0.97
PRIORS = {
    "agree": lambda E: matched_priors(E),
    "uniform": lambda E: uniform_priors(E.C),
    "disagree": lambda E: logic_priors(E.C, E.C.gate_table, p_other=P_WRONG),
}


def configs():
    return [Config(dict(n_inputs=n, prior=p), lambda n=n: binary_tree(n, "AND"), PRIORS[p])
            for p in PRIORS for n in N_INPUTS]


def plot(rows, save):
    fig, ax = new_figure()
    for prior, style, label in (("agree", "o-", "Agree"), ("uniform", "s-", "Uniform"), ("disagree", "^-", "Disagree")):
        R = select(rows, prior=prior)
        ax.plot([r["gates"] for r in R], [r["mmc"] for r in R], style, label=label)
    finish(ax, "Circuit Size", "Mismatch Cost", ncol=2)
    save(fig, "mix_binary_tree5")


def caption(rows):
    big = max(r["gates"] for r in rows)
    v = {p: select(rows, prior=p, gates=big)[0]["mmc"] for p in PRIORS}
    return rf"""\caption{{Mismatch cost for a family of binary-tree AND circuits under different gate
priors, for a uniform input distribution and gate-by-gate execution [Eq.~(26)].
Because the inputs are re-sampled on every run, the value a gate holds just before it
updates is its output from the previous run, which is independent of its new inputs
[Eq.~(21)]; the mismatch cost of the update measures how well the gate's prior matches
this pre-update state. The priors are: (blue) a prior that agrees with the pre-update
state, $q_{{g|\mathrm{{pa}}(g)}} = p_g$, for which the total mismatch cost reduces to
$\sum_g S(X_g)$; (orange) a uniform prior, for which every gate contributes exactly one
bit; (green) a prior that assigns probability {P_WRONG} to the output opposite to that of an
ideal AND gate given its inputs (`disagrees'). The agreeing prior gives the lowest
mismatch cost ({v["agree"]:.1f} bits for {big} gates, compared with {v["uniform"]:.1f} bits for the uniform prior and
{v["disagree"]:.1f} bits for the disagreeing prior). Mismatch cost in bits.}}"""
