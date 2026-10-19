"""Fig. 8: homogeneous vs heterogeneous priors on a binary tree with one OR gate per level.

Uniform inputs, gate-by-gate. Homogeneous: every gate uniform -> MMC = |C| exactly.
Heterogeneous: OR gates have q_OR(1) = 0.01, AND gates uniform -> weighted gate count (Theorem 2).
"""

from mmc import and_tree_one_or_per_level, per_gate_priors, uniform_priors

from .common import Config, finish, new_figure, select

ID = "fig08"
TITLE = "Homogeneous vs heterogeneous priors (AND tree with one OR per level)"
N_INPUTS = (2, 4, 8, 16)
Q_OR = 0.01


def _heterogeneous(E):
    C = E.C
    return per_gate_priors(C, lambda g: Q_OR if C.kind[g] == "OR" else 0.5)


def configs():
    out = []
    for n in N_INPUTS:
        build = lambda n=n: and_tree_one_or_per_level(n)
        out.append(Config(dict(n_inputs=n, priors="heterogeneous"), build, _heterogeneous))
        out.append(Config(dict(n_inputs=n, priors="homogeneous"), build, lambda E: uniform_priors(E.C)))
    return out


def plot(rows, save):
    fig, ax = new_figure()
    het, hom = select(rows, priors="heterogeneous"), select(rows, priors="homogeneous")
    ax.plot([r["gates"] for r in het], [r["mmc"] for r in het], "o-", label="TOTAL — OR Non-Uniform Prior")
    ax.plot([r["gates"] for r in hom], [r["mmc"] for r in hom], "s-", label="TOTAL — OR Uniform Prior")
    finish(ax, "Number of Non-Input Gates", "Mismatch Cost")
    save(fig, "mix_binary_tree_3")


def caption(rows):
    het = select(rows, priors="heterogeneous")
    vals = ", ".join(f"{r['mmc']:.1f}" for r in het[:-1]) + f" and {het[-1]['mmc']:.1f}"
    sizes = ", ".join(str(r["gates"]) for r in het[:-1]) + f" and {het[-1]['gates']}"
    return rf"""\caption{{Mismatch cost scaling for a circuit family with homogeneous (orange) versus
heterogeneous (blue) priors, for uniformly random inputs and gate-by-gate execution
[Eq.~(26)]. The circuit family consists of binary trees of AND and OR gates in which each
level has one OR gate and the remaining gates are AND gates, so the number of OR gates
equals the depth, $\log_2(|C_n| + 1)$. With identical uniform priors for all gates
(orange), every gate contributes exactly one bit and the mismatch cost equals the circuit
size, $\mathcal{{MC}}(C_n) = |C_n|$. When the OR gates instead have prior
$q_{{\mathrm{{OR}}}}(1) = {Q_OR}$ while the AND gates keep the uniform prior (blue), each OR
gate contributes about 4 additional bits, giving {vals} bits for
{sizes} gates. The mismatch cost then grows as a weighted gate count
(Theorem~2) rather than linearly with circuit size. Mismatch cost in bits.}}"""
