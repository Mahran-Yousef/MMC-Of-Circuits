"""Fig. 12: total MMC of a 4-layer AND/OR circuit as the OR-gate prior is varied.

3 inputs; each layer computes AND(n0,n1), AND(n1,n2), OR(n2,n0) of the previous layer's nodes.
AND priors fixed and uniform; OR prior q_g(x_g = 1) varied. Uniform inputs, gate-by-gate.
At q = 0.5 every prior is uniform and MMC = |C| = 12 bits exactly, for any wiring.
"""

import numpy as np

from mmc import four_layer_and_or, per_gate_priors

from .common import Config, finish, new_figure

ID = "fig12"
TITLE = "Four-layer AND/OR circuit: total MMC vs OR-gate prior"
QS = np.round(np.concatenate([[0.01], np.arange(0.05, 0.951, 0.05), [0.99]]), 2)


def configs():
    return [Config(dict(q_or=float(q)), four_layer_and_or,
                   lambda E, q=q: per_gate_priors(E.C, lambda g: q if E.C.kind[g] == "OR" else 0.5))
            for q in QS]


def plot(rows, save):
    fig, ax = new_figure()
    ax.plot([r["q_or"] for r in rows], [r["mmc"] for r in rows], "o-")
    finish(ax, r"Gate prior: $q_g(x_g = 1)$", "Mismatch Cost", legend=False)
    save(fig, "L1distance2")


def caption(rows):
    best = min(rows, key=lambda r: r["mmc"])
    half = [r for r in rows if abs(r["q_or"] - 0.5) < 1e-9][0]
    return rf"""\caption{{Total mismatch cost of a circuit with 4 layers, each containing two AND gates
and one OR gate, as a function of the prior of the OR gates, for uniformly random inputs
and gate-by-gate execution [Eq.~(26)]. The circuit has three inputs, and each layer
computes $\mathrm{{AND}}(n_0, n_1)$, $\mathrm{{AND}}(n_1, n_2)$ and $\mathrm{{OR}}(n_2, n_0)$ of
the three nodes $(n_0, n_1, n_2)$ of the previous layer. The prior of the AND gates is
fixed and uniform, while the prior of the OR gates, $q_g(x_g = 1)$, is varied from {rows[0]["q_or"]}
to {rows[-1]["q_or"]}. At $q_g(x_g = 1) = 0.5$ all priors are uniform and the total mismatch cost
equals the circuit size, {half["mmc"]:.0f} bits. The minimum, {best["mmc"]:.1f} bits, occurs at
$q_g(x_g = 1) = {best["q_or"]}$, the probability with which every OR gate in this circuit outputs 1;
the cost rises steeply as the OR prior moves away from it ({rows[0]["mmc"]:.1f} bits at {rows[0]["q_or"]} and
{rows[-1]["mmc"]:.1f} bits at {rows[-1]["q_or"]}). Mismatch cost in bits.}}"""
