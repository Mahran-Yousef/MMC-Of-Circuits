"""Fig. 11: mismatch cost of a single AND / OR / XOR gate as a function of its prior.

Uniform inputs, parent-independent gate prior q_g(x_g = 1) varied. Each curve is minimised where
the prior equals the gate's output distribution (1/4 for AND, 3/4 for OR, 1/2 for XOR).
"""

import numpy as np

from mmc import per_gate_priors, single_gate

from .common import Config, finish, new_figure, select

ID = "fig11"
TITLE = "Single gates: mismatch cost vs gate prior"
KINDS = ("AND", "OR", "XOR")
QS = np.round(np.linspace(0.01, 0.99, 50), 4)


def configs():
    return [Config(dict(gate=k, q=float(q)), lambda k=k: single_gate(k),
                   lambda E, q=q: per_gate_priors(E.C, lambda g: q))
            for k in KINDS for q in QS]


def plot(rows, save):
    fig, ax = new_figure()
    for k in KINDS:
        R = select(rows, gate=k)
        ax.plot([r["q"] for r in R], [r["mmc"] for r in R], "o-", label=k)
    finish(ax, r"Gate prior: $q_g(x_g = 1)$", "Mismatch Cost", loc="upper center")
    save(fig, "L1distance")


def caption(rows):
    return r"""\caption{Mismatch cost of individual gates as a function of their prior distributions. The
input distribution to the gate is uniform. The x-axis is the prior probability
$q_g(x_g = 1)$. Each curve is minimized when the prior equals the gate's output distribution
($q_g(x_g = 1) = 1/4$ for AND, $3/4$ for OR and $1/2$ for XOR).}"""
