"""Fig. 6: AND binary tree saturating the Theorem 1 bound.

Uniform priors for every gate and input. All-ones inputs attain the bound |C| + n of Eq. (C13)
(q_star = q_i = 1/2); i.i.d. uniform inputs give exactly |C| bits.
"""

import numpy as np

from mmc import binary_tree, uniform_priors

from .common import Config, finish, new_figure, select

ID = "fig06"
TITLE = "AND binary tree: all-ones input saturates the upper bound"
N_INPUTS = (2, 4, 8, 16)


def _all_ones(C):
    p = np.zeros(2 ** len(C.inputs))
    p[-1] = 1.0                      # the last input string is all ones
    return p


def configs():
    out = []
    for n in N_INPUTS:
        build = lambda n=n: binary_tree(n, "AND")
        out.append(Config(dict(n_inputs=n, input_dist="all-ones"), build, lambda E: uniform_priors(E.C), _all_ones))
        out.append(Config(dict(n_inputs=n, input_dist="uniform"), build, lambda E: uniform_priors(E.C)))
    return out


def plot(rows, save):
    fig, ax = new_figure()
    ones, unif = select(rows, input_dist="all-ones"), select(rows, input_dist="uniform")
    x = [r["gates"] for r in ones]
    ax.plot(x, [r["mmc"] for r in ones], "o-", label="All-Ones — Simulation")
    ax.plot(x, [r["gates"] + r["inputs"] for r in ones], "s--", label="All-Ones — Theory (exact)")
    ax.plot([r["gates"] for r in unif], [r["mmc"] for r in unif], "o-", label="IID Uniform — Simulation")
    finish(ax, "Number of Non-Input Gates", "Mismatch Cost (bits)")
    save(fig, "saturating_max4")


def caption(rows):
    return r"""\caption{Saturating the upper bound on MMC for the AND binary tree circuit family, with
uniform priors for all gates and inputs. The solid blue line is the total mismatch
cost from Eq.~(26) for an input distribution concentrated on the all-ones string. It
coincides with the theoretical upper bound $|C_n| + n$ of Eq.~(C13) (orange dashed):
every gate contributes $\log 2$, and re-initializing the inputs contributes
$D(p_{\mathrm{in}}\|q_{\mathrm{in}}) = n \log 2$. For an i.i.d.\ uniform input
distribution (green), $D(p_{\mathrm{in}}\|q_{\mathrm{in}}) = 0$ and the total mismatch
cost is $|C_n|$ bits, below the bound. With uniform priors, each deterministic gate
contributes exactly one bit for any input distribution, so the two curves differ only
through the input re-initialization term.}"""
