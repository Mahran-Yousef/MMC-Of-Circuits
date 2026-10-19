"""Fig. 10: a mixed binary-tree family whose MMC is strongly nonlinear in circuit size.

Depth-d tree uses one gate type, chosen in increasing order of per-gate cost under a shared prior
that strongly favours AND behaviour: q_pa uniform, q_{g|pa}(x_g | x_pa) = 1 - eps if x_g = x AND y,
eps otherwise, eps = 1e-12. Uniform inputs, gate-by-gate.
    depth 1: CONST gate that always outputs 0, depth 2: AND, depth 3: NOR, depth 4: NAND
"""

from mmc import binary_tree, logic_priors
from mmc.priors import AND_TABLE

from .common import Config, finish, new_figure

ID = "fig10"
TITLE = "Mixed binary-tree family (CONST0, AND, NOR, NAND) under an AND-favouring prior"
EPS = 1e-12
TREES = ((1, "ZERO"), (2, "AND"), (3, "NOR"), (4, "NAND"))


def _tree(depth, kind):
    C = binary_tree(2 ** depth, "AND")
    for g in C.gates:
        C.kind[g] = kind
    return C


def configs():
    return [Config(dict(depth=d, gate_type=k), lambda d=d, k=k: _tree(d, k),
                   lambda E: logic_priors(E.C, lambda g: AND_TABLE, p_other=EPS))
            for d, k in TREES]


def plot(rows, save):
    fig, ax = new_figure()
    ax.plot([r["gates"] for r in rows], [r["mmc"] for r in rows], "o-", ms=10)
    finish(ax, "Circuit Size (internal gates)", "Mismatch Cost", legend=False)
    save(fig, "mixed_tree_nonlinear")


def caption(rows):
    # The authors' caption, with only the gate order changed (and the subscript braces fixed).
    return r"""\caption{Mismatch cost of a mixed binary tree circuit family. This figure illustrates that
the linear cost--size relation need not hold for circuit families in the finite input regime.
The circuit family is constructed as follows: starting with a depth-1 tree which consists of
a CONST gate that always outputs zero, the depth-2 tree is made of AND gates, depth-3 tree is
made from NOR gates and lastly, depth 4 tree is a tree of NAND gates. All gates are evaluated
under the same prior: inputs are assumed independent and uniformly distributed, so that
$q_{\mathrm{pa}(g)}(\boldsymbol{x}_{\mathrm{pa}(g)}) = \tfrac{1}{4}$ for all
$\boldsymbol{x}_{\mathrm{pa}(g)} \in \{0,1\}^2$, and
$q_{g|\mathrm{pa}(g)}(\boldsymbol{x}_g \mid \boldsymbol{x}_{\mathrm{pa}(g)}) =
\begin{cases} 1-\varepsilon & \text{if } \boldsymbol{x}_g = x \land y, \\ \varepsilon & \text{otherwise}, \end{cases}$
with $\varepsilon = 10^{-12}$. This means that the prior strongly favors AND behavior, and
deviations from AND incur mismatch cost. Jointly, $q_{\mathrm{pa}(g)}$ and $q_{g|\mathrm{pa}(g)}$
determine the prior
$q_{g,\mathrm{pa}(g)}(\boldsymbol{x}_g, \boldsymbol{x}_{\mathrm{pa}(g)}) =
q_{g|\mathrm{pa}(g)}(\boldsymbol{x}_g \mid \boldsymbol{x}_{\mathrm{pa}(g)})\,
q_{\mathrm{pa}(g)}(\boldsymbol{x}_{\mathrm{pa}(g)})$.}"""
