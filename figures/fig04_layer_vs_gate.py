"""Fig. 4: ripple-carry adder, gate-by-gate vs layer-by-layer execution.

Uniform inputs; each schedule uses the priors that minimise its own mismatch cost:
  gate-by-gate   q_{g|pa} = p_g (per gate)                         -> Eq. (26) = sum_g S(X_g)
  layer-by-layer joint q_{l|pa(l)} = p_l (per layer, correlated)   -> Eq. (33) = sum_l S(X_l)
The gap is sum_l TC(X_l), the total correlation within each layer. With product layer priors
(e.g. uniform priors) the two schedules cost exactly the same.
"""

from mmc import matched_priors, ripple_carry_adder, with_joint_layer_priors

from .common import Config, finish, new_figure, select

ID = "fig04"
TITLE = "Ripple-carry adder: gate-by-gate vs layer-by-layer (matched / joint layer priors)"
NS = range(2, 6)


def configs():
    out = []
    for n in NS:
        build = lambda n=n: ripple_carry_adder(n)
        out.append(Config(dict(n=n, schedule="gate"), build, lambda E: matched_priors(E), mode="gate"))
        out.append(Config(dict(n=n, schedule="layer"), build,
                          lambda E: with_joint_layer_priors(E, matched_priors(E)), mode="layer"))
    return out


def plot(rows, save):
    fig, ax = new_figure()
    g, l = select(rows, schedule="gate"), select(rows, schedule="layer")
    ax.plot([r["gates"] for r in g], [r["mmc"] for r in g], "o-", label="Ripple-Carry (Gate-by-gate)")
    ax.plot([r["gates"] for r in l], [r["mmc"] for r in l], "o-", label="Ripple-Carry Layer-by-layer")
    finish(ax, "Circuit Size", "Mismatch Cost")
    save(fig, "ripple_layer_vs_gate")


def caption(rows):
    gaps = [g["mmc"] - l["mmc"] for g, l in zip(select(rows, schedule="gate"), select(rows, schedule="layer"))]
    return rf"""\caption{{Comparison of layer-by-layer and gate-by-gate mismatch costs for the
ripple-carry adder circuit family ($n = {NS[0]},\dots,{NS[-1]}$ bits per operand, circuit size
$|C_n| = 5n$), with uniformly random inputs. Each implementation uses the priors that
minimize its own mismatch cost for this input distribution. For gate-by-gate execution
[Eq.~(26)], every gate is a separate subsystem process with prior
$q_{{g|\mathrm{{pa}}(g)}} = p_g$. For layer-by-layer execution [Eq.~(33)], every layer is a
single subsystem process whose joint prior $q_{{l|\mathrm{{pa}}(l)}} = p_l$ matches the
correlated distribution of the gates in that layer. The layer-by-layer mismatch cost is
lower by $\sum_l \mathrm{{TC}}(X_l)$, the total correlation among the gates within each
layer (between {min(gaps):.2f} and {max(gaps):.2f} bits here). If instead each layer's prior is the product
of its gates' priors, as for uniform priors, the two costs coincide.
Mismatch cost in bits.}}"""
