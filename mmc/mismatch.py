"""Total mismatch cost of one run: Eq. (26) gate-by-gate and Eq. (33) layer-by-layer.

    MC(C, p_in) = D(p_in || q_in)
                + sum_t I_{t+1}(X_{g_t}; X_{pa(g_t)})                                   (post-update)
                + sum_t [ D(p_{g_t} p_{pa(g_t)} || q_{g_t,pa(g_t)}) - D(p_{pa(g_t)} || q_{pa(g_t)}) ]
                                                                                        (pre-update)
The KL bracket equals sum_{x_pa} p(x_pa) D(p_g || q_{g|x_pa}) (Eq. 28 / App. A), which is how it is
evaluated here; it does not depend on q_pa. The layer version replaces g by the layer's joint state.

Each function also returns the per-step decomposition MC_ow = I_0(X_in; X_nin) + D(p_in||q_in)
(Eq. 20) and MC_t = I(X_{g_t}; X_{>t}) + KL bracket (Eqs. B10 and C31), whose sum must equal the
total; `consistent` reports that it does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ensemble import LOG, entropy


@dataclass
class Result:
    total: float          # Eq. (26) or Eq. (33)
    steps: list           # per-step costs, input re-initialisation first
    consistent: bool      # sum(steps) == total


def input_kl(E, priors):
    """D(p_in || q_in) for a product input prior."""
    X = E.X[:, E.C.inputs]
    logq = (X * LOG(priors.q_in) + (1 - X) * LOG(1 - priors.q_in)).sum(axis=1)
    m = E.w > 0
    return float((E.w[m] * (LOG(E.w[m]) - logq[m])).sum())


def _cross_entropy(p_pa, p_x, q_rows):
    """sum_pa p(pa) sum_x p(x) [-log q(x | pa)]; q_rows has 1 row (parent-independent) or len(p_pa) rows."""
    m = p_x > 0
    ce_rows = -(p_x[None, m] * LOG(q_rows[:, m])).sum(axis=1)
    return float(ce_rows[0]) if q_rows.shape[0] == 1 else float((p_pa * ce_rows).sum())


def gate_kl_term(E, priors, g):
    """D(p_g p_pa || q_{g,pa}) - D(p_pa || q_pa) = sum_pa p(pa) D(p_g || q_{g|pa})."""
    pa = E.C.parents[g]
    p_g = E.dist([g])
    return _cross_entropy(E.dist(pa), p_g, priors.q_gate[g]) - entropy(p_g)


def layer_kl_term(E, priors, L):
    """Layer analogue: sum_pa p(pa) D(p_l || q_{l|pa(l)}), p_l the layer's joint pre-update distribution."""
    key = tuple(L)
    if key in priors.q_layer:
        pa = sorted(set().union(*(E.C.parents[g] for g in L)))
        q = priors.q_layer[key]
        p_pa = E.dist(pa) if q.shape[0] > 1 else np.ones(1)
        return _cross_entropy(p_pa, E.dist(L), q) - E.H(L)
    # product of the gates' priors: the cross-entropy splits into a sum over gates
    return sum(_cross_entropy(E.dist(E.C.parents[g]), E.dist([g]), priors.q_gate[g]) for g in L) - E.H(L)


def mmc_gate_by_gate(E, priors, order=None):
    C = E.C
    order = list(C.gates) if order is None else list(order)
    nin = [v for v in range(C.n_nodes) if v not in C.inputs]
    d_in = input_kl(E, priors)
    total = d_in + sum(E.I([g], C.parents[g]) + gate_kl_term(E, priors, g) for g in order)
    steps = [E.I(C.inputs, nin) + d_in]
    for t, g in enumerate(order):
        later = order[t + 1:]
        steps.append((E.I([g], later) if later else 0.0) + gate_kl_term(E, priors, g))
    return Result(total, steps, abs(sum(steps) - total) < 1e-8 * max(1.0, total))


def mmc_layer_by_layer(E, priors):
    C = E.C
    layers = C.layers()
    nin = [v for v in range(C.n_nodes) if v not in C.inputs]
    d_in = input_kl(E, priors)
    total = d_in
    steps = [E.I(C.inputs, nin) + d_in]
    for l, L in enumerate(layers):
        pa = set().union(*(C.parents[g] for g in L))
        kl = layer_kl_term(E, priors, L)
        total += E.I(L, pa) + kl
        later = [g for LL in layers[l + 1:] for g in LL]
        steps.append((E.I(L, later) if later else 0.0) + kl)
    return Result(total, steps, abs(sum(steps) - total) < 1e-8 * max(1.0, total))
