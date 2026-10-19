"""Priors used in the figures.

A gate prior is the conditional island prior q_{g|pa(g)}(x_g | x_pa) of App. A, stored as an
array of shape (2^|pa(g)|, 2): row = parent configuration c = sum_j x_{pa_j} 2^j, column = x_g.
The input prior is a product over input bits, stored as q_in[i] = q(x_i = 1).
A layer prior (layer-by-layer execution only) is a joint q_{l|pa(l)}(x_l | x_pa(l)) of shape
(1 or 2^|pa(l)|, 2^|l|); layers without one use the product of their gates' priors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Priors:
    q_in: np.ndarray
    q_gate: dict
    q_layer: dict = field(default_factory=dict)


def _rows(C, g, q1):
    return np.tile([1 - q1, q1], (2 ** len(C.parents[g]), 1))


def uniform_priors(C):
    return per_gate_priors(C, lambda g: 0.5)


def per_gate_priors(C, q1_of, q_in=None):
    """Parent-independent prior q_g(x_g = 1) = q1_of(g) for every gate; inputs uniform by default."""
    return Priors(q_in=np.full(len(C.inputs), 0.5) if q_in is None else np.asarray(q_in, float),
                  q_gate={g: _rows(C, g, q1_of(g)) for g in C.gates})


def type_basis_priors(C, basis=None):
    """One fixed prior per gate type (a basis with priors, Sec. IV). Default: each type's output
    distribution on independent uniform inputs: AND 1/4, OR 3/4, XOR 1/2."""
    basis = {"AND": 0.25, "OR": 0.75, "XOR": 0.5} if basis is None else basis
    return per_gate_priors(C, lambda g: basis[C.kind[g]])


def matched_priors(E, eps=1e-12):
    """Per-gate prior equal to the distribution of the value the gate holds when it fires,
    q_{g|pa}(x_g | x_pa) = p_g(x_g) (its previous-run output, independent of its new parents,
    Eq. 21). This minimises every gate's KL bracket in Eq. (26), so MMC = D(p_in||q_in) + sum_g S(X_g).
    The input prior is set to the input marginals. eps of the uniform distribution is mixed in so
    every prior has full support (App. A)."""
    C = E.C
    mix = lambda p: (1 - eps) * p + eps / 2
    return per_gate_priors(C, lambda g: mix(E.p1(g)), q_in=[mix(E.p1(v)) for v in C.inputs])


def logic_priors(C, target_of, p_other):
    """Prior that puts probability p_other on x_g != target(x_pa) and 1 - p_other on
    x_g = target(x_pa). target_of(g) returns the target truth table over parent configurations,
    e.g. C.gate_table(g) (the gate's own function) or AND_TABLE. Pass the small probability
    directly (e.g. p_other = 1e-12) so it is represented exactly."""
    q_gate = {}
    for g in C.gates:
        t = np.asarray(target_of(g))
        other = np.arange(2)[None, :] != t[:, None]
        q_gate[g] = np.where(other, p_other, 1.0 - p_other)
    return Priors(q_in=np.full(len(C.inputs), 0.5), q_gate=q_gate)


AND_TABLE = np.array([0, 0, 0, 1], dtype=np.uint8)   # parent configs 00, 10, 01, 11


def with_joint_layer_priors(E, priors, lam=1.0, eps=1e-12):
    """Add a joint prior for every layer: q_l = (1 - lam) prod_{g in l} q_g + lam p_l, mixed with
    eps of uniform. p_l is the joint distribution of the layer's (previous-run) values, so lam = 1
    matches the layer's correlations. Requires parent-independent gate priors."""
    C = E.C
    q_layer = {}
    for L in C.layers():
        k = len(L)
        cfg = np.arange(2 ** k)
        prod = np.ones(2 ** k)
        for j, g in enumerate(L):
            qg = priors.q_gate[g]
            assert np.allclose(qg, qg[0]), "gate prior depends on its parents"
            prod *= np.where((cfg >> j) & 1, qg[0, 1], qg[0, 0])
        q = (1 - lam) * prod + lam * E.dist(L)
        q_layer[tuple(L)] = ((1 - eps) * q + eps / 2 ** k)[None, :]
    return Priors(q_in=priors.q_in, q_gate=priors.q_gate, q_layer=q_layer)
