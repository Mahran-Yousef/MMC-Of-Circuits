"""Verification suite.

check_equations     Eq. (26) / Eq. (33) and their per-step forms (Eqs. 20, C31) against the
                    brute-force Eq. (B5), on small circuits with random priors, random input
                    distributions, random topological orders, and random q_pa / q_rest / q_nin.
check_figure_points every plotted point small enough for brute force, recomputed with Eq. (B5).
"""

import numpy as np

from .bruteforce import BruteForce
from .circuits import (and_tree_one_or_per_level, binary_tree, four_layer_and_or, kogge_stone_adder,
                       ripple_carry_adder)
from .ensemble import Ensemble
from .mismatch import mmc_gate_by_gate, mmc_layer_by_layer
from .priors import Priors, per_gate_priors

TOL = 1e-9


def _random_priors(C, rng, joint_layers=False):
    """Random parent-dependent gate priors in [0.1, 0.9]; optionally random parent-dependent joint layer priors."""
    q_gate = {}
    for g in C.gates:
        q1 = rng.uniform(0.1, 0.9, size=2 ** len(C.parents[g]))
        q_gate[g] = np.stack([1 - q1, q1], axis=1)
    q_layer = {}
    if joint_layers:
        for L in C.layers():
            n_pa = len(set().union(*(C.parents[g] for g in L)))
            q_layer[tuple(L)] = rng.dirichlet(np.ones(2 ** len(L)), size=2 ** n_pa)
    return Priors(q_in=rng.uniform(0.1, 0.9, size=len(C.inputs)), q_gate=q_gate, q_layer=q_layer)


def check_equations(quick=False, seed=0):
    """Returns a list of (description, max abs error in bits)."""
    rng = np.random.default_rng(seed)
    circuits = [("RCA n=1 (carry-in input)", ripple_carry_adder(1, "input")),
                ("RCA n=1 (carry-in tied to 0)", ripple_carry_adder(1)),
                ("Kogge-Stone n=2", kogge_stone_adder(2)),
                ("AND tree, 4 inputs", binary_tree(4, "AND"))]
    if not quick:
        circuits += [("RCA n=2 (carry-in input)", ripple_carry_adder(2, "input")),
                     ("AND tree with one OR per level, 8 inputs", and_tree_one_or_per_level(8)),
                     ("four-layer AND/OR circuit", four_layer_and_or())]
    err = {"Eq.26 total vs brute force": 0.0, "Eqs.20/C31 per step vs brute force": 0.0,
           "Eq.33 (product layer priors) vs Eq.26": 0.0, "Eq.33 (product layer priors) vs brute force": 0.0,
           "Eq.33 (joint layer priors) vs brute force": 0.0, "run returns to p^0": 0.0,
           "per-step sum vs closed form": 0.0}
    for _, C in circuits:
        for dist in ("uniform", "random"):
            m = len(C.inputs)
            p_in = None if dist == "uniform" else rng.dirichlet(np.ones(2 ** m))
            E = Ensemble(C, p_in)
            for pr in (per_gate_priors(C, lambda g: 0.5), _random_priors(C, rng)):
                for order in [C.gates] + [C.random_topological_order(rng) for _ in range(2)]:
                    res = mmc_gate_by_gate(E, pr, order)
                    total, steps, back = BruteForce(C, pr, p_in, rng=rng).run([[g] for g in order])
                    err["Eq.26 total vs brute force"] = max(err["Eq.26 total vs brute force"], abs(res.total - total))
                    err["Eqs.20/C31 per step vs brute force"] = max(
                        err["Eqs.20/C31 per step vs brute force"], max(abs(a - b) for a, b in zip(res.steps, steps)))
                    err["run returns to p^0"] = max(err["run returns to p^0"], back)
                    err["per-step sum vs closed form"] = max(err["per-step sum vs closed form"],
                                                             abs(sum(res.steps) - res.total))
                lay = mmc_layer_by_layer(E, pr)
                bl = BruteForce(C, pr, p_in, rng=rng).run(C.layers())[0]
                err["Eq.33 (product layer priors) vs Eq.26"] = max(
                    err["Eq.33 (product layer priors) vs Eq.26"], abs(lay.total - mmc_gate_by_gate(E, pr).total))
                err["Eq.33 (product layer priors) vs brute force"] = max(
                    err["Eq.33 (product layer priors) vs brute force"], abs(lay.total - bl))
            prj = _random_priors(C, rng, joint_layers=True)
            lay = mmc_layer_by_layer(E, prj)
            bl = BruteForce(C, prj, p_in, rng=rng).run(C.layers())[0]
            err["Eq.33 (joint layer priors) vs brute force"] = max(
                err["Eq.33 (joint layer priors) vs brute force"], abs(lay.total - bl))
    return list(err.items())


def check_figure_points(figure_modules, rows_by_fig, max_nodes=22, seed=1):
    """Recompute every plotted point with <= max_nodes nodes by brute force (random q_pa/q_rest).
    Returns a list of (figure id, point description, plotted value, brute-force value)."""
    from figures.common import circuit_and_priors   # local import: figures depends on mmc
    rng = np.random.default_rng(seed)
    out = []
    for mod in figure_modules:
        for cfg, row in zip(mod.configs(), rows_by_fig[mod.ID]):
            C, p_in, pr = circuit_and_priors(cfg)
            if C.n_nodes > max_nodes:
                continue
            sched = C.layers() if cfg.mode == "layer" else [[g] for g in C.gates]
            bf = BruteForce(C, pr, p_in, rng=rng).run(sched)[0]
            out.append((mod.ID, ", ".join(f"{k}={v}" for k, v in cfg.key.items()), row["mmc"], bf))
    return out
