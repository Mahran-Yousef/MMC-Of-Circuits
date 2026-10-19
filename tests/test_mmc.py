"""pytest entry point: `python -m pytest tests` (quick versions of the checks in run_all.py --verify)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

import mmc  # noqa: E402
from mmc.verify import check_equations  # noqa: E402


def test_equations_match_bruteforce():
    for name, err in check_equations(quick=True):
        assert err < 1e-9, f"{name}: {err}"


def test_adders_add():
    for n in range(1, 6):
        mmc.check_adder(mmc.ripple_carry_adder(n), n)
        mmc.check_adder(mmc.ripple_carry_adder(n, "input"), n)
        mmc.check_adder(mmc.kogge_stone_adder(n), n)


def test_uniform_priors_give_one_bit_per_gate():
    for C in (mmc.ripple_carry_adder(4), mmc.kogge_stone_adder(4), mmc.binary_tree(8, "AND")):
        assert abs(mmc.mmc_gate_by_gate(mmc.Ensemble(C), mmc.uniform_priors(C)).total - C.size) < 1e-9


def test_matched_priors_give_sum_of_entropies():
    C = mmc.kogge_stone_adder(4)
    E = mmc.Ensemble(C)
    total = mmc.mmc_gate_by_gate(E, mmc.matched_priors(E)).total
    assert abs(total - sum(E.H([g]) for g in C.gates)) < 1e-6


def test_layer_equals_gate_with_product_layer_priors():
    C = mmc.ripple_carry_adder(3)
    E = mmc.Ensemble(C)
    pr = mmc.matched_priors(E)
    gate = mmc.mmc_gate_by_gate(E, pr).total
    assert abs(mmc.mmc_layer_by_layer(E, pr).total - gate) < 1e-9
    joint = mmc.mmc_layer_by_layer(E, mmc.with_joint_layer_priors(E, pr)).total
    tc = sum(sum(E.H([g]) for g in L) - E.H(L) for L in C.layers())
    assert abs((gate - joint) - tc) < 1e-6


def test_figure_point_bruteforce():
    rng = np.random.default_rng(0)
    C = mmc.four_layer_and_or()
    pr = mmc.per_gate_priors(C, lambda g: 0.3 if C.kind[g] == "OR" else 0.5)
    bf = mmc.BruteForce(C, pr, rng=rng).run([[g] for g in C.gates])[0]
    assert abs(bf - mmc.mmc_gate_by_gate(mmc.Ensemble(C), pr).total) < 1e-9
