"""Mismatch cost (MMC) of Boolean circuits run repeatedly, gate-by-gate or layer-by-layer.

Implements the execution model and equations of Yadav, Yousef & Wolpert,
"A strictly positive lower bound on the thermodynamic cost of running a Boolean circuit".

    circuits     circuit DAGs and the circuit families used in the figures
    ensemble     exact joint distribution p^0 of all nodes (Eq. 13) and its information measures
    priors       gate / input / layer priors used in the figures
    mismatch     Eq. (26) gate-by-gate and Eq. (33) layer-by-layer mismatch cost
    bruteforce   independent check: Eq. (B5) evaluated on the full joint distribution
    verify       test suite comparing the two

All quantities are in bits (base-2 logarithms).
"""

from .circuits import (GATES, Circuit, and_tree_one_or_per_level, binary_tree, check_adder,
                       four_layer_and_or, kogge_stone_adder, ripple_carry_adder, single_gate)
from .ensemble import Ensemble, all_inputs
from .mismatch import mmc_gate_by_gate, mmc_layer_by_layer
from .priors import (Priors, logic_priors, matched_priors, per_gate_priors, type_basis_priors,
                     uniform_priors, with_joint_layer_priors)
from .bruteforce import BruteForce
