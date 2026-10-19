"""One module per paper figure. ALL lists them in paper order."""

from . import (fig04_layer_vs_gate, fig06_and_tree_bound, fig07_rca_vs_kogge_stone, fig08_heterogeneous,
               fig09_and_tree_priors, fig10_mixed_tree, fig11_single_gates, fig12_four_layer_or_prior)

ALL = [fig04_layer_vs_gate, fig06_and_tree_bound, fig07_rca_vs_kogge_stone, fig08_heterogeneous,
       fig09_and_tree_priors, fig10_mixed_tree, fig11_single_gates, fig12_four_layer_or_prior]
BY_ID = {m.ID: m for m in ALL}
