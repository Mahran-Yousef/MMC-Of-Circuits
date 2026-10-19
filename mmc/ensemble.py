"""The joint distribution p^0 of all nodes at the end of a run (Eq. 13), computed exactly.

p^0(x) = R(x_nin | x_in) p_in(x_in). For deterministic gates every input string determines all
node values, so p^0 is stored as one row per input string (weight p_in). Every distribution that
appears during a run is a product of marginals of p^0 (Eqs. 15, 21, 22), so all the entropies and
mutual informations needed by Eqs. (26) and (33) are entropies of marginals of p^0.
"""

import numpy as np

LOG = np.log2


def all_inputs(m):
    """All 2^m input strings; row r is the binary expansion of r (bit j = input j)."""
    r = np.arange(2 ** m, dtype=np.int64)
    return ((r[:, None] >> np.arange(m)) & 1).astype(np.uint8)


def entropy(p):
    p = np.asarray(p, float)
    p = p[p > 0]
    return float(-(p * LOG(p)).sum())


class Ensemble:
    def __init__(self, C, p_in=None):
        self.C = C
        m = len(C.inputs)
        self.X = C.evaluate(all_inputs(m))
        self.w = np.full(2 ** m, 2.0 ** -m) if p_in is None else np.asarray(p_in, float)
        assert self.w.shape == (2 ** m,) and abs(self.w.sum() - 1) < 1e-12
        self._H = {}

    def config(self, nodes):
        """Integer configuration of the given nodes for every row (bit j = nodes[j])."""
        c = np.zeros(len(self.w), dtype=np.int64)
        for j, v in enumerate(nodes):
            c |= self.X[:, v].astype(np.int64) << j
        return c

    def dist(self, nodes):
        """Marginal distribution over the 2^len(nodes) configurations of `nodes`."""
        nodes = list(nodes)
        return np.bincount(self.config(nodes), weights=self.w, minlength=2 ** len(nodes))

    def H(self, nodes):
        key = tuple(sorted(set(nodes)))
        if not key:
            return 0.0
        if key not in self._H:
            if len(key) <= 62:
                _, inv = np.unique(self.config(key), return_inverse=True)
            else:
                packed = np.ascontiguousarray(np.packbits(self.X[:, list(key)], axis=1))
                _, inv = np.unique(packed.view(np.dtype((np.void, packed.shape[1]))).ravel(),
                                   return_inverse=True)
            self._H[key] = entropy(np.bincount(inv.ravel(), weights=self.w))
        return self._H[key]

    def I(self, A, B):
        A, B = set(A), set(B)
        return self.H(A) + self.H(B) - self.H(A | B)

    def p1(self, v):
        """P(x_v = 1)."""
        return float((self.w * self.X[:, v]).sum())
