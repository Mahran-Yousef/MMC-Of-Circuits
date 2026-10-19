"""Independent check: mismatch cost from its definition, Eq. (B5), on the full joint distribution.

The joint distribution over all 2^|V| node configurations is evolved through one run:
    input step   p^0 -> p^1 = p_in * p^0_nin                                     (Eq. 15)
                 MC_ow = D(p^0 || q^0) - D(p^1 || q'^0),  q^0 = q_in q_nin,  q'^0 = p_in q_nin   (Eqs. 17-19)
    update step  update the gate (or all gates of a layer) S from its parents:
                 MC_t = D(p^t || q) - D(p^{t+1} || q'),  q = q_{S|pa} q_pa q_rest,  q' = pi_S q_pa q_rest
                                                                                  (Eqs. B3-B5)
By App. A the result does not depend on the arbitrary full-support distributions q_pa, q_rest,
q_nin; with rng given they are drawn at random to test exactly that, otherwise they are uniform.
Cost is exponential in the number of nodes; intended for circuits of up to ~22 nodes.
"""

import numpy as np

from .ensemble import LOG, all_inputs


def _kl(p, q):
    m = p > 0
    assert np.all(q[m] > 0), "prior lacks support"
    return float((p[m] * (LOG(p[m]) - LOG(q[m]))).sum())


class BruteForce:
    def __init__(self, C, priors, p_in=None, rng=None, max_nodes=24):
        V = C.n_nodes
        assert V <= max_nodes, f"{V} nodes is too many for brute force"
        self.C, self.pr, self.V, self.rng = C, priors, V, rng
        self.idx = np.arange(2 ** V, dtype=np.int64)
        m = len(C.inputs)
        self.p_in = np.full(2 ** m, 2.0 ** -m) if p_in is None else np.asarray(p_in, float)
        X = C.evaluate(all_inputs(m))
        states = (X.astype(np.int64) << np.arange(V)).sum(axis=1)
        self.p0 = np.bincount(states, weights=self.p_in, minlength=2 ** V)    # Eq. (13)

    def _cfg(self, nodes):
        c = np.zeros_like(self.idx)
        for j, v in enumerate(nodes):
            c |= ((self.idx >> v) & 1) << j
        return c

    def _marg(self, p, nodes):
        return np.bincount(self._cfg(nodes), weights=p, minlength=2 ** len(nodes))

    def _arbitrary(self, k):
        return np.full(2 ** k, 2.0 ** -k) if self.rng is None else self.rng.dirichlet(np.ones(2 ** k))

    def _cond_table(self, S, pa):
        """q_{S|pa}(x_S | x_pa) as [pa_cfg, S_cfg]: joint layer prior if given, else product of gate priors."""
        if tuple(S) in self.pr.q_layer:
            tab = self.pr.q_layer[tuple(S)]
            return np.repeat(tab, 2 ** len(pa), axis=0) if tab.shape[0] == 1 else tab
        T = np.ones((2 ** len(pa), 2 ** len(S)))
        pa_bits = (np.arange(2 ** len(pa))[:, None] >> np.arange(len(pa))) & 1
        pos = {v: j for j, v in enumerate(pa)}
        for k, g in enumerate(S):
            gcfg = sum(pa_bits[:, pos[p]] << j for j, p in enumerate(self.C.parents[g]))
            xg = (np.arange(2 ** len(S)) >> k) & 1
            T *= self.pr.q_gate[g][gcfg[:, None], xg[None, :]]
        return T

    def input_step(self, p):
        C = self.C
        ins = C.inputs
        nin = [v for v in range(self.V) if v not in ins]
        X = all_inputs(len(ins))
        q_in = np.prod(np.where(X == 1, self.pr.q_in, 1 - self.pr.q_in), axis=1)
        q_nin = self._arbitrary(len(nin))
        ci, cn = self._cfg(ins), self._cfg(nin)
        p_next = self.p_in[ci] * self._marg(p, nin)[cn]
        q, q_next = q_in[ci] * q_nin[cn], self.p_in[ci] * q_nin[cn]
        return p_next, _kl(p, q) - _kl(p_next, q_next)

    def update_step(self, p, S):
        C = self.C
        pa = sorted(set().union(*(C.parents[g] for g in S)))
        rest = [v for v in range(self.V) if v not in S and v not in pa]
        q = (self._cond_table(S, pa)[self._cfg(pa), self._cfg(S)]
             * self._arbitrary(len(pa))[self._cfg(pa)] * self._arbitrary(len(rest))[self._cfg(rest)])
        new = self.idx.copy()
        for g in S:
            val = C.gate_table(g)[self._cfg(C.parents[g])].astype(np.int64)
            new = (new & ~(1 << g)) | (val << g)
        p_next = np.bincount(new, weights=p, minlength=len(p))
        q_next = np.bincount(new, weights=q, minlength=len(q))
        return p_next, _kl(p, q) - _kl(p_next, q_next)

    def run(self, schedule):
        """schedule: list of gate lists (singletons for gate-by-gate, C.layers() for layer-by-layer).
        Returns (total, per-step costs, max |p_end - p^0|)."""
        p, mc = self.input_step(self.p0)
        steps = [mc]
        for S in schedule:
            p, mc = self.update_step(p, S)
            steps.append(mc)
        return sum(steps), steps, float(np.abs(p - self.p0).max())
