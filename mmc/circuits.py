"""Boolean circuits as DAGs, and the circuit families used in the paper's figures.

Node kinds:
    "IN"      input node, re-sampled from p_in at the start of every run
    "CONST0"  node hard-wired to 0 (e.g. a grounded carry-in); never re-sampled or updated
    gate      any key of GATES (two-input deterministic gates)
Gates are stored in construction order, which is always a valid topological order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GATES = {
    "AND": lambda a, b: a & b,
    "OR": lambda a, b: a | b,
    "XOR": lambda a, b: a ^ b,
    "NAND": lambda a, b: 1 - (a & b),
    "NOR": lambda a, b: 1 - (a | b),
    "XNOR": lambda a, b: 1 - (a ^ b),
    "ZERO": lambda a, b: np.zeros_like(a),   # CONST gate that always outputs 0
}


@dataclass
class Circuit:
    names: list = field(default_factory=list)
    kind: list = field(default_factory=list)
    parents: list = field(default_factory=list)
    index: dict = field(default_factory=dict)

    # -- construction -------------------------------------------------------------------
    def _add(self, name, kind, parents):
        assert name not in self.index, f"duplicate node {name}"
        self.index[name] = len(self.names)
        self.names.append(name)
        self.kind.append(kind)
        self.parents.append([self.index[p] for p in parents])
        return self.index[name]

    def add_input(self, name):
        return self._add(name, "IN", [])

    def add_const0(self, name):
        return self._add(name, "CONST0", [])

    def add_gate(self, name, kind, parents):
        assert kind in GATES, kind
        return self._add(name, kind, parents)

    def alias(self, new_name, existing):
        self.index[new_name] = self.index[existing]

    # -- structure ----------------------------------------------------------------------
    @property
    def n_nodes(self):
        return len(self.names)

    @property
    def inputs(self):
        return [v for v, k in enumerate(self.kind) if k == "IN"]

    @property
    def gates(self):
        return [v for v, k in enumerate(self.kind) if k not in ("IN", "CONST0")]

    @property
    def size(self):
        return len(self.gates)

    def layers(self):
        """Eq. (32): each gate goes to the earliest layer in which all its parents are evaluated."""
        level = {}
        for v in range(self.n_nodes):
            level[v] = 0 if not self.parents[v] else 1 + max(level[p] for p in self.parents[v])
        depth = max(level[g] for g in self.gates)
        return [[g for g in self.gates if level[g] == l] for l in range(1, depth + 1)]

    def depth(self):
        return len(self.layers())

    def random_topological_order(self, rng):
        done = set(v for v in range(self.n_nodes) if self.kind[v] in ("IN", "CONST0"))
        remaining, order = set(self.gates), []
        while remaining:
            ready = sorted(g for g in remaining if all(p in done for p in self.parents[g]))
            g = ready[rng.integers(len(ready))]
            order.append(g)
            done.add(g)
            remaining.remove(g)
        return order

    # -- evaluation ---------------------------------------------------------------------
    def evaluate(self, X_in):
        """X_in: (N, #inputs) 0/1 array -> (N, #nodes) array with every node's value."""
        X = np.zeros((X_in.shape[0], self.n_nodes), dtype=np.uint8)
        X[:, self.inputs] = X_in
        for g in self.gates:
            X[:, g] = GATES[self.kind[g]](*(X[:, p] for p in self.parents[g]))
        return X

    def gate_table(self, g):
        """Output of gate g for every parent configuration c = sum_j x_{pa_j} 2^j."""
        k = len(self.parents[g])
        cfg = (np.arange(2 ** k)[:, None] >> np.arange(k)) & 1
        return np.asarray(GATES[self.kind[g]](*(cfg[:, j].astype(np.uint8) for j in range(k))), dtype=np.uint8)


# ---------------------------------------------------------------------------- adders
def ripple_carry_adder(n, carry_in="const0"):
    """n-bit ripple-carry adder of 2-input gates; |C| = 5n.
    Full adder at bit i: x=a^b, g=a&b, s=x^c, t=x&c, c_out=g|t.
    carry_in: "const0" (tied to 0, used in the figures) or "input" (an extra input bit)."""
    C = Circuit()
    for i in range(n):
        C.add_input(f"a{i}")
    for i in range(n):
        C.add_input(f"b{i}")
    (C.add_const0 if carry_in == "const0" else C.add_input)("c0")
    for i in range(n):
        C.add_gate(f"x{i}", "XOR", [f"a{i}", f"b{i}"])
        C.add_gate(f"g{i}", "AND", [f"a{i}", f"b{i}"])
        C.add_gate(f"s{i}", "XOR", [f"x{i}", f"c{i}"])
        C.add_gate(f"t{i}", "AND", [f"x{i}", f"c{i}"])
        C.add_gate(f"c{i + 1}", "OR", [f"g{i}", f"t{i}"])
    return C


def kogge_stone_adder(n):
    """n-bit Kogge-Stone parallel-prefix adder of 2-input gates, carry-in tied to 0.
      preprocessing      p_i = a_i XOR b_i,  g_i = a_i AND b_i
      prefix level (span d = 1, 2, 4, ...), for every i >= d, j = i - d:
                         G_i <- G_i OR (P_i AND G_j),   P_i <- P_i AND P_j
      sums               s_0 = p_0 XOR c_0,  s_i = p_i XOR G_{i-1},  carry out c_n = G_{n-1}
    |C| = 9, 18, 27, 39, 51, 63, 75, 90 for n = 2..9."""
    C = Circuit()
    for i in range(n):
        C.add_input(f"a{i}")
    for i in range(n):
        C.add_input(f"b{i}")
    C.add_const0("c0")
    for i in range(n):
        C.add_gate(f"p{i}", "XOR", [f"a{i}", f"b{i}"])
        C.add_gate(f"g{i}", "AND", [f"a{i}", f"b{i}"])
    G = {i: f"g{i}" for i in range(n)}
    P = {i: f"p{i}" for i in range(n)}
    d, lvl = 1, 1
    while d < n:
        newG, newP = dict(G), dict(P)
        for i in range(d, n):
            j = i - d
            C.add_gate(f"PG{lvl}_{i}", "AND", [P[i], G[j]])
            C.add_gate(f"G{lvl}_{i}", "OR", [G[i], f"PG{lvl}_{i}"])
            C.add_gate(f"P{lvl}_{i}", "AND", [P[i], P[j]])
            newG[i], newP[i] = f"G{lvl}_{i}", f"P{lvl}_{i}"
        G, P = newG, newP
        d, lvl = 2 * d, lvl + 1
    C.add_gate("s0", "XOR", ["p0", "c0"])
    for i in range(1, n):
        C.add_gate(f"s{i}", "XOR", [f"p{i}", G[i - 1]])
    C.alias(f"c{n}", G[n - 1])
    return C


def check_adder(C, n):
    """Assert that C adds two n-bit numbers correctly on every input."""
    m = len(C.inputs)
    X_in = ((np.arange(2 ** m)[:, None] >> np.arange(m)) & 1).astype(np.uint8)
    X = C.evaluate(X_in)
    a = sum(X_in[:, i].astype(int) << i for i in range(n))
    b = sum(X_in[:, n + i].astype(int) << i for i in range(n))
    c0 = X_in[:, 2 * n].astype(int) if m == 2 * n + 1 else 0
    out = sum(X[:, C.index[f"s{i}"]].astype(int) << i for i in range(n)) + (X[:, C.index[f"c{n}"]].astype(int) << n)
    assert np.array_equal(out, a + b + c0), "adder is wrong"


# ---------------------------------------------------------------------------- trees and small circuits
def binary_tree(n_inputs, kind="AND"):
    """Balanced binary tree of one gate type on n_inputs = 2^k inputs; |C| = n_inputs - 1."""
    assert n_inputs >= 2 and n_inputs & (n_inputs - 1) == 0
    C = Circuit()
    level = [f"x{i}" for i in range(n_inputs)]
    for x in level:
        C.add_input(x)
    d = 1
    while len(level) > 1:
        nxt = []
        for j in range(0, len(level), 2):
            name = f"{kind}{d}_{j // 2}"
            C.add_gate(name, kind, [level[j], level[j + 1]])
            nxt.append(name)
        level, d = nxt, d + 1
    return C


def and_tree_one_or_per_level(n_inputs):
    """Binary tree in which the first (leftmost) gate of every level is an OR, all others AND."""
    C = Circuit()
    level = [f"x{i}" for i in range(n_inputs)]
    for x in level:
        C.add_input(x)
    d = 1
    while len(level) > 1:
        nxt = []
        for j in range(0, len(level), 2):
            kind = "OR" if j == 0 else "AND"
            name = f"{kind}{d}_{j // 2}"
            C.add_gate(name, kind, [level[j], level[j + 1]])
            nxt.append(name)
        level, d = nxt, d + 1
    return C


def four_layer_and_or():
    """3 inputs; 4 layers, each computing AND(n0,n1), AND(n1,n2), OR(n2,n0) of the previous
    layer's three nodes (n0, n1, n2)."""
    C = Circuit()
    prev = ["x0", "x1", "x2"]
    for x in prev:
        C.add_input(x)
    for l in range(1, 5):
        C.add_gate(f"A{l}a", "AND", [prev[0], prev[1]])
        C.add_gate(f"A{l}b", "AND", [prev[1], prev[2]])
        C.add_gate(f"O{l}", "OR", [prev[2], prev[0]])
        prev = [f"A{l}a", f"A{l}b", f"O{l}"]
    return C


def single_gate(kind):
    C = Circuit()
    C.add_input("a")
    C.add_input("b")
    C.add_gate("g", kind, ["a", "b"])
    return C
