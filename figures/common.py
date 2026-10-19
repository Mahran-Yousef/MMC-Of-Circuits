"""Shared machinery for the figure modules.

Every figure module defines:
    ID          short id, e.g. "fig07"
    TITLE       one-line description
    configs()   list of Config: one per plotted point (circuit, input distribution, priors, schedule)
    plot(rows, save)   draws the figure(s) from the computed rows; save(fig, stem) writes them
    caption(rows)      LaTeX caption(s); numbers quoted in captions come from the computed rows
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Callable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mmc import Ensemble, mmc_gate_by_gate, mmc_layer_by_layer  # noqa: E402

FIGSIZE = (10, 6.5)          # plain matplotlib defaults, as in the paper's figures
DPI = 200


@dataclass
class Config:
    key: dict                                   # identifies the point; written to the CSV
    build: Callable                             # () -> Circuit
    priors: Callable                            # (Ensemble) -> Priors
    p_in: Optional[Callable] = None             # (Circuit) -> input distribution; None = uniform
    mode: str = "gate"                          # "gate" (Eq. 26) or "layer" (Eq. 33)


def circuit_and_priors(cfg):
    C = cfg.build()
    p_in = None if cfg.p_in is None else cfg.p_in(C)
    return C, p_in, cfg.priors(Ensemble(C, p_in))


def evaluate(cfg):
    """Mismatch cost of one configuration -> CSV row."""
    C = cfg.build()
    p_in = None if cfg.p_in is None else cfg.p_in(C)
    E = Ensemble(C, p_in)
    pr = cfg.priors(E)
    res = mmc_layer_by_layer(E, pr) if cfg.mode == "layer" else mmc_gate_by_gate(E, pr)
    assert res.consistent, f"per-step and closed-form totals disagree for {cfg.key}"
    return dict(cfg.key, gates=C.size, inputs=len(C.inputs), depth=C.depth(), mmc=res.total)


def new_figure():
    plt.rcParams.update(plt.rcParamsDefault)
    return plt.subplots(figsize=FIGSIZE)


def finish(ax, xlabel, ylabel, legend=True, **legend_kw):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(**legend_kw)


class Saver:
    def __init__(self, outdir, exts=("pdf", "png")):
        self.outdir, self.exts, self.written = outdir, exts, []
        os.makedirs(outdir, exist_ok=True)

    def __call__(self, fig, stem):
        fig.tight_layout()
        for ext in self.exts:
            path = os.path.join(self.outdir, f"{stem}.{ext}")
            fig.savefig(path, dpi=DPI)
            self.written.append(path)
        plt.close(fig)


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def select(rows, **match):
    return [r for r in rows if all(r[k] == v for k, v in match.items())]
