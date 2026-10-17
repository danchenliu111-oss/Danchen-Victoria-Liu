"""
metrics.py -- summary pharmacokinetic numbers derived from a PKResult.

Author:         AI-generated draft (Claude, Anthropic) -- module owner: Person 3
Human reviewer: TODO (Person 3)

WHAT THIS FILE DOES (plain language)
------------------------------------
A whole concentration-versus-time curve is hard to compare at a glance, so pharmacologists
summarise it with a few standard numbers:

    Cmax  : the highest blood concentration reached [mg/L]      ("how strong is the peak?")
    Tmax  : the time at which that peak happens [h]              ("how fast does it act?")
    AUC   : Area Under the Curve = total exposure [mg*h/L]       ("how much drug did the
            body see overall?"); computed with the trapezoidal rule (sum of thin trapezoids)
    t_1/2 : half-life [h] = time for the concentration to halve during the final,
            elimination-only phase. Found by fitting a straight line to log(concentration).
"""

from __future__ import annotations

import numpy as np

from .pk_model import PKResult


class PKMetrics:
    """Computes Cmax, Tmax, AUC and terminal half-life for one concentration-time curve."""

    def __init__(self, time: np.ndarray, concentration: np.ndarray):
        self.t = np.asarray(time, dtype=float)            # time points [h]
        self.c = np.asarray(concentration, dtype=float)   # concentration at each point [mg/L]

    @classmethod
    def from_result(cls, result: PKResult, compartment: str = "blood") -> "PKMetrics":
        """Shortcut: build metrics directly from a PKResult (default: central blood)."""
        return cls(result.time, result.concentrations[compartment])

    def cmax(self) -> float:
        """Peak concentration."""
        return float(np.max(self.c))

    def tmax(self) -> float:
        """Time of the peak: look up the time at the index where the maximum occurs."""
        return float(self.t[int(np.argmax(self.c))])

    def auc(self) -> float:
        """Area under the curve with the trapezoidal rule."""
        return float(np.trapezoid(self.c, self.t))

    def half_life(self, tail_fraction: float = 0.3) -> float:
        """Terminal half-life ln(2) / lambda_z.

        In the final phase concentration decays exponentially, C ~ exp(-lambda_z t), so
        log(C) is a straight line with slope -lambda_z. We fit that line to the last
        `tail_fraction` (default 30 %) of the curve.
        """
        n = max(3, int(len(self.t) * tail_fraction))      # number of tail points used
        t, c = self.t[-n:], self.c[-n:]
        ok = c > 0                                        # log(0) is undefined
        if ok.sum() < 3:
            return float("nan")
        slope = np.polyfit(t[ok], np.log(c[ok]), 1)[0]    # straight-line fit, keep the slope
        return float(np.log(2) / -slope) if slope < 0 else float("inf")

    def summary(self) -> dict:
        """All four numbers in one dictionary."""
        return dict(Cmax=self.cmax(), Tmax=self.tmax(), AUC=self.auc(), half_life=self.half_life())
