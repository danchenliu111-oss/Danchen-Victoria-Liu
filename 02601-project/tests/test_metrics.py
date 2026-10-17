"""
test_metrics.py -- tests for PKMetrics and parameter loading.

Author:         AI-generated draft (Claude, Anthropic) -- owner: Person 3
Human reviewer: TODO (Person 3)
"""
import unittest

import numpy as np

import _helpers  # noqa: F401
from src import PKMetrics, load_drug, load_physiology, load_scenario, available_drugs


class TestMetrics(unittest.TestCase):

    def test_exponential_decay(self):
        """C = C0 exp(-k t): Cmax = C0, Tmax = 0, AUC = C0/k (1 - e^{-kT}), t1/2 = ln2/k."""
        t = np.linspace(0, 20, 20001)
        C0, k = 10.0, 0.3
        m = PKMetrics(t, C0 * np.exp(-k * t))
        self.assertAlmostEqual(m.cmax(), C0)
        self.assertEqual(m.tmax(), 0.0)
        self.assertAlmostEqual(m.auc(), C0 / k * (1 - np.exp(-k * 20)), places=4)
        self.assertAlmostEqual(m.half_life(), np.log(2) / k, places=4)

    def test_bateman_peak(self):
        """One-compartment oral (Bateman) curve: Tmax = ln(ka/ke)/(ka-ke)."""
        ka, ke = 2.0, 0.3
        t = np.linspace(0, 24, 240001)
        c = ka / (ka - ke) * (np.exp(-ke * t) - np.exp(-ka * t))
        self.assertAlmostEqual(PKMetrics(t, c).tmax(), np.log(ka / ke) / (ka - ke), places=3)


class TestParameters(unittest.TestCase):

    def test_load_all_drugs_and_scenarios(self):
        """Every drug in drugs.json loads, and has usable fed and fasted settings."""
        for key in available_drugs():
            d = load_drug(key)
            for meal in ("fasted", "fed"):
                s = load_scenario(d, meal, "oral")
                self.assertGreater(s.gastric_emptying_rate, 0)

    def test_invalid_inputs_rejected(self):
        """Impossible inputs (negative clearance, unknown drug or route) raise clear errors."""
        with self.assertRaises(ValueError):
            load_drug("acetaminophen", hepatic_clearance=-1.0)
        with self.assertRaises(KeyError):
            load_drug("not_a_drug")
        with self.assertRaises(ValueError):
            load_scenario(load_drug("caffeine"), "fasted", "inhaled")

    def test_stokes_einstein_order_of_magnitude(self):
        """Small drug molecules in plasma: D ~ 1e-10 .. 1e-9 m^2/s (= 1e-4 .. 1e-3 mm^2/s)."""
        p = load_physiology()
        D = load_drug("acetaminophen").diffusion_coefficient(p.plasma_viscosity, p.temperature)
        self.assertTrue(1e-4 < D < 1.5e-3)


if __name__ == "__main__":
    unittest.main()
