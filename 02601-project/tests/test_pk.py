"""
test_pk.py -- tests for the five-compartment ODE model (validation plan, section 11).

Author:         AI-generated draft (Claude, Anthropic) -- owner: Person 1
Human reviewer: TODO (Person 1)
"""
import unittest

import numpy as np

import _helpers  # noqa: F401
from src import PKModel, Scenario, load_drug, load_physiology, load_scenario
from src.pk_model import COMPARTMENTS


def model(drug="acetaminophen", meal="fasted", route="oral", **drug_overrides):
    """Build a PKModel; keyword overrides change drug parameters (e.g. absorption_rate=0)."""
    d = load_drug(drug, **drug_overrides)
    return PKModel(d, load_physiology(), load_scenario(d, meal, route))


class TestPKModel(unittest.TestCase):

    def test_mass_conservation(self):
        """Amounts in all compartments + eliminated drug = dose at every time point."""
        for route in ("oral", "iv"):
            for drug in ("acetaminophen", "caffeine"):
                res = model(drug, route=route).solve((0, 24))
                self.assertLess(res.mass_balance_error(), 1e-6, f"{drug} {route}")

    def test_zero_absorption_gives_no_systemic_exposure(self):
        """If the intestine absorbs nothing (k_a = 0), no drug may ever appear in blood."""
        res = model(absorption_rate=0.0).solve((0, 12))
        self.assertAlmostEqual(float(np.max(res.blood)), 0.0, places=12)

    def test_zero_clearance_gives_no_elimination(self):
        """With liver and kidney clearance switched off, nothing leaves: all drug stays in the body."""
        res = model(hepatic_clearance=0.0, renal_clearance=0.0).solve((0, 12))
        self.assertLess(float(sum(v[-1] for v in res.eliminated.values())), 1e-6)
        total = sum(res.amounts[c][-1] for c in COMPARTMENTS)
        self.assertAlmostEqual(total, res.dose, places=4)

    def test_iv_rises_immediately_oral_is_delayed(self):
        """IV injection puts drug straight into blood; a swallowed pill needs time to be absorbed."""
        iv = model(route="iv").solve((0, 12))
        oral = model(route="oral").solve((0, 12))
        self.assertGreater(iv.blood[0], 0.0)          # IV: C_C(0) = Dose / V_C
        self.assertEqual(oral.blood[0], 0.0)          # oral: nothing in blood at t = 0
        self.assertAlmostEqual(iv.blood[0], iv.dose / load_physiology().blood_volume)

    def test_non_negative(self):
        """Amounts of drug can never be negative."""
        res = model().solve((0, 24))
        for c in COMPARTMENTS:
            self.assertGreaterEqual(float(np.min(res.amounts[c])), 0.0)

    def test_iv_one_compartment_limit_matches_analytic(self):
        """With no liver/tissue exchange, IV blood follows C = Dose/V exp(-CL_R t / V)."""
        d = load_drug("acetaminophen", hepatic_clearance=0.0)
        p = load_physiology(hepatic_flow=0.0, tissue_flow=0.0)
        res = PKModel(d, p, Scenario("iv", "fasted", 4.0)).solve((0, 10))
        V, CL = p.blood_volume, d.renal_clearance
        exact = d.dose_mg / V * np.exp(-CL / V * res.time)
        self.assertLess(np.max(np.abs(res.blood - exact)) / exact[0], 1e-6)

    def test_auc_matches_bioavailability_over_clearance(self):
        """Long-time oral AUC -> F * Dose / CL (well-stirred liver), a classic PK identity."""
        m = model()
        res = m.solve((0, 60), n_points=6001)
        auc = np.trapezoid(res.blood, res.time)
        expected = m.oral_bioavailability() * m.drug.dose_mg / m.systemic_clearance()
        self.assertAlmostEqual(auc / expected, 1.0, delta=0.01)

    def test_fed_delays_tmax(self):
        """A full stomach empties more slowly, so the peak should come later and be lower."""
        fasted = model(meal="fasted").solve((0, 12))
        fed = model(meal="fed").solve((0, 12))
        self.assertGreater(fed.time[np.argmax(fed.blood)], fasted.time[np.argmax(fasted.blood)])
        self.assertLess(np.max(fed.blood), np.max(fasted.blood))


if __name__ == "__main__":
    unittest.main()
