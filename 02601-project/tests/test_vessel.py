"""
test_vessel.py -- tests for the 2-D vessel advection-diffusion solver.

Author:         AI-generated draft (Claude, Anthropic) -- owner: Person 2
Human reviewer: TODO (Person 2)
"""
import unittest

import numpy as np

import _helpers  # noqa: F401
from src import VesselModel, Simulation


def vessel(**kw):
    """Small, fast VesselModel used by most tests; keyword arguments override defaults."""
    args = dict(diffusion_coefficient=5e-4, vessel_length=10.0, vessel_radius=0.1,
                blood_velocity=2.0, nx=100, ny=16)
    args.update(kw)
    return VesselModel(**args)


class TestVessel(unittest.TestCase):

    def test_poiseuille_mean_velocity(self):
        """Tube profile 2*u_mean*(1-(r/R)^2): its cross-section (area-weighted) mean is u_mean."""
        v = vessel(ny=400)
        r = np.abs(v.y)
        u = v.velocity_field()[:, 0]
        mean_tube = np.sum(u * r) / np.sum(r)          # area weighting 2*pi*r dr
        self.assertAlmostEqual(mean_tube, 2.0, delta=0.01)
        self.assertAlmostEqual(u.max(), 4.0, delta=0.01)

    def test_mass_conservation(self):
        """Drug that entered = drug that left + drug still inside (no numerical loss)."""
        res = vessel().solve(3.0, duration=8.0, n_frames=9)
        self.assertLess(res.mass_balance_error(), 1e-10)

    def test_mass_conservation_with_leaky_wall(self):
        """Same check with the stretch-goal leaky wall and tissue uptake switched on."""
        res = vessel(wall_permeability=2e-3, tissue_thickness=0.1, tissue_uptake=0.05).solve(
            1.0, duration=8.0, n_frames=9)
        self.assertLess(res.mass_balance_error(), 1e-10)
        self.assertGreater(res.mass_eliminated[-1], 0.0)

    def test_non_negative_and_bounded(self):
        """Concentration stays between 0 and the inlet value (no artificial wiggles)."""
        res = vessel().solve(2.5, duration=6.0, n_frames=7)
        self.assertGreaterEqual(res.concentration.min(), -1e-12)
        self.assertLessEqual(res.concentration.max(), 2.5 + 1e-9)

    def test_reaches_inlet_concentration(self):
        """Constant inlet, impermeable wall: the vessel eventually fills to C_in everywhere."""
        res = vessel().solve(4.0, duration=40.0, n_frames=5)
        self.assertLess(np.max(np.abs(res.concentration[-1] - 4.0)), 0.05)

    def test_filled_vessel_stays_uniform(self):
        """A vessel already at the inlet concentration must stay exactly there."""
        res = vessel().solve(1.7, duration=3.0, n_frames=4, initial="filled")
        self.assertLess(np.max(np.abs(res.concentration[-1] - 1.7)), 1e-10)

    def test_pure_diffusion_matches_analytic(self):
        """u = 0, Gaussian in x: variance grows as sigma0^2 + 2 D t."""
        D, s0, T = 1e-3, 0.3, 20.0
        v = vessel(blood_velocity=0.0, diffusion_coefficient=D, nx=200, ny=4)
        c0 = np.exp(-(v.x - 5.0) ** 2 / (2 * s0 ** 2))
        c = np.repeat(c0[None, :], v.ny, axis=0)
        t = 0.0
        while t < T - 1e-12:
            c, _ = v.step(c, 0.0)
            t += v.dt
        st = np.sqrt(s0 ** 2 + 2 * D * t)
        exact = s0 / st * np.exp(-(v.x - 5.0) ** 2 / (2 * st ** 2))
        self.assertLess(np.max(np.abs(c[0] - exact)), 2e-3)

    def test_grid_convergence(self):
        """Mean lumen concentration changes little under grid refinement."""
        means = []
        for nx, ny in ((100, 12), (200, 24)):
            res = vessel(nx=nx, ny=ny).solve(1.0, duration=3.0, n_frames=2)
            means.append(res.mean_lumen_concentration()[-1])
        self.assertLess(abs(means[0] - means[1]) / means[1], 0.02)


class TestCoupling(unittest.TestCase):

    def test_inlet_equals_pk_blood_concentration(self):
        """ODE -> PDE: the vessel inlet must equal C_C(t_body) from the PK model."""
        sim = Simulation.from_names("acetaminophen", "fasted", "oral")
        pk = sim.run_pk()
        sim.vessel_settings.duration_s = 2.0
        sim.vessel_settings.n_frames = 3
        for tb in (0.5, 1.0, 4.0):
            v = sim.run_vessel(tb)
            self.assertAlmostEqual(v.inlet_values[0], pk.blood_concentration_at(tb), places=10)
            self.assertEqual(v.body_time_h, tb)


if __name__ == "__main__":
    unittest.main()
