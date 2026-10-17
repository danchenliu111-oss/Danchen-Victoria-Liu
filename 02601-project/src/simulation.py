"""
simulation.py -- orchestrates the whole-body PK model and the local vessel PDE.

Author:         AI-generated draft (Claude, Anthropic) -- module owner: Person 4
Human reviewer: TODO (Person 4)

WHAT THIS FILE DOES (plain language)
------------------------------------
This is the "conductor" that connects the two models:

Pipeline:   Dose -> 5-compartment ODE -> C_C(t) -> 2-D vessel PDE -> visualisation

1. run_pk()        : simulate the whole body for several hours -> blood concentration curve.
2. run_vessel(t)   : pick one moment t on that curve, read the blood concentration there,
                     and use it as the concentration of blood ENTERING the small vessel.
3. compare_scenarios(): run the body model several times (fed vs fasted, oral vs IV).
The plotting code and the app only talk to this class, never to the solvers directly.

One-way multiscale coupling (ODE -> PDE)
----------------------------------------
The PK model runs over hours; the vessel PDE over seconds. For a chosen whole-body time
t_body (h), the vessel inlet is

    c(x = 0, y, tau) = C_C(t_body + tau / 3600),    tau in seconds,

i.e. the systemic blood concentration read from the PK result. Because tau spans only
seconds, C_C is essentially constant over a vessel run, which is exactly the time-scale
separation that justifies the multiscale split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .drug import Drug, load_drug
from .metrics import PKMetrics
from .physiology import Physiology, Scenario, load_physiology, load_scenario
from .pk_model import PKModel, PKResult
from .vessel import VesselModel, VesselResult


@dataclass
class VesselSettings:
    """Numerical / optional-physics settings for the local vessel run."""

    nx: int = 200                     # grid cells along the vessel
    ny: int = 24                      # grid cells across the vessel diameter
    duration_s: float = 10.0          # how many seconds of vessel time to simulate
    n_frames: int = 101               # snapshots stored (for plots / movies)
    initial: str = "empty"            # "empty": watch the drug front enter; "filled": steady vessel
    wall_permeability: float = 0.0    # stretch goal (mm/s); 0 = impermeable wall
    tissue_thickness: float = 0.0     # stretch goal (mm)
    tissue_uptake: float = 0.0        # stretch goal (1/s)


class Simulation:
    """Couples Drug + Physiology + Scenario -> PKModel -> VesselModel."""

    def __init__(self, drug: Drug, physiology: Physiology, scenario: Scenario,
                 pk_duration_h: float = 12.0, vessel: Optional[VesselSettings] = None):
        self.drug = drug
        self.physiology = physiology
        self.scenario = scenario
        self.pk_duration_h = pk_duration_h
        self.vessel_settings = vessel or VesselSettings()
        # The PK result is stored after the first run ("cached") so that several vessel runs
        # at different body times do not re-solve the same ODE system.
        self._pk_result: Optional[PKResult] = None

    @classmethod
    def from_names(cls, drug: str = "acetaminophen", meal_state: str = "fasted",
                   administration: str = "oral", dose_mg: Optional[float] = None,
                   **kwargs) -> "Simulation":
        """Convenience constructor from the JSON databases.

        Example: Simulation.from_names("caffeine", meal_state="fed", administration="oral")
        """
        d = load_drug(drug) if dose_mg is None else load_drug(drug, dose_mg=dose_mg)
        return cls(d, load_physiology(), load_scenario(d, meal_state, administration), **kwargs)

    # ------------------------------------------------------------------ PK
    def run_pk(self) -> PKResult:
        """Solve the whole-body ODE model (cached)."""
        if self._pk_result is None:
            self._pk_result = PKModel(self.drug, self.physiology, self.scenario).solve(
                (0.0, self.pk_duration_h))
        return self._pk_result

    def pk_metrics(self) -> dict:
        """Cmax, Tmax, AUC and half-life of the central blood curve."""
        return PKMetrics.from_result(self.run_pk()).summary()

    # --------------------------------------------------------------- vessel
    def build_vessel_model(self) -> VesselModel:
        """VesselModel whose diffusion coefficient comes from the drug (Stokes-Einstein)."""
        p, s = self.physiology, self.vessel_settings
        # D is computed from the molecule's size and the plasma viscosity, not typed in by hand.
        D = self.drug.diffusion_coefficient(p.plasma_viscosity, p.temperature)
        return VesselModel(D, p.vessel_length, p.vessel_radius, p.blood_velocity, s.nx, s.ny,
                           wall_permeability=s.wall_permeability,
                           tissue_thickness=s.tissue_thickness, tissue_uptake=s.tissue_uptake)

    def run_vessel(self, body_time: float) -> VesselResult:
        """Local vessel simulation at whole-body time `body_time` [h] (ODE -> PDE coupling)."""
        pk = self.run_pk()
        s = self.vessel_settings
        # THE LINK BETWEEN THE TWO MODELS:
        # the vessel's clock tau runs in seconds, the body's clock in hours, so a vessel time
        # of tau seconds corresponds to body time  body_time + tau/3600  hours.
        inlet = lambda tau: pk.blood_concentration_at(body_time + tau / 3600.0)  # noqa: E731
        res = self.build_vessel_model().solve(inlet, s.duration_s, s.n_frames, initial=s.initial)
        res.body_time_h = body_time        # remember which body time this vessel picture shows
        return res

    # ------------------------------------------------------------ scenarios
    def compare_scenarios(self, meal_states=("fasted", "fed"), administrations=("oral",)) -> dict:
        """Run the PK model for each (route, meal) combination; returns label -> (PKResult, metrics)."""
        out = {}
        # Loop over every requested combination, e.g. (oral, fasted) and (oral, fed).
        for route in administrations:
            for meal in meal_states:
                sc = load_scenario(self.drug, meal, route)
                res = PKModel(self.drug, self.physiology, sc).solve((0.0, self.pk_duration_h))
                out[sc.label] = (res, PKMetrics.from_result(res).summary())
        return out
