"""
physiology.py -- Physiology and Scenario data classes and their loaders.

Author:         AI-generated draft (Claude, Anthropic) -- module owner: Person 3
Human reviewer: TODO (Person 3)

WHAT THIS FILE DOES (plain language)
------------------------------------
* `Physiology` describes the PERSON: how much blood they have, how big the liver is, how
  much blood flows through each organ per hour, and the size of the small vessel we zoom
  into. Default values describe a typical 70-kg adult (data/physiology.json).
* `Scenario` describes the SITUATION: was the drug swallowed ("oral") or injected into a
  vein ("iv"), and was the stomach empty ("fasted") or full ("fed")? Food slows down how
  fast the stomach empties, which is the main lever for the fed/fasted comparison.

Unit conventions
----------------
Whole-body model : volumes L, flows L/h, time h, amounts mg, concentrations mg/L
Vessel model     : lengths mm, velocity mm/s, time s, concentration mg/L
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .drug import Drug, DATA_DIR

ADMINISTRATION_ROUTES = ("oral", "iv")
MEAL_STATES = ("fasted", "fed")


@dataclass
class Physiology:
    """Representative human physiological parameters."""

    # --- whole-body model (ODE)
    blood_volume: float            # V_C  [L]   all circulating blood, treated as one well-mixed tank
    liver_volume: float            # V_L  [L]   size of the liver
    tissue_volume: float           # V_P  [L]   muscle, fat, skin ... lumped into one "tissue" tank
    hepatic_flow: float            # Q_H  [L/h] blood flowing through the liver every hour
    tissue_flow: float             # Q_P  [L/h] blood flowing through all other tissue every hour
    blood_viscosity: float         # mu_blood [Pa s] "thickness" of whole blood
    # --- local vessel model (PDE)
    vessel_radius: float           # R [mm]    radius of the small vessel we zoom into
    vessel_length: float           # L [mm]    length of that vessel segment
    blood_velocity: float = 2.0    # average blood speed in that vessel [mm/s]
    plasma_viscosity: float = 1.2e-3   # [Pa s] liquid part of blood; sets molecular diffusion
    temperature: float = 310.15        # [K] body temperature (37 C)
    body_weight: float = 70.0          # [kg]
    heart_rate: Optional[float] = None  # [beats/min] optional, for pulsatile flow (stretch goal)

    def pressure_gradient(self) -> float:
        """Axial pressure gradient [Pa/m] that drives the mean velocity in a tube (Hagen-Poiseuille).

        dp/dx = 8 mu u_mean / R^2. Reported for interpretation; the PDE uses the velocity directly.
        """
        R = self.vessel_radius * 1e-3      # mm -> m
        u = self.blood_velocity * 1e-3     # mm/s -> m/s
        return 8.0 * self.blood_viscosity * u / R ** 2


@dataclass
class Scenario:
    """Simulation condition: route of administration and meal state."""

    administration: str               # "oral" or "iv"
    meal_state: str                   # "fasted" or "fed"
    gastric_emptying_rate: float      # k_GE [1/h]
    absorption_modifier: float = 1.0  # multiplies k_a (meal effect on absorption)

    def __post_init__(self):
        # Runs automatically after creation: reject typos such as "IV" or "fasting".
        if self.administration not in ADMINISTRATION_ROUTES:
            raise ValueError(f"administration must be one of {ADMINISTRATION_ROUTES}")
        if self.meal_state not in MEAL_STATES:
            raise ValueError(f"meal_state must be one of {MEAL_STATES}")

    @property
    def label(self) -> str:
        """Short text used in plot legends, e.g. 'ORAL, fed'."""
        return f"{self.administration.upper()}, {self.meal_state}"


def load_physiology(key: str = "default_adult", path: Path | str = DATA_DIR / "physiology.json",
                    **overrides) -> Physiology:
    """Load a physiology parameter set from data/physiology.json."""
    with open(path, encoding="utf-8") as fh:
        entry = dict(json.load(fh)[key])
    entry.update(overrides)              # e.g. load_physiology(hepatic_flow=0) in tests
    return Physiology(**entry)


def load_scenario(drug: Drug, meal_state: str = "fasted", administration: str = "oral") -> Scenario:
    """Build the fed/fasted scenario for a drug from the parameters stored in drugs.json."""
    if meal_state not in drug.scenarios:
        raise KeyError(f"{drug.name} has no '{meal_state}' scenario")
    s = drug.scenarios[meal_state]       # e.g. {"gastric_emptying_rate": 1.0, ...}
    return Scenario(administration=administration, meal_state=meal_state,
                    gastric_emptying_rate=s["gastric_emptying_rate"],
                    absorption_modifier=s.get("absorption_modifier", 1.0))
