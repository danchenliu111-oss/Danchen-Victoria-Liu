"""
drug.py -- the Drug data class and the drug-parameter loader.

Author:         AI-generated draft (Claude, Anthropic) -- module owner: Person 3
Human reviewer: TODO (Person 3)

WHAT THIS FILE DOES (plain language)
------------------------------------
A `Drug` object is simply a labelled box of numbers describing one medicine: how big
the dose is, how fast it is absorbed, how quickly the liver and kidneys remove it, etc.
The numbers are stored in data/drugs.json (so they can be changed without touching the
code) and every value's source is listed in data/references.csv.

Physical-chemistry helper: how fast does the drug spread (diffuse) in blood plasma?
--------------------------------------------------------------------------------------
The vessel PDE needs a diffusion coefficient D. Instead of guessing it, we estimate it
from the size of the molecule with the Stokes-Einstein relation:

    D = k_B T / (6 pi mu r)

    k_B : Boltzmann constant (links temperature to molecular jiggling energy)
    T   : body temperature in kelvin (310 K = 37 C)
    mu  : viscosity ("thickness") of plasma -- thicker liquid -> slower spreading
    r   : radius of the drug molecule -- bigger molecule -> slower spreading

If r is not known, we estimate it from the molecular weight MW by treating the molecule as
a small compact ball of density ~1.3 g/cm^3:   r = (3 MW / (4 pi rho N_A))^(1/3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Folder that holds drugs.json, physiology.json and references.csv
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

K_BOLTZMANN = 1.380649e-23   # Boltzmann constant [J/K]
AVOGADRO = 6.02214076e23     # molecules per mole [1/mol]


@dataclass
class Drug:
    """Drug-specific physicochemical and pharmacokinetic parameters.

    Units: dose mg, molecular_weight g/mol, molecular_radius nm, solubility mg/mL,
    absorption_rate 1/h, clearances L/h, partition coefficients dimensionless.
    """

    name: str                       # display name, e.g. "Caffeine"
    dose_mg: float                  # amount given [mg]
    molecular_weight: float         # mass of one mole of molecules [g/mol]
    molecular_radius: Optional[float]  # size of the molecule [nm]; None = estimate from MW
    solubility: Optional[float]     # how much dissolves in water [mg/mL] (informational for now)
    logP: Optional[float]           # oil-vs-water preference (informational for now)
    absorption_rate: float          # k_a  [1/h]  fraction of intestinal drug absorbed per hour
    hepatic_clearance: float        # CL_H [L/h]  liver's ability to destroy the drug
    renal_clearance: float          # CL_R [L/h]  kidneys' ability to remove the drug
    liver_partition: float          # K_L  [-]    liver/blood concentration ratio at equilibrium
    tissue_partition: float         # K_P  [-]    tissue/blood concentration ratio at equilibrium
    fraction_absorbed: float = 1.0  # F_g  [-]    fraction that survives the gut wall
    scenarios: dict = field(default_factory=dict, repr=False)  # fed/fasted settings from JSON

    def hydrodynamic_radius_m(self, density_g_cm3: float = 1.3) -> float:
        """Radius of the molecule in metres (given value, or estimated from molecular weight)."""
        if self.molecular_radius is not None:
            return self.molecular_radius * 1e-9                    # nm -> m
        rho = density_g_cm3 * 1e3 * 1e3                            # g/cm^3 -> g/m^3
        volume = self.molecular_weight / (rho * AVOGADRO)          # volume of ONE molecule [m^3]
        return (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)       # sphere: V = 4/3 pi r^3

    def diffusion_coefficient(self, viscosity_pa_s: float, temperature_k: float = 310.15) -> float:
        """Stokes-Einstein diffusion coefficient, returned in mm^2/s (the vessel model's unit)."""
        r = self.hydrodynamic_radius_m()
        d_m2_s = K_BOLTZMANN * temperature_k / (6.0 * np.pi * viscosity_pa_s * r)
        return d_m2_s * 1e6                                        # m^2/s -> mm^2/s

    def validate(self) -> None:
        """Stop early with a clear message if a parameter makes no physical sense."""
        for attr in ("dose_mg", "molecular_weight", "absorption_rate", "hepatic_clearance",
                     "renal_clearance", "liver_partition", "tissue_partition"):
            if getattr(self, attr) < 0:
                raise ValueError(f"{self.name}: {attr} must be non-negative")
        if self.liver_partition == 0 or self.tissue_partition == 0:
            raise ValueError(f"{self.name}: partition coefficients must be > 0")   # we divide by them
        if not 0.0 <= self.fraction_absorbed <= 1.0:
            raise ValueError(f"{self.name}: fraction_absorbed must lie in [0, 1]")


def available_drugs(path: Path | str = DATA_DIR / "drugs.json") -> list[str]:
    """Names (keys) of all drugs in the database; keys starting with '_' are comments."""
    with open(path, encoding="utf-8") as fh:
        return [k for k in json.load(fh) if not k.startswith("_")]


def load_drug(key: str, path: Path | str = DATA_DIR / "drugs.json", **overrides) -> Drug:
    """Read one drug from data/drugs.json.

    Keyword overrides replace stored values, e.g. load_drug("caffeine", dose_mg=200).
    The tests use this to switch processes off (e.g. absorption_rate=0).
    """
    with open(path, encoding="utf-8") as fh:
        db = json.load(fh)
    if key not in db:
        raise KeyError(f"Unknown drug '{key}'. Available: {available_drugs(path)}")
    entry = dict(db[key])          # copy so the loaded database is not modified
    entry.update(overrides)
    drug = Drug(**entry)           # turn the dictionary into a Drug object
    drug.validate()
    return drug
