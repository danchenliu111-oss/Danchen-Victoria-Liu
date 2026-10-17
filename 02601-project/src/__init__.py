"""
Multiscale drug absorption, distribution and vascular transport model.

Author:         AI-generated draft (Claude, Anthropic)
Human reviewer: TODO

Dose -> 5-compartment ODE (pk_model) -> C_C(t) -> 2-D vessel PDE (vessel) -> visualization
"""

from .drug import Drug, load_drug, available_drugs
from .physiology import Physiology, Scenario, load_physiology, load_scenario
from .pk_model import PKState, PKModel, PKResult
from .metrics import PKMetrics
from .vessel import VesselModel, VesselResult
from .simulation import Simulation, VesselSettings

__all__ = ["Drug", "load_drug", "available_drugs", "Physiology", "Scenario", "load_physiology",
           "load_scenario", "PKState", "PKModel", "PKResult", "PKMetrics", "VesselModel",
           "VesselResult", "Simulation", "VesselSettings"]
