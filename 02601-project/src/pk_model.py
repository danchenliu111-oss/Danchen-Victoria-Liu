"""
pk_model.py -- five-compartment whole-body pharmacokinetic (ODE) model.

Author:         AI-generated draft (Claude, Anthropic) -- module owner: Person 1
Human reviewer: TODO (Person 1)

WHAT THIS FILE DOES (plain language)
------------------------------------
"Pharmacokinetics" (PK) = what the body does to a drug: how it is absorbed, where it goes,
and how it is removed. We picture the body as five connected "tanks" (compartments).
Each tank is assumed to be perfectly stirred, so it is described by ONE number at each moment:
how many milligrams of drug it currently holds.

    Stomach -> Intestine -> Liver <-> Central Blood <-> Peripheral Tissue
                                           |
                                      kidneys remove drug (a "drain", not a tank)

Every arrow is a flow of drug. For every tank we write
    (rate of change of the amount) = (what flows in) - (what flows out).
Such equations are "ordinary differential equations" (ODEs). The computer solves them by
taking many tiny time steps (we use scipy's solve_ivp for this).

THE EQUATIONS (time in hours, amounts A in mg, concentration C = A / V in mg/L)
-------------------------------------------------------------------------------
    dA_S/dt = -k_GE A_S                                  stomach empties into intestine
    dA_I/dt =  k_GE A_S - k_a A_I                        intestine receives, then absorbs
    dA_L/dt =  F_g k_a A_I + Q_H (C_C - C_L/K_L) - CL_H (C_L/K_L)
               (absorbed drug) + (exchange with blood) - (liver metabolism)
    dA_C/dt =  Q_H (C_L/K_L - C_C) - Q_P (C_C - C_P/K_P) - CL_R C_C
               (exchange with liver) - (exchange with tissue) - (kidney removal)
    dA_P/dt =  Q_P (C_C - C_P/K_P)                       exchange with blood

Symbols:
    k_GE  gastric-emptying rate [1/h]: fraction of stomach content leaving per hour
    k_a   absorption rate [1/h]: fraction of intestinal drug absorbed per hour
    F_g   fraction of absorbed drug that survives the gut wall [-]
    Q_H   blood flow through the liver [L/h];  Q_P blood flow through other tissue [L/h]
    K_L, K_P  partition coefficients [-]: how much the organ "prefers" to hold drug vs. blood
    CL_H  hepatic (liver) clearance [L/h];  CL_R renal (kidney) clearance [L/h]
          ("clearance" = volume of blood fully cleaned of drug per hour)

Starting point: oral dose -> all drug in the stomach;  IV injection -> all drug in blood.

Mass-balance bookkeeping: drug that leaves the body is collected in three extra
"counters" (gut loss, liver metabolism, kidney), so we can always check
    (drug in the 5 tanks) + (drug removed) = dose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .drug import Drug
from .physiology import Physiology, Scenario

# Fixed order of the five compartments; used everywhere to index arrays.
COMPARTMENTS = ("stomach", "intestine", "liver", "blood", "tissue")


@dataclass
class PKState:
    """Drug amounts [mg] in the five compartments at one instant (a 'snapshot' of the body)."""

    stomach: float = 0.0
    intestine: float = 0.0
    liver: float = 0.0
    blood: float = 0.0
    tissue: float = 0.0

    def to_array(self) -> np.ndarray:
        """Pack the five amounts into a numpy array (the format the ODE solver needs)."""
        return np.array([self.stomach, self.intestine, self.liver, self.blood, self.tissue])

    @classmethod
    def from_array(cls, a) -> "PKState":
        """Unpack the first five numbers of an array back into a PKState."""
        return cls(*[float(v) for v in a[:5]])

    def total(self) -> float:
        """Total drug still inside the body [mg]."""
        return float(self.to_array().sum())


@dataclass
class PKResult:
    """Everything the ODE solver produced: amounts and concentrations over time.

    time            : 1-D array of time points [h]
    amounts         : dict  compartment name -> array of amounts [mg]
    concentrations  : dict  'liver'/'blood'/'tissue' -> array of concentrations [mg/L]
    eliminated      : dict  'gut'/'hepatic'/'renal' -> cumulative drug removed [mg]
    """

    time: np.ndarray
    amounts: dict
    concentrations: dict
    eliminated: dict
    dose: float
    drug_name: str
    scenario_label: str

    @property
    def blood(self) -> np.ndarray:
        """Central blood concentration C_C(t) [mg/L] -- this is what we hand to the vessel PDE."""
        return self.concentrations["blood"]

    def blood_concentration_at(self, t_hours: float) -> float:
        """C_C at any time, even between stored time points (straight-line interpolation)."""
        return float(np.interp(t_hours, self.time, self.blood))

    def state_at(self, t_hours: float) -> PKState:
        """Amounts in all five compartments at an arbitrary time (interpolated)."""
        return PKState(*[float(np.interp(t_hours, self.time, self.amounts[c])) for c in COMPARTMENTS])

    def mass_balance_error(self) -> float:
        """Largest relative violation of: drug in body + drug removed = dose (should be ~0)."""
        total = sum(self.amounts[c] for c in COMPARTMENTS) + sum(self.eliminated.values())
        return float(np.max(np.abs(total - self.dose)) / self.dose)


class PKModel:
    """Five-compartment ODE model, solved with scipy's solve_ivp."""

    def __init__(self, drug: Drug, physiology: Physiology, scenario: Scenario):
        self.drug = drug
        self.phys = physiology
        self.scenario = scenario

        # Copy every parameter into a short, equation-like name once, so that
        # derivatives() below reads almost exactly like the equations in the docstring.
        self.k_GE = scenario.gastric_emptying_rate               # stomach emptying [1/h]
        self.k_a = drug.absorption_rate * scenario.absorption_modifier  # meal can slow absorption
        self.F_g = drug.fraction_absorbed                        # survives the gut wall [-]
        self.Q_H, self.Q_P = physiology.hepatic_flow, physiology.tissue_flow  # blood flows [L/h]
        self.CL_H, self.CL_R = drug.hepatic_clearance, drug.renal_clearance   # clearances [L/h]
        self.K_L, self.K_P = drug.liver_partition, drug.tissue_partition      # partitions [-]
        # Volumes are needed to turn amounts (mg) into concentrations (mg/L).
        # Stomach and intestine have no volume here: only their amounts matter.
        self.V = dict(liver=physiology.liver_volume, blood=physiology.blood_volume,
                      tissue=physiology.tissue_volume)

    def initial_state(self) -> PKState:
        """Where the dose starts: oral -> stomach; IV (intravenous injection) -> blood."""
        if self.scenario.administration == "oral":
            return PKState(stomach=self.drug.dose_mg)
        return PKState(blood=self.drug.dose_mg)

    def derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        """How fast each amount is changing right now [mg/h].

        `state` holds 8 numbers: the 5 compartment amounts + 3 elimination counters.
        The solver calls this function thousands of times with different states.
        """
        # --- unpack the current amounts (the counters are not needed to compute rates)
        A_S, A_I, A_L, A_C, A_P = state[:5]

        # --- convert amounts to concentrations: C = A / V
        C_L = A_L / self.V["liver"]
        C_C = A_C / self.V["blood"]
        C_P = A_P / self.V["tissue"]

        # --- individual flows of drug [mg/h]
        absorbed = self.k_a * A_I                     # intestine -> (gut wall) -> liver
        liver_out = C_L / self.K_L                    # concentration in blood leaving the liver
        hepatic_elim = self.CL_H * liver_out          # destroyed by liver metabolism
        renal_elim = self.CL_R * C_C                  # removed by the kidneys
        # Exchange terms: drug moves from high to low concentration, carried by blood flow.
        # Positive value = net movement blood -> tissue (or blood -> liver).
        tissue_exchange = self.Q_P * (C_C - C_P / self.K_P)
        hepatic_exchange = self.Q_H * (C_C - liver_out)

        # --- balance for each compartment: inflow - outflow
        dA_S = -self.k_GE * A_S                                   # stomach only empties
        dA_I = self.k_GE * A_S - absorbed                         # in from stomach, out by absorption
        dA_L = self.F_g * absorbed + hepatic_exchange - hepatic_elim
        dA_C = -hepatic_exchange - tissue_exchange - renal_elim   # blood loses what others gain
        dA_P = tissue_exchange

        # --- the three bookkeeping counters grow by whatever left the body
        d_gut_loss = (1.0 - self.F_g) * absorbed
        return np.array([dA_S, dA_I, dA_L, dA_C, dA_P, d_gut_loss, hepatic_elim, renal_elim])

    def solve(self, t_span=(0.0, 12.0), n_points: int = 721) -> PKResult:
        """Run the simulation from t_span[0] to t_span[1] hours and return a PKResult.

        n_points controls how many time points are stored (721 over 12 h = one per minute).
        """
        # Starting vector: 5 compartment amounts followed by three counters at zero.
        y0 = np.concatenate([self.initial_state().to_array(), np.zeros(3)])
        t_eval = np.linspace(t_span[0], t_span[1], n_points)

        # LSODA automatically switches between methods for "easy" and "stiff" problems.
        # Stiff = some processes are much faster than others (here: blood-tissue exchange is
        # fast, elimination is slow). Tight tolerances make mass balance accurate to ~1e-12.
        sol = solve_ivp(self.derivatives, t_span, y0, t_eval=t_eval, method="LSODA",
                        rtol=1e-8, atol=1e-10 * max(self.drug.dose_mg, 1.0))
        if not sol.success:
            raise RuntimeError(f"PK integration failed: {sol.message}")

        # Tiny negative values (like -1e-15) can appear from rounding; amounts cannot be negative.
        y = np.clip(sol.y, 0.0, None)
        amounts = {c: y[i] for i, c in enumerate(COMPARTMENTS)}
        conc = {c: amounts[c] / self.V[c] for c in ("liver", "blood", "tissue")}
        return PKResult(time=sol.t, amounts=amounts, concentrations=conc,
                        eliminated=dict(gut=y[5], hepatic=y[6], renal=y[7]),
                        dose=self.drug.dose_mg, drug_name=self.drug.name,
                        scenario_label=self.scenario.label)

    # ---- closed-form numbers used to check the simulation ---------------------------
    def systemic_clearance(self) -> float:
        """Total clearance from blood [L/h] = effective liver clearance + kidney clearance.

        The liver can only clean the blood that actually flows through it, so its
        effective clearance is Q_H*CL_H / (Q_H + CL_H)  ("well-stirred liver" formula).
        """
        hep = self.Q_H * self.CL_H / (self.Q_H + self.CL_H) if (self.Q_H + self.CL_H) > 0 else 0.0
        return hep + self.CL_R

    def oral_bioavailability(self) -> float:
        """Fraction of an oral dose that reaches the blood unchanged: F = F_g * (1 - E_H).

        E_H = CL_H / (Q_H + CL_H) is the fraction the liver destroys on the first pass
        ("first-pass effect"), before the drug ever reaches the rest of the body.
        """
        E_H = self.CL_H / (self.Q_H + self.CL_H) if (self.Q_H + self.CL_H) > 0 else 0.0
        return self.F_g * (1.0 - E_H)
