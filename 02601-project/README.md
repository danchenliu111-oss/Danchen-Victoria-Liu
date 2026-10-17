# Multiscale Modeling and Visualization of Drug Absorption, Distribution and Vascular Transport

**02-601 Programming for Scientists, final project (work in progress, checkpoint 1)**
Team _[Team name]_: _[Person 1]_, _[Person 2]_, _[Person 3]_, _[Person 4]_

```
Dose ─► 5-compartment ODE model ─► C_blood(t) ─► 2-D vessel PDE ─► interactive visualization
        (whole body, hours)                     (one local vessel, seconds)
```

Whole-body absorption and distribution decide the systemic blood concentration C_C(t).
That concentration then becomes the inlet boundary condition of a spatially resolved 2-D
advection–diffusion model of a single blood vessel. Coupling is one-way (ODE → PDE).

![multiscale coupling](docs/figures/multiscale_coupling.png)

> Educational, mechanistic model. It is **not** clinical or personalized dosing advice.

## Quick start

```bash
pip install -r requirements.txt            # numpy, scipy, matplotlib, streamlit
python scripts/run_checkpoint1.py          # all figures + movies -> docs/figures (~40 s)
python -m unittest discover -s tests -v    # 22 tests, ~2 s
streamlit run app.py                       # interactive app
```

```python
from src import Simulation
sim = Simulation.from_names("acetaminophen", meal_state="fed", administration="oral")
pk = sim.run_pk()                # PKResult: 5 compartments over 12 h
print(sim.pk_metrics())          # Cmax, Tmax, AUC, half-life
v = sim.run_vessel(body_time=1.0)  # VesselResult: c[t, y, x] with inlet C_C(1.0 h)
```

## Model

### 1. Whole body: five well-mixed compartments (ODE, time in h)
State variables are amounts A_i (mg); concentrations are C_i = A_i / V_i.

| Compartment | Equation |
|---|---|
| Stomach | dA_S/dt = −k_GE A_S |
| Intestine | dA_I/dt = k_GE A_S − k_a A_I |
| Liver | dA_L/dt = F_g k_a A_I + Q_H (C_C − C_L/K_L) − CL_H C_L/K_L |
| Central blood | dA_C/dt = Q_H (C_L/K_L − C_C) − Q_P (C_C − C_P/K_P) − CL_R C_C |
| Peripheral tissue | dA_P/dt = Q_P (C_C − C_P/K_P) |

Oral: A_S(0) = Dose. IV: A_C(0) = Dose. The kidney is a clearance sink (CL_R). Eliminated drug is tracked so that
Σ A_i + eliminated = Dose can be checked. The system is solved with `scipy.integrate.solve_ivp` (LSODA).
Fed and fasted scenarios change the gastric-emptying rate k_GE and, optionally, an absorption modifier on k_a.

### 2. Local vessel: 2-D advection–diffusion (PDE, time in s, lengths in mm)

  ∂c/∂t + u(y) ∂c/∂x = D ∇²c,  u(y) = 2 ū (1 − (y/R)²)

The domain is an axial slice of an idealized cylindrical vessel. Boundary conditions: inlet **c(0, y, τ) = C_C(t_body + τ)**
(from the PK model), zero-gradient outlet, no-flux walls. D comes from the Stokes–Einstein relation
D = k_BT / (6πμr), using plasma viscosity and a molecular radius estimated from the molecular weight.
Numerics: finite volumes, MUSCL/van Leer advection, SSP-RK2 time stepping, dt from the CFL and diffusion limits;
mass is conserved to about 1e-14.
Stretch goal, already prototyped and off by default: a permeable wall J = P(c_lumen − c_tissue) plus a tissue layer.

## Classes (API contracts)

| Class | File | Responsibility |
|---|---|---|
| `Drug`, `load_drug()` | `src/drug.py` | drug parameters, Stokes–Einstein D |
| `Physiology`, `Scenario`, `load_scenario()` | `src/physiology.py` | body parameters; oral/IV, fed/fasted |
| `PKState`, `PKModel`, `PKResult` | `src/pk_model.py` | 5-compartment ODE: `derivatives()`, `solve()` |
| `PKMetrics` | `src/metrics.py` | Cmax, Tmax, AUC, half-life |
| `VesselModel`, `VesselResult` | `src/vessel.py` | `velocity_field()`, `step()`, `solve(inlet, duration)` |
| `Simulation` | `src/simulation.py` | `run_pk()`, `run_vessel(body_time)`, `compare_scenarios()` |
| `Visualizer` | `src/visualization.py` | PK curves, fed/fasted, body view, vessel heat maps/movies |

## Data
`data/drugs.json` (acetaminophen, caffeine), `data/physiology.json` (70-kg adult) and `data/references.csv`
(source and verification status of every parameter). Several values are documented simplifying assumptions or were
chosen to reproduce literature clearance, volume of distribution and Tmax. Checking them is ongoing work.

## Repository layout
```
src/  drug.py physiology.py pk_model.py metrics.py vessel.py simulation.py visualization.py
data/ drugs.json physiology.json references.csv
tests/ test_pk.py test_vessel.py test_metrics.py
scripts/run_checkpoint1.py   app.py   docs/figures/   report/
```

## Authorship and AI use
Every file states its author and a human reviewer. The checkpoint-1 code skeleton was drafted with an AI assistant
(Claude, Anthropic) following our technical design document. The team reviews and tests it and is responsible for it.
Later extensions will be written by team members, keeping AI-written code under the course's 50 % limit.

## References
1. Forrest JA, Clements JA, Prescott LF (1982). Clinical pharmacokinetics of paracetamol. *Clin Pharmacokinet* 7:93–107.
2. Arnaud MJ (2011). Pharmacokinetics and metabolism of natural methylxanthines in animal and man. *Handb Exp Pharmacol* 200:33–91.
3. Davies B, Morris T (1993). Physiological parameters in laboratory animals and humans. *Pharm Res* 10:1093–1095.
4. Brown RP et al. (1997). Physiological parameter values for physiologically based pharmacokinetic models. *Toxicol Ind Health* 13:407–484.
5. Rowland M, Tozer TN. *Clinical Pharmacokinetics and Pharmacodynamics*. Lippincott Williams & Wilkins.
6. LeVeque RJ (2002). *Finite Volume Methods for Hyperbolic Problems*. Cambridge University Press.
