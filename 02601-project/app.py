"""
app.py -- interactive Streamlit front end for the multiscale drug model.

Author:         AI-generated draft (Claude, Anthropic) -- owner: Person 4
Human reviewer: TODO (Person 4)

Run with:   streamlit run app.py

The app contains no solver logic: it only builds a Simulation from the user's choices and
passes PKResult / VesselResult objects to the Visualizer.
Educational model only -- not clinical dosing advice.
"""
import streamlit as st

from src import Simulation, VesselSettings, available_drugs, load_drug
from src.visualization import Visualizer

st.set_page_config(page_title="Multiscale drug transport", layout="wide")
st.title("Drug absorption, distribution and vascular transport")
st.caption("Dose → 5-compartment ODE → C_blood(t) → 2-D vessel PDE.  Educational model, not dosing advice.")

# ---------------------------------------------------------------- controls
# Everything inside the sidebar becomes an input widget; Streamlit re-runs this whole
# script from top to bottom every time the user changes one of them.
with st.sidebar:
    st.header("Scenario")
    drug_key = st.selectbox("Drug", available_drugs())
    default_dose = load_drug(drug_key).dose_mg
    dose = st.number_input("Dose (mg)", min_value=1.0, value=float(default_dose), step=50.0)
    route = st.radio("Administration", ["oral", "iv"], horizontal=True)
    meal = st.radio("Meal state", ["fasted", "fed"], horizontal=True, disabled=(route == "iv"))
    hours = st.slider("Whole-body simulation length (h)", 4, 48, 12)
    st.header("Local vessel")
    duration = st.slider("Vessel simulation length (s)", 2, 30, 10)
    leaky = st.checkbox("Stretch goal: leaky wall + tissue layer", value=False)


# @st.cache_data remembers results for inputs already seen, so moving a slider back and
# forth does not re-run the (slow) simulations.
@st.cache_data(show_spinner=False)
def run_pk(drug_key, dose, route, meal, hours):
    sim = Simulation.from_names(drug_key, meal, route, dose_mg=dose, pk_duration_h=hours)
    return sim.run_pk(), sim.pk_metrics(), sim.compare_scenarios(("fasted", "fed"), ("oral",))


@st.cache_data(show_spinner=False)
def run_vessel(drug_key, dose, route, meal, hours, body_time, duration, leaky):
    # Optional stretch-goal physics: a leaky wall and a 0.25 mm layer of tissue.
    settings = VesselSettings(duration_s=duration, n_frames=4 * duration + 1,
                              wall_permeability=2e-3 if leaky else 0.0,
                              tissue_thickness=0.25 if leaky else 0.0)
    sim = Simulation.from_names(drug_key, meal, route, dose_mg=dose, pk_duration_h=hours,
                                vessel=settings)
    return sim.run_vessel(body_time)


pk, metrics, comparison = run_pk(drug_key, dose, route, meal, hours)
viz = Visualizer()

# ------------------------------------------------------------ whole body
st.subheader("1 · Whole-body pharmacokinetics (hours)")
c1, c2, c3 = st.columns(3)
c1.metric("Cmax", f"{metrics['Cmax']:.2f} mg/L")
c2.metric("Tmax", f"{metrics['Tmax']:.2f} h")
c3.metric("AUC", f"{metrics['AUC']:.1f} mg·h/L")

# The key interaction: choose a moment on the hours-long body curve ...
body_time = st.slider("Whole-body time point (h)", 0.0, float(hours), min(1.0, float(hours)), 0.05)
left, mid, right = st.columns([1.3, 1.3, 1])
left.pyplot(viz.plot_blood_curve(pk, marker_time=body_time))
mid.pyplot(viz.plot_compartments(pk))
right.pyplot(viz.plot_body_view(pk, body_time))

if route == "oral":
    st.subheader("Fed vs. fasted")
    st.pyplot(viz.plot_scenario_comparison(comparison))

# ----------------------------------------------------------------- vessel
st.subheader(f"2 · Local vessel at body time {body_time:.2f} h (seconds)")
st.write(f"Inlet concentration from the PK model: **C_C = {pk.blood_concentration_at(body_time):.3f} mg/L**")
# ... and simulate the small vessel at that moment (its inlet = blood concentration then).
vres = run_vessel(drug_key, dose, route, meal, hours, body_time, duration, leaky)
frame = st.slider("Local vessel time (s)", 0.0, float(duration), float(duration) / 2, 0.25)
k = int(round(frame / duration * (len(vres.time) - 1)))   # slider time -> stored frame index
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(9, 2.8), constrained_layout=True)
im = viz.plot_vessel_frame(vres, k, ax)
fig.colorbar(im, ax=ax, label="c (mg/L)")
st.pyplot(fig)
st.caption(f"Mass-balance error of the vessel solver: {vres.mass_balance_error():.1e}")
