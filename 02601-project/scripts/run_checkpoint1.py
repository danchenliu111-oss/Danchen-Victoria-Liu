"""
run_checkpoint1.py -- reproduce every figure and movie used in progress report 1.

Author:         AI-generated draft (Claude, Anthropic) -- owner: Person 4
Human reviewer: TODO (Person 4)

Usage (from the repository root):
    python scripts/run_checkpoint1.py [--drug acetaminophen] [--out docs/figures] [--no-movie]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import Simulation, VesselSettings  # noqa: E402
from src.visualization import Visualizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drug", default="acetaminophen")
    ap.add_argument("--out", default=str(ROOT / "docs" / "figures"))
    ap.add_argument("--no-movie", action="store_true")
    args = ap.parse_args()
    viz = Visualizer(args.out)          # all figures are written to this folder

    # 1) whole-body PK, oral fasted
    sim = Simulation.from_names(args.drug, "fasted", "oral")
    pk = sim.run_pk()
    print(f"{pk.drug_name} {pk.scenario_label}: {sim.pk_metrics()}  mass error {pk.mass_balance_error():.1e}")
    viz.plot_compartments(pk, save="pk_compartments.png")
    viz.plot_blood_curve(pk, save="pk_blood_curve.png")

    # 2) fed vs fasted, oral and IV
    comp = sim.compare_scenarios(("fasted", "fed"), ("oral",))
    for k, (_, m) in comp.items():
        print(f"  {k:14s} Cmax={m['Cmax']:.2f} mg/L  Tmax={m['Tmax']:.2f} h  AUC={m['AUC']:.1f} mg h/L")
    viz.plot_scenario_comparison(comp, save="fed_vs_fasted.png")
    viz.plot_scenario_comparison(comp, save="fed_vs_fasted_curve.png", show_table=False)

    # 3) ODE -> PDE coupling at three whole-body times
    vessels = [sim.run_vessel(tb) for tb in (0.25, 1.0, 6.0)]
    for v in vessels:
        print(f"  vessel @ {v.body_time_h} h: inlet {v.inlet_values[0]:.3f} mg/L, "
              f"mass error {v.mass_balance_error():.1e}")
    viz.plot_multiscale(pk, vessels, save="multiscale_coupling.png")
    viz.plot_vessel_snapshots(vessels[1], times=(0.5, 2, 4, 8), save="vessel_snapshots.png")
    viz.plot_body_view(pk, 1.0, save="body_view.png")

    # 4) stretch-goal prototype: leaky wall + tissue layer.
    #    We reuse the PK result already computed above instead of solving it again.
    leaky = Simulation(sim.drug, sim.physiology, sim.scenario,
                       vessel=VesselSettings(duration_s=30, n_frames=121, wall_permeability=2e-3,
                                             tissue_thickness=0.25, tissue_uptake=0.0))
    leaky._pk_result = pk
    vl = leaky.run_vessel(1.0)
    viz.plot_vessel_snapshots(vl, times=(2, 8, 16, 30), save="stretch_leaky_wall.png")

    if not args.no_movie:
        print("saved", viz.animate_vessel(vessels[1], "vessel_1h.mp4"))
        print("saved", viz.animate_vessel(vl, "stretch_leaky_wall.mp4"))


if __name__ == "__main__":
    main()
