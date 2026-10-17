"""
visualization.py -- plots and animations for PKResult and VesselResult.

Author:         AI-generated draft (Claude, Anthropic) -- module owner: Person 4
Human reviewer: TODO (Person 4)

WHAT THIS FILE DOES (plain language)
------------------------------------
Turns simulation results into pictures:
    plot_compartments        drug amount in each body compartment over time
    plot_blood_curve         blood concentration curve with Cmax / Tmax / AUC
    plot_scenario_comparison fed vs fasted curves + table of metrics
    plot_body_view           cartoon of the body; each organ box is coloured by its drug level
    plot_vessel_frame /
    plot_vessel_snapshots    heat map of drug inside the vessel (darker green = more drug)
    plot_multiscale          the key figure: body curve -> chosen times -> vessel heat maps
    animate_vessel           movie of the vessel heat map

The Visualizer only consumes result objects; it contains no solver logic, so the UI
(app.py) and the report scripts can be developed against mock results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")      # draw to files/memory, no window needed (works on servers)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

import numpy as np  # noqa: E402

from .metrics import PKMetrics  # noqa: E402
from .pk_model import PKResult, COMPARTMENTS  # noqa: E402
from .vessel import VesselResult  # noqa: E402

# One fixed colour per compartment / scenario, so every figure uses the same colour code.
COLORS = dict(stomach="#d9822b", intestine="#b5651d", liver="#8e3b46",
              blood="#c0392b", tissue="#2e7d5b", fasted="#1f5fa8", fed="#d9822b")
SCENARIO_STYLES = ["-", "--", ":", "-."]


class Visualizer:
    """Figure factory. Every method returns a matplotlib Figure (and saves it if `save` is given)."""

    def __init__(self, out_dir: str | Path | None = None):
        self.out = Path(out_dir) if out_dir else None
        if self.out:
            self.out.mkdir(parents=True, exist_ok=True)

    def _finish(self, fig, save):
        """Save the figure to the output folder if a file name was given; always return it."""
        if save and self.out:
            fig.savefig(self.out / save, dpi=160)
        return fig

    # ================================================================ PK
    def plot_compartments(self, pk: PKResult, save: str | None = None):
        """Drug amount in each compartment and cumulative elimination vs time."""
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        for c in COMPARTMENTS:
            ax.plot(pk.time, pk.amounts[c], color=COLORS[c], lw=2, label=c.capitalize())
        elim = sum(pk.eliminated.values())          # liver + kidney + gut losses together
        ax.plot(pk.time, elim, color="0.4", lw=1.5, ls="--", label="eliminated (cumulative)")
        ax.set_xlabel("time (h)")
        ax.set_ylabel("drug amount (mg)")
        ax.set_title(f"{pk.drug_name}, {pk.scenario_label}: compartment amounts")
        ax.legend(fontsize=8, ncol=2)
        return self._finish(fig, save)

    def plot_blood_curve(self, pk: PKResult, ax=None, marker_time: float | None = None,
                         save: str | None = None):
        """Central blood concentration C_C(t) with Cmax/Tmax/AUC annotation."""
        own = ax is None
        if own:
            fig, ax = plt.subplots(figsize=(6, 3.8), constrained_layout=True)
        m = PKMetrics.from_result(pk)
        ax.plot(pk.time, pk.blood, color=COLORS["blood"], lw=2)
        ax.fill_between(pk.time, pk.blood, color=COLORS["blood"], alpha=0.08)
        ax.plot(m.tmax(), m.cmax(), "o", color="k", ms=5)
        ax.annotate(f"Cmax={m.cmax():.2f} mg/L\nTmax={m.tmax():.2f} h\nAUC={m.auc():.1f} mg·h/L",
                    xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round", fc="white", ec="0.8"))
        if marker_time is not None:                 # highlight the time chosen in the app
            cc = pk.blood_concentration_at(marker_time)
            ax.axvline(marker_time, color="0.3", ls=":", lw=1)
            ax.plot(marker_time, cc, "s", color="#1f5fa8", ms=7, zorder=5)
        ax.set_xlabel("time (h)")
        ax.set_ylabel("central blood conc. C_C (mg/L)")
        ax.set_title(f"{pk.drug_name}: {pk.scenario_label}")
        if own:
            return self._finish(ax.figure, save)
        return ax

    def plot_scenario_comparison(self, comparison: dict, save: str | None = None,
                                 show_table: bool = True):
        """Overlay C_C(t) for several scenarios (e.g. fed vs fasted), optionally with a metrics table."""
        if show_table:
            fig, (ax, axt) = plt.subplots(1, 2, figsize=(10, 3.8), width_ratios=[1.6, 1],
                                          constrained_layout=True)
        else:
            fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
            axt = None
        rows = []
        for k, (label, (pk, met)) in enumerate(comparison.items()):
            color = COLORS["fed"] if "fed" in label else COLORS["fasted"]
            ax.plot(pk.time, pk.blood, color=color, lw=2, ls=SCENARIO_STYLES[k % 4], label=label)
            rows.append([label, f"{met['Cmax']:.2f}", f"{met['Tmax']:.2f}", f"{met['AUC']:.1f}"])
        ax.set_xlabel("time (h)")
        ax.set_ylabel("C_C (mg/L)")
        ax.set_title(f"{pk.drug_name}: fed vs fasted")
        ax.legend(fontsize=8)
        if axt is None:                              # curve only (used in the report)
            return self._finish(fig, save)
        axt.axis("off")
        tab = axt.table(cellText=rows, colLabels=["scenario", "Cmax\n(mg/L)", "Tmax\n(h)", "AUC\n(mg·h/L)"],
                        loc="center", cellLoc="center")
        tab.auto_set_font_size(False)
        tab.set_fontsize(8)
        tab.scale(1, 1.8)
        return self._finish(fig, save)

    def plot_body_view(self, pk: PKResult, t_hours: float, ax=None, save: str | None = None):
        """Schematic body view: each compartment glows with its concentration at time t."""
        own = ax is None
        if own:
            fig, ax = plt.subplots(figsize=(4.2, 4.4), constrained_layout=True)
        st = pk.state_at(t_hours)                   # amounts in all five tanks at time t
        conc = dict(stomach=st.stomach, intestine=st.intestine)   # GI shown as amount (mg)
        for c in ("liver", "blood", "tissue"):
            conc[c] = float(np.interp(t_hours, pk.time, pk.concentrations[c]))
        peak = {c: max(np.max(pk.concentrations[c]) if c in pk.concentrations else np.max(pk.amounts[c]), 1e-12)
                for c in COMPARTMENTS}
        # Colour intensity = current level / highest level ever reached in that compartment.
        # (x, y) position of each organ box in the cartoon:
        boxes = dict(stomach=(0.1, 0.72), intestine=(0.1, 0.42), liver=(0.1, 0.12),
                     blood=(0.55, 0.42), tissue=(0.55, 0.72))
        cmap = plt.get_cmap("Reds")
        for c, (x0, y0) in boxes.items():
            level = conc[c] / peak[c]
            ax.add_patch(FancyBboxPatch((x0, y0), 0.35, 0.18, boxstyle="round,pad=0.02",
                                        fc=cmap(0.1 + 0.85 * level), ec="0.3"))
            unit = "mg" if c in ("stomach", "intestine") else "mg/L"
            ax.text(x0 + 0.175, y0 + 0.09, f"{c.capitalize()}\n{conc[c]:.2f} {unit}",
                    ha="center", va="center", fontsize=8,
                    color="white" if level > 0.6 else "black")
        arrows = [((0.275, 0.72), (0.275, 0.62)), ((0.275, 0.42), (0.275, 0.32)),
                  ((0.47, 0.21), (0.6, 0.40)), ((0.725, 0.62), (0.725, 0.70))]
        for a, b in arrows:
            ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="<|-|>" if a[0] > 0.4 else "-|>",
                                                            color="0.35"))
        ax.text(0.725, 0.33, "renal CL ↓", ha="center", fontsize=7, color="0.35")
        ax.set_xlim(0, 1)
        ax.set_ylim(0.05, 0.97)
        ax.axis("off")
        ax.set_title(f"Whole body at t = {t_hours:.1f} h", fontsize=10)
        if own:
            return self._finish(ax.figure, save)
        return ax

    # ============================================================ vessel
    def _vessel_extent(self, v: VesselResult):
        """[x_min, x_max, y_min, y_max] of the grid, so imshow draws cells at true positions."""
        dy = v.y[1] - v.y[0]
        return [0, v.x[-1] + 0.5 * (v.x[1] - v.x[0]), v.y[0] - 0.5 * dy, v.y[-1] + 0.5 * dy]

    def plot_vessel_frame(self, v: VesselResult, k: int, ax, vmax: float | None = None):
        """One concentration heat-map frame on `ax`; returns the image artist."""
        # Colour scale from 0 to the inlet concentration: full dark green = same as body blood.
        vmax = vmax or max(float(np.max(v.inlet_values)), 1e-12)
        im = ax.imshow(v.concentration[k], origin="lower", extent=self._vessel_extent(v),
                       aspect="auto", cmap="Greens", vmin=0, vmax=vmax, interpolation="bilinear")
        for yw in (-v.vessel_radius, v.vessel_radius):   # black lines = vessel walls
            ax.axhline(yw, color="k", lw=1.5)
        title = f"t = {v.time[k]:.1f} s"
        if v.body_time_h is not None:
            title = f"Local vessel at body time {v.body_time_h:g} h,  " + title
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x (mm) along vessel")
        ax.set_ylabel("y (mm)")
        return im

    def plot_vessel_snapshots(self, v: VesselResult, times=(0.5, 2, 4, 8), save: str | None = None):
        fig, axes = plt.subplots(len(times), 1, figsize=(7, 1.55 * len(times) + 0.5),
                                 constrained_layout=True, sharex=True)
        im = None
        for ax, t in zip(np.atleast_1d(axes), times):
            im = self.plot_vessel_frame(v, int(np.argmin(np.abs(v.time - t))), ax)
            ax.set_xlabel("")
        np.atleast_1d(axes)[-1].set_xlabel("x (mm) along vessel")
        fig.colorbar(im, ax=axes, label="drug concentration (mg/L)", shrink=0.9)
        return self._finish(fig, save)

    def plot_multiscale(self, pk: PKResult, vessels: list, save: str | None = None):
        """The key figure: PK curve with selected body times -> local vessel fields at those times."""
        n = len(vessels)
        fig = plt.figure(figsize=(11, 1.5 + 1.45 * n), constrained_layout=True)
        gs = fig.add_gridspec(n, 2, width_ratios=[1, 1.5])
        ax_pk = fig.add_subplot(gs[:, 0])
        self.plot_blood_curve(pk, ax=ax_pk)
        # Shared colour scale so the three vessel panels can be compared directly.
        vmax = max(float(np.max(v.inlet_values)) for v in vessels)
        im = None
        for i, v in enumerate(vessels):
            cc = pk.blood_concentration_at(v.body_time_h)
            ax_pk.axvline(v.body_time_h, color="0.4", ls=":", lw=1)
            ax_pk.plot(v.body_time_h, cc, "s", color="#1f5fa8", ms=6, zorder=5)
            ax_pk.annotate(f"{'ABCDEF'[i]}", (v.body_time_h, cc), xytext=(4, 4),
                           textcoords="offset points", fontsize=9, weight="bold", color="#1f5fa8")
            ax = fig.add_subplot(gs[i, 1])
            k = len(v.time) // 2                    # show the middle frame of each vessel run
            im = self.plot_vessel_frame(v, k, ax, vmax=vmax)
            ax.set_title(f"({'ABCDEF'[i]}) body time {v.body_time_h:g} h → inlet C_C = {cc:.2f} mg/L,"
                         f"  local t = {v.time[k]:.0f} s", fontsize=9)
            if i < n - 1:
                ax.set_xlabel("")
        fig.colorbar(im, ax=fig.axes[1:], label="c (mg/L)", shrink=0.9)
        return self._finish(fig, save)

    def animate_vessel(self, v: VesselResult, fname: str = "vessel.mp4", fps: int = 12):
        """Movie of c(x, y, t) in the local vessel (mp4 via ffmpeg, else gif)."""
        fig, ax = plt.subplots(figsize=(8, 2.8), constrained_layout=True)
        im = self.plot_vessel_frame(v, 0, ax)
        fig.colorbar(im, ax=ax, label="c (mg/L)")

        def update(k):                               # called once per movie frame
            im.set_data(v.concentration[k])
            ax.set_title(f"Local vessel at body time {v.body_time_h:g} h,  t = {v.time[k]:.1f} s"
                         if v.body_time_h is not None else f"t = {v.time[k]:.1f} s", fontsize=9)
            return (im,)

        anim = animation.FuncAnimation(fig, update, frames=len(v.time))
        path = (self.out or Path(".")) / fname
        # Prefer mp4 (needs the ffmpeg program); otherwise fall back to an animated gif.
        if path.suffix == ".mp4" and animation.writers.is_available("ffmpeg"):
            anim.save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=1600))
        else:
            path = path.with_suffix(".gif")
            anim.save(path, writer=animation.PillowWriter(fps=fps))
        plt.close(fig)
        return path
