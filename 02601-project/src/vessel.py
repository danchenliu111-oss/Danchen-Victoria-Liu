"""
vessel.py -- local blood-vessel model: 2-D advection-diffusion PDE.

Author:         AI-generated draft (Claude, Anthropic) -- module owner: Person 2
Human reviewer: TODO (Person 2)

WHAT THIS FILE DOES (plain language)
------------------------------------
We zoom into ONE small blood vessel (0.2 mm wide, 10 mm long) and ask: where exactly is
the drug inside it, second by second? Two things move the drug:

  1. Advection = being carried along by the flowing blood (like a leaf on a river).
     Blood flows fastest in the centre of the vessel and is almost still at the wall
     (a parabolic "Poiseuille" profile), so drug in the centre races ahead.
  2. Diffusion = random molecular jiggling that spreads drug from crowded to empty regions
     (like a drop of ink slowly spreading in still water).

Because concentration now depends on position (x, y) AND time, the equation is a
"partial differential equation" (PDE):

    dc/dt + u(y) dc/dx = D (d2c/dx2 + d2c/dy2)
    (change)  (carried by flow)  (spreading by diffusion)

    u(y) = 2 u_mean (1 - (y/R)^2)   -- speed profile of blood in a round tube

Boundary conditions (what happens at the edges):
    inlet  x = 0  : blood arrives with concentration C_in(t)  <-- comes from the PK model!
    outlet x = L  : blood simply leaves ("zero-gradient")
    walls  y = +-R: drug cannot pass through the wall (core model)

HOW THE COMPUTER SOLVES IT (finite-volume method)
-------------------------------------------------
The vessel is cut into a grid of small rectangular cells (e.g. 200 x 24). Each cell stores
one average concentration. Every time step we compute how much drug crosses each cell face
(the "flux") and update every cell:   new = old + (flux in - flux out) * dt / cell size.
Because whatever leaves one cell enters its neighbour, no drug is ever created or lost
by mistake ("mass conservation") -- our tests check this to ~1e-14.

    * Advection flux uses the MUSCL scheme with a van Leer "limiter": a smart way to estimate
      the concentration at a cell face that is accurate yet never creates fake wiggles.
    * Diffusion flux = D * (difference between neighbours) / distance.
    * Time stepping: SSP-RK2 ("Heun's method") = take a trial step, then average it with
      a corrected step; more accurate than one simple step.
    * Stability: the time step must be small enough that drug never jumps more than one
      cell per step:   dt = CFL * min( dx / u_max ,  1 / (2 D (1/dx^2 + 1/dy^2)) ).

Stretch goal (prototyped, OFF by default): a leaky wall with flux J = P (c_lumen - c_tissue)
and a layer of tissue around the vessel in which drug diffuses and is taken up by cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Union

import numpy as np

# The inlet can be a fixed number or a function of time (seconds) -> concentration.
Inlet = Union[float, Callable[[float], float]]


def van_leer(r: np.ndarray) -> np.ndarray:
    """Van Leer limiter phi(r) = (r + |r|) / (1 + |r|).

    r compares the slope on one side of a cell with the slope on the other side.
    Smooth region (r ~ 1)  -> phi ~ 1  : use the accurate 2nd-order estimate.
    Peak / sharp front (r <= 0) -> phi = 0 : fall back to the safe 1st-order estimate,
    which prevents unphysical over- and undershoots.
    """
    return (r + np.abs(r)) / (1.0 + np.abs(r))


@dataclass
class VesselResult:
    """What a vessel simulation produced.

    time          : frame times [s]
    x, y          : cell-centre coordinates [mm] (y = 0 is the vessel centre line)
    concentration : array [frame, y, x] of drug concentration [mg/L]
    mass_*        : bookkeeping used to prove that no drug was lost numerically
    """

    time: np.ndarray
    x: np.ndarray
    y: np.ndarray
    concentration: np.ndarray
    lumen_rows: np.ndarray              # True for grid rows inside the vessel (not tissue)
    vessel_radius: float
    mass_in: np.ndarray = field(default=None)       # total drug that entered (cumulative)
    mass_out: np.ndarray = field(default=None)      # total drug that left at the outlet
    mass_eliminated: np.ndarray = field(default=None)  # taken up by tissue (stretch goal)
    mass_stored: np.ndarray = field(default=None)   # drug currently inside the domain
    inlet_values: np.ndarray = field(default=None)  # inlet concentration at each frame
    dt: float = 0.0
    body_time_h: float | None = None    # which whole-body time this local run represents

    def mass_balance_error(self) -> float:
        """Relative error of: in - out - eliminated = stored (should be ~0)."""
        r = self.mass_in[-1] - self.mass_out[-1] - self.mass_eliminated[-1] - self.mass_stored[-1]
        return float(abs(r) / max(self.mass_in[-1], 1e-30))

    def mean_lumen_concentration(self) -> np.ndarray:
        """Average concentration inside the vessel for every frame [mg/L]."""
        return self.concentration[:, self.lumen_rows, :].mean(axis=(1, 2))


class VesselModel:
    """Finite-volume solver for drug transport in one local vessel segment."""

    def __init__(self, diffusion_coefficient: float, vessel_length: float, vessel_radius: float,
                 blood_velocity: float, nx: int = 200, ny: int = 24, cfl: float = 0.4,
                 wall_permeability: float = 0.0, tissue_thickness: float = 0.0,
                 tissue_diffusion: float | None = None, tissue_uptake: float = 0.0):
        """
        Parameters
        ----------
        diffusion_coefficient : D, how fast the drug spreads by diffusion [mm^2/s]
        vessel_length, vessel_radius : L and R [mm]
        blood_velocity : average blood speed u_mean [mm/s]
        nx, ny : number of grid cells along the vessel and across its diameter
        cfl    : safety factor (< 1) applied to the largest stable time step
        wall_permeability, tissue_thickness, tissue_diffusion, tissue_uptake :
            stretch-goal options; with the defaults the wall is sealed and there is no tissue.
        """
        # --- physical parameters
        self.D = diffusion_coefficient
        self.L = vessel_length
        self.R = vessel_radius
        self.u_mean = blood_velocity
        self.P = wall_permeability
        # diffusion is slower in crowded tissue; default guess: 30 % of the value in blood
        self.D_t = tissue_diffusion if tissue_diffusion is not None else 0.3 * diffusion_coefficient
        self.k_t = tissue_uptake

        # --- build the grid
        self.nx, self.ny_lumen = int(nx), int(ny)
        self.dx = self.L / self.nx                    # cell length along the vessel
        self.dy = 2.0 * self.R / self.ny_lumen        # cell height across the vessel
        # optional tissue rows above and below the vessel (0 rows in the core model)
        self.n_tissue = int(round(tissue_thickness / self.dy)) if tissue_thickness > 0 else 0
        self.ny = self.ny_lumen + 2 * self.n_tissue

        # cell-centre coordinates; y = 0 is the centre line of the vessel
        self.x = (np.arange(self.nx) + 0.5) * self.dx
        self.y = (np.arange(self.ny) + 0.5) * self.dy - (self.R + self.n_tissue * self.dy)
        self.lumen_rows = np.abs(self.y) < self.R     # True = inside the vessel ("lumen")

        # --- per-row material properties (everything is uniform along x)
        self.D_row = np.where(self.lumen_rows, self.D, self.D_t)       # diffusion coefficient
        self.k_row = np.where(self.lumen_rows, 0.0, self.k_t)         # uptake only in tissue
        self.u_row = self.velocity_field()[:, 0]                       # blood speed per row
        self.G_y = self._face_conductances()                           # see method below
        self.dt = self.stable_time_step(cfl)

    # ---------------------------------------------------------------- set-up
    def velocity_field(self) -> np.ndarray:
        """Blood speed u [mm/s] in every cell, shape (ny, nx).

        Parabolic profile of a round tube: 2*u_mean on the centre line, 0 at the wall,
        and 0 everywhere in the (optional) tissue.
        """
        eta = self.y / self.R                          # -1 at one wall, 0 at centre, +1 at other wall
        u = np.where(np.abs(eta) < 1.0, 2.0 * self.u_mean * (1.0 - eta ** 2), 0.0)
        return np.repeat(u[:, None], self.nx, axis=1)  # same profile at every x position

    def _face_conductances(self) -> np.ndarray:
        """How easily drug diffuses through each horizontal cell face [mm/s].

        Like resistors in series: half a cell of blood + (wall) + half a cell of tissue.
        The vertical diffusive flux across a face is  -G * (c_above - c_below).
        A sealed wall has infinite resistance -> G = 0 -> no drug crosses.
        """
        G = np.zeros(self.ny + 1)                       # ny cells have ny+1 faces
        h = 0.5 * self.dy                               # distance from cell centre to face
        for j in range(1, self.ny):                     # interior faces only
            resistance = h / self.D_row[j - 1] + h / self.D_row[j]
            if self.lumen_rows[j - 1] != self.lumen_rows[j]:          # this face IS the wall
                resistance = resistance + (1.0 / self.P if self.P > 0 else np.inf)
            G[j] = 0.0 if np.isinf(resistance) else 1.0 / resistance
        return G                                        # outermost faces stay 0: no flux

    def stable_time_step(self, cfl: float = 0.4) -> float:
        """Largest time step [s] for which the explicit scheme stays stable, times `cfl`.

        dt_adv : time for the fastest blood to cross one cell (advection limit)
        dt_diff: time for diffusion to smooth one cell (diffusion limit)
        """
        u_max = max(2.0 * self.u_mean, 1e-12)
        D_max = float(np.max(self.D_row))
        dt_adv = self.dx / u_max
        dt_diff = 1.0 / (2.0 * D_max * (1.0 / self.dx ** 2 + 1.0 / self.dy ** 2))
        return cfl * min(dt_adv, dt_diff)

    @staticmethod
    def _as_function(inlet: Inlet) -> Callable[[float], float]:
        """Turn a constant inlet value into a function of time, so both cases look the same."""
        return inlet if callable(inlet) else (lambda t, v=float(inlet): v)

    # ------------------------------------------------------- right-hand side
    def _rhs(self, c: np.ndarray, c_in: float):
        """Rate of change dc/dt in every cell, plus boundary bookkeeping.

        Returns (dcdt, rates) where rates = [inflow, outflow, eliminated] per second.
        """
        ny, nx, dx, dy = self.ny, self.nx, self.dx, self.dy

        # ---- 1. Pad the grid with two "ghost" columns on each side.
        #      Left ghosts = inlet concentration (the drug arriving from the body).
        #      Right ghosts = copy of the last column (drug just flows out).
        ext = np.empty((ny, nx + 4))
        ext[:, 2:-2] = c
        ext[:, :2] = c_in
        ext[:, -2:] = c[:, -1:]

        # ---- 2. Advective flux through every vertical face (nx+1 faces).
        # Blood flows left -> right, so the "upwind" cell (where drug comes from) is on the left.
        c_up, c_upup, c_down = ext[:, 1:-2], ext[:, 0:-3], ext[:, 2:-1]
        den = c_up - c_upup                                  # slope behind the face
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(np.abs(den) > 1e-14, (c_down - c_up) / den, 0.0)   # slope ratio
        c_face = c_up + 0.5 * van_leer(r) * den              # limited face concentration
        Fx = self.u_row[:, None] * c_face                    # flux = speed * concentration

        # ---- 3. Add axial diffusion between neighbouring columns (none through inlet/outlet).
        Fx[:, 1:-1] -= self.D_row[:, None] * (c[:, 1:] - c[:, :-1]) / dx

        # ---- 4. Transverse diffusion through every horizontal face (and the wall, if leaky).
        Fy = np.zeros((ny + 1, nx))
        Fy[1:-1] = -self.G_y[1:-1, None] * (c[1:] - c[:-1])

        # ---- 5. Net change per cell = (in - out) / cell size, minus uptake in tissue.
        dcdt = -(Fx[:, 1:] - Fx[:, :-1]) / dx - (Fy[1:] - Fy[:-1]) / dy - self.k_row[:, None] * c

        # ---- 6. Bookkeeping for the mass-balance check.
        rates = np.array([Fx[:, 0].sum() * dy,               # entering at the inlet
                          Fx[:, -1].sum() * dy,              # leaving at the outlet
                          (self.k_row[:, None] * c).sum() * dx * dy])   # taken up by tissue
        return dcdt, rates

    def step(self, concentration: np.ndarray, inlet: float, t: float = 0.0,
             inlet_next: float | None = None):
        """Advance one time step with SSP-RK2 (Heun's method).

        1. Euler trial step:     c1 = c + dt * f(c)
        2. second Euler step from c1, then average with the start: c_new = (c + c1 + dt*f(c1)) / 2
        Returns (new concentration, mass moved across boundaries during this step).
        """
        inlet_next = inlet if inlet_next is None else inlet_next
        k1, r1 = self._rhs(concentration, inlet)
        c1 = concentration + self.dt * k1
        k2, r2 = self._rhs(c1, inlet_next)
        c_new = 0.5 * (concentration + c1 + self.dt * k2)
        return c_new, 0.5 * (r1 + r2) * self.dt

    def solve(self, inlet_concentration: Inlet, duration: float, n_frames: int = 121,
              initial: str = "empty") -> VesselResult:
        """Simulate `duration` seconds and store `n_frames` evenly spaced snapshots.

        inlet_concentration : constant C_in [mg/L] or a function C_in(t_seconds)
        initial             : "empty"  -> vessel starts drug-free, we watch the drug wash in;
                              "filled" -> vessel starts already at the inlet concentration.
        """
        cin = self._as_function(inlet_concentration)
        c = np.zeros((self.ny, self.nx))                 # concentration in every cell
        if initial == "filled":
            c[self.lumen_rows, :] = cin(0.0)

        frame_t = np.linspace(0.0, duration, n_frames)   # times at which we save a snapshot
        frames = np.empty((n_frames, self.ny, self.nx))
        led = np.zeros((n_frames, 3))                    # cumulative in / out / eliminated
        stored = np.zeros(n_frames)                      # drug inside the domain per frame
        totals = np.zeros(3)
        cell = self.dx * self.dy                         # area of one cell
        stored0 = c.sum() * cell                         # drug present at the start
        frames[0], stored[0] = c, stored0

        # Main time loop: keep stepping until every frame has been recorded.
        t, k = 0.0, 1
        while k < n_frames:
            c, r = self.step(c, cin(t), t, cin(t + self.dt))
            totals += r
            t += self.dt
            # store a snapshot whenever we pass the next frame time
            while k < n_frames and t >= frame_t[k] - 1e-12:
                frames[k], led[k], stored[k] = c, totals, c.sum() * cell
                k += 1

        return VesselResult(time=frame_t, x=self.x, y=self.y, concentration=frames,
                            lumen_rows=self.lumen_rows, vessel_radius=self.R,
                            mass_in=led[:, 0] + stored0, mass_out=led[:, 1],
                            mass_eliminated=led[:, 2], mass_stored=stored,
                            inlet_values=np.array([cin(tt) for tt in frame_t]), dt=self.dt)

    def peclet_number(self) -> float:
        """Peclet number = u_mean R / D: how much stronger flow is than diffusion.

        Pe >> 1 (ours is ~400) means the drug is carried along far faster than it spreads
        sideways, which is why the drug front in the movie looks like a stretched bullet.
        """
        return self.u_mean * self.R / self.D
