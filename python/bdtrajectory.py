"""
bdtrajectory.py — Python interface for GPUParticles HDF5 trajectory files.

The BDTrajectory class reads the HDF5 files written by the simulation engine
and exposes the trajectory data through a clean, Pythonic API.  All
visualisation helpers live on the class so a single import covers both data
access and plotting.

HDF5 layout (produced by TrajectoryWriter):

    /                        root group
      attrs: N, gamma, phi, sigma, epsilon, kT, L, dt,
             n_steps, output_every, output_file,
             init_mode, seed, r_skin, verlet_check_every,
             f0, tau_theta, max_drift, k_a, gamma_a
             (D = kT/gamma is derived, not stored)

    /frame_00000000          one group per output frame
      attrs: step (uint64)   simulation step number
             time (float64)  physical time  = step * dt
             frame (uint64)  sequential frame index (0-based)
             Lx, Ly (float64) box dimensions
      dset: positions        shape (N, 2), float64, columns = [x, y]
      dset: velocities       shape (N, 2), float64, columns = [vx, vy]
      dset: orientations     shape (N,),   float64, theta in radians
      dset: anchors          shape (N, 2), float64, columns = [ax, ay]
      dset: forces           shape (N, 2), float64, columns = [fx, fy]
                             — net pair-interaction force on each particle.
                             Active and spring contributions are NOT included.

    /frame_00000001
      ...

Usage
-----
>>> traj = BDTrajectory("trajectory.h5")
>>> traj.plot_frame(0)
>>> plt.show()

>>> with BDTrajectory("trajectory.h5") as traj:
...     for fd in traj:
...         print(fd.step, fd.positions.mean(axis=0))
"""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

import h5py
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection

from cluster_utils import ClusterUtils

# ---------------------------------------------------------------------------
# Named tuple for per-frame data returned by iteration
# ---------------------------------------------------------------------------
FrameData = namedtuple(
    "FrameData",
    ["frame", "step", "time", "positions", "velocities", "orientations", "anchors", "forces"],
)


# ---------------------------------------------------------------------------
# BDTrajectory
# ---------------------------------------------------------------------------
class BDTrajectory:
    """
    Read-only interface to a GPUParticles Brownian-dynamics trajectory.

    Parameters
    ----------
    path : str or Path
        Path to the HDF5 trajectory file.

    Examples
    --------
    >>> traj = BDTrajectory("trajectory.h5")
    >>> print(traj)
    BDTrajectory('trajectory.h5', N=256, frames=101, sigma=1.0, L=22.42)

    >>> pos = traj.positions(0)        # (N, 2) array for frame 0
    >>> vel = traj.velocities(0)       # (N, 2) array or None for passive runs
    >>> ori = traj.orientations(0)     # (N,)   array or None for passive runs
    >>> traj.plot_frame(50)            # render frame 50
    >>> plt.show()
    """

    # ------------------------------------------------------------------
    # Construction / lifecycle
    # ------------------------------------------------------------------

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {self._path}")

        self._file = h5py.File(self._path, "r")

        # Cache root attributes as plain Python scalars / strings.
        self._params: dict = {}
        for k, v in self._file.attrs.items():
            val = v
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = float(val)
            elif isinstance(val, (bytes, np.bytes_)):
                val = val.decode()
            elif isinstance(val, np.ndarray) and val.ndim == 0:
                val = val.item()
            self._params[k] = val

        # Sorted list of frame group names  (frame_00000000, …)
        self._frame_names: list[str] = sorted(
            k for k in self._file.keys() if k.startswith("frame_")
        )

    def __repr__(self) -> str:
        active = f", f0={self.f0:.3g}" if self.f0 > 0 else ""
        return (
            f"BDTrajectory('{self._path.name}', "
            f"N={self.N}, frames={self.num_frames}, "
            f"sigma={self.sigma}, L={self.Lx:.4g}{active})"
        )

    def __len__(self) -> int:
        return self.num_frames

    def __iter__(self) -> Iterator[FrameData]:
        """Yield a FrameData namedtuple for every frame in order."""
        for i in range(self.num_frames):
            yield FrameData(
                frame=i,
                step=self.step(i),
                time=self.time(i),
                positions=self.positions(i),
                velocities=self.velocities(i),
                orientations=self.orientations(i),
                anchors=self.anchors(i),
                forces=self.forces(i),
            )

    def __getitem__(self, idx: int) -> np.ndarray:
        """Return positions array for frame *idx* (supports negative indexing)."""
        if idx < 0:
            idx = self.num_frames + idx
        return self.positions(idx)

    def __enter__(self) -> "BDTrajectory":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HDF5 file handle."""
        if self._file.id.valid:
            self._file.close()

    # ------------------------------------------------------------------
    # Simulation-parameter properties
    # ------------------------------------------------------------------

    @property
    def params(self) -> dict:
        """All root-group attributes as a plain dict."""
        return dict(self._params)

    @property
    def N(self) -> int:
        """Number of particles."""
        return int(self._params["N"])

    @property
    def sigma(self) -> float:
        """Particle diameter."""
        return float(self._params.get("sigma", 1.0))

    @property
    def Lx(self) -> float:
        """Box width (x).  Uses root attribute L for a square box."""
        return float(self._params.get("L", self._params.get("Lx", 1.0)))

    @property
    def Ly(self) -> float:
        """Box height (y).  Uses root attribute L for a square box."""
        return float(self._params.get("L", self._params.get("Ly", 1.0)))

    @property
    def dt(self) -> float:
        """Integration timestep."""
        return float(self._params.get("dt", 0.0))

    @property
    def gamma(self) -> float:
        """Particle friction coefficient (input parameter)."""
        return float(self._params["gamma"])

    @property
    def kT(self) -> float:
        """Thermal energy."""
        return float(self._params.get("kT", 1.0))

    @property
    def D(self) -> float:
        """Diffusion coefficient (derived via Einstein relation: D = kT/gamma)."""
        return self.kT / self.gamma

    @property
    def f0(self) -> float:
        """Active self-propulsion force magnitude (0 for passive runs)."""
        return float(self._params.get("f0", 0.0))

    @property
    def tau_theta(self) -> float:
        """Orientational persistence time (0 = no rotational diffusion)."""
        return float(self._params.get("tau_theta", 0.0))

    @property
    def is_active(self) -> bool:
        """True if f0 > 0 (active Brownian motion)."""
        return self.f0 > 0.0

    @property
    def max_drift(self) -> float:
        """Drift-displacement cap per step (0.0 = disabled)."""
        return float(self._params.get("max_drift", 0.0))

    @property
    def k_a(self) -> float:
        """Anchor spring stiffness (0.0 = no coupling)."""
        return float(self._params.get("k_a", 0.0))

    @property
    def gamma_a(self) -> float:
        """Anchor friction coefficient (0.0 = anchors frozen)."""
        return float(self._params.get("gamma_a", 0.0))

    @property
    def has_anchors(self) -> bool:
        """True if k_a > 0 (particles tethered to anchors)."""
        return self.k_a > 0.0

    @property
    def num_frames(self) -> int:
        """Total number of frames in the trajectory."""
        return len(self._frame_names)

    # ------------------------------------------------------------------
    # Per-frame data access
    # ------------------------------------------------------------------

    def _grp(self, frame: int) -> h5py.Group:
        if frame < 0:
            frame = self.num_frames + frame
        if not 0 <= frame < self.num_frames:
            raise IndexError(
                f"Frame {frame} out of range [0, {self.num_frames})"
            )
        return self._file[self._frame_names[frame]]

    def positions(self, frame: int) -> np.ndarray:
        """
        Return particle positions for one frame.

        Returns
        -------
        np.ndarray, shape (N, 2), dtype float64
            Column 0 = x, column 1 = y.
        """
        return self._grp(frame)["positions"][:]

    def velocities(self, frame: int) -> Optional[np.ndarray]:
        """
        Return instantaneous velocities (Delta_r/dt) for one frame.

        Returns None for trajectory files written before ABP support.

        Returns
        -------
        np.ndarray, shape (N, 2), dtype float64, or None
            Column 0 = vx, column 1 = vy.
        """
        grp = self._grp(frame)
        if "velocities" not in grp:
            return None
        return grp["velocities"][:]

    def orientations(self, frame: int) -> Optional[np.ndarray]:
        """
        Return orientation angles theta for one frame (radians).

        Returns None for trajectory files written before ABP support.

        Returns
        -------
        np.ndarray, shape (N,), dtype float64, or None
        """
        grp = self._grp(frame)
        if "orientations" not in grp:
            return None
        return grp["orientations"][:]

    def anchors(self, frame: int) -> Optional[np.ndarray]:
        """
        Return per-particle anchor positions for one frame.

        Returns None for trajectory files written before anchor support.

        Returns
        -------
        np.ndarray, shape (N, 2), dtype float64, or None
            Column 0 = ax, column 1 = ay.
        """
        grp = self._grp(frame)
        if "anchors" not in grp:
            return None
        return grp["anchors"][:]

    def forces(self, frame: int) -> Optional[np.ndarray]:
        """
        Return net pair-interaction forces on each particle for one frame.

        These are the forces written by ForceCalculator::compute on the C++
        side — i.e. only the pair-potential contribution. Active drive
        (f0) and anchor-spring forces are evaluated inside the integrator
        step and are NOT included here.

        Returns None for trajectory files written before forces were
        added to the writer.

        Returns
        -------
        np.ndarray, shape (N, 2), dtype float64, or None
            Column 0 = fx, column 1 = fy.
        """
        grp = self._grp(frame)
        if "forces" not in grp:
            return None
        return grp["forces"][:]

    def step(self, frame: int) -> int:
        """Simulation step number for a given frame index."""
        return int(self._grp(frame).attrs["step"])

    def time(self, frame: int) -> float:
        """Physical time for a given frame index."""
        return float(self._grp(frame).attrs["time"])

    def frame_box(self, frame: int) -> tuple[float, float]:
        """Return (Lx, Ly) stored in the frame group (constant for rigid box)."""
        g = self._grp(frame)
        return float(g.attrs["Lx"]), float(g.attrs["Ly"])

    def all_steps(self) -> np.ndarray:
        """1-D array of simulation step numbers across all frames."""
        return np.array([self.step(i) for i in range(self.num_frames)], dtype=np.int64)

    def all_times(self) -> np.ndarray:
        """1-D array of physical times across all frames."""
        return np.array([self.time(i) for i in range(self.num_frames)])

    # ------------------------------------------------------------------
    # Derived observables
    # ------------------------------------------------------------------

    def mean_squared_displacement(
        self, reference_frame: int = 0
    ) -> np.ndarray:
        """
        Compute ensemble-averaged MSD relative to *reference_frame*.

        Returns a 1-D array of length *num_frames* where entry i is

            MSD(i) = < |r_i - r_ref|^2 >_particles

        Trajectory positions are stored unwrapped by the engine, so the
        naive squared difference is the true displacement at any lag time
        (no minimum-image correction needed).
        """
        ref = self.positions(reference_frame)
        msd = np.empty(self.num_frames)
        for i in range(self.num_frames):
            delta = self.positions(i) - ref
            msd[i] = np.mean(np.sum(delta**2, axis=1))
        return msd

    def mean_speed(self) -> np.ndarray:
        """
        Ensemble-averaged instantaneous speed |v| across all frames.

        Returns a 1-D array of length *num_frames*.  Returns None for
        files without a 'velocities' dataset.
        """
        v0 = self.velocities(0)
        if v0 is None:
            return None
        speeds = np.empty(self.num_frames)
        for i in range(self.num_frames):
            v = self.velocities(i)
            speeds[i] = np.mean(np.sqrt(v[:, 0]**2 + v[:, 1]**2))
        return speeds
    
    def max_cluster_fraction(self, sigma_frac : float = 0.95) -> np.ndarray:
        """max_cluster_fraction
        
        Computes the fraction of the system in the largest cluster for each frame, using the cluster identification algorithm in cluster_utils.py with a distance threshold of sigma_frac * sigma.

        Returns
        -------
        np.ndarray
            _description_
        """
        sigma = self.sigma
        threshold = sigma_frac * sigma
        fractions = np.empty(self.num_frames)
        for i in range(self.num_frames):
            x = self.positions(i)[:, 0]
            y = self.positions(i)[:, 1]
            labels, cluster_sizes = ClusterUtils.find_clusters(x, y, threshold, self.frame_box(i)[0])
            fractions[i] = np.max(cluster_sizes) / len(x)
        return fractions

    # ------------------------------------------------------------------
    # Visualisation: single frame
    # ------------------------------------------------------------------

    def plot_frame(
        self,
        frame_idx: int = 0,
        *,
        ax: Optional[plt.Axes] = None,
        figsize: tuple[float, float] = (6, 6),
        particle_color: Union[str, tuple] = "steelblue",
        edgecolor: Union[str, tuple] = "navy",
        alpha: float = 0.85,
        linewidth: float = 0.5,
        show_box: bool = True,
        box_color: Union[str, tuple] = "black",
        box_linewidth: float = 1.5,
        title: Optional[str] = None,
        show_axes: bool = False,
        show_orientations: bool = False,
        orientation_color: Union[str, tuple] = "white",
        orientation_scale: float = 0.4,
        show_clusters: bool = False,
        cluster_size_cmap: Optional[plt.Colormap] = None,
        clus_sigma_frac: float = 1.05,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Render one simulation frame as filled circles on a 2-D plane.

        Each circle has radius ``sigma / 2`` so that touching circles just
        make contact at the particle surface.

        Parameters
        ----------
        frame_idx : int
            Frame index to visualise (0-based).
        ax : Axes, optional
            Existing axes to draw on.  A new figure is created when *None*.
        figsize : (float, float)
            Figure size in inches (ignored when *ax* is supplied).
        particle_color : color
            Matplotlib color spec for the circle fill.
        edgecolor : color
            Circle edge colour.
        alpha : float
            Circle opacity in [0, 1].
        linewidth : float
            Circle edge line width in points.
        show_box : bool
            Draw the periodic-cell boundary rectangle.
        box_color : color
            Colour of the cell boundary line.
        box_linewidth : float
            Line width of the cell boundary in points.
        title : str, optional
            Axes title.  Defaults to ``"frame {i}  step {s}  t = {t:.4g}"``.
        show_axes : bool
            Show axis ticks and labels (default False for a cleaner snapshot).
        show_orientations : bool
            Draw an arrow on each particle indicating its orientation theta.
            Only drawn when the 'orientations' dataset exists in the file.
        orientation_color : color
            Colour of orientation arrows.
        orientation_scale : float
            Arrow length as a fraction of sigma/2.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax  : matplotlib.axes.Axes
        """
        pos = self.positions(frame_idx)
        Lx, Ly = self.frame_box(frame_idx)
        r = self.sigma / 2.0

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()            
            
        # determine clusters + colors if requested
        if show_clusters:
            x = pos[:, 0]
            y = pos[:, 1]
            
            # replace in box
            x = np.mod(x, Lx)
            y = np.mod(y, Ly)
            pos[:, 0] = x
            pos[:, 1] = y

            
            # color particles by cluster size
            MAX_CLUSTER_SIZE = x.shape[0]  # max cluster size is total number of particles
            
            # get colormap
            if cluster_size_cmap is None:
                cluster_size_cmap = plt.get_cmap("inferno")
            else:
                cluster_size_cmap = plt.get_cmap(cluster_size_cmap)
                                
            # cluster size norm (logarithmic, since cluster sizes can vary widely)                                
            cluster_size_norm = mcolors.LogNorm(vmin=1, vmax=MAX_CLUSTER_SIZE)
            
            # compute cluster sizes using utility function
            labels, cluster_sizes = ClusterUtils.find_clusters(x, y, clus_sigma_frac * self.sigma, Lx)
            
            # get particle colors based on cluster size
            particle_color = cluster_size_cmap(cluster_size_norm(cluster_sizes[labels]))                        

        # One Circle patch per particle, rendered as a single PatchCollection
        # for efficient GPU-accelerated drawing even at large N.
        circles = [mpatches.Circle(pos[i], radius=r) for i in range(self.N)]
        col = PatchCollection(
            circles,
            facecolor=particle_color,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=linewidth,
        )
        ax.add_collection(col)

        # Orientation arrows (only when requested and dataset is present).
        if show_orientations:
            theta = self.orientations(frame_idx)
            if theta is not None:
                arrow_len = r * orientation_scale
                for i in range(self.N):
                    ax.annotate(
                        "",
                        xy=(pos[i, 0] + arrow_len * np.cos(theta[i]),
                            pos[i, 1] + arrow_len * np.sin(theta[i])),
                        xytext=(pos[i, 0], pos[i, 1]),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color=orientation_color,
                            lw=0.6,
                        ),
                    )

        if show_box:
            ax.add_patch(
                mpatches.Rectangle(
                    (0, 0), Lx, Ly,
                    fill=False,
                    edgecolor=box_color,
                    linewidth=box_linewidth,
                )
            )

        # Axes limits: one radius of margin so boundary circles aren't clipped.
        ax.set_xlim(-r, Lx + r)
        ax.set_ylim(-r, Ly + r)
        ax.set_aspect("equal")

        if not show_axes:
            ax.set_axis_off()

        if title is None:
            s = self.step(frame_idx)
            t = self.time(frame_idx)
            title = f"frame {frame_idx}   step {s}   t = {t:.4g}"
        ax.set_title(title, fontsize=10)
        
        
        if show_clusters:
            # add colorbar
            sm = plt.cm.ScalarMappable(cmap=cluster_size_cmap, norm=cluster_size_norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label="Cluster size")

        return fig, ax

    # ------------------------------------------------------------------
    # Visualisation: grid of frames
    # ------------------------------------------------------------------

    def plot_frames_grid(
        self,
        frame_indices: Optional[Sequence[int]] = None,
        *,
        ncols: int = 3,
        subplot_size: float = 3.5,
        **plot_frame_kwargs,
    ) -> tuple[plt.Figure, np.ndarray]:
        """
        Plot multiple frames arranged in a grid.

        Parameters
        ----------
        frame_indices : sequence of int, optional
            Which frames to plot.  Defaults to evenly spaced across the
            whole trajectory (up to 9 frames).
        ncols : int
            Number of columns in the grid.
        subplot_size : float
            Side length of each subplot panel in inches.
        **plot_frame_kwargs
            Forwarded verbatim to :meth:`plot_frame`.

        Returns
        -------
        fig : Figure
        axes : ndarray of Axes, shape (nrows, ncols)
        """
        if frame_indices is None:
            n = min(9, self.num_frames)
            frame_indices = list(
                np.linspace(0, self.num_frames - 1, n, dtype=int)
            )

        n = len(frame_indices)
        ncols = min(ncols, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * subplot_size, nrows * subplot_size),
        )
        axes_flat = np.array(axes).flatten()

        for k, fidx in enumerate(frame_indices):
            self.plot_frame(fidx, ax=axes_flat[k], **plot_frame_kwargs)

        # Hide unused panels.
        for k in range(n, len(axes_flat)):
            axes_flat[k].set_visible(False)

        fig.tight_layout()
        return fig, axes

    # ------------------------------------------------------------------
    # Visualisation: animation
    # ------------------------------------------------------------------

    def animate(
        self,
        frame_indices: Optional[Sequence[int]] = None,
        *,
        interval: int = 80,
        figsize: tuple[float, float] = (5, 5),
        repeat: bool = True,
        **plot_frame_kwargs,
    ) -> animation.FuncAnimation:
        """
        Create a :class:`matplotlib.animation.FuncAnimation` that sweeps
        through the trajectory.

        Parameters
        ----------
        frame_indices : sequence of int, optional
            Subset of frames to animate.  Defaults to all frames.
        interval : int
            Delay between frames in milliseconds.
        figsize : (float, float)
            Figure size in inches.
        repeat : bool
            Whether the animation loops.
        **plot_frame_kwargs
            Forwarded to :meth:`plot_frame` (except *ax* and *figsize*).

        Returns
        -------
        anim : matplotlib.animation.FuncAnimation
            Call ``anim.save(...)`` or display inline in a Jupyter notebook
            with ``from IPython.display import HTML; HTML(anim.to_jshtml())``.
        """
        if frame_indices is None:
            frame_indices = list(range(self.num_frames))

        fig, ax = plt.subplots(figsize=figsize)

        def _update(fidx: int) -> list:
            ax.cla()
            self.plot_frame(fidx, ax=ax, **plot_frame_kwargs)
            return []

        _update(frame_indices[0])

        anim = animation.FuncAnimation(
            fig,
            _update,
            frames=frame_indices,
            interval=interval,
            repeat=repeat,
            blit=False,
        )
        return anim
