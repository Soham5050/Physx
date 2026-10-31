# PROJECT EVENT HORIZON — CINEMATIC EDITION

A self-contained procedural 3D space observatory built with Python, Pygame and ModernGL. The renderer creates its nebula field, starfield, embedding surfaces, accretion-disk artwork, photon paths, glow and interface procedurally; no external Blender assets are required.

## Quick start (Windows 10/11)

1. Install **64-bit Python 3.10 or newer**. During installation, enable the Python launcher / PATH options.
2. Extract this folder somewhere convenient.
3. Double-click `INSTALL_AND_RUN.bat`.

Or use PowerShell in this folder:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python event_horizon_cinematic.py
```

If PowerShell blocks activation, either run `Set-ExecutionPolicy -Scope Process Bypass` in that PowerShell session or use `INSTALL_AND_RUN.bat`.

## Run physics sanity checks

```powershell
py verify_physics.py
```

This verifies characteristic Schwarzschild radii, captured/scattered ray cases, Ellis throat crossing/reflection and approximate conservation-constraint residuals. The Ellis residual is estimated with finite differences, so it is a numerical diagnostic rather than an exact symbolic test.

## Interaction

| Input | Action |
|---|---|
| Left mouse drag | Orbit camera |
| Right mouse drag | Pan camera target |
| Mouse wheel | Zoom |
| `1` | Focus black hole / Schwarzschild geometry |
| `2` | Focus Ellis wormhole geometry |
| `3` | Show both geometries side by side |
| `TAB` | Toggle Exploration / Laboratory mode |
| `E` | Exploration mode (minimal HUD) |
| `L` | Laboratory mode (physics panels) |
| `[` / `]` (Lab mode) | Decrease / increase black-hole mass `M` |
| `,` / `.` (Lab mode) | Decrease / increase Ellis throat radius `a` |
| `SPACE` | Pause / resume ray markers |
| `UP` / `DOWN` | Increase / decrease marker animation speed |
| `R` | Reset animation time |
| `H` | Hide / show HUD |
| `ESC` | Quit |

The `1`, `2`, and `3` keys select the physical background view; the `TAB` key selects how much scientific instrumentation is displayed.

## What the scene contains

- Procedural dark-space nebula and animated star-like field.
- A shaded Flamm paraboloid and wireframe for the equatorial, constant-time Schwarzschild spatial slice.
- A stylized dark horizon visual, photon-sphere ring and an **illustrative** animated accretion-disk mesh.
- A shaded Ellis-Bronnikov catenoid embedding with a luminous throat and guide rings.
- Families of animated null-geodesic paths on the two separate fixed backgrounds.
- Bloom extraction / blur, final tone mapping, vignette and a translucent scientific HUD.

## Physics retained from the 3D prototype

All formulas use geometrized units `G = c = 1`.

### Schwarzschild black hole

The metric is

\[
ds^2 = -(1-2M/r)dt^2 +(1-2M/r)^{-1}dr^2+r^2d\Omega^2.
\]

For equatorial null geodesics, set `u = 1/r` and use the azimuth `phi` as independent variable:

\[
\frac{d^2u}{d\phi^2}+u=3Mu^2.
\]

Characteristic scales:

\[
r_s=2M,\qquad r_{ph}=3M,\qquad b_{crit}=3\sqrt{3}M.
\]

The equatorial spatial embedding used for display is

\[
z(r)=2\sqrt{r_s(r-r_s)}.
\]

### Ellis-Bronnikov wormhole

In proper radial coordinate `l`, the fixed Ellis metric is

\[
ds^2=-dt^2+dl^2+(l^2+a^2)d\Omega^2.
\]

For equatorial null rays with `E=1`, `L=b`:

\[
\dot l^2+\frac{b^2}{l^2+a^2}=1,\qquad
\ddot l=\frac{b^2l}{(l^2+a^2)^2},\qquad
\dot\phi=\frac{b}{l^2+a^2}.
\]

Its equatorial spatial embedding is

\[
R(l)=\sqrt{l^2+a^2},\qquad z(l)=a\,\mathrm{asinh}(l/a).
\]

For these idealized Ellis null rays, `b < a` can traverse the throat; `b > a` turns back. The Ellis solution uses an exotic/phantom scalar source and is not evidence that traversable astrophysical wormholes exist.

## Rendering and dynamic resolution

- **OpenGL 3.3 core profile** is required. The application prints the actual OpenGL vendor, renderer and version at startup.
- ModernGL renders the 3D scene, procedural sky, bloom passes and final upscale.
- The initial render scale is 90%; it can adapt between 55% and 100% of the window resolution.
- The adaptive scaler uses rolling end-to-end CPU frame time as a practical frame-budget signal. It does not currently use vendor-specific GPU timer queries.
- The CPU (NumPy + RK4) calculates the geodesic samples. OpenGL accelerates visualization and post-processing, not the integration of Einstein's equations.
- Maximize the window on a WQHD (2560 x 1440) display for a native-size cinematic viewport. Dynamic scaling changes the internal scene render resolution while the HUD remains at display resolution.

### Prefer the RTX 5060 on Windows

Windows may select the Radeon 780M or RTX 5060 depending on its graphics preference. To request the RTX 5060:

1. Open **Settings → System → Display → Graphics**.
2. Add the `python.exe` located inside this project's `.venv\Scripts` folder (or the Python executable used to launch the project).
3. Choose **Options → High performance (NVIDIA GeForce RTX 5060)**.
4. Save and restart the application.
5. Check the printed `OpenGL renderer` line at launch.

Choose Power saving to test the Radeon 780M. GPU choice is controlled by the driver / Windows, not by a Python runtime switch.

## Scientific scope and limitations

**This is not a black-hole / wormhole collision simulation.** The scene shows null geodesics on two independent, exact, fixed background metrics. The code does not add the metrics together, evolve the stress-energy tensor, or solve a time-dependent Einstein-matter initial-value problem.

Additional distinctions:

- The displayed 3D surfaces are Euclidean embedding diagrams of 2D spatial slices. The vertical embedding coordinate is a visualization aid, not a literal extra spatial dimension.
- The plotted rays are equatorial null geodesics embedded onto those spatial slices, not full 3D spacetime trajectories.
- The dark horizon sphere and the accretion disk are visual design elements. The sphere is not an exact embedding of the event horizon; the disk is not a GRMHD calculation.
- The Ellis wormhole is a mathematical solution requiring an exotic/phantom source in this idealized model.
- Real interaction / collision research would require constraint-satisfying initial data and a multidimensional numerical-relativity evolution for a specifically defined Einstein-matter theory. A visual combination of two separate solutions is not a valid collision metric.

## Troubleshooting

### `No module named pygame` / `moderngl`

Activate the project virtual environment and run `python -m pip install -r requirements.txt`.

### OpenGL context creation fails

Update the NVIDIA and AMD graphics drivers. OpenGL 3.3 core is required. Some Remote Desktop sessions expose a software renderer or an older OpenGL context; test locally on the laptop screen.

### Black window or shader error

Update the GPU driver first. If it persists, launch from PowerShell and share the complete terminal output, including the OpenGL vendor/renderer/version lines and the shader error.

### Low frame rate

Maximize the window and let the adaptive scaler settle. You can also use Windows Graphics settings to select the RTX 5060. The adaptive render scale will lower internal resolution to protect responsiveness.

## Project files

- `event_horizon_cinematic.py` — OpenGL renderer, camera, UI and application loop.
- `event_horizon_physics.py` — Schwarzschild and Ellis geodesic integrators and embedding functions.
- `verify_physics.py` — numerical sanity checks.
- `requirements.txt` — Python dependencies.
- `INSTALL_AND_RUN.bat` — Windows virtual-environment setup and launcher.
