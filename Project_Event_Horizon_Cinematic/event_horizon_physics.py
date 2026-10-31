"""Relativistic test-particle geodesics for Project Event Horizon.

All calculations use geometrized units G = c = 1.

This module intentionally keeps the two backgrounds independent:
  * Schwarzschild vacuum black-hole spacetime.
  * Ellis-Bronnikov traversable-wormhole spacetime (phantom scalar source).

It does NOT construct a black-hole/wormhole collision metric.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import math
import numpy as np


@dataclass
class ModelParameters:
    black_hole_mass: float = 1.0
    wormhole_throat: float = 2.0

    @property
    def schwarzschild_radius(self) -> float:
        return 2.0 * self.black_hole_mass

    @property
    def photon_sphere_radius(self) -> float:
        return 3.0 * self.black_hole_mass

    @property
    def critical_impact_parameter(self) -> float:
        return 3.0 * math.sqrt(3.0) * self.black_hole_mass


@dataclass
class Geodesic:
    name: str
    model: str                    # "bh" or "wh"
    impact_parameter: float
    coordinates: np.ndarray       # (r, phi) for BH; (l, phi) for Ellis
    points_3d: np.ndarray         # points on the corresponding spatial embedding
    color: tuple[float, float, float]
    outcome: str


def rk4_step(rhs: Callable[[np.ndarray], np.ndarray], y: np.ndarray, h: float) -> np.ndarray:
    k1 = rhs(y)
    k2 = rhs(y + 0.5 * h * k1)
    k3 = rhs(y + 0.5 * h * k2)
    k4 = rhs(y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def schwarzschild_embedding(r: float | np.ndarray, phi: float | np.ndarray,
                            mass: float) -> np.ndarray:
    """Flamm paraboloid embedding of the equatorial t=constant spatial slice."""
    rs = 2.0 * mass
    r_arr = np.asarray(r, dtype=np.float64)
    p_arr = np.asarray(phi, dtype=np.float64)
    z = 2.0 * np.sqrt(np.maximum(0.0, rs * (r_arr - rs)))
    return np.stack((r_arr * np.cos(p_arr), r_arr * np.sin(p_arr), z), axis=-1)


def ellis_embedding(l: float | np.ndarray, phi: float | np.ndarray,
                    throat: float) -> np.ndarray:
    """Catenoid embedding of an equatorial Ellis wormhole spatial slice."""
    l_arr = np.asarray(l, dtype=np.float64)
    p_arr = np.asarray(phi, dtype=np.float64)
    radius = np.sqrt(l_arr * l_arr + throat * throat)
    z = throat * np.arcsinh(l_arr / throat)
    return np.stack((radius * np.cos(p_arr), radius * np.sin(p_arr), z), axis=-1)


def integrate_schwarzschild_ray(impact_parameter: float, mass: float,
                                max_steps: int = 42000,
                                dphi: float = 0.0012) -> tuple[np.ndarray, str]:
    """Integrate an equatorial Schwarzschild null geodesic.

    With u=1/r and azimuth phi as independent variable:
        u'' + u = 3 M u^2
    Initial data are set on the incoming branch at r0=22M. The impact
    parameter is b=L/E. Rays below b_crit are captured; rays above it scatter.
    """
    b = float(impact_parameter)
    r_s = 2.0 * mass
    r0 = 22.0 * mass
    u0 = 1.0 / r0
    radicand = 1.0 / (b * b) - u0 * u0 * (1.0 - 2.0 * mass * u0)
    if radicand <= 0.0 or b <= 0.0:
        return np.empty((0, 2), dtype=np.float64), "invalid initial data"

    state = np.array([u0, math.sqrt(radicand)], dtype=np.float64)

    def rhs(y: np.ndarray) -> np.ndarray:
        u, du_dphi = y
        return np.array([du_dphi, -u + 3.0 * mass * u * u], dtype=np.float64)

    samples: list[tuple[float, float]] = []
    outcome = "captured / horizon cutoff"
    for step in range(max_steps):
        u, du = state
        if not np.isfinite(u) or not np.isfinite(du) or u >= 1.0 / r_s or u <= 0.0:
            break
        phi = step * dphi
        samples.append((1.0 / u, phi))
        state = rk4_step(rhs, state, dphi)
        # On a scattering orbit, du/dphi changes sign at periapsis, then
        # the ray returns to its launch radius on the outgoing branch.
        if step > 30 and state[1] < 0.0 and state[0] < u0:
            outcome = "scattered"
            break
    return np.asarray(samples, dtype=np.float64), outcome


def integrate_ellis_ray(impact_parameter: float, throat: float,
                        max_steps: int = 16000) -> tuple[np.ndarray, str]:
    """Integrate an equatorial null geodesic in the Ellis metric.

    ds^2=-dt^2+dl^2+(l^2+a^2)dOmega^2. For E=1 and L=b:
        (dl/dlambda)^2 + b^2/(l^2+a^2) = 1
        d2l/dlambda2 = b^2 l/(l^2+a^2)^2
        dphi/dlambda = b/(l^2+a^2)
    """
    a = float(throat)
    b = float(impact_parameter)
    l0 = 8.0 * a
    q0 = l0 * l0 + a * a
    radial2 = 1.0 - b * b / q0
    if radial2 <= 0.0 or a <= 0.0 or b <= 0.0:
        return np.empty((0, 2), dtype=np.float64), "invalid initial data"

    state = np.array([l0, -math.sqrt(radial2), 0.0], dtype=np.float64)

    def rhs(y: np.ndarray) -> np.ndarray:
        l, dl_dlambda, _phi = y
        q = l * l + a * a
        return np.array([dl_dlambda, b * b * l / (q * q), b / q], dtype=np.float64)

    dlambda = 0.008 * a
    samples: list[tuple[float, float]] = []
    crossed_throat = False
    outcome = "turned back"
    for _ in range(max_steps):
        l, velocity, phi = state
        if not np.all(np.isfinite(state)):
            break
        samples.append((l, phi))
        if l < 0.0:
            crossed_throat = True
        state = rk4_step(rhs, state, dlambda)
        if crossed_throat and state[0] <= -l0:
            outcome = "traversed throat"
            break
        if (not crossed_throat and len(samples) > 40 and state[1] > 0.0
                and state[0] >= l0):
            outcome = "turned back"
            break
    return np.asarray(samples, dtype=np.float64), outcome


def build_geodesics(params: ModelParameters,
                    bh_center: np.ndarray,
                    wh_center: np.ndarray) -> list[Geodesic]:
    """Build illustrative families of exact-background null geodesics."""
    colors = [
        (1.00, 0.35, 0.22), (1.00, 0.58, 0.20), (1.00, 0.79, 0.30),
        (0.30, 0.85, 1.00), (0.39, 0.59, 1.00), (0.75, 0.48, 1.00),
        (1.00, 0.43, 0.72), (0.43, 0.95, 0.68), (0.90, 0.84, 1.00),
    ]
    result: list[Geodesic] = []
    m = params.black_hole_mass
    bcrit = params.critical_impact_parameter
    bh_b_values = [0.78, 0.90, 0.98, 1.015, 1.035, 1.08, 1.18, 1.34, 1.55]
    for i, factor in enumerate(bh_b_values):
        b = factor * bcrit
        coords, outcome = integrate_schwarzschild_ray(b, m)
        if len(coords) < 3:
            continue
        points = schwarzschild_embedding(coords[:, 0], coords[:, 1], m) + bh_center
        result.append(Geodesic(
            name=f"BH photon {i + 1}", model="bh", impact_parameter=b,
            coordinates=coords, points_3d=points.astype(np.float32),
            color=colors[i % len(colors)], outcome=outcome))

    a = params.wormhole_throat
    wh_factors = [0.18, 0.35, 0.55, 0.72, 0.86, 0.94, 1.04, 1.16, 1.38, 1.65]
    for i, factor in enumerate(wh_factors):
        b = factor * a
        coords, outcome = integrate_ellis_ray(b, a)
        if len(coords) < 3:
            continue
        points = ellis_embedding(coords[:, 0], coords[:, 1], a) + wh_center
        color = colors[(i + 3) % len(colors)]
        if outcome == "traversed throat":
            color = (0.27, 0.88, 1.00) if i % 2 == 0 else (0.72, 0.48, 1.00)
        result.append(Geodesic(
            name=f"Ellis photon {i + 1}", model="wh", impact_parameter=b,
            coordinates=coords, points_3d=points.astype(np.float32),
            color=color, outcome=outcome))
    return result


def schwarzschild_null_constraint_residual(b: float, samples: np.ndarray,
                                           mass: float) -> float:
    """Maximum absolute residual of (du/dphi)^2 energy constraint.

    This diagnostic estimates du/dphi from the sampled u(phi) trajectory.
    It is useful as a numerical convergence check, not as a proof of a
    full spacetime evolution.
    """
    if len(samples) < 5:
        return float("nan")
    r = samples[:, 0]
    phi = samples[:, 1]
    u = 1.0 / r
    du = np.gradient(u, phi)
    residual = du * du - (1.0 / (b * b) - u * u * (1.0 - 2.0 * mass * u))
    return float(np.nanmax(np.abs(residual)))


def ellis_null_constraint_residual(b: float, samples: np.ndarray,
                                   throat: float, dlambda: float | None = None) -> float:
    """Residual using finite-difference dl/dlambda along sampled Ellis ray."""
    if len(samples) < 5:
        return float("nan")
    l = samples[:, 0]
    phi = samples[:, 1]
    # dphi/dlambda = b/(l^2+a^2), so d lambda / d index follows from dphi.
    dphi = np.gradient(phi)
    dlam = dphi * (l * l + throat * throat) / b
    dl = np.gradient(l)
    velocity = np.divide(dl, dlam, out=np.zeros_like(dl), where=np.abs(dlam) > 1e-15)
    residual = velocity * velocity + b * b / (l * l + throat * throat) - 1.0
    return float(np.nanmax(np.abs(residual)))
