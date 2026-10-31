"""Small numerical sanity checks for Project Event Horizon physics routines."""
from __future__ import annotations
import numpy as np
from event_horizon_physics import (
    ModelParameters,
    integrate_schwarzschild_ray,
    integrate_ellis_ray,
    schwarzschild_null_constraint_residual,
    ellis_null_constraint_residual,
)


def main() -> int:
    p = ModelParameters()
    assert np.isclose(p.schwarzschild_radius, 2.0 * p.black_hole_mass)
    assert np.isclose(p.photon_sphere_radius, 3.0 * p.black_hole_mass)
    assert np.isclose(p.critical_impact_parameter,
                      3.0 * np.sqrt(3.0) * p.black_hole_mass)

    # The test set intentionally includes captured and scattered Schwarzschild rays.
    bh_results = []
    for factor in (0.78, 0.98, 1.015, 1.18):
        b = factor * p.critical_impact_parameter
        coords, outcome = integrate_schwarzschild_ray(b, p.black_hole_mass)
        assert len(coords) > 100 and np.isfinite(coords).all()
        residual = schwarzschild_null_constraint_residual(b, coords, p.black_hole_mass)
        assert residual < 2.0e-4, (factor, residual)
        bh_results.append((factor, outcome, residual))
    assert "captured" in bh_results[0][1]
    assert bh_results[-1][1] == "scattered"

    # Ellis rays below the throat scale pass through; larger impact parameters turn back.
    wh_results = []
    for factor in (0.55, 0.94, 1.04, 1.38):
        b = factor * p.wormhole_throat
        coords, outcome = integrate_ellis_ray(b, p.wormhole_throat)
        assert len(coords) > 100 and np.isfinite(coords).all()
        residual = ellis_null_constraint_residual(b, coords, p.wormhole_throat)
        assert residual < 5.0e-3, (factor, residual)
        wh_results.append((factor, outcome, residual))
    assert wh_results[0][1] == "traversed throat"
    assert wh_results[-1][1] == "turned back"

    print("PROJECT EVENT HORIZON | physics checks")
    print("PASS: Schwarzschild horizon / photon sphere / critical impact parameter")
    print("PASS: captured and scattered Schwarzschild ray families")
    print("PASS: traversing and reflected Ellis wormhole ray families")
    print("PASS: approximate null-constraint residuals")
    print("\nSchwarzschild residuals:")
    for factor, outcome, residual in bh_results:
        print(f"  b/bcrit={factor:5.3f}  {outcome:24s} residual={residual:.3e}")
    print("Ellis residuals:")
    for factor, outcome, residual in wh_results:
        print(f"  b/a={factor:5.3f}       {outcome:24s} residual={residual:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
