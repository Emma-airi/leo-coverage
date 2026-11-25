"""Validation of the geometry and coverage functions."""

import numpy as np
import pytest

from leo_coverage.constants import R_EARTH
from leo_coverage.constellation import Constellation, eutelsat_next_like
from leo_coverage.coverage import coverage_half_angle_deg, sats_in_view
from leo_coverage.frames import elevation_deg, geodetic_to_ecef, local_up


def test_satellite_overhead_is_at_90_degrees():
    site = geodetic_to_ecef(51.5, -0.13)
    up = local_up(51.5, -0.13)
    sat = site + 1220.0 * up
    assert elevation_deg(sat, site, up) == pytest.approx(90.0, abs=1e-9)


def test_footprint_edge_is_at_minimum_elevation():
    # Equatorial site, where geodetic and geocentric up coincide
    h, eps = 1220.0, 25.0
    lam = np.radians(coverage_half_angle_deg(h, eps))
    site = geodetic_to_ecef(0.0, 0.0)
    sat = (R_EARTH + h) * np.array([np.cos(lam), np.sin(lam), 0.0])
    assert elevation_deg(sat, site, local_up(0.0, 0.0)) == pytest.approx(eps, abs=1e-6)


def test_walker_layout():
    con = eutelsat_next_like()
    raan, u = con.initial_angles()
    assert con.n_sats == 528 and raan.size == 528
    assert np.isclose(np.unique(np.round(raan, 9)).size, 24)
    assert raan.max() < np.pi                      # star pattern spans 180 deg


def test_polar_constellation_sees_more_at_high_latitude_than_equator():
    # Known property of polar constellations: planes converge near the poles
    con = eutelsat_next_like()
    t = np.arange(0.0, 6 * 3600.0, 120.0)
    n_eq, _ = sats_in_view(con, 0.0, 0.0, t)
    n_hi, _ = sats_in_view(con, 70.0, 0.0, t)
    assert n_hi.mean() > n_eq.mean()


def test_bad_pattern_rejected():
    with pytest.raises(ValueError):
        Constellation(4, 4, 1000.0, 50.0, pattern="flower").initial_angles()


def test_global_mean_in_view_matches_footprint_area():
    """Each footprint covers a fixed fraction of the sphere, (1 - cos lambda)/2,
    so the area-weighted global mean of satellites in view must equal
    n_sats * (1 - cos lambda) / 2, whatever the constellation geometry."""
    from leo_coverage.coverage import global_snapshot

    con = eutelsat_next_like()
    lats, _, count = global_snapshot(con, 0.0, 25.0, step_deg=2.0)
    w = np.cos(np.radians(lats))[:, None] * np.ones_like(count)
    simulated = np.sum(count * w) / np.sum(w)
    lam = np.radians(coverage_half_angle_deg(con.altitude_km, 25.0))
    expected = con.n_sats * (1 - np.cos(lam)) / 2
    assert simulated == pytest.approx(expected, rel=0.03)
