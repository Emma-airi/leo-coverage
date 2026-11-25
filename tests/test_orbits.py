"""Validation of the propagators against hand formulas and each other."""

import numpy as np
import pytest

from leo_coverage.constants import MU_EARTH, R_EARTH
from leo_coverage.orbits import (elements_to_state, j2_secular_rates, orbital_period,
                                 propagate_numerical, propagate_secular_j2, raan_from_state)

A_ISS = R_EARTH + 420.0
A_NEXT = R_EARTH + 1220.0


def test_period_matches_hand_formula():
    # Hand calculation for a 1,220 km circular orbit: T = 2*pi*sqrt(a^3/mu)
    a = A_NEXT
    expected = 2 * np.pi * np.sqrt(a**3 / MU_EARTH)
    assert orbital_period(a) == pytest.approx(expected, rel=1e-12)
    assert 109 * 60 < orbital_period(a) < 110 * 60   # about 109.9 minutes


def test_two_body_returns_to_start_after_one_period():
    r0, v0 = elements_to_state(A_NEXT, 0.0, np.radians(87.9), 0.3, 0.0, 0.0)
    T = orbital_period(A_NEXT)
    r, _ = propagate_numerical(r0, v0, np.array([0.0, T]), include_j2=False)
    assert np.linalg.norm(r[-1] - r0) < 1e-3   # better than 1 m


def test_two_body_conserves_energy():
    r0, v0 = elements_to_state(A_NEXT, 0.001, np.radians(87.9), 0.0, 0.0, 0.0)
    t = np.linspace(0.0, 86400.0, 200)
    r, v = propagate_numerical(r0, v0, t, include_j2=False)
    energy = 0.5 * np.sum(v**2, axis=1) - MU_EARTH / np.linalg.norm(r, axis=1)
    assert np.ptp(energy) / abs(energy[0]) < 1e-9


def test_numerical_j2_raan_drift_matches_analytic_rate():
    # ISS-type orbit, where the nodal drift is large (about -5 deg/day)
    i = np.radians(51.64)
    r0, v0 = elements_to_state(A_ISS, 0.0, i, 0.0, 0.0, 0.0)
    t = np.linspace(0.0, 5 * 86400.0, 2000)
    r, v = propagate_numerical(r0, v0, t, include_j2=True)
    raan = np.unwrap(raan_from_state(r, v))
    fitted_rate = np.polyfit(t, raan, 1)[0]          # fit removes short-period wobble
    analytic_rate, _, _ = j2_secular_rates(A_ISS, 0.0, i)
    assert fitted_rate == pytest.approx(analytic_rate, rel=0.02)


def test_secular_propagator_against_numerical_over_one_orbit():
    """Plane and radius agree closely; along-track drift is a known limitation.

    The secular model treats the initial elements as mean elements, while the
    numerical run starts from osculating ones, so their orbital periods differ
    slightly and an along-track offset grows each orbit. Converting osculating
    to mean elements (Brouwer or Kozai theory) would remove it.
    """
    i = np.radians(87.9)
    r0, v0 = elements_to_state(A_NEXT, 0.0, i, 0.5, 0.0, 0.0)
    t = np.linspace(0.0, orbital_period(A_NEXT), 50)
    r_num, _ = propagate_numerical(r0, v0, t, include_j2=True)
    r_sec = propagate_secular_j2(A_NEXT, i, np.array([0.5]), np.array([0.0]), t)[0]

    d = r_num - r_sec
    r_hat = r_sec / np.linalg.norm(r_sec, axis=1)[:, None]
    h_hat = np.cross(r0, v0) / np.linalg.norm(np.cross(r0, v0))
    radial = np.sum(d * r_hat, axis=1)
    cross = d @ h_hat
    along = np.linalg.norm(d - radial[:, None] * r_hat - cross[:, None] * h_hat, axis=1)

    assert np.max(np.abs(radial)) < 10.0      # km
    assert np.max(np.abs(cross)) < 1.0        # km: orbit planes agree
    assert np.max(along) < 100.0              # km per orbit: documented drift


def test_secular_without_j2_is_a_fixed_circle():
    t = np.linspace(0.0, 86400.0, 100)
    r = propagate_secular_j2(A_NEXT, np.radians(87.9), np.array([0.0]), np.array([0.0]),
                             t, include_j2=False)[0]
    assert np.allclose(np.linalg.norm(r, axis=1), A_NEXT)
