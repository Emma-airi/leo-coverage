"""Orbit propagation.

Two routes are provided:

* ``propagate_numerical`` integrates the equations of motion (two-body plus
  optional J2 acceleration) with SciPy. It is accurate but slow, so it is used
  for single satellites and for validation.
* ``propagate_secular_j2`` advances circular orbits with the analytic secular
  J2 rates. It is fast enough for a whole constellation and is validated
  against the numerical route in the tests.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .constants import J2, MU_EARTH, R_EARTH


def orbital_period(a_km: float, mu: float = MU_EARTH) -> float:
    """Keplerian orbital period [s] for semi-major axis ``a_km``."""
    return 2.0 * np.pi * np.sqrt(a_km**3 / mu)


def elements_to_state(a, e, i, raan, argp, nu, mu: float = MU_EARTH):
    """Classical orbital elements (angles in rad) to ECI position and velocity.

    Uses the perifocal frame and the 3-1-3 rotation (RAAN, inclination,
    argument of perigee), as in the original thesis propagator.
    """
    p = a * (1.0 - e**2)
    r_pf = (p / (1.0 + e * np.cos(nu))) * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(i), np.sin(i)
    cw, sw = np.cos(argp), np.sin(argp)
    rot = np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si, cw * si, ci],
    ])
    return rot @ r_pf, rot @ v_pf


def acceleration(r, include_j2: bool = True, mu: float = MU_EARTH):
    """Gravitational acceleration [km/s^2] at ECI position ``r`` [km]."""
    x, y, z = r
    rn = np.linalg.norm(r)
    a = -mu * r / rn**3
    if include_j2:
        k = -1.5 * J2 * mu * R_EARTH**2 / rn**5
        zr2 = (z / rn) ** 2
        a = a + k * np.array([x * (1 - 5 * zr2), y * (1 - 5 * zr2), z * (3 - 5 * zr2)])
    return a


def propagate_numerical(r0, v0, t_eval, include_j2: bool = True,
                        rtol: float = 1e-11, atol: float = 1e-11):
    """Integrate one satellite. Returns positions and velocities, shape (N, 3)."""

    def rhs(_t, y):
        return np.concatenate([y[3:], acceleration(y[:3], include_j2)])

    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), np.concatenate([r0, v0]),
                    t_eval=t_eval, method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[:3].T, sol.y[3:].T


def j2_secular_rates(a, e, i, mu: float = MU_EARTH):
    """Secular J2 rates [rad/s] of RAAN, argument of perigee and mean anomaly."""
    n = np.sqrt(mu / a**3)
    p = a * (1.0 - e**2)
    f = (R_EARTH / p) ** 2
    raan_dot = -1.5 * n * J2 * f * np.cos(i)
    argp_dot = 0.75 * n * J2 * f * (5.0 * np.cos(i) ** 2 - 1.0)
    m_dot = n * (1.0 + 0.75 * J2 * f * np.sqrt(1.0 - e**2) * (3.0 * np.cos(i) ** 2 - 1.0))
    return raan_dot, argp_dot, m_dot


def propagate_secular_j2(a, i, raan0, u0, t, include_j2: bool = True):
    """Propagate many circular orbits at once.

    Parameters
    ----------
    a, i : float
        Common semi-major axis [km] and inclination [rad].
    raan0, u0 : array (n_sat,)
        Initial RAAN and argument of latitude [rad] for each satellite.
    t : array (n_t,)
        Times since epoch [s].

    Returns
    -------
    r_eci : array (n_sat, n_t, 3) of ECI positions [km].
    """
    raan0 = np.atleast_1d(raan0)[:, None]
    u0 = np.atleast_1d(u0)[:, None]
    t = np.atleast_1d(t)[None, :]
    if include_j2:
        raan_dot, argp_dot, m_dot = j2_secular_rates(a, 0.0, i)
        u_dot = argp_dot + m_dot
    else:
        raan_dot, u_dot = 0.0, np.sqrt(MU_EARTH / a**3)
    raan = raan0 + raan_dot * t
    u = u0 + u_dot * t

    cO, sO = np.cos(raan), np.sin(raan)
    cu, su = np.cos(u), np.sin(u)
    ci, si = np.cos(i), np.sin(i)
    x = a * (cO * cu - sO * su * ci)
    y = a * (sO * cu + cO * su * ci)
    z = a * (su * si)
    return np.stack([x, y, z], axis=-1)


def raan_from_state(r, v):
    """Right ascension of the ascending node [rad] from ECI state vectors."""
    h = np.cross(r, v)
    return np.arctan2(h[..., 0], -h[..., 1])
