"""Reference frames: ECI to ECEF, geodetic sites and look angles."""

from __future__ import annotations

import numpy as np

from .constants import OMEGA_EARTH, R_EARTH, WGS84_E2


def eci_to_ecef(r_eci, t, theta0: float = 0.0):
    """Rotate ECI positions to ECEF about the z axis.

    Simplification: Earth rotation angle = theta0 + OMEGA_EARTH * t, with no
    precession, nutation or polar motion. Adequate for coverage statistics.
    ``r_eci`` has shape (..., n_t, 3) with time on the second-last axis.
    """
    theta = theta0 + OMEGA_EARTH * np.asarray(t)
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = r_eci[..., 0], r_eci[..., 1], r_eci[..., 2]
    return np.stack([c * x + s * y, -s * x + c * y, z], axis=-1)


def geodetic_to_ecef(lat_deg, lon_deg, h_km=0.0):
    """WGS-84 geodetic latitude, longitude and height to ECEF [km]."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    n = R_EARTH / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    x = (n + h_km) * np.cos(lat) * np.cos(lon)
    y = (n + h_km) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + h_km) * np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def local_up(lat_deg, lon_deg):
    """Unit vector normal to the WGS-84 ellipsoid (geodetic 'up')."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def elevation_deg(r_sat_ecef, r_site_ecef, up):
    """Elevation angle [deg] of satellites seen from a site.

    Broadcasting: r_sat_ecef (..., 3), r_site_ecef and up (3,) or broadcastable.
    """
    rho = r_sat_ecef - r_site_ecef
    sin_el = np.sum(rho * up, axis=-1) / np.linalg.norm(rho, axis=-1)
    return np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0)))
