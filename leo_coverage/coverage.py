"""Coverage analysis for a ground site and for a global grid."""

from __future__ import annotations

import numpy as np

from .constants import R_EARTH
from .constellation import Constellation
from .frames import eci_to_ecef, elevation_deg, geodetic_to_ecef, local_up
from .orbits import propagate_secular_j2


def coverage_half_angle_deg(altitude_km: float, min_elev_deg: float) -> float:
    """Earth central angle [deg] of one satellite's footprint (spherical Earth).

    lambda = arccos(R cos(eps) / (R + h)) - eps
    """
    eps = np.radians(min_elev_deg)
    lam = np.arccos(R_EARTH * np.cos(eps) / (R_EARTH + altitude_km)) - eps
    return float(np.degrees(lam))


def constellation_ecef(con: Constellation, t, include_j2: bool = True):
    """ECEF positions of every satellite, shape (n_sat, n_t, 3)."""
    raan0, u0 = con.initial_angles()
    r_eci = propagate_secular_j2(con.a_km, con.inclination_rad, raan0, u0, t, include_j2)
    return eci_to_ecef(r_eci, t)


def sats_in_view(con: Constellation, lat_deg, lon_deg, t, min_elev_deg: float = 25.0,
                 include_j2: bool = True):
    """Number of satellites above ``min_elev_deg`` at each time, plus the
    maximum elevation available at each time."""
    r = constellation_ecef(con, t, include_j2)
    site = geodetic_to_ecef(lat_deg, lon_deg)
    el = elevation_deg(r, site, local_up(lat_deg, lon_deg))   # (n_sat, n_t)
    return (el >= min_elev_deg).sum(axis=0), el.max(axis=0)


def global_snapshot(con: Constellation, t_s: float = 0.0, min_elev_deg: float = 25.0,
                    step_deg: float = 2.0, include_j2: bool = True):
    """Satellites in view over a latitude/longitude grid at a single time."""
    lats = np.arange(-90.0 + step_deg / 2, 90.0, step_deg)
    lons = np.arange(-180.0 + step_deg / 2, 180.0, step_deg)
    lat_g, lon_g = np.meshgrid(lats, lons, indexing="ij")
    sites = geodetic_to_ecef(lat_g, lon_g)          # (n_lat, n_lon, 3)
    ups = local_up(lat_g, lon_g)
    r = constellation_ecef(con, np.array([t_s]), include_j2)[:, 0, :]  # (n_sat, 3)
    count = np.zeros(lat_g.shape, dtype=int)
    for rs in r:                                     # loop sats, vectorise grid
        count += elevation_deg(rs, sites, ups) >= min_elev_deg
    return lats, lons, count
