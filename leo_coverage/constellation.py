"""Walker constellation generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import R_EARTH


@dataclass(frozen=True)
class Constellation:
    """A Walker constellation of circular orbits.

    pattern = "star": planes spread over 180 deg of RAAN (polar systems).
    pattern = "delta": planes spread over 360 deg of RAAN (inclined systems).
    """

    n_planes: int
    sats_per_plane: int
    altitude_km: float
    inclination_deg: float
    phasing: int = 1
    pattern: str = "star"

    @property
    def n_sats(self) -> int:
        return self.n_planes * self.sats_per_plane

    @property
    def a_km(self) -> float:
        return R_EARTH + self.altitude_km

    @property
    def inclination_rad(self) -> float:
        return np.radians(self.inclination_deg)

    def initial_angles(self):
        """Initial RAAN and argument of latitude [rad] for every satellite."""
        if self.pattern not in ("star", "delta"):
            raise ValueError("pattern must be 'star' or 'delta'")
        spread = np.pi if self.pattern == "star" else 2.0 * np.pi
        p = np.repeat(np.arange(self.n_planes), self.sats_per_plane)
        s = np.tile(np.arange(self.sats_per_plane), self.n_planes)
        raan = p * spread / self.n_planes
        u = (2.0 * np.pi * s / self.sats_per_plane
             + 2.0 * np.pi * self.phasing * p / self.n_sats)
        return raan, u


def eutelsat_next_like(inclination_deg: float = 87.9, **kwargs) -> Constellation:
    """Geometry from the public July 2026 FCC filing for Eutelsat Next:
    528 satellites in 24 planes at 1,220 km.

    The inclination is NOT taken from the filing: 87.9 deg is an assumption
    (close to the current OneWeb fleet). Treat it as a variable to study.
    """
    return Constellation(n_planes=24, sats_per_plane=22, altitude_km=1220.0,
                         inclination_deg=inclination_deg, **kwargs)
