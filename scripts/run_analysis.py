"""Run the baseline analysis and save figures and a results summary.

Usage:  python scripts/run_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from leo_coverage.constellation import eutelsat_next_like
from leo_coverage.coverage import coverage_half_angle_deg, global_snapshot, sats_in_view

LONDON = (51.5074, -0.1278)
MIN_ELEV = 25.0                         # deg: assumed user-terminal mask
DAY = np.arange(0.0, 86400.0, 30.0)     # 24 h at 30 s steps
OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)


def london_day(con):
    n, max_el = sats_in_view(con, *LONDON, DAY, MIN_ELEV)
    return {
        "mean_in_view": float(n.mean()),
        "min_in_view": int(n.min()),
        "max_in_view": int(n.max()),
        "percent_time_covered": float(100.0 * np.mean(n >= 1)),
        "mean_best_elevation_deg": float(max_el.mean()),
    }, n, max_el


def main():
    base = eutelsat_next_like()
    summary = {"footprint_half_angle_deg": coverage_half_angle_deg(base.altitude_km, MIN_ELEV)}

    # 1. London over one day
    stats, n, max_el = london_day(base)
    summary["london_baseline"] = stats
    hours = DAY / 3600.0
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    ax[0].step(hours, n, where="post", lw=0.8)
    ax[0].set_ylabel(f"Satellites above {MIN_ELEV:.0f} deg")
    ax[0].set_title("Eutelsat Next-like geometry (528 sats, 24 planes, 1,220 km): London")
    ax[1].plot(hours, max_el, lw=0.8)
    ax[1].set_ylabel("Best elevation [deg]")
    ax[1].set_xlabel("Time since epoch [h]")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "london_24h.png", dpi=150)

    # 2. Sensitivity to the (unpublished) inclination
    incs = np.arange(80.0, 98.1, 2.0)
    sweep = [london_day(eutelsat_next_like(inclination_deg=float(i)))[0] for i in incs]
    summary["inclination_sweep"] = {f"{i:.0f}": s for i, s in zip(incs, sweep)}
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(incs, [s["mean_in_view"] for s in sweep], marker="o", label="mean")
    ax.plot(incs, [s["min_in_view"] for s in sweep], marker="s", label="minimum")
    ax.set_xlabel("Inclination [deg]")
    ax.set_ylabel("Satellites in view over London")
    ax.set_title("Coverage sensitivity to inclination (24 h)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "inclination_sweep.png", dpi=150)

    # 3. Global snapshot
    lats, lons, count = global_snapshot(base, 0.0, MIN_ELEV, step_deg=2.0)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(lons, lats, count, shading="auto", cmap="viridis")
    ax.plot(LONDON[1], LONDON[0], "r*", ms=10, label="London")
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(f"Satellites above {MIN_ELEV:.0f} deg elevation at epoch")
    fig.colorbar(im, ax=ax, label="Satellites in view")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "global_snapshot.png", dpi=150)
    summary["global_snapshot_percent_covered"] = float(100.0 * np.mean(count >= 1))

    (OUT / "results.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
