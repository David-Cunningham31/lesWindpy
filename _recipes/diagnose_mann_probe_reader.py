# -*- coding: utf-8 -*-
"""Diagnose postProcessing/probes2 parsing for MannHybrid calibration scripts."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mann_calibration_common import case_paths, load_sim_init, load_downstream_probe_velocity, clean_time_series_for_spectra


def main() -> int:
    case_dir = os.environ.get("CASE_DIR", os.getcwd())
    p = case_paths(case_dir)
    sim = load_sim_init(case_dir)
    burn = float(sim.get("burn_in_time", 0.0))
    print(f"CASE_DIR={case_dir}")
    print(f"probes2={p['probes2']}")
    print(f"burn_in_time={burn}")
    vel, t, info = load_downstream_probe_velocity(p["probes2"])
    print("Raw concatenated probe data:")
    print(f"  vel shape: {vel.shape}")
    print(f"  time rows: {t.size}")
    print(f"  finite times: {np.isfinite(t).sum()}")
    print(f"  time range: {np.nanmin(t):.12g} -> {np.nanmax(t):.12g}")
    diffs = np.diff(np.sort(t[np.isfinite(t)]))
    print(f"  duplicate/zero diffs before burn filter: {np.sum(np.abs(diffs) <= 1e-12)}")
    vel_c, t_c, dt = clean_time_series_for_spectra(vel, t, burn=burn, min_samples=8)
    print("After burn-in filter + duplicate removal:")
    print(f"  vel shape: {vel_c.shape}")
    print(f"  time rows: {t_c.size}")
    print(f"  time range: {t_c[0]:.12g} -> {t_c[-1]:.12g}")
    print(f"  dt: {dt:.12g}")
    print(f"  fs: {1.0/dt:.12g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
