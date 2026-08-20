# -*- coding: utf-8 -*-
"""
Recommended MannHybridTurb downstream calibration recipe.

Method: bounded Wong-style frequency-by-frequency residual update.
- Multitaper downstream spectra.
- Smooth log-bin/PCHIP PSD fit.
- Wong-style adaptive spectral correction, bounded per iteration.
- Resolved-band variance renormalisation.
- Signed u-w co-spectrum updated through bounded coherence rho_uw.
- Writes MannHybridTurb-compatible spectraProfile and uwCoSpectrumProfile.
- Writes Melaku-style 8-panel profiles between iterations.

Exit status:
    0 = converged/stagnated; stop Slurm loop
    1 = update written; regenerate MannHybridTurb and continue Slurm loop
    >1 = error
"""

import os
import sys
from pathlib import Path

# Allow importing common file when copied into windlespy/_recipes.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mann_calibration_common import CalibrationConfig, run_calibration


def _env_float(name, default):
    return float(os.environ.get(name, default))


def _env_bool(name, default):
    v = os.environ.get(name, str(default)).strip().lower()
    return v in ("1", "true", "yes", "on")


cfg = CalibrationConfig(
    method_name="wong",
    spectral_relaxation=_env_float("MANN_CAL_SPECTRAL_RELAX", 0.45),
    mean_relaxation=_env_float("MANN_CAL_MEAN_RELAX", 0.45),
    variance_relaxation=_env_float("MANN_CAL_VARIANCE_RELAX", 0.35),
    uw_relaxation=_env_float("MANN_CAL_UW_RELAX", 0.30),
    min_factor=_env_float("MANN_CAL_MIN_FACTOR", 0.50),
    max_factor=_env_float("MANN_CAL_MAX_FACTOR", 2.00),
    psd_estimator=os.environ.get("MANN_CAL_PSD_ESTIMATOR", "multitaper"),
    multitaper_time_bandwidth=_env_float("MANN_CAL_MT_NW", 3.5),
    smooth_bins=int(os.environ.get("MANN_CAL_SMOOTH_BINS", 56)),
    low_frequency_mode=os.environ.get("MANN_CAL_LOW_FREQ_MODE", os.environ.get("MANN_CAL_LOW_FREQUENCY_MODE", "plateau")),
    low_plateau_hz=float(os.environ["MANN_CAL_LOW_PLATEAU_HZ"]) if "MANN_CAL_LOW_PLATEAU_HZ" in os.environ else None,
    tail_slope_after_hz=float(os.environ["MANN_CAL_TAIL_AFTER_HZ"]) if "MANN_CAL_TAIL_AFTER_HZ" in os.environ else None,
    renormalise_variance=_env_bool("MANN_CAL_RENORMALISE_VARIANCE", True),
    update_cospectrum=_env_bool("MANN_CAL_UPDATE_UW", True),
    update_profile=_env_bool("MANN_CAL_UPDATE_PROFILE", True),
)

raise SystemExit(run_calibration(cfg))
