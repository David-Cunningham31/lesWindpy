#!/usr/bin/env python3
from mann_spectral_tilt_common_windlespyResolved import default_config, run_calibration

# Aggressive all-statistics update. Use to test whether the LES responds to strong spectral tilting.
cfg = default_config(
    mode="hybrid",
    components=("u", "v", "w"),
    variance_relax=0.75,
    moment_relax=0.85,
    shape_relax=0.70,
    profile_relax_U=0.20,
    profile_relax_I=0.40,
    profile_relax_uw=0.30,
    max_log_band_update=1.50,
    max_log_tilt=1.60,
    max_log_uw_update=1.10,
    update_profile_intensity=True,
    update_profile_length=False,
    update_uw_stress=True,
    update_uw_cospectrum=True,
)
raise SystemExit(run_calibration(cfg))
