#!/usr/bin/env python3
from mann_spectral_tilt_common import default_config, run_calibration

# Conservative full-height all-statistics update.
# Explicitly calibrates the full vertical profile, not just z <= 3H.
# Override by exporting MST_FULL_HEIGHT=false or MST_Z_CAL_MAX=<height>.
cfg = default_config(
    mode="hybrid",
    components=("u", "v", "w"),
    variance_relax=0.35,
    moment_relax=0.35,
    shape_relax=0.25,
    profile_relax_U=0.12,
    profile_relax_I=0.25,
    profile_relax_uw=0.15,
    max_log_band_update=0.65,
    max_log_tilt=0.85,
    max_log_uw_update=0.45,
    update_profile_intensity=True,
    update_profile_length=False,
    update_uw_stress=True,
    update_uw_cospectrum=True,
    z_cal_min=-1.0e30,
    z_cal_max=1.0e30,
    freeze_above_zmax=False,
)
raise SystemExit(run_calibration(cfg))
