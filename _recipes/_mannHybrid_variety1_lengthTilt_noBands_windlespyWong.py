#!/usr/bin/env python3
from mann_spectral_tilt_common_windlespyWong_timeSeriesReStress import default_config, run_calibration

# Variety 1:
#   - Profile U, R11, R22, R33 and u'w' use the actual windlespy Wong update.
#   - Spectral tilt direction is controlled by log(L_down/L_target).
#   - No band-energy correction is applied.
#   - Final auto-spectrum area is renormalised over the resolved frequency band
#     to the Wong-updated variance.

cfg = default_config(
    mode="length_tilt",
    components=("u", "v", "w"),
    log_dir_name="spectralTiltCalibration_lengthTilt_noBands",
    variance_relax=0.0,
    moment_relax=0.70,
    shape_relax=0.0,
    use_windlespy_wong_update=True,
    wong_relaxation_factor=0.9,
    max_log_band_update=1.50,
    max_log_tilt=1.40,
    max_log_uw_update=1.10,
    update_profile_intensity=True,
    update_profile_length=False,
    update_uw_stress=True,
    update_uw_cospectrum=True,
    use_windlespy_resolved_band=True,
    uw_cospectrum_resolved_band_only=True,
)
raise SystemExit(run_calibration(cfg))
