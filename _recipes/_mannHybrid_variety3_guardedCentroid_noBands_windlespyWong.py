#!/usr/bin/env python3
from mann_spectral_tilt_common_windlespyWong_timeSeriesReStress import default_config, run_calibration

# Variety 3:
#   - Uses centroid difference mu_target - mu_downstream as the spectral-tilt
#     control only when it implies the same tilt direction as log(L_down/L_target).
#   - If centroid direction and integral-length direction contradict each other,
#     the update falls back to the direct log(L_down/L_target) controller.
#   - No band-energy correction is applied.
#   - Final auto-spectrum area is renormalised over the resolved frequency band
#     to the Wong-updated variance.

cfg = default_config(
    mode="guarded_centroid",
    components=("u", "v", "w"),
    log_dir_name="spectralTiltCalibration_guardedCentroid_noBands",
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
