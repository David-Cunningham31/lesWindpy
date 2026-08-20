#!/usr/bin/env python3
from mann_spectral_tilt_common_windlespyWong_timeSeriesReStress import default_config, run_calibration

# Variety 2:
#   - Same as Variety 1, but also applies a local resolved-band energy correction.
#   - The band correction is computed over each height's local resolved frequency
#     range, not over one global frequency range.
#   - Final auto-spectrum area is still renormalised over the resolved frequency
#     band to the Wong-updated variance.

cfg = default_config(
    mode="length_tilt_bands",
    components=("u", "v", "w"),
    log_dir_name="spectralTiltCalibration_lengthTilt_withBands",
    variance_relax=0.0,
    moment_relax=0.70,
    # Conservative default so the band-shape objective does not dominate the
    # direct L-error tilt. Override with MST_SHAPE_RELAX if needed.
    shape_relax=0.35,
    use_windlespy_wong_update=True,
    wong_relaxation_factor=0.9,
    band_correction_uses_resolved_range=True,
    max_log_band_update=1.00,
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
