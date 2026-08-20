#!/usr/bin/env python3
from mann_spectral_tilt_common_windlespyWong_timeSeriesReStress import default_config, run_calibration

# MannHybridTurb generator behaviour:
#   rawMann                 : Reynolds shear stress is plotted, not calibrated.
#   sameComponentCoherence  : auto-spectra use the established calibration;
#                             Reynolds shear stress is plotted, not calibrated.
#   reynoldsShearStress     : the established auto-spectrum calibration is used
#                             and the profile Reynolds shear stress is calibrated.
#
# No standalone u-w cospectrum file is tilted in any revised mode.  The utility
# constructs the Mann-shaped u-w cospectrum internally from the profile stress.
INFLOW_MODE = "sameComponentCoherence"

# Variety 1 (unchanged algorithm and tuned values):
#   - Profile U, R11, R22, R33 and, when enabled by INFLOW_MODE, u'w' use the
#     actual windlespy Wong update.
#   - Spectral tilt direction is controlled by log(L_down/L_target).
#   - No band-energy correction is applied.
#   - Final auto-spectrum area is renormalised over the resolved frequency band
#     to the Wong-updated variance.

cfg = default_config(
    mode="length_tilt",
    inflow_mode=INFLOW_MODE,
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
    # These are enforced from inflow_mode by the common engine. They are shown
    # explicitly here so the intended behaviour is immediately visible.
    update_uw_stress=(INFLOW_MODE == "reynoldsShearStress"),
    update_uw_cospectrum=False,
    use_windlespy_resolved_band=True,
    uw_cospectrum_resolved_band_only=True,
)
raise SystemExit(run_calibration(cfg))
