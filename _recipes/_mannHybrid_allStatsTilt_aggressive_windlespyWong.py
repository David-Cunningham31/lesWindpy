#!/usr/bin/env python3
from mann_spectral_tilt_common_windlespyWong_timeSeriesReStress import default_config, run_calibration

# Aggressive all-statistics spectral-tilt update with the actual windlespy Wong
# profile update.
#
# Important:
#   - U, R_11, R_22, R_33 and u'w' are updated through
#     windlespy._profileCalibration.new_dfsr_profile_array using the classic
#     Wong adaptive correction and relaxation_factor=0.9 by default.
#   - The downstream Reynolds stresses are calculated from the post-burn-in LES
#     velocity time series directly: R_ii = var(u_i) and u'w' = mean(u'w').
#     They are NOT computed from resolved-band spectral integrals.
#   - Lu/Lv/Lw profile columns are kept unchanged in this wrapper because
#     length-scale correction is handled by spectral shape/centroid tilting.
#     Set update_profile_length=True below if you deliberately want direct
#     Wong updates to the L columns as well.
#   - Spectral centroids and final spectrum renormalisation use the
#     windlespy/Geleta resolved frequency limits.
#   - Spectral tilt is applied only to the auto-spectra S_uu/S_vv/S_ww.
#   - C_uw is not spectrally tilted; its shape is preserved and its resolved-band
#     area is scaled to the Wong-updated u'w'.

cfg = default_config(
    mode="hybrid",
    components=("u", "v", "w"),
    # Area/normal-stress update is controlled by the actual windlespy Wong
    # function. variance_relax is retained for legacy mode only.
    variance_relax=0.0,
    moment_relax=0.85,
    shape_relax=0.70,
    use_windlespy_wong_update=True,
    wong_relaxation_factor=0.9,
    max_log_band_update=1.50,
    max_log_tilt=1.60,
    max_log_uw_update=1.10,
    update_profile_intensity=True,
    update_profile_length=False,
    update_uw_stress=True,
    update_uw_cospectrum=True,
    use_windlespy_resolved_band=True,
    uw_cospectrum_resolved_band_only=True,
)
raise SystemExit(run_calibration(cfg))
