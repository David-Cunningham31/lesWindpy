# DFSR hybrid covariance/spectral calibration recipe

Created from `_downstreamDFSRHybridSpectralAutocorrelationCalibration.py`.

## New recipe

`_downstreamDFSRHybridCovarianceSpectralCalibration.py`

This keeps the existing auto-spectrum hybrid calibration logic and adds a u-w co-spectral calibration branch:

- high-frequency branch: downstream measured `Cuw(f,z)` from a multitaper cross-spectrum;
- low-frequency branch: downstream measured `Ruw(tau,z)` from u-w cross-covariance;
- signed join-matched PCHIP over `log(-Cuw)` for the usual negative ABL Reynolds shear-stress convention;
- optional diagnostic-only Reynolds-stress area tracking, with no forced area normalization by default;
- realizability clipping against `|Cuw| <= rhoMax * sqrt(Suu*Sww)` using the final calibrated auto-spectra;
- output of the augmented `spectraProfile` with `uwStress`, plus a separate `uwCoSpectrumProfile` containing the calibrated co-spectrum.

## Important default behaviour

The recipe does not globally renormalize the auto-spectra or u-w co-spectrum by default. It reports area/stress errors as diagnostics. This mirrors the existing hybrid approach, where local continuity at the spectral/autocorrelation join is preferred over global variance rescaling.

## New/important settings

```python
INCLUDE_UW_COSPECTRAL_CALIBRATION = True
UW_COSPECTRAL_RELAXATION_FACTOR = 0.35
UW_CROSSCOV_RELAXATION_FACTOR = 0.35
UW_STRESS_RELAXATION_FACTOR = 0.35
RENORMALISE_UW_COSPECTRUM_TO_UPDATED_STRESS = False
UW_FALLBACK_RHO = -0.30
UW_ENFORCE_NEGATIVE_COSPECTRUM = True
UW_RHO_MAX = 0.95
WRITE_AUGMENTED_SPECTRA_PROFILE_WITH_UWSTRESS = True
WRITE_UW_COSPECTRA_PROFILE = True
SPECTRA_PROFILE_UW_FILENAME = "uwCoSpectrumProfile"
```

## File formats written

Augmented `spectraProfile`:

```text
nHeights nFreq
z uwStress Su[0:nFreq] Sv[0:nFreq] Sw[0:nFreq]
```

Separate `uwCoSpectrumProfile`:

```text
nHeights nFreq
z uwStress Cuw[0:nFreq]
```

A backup legacy three-component file is written as `spectraProfile_legacy3comp` when enabled.

## Assumptions

1. The ABL target co-spectrum is predominantly negative, so the stable interpolation variable is `-Cuw`.
2. If a calibrated target or inlet `Cuw` profile is not present, a neutral Kaimal shape is used and normalized to the chosen `uwStress(z)`.
3. The target `uwStress(z)` is read from a profile/spectra column if present; otherwise it is approximated as `UW_FALLBACK_RHO * sigma_u * sigma_w`.
4. The existing DFSR frequency grid is cyclic frequency in Hz, matching the current spectra profile convention.
5. The recipe writes both the augmented spectra profile and a separate co-spectrum profile because DFSR readers may evolve differently.

## Integration note

Place the recipe in `lesWindpy/_recipes/`. It is intended as a drop-in alternative to `_downstreamDFSRHybridSpectralAutocorrelationCalibration.py`.
