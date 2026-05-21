# -*- coding: utf-8 -*-
"""
Standalone downstream DFSR hybrid spectral/autocorrelation + uw co-spectral/cross-covariance calibration recipe.

This script combines the two testing branches we have been developing:

1. Spectral calibration supplies the baseline updated spectrum using the fitted
   downstream multitaper/spline spectrum and the Wong-style residual update.
2. Autocorrelation calibration now uses the full raw updated autocorrelation
   function to construct the low-frequency branch. That branch is vertically
   shifted with a single multiplier so that it matches the first
   spectral-calibration knot.
3. Eight low-frequency autocorrelation knots are then combined with the normal
   spectral-calibration knots, and one log-log PCHIP is fitted through the
   merged knot set.
4. The final hybrid spectrum is not globally renormalised by default; continuity
   is controlled locally at the spectral/autocorrelation join.

The script is intentionally self-contained and does not require editing
windLespy. It reuses windLespy for case IO, profile processing, downstream
spectral estimation, and writing DFSR files. This version additionally calibrates
the u-w co-spectrum using downstream co-spectra at higher frequencies and
downstream cross-covariance at lower frequencies.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.signal.windows import dpss
from scipy.fft import rfft, rfftfreq

try:
    from scipy.optimize import curve_fit
except Exception:
    curve_fit = None

# -----------------------------------------------------------------------------
# Robust paths for Windows/OneDrive
# -----------------------------------------------------------------------------

def _windows_long_path(path):
    path = os.path.abspath(path)
    if os.name != "nt":
        return path
    bs = chr(92)
    long_prefix = bs * 2 + "?" + bs
    unc_prefix = bs * 2 + "?" + bs + "UNC" + bs
    if path.startswith(long_prefix):
        return path
    if path.startswith(bs * 2):
        return unc_prefix + path[2:]
    return long_prefix + path


def safe_makedirs(path):
    os.makedirs(_windows_long_path(path), exist_ok=True)


def safe_savefig(fig, path, dpi=300, bbox_inches="tight"):
    path = os.path.abspath(path)
    safe_makedirs(os.path.dirname(path))
    fig.savefig(_windows_long_path(path), dpi=dpi, bbox_inches=bbox_inches)


cwd = os.path.dirname(os.path.abspath(__file__))
windlespy_path = os.path.abspath(os.path.join(cwd, "..", ".."))
sys.path.append(windlespy_path)
import windlespy as LES
sys.path.remove(windlespy_path)


#%% --------------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------------

case_path = os.environ["CASE_DIR"]
# For manual testing, comment the line above and uncomment/edit this line:
# case_path = r"C:\Users\david\OneDrive\Documents\PhD\Year 1\Spectral Calibration Method\empty_domain_test_case\spectralDFSR"

downstream_probes_folder = os.path.join(case_path, "postProcessing", "probes2")

WRITE_RESULTS = True
WRITE_ITER_SPECTRA = True

MEAN_PROFILE_RELAXATION_FACTOR = 0.5
SPECTRAL_RELAXATION_FACTOR = 0.35
AUTOCORR_RELAXATION_FACTOR = 0.35
VARIANCE_RELAXATION_FACTOR = 0.35

# Downstream variance used in the resolved-band Wong variance update.
# "spectra_smoothed" is most consistent with the spectral update; "time_series"
# uses downstream_profile_array variances.
DOWNSTREAM_VARIANCE_SOURCE = "spectra_smoothed"

# Spectral-calibration downstream PSD settings.
PSD_FIT_MIN = 0.5
PSD_FIT_MAX = None
PSD_MIN_POINTS_PER_BIN = 4
PSD_LOW_PLATEAU_BAND = (0.20, 1.00)
PSD_LEFT_MODE = "band_median"
PSD_RIGHT_MODE = "slope"
PSD_RIGHT_SLOPE_CLIP = (-8.0, 0.0)
PSD_BAND_CONFIG = [
    (1.0, 2.0, 3),
    (2.0, 5.0, 4),
    (5.0, 10.0, 4),
    (10.0, None, 10),
]
PSD_SMOOTH_KNOTS = True
PSD_KNOT_SMOOTH_KERNEL = [1, 2, 1]
PSD_HEIGHT_KERNEL = [1, 2, 1]

# Transfer-function diagnostics only. Final spectral update uses direct Wong
# residual form through windLespy.
TRANSFER_SAVGOL_WINDOW = 121
TRANSFER_SAVGOL_POLYORDER = 2

# Resolved-band settings.
USE_HEIGHT_DEPENDENT_RESOLVED_FMAX = True
RESOLVED_F_MIN_OVERRIDE = None
RESOLVED_F_MAX_OVERRIDE = None

# Autocorrelation settings.
TAU_MAX_FACTOR_OF_MAX_TARGET_T = 20.0
TAU_MAX_MIN_SECONDS = None
TAU_MAX_MAX_SECONDS = None

USE_EXPONENTIAL_RHO_FIT = False
EXP_FIT_RHO_MIN = 0.08
EXP_FIT_RHO_MAX = 0.98
EXP_FIT_MIN_POINTS = 8
EXP_FIT_P_BOUNDS = (0.45, 4.0)
EXP_FIT_T_BOUNDS_FACTOR = (0.02, 20.0)
EXP_ZERO_TOL = 1e-2
EXP_ACCEPT_RAW_ZERO_FACTOR = 1.75
EXP_ACCEPT_RAW_ZERO_ABS_PAD = 4.0
EXP_ZERO_REF_FACTOR = 2.25
EXP_MIN_ZERO_TAU_FACTOR = 0.35
EXP_MAX_ZERO_TAU_FACTOR = 2.50
AUTOCORR_ZERO_TOL = EXP_ZERO_TOL
AUTOCORR_ZERO_PERSISTENCE_POINTS = 3
AUTOCORR_ZERO_LOOKAHEAD_POINTS = 8

APPLY_AUTOCOVARIANCE_TAPER = True  # still taper the raw autocovariance tail before the cosine transform
TAPER_START_FRACTION = 0.75
CLIP_NEGATIVE_SPECTRUM_TO_FLOOR = True

# Combined-knot low-frequency autocorrelation branch.
HYBRID_ENABLE_AUTOCORR_LOW_FREQ_KNOTS = True

# Vertically shift the full autocorrelation-derived low-frequency branch so it
# matches the first spectral-calibration knot. This is the key continuity fix.
HYBRID_MATCH_LOW_FREQ_TO_SPECTRAL_JOIN = True
HYBRID_JOIN_SCALE_MIN = 0.33
HYBRID_JOIN_SCALE_MAX = 3.0

# Keep 8 explicit log-spaced low-frequency knots from the shifted
# autocorrelation branch, below the first spectral knot.
HYBRID_LOW_FREQ_N_KNOTS = 8
HYBRID_LOW_FREQ_MAX_FRACTION_OF_FIRST_KNOT = 0.95
HYBRID_LOW_FREQ_MIN_POINTS_PER_BIN = 2  # fallback only

# Post-shift safety bounds relative to the spectral branch. These are not the
# main matching mechanism; they only stop extreme autocorrelation artefacts.
HYBRID_LOW_FREQ_RATIO_MIN = 0.50
HYBRID_LOW_FREQ_RATIO_MAX = 2.00
HYBRID_FIRST_KNOT_FALLBACK = PSD_BAND_CONFIG[0][0]
HYBRID_ADD_ENDPOINT_ANCHORS = True

# Variance enforcement.
# Leave these off for this hybrid version: the low-frequency branch is anchored
# locally to the spectral branch instead of globally re-scaling the full result.
RENORMALISE_SPECTRAL_BASELINE_TO_UPDATED_VARIANCE = False
RENORMALISE_HYBRID_TO_UPDATED_VARIANCE = False


# u-w co-spectrum / cross-covariance calibration.
# This branch mirrors the auto-spectrum hybrid method but uses:
#   high f: downstream Cuw(f) co-spectral residual update
#   low f : downstream Ruw(tau) cross-covariance update
# The final area (integrated uw stress) is diagnostic by default, not forced,
# matching the auto-spectral calibration philosophy above.
INCLUDE_UW_COSPECTRAL_CALIBRATION = True
UW_COSPECTRAL_RELAXATION_FACTOR = 0.35
UW_CROSSCOV_RELAXATION_FACTOR = 0.35
UW_STRESS_RELAXATION_FACTOR = 0.35
RENORMALISE_UW_COSPECTRUM_TO_UPDATED_STRESS = False

# If no uwStress column is found in the target/profile/spectraProfile files,
# fall back to rho_uw * sigma_u * sigma_w. Use a negative rho for neutral ABL shear.
UW_FALLBACK_RHO = -0.30
UW_TARGET_STRESS_COLUMN_CANDIDATES = (
    "uwStress", "UWStress", "Ruw", "R13", "uw", "u'w'", "u_w", "cov_uw"
)

# Kaimal is used only to initialise the target/inlet Cuw shape when a calibrated
# Cuw profile is not already present. The discrete shape is rescaled so its area
# equals the chosen uwStress profile.
UW_USE_KAIMAL_IF_PROFILE_COSPECTRUM_MISSING = True
UW_ENFORCE_NEGATIVE_COSPECTRUM = True
UW_MAGNITUDE_FLOOR = 1e-20
UW_RHO_MAX = 0.95

# Smoothing/binning for measured downstream Cuw. These default to the auto-PSD
# settings, but are separate because co-spectra are much noisier.
UW_PSD_FIT_MIN = PSD_FIT_MIN
UW_PSD_FIT_MAX = PSD_FIT_MAX
UW_MIN_POINTS_PER_BIN = PSD_MIN_POINTS_PER_BIN
UW_BAND_CONFIG = PSD_BAND_CONFIG
UW_LOW_PLATEAU_BAND = PSD_LOW_PLATEAU_BAND
UW_LEFT_MODE = PSD_LEFT_MODE
UW_RIGHT_MODE = PSD_RIGHT_MODE
UW_RIGHT_SLOPE_CLIP = PSD_RIGHT_SLOPE_CLIP
UW_SMOOTH_KNOTS = PSD_SMOOTH_KNOTS
UW_KNOT_SMOOTH_KERNEL = PSD_KNOT_SMOOTH_KERNEL
UW_HEIGHT_KERNEL = PSD_HEIGHT_KERNEL

# Join-matched low-frequency cross-covariance branch.
UW_HYBRID_LOW_FREQ_N_KNOTS = HYBRID_LOW_FREQ_N_KNOTS
UW_HYBRID_LOW_FREQ_MAX_FRACTION_OF_FIRST_KNOT = HYBRID_LOW_FREQ_MAX_FRACTION_OF_FIRST_KNOT
UW_HYBRID_LOW_FREQ_RATIO_MIN = HYBRID_LOW_FREQ_RATIO_MIN
UW_HYBRID_LOW_FREQ_RATIO_MAX = HYBRID_LOW_FREQ_RATIO_MAX
UW_HYBRID_JOIN_SCALE_MIN = HYBRID_JOIN_SCALE_MIN
UW_HYBRID_JOIN_SCALE_MAX = HYBRID_JOIN_SCALE_MAX
UW_HYBRID_FIRST_KNOT_FALLBACK = HYBRID_FIRST_KNOT_FALLBACK
UW_HYBRID_ADD_ENDPOINT_ANCHORS = HYBRID_ADD_ENDPOINT_ANCHORS

# Write formats for the modified DFSR utility. The augmented spectraProfile is:
#   nHeights nFreq
#   z uwStress Su[0:nFreq] Sv[0:nFreq] Sw[0:nFreq]
# and the separate uwCoSpectrumProfile is:
#   nHeights nFreq
#   z uwStress Cuw[0:nFreq]
WRITE_AUGMENTED_SPECTRA_PROFILE_WITH_UWSTRESS = True
WRITE_UW_COSPECTRA_PROFILE = True
WRITE_LEGACY_3COMP_SPECTRA_BACKUP = True
SPECTRA_PROFILE_UW_FILENAME = "uwCoSpectrumProfile"


# Tail treatment.
APPLY_POWER_LAW_TAIL = True
POWER_LAW_TAIL_CUTOFF_FACTOR = 0.5
POWER_LAW_TAIL_SLOPE = -5.0 / 3.0

FLOOR = 1e-16
SPECTRUM_FLOOR_FOR_WRITE = 1e-16

# Plot settings.
Z_MAX_FACTOR_PLOTS = 1.5
Z_MAX_FACTOR_FINAL_SPECTRA = 3.0
N_HEIGHTS_TO_PLOT = 8
COMPONENT_NAMES = ("u", "v", "w")


#%% --------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _trapz(y, x, axis=-1):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    return np.trapz(y, x, axis=axis)


def _band_mask_for_height(freq_array, h_id, f_min=None, f_max=None):
    f = np.asarray(freq_array, dtype=float)
    mask = np.isfinite(f) & (f > 0.0)
    if f_min is not None:
        mask &= f >= float(f_min)
    if f_max is not None:
        if np.ndim(f_max) == 0:
            f_hi = float(f_max)
        else:
            f_hi = float(np.asarray(f_max, dtype=float)[h_id])
        mask &= f <= f_hi
    return mask


def integrate_spectra_area(freq_array, spectra_array, f_min=None, f_max=None):
    f = np.asarray(freq_array, dtype=float)
    spectra = np.asarray(spectra_array, dtype=float)
    area = np.zeros(spectra.shape[:2], dtype=float)
    for h_id in range(spectra.shape[1]):
        mask = _band_mask_for_height(f, h_id, f_min=f_min, f_max=f_max)
        if np.count_nonzero(mask) < 2:
            raise ValueError(f"Not enough frequency points in integration band for height index {h_id}.")
        for comp_id in range(spectra.shape[0]):
            area[comp_id, h_id] = _trapz(spectra[comp_id, h_id, mask], f[mask], axis=0)
    return area


def renormalise_spectra_to_variance(freq_array, spectra_array, target_variance, floor=1e-16, f_min=None, f_max=None):
    spectra = np.maximum(np.asarray(spectra_array, dtype=float), floor)
    target = np.maximum(np.asarray(target_variance, dtype=float), floor)
    area = np.maximum(integrate_spectra_area(freq_array, spectra, f_min=f_min, f_max=f_max), floor)
    out = spectra.copy()
    for comp_id in range(out.shape[0]):
        for h_id in range(out.shape[1]):
            out[comp_id, h_id, :] *= target[comp_id, h_id] / area[comp_id, h_id]
    return np.maximum(out, floor)


def wong_update_variance(inlet_sigma2, target_sigma2, downstream_sigma2, relaxation_factor=0.35, floor=1e-16):
    inlet = np.maximum(np.asarray(inlet_sigma2, dtype=float), floor)
    target = np.maximum(np.asarray(target_sigma2, dtype=float), floor)
    downstream = np.maximum(np.asarray(downstream_sigma2, dtype=float), floor)
    updated = inlet + relaxation_factor * (inlet / downstream) * (target - downstream)
    return np.maximum(updated, floor)


def _selected_height_ids(z_array, body_height, z_min=0.0, z_max_factor=1.5, n_heights=8):
    z = np.asarray(z_array, dtype=float)
    mask = (z >= z_min) & (z <= z_max_factor * body_height)
    ids = np.where(mask)[0]
    if len(ids) == 0:
        return np.array([], dtype=int)
    n = min(n_heights, len(ids))
    return np.unique(np.round(np.linspace(ids[0], ids[-1], n)).astype(int))


#%% --------------------------------------------------------------------------
# Autocorrelation helpers
# ---------------------------------------------------------------------------

def make_tau_array_from_target_lengths(target_profile_array, dt, factor=20.0, tau_min=None, tau_max=None):
    U = np.maximum(target_profile_array[:, 0], 1e-12)
    L = np.maximum(target_profile_array[:, 4:7], 0.0)
    T_targets = L / U[:, np.newaxis]
    T_ref = float(np.nanmax(T_targets))
    if not np.isfinite(T_ref) or T_ref <= 0.0:
        T_ref = 1.0
    tau_end = factor * T_ref
    if tau_min is not None:
        tau_end = max(tau_end, float(tau_min))
    if tau_max is not None:
        tau_end = min(tau_end, float(tau_max))
    n_lags = max(4, int(np.floor(tau_end / dt)) + 1)
    return np.arange(n_lags, dtype=float) * dt


def autocorrelation_fft_1d(x, max_lags=None):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return np.full(max_lags if max_lags is not None else 1, np.nan)
    x = x - np.mean(x)
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(2 * n - 1)))
    X = np.fft.rfft(x, n=nfft)
    acov = np.fft.irfft(X * np.conjugate(X), n=nfft)[:n]
    acov = acov / n
    if abs(acov[0]) <= 0.0 or not np.isfinite(acov[0]):
        rho = np.full(n, np.nan)
    else:
        rho = acov / acov[0]
    if max_lags is not None:
        rho = rho[: min(int(max_lags), len(rho))]
    return rho


def autocorrelation_array_from_velocity(vel_array_3d, max_lags):
    n_comp, _, n_heights = vel_array_3d.shape
    rho = np.zeros((n_comp, n_heights, max_lags), dtype=float)
    for comp_id in range(n_comp):
        for h_id in range(n_heights):
            rho_1d = autocorrelation_fft_1d(vel_array_3d[comp_id, :, h_id], max_lags=max_lags)
            if len(rho_1d) < max_lags:
                tmp = np.full(max_lags, np.nan)
                tmp[: len(rho_1d)] = rho_1d
                rho_1d = tmp
            rho[comp_id, h_id, :] = rho_1d
    return rho


def autocovariance_from_spectrum_array(freq_array, spectra_array, tau_array, floor=1e-16, f_min=None, f_max=None):
    f = np.asarray(freq_array, dtype=float)
    tau = np.asarray(tau_array, dtype=float)
    spectra = np.maximum(np.asarray(spectra_array, dtype=float), floor)
    out = np.zeros((spectra.shape[0], spectra.shape[1], len(tau)), dtype=float)
    for h_id in range(spectra.shape[1]):
        mask = _band_mask_for_height(f, h_id, f_min=f_min, f_max=f_max)
        if np.count_nonzero(mask) < 2:
            raise ValueError(f"Not enough frequency points in resolved band for height index {h_id}.")
        f_band = f[mask]
        cos_matrix = np.cos(2.0 * np.pi * tau[:, np.newaxis] * f_band[np.newaxis, :])
        for comp_id in range(spectra.shape[0]):
            integrand = spectra[comp_id, h_id, mask][np.newaxis, :] * cos_matrix
            out[comp_id, h_id, :] = _trapz(integrand, f_band, axis=1)
    return out


def normalise_autocovariance_to_rho(autocovariance_array, floor=1e-30):
    R = np.asarray(autocovariance_array, dtype=float)
    R0 = R[:, :, 0]
    rho = np.full_like(R, np.nan)
    for comp_id in range(R.shape[0]):
        for h_id in range(R.shape[1]):
            if np.isfinite(R0[comp_id, h_id]) and abs(R0[comp_id, h_id]) > floor:
                rho[comp_id, h_id, :] = R[comp_id, h_id, :] / R0[comp_id, h_id]
    return rho


def integral_time_scale_from_rho(tau_array, rho_array, first_zero=True):
    tau = np.asarray(tau_array, dtype=float)
    rho = np.asarray(rho_array, dtype=float)
    valid = np.isfinite(tau) & np.isfinite(rho)
    tau = tau[valid]
    rho = rho[valid]
    if len(tau) < 2:
        return np.nan
    if first_zero:
        zero_ids = np.where(rho <= 0.0)[0]
        i_end = int(zero_ids[0]) if len(zero_ids) > 0 else len(rho) - 1
    else:
        i_end = len(rho) - 1
    if i_end < 1:
        return 0.0
    return float(_trapz(rho[: i_end + 1], tau[: i_end + 1], axis=0))


def integral_length_array_from_rho(tau_array, rho_array, U_array):
    out = np.zeros((rho_array.shape[0], rho_array.shape[1]), dtype=float)
    for comp_id in range(rho_array.shape[0]):
        for h_id in range(rho_array.shape[1]):
            T = integral_time_scale_from_rho(tau_array, rho_array[comp_id, h_id, :])
            out[comp_id, h_id] = U_array[h_id] * T
    return out


def first_zero_index(rho_1d, start_index=1, zero_tol=0.0, persistence_points=1, lookahead_points=0):
    rho = np.asarray(rho_1d, dtype=float)
    n = len(rho)
    if n == 0:
        return 0
    start_index = max(int(start_index), 0)
    zero_tol = max(float(zero_tol), 0.0)
    persistence_points = max(int(persistence_points), 1)
    lookahead_points = max(int(lookahead_points), 0)
    finite = np.isfinite(rho)

    ids = np.where(finite & (rho <= 0.0))[0]
    ids = ids[ids >= start_index]
    if len(ids) > 0:
        return int(ids[0])

    if zero_tol > 0.0:
        for i in range(start_index, n):
            if (not finite[i]) or rho[i] > zero_tol:
                continue
            j_persist = min(n, i + persistence_points)
            persist = rho[i:j_persist]
            persist = persist[np.isfinite(persist)]
            if len(persist) == 0 or np.nanmax(persist) > zero_tol:
                continue
            j_look = min(n, i + persistence_points + lookahead_points)
            look = rho[i:j_look]
            look = look[np.isfinite(look)]
            if len(look) > 0 and np.nanmedian(look) <= zero_tol:
                return int(i)
    return n - 1


def _moving_average_1d(x, window=5):
    x = np.asarray(x, dtype=float)
    window = int(window)
    if window <= 1 or len(x) < 3:
        return x.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    xp = np.pad(x, pad, mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(xp, kernel, mode="valid")


def _stretched_exp(tau, T, p):
    tau = np.asarray(tau, dtype=float)
    T = max(float(T), 1e-12)
    p = max(float(p), 1e-6)
    return np.exp(-np.power(np.maximum(tau, 0.0) / T, p))


def _zero_for_stretched_exp(T, p, zero_tol):
    zero_tol = float(np.clip(zero_tol, 1e-8, 0.95))
    return float(T * (-np.log(zero_tol)) ** (1.0 / p))


def _zero_shifted_stretched_exp(tau, T, p, tau_zero):
    tau = np.asarray(tau, dtype=float)
    tau_zero = max(float(tau_zero), 1e-12)
    e0 = _stretched_exp(tau_zero, T, p)
    denom = max(1.0 - e0, 1e-12)
    rho = (_stretched_exp(tau, T, p) - e0) / denom
    return np.clip(rho, -1.0, 1.0)


def fit_stretched_exponential_decay(
    tau_array,
    rho_raw_1d,
    raw_zero_tau,
    reference_zero_tau,
    dt,
    rho_min=0.08,
    rho_max=0.98,
    min_points=8,
    p_bounds=(0.45, 4.0),
    T_bounds_factor=(0.02, 20.0),
    zero_tol=1e-2,
    accept_raw_zero_factor=1.75,
    accept_raw_zero_abs_pad=4.0,
    zero_ref_factor=2.25,
    min_zero_tau_factor=0.35,
    max_zero_tau_factor=2.50,
):
    tau = np.asarray(tau_array, dtype=float)
    rho = np.asarray(rho_raw_1d, dtype=float)
    dt = float(dt)
    finite = np.isfinite(tau) & np.isfinite(rho)
    tau_fit_limit = max(float(raw_zero_tau), float(reference_zero_tau), 8.0 * dt)
    rho_smooth = _moving_average_1d(rho, window=5)
    fit_mask = finite & (tau > 0.0) & (tau <= tau_fit_limit) & (rho > float(rho_min)) & (rho < float(rho_max))
    low_ids = np.where(finite & (tau > 0.0) & (rho_smooth < float(rho_min)))[0]
    if len(low_ids) > 0:
        fit_mask[int(low_ids[0]) + 1 :] = False
    ids = np.where(fit_mask)[0]
    if len(ids) < int(min_points):
        fit_mask = finite & (tau > 0.0) & (tau <= tau_fit_limit) & (rho > max(0.02, 0.5 * float(rho_min))) & (rho < float(rho_max))
        ids = np.where(fit_mask)[0]

    used_fallback = False
    ref = max(float(reference_zero_tau), dt)
    T_guess = max(ref / max((-np.log(max(zero_tol, 1e-8))), 1e-12), dt)
    p_guess = 1.0

    if len(ids) < max(3, int(min_points) // 2):
        used_fallback = True
        T_fit = T_guess
        p_fit = p_guess
    else:
        tau_data = tau[ids]
        rho_data = np.clip(rho[ids], 1e-6, 0.999999)
        if curve_fit is not None:
            try:
                T_low = max(dt, T_bounds_factor[0] * ref)
                T_high = max(T_low * 1.01, T_bounds_factor[1] * ref)
                popt, _ = curve_fit(
                    _stretched_exp,
                    tau_data,
                    rho_data,
                    p0=(T_guess, p_guess),
                    bounds=([T_low, p_bounds[0]], [T_high, p_bounds[1]]),
                    maxfev=20000,
                )
                T_fit, p_fit = float(popt[0]), float(popt[1])
            except Exception:
                used_fallback = True
                log_tau = np.log(np.maximum(tau_data, dt))
                log_y = np.log(np.maximum(-np.log(rho_data), 1e-8))
                slope, intercept = np.polyfit(log_tau, log_y, 1)
                p_fit = float(np.clip(slope, p_bounds[0], p_bounds[1]))
                T_fit = float(np.exp(-intercept / max(p_fit, 1e-8)))
        else:
            used_fallback = True
            log_tau = np.log(np.maximum(tau_data, dt))
            log_y = np.log(np.maximum(-np.log(rho_data), 1e-8))
            slope, intercept = np.polyfit(log_tau, log_y, 1)
            p_fit = float(np.clip(slope, p_bounds[0], p_bounds[1]))
            T_fit = float(np.exp(-intercept / max(p_fit, 1e-8)))

    fitted_zero_tau = _zero_for_stretched_exp(T_fit, p_fit, zero_tol)
    fitted_zero_tau = float(np.clip(
        fitted_zero_tau,
        max(min_zero_tau_factor * ref, 2.0 * dt),
        max(max_zero_tau_factor * ref, 3.0 * dt),
    ))
    raw_zero_tau = float(raw_zero_tau)
    raw_is_reasonable = (
        np.isfinite(raw_zero_tau)
        and raw_zero_tau > 2.0 * dt
        and raw_zero_tau <= accept_raw_zero_factor * fitted_zero_tau + accept_raw_zero_abs_pad * dt
        and raw_zero_tau <= zero_ref_factor * ref + accept_raw_zero_abs_pad * dt
    )
    if raw_is_reasonable:
        selected_zero_tau = raw_zero_tau
        zero_source = "raw"
    else:
        selected_zero_tau = fitted_zero_tau
        zero_source = "fit"
    return {
        "T_fit": float(T_fit),
        "p_fit": float(p_fit),
        "fitted_zero_tau": float(fitted_zero_tau),
        "selected_zero_tau": float(selected_zero_tau),
        "zero_source": zero_source,
        "used_fallback": bool(used_fallback),
        "n_fit_points": int(len(ids)),
    }


def build_exponential_fitted_updated_rho(tau_array, rho_raw, rho_target, rho_downstream, dt, zero_tol=1e-2, zero_persistence_points=3, zero_lookahead_points=8, **fit_kwargs):
    tau = np.asarray(tau_array, dtype=float)
    rho_raw = np.asarray(rho_raw, dtype=float)
    rho_target = np.asarray(rho_target, dtype=float)
    rho_downstream = np.asarray(rho_downstream, dtype=float)
    rho_out = np.zeros_like(rho_raw)
    n_tau = rho_raw.shape[2]
    diagnostics = {
        "raw_zero_id": np.zeros(rho_raw.shape[:2], dtype=int),
        "selected_zero_id": np.zeros(rho_raw.shape[:2], dtype=int),
        "target_zero_id": np.zeros(rho_raw.shape[:2], dtype=int),
        "downstream_zero_id": np.zeros(rho_raw.shape[:2], dtype=int),
        "fit_T": np.zeros(rho_raw.shape[:2], dtype=float),
        "fit_p": np.zeros(rho_raw.shape[:2], dtype=float),
        "fitted_zero_tau": np.zeros(rho_raw.shape[:2], dtype=float),
        "selected_zero_tau": np.zeros(rho_raw.shape[:2], dtype=float),
        "zero_source": np.empty(rho_raw.shape[:2], dtype=object),
        "n_fit_points": np.zeros(rho_raw.shape[:2], dtype=int),
        "fit_used_fallback": np.zeros(rho_raw.shape[:2], dtype=bool),
    }
    for comp_id in range(rho_raw.shape[0]):
        for h_id in range(rho_raw.shape[1]):
            raw_1d = rho_raw[comp_id, h_id, :]
            target_1d = rho_target[comp_id, h_id, :]
            downstream_1d = rho_downstream[comp_id, h_id, :]
            raw_zero_id = first_zero_index(raw_1d, 1, zero_tol, zero_persistence_points, zero_lookahead_points)
            target_zero_id = first_zero_index(target_1d, 1, zero_tol, zero_persistence_points, zero_lookahead_points)
            downstream_zero_id = first_zero_index(downstream_1d, 1, zero_tol, zero_persistence_points, zero_lookahead_points)
            raw_zero_tau = tau[raw_zero_id]
            target_zero_tau = tau[target_zero_id]
            downstream_zero_tau = tau[downstream_zero_id]
            reference_zero_tau = float(np.nanmedian([target_zero_tau, downstream_zero_tau, raw_zero_tau]))
            if not np.isfinite(reference_zero_tau) or reference_zero_tau <= 0.0:
                reference_zero_tau = max(target_zero_tau, downstream_zero_tau, 4.0 * dt)
            fit = fit_stretched_exponential_decay(tau, raw_1d, raw_zero_tau, reference_zero_tau, dt, zero_tol=zero_tol, **fit_kwargs)
            selected_zero_id = int(np.argmin(np.abs(tau - fit["selected_zero_tau"])))
            selected_zero_id = int(np.clip(selected_zero_id, 1, n_tau - 1))
            fitted = _zero_shifted_stretched_exp(tau, fit["T_fit"], fit["p_fit"], tau[selected_zero_id])
            updated_1d = np.zeros(n_tau, dtype=float)
            updated_1d[: selected_zero_id + 1] = fitted[: selected_zero_id + 1]
            updated_1d[0] = 1.0
            updated_1d[selected_zero_id] = 0.0
            for i in range(selected_zero_id + 1, n_tau):
                offset = i - selected_zero_id
                target_i = target_zero_id + offset
                updated_1d[i] = target_1d[target_i] if target_i < n_tau else target_1d[-1]
            rho_out[comp_id, h_id, :] = np.clip(updated_1d, -1.0, 1.0)
            diagnostics["raw_zero_id"][comp_id, h_id] = raw_zero_id
            diagnostics["selected_zero_id"][comp_id, h_id] = selected_zero_id
            diagnostics["target_zero_id"][comp_id, h_id] = target_zero_id
            diagnostics["downstream_zero_id"][comp_id, h_id] = downstream_zero_id
            diagnostics["fit_T"][comp_id, h_id] = fit["T_fit"]
            diagnostics["fit_p"][comp_id, h_id] = fit["p_fit"]
            diagnostics["fitted_zero_tau"][comp_id, h_id] = fit["fitted_zero_tau"]
            diagnostics["selected_zero_tau"][comp_id, h_id] = tau[selected_zero_id]
            diagnostics["zero_source"][comp_id, h_id] = fit["zero_source"]
            diagnostics["n_fit_points"][comp_id, h_id] = fit["n_fit_points"]
            diagnostics["fit_used_fallback"][comp_id, h_id] = fit["used_fallback"]
    diagnostics["updated_zero_id"] = diagnostics["selected_zero_id"]
    return rho_out, diagnostics



def residual_update_autocorrelation_full_raw(rho_inlet, rho_target, rho_downstream, tau_array=None, relaxation_factor=0.35, clip=True, zero_tol=0.0, zero_persistence_points=1, zero_lookahead_points=0):
    """Residual autocorrelation update using the full raw updated autocorrelation.

    Uses the raw updated rho = rho_inlet + relaxation_factor*(rho_target-rho_downstream)
    over the full tau range. This is then transformed to an autocorrelation-derived
    spectrum, but only the low-frequency portion is used in the final hybrid blend.
    Any resulting high-frequency noise from the raw rho tail is therefore acceptable.
    """
    rho_updated = rho_inlet + relaxation_factor * (rho_target - rho_downstream)
    if clip:
        rho_updated = np.clip(rho_updated, -1.0, 1.0)
    rho_updated[:, :, 0] = 1.0

    diagnostics = {
        "updated_zero_id": np.zeros(rho_updated.shape[:2], dtype=int),
        "selected_zero_id": np.zeros(rho_updated.shape[:2], dtype=int),
        "selected_zero_tau": np.full(rho_updated.shape[:2], np.nan, dtype=float),
        "target_zero_id": np.zeros(rho_updated.shape[:2], dtype=int),
        "downstream_zero_id": np.zeros(rho_updated.shape[:2], dtype=int),
        "fit_T": np.full(rho_updated.shape[:2], np.nan, dtype=float),
        "fit_p": np.full(rho_updated.shape[:2], np.nan, dtype=float),
        "fitted_zero_tau": np.full(rho_updated.shape[:2], np.nan, dtype=float),
        "n_fit_points": np.zeros(rho_updated.shape[:2], dtype=int),
        "fit_used_fallback": np.zeros(rho_updated.shape[:2], dtype=bool),
        "zero_source": np.empty(rho_updated.shape[:2], dtype=object),
    }
    tau_values = np.asarray(tau_array, dtype=float) if tau_array is not None else None

    for comp_id in range(rho_updated.shape[0]):
        for h_id in range(rho_updated.shape[1]):
            updated_1d = rho_updated[comp_id, h_id, :]
            target_1d = rho_target[comp_id, h_id, :]
            downstream_1d = rho_downstream[comp_id, h_id, :]

            updated_zero_id = first_zero_index(updated_1d, 1, zero_tol, zero_persistence_points, zero_lookahead_points)
            target_zero_id = first_zero_index(target_1d, 1, zero_tol, zero_persistence_points, zero_lookahead_points)
            downstream_zero_id = first_zero_index(downstream_1d, 1, zero_tol, zero_persistence_points, zero_lookahead_points)

            diagnostics["updated_zero_id"][comp_id, h_id] = updated_zero_id
            diagnostics["selected_zero_id"][comp_id, h_id] = updated_zero_id
            diagnostics["selected_zero_tau"][comp_id, h_id] = float(tau_values[updated_zero_id]) if tau_values is not None and updated_zero_id < len(tau_values) else float(updated_zero_id)
            diagnostics["target_zero_id"][comp_id, h_id] = target_zero_id
            diagnostics["downstream_zero_id"][comp_id, h_id] = downstream_zero_id
            diagnostics["zero_source"][comp_id, h_id] = "raw_full"

    return rho_updated, diagnostics

def cosine_taper_tail(tau_array, start_fraction=0.75):
    tau = np.asarray(tau_array, dtype=float)
    if len(tau) < 3:
        return np.ones_like(tau)
    x = np.linspace(0.0, 1.0, len(tau))
    w = np.ones_like(x)
    m = x > start_fraction
    if np.any(m):
        xi = (x[m] - start_fraction) / (1.0 - start_fraction)
        w[m] = 0.5 * (1.0 + np.cos(np.pi * xi))
    return w


def spectrum_from_autocovariance_array(tau_array, autocovariance_array, freq_array, floor=1e-16, apply_taper=True, taper_start_fraction=0.75, clip_negative=True):
    tau = np.asarray(tau_array, dtype=float)
    f = np.asarray(freq_array, dtype=float)
    R = np.asarray(autocovariance_array, dtype=float)
    cos_matrix = np.cos(2.0 * np.pi * f[:, np.newaxis] * tau[np.newaxis, :])
    taper = cosine_taper_tail(tau, start_fraction=taper_start_fraction) if apply_taper else np.ones_like(tau)
    spectra = np.zeros((R.shape[0], R.shape[1], len(f)), dtype=float)
    negative_fraction = np.zeros((R.shape[0], R.shape[1]), dtype=float)
    for comp_id in range(R.shape[0]):
        for h_id in range(R.shape[1]):
            R_use = R[comp_id, h_id, :] * taper
            S = 4.0 * _trapz(cos_matrix * R_use[np.newaxis, :], tau, axis=1)
            negative_fraction[comp_id, h_id] = np.mean(S <= 0.0)
            if clip_negative:
                S = np.maximum(S, floor)
            spectra[comp_id, h_id, :] = np.maximum(S, floor)
    return spectra, negative_fraction


def reconstruct_rho_from_spectra_resolved_band(freq_array, spectra_array, tau_array, f_min=None, f_max=None, floor=1e-16):
    R = autocovariance_from_spectrum_array(freq_array, spectra_array, tau_array, floor=floor, f_min=f_min, f_max=f_max)
    return normalise_autocovariance_to_rho(R, floor=1e-30)


#%% --------------------------------------------------------------------------
# Hybrid low-frequency correction helpers
# ---------------------------------------------------------------------------

def unpack_packed_raw_or_binned_array(arr, freq_array=None, name="array"):
    """Handle windLespy packed raw/binned multitaper outputs when return_raw=True.

    Packed raw often has shape (3, nHeights, 2, nFreq): [:,:,0,:] frequencies,
    [:,:,1,:] PSD. Packed binned often has shape (3, nHeights, 2, nBins).
    This helper returns the PSD part if such a packed structure is detected.
    """
    a = np.asarray(arr, dtype=object if isinstance(arr, list) else float)
    if isinstance(arr, list):
        return arr
    if a.ndim == 4 and a.shape[2] == 2:
        if freq_array is not None and a.shape[-1] == len(freq_array):
            maxdiff = np.nanmax(np.abs(np.asarray(a[0, 0, 0, :], dtype=float) - np.asarray(freq_array, dtype=float)))
            print(f"{name}: unpacked packed PSD array; max frequency-grid difference = {maxdiff:.3e}")
        else:
            print(f"{name}: unpacked packed PSD/bin array")
        return np.asarray(a[:, :, 1, :], dtype=float)
    return np.asarray(arr, dtype=float)


def get_first_spline_frequency_array(downstream_binned_multitaper, n_comp, n_heights, fallback=1.0):
    """Return first binned/spline knot frequency for each component and height.

    Supports packed binned arrays of shape (3,nH,2,nBins) and nested list formats.
    Falls back to a scalar when the bin structure is unavailable.
    """
    first = np.full((n_comp, n_heights), float(fallback), dtype=float)
    try:
        arr = np.asarray(downstream_binned_multitaper, dtype=object if isinstance(downstream_binned_multitaper, list) else float)
        if not isinstance(downstream_binned_multitaper, list) and arr.ndim == 4 and arr.shape[2] == 2:
            for c in range(n_comp):
                for h in range(n_heights):
                    f = np.asarray(arr[c, h, 0, :], dtype=float)
                    valid = np.isfinite(f) & (f > 0.0)
                    if np.any(valid):
                        first[c, h] = float(f[valid][0])
            return first
        if isinstance(downstream_binned_multitaper, list):
            for c in range(min(n_comp, len(downstream_binned_multitaper))):
                for h in range(min(n_heights, len(downstream_binned_multitaper[c]))):
                    item = downstream_binned_multitaper[c][h]
                    if isinstance(item, dict):
                        f = np.asarray(item.get("freq", item.get("f", [])), dtype=float)
                    elif isinstance(item, (tuple, list)) and len(item) >= 2:
                        f = np.asarray(item[0], dtype=float)
                    else:
                        f = np.asarray([], dtype=float)
                    valid = np.isfinite(f) & (f > 0.0)
                    if np.any(valid):
                        first[c, h] = float(f[valid][0])
            return first
    except Exception as exc:
        print(f"Warning: could not infer first spline/bin frequencies; using fallback {fallback}. Error: {exc}")
    return first



def get_spectral_knot_frequency_lists(downstream_binned_multitaper, n_comp, n_heights, fallback=1.0):
    """Return all spectral-calibration knot frequencies for each component/height."""
    knot_freqs = [[np.asarray([float(fallback)], dtype=float) for _ in range(n_heights)] for _ in range(n_comp)]
    try:
        arr = np.asarray(downstream_binned_multitaper, dtype=object if isinstance(downstream_binned_multitaper, list) else float)
        if not isinstance(downstream_binned_multitaper, list) and arr.ndim == 4 and arr.shape[2] == 2:
            for c in range(n_comp):
                for h in range(n_heights):
                    f = np.asarray(arr[c, h, 0, :], dtype=float)
                    f = f[np.isfinite(f) & (f > 0.0)]
                    if len(f) > 0:
                        knot_freqs[c][h] = np.unique(np.sort(f))
            return knot_freqs
        if isinstance(downstream_binned_multitaper, list):
            for c in range(min(n_comp, len(downstream_binned_multitaper))):
                for h in range(min(n_heights, len(downstream_binned_multitaper[c]))):
                    item = downstream_binned_multitaper[c][h]
                    if isinstance(item, dict):
                        f = np.asarray(item.get("freq", item.get("f", [])), dtype=float)
                    elif isinstance(item, (tuple, list)) and len(item) >= 2:
                        f = np.asarray(item[0], dtype=float)
                    else:
                        f = np.asarray([], dtype=float)
                    f = f[np.isfinite(f) & (f > 0.0)]
                    if len(f) > 0:
                        knot_freqs[c][h] = np.unique(np.sort(f))
    except Exception as exc:
        print(f"Warning: could not infer all spline/bin frequencies; using fallback {fallback}. Error: {exc}")
    return knot_freqs


def merge_duplicate_x_log_knots(f_knots, s_knots, floor=1e-16):
    f = np.asarray(f_knots, dtype=float)
    S = np.asarray(s_knots, dtype=float)
    mask = np.isfinite(f) & np.isfinite(S) & (f > 0.0) & (S > floor)
    f = f[mask]
    S = S[mask]
    if len(f) == 0:
        return f, S
    order = np.argsort(f)
    f = f[order]
    S = S[order]
    logf = np.log10(f)
    logS = np.log10(np.maximum(S, floor))
    out_f = []
    out_S = []
    i = 0
    while i < len(logf):
        j = i + 1
        while j < len(logf) and np.isclose(logf[j], logf[i], rtol=1e-10, atol=1e-12):
            j += 1
        out_f.append(10.0 ** np.mean(logf[i:j]))
        out_S.append(10.0 ** np.mean(logS[i:j]))
        i = j
    return np.asarray(out_f), np.asarray(out_S)


def interp_loglog_1d(freq_old, spectrum_old, freq_new, floor=1e-16):
    f_old = np.asarray(freq_old, dtype=float)
    S_old = np.asarray(spectrum_old, dtype=float)
    f_new = np.asarray(freq_new, dtype=float)
    mask = np.isfinite(f_old) & np.isfinite(S_old) & (f_old > 0.0) & (S_old > floor)
    if np.count_nonzero(mask) < 2:
        return np.full_like(f_new, floor, dtype=float)
    x_old = np.log(f_old[mask])
    y_old = np.log(np.maximum(S_old[mask], floor))
    order = np.argsort(x_old)
    x_old = x_old[order]
    y_old = y_old[order]
    return np.maximum(np.exp(np.interp(np.log(f_new), x_old, y_old, left=y_old[0], right=y_old[-1])), floor)


def pchip_loglog_from_knots(freq_array, f_knots, s_knots, floor=1e-16):
    f = np.asarray(freq_array, dtype=float)
    fk, Sk = merge_duplicate_x_log_knots(f_knots, s_knots, floor=floor)
    if len(fk) < 2:
        return np.full_like(f, floor, dtype=float)
    x = np.log10(np.maximum(f, floor))
    xk = np.log10(fk)
    yk = np.log10(np.maximum(Sk, floor))
    if len(fk) == 2:
        y = np.interp(x, xk, yk, left=yk[0], right=yk[-1])
        return np.maximum(10.0 ** y, floor)
    pchip = PchipInterpolator(xk, yk, extrapolate=False)
    y = np.empty_like(x)
    mid = (x >= xk[0]) & (x <= xk[-1])
    y[mid] = pchip(x[mid])
    y[x < xk[0]] = yk[0]
    right = x > xk[-1]
    if np.any(right):
        slope = (yk[-1] - yk[-2]) / max(xk[-1] - xk[-2], 1e-12)
        slope = np.clip(slope, -8.0, 2.0)
        y[right] = yk[-1] + slope * (x[right] - xk[-1])
    return np.maximum(10.0 ** y, floor)


def get_autocorr_low_frequency_knots_explicit(
    freq_array,
    autocorr_spectrum_1d,
    first_spectral_knot_freq,
    resolved_f_min,
    n_low_knots=8,
    max_fraction_of_first_knot=0.95,
    floor=1e-16,
):
    """Return explicit log-spaced low-frequency knots from the autocorrelation branch."""
    f = np.asarray(freq_array, dtype=float)
    S = np.asarray(autocorr_spectrum_1d, dtype=float)
    positive_f = f[np.isfinite(f) & (f > 0.0)]
    if len(positive_f) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    f_low = float(np.nanmin(positive_f)) if resolved_f_min is None else max(float(resolved_f_min), float(np.nanmin(positive_f)))
    f_high = float(max_fraction_of_first_knot) * float(first_spectral_knot_freq)
    if (not np.isfinite(f_low)) or (not np.isfinite(f_high)) or f_high <= f_low:
        return np.array([], dtype=float), np.array([], dtype=float)
    valid = np.isfinite(f) & np.isfinite(S) & (f > 0.0) & (S > floor) & (f >= f_low) & (f <= f_high)
    if np.count_nonzero(valid) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    f_valid = f[valid]
    S_valid = np.maximum(S[valid], floor)
    f_knots = np.geomspace(f_low, f_high, int(n_low_knots))
    S_knots = np.exp(np.interp(np.log(f_knots), np.log(f_valid), np.log(S_valid)))
    good = np.isfinite(f_knots) & np.isfinite(S_knots) & (f_knots > 0.0) & (S_knots > floor)
    return f_knots[good], S_knots[good]


def build_join_matched_combined_knot_spectrum(
    freq_array,
    spectral_baseline_array,
    autocorr_spectra_array,
    spectral_knot_freqs,
    first_knot_freqs,
    floor=1e-16,
    low_n_knots=8,
    low_max_fraction_of_first_knot=0.95,
    low_min_points_per_bin=2,
    ratio_min=0.5,
    ratio_max=2.0,
    f_min_for_low=None,
    add_endpoint_anchors=True,
    match_low_freq_to_spectral_join=True,
    join_scale_min=0.33,
    join_scale_max=3.0,
):
    """Build final spectrum from matched autocorrelation low-frequency knots plus spectral knots."""
    n_comp, n_heights, _ = spectral_baseline_array.shape
    hybrid = np.zeros_like(spectral_baseline_array)
    rows = []
    knot_store = [[None for _ in range(n_heights)] for _ in range(n_comp)]
    matched_autocorr = np.zeros_like(autocorr_spectra_array)

    for comp_id in range(n_comp):
        for h_id in range(n_heights):
            f_join = float(first_knot_freqs[comp_id, h_id])
            if not np.isfinite(f_join) or f_join <= 0.0:
                f_join = HYBRID_FIRST_KNOT_FALLBACK

            S_spec = np.maximum(spectral_baseline_array[comp_id, h_id, :], floor)
            S_ac = np.maximum(autocorr_spectra_array[comp_id, h_id, :], floor)

            S_spec_join = float(interp_loglog_1d(freq_array, S_spec, np.asarray([f_join]), floor=floor)[0])
            S_ac_join = float(interp_loglog_1d(freq_array, S_ac, np.asarray([f_join]), floor=floor)[0])
            raw_join_scale = S_spec_join / max(S_ac_join, floor)
            if match_low_freq_to_spectral_join:
                join_scale = float(np.clip(raw_join_scale, join_scale_min, join_scale_max))
            else:
                join_scale = 1.0
            S_ac_matched = np.maximum(S_ac * join_scale, floor)

            # Keep the full low-frequency autocorrelation shape after vertical shifting,
            # but bound it relative to the spectral branch to prevent pathological jumps.
            ratio = np.clip(S_ac_matched / np.maximum(S_spec, floor), ratio_min, ratio_max)
            S_ac_bounded = np.maximum(S_spec * ratio, floor)
            matched_autocorr[comp_id, h_id, :] = S_ac_bounded

            if HYBRID_ENABLE_AUTOCORR_LOW_FREQ_KNOTS:
                f_low, S_low = get_autocorr_low_frequency_knots_explicit(
                    freq_array=freq_array,
                    autocorr_spectrum_1d=S_ac_bounded,
                    first_spectral_knot_freq=f_join,
                    resolved_f_min=f_min_for_low,
                    n_low_knots=low_n_knots,
                    max_fraction_of_first_knot=low_max_fraction_of_first_knot,
                    floor=floor,
                )
                if len(f_low) < max(2, min(4, int(low_n_knots) // 2)):
                    f_low, S_low = log_binned_median_low_points(
                        freq_array,
                        S_ac_bounded,
                        f_min=f_min_for_low,
                        f_max=f_join,
                        n_bins=low_n_knots,
                        min_points_per_bin=low_min_points_per_bin,
                        floor=floor,
                    )
            else:
                f_low = np.array([], dtype=float)
                S_low = np.array([], dtype=float)

            f_spec_knots = np.asarray(spectral_knot_freqs[comp_id][h_id], dtype=float)
            f_spec_knots = f_spec_knots[np.isfinite(f_spec_knots) & (f_spec_knots > 0.0) & (f_spec_knots >= f_join * (1.0 - 1e-10))]
            if len(f_spec_knots) == 0:
                f_spec_knots = np.asarray([f_join], dtype=float)
            S_spec_knots = interp_loglog_1d(freq_array, S_spec, f_spec_knots, floor=floor)

            f_all = []
            S_all = []
            if add_endpoint_anchors:
                f_all.append(freq_array[0])
                S_all.append(S_ac_bounded[0] if freq_array[0] < f_join else S_spec[0])
            f_all.extend(list(f_low))
            S_all.extend(list(S_low))
            f_all.extend(list(f_spec_knots))
            S_all.extend(list(S_spec_knots))
            if add_endpoint_anchors and freq_array[-1] > np.max(f_spec_knots):
                f_all.append(freq_array[-1])
                S_all.append(S_spec[-1])

            f_all, S_all = merge_duplicate_x_log_knots(f_all, S_all, floor=floor)
            hybrid[comp_id, h_id, :] = pchip_loglog_from_knots(freq_array, f_all, S_all, floor=floor) if len(f_all) >= 2 else S_spec
            knot_store[comp_id][h_id] = {"f": f_all, "S": S_all, "f_join": f_join}
            rows.append({
                "component": COMPONENT_NAMES[comp_id],
                "height_id": h_id,
                "first_spectral_knot_freq": f_join,
                "raw_join_scale": raw_join_scale,
                "applied_join_scale": join_scale,
                "join_scale_was_clipped": bool(not np.isclose(raw_join_scale, join_scale)),
                "n_low_knots": len(f_low),
                "n_low_knots_requested": int(low_n_knots),
                "n_spec_knots": len(f_spec_knots),
                "n_total_knots": len(f_all),
            })

    return np.maximum(hybrid, floor), pd.DataFrame(rows), knot_store, np.maximum(matched_autocorr, floor)


def log_binned_median_low_points(freq_array, spectrum_1d, f_min, f_max, n_bins=8, min_points_per_bin=2, floor=1e-16):
    """Fallback low-frequency knot extraction by log-frequency bin medians."""
    f = np.asarray(freq_array, dtype=float)
    S = np.maximum(np.asarray(spectrum_1d, dtype=float), floor)
    mask = np.isfinite(f) & np.isfinite(S) & (f > 0.0) & (S > floor)
    if f_min is not None:
        mask &= f >= float(f_min)
    if f_max is not None:
        mask &= f < float(f_max)
    if np.count_nonzero(mask) < max(2, min_points_per_bin):
        return np.array([], dtype=float), np.array([], dtype=float)
    logf = np.log10(f[mask])
    logS = np.log10(S[mask])
    n_bins_eff = min(int(n_bins), max(1, len(logf) // max(1, min_points_per_bin)))
    edges = np.linspace(logf.min(), logf.max(), n_bins_eff + 1)
    fk = []
    Sk = []
    for i in range(n_bins_eff):
        m = (logf >= edges[i]) & ((logf < edges[i + 1]) if i < n_bins_eff - 1 else (logf <= edges[i + 1]))
        if np.count_nonzero(m) >= min_points_per_bin:
            fk.append(10.0 ** np.median(logf[m]))
            Sk.append(10.0 ** np.median(logS[m]))
    return np.asarray(fk), np.asarray(Sk)


def smoothstep(x):
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def build_low_frequency_ratio_correction(
    freq_array,
    spectral_baseline,
    autocorr_updated_spectrum,
    first_knot_freqs,
    floor=1e-16,
    ratio_min=0.2,
    ratio_max=5.0,
    poly_degree=2,
    min_fit_points=4,
    blend_start_fraction=0.70,
):
    """Build smooth ratio G(f) from autocorr/spectral spectra below first knot.

    The correction is fitted in log-log space and forced to G=1 at the first
    spline/bin frequency. Above the first knot G=1, so the high-frequency
    spectral calibration shape is retained.
    """
    f = np.asarray(freq_array, dtype=float)
    S_spec = np.maximum(np.asarray(spectral_baseline, dtype=float), floor)
    S_ac = np.maximum(np.asarray(autocorr_updated_spectrum, dtype=float), floor)
    G = np.ones_like(S_spec)
    diag_rows = []

    for comp_id in range(S_spec.shape[0]):
        for h_id in range(S_spec.shape[1]):
            f_knot = float(first_knot_freqs[comp_id, h_id])
            fit_mask = np.isfinite(f) & (f > 0.0) & (f <= f_knot)
            if np.count_nonzero(fit_mask) < min_fit_points:
                fit_mask = np.isfinite(f) & (f > 0.0) & (f <= max(f_knot, np.nanmin(f[f > 0]) * min_fit_points))

            x = np.log(np.maximum(f[fit_mask], floor))
            ratio = np.clip(S_ac[comp_id, h_id, fit_mask] / S_spec[comp_id, h_id, fit_mask], ratio_min, ratio_max)
            y = np.log(np.maximum(ratio, floor))
            finite = np.isfinite(x) & np.isfinite(y)

            n_fit = int(np.count_nonzero(finite))
            if n_fit >= max(2, min_fit_points):
                deg = min(int(poly_degree), n_fit - 1)
                try:
                    coef = np.polyfit(x[finite], y[finite], deg)
                    x_all = np.log(np.maximum(f, floor))
                    y_all = np.polyval(coef, x_all)
                    y_knot = float(np.polyval(coef, np.log(max(f_knot, floor))))
                    y_all = y_all - y_knot  # enforce G(f_knot)=1
                    g_fit = np.exp(np.clip(y_all, np.log(ratio_min), np.log(ratio_max)))
                except Exception:
                    g_fit = np.ones_like(f)
            else:
                g_fit = np.ones_like(f)

            # Full correction below blend_start; taper to unity at f_knot.
            g_1d = np.ones_like(f)
            low = (f > 0.0) & (f < f_knot)
            f_blend_start = max(np.nanmin(f[f > 0]), blend_start_fraction * f_knot)
            full = low & (f <= f_blend_start)
            blend = low & (f > f_blend_start)
            g_1d[full] = g_fit[full]
            if np.any(blend):
                w = smoothstep((f[blend] - f_blend_start) / max(f_knot - f_blend_start, floor))
                g_1d[blend] = np.exp((1.0 - w) * np.log(np.maximum(g_fit[blend], floor)))
            g_1d[f >= f_knot] = 1.0
            G[comp_id, h_id, :] = np.clip(g_1d, ratio_min, ratio_max)
            diag_rows.append({
                "component_id": comp_id,
                "height_id": h_id,
                "first_knot_freq": f_knot,
                "n_ratio_fit_points": n_fit,
                "ratio_min_applied": float(np.nanmin(G[comp_id, h_id, low])) if np.any(low) else 1.0,
                "ratio_max_applied": float(np.nanmax(G[comp_id, h_id, low])) if np.any(low) else 1.0,
            })
    return G, pd.DataFrame(diag_rows)


def choose_hybrid_beta_by_length(
    freq_array,
    spectral_baseline,
    ratio_correction,
    beta_grid,
    updated_sigma2,
    tau_array,
    target_L_for_update,
    U_for_update,
    f_min,
    f_max,
    floor=1e-16,
):
    """Choose beta per comp/height by matching L from final spectrum to target_L."""
    beta_grid = np.asarray(beta_grid, dtype=float)
    beta_selected = np.zeros(spectral_baseline.shape[:2], dtype=float)
    best_L = np.zeros_like(beta_selected)
    for comp_id in range(spectral_baseline.shape[0]):
        for h_id in range(spectral_baseline.shape[1]):
            best_err = np.inf
            best_beta = float(beta_grid[0])
            best_L_val = np.nan
            for beta in beta_grid:
                S = spectral_baseline.copy()
                S[comp_id, h_id, :] = spectral_baseline[comp_id, h_id, :] * np.power(ratio_correction[comp_id, h_id, :], beta)
                S = renormalise_spectra_to_variance(freq_array, S, updated_sigma2, floor=floor, f_min=f_min, f_max=f_max)
                rho = reconstruct_rho_from_spectra_resolved_band(freq_array, S, tau_array, f_min=f_min, f_max=f_max, floor=floor)
                L = integral_length_array_from_rho(tau_array, rho, U_for_update)
                L_val = L[comp_id, h_id]
                err = abs(L_val - target_L_for_update[comp_id, h_id]) / max(abs(target_L_for_update[comp_id, h_id]), 1e-12)
                if np.isfinite(err) and err < best_err:
                    best_err = err
                    best_beta = float(beta)
                    best_L_val = float(L_val)
            beta_selected[comp_id, h_id] = best_beta
            best_L[comp_id, h_id] = best_L_val
    return beta_selected, best_L


def apply_ratio_with_beta(spectral_baseline, ratio_correction, beta):
    S = np.asarray(spectral_baseline, dtype=float).copy()
    if np.ndim(beta) == 0:
        return S * np.power(ratio_correction, float(beta))
    beta = np.asarray(beta, dtype=float)
    for comp_id in range(S.shape[0]):
        for h_id in range(S.shape[1]):
            S[comp_id, h_id, :] *= np.power(ratio_correction[comp_id, h_id, :], beta[comp_id, h_id])
    return S


#%% --------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_autocorrelation_comparison(fig_dir, z_array, tau_array, rho_inlet, rho_downstream, rho_target, rho_raw_update, rho_fitted, body_height, z_max_factor=1.5, n_heights=8, components=("u", "v", "w")):
    safe_makedirs(fig_dir)
    ids = _selected_height_ids(z_array, body_height, z_max_factor=z_max_factor, n_heights=n_heights)
    for comp_id, comp in enumerate(components):
        comp_dir = os.path.join(fig_dir, f"rho_{comp}")
        safe_makedirs(comp_dir)
        for h_id in ids:
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.plot(tau_array, rho_inlet[comp_id, h_id, :], label="Inlet from spectra")
            ax.plot(tau_array, rho_downstream[comp_id, h_id, :], label="Downstream from time series")
            ax.plot(tau_array, rho_target[comp_id, h_id, :], label="Target from spectra")
            ax.plot(tau_array, rho_raw_update[comp_id, h_id, :], linestyle="--", alpha=0.8, label="Raw Wong-updated rho")
            ax.plot(tau_array, rho_fitted[comp_id, h_id, :], label="Updated rho (full raw)")
            ax.axhline(0.0, linestyle="--", linewidth=1.0, color="k", alpha=0.5)
            ax.set_xlabel(r"$\tau$ [s]")
            ax.set_ylabel(r"$\rho(\tau)$")
            ax.set_title(f"{comp}-component autocorrelation, z/H = {z_array[h_id] / body_height:.2f}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            safe_savefig(fig, os.path.join(comp_dir, f"rho_{comp}_zH_{z_array[h_id] / body_height:.2f}.png"))
            plt.close(fig)


def plot_hybrid_spectra(fig_dir, z_array, freq_array, inlet_spectra, target_spectra, spectral_baseline, autocorr_spectra, hybrid_spectra, first_knot_freqs, body_height, z_max_factor=3.0, n_heights=8, components=("u", "v", "w"), floor=1e-16):
    safe_makedirs(fig_dir)
    ids = _selected_height_ids(z_array, body_height, z_max_factor=z_max_factor, n_heights=n_heights)
    for comp_id, comp in enumerate(components):
        comp_dir = os.path.join(fig_dir, f"S_{comp}")
        safe_makedirs(comp_dir)
        for h_id in ids:
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.loglog(freq_array, np.maximum(inlet_spectra[comp_id, h_id, :], floor), label="Current inlet")
            ax.loglog(freq_array, np.maximum(target_spectra[comp_id, h_id, :], floor), label="Target")
            ax.loglog(freq_array, np.maximum(spectral_baseline[comp_id, h_id, :], floor), label="Spectral baseline")
            ax.loglog(freq_array, np.maximum(autocorr_spectra[comp_id, h_id, :], floor), linestyle="--", alpha=0.75, label="Autocorr-derived")
            ax.loglog(freq_array, np.maximum(hybrid_spectra[comp_id, h_id, :], floor), linewidth=2.0, label="Hybrid final")
            ax.axvline(first_knot_freqs[comp_id, h_id], linestyle=":", color="k", alpha=0.5, label="First spline point")
            ax.set_xlabel("f [Hz]")
            ax.set_ylabel(fr"$S_{{{comp}{comp}}}(f)$")
            ax.set_title(f"{comp}-component hybrid spectrum, z/H = {z_array[h_id] / body_height:.2f}")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
            safe_savefig(fig, os.path.join(comp_dir, f"hybrid_S_{comp}_zH_{z_array[h_id] / body_height:.2f}.png"))
            plt.close(fig)


def plot_low_frequency_ratio(fig_dir, z_array, freq_array, ratio_correction, beta_selected, first_knot_freqs, body_height, z_max_factor=1.5, n_heights=8, components=("u", "v", "w")):
    safe_makedirs(fig_dir)
    ids = _selected_height_ids(z_array, body_height, z_max_factor=z_max_factor, n_heights=n_heights)
    for comp_id, comp in enumerate(components):
        comp_dir = os.path.join(fig_dir, f"G_{comp}")
        safe_makedirs(comp_dir)
        for h_id in ids:
            beta = beta_selected[comp_id, h_id] if np.ndim(beta_selected) == 2 else float(beta_selected)
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.semilogx(freq_array, ratio_correction[comp_id, h_id, :], label="Raw smooth correction G")
            ax.semilogx(freq_array, np.power(ratio_correction[comp_id, h_id, :], beta), label=f"Applied G^beta, beta={beta:.2f}")
            ax.axhline(1.0, linestyle="--", color="k", alpha=0.5)
            ax.axvline(first_knot_freqs[comp_id, h_id], linestyle=":", color="k", alpha=0.5)
            ax.set_xlabel("f [Hz]")
            ax.set_ylabel("Low-frequency multiplicative correction")
            ax.set_title(f"{comp}-component low-frequency correction, z/H = {z_array[h_id] / body_height:.2f}")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
            safe_savefig(fig, os.path.join(comp_dir, f"G_{comp}_zH_{z_array[h_id] / body_height:.2f}.png"))
            plt.close(fig)


def plot_length_profiles(fig_dir, z_array, body_height, L_inlet, L_downstream, L_target, L_acorr_update, L_spectral, L_hybrid, components=("u", "v", "w")):
    safe_makedirs(fig_dir)
    y = z_array / body_height
    for comp_id, comp in enumerate(components):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(L_inlet[comp_id, :], y, label="Current inlet")
        ax.plot(L_downstream[comp_id, :], y, label="Downstream")
        ax.plot(L_target[comp_id, :], y, label="Target")
        ax.plot(L_acorr_update[comp_id, :], y, label="Autocorr intended")
        ax.plot(L_spectral[comp_id, :], y, label="Spectral baseline")
        ax.plot(L_hybrid[comp_id, :], y, label="Hybrid final")
        ax.set_xlabel(f"L_{comp} [m]")
        ax.set_ylabel("z/H")
        ax.set_title(f"{comp}-component resolved-band length-scale diagnostics")
        ax.grid(True, alpha=0.3)
        ax.legend()
        safe_savefig(fig, os.path.join(fig_dir, f"L_{comp}.png"))
        plt.close(fig)



#%% --------------------------------------------------------------------------
# u-w co-spectrum / cross-covariance helpers
# ---------------------------------------------------------------------------

def _find_first_existing_profile_column(df, candidates):
    if df is None:
        return None
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in lower_map:
            return lower_map[key]
    return None


def get_uw_stress_from_profile_df_or_fallback(profile_df, profile_array, candidates=UW_TARGET_STRESS_COLUMN_CANDIDATES, fallback_rho=UW_FALLBACK_RHO):
    """Return uwStress(z), preferring an explicit profile column if present.

    profile_array is expected to contain [U, var_u, var_v, var_w, Lu, Lv, Lw].
    The fallback is rho_uw * sigma_u * sigma_w.
    """
    col = _find_first_existing_profile_column(profile_df, candidates)
    if col is not None:
        values = pd.to_numeric(profile_df[col], errors="coerce").to_numpy(dtype=float)
        if np.all(np.isfinite(values)):
            return values, f"profile_column:{col}"
    arr = np.asarray(profile_array, dtype=float)
    sigma_u = np.sqrt(np.maximum(arr[:, 1], 0.0))
    sigma_w = np.sqrt(np.maximum(arr[:, 3], 0.0))
    return float(fallback_rho) * sigma_u * sigma_w, f"fallback_rho:{fallback_rho}"


def read_spectra_profile_file_extended(case_path, filename, n_freq_expected=None):
    """Read spectraProfile variants.

    Supported rows after the header:
      legacy:      z Su Sv Sw
      augmented:   z uwStress Su Sv Sw
      full:        z uwStress Su Sv Sw Cuw

    where each spectral block has nFreq entries. Returns
    (spectra_array, z_array, uw_stress_or_None, cuw_array_or_None).
    """
    filepath = os.path.join(case_path, "constant", "boundaryData", "windProfile", filename)
    with open(filepath, "r") as f:
        header = f.readline().split()
    if len(header) != 2:
        raise ValueError(f"Invalid spectra profile header in {filepath}")
    n_heights = int(header[0])
    n_freq = int(header[1])
    if n_freq_expected is not None and int(n_freq_expected) != n_freq:
        raise ValueError(f"{filepath} has nFreq={n_freq}, expected {n_freq_expected}")
    data = np.loadtxt(filepath, skiprows=1)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[0] != n_heights:
        raise ValueError(f"Expected {n_heights} rows in {filepath}, found {data.shape[0]}")
    z = data[:, 0].astype(float)
    ncols = data.shape[1]
    legacy_cols = 1 + 3 * n_freq
    augmented_cols = 2 + 3 * n_freq
    full_cols = 2 + 4 * n_freq
    uw = None
    cuw = None
    if ncols == legacy_cols:
        start = 1
    elif ncols == augmented_cols:
        uw = data[:, 1].astype(float)
        start = 2
    elif ncols == full_cols:
        uw = data[:, 1].astype(float)
        start = 2
        cuw = data[:, start + 3 * n_freq:start + 4 * n_freq].astype(float)
    else:
        raise ValueError(
            f"Unexpected number of columns in {filepath}: {ncols}. "
            f"Expected {legacy_cols}, {augmented_cols}, or {full_cols}."
        )
    spectra = np.zeros((3, n_heights, n_freq), dtype=float)
    spectra[0, :, :] = data[:, start:start + n_freq]
    spectra[1, :, :] = data[:, start + n_freq:start + 2 * n_freq]
    spectra[2, :, :] = data[:, start + 2 * n_freq:start + 3 * n_freq]
    return spectra, z, uw, cuw


def read_uw_cospectrum_profile(case_path, filename=SPECTRA_PROFILE_UW_FILENAME, n_freq_expected=None):
    filepath = os.path.join(case_path, "constant", "boundaryData", "windProfile", filename)
    if not os.path.exists(filepath):
        return None, None, None
    with open(filepath, "r") as f:
        header = f.readline().split()
    if len(header) != 2:
        raise ValueError(f"Invalid uw co-spectrum profile header in {filepath}")
    n_heights = int(header[0])
    n_freq = int(header[1])
    if n_freq_expected is not None and int(n_freq_expected) != n_freq:
        raise ValueError(f"{filepath} has nFreq={n_freq}, expected {n_freq_expected}")
    data = np.loadtxt(filepath, skiprows=1)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[0] != n_heights:
        raise ValueError(f"Expected {n_heights} rows in {filepath}, found {data.shape[0]}")
    expected_cols = 2 + n_freq
    if data.shape[1] != expected_cols:
        raise ValueError(f"Expected {expected_cols} columns in {filepath}, found {data.shape[1]}")
    return data[:, 2:].astype(float), data[:, 0].astype(float), data[:, 1].astype(float)


def write_spectra_profile_with_optional_uw(spectra_array, z_array, output_path, uw_stress=None, cuw_array=None, clip_min=1e-16):
    spectra = np.asarray(spectra_array, dtype=float)
    z = np.asarray(z_array, dtype=float)
    n_comp, n_heights, n_freq = spectra.shape
    if n_comp != 3:
        raise ValueError("spectra_array must have shape (3, nHeights, nFreq)")
    safe_makedirs(os.path.dirname(output_path))
    rows = []
    for h in range(n_heights):
        row = [z[h]]
        if uw_stress is not None:
            row.append(float(np.asarray(uw_stress, dtype=float)[h]))
        row.extend(np.maximum(spectra[0, h, :], clip_min))
        row.extend(np.maximum(spectra[1, h, :], clip_min))
        row.extend(np.maximum(spectra[2, h, :], clip_min))
        if cuw_array is not None:
            row.extend(np.asarray(cuw_array, dtype=float)[h, :])
        rows.append(row)
    with open(_windows_long_path(output_path), "w") as f:
        f.write(f"{n_heights} {n_freq}\n")
        np.savetxt(f, np.asarray(rows), fmt="%.10e", delimiter="\t")


def write_uw_cospectrum_profile(cuw_array, z_array, uw_stress, output_path):
    C = np.asarray(cuw_array, dtype=float)
    z = np.asarray(z_array, dtype=float)
    uw = np.asarray(uw_stress, dtype=float)
    n_heights, n_freq = C.shape
    safe_makedirs(os.path.dirname(output_path))
    rows = []
    for h in range(n_heights):
        rows.append(np.r_[z[h], uw[h], C[h, :]])
    with open(_windows_long_path(output_path), "w") as f:
        f.write(f"{n_heights} {n_freq}\n")
        np.savetxt(f, np.asarray(rows), fmt="%.10e", delimiter="\t")


def write_new_dfsr_inlet_profile_with_uw(new_inlet_profile_array, target_profile_df, case_path, uw_stress=None):
    output_path = os.path.join(case_path, "constant", "boundaryData", "windProfile", "profile")
    arr = np.asarray(new_inlet_profile_array, dtype=float)
    df = pd.DataFrame({
        "z": target_profile_df["z"].to_numpy(dtype=float),
        "U": arr[:, 0],
        "Iu": np.sqrt(np.maximum(arr[:, 1], 0.0)) / np.maximum(arr[:, 0], 1e-12),
        "Iv": np.sqrt(np.maximum(arr[:, 2], 0.0)) / np.maximum(arr[:, 0], 1e-12),
        "Iw": np.sqrt(np.maximum(arr[:, 3], 0.0)) / np.maximum(arr[:, 0], 1e-12),
        "Lu": arr[:, 4],
        "Lv": arr[:, 5],
        "Lw": arr[:, 6],
    })
    if uw_stress is not None:
        df["uwStress"] = np.asarray(uw_stress, dtype=float)
    np.savetxt(_windows_long_path(output_path), df.to_numpy(), fmt="%.10e", delimiter="\t")


def kaimal_uw_cospectrum_shape(freq_array, z_array, U_array, floor=1e-30):
    """Neutral Kaimal uw co-spectrum shape per Hz, with negative sign.

    Formula: -n Cuw(n) / u_*^2 = 14 f / (1 + 9.6 f)^2.4,
    f = n z / U. Since only shape is needed, u_*^2 is omitted and
    Cuw_shape = -14 z/U / (1 + 9.6 f)^2.4.
    """
    f = np.asarray(freq_array, dtype=float)
    z = np.maximum(np.asarray(z_array, dtype=float), floor)
    U = np.maximum(np.asarray(U_array, dtype=float), floor)
    out = np.zeros((len(z), len(f)), dtype=float)
    for h in range(len(z)):
        fr = f * z[h] / U[h]
        out[h, :] = -14.0 * (z[h] / U[h]) / np.power(1.0 + 9.6 * fr, 2.4)
    return out


def normalise_cospectrum_to_stress(freq_array, shape_array, uw_stress, floor=1e-30):
    f = np.asarray(freq_array, dtype=float)
    shape = np.asarray(shape_array, dtype=float).copy()
    uw = np.asarray(uw_stress, dtype=float)
    out = np.zeros_like(shape)
    for h in range(shape.shape[0]):
        area = _trapz(shape[h, :], f)
        if not np.isfinite(area) or abs(area) < floor:
            out[h, :] = 0.0
        else:
            out[h, :] = shape[h, :] * (uw[h] / area)
    return out


def integrate_cospectrum_area(freq_array, cospectrum_array, f_min=None, f_max=None):
    f = np.asarray(freq_array, dtype=float)
    C = np.asarray(cospectrum_array, dtype=float)
    area = np.zeros(C.shape[0], dtype=float)
    for h in range(C.shape[0]):
        mask = _band_mask_for_height(f, h, f_min=f_min, f_max=f_max)
        if np.count_nonzero(mask) >= 2:
            area[h] = _trapz(C[h, mask], f[mask])
        else:
            area[h] = np.nan
    return area


def wong_update_uw_stress(inlet_uw, target_uw, downstream_uw, relaxation_factor=0.35):
    """Additive relaxed stress update. Ratio-Wong update is avoided for signed stresses."""
    inlet = np.asarray(inlet_uw, dtype=float)
    target = np.asarray(target_uw, dtype=float)
    downstream = np.asarray(downstream_uw, dtype=float)
    return inlet + relaxation_factor * (target - downstream)


def _multitaper_csd_1d(x, y, fs, time_bandwidth=4.0, num_tapers=None, detrend=True):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 8:
        raise ValueError("Time series too short for multitaper CSD.")
    if detrend:
        x = x - np.mean(x)
        y = y - np.mean(y)
    n = x.size
    if num_tapers is None:
        num_tapers = max(1, int(2 * time_bandwidth) - 1)
    tapers, eigvals = dpss(n, time_bandwidth, Kmax=num_tapers, return_ratios=True)
    spectra = []
    for taper in tapers:
        X = rfft(x * taper)
        Y = rfft(y * taper)
        scale = fs * np.sum(taper ** 2)
        Sxy = (np.conj(X) * Y) / scale
        if n % 2 == 0:
            Sxy[1:-1] *= 2.0
        else:
            Sxy[1:] *= 2.0
        spectra.append(Sxy)
    S_mt = np.average(np.vstack(spectra), axis=0, weights=eigvals)
    return rfftfreq(n, d=1.0 / fs), S_mt


def _interp_signed_linear(f_old, y_old, f_new):
    f_old = np.asarray(f_old, dtype=float)
    y_old = np.asarray(y_old, dtype=float)
    f_new = np.asarray(f_new, dtype=float)
    mask = np.isfinite(f_old) & np.isfinite(y_old) & (f_old > 0.0)
    if np.count_nonzero(mask) < 2:
        return np.zeros_like(f_new, dtype=float)
    f_old = f_old[mask]
    y_old = y_old[mask]
    order = np.argsort(f_old)
    return np.interp(f_new, f_old[order], y_old[order], left=y_old[order][0], right=y_old[order][-1])


def smooth_signed_negative_cospectrum_1d(freq_array, C_1d, band_config, floor=1e-20, f_fit_min=1.0, f_fit_max=None, min_points_per_bin=4, left_mode="band_median", right_mode="slope", right_slope_clip=(-8.0, 0.0), low_plateau_band=(0.2, 1.0), smooth_knots=True, knot_smooth_kernel=(1,2,1)):
    """Smooth a mostly-negative co-spectrum by fitting log(-C)."""
    f = np.asarray(freq_array, dtype=float)
    C = np.asarray(C_1d, dtype=float)
    M = np.maximum(-C, floor)
    try:
        S_fit, knot_dict, _ = LES._profileCalibration.smooth_log_spectrum_1d_binned_pchip(
            f,
            M,
            band_config=band_config,
            floor=floor,
            f_fit_min=f_fit_min,
            f_fit_max=f_fit_max,
            min_points_per_bin=min_points_per_bin,
            left_mode=left_mode,
            right_mode=right_mode,
            right_slope_clip=right_slope_clip,
            low_plateau_band=low_plateau_band,
            smooth_knots=smooth_knots,
            knot_smooth_kernel=knot_smooth_kernel,
        )
        C_fit = -np.maximum(S_fit, floor)
        if knot_dict is not None:
            knot_dict = dict(knot_dict)
            knot_dict["C_knots"] = -np.maximum(knot_dict.get("S_knots", np.array([])), floor)
        return C_fit, knot_dict
    except Exception:
        return -M, {"f_knots": np.array([]), "S_knots": np.array([]), "C_knots": np.array([])}


def get_downstream_uw_cospectrum_array(fMax, nFreq, vel_array_3d, time_step, time_bandwidth=4.0, num_tapers=None):
    fs = 1.0 / time_step
    target_f = LES._profileCalibration.get_freq_array(fMax, nFreq)
    n_heights = vel_array_3d.shape[2]
    C_fit_array = np.zeros((n_heights, nFreq), dtype=float)
    raw = [None for _ in range(n_heights)]
    binned = [None for _ in range(n_heights)]
    for h in range(n_heights):
        u = vel_array_3d[0, :, h]
        w = vel_array_3d[2, :, h]
        raw_f, raw_Suw = _multitaper_csd_1d(u, w, fs, time_bandwidth=time_bandwidth, num_tapers=num_tapers)
        raw_C = np.real(raw_Suw)
        raw_on_grid = _interp_signed_linear(raw_f[(raw_f > 0.0) & (raw_f <= fMax)], raw_C[(raw_f > 0.0) & (raw_f <= fMax)], target_f)
        if UW_ENFORCE_NEGATIVE_COSPECTRUM:
            raw_on_grid = -np.maximum(-raw_on_grid, UW_MAGNITUDE_FLOOR)
        C_fit, knot_dict = smooth_signed_negative_cospectrum_1d(
            target_f,
            raw_on_grid,
            band_config=UW_BAND_CONFIG,
            floor=UW_MAGNITUDE_FLOOR,
            f_fit_min=UW_PSD_FIT_MIN,
            f_fit_max=UW_PSD_FIT_MAX,
            min_points_per_bin=UW_MIN_POINTS_PER_BIN,
            left_mode=UW_LEFT_MODE,
            right_mode=UW_RIGHT_MODE,
            right_slope_clip=UW_RIGHT_SLOPE_CLIP,
            low_plateau_band=UW_LOW_PLATEAU_BAND,
            smooth_knots=UW_SMOOTH_KNOTS,
            knot_smooth_kernel=UW_KNOT_SMOOTH_KERNEL,
        )
        C_fit_array[h, :] = C_fit
        raw[h] = (target_f, raw_on_grid)
        if knot_dict is None:
            binned[h] = (np.array([]), np.array([]))
        else:
            binned[h] = (np.asarray(knot_dict.get("f_knots", []), dtype=float), np.asarray(knot_dict.get("C_knots", []), dtype=float))
    return C_fit_array, raw, binned


def smooth_cospectrum_height_kernel(z_array, C_array, kernel_weights=(1,2,1)):
    C = np.asarray(C_array, dtype=float).copy()
    if kernel_weights is None or len(kernel_weights) <= 1:
        return C
    out = C.copy()
    for f_id in range(C.shape[1]):
        # Smooth log magnitude, preserve negative sign.
        mag = np.maximum(-C[:, f_id], UW_MAGNITUDE_FLOOR)
        sm = _smooth_1d_with_kernel(np.log(mag), kernel_weights)
        out[:, f_id] = -np.exp(sm)
    return out


def get_first_uw_knot_frequency_array(uw_binned, n_heights, fallback=1.0):
    out = np.full(n_heights, fallback, dtype=float)
    for h in range(n_heights):
        item = uw_binned[h]
        if item is None:
            continue
        f_knots = np.asarray(item[0], dtype=float)
        f_knots = f_knots[np.isfinite(f_knots) & (f_knots > 0.0)]
        if len(f_knots) > 0:
            out[h] = f_knots[0]
    return out


def get_uw_knot_frequency_lists(uw_binned, n_heights, fallback=1.0):
    out = []
    for h in range(n_heights):
        item = uw_binned[h]
        if item is None:
            out.append(np.array([fallback], dtype=float))
            continue
        f_knots = np.asarray(item[0], dtype=float)
        f_knots = f_knots[np.isfinite(f_knots) & (f_knots > 0.0)]
        if len(f_knots) == 0:
            f_knots = np.array([fallback], dtype=float)
        out.append(f_knots)
    return out


def cross_covariance_fft_1d(x, y, max_lags=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask] - np.nanmean(x[mask])
    y = y[mask] - np.nanmean(y[mask])
    n = len(x)
    if max_lags is None or max_lags > n:
        max_lags = n
    nfft = 1 << (2 * n - 1).bit_length()
    X = np.fft.fft(x, nfft)
    Y = np.fft.fft(y, nfft)
    # R_uw(tau) = <u(t+tau) w(t)> for tau >= 0.
    corr = np.fft.ifft(np.conj(X) * Y).real[:max_lags]
    denom = np.arange(n, n - max_lags, -1, dtype=float)
    return corr / np.maximum(denom, 1.0)


def cross_covariance_array_from_velocity(vel_array_3d, max_lags):
    n_heights = vel_array_3d.shape[2]
    out = np.zeros((n_heights, max_lags), dtype=float)
    for h in range(n_heights):
        out[h, :] = cross_covariance_fft_1d(vel_array_3d[0, :, h], vel_array_3d[2, :, h], max_lags=max_lags)
    return out


def cross_covariance_from_cospectrum(freq_array, C_array, tau_array, f_min=None, f_max=None):
    f = np.asarray(freq_array, dtype=float)
    tau = np.asarray(tau_array, dtype=float)
    C = np.asarray(C_array, dtype=float)
    out = np.zeros((C.shape[0], len(tau)), dtype=float)
    for h in range(C.shape[0]):
        mask = _band_mask_for_height(f, h, f_min=f_min, f_max=f_max)
        if np.count_nonzero(mask) < 2:
            continue
        cos_matrix = np.cos(2.0 * np.pi * tau[:, np.newaxis] * f[mask][np.newaxis, :])
        out[h, :] = _trapz(C[h, mask][np.newaxis, :] * cos_matrix, f[mask], axis=1)
    return out


def cospectrum_from_cross_covariance(tau_array, R_array, freq_array, apply_taper=True, taper_start_fraction=0.75):
    tau = np.asarray(tau_array, dtype=float)
    f = np.asarray(freq_array, dtype=float)
    R = np.asarray(R_array, dtype=float).copy()
    if apply_taper and len(tau) > 4:
        start = int(np.clip(round(taper_start_fraction * len(tau)), 1, len(tau) - 1))
        taper = np.ones(len(tau), dtype=float)
        tail = len(tau) - start
        taper[start:] = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, tail)))
        R *= taper[np.newaxis, :]
    out = np.zeros((R.shape[0], len(f)), dtype=float)
    for h in range(R.shape[0]):
        cos_matrix = np.cos(2.0 * np.pi * f[:, np.newaxis] * tau[np.newaxis, :])
        # For a one-sided co-spectrum convention: R(tau) = int C(f) cos(2pi f tau) df.
        # The inverse is approximated with 2 * int R(tau) cos(2pi f tau) d tau.
        out[h, :] = 2.0 * _trapz(R[h, :][np.newaxis, :] * cos_matrix, tau, axis=1)
    return out


def build_join_matched_signed_cospectrum(freq_array, spectral_baseline_C, crosscov_C, spectral_knot_freqs, first_knot_freqs, floor=1e-20, low_n_knots=8, low_max_fraction_of_first_knot=0.95, ratio_min=0.5, ratio_max=2.0, f_min_for_low=None, add_endpoint_anchors=True, join_scale_min=0.33, join_scale_max=3.0):
    f = np.asarray(freq_array, dtype=float)
    C_spec = np.asarray(spectral_baseline_C, dtype=float)
    C_low = np.asarray(crosscov_C, dtype=float)
    n_heights = C_spec.shape[0]
    hybrid = np.zeros_like(C_spec)
    matched_low = np.zeros_like(C_low)
    rows = []
    knot_store = [[None] for _ in range(n_heights)]
    for h in range(n_heights):
        spec = C_spec[h, :].copy()
        low = C_low[h, :].copy()
        if UW_ENFORCE_NEGATIVE_COSPECTRUM:
            spec = -np.maximum(-spec, floor)
            low = -np.maximum(-low, floor)
        f_join = float(first_knot_freqs[h]) if np.isfinite(first_knot_freqs[h]) else UW_HYBRID_FIRST_KNOT_FALLBACK
        f_join = float(np.clip(f_join, f[0], f[-1]))
        spec_join = np.interp(np.log(f_join), np.log(f), spec)
        low_join = np.interp(np.log(f_join), np.log(f), low)
        raw_scale = spec_join / low_join if abs(low_join) > floor else 1.0
        # Use only positive scale. If signs disagree, use no scale and rely on clipping/fallback magnitude.
        if not np.isfinite(raw_scale) or raw_scale <= 0.0:
            raw_scale = 1.0
        scale = float(np.clip(raw_scale, join_scale_min, join_scale_max))
        low_shift = low * scale
        if UW_ENFORCE_NEGATIVE_COSPECTRUM:
            # Bound shifted low branch magnitude relative to spectral branch.
            low_mag = np.maximum(-low_shift, floor)
            spec_mag = np.maximum(-spec, floor)
            low_mag = np.clip(low_mag, ratio_min * spec_mag, ratio_max * spec_mag)
            low_shift = -low_mag
        matched_low[h, :] = low_shift
        low_knots_f, low_knots_C = get_autocorr_low_frequency_knots_explicit(
            freq_array=f,
            autocorr_spectrum_1d=-low_shift if UW_ENFORCE_NEGATIVE_COSPECTRUM else np.abs(low_shift),
            first_spectral_knot_freq=f_join,
            n_knots=low_n_knots,
            max_fraction_of_first_knot=low_max_fraction_of_first_knot,
            min_points_per_bin=HYBRID_LOW_FREQ_MIN_POINTS_PER_BIN,
            floor=floor,
            f_min_for_low=f_min_for_low,
        )
        low_knots_C = -np.maximum(low_knots_C, floor) if UW_ENFORCE_NEGATIVE_COSPECTRUM else low_knots_C
        f_spec_knots = np.asarray(spectral_knot_freqs[h], dtype=float)
        f_spec_knots = f_spec_knots[np.isfinite(f_spec_knots) & (f_spec_knots > 0.0)]
        if len(f_spec_knots) == 0:
            f_spec_knots = np.array([f_join], dtype=float)
        C_spec_knots = np.interp(np.log(f_spec_knots), np.log(f), spec)
        all_f = np.r_[low_knots_f, f_spec_knots]
        all_C = np.r_[low_knots_C, C_spec_knots]
        if add_endpoint_anchors:
            all_f = np.r_[f[0], all_f, f[-1]]
            all_C = np.r_[low_shift[0], all_C, spec[-1]]
        valid = np.isfinite(all_f) & np.isfinite(all_C) & (all_f > 0.0) & (np.abs(all_C) > floor)
        all_f = all_f[valid]
        all_C = all_C[valid]
        order = np.argsort(all_f)
        all_f = all_f[order]
        all_C = all_C[order]
        all_f, all_C = _merge_duplicate_x(all_f, all_C)
        if len(all_f) >= 2:
            logf = np.log(all_f)
            logmag = np.log(np.maximum(np.abs(all_C), floor))
            interp = PchipInterpolator(logf, logmag, extrapolate=True)
            mag = np.exp(interp(np.log(f)))
            sign = -1.0 if UW_ENFORCE_NEGATIVE_COSPECTRUM else np.sign(np.nanmedian(all_C)) or -1.0
            hybrid[h, :] = sign * mag
        else:
            hybrid[h, :] = spec
        knot_store[h][0] = {"f": all_f, "C": all_C}
        rows.append({
            "height_id": h,
            "first_spectral_knot_freq": f_join,
            "raw_join_scale": raw_scale,
            "applied_join_scale": scale,
            "join_scale_was_clipped": bool(abs(scale - raw_scale) > 1e-12),
            "n_low_knots": int(len(low_knots_f)),
            "n_total_knots": int(len(all_f)),
        })
    return hybrid, pd.DataFrame(rows), knot_store, matched_low


def clip_cospectrum_to_realisability(C_array, Su_array, Sw_array, rho_max=0.95, floor=1e-30):
    C = np.asarray(C_array, dtype=float).copy()
    bound = float(rho_max) * np.sqrt(np.maximum(Su_array, floor) * np.maximum(Sw_array, floor))
    C_clipped = np.clip(C, -bound, bound)
    clipped_fraction = np.mean(np.abs(C_clipped - C) > 1e-12)
    return C_clipped, clipped_fraction, bound


def plot_uw_cospectra(fig_dir, z_array, freq_array, inlet_C, target_C, downstream_C, spectral_C, crosscov_C, hybrid_C, first_knot_freqs, body_height, z_max_factor=3.0, n_heights=8):
    safe_makedirs(fig_dir)
    z = np.asarray(z_array, dtype=float)
    candidates = np.where(z <= z_max_factor * body_height)[0]
    if len(candidates) == 0:
        candidates = np.arange(len(z))
    ids = np.linspace(0, len(candidates)-1, min(n_heights, len(candidates)), dtype=int)
    for h in candidates[ids]:
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        ax.loglog(freq_array, np.maximum(-inlet_C[h], UW_MAGNITUDE_FLOOR), label="Current inlet -Cuw")
        ax.loglog(freq_array, np.maximum(-target_C[h], UW_MAGNITUDE_FLOOR), label="Target -Cuw")
        ax.loglog(freq_array, np.maximum(-downstream_C[h], UW_MAGNITUDE_FLOOR), label="Downstream smoothed -Cuw")
        ax.loglog(freq_array, np.maximum(-spectral_C[h], UW_MAGNITUDE_FLOOR), label="Spectral baseline -Cuw")
        ax.loglog(freq_array, np.maximum(-crosscov_C[h], UW_MAGNITUDE_FLOOR), linestyle="--", alpha=0.75, label="Cross-cov derived -Cuw")
        ax.loglog(freq_array, np.maximum(-hybrid_C[h], UW_MAGNITUDE_FLOOR), linewidth=2.0, label="Hybrid final -Cuw")
        ax.axvline(first_knot_freqs[h], linestyle=":", alpha=0.5, label="join")
        ax.set_xlabel("f [Hz]")
        ax.set_ylabel("-Cuw [m2/s2/Hz]")
        ax.set_title(f"u-w co-spectrum, z/H={z[h]/body_height:.2f}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        safe_savefig(fig, os.path.join(fig_dir, f"uw_cospectrum_h{h:03d}.png"))
        plt.close(fig)


def plot_uw_stress_profiles(fig_dir, z_array, body_height, inlet_uw, downstream_uw, target_uw, updated_uw, final_uw):
    safe_makedirs(fig_dir)
    y = np.asarray(z_array, dtype=float) / body_height
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.plot(inlet_uw, y, label="Current inlet")
    ax.plot(downstream_uw, y, label="Downstream time series")
    ax.plot(target_uw, y, label="Target")
    ax.plot(updated_uw, y, label="Updated diagnostic")
    ax.plot(final_uw, y, label="Final hybrid area")
    ax.set_xlabel("uw stress [m2/s2]")
    ax.set_ylabel("z/H")
    ax.grid(True, alpha=0.3)
    ax.legend()
    safe_savefig(fig, os.path.join(fig_dir, "uw_stress_profiles.png"))
    plt.close(fig)

#%% --------------------------------------------------------------------------
# Case setup and baseline downstream profile calculation
# ---------------------------------------------------------------------------

variable_dict = LES._caseFiles.parse_setup_file(case_path)
building_height = variable_dict["buildingHeight"]
lower_z_threshold = variable_dict["lowerZThreshold"]
upper_z_threshold = variable_dict["upperZThreshold"]
rmse_threshold = variable_dict["rmseThreshold"]
mesh_size = variable_dict["meshSize"]
fMax = variable_dict["fMax"]
nFreq = variable_dict["nFreq"]

json_path = os.path.join(case_path, "log", "downstreamCalibration", "sim_init.json")
with open(json_path, "r") as f:
    dfsr_les_init_dict = json.load(f)
burn_in_time = dfsr_les_init_dict["burn_in_time"]

target_profile_df = LES._profileCalibration.get_dfsr_target_profile_df(case_path)
target_profile_array = LES._profileCalibration.get_dfsr_target_profile_array(case_path)
inlet_profile_array = LES._profileCalibration.get_current_dfsr_inlet_profile_array(case_path)

vel_array_3d_full = LES._profileAnalysis.get_velocity_components(downstream_probes_folder)
time_steps_full = LES._profileAnalysis.get_time_steps_probe_data(downstream_probes_folder)
mask = time_steps_full > burn_in_time
vel_array_3d = vel_array_3d_full[:, mask, :]
time_steps = time_steps_full[mask]
time_step = float(np.mean(np.diff(time_steps)))

# Burn-in is already applied, so avoid double filtering inside windLespy.
downstream_profile_array = LES._profileCalibration.get_downstream_dfsr_profile_array(
    vel_array_3d,
    time_step,
    inlet_or_downstream="inlet",
    burn_in_time=None,
    time_steps=None,
)

new_inlet_profile_array = LES._profileCalibration.update_mean_profile_only(
    inlet_profile_array,
    target_profile_array,
    downstream_profile_array,
    relaxation_factor=MEAN_PROFILE_RELAXATION_FACTOR,
)

lower_z_threshold_id, upper_z_threshold_id = LES._profileCalibration.get_avg_z_thresolds_ids(
    target_profile_df,
    lower_z_threshold,
    upper_z_threshold,
)
rmse_array = LES._profileCalibration.get_rmse(
    downstream_profile_array,
    target_profile_array,
    lower_z_threshold_id,
    upper_z_threshold_id,
)
iter_status = LES._profileCalibration.dfsr_iter_status(case_path, rmse_array, rmse_threshold, "downstream")
LES._caseFiles.write_dfsr_iter_json(case_path, iter_status, "downstream")
iteration = iter_status["iteration"]
converged = iter_status["converged"]
stagnated = iter_status["stagnated"]

freq_array = LES._profileCalibration.get_freq_array(fMax, nFreq)
z_array = target_profile_df["z"].to_numpy(dtype=float)

target_spectra_array, target_spectra_z, target_uw_from_spectra, target_cuw_from_spectra = read_spectra_profile_file_extended(case_path, "targetSpectraProfile", n_freq_expected=nFreq)
inlet_spectra_array, inlet_spectra_z, inlet_uw_from_spectra, inlet_cuw_from_spectra = read_spectra_profile_file_extended(case_path, "spectraProfile", n_freq_expected=nFreq)

# Optional separate u-w co-spectrum profile files override embedded Cuw blocks if present.
_target_cuw_sep, _target_cuw_z, _target_cuw_stress = read_uw_cospectrum_profile(case_path, "targetUWCoSpectrumProfile", n_freq_expected=nFreq)
_inlet_cuw_sep, _inlet_cuw_z, _inlet_cuw_stress = read_uw_cospectrum_profile(case_path, SPECTRA_PROFILE_UW_FILENAME, n_freq_expected=nFreq)
if _target_cuw_sep is not None:
    target_cuw_from_spectra = _target_cuw_sep
    target_uw_from_spectra = _target_cuw_stress
if _inlet_cuw_sep is not None:
    inlet_cuw_from_spectra = _inlet_cuw_sep
    inlet_uw_from_spectra = _inlet_cuw_stress

fig_root = os.path.join(case_path, "log", f"it{iteration}", "hybrid")
safe_makedirs(fig_root)


#%% --------------------------------------------------------------------------
# Standard profile diagnostic plots
# ---------------------------------------------------------------------------

fig_folder = os.path.join(fig_root, "profiles")
safe_makedirs(fig_folder)
height_mask = (target_profile_df["z"] <= (3.0 * building_height)).to_numpy()
norm_heights = target_profile_df["z"].to_numpy(dtype=float) / building_height
norm_heights = norm_heights[height_mask]

for col_index, x_axis_desc in enumerate(target_profile_df.columns[1:]):
    profile_list = []
    plot_descs = []
    if "I" in x_axis_desc:
        downstream_profile = np.sqrt(downstream_profile_array[height_mask, col_index]) / downstream_profile_array[height_mask, 0]
        target_profile = np.sqrt(target_profile_array[height_mask, col_index]) / target_profile_array[height_mask, 0]
    else:
        downstream_profile = downstream_profile_array[height_mask, col_index]
        target_profile = target_profile_array[height_mask, col_index]
    profile_list.append(downstream_profile)
    plot_descs.append("Downstream Profile")
    profile_list.append(target_profile)
    plot_descs.append("Target Profile")
    profiles_array = np.stack(profile_list, axis=0)
    fig = LES._plot.plot_profile(profiles_array, norm_heights, x_axis_desc, "z/H", xlims=None, ylims=None, several=True, descs=plot_descs)
    safe_savefig(fig, os.path.join(fig_folder, f"{x_axis_desc}_profiles.png"))
    plt.close(fig)


#%% --------------------------------------------------------------------------
# Downstream fitted spectrum and spectral baseline update
# ---------------------------------------------------------------------------

(
    downstream_spectra_spline_array,
    downstream_raw_multitaper,
    downstream_binned_multitaper,
) = LES._profileCalibration.get_downstream_spectra_array(
    fMax,
    nFreq,
    vel_array_3d,
    time_step,
    building_height,
    downstream_profile_array,
    inlet_or_downstream="inlet",
    burn_in_time=None,
    time_steps=None,
    method="multitaper",
    psd_fit_min=PSD_FIT_MIN,
    psd_fit_max=PSD_FIT_MAX,
    psd_band_config=PSD_BAND_CONFIG,
    min_points_per_bin=PSD_MIN_POINTS_PER_BIN,
    psd_left_mode=PSD_LEFT_MODE,
    psd_right_mode=PSD_RIGHT_MODE,
    psd_right_slope_clip=PSD_RIGHT_SLOPE_CLIP,
    psd_low_plateau_band=PSD_LOW_PLATEAU_BAND,
    psd_smooth_knots=PSD_SMOOTH_KNOTS,
    psd_knot_smooth_kernel=PSD_KNOT_SMOOTH_KERNEL,
    time_bandwidth=4.0,
    num_tapers=None,
    return_raw=True,
    floor=FLOOR,
)

downstream_raw_psd = unpack_packed_raw_or_binned_array(downstream_raw_multitaper, freq_array=freq_array, name="downstream_raw_multitaper")
first_knot_freqs = get_first_spline_frequency_array(
    downstream_binned_multitaper,
    n_comp=3,
    n_heights=len(z_array),
    fallback=HYBRID_FIRST_KNOT_FALLBACK,
)
spectral_knot_freqs = get_spectral_knot_frequency_lists(
    downstream_binned_multitaper,
    n_comp=3,
    n_heights=len(z_array),
    fallback=HYBRID_FIRST_KNOT_FALLBACK,
)

downstream_spectra_smoothed_array = LES._profileCalibration.smooth_spectra_array_height_kernel(
    z_array,
    downstream_spectra_spline_array,
    kernel_weights=PSD_HEIGHT_KERNEL,
    floor=FLOOR,
)

inverse_transfer_raw = LES._profileCalibration.get_inverse_transfer_function(
    inlet_spectra_array,
    downstream_spectra_smoothed_array,
    floor=FLOOR,
)
inverse_transfer_smoothed = LES._profileCalibration.smooth_spectral_ratio_array(
    freq_array,
    inverse_transfer_raw,
    window_length=TRANSFER_SAVGOL_WINDOW,
    polyorder=TRANSFER_SAVGOL_POLYORDER,
    floor=FLOOR,
)

spectral_baseline_array = LES._profileCalibration.get_updated_spectra_array_wong(
    inlet_spectra_array,
    target_spectra_array,
    downstream_spectra_smoothed_array,
    inverse_transfer_function=None,
    relaxation_factor=SPECTRAL_RELAXATION_FACTOR,
    floor=FLOOR,
)


#%% --------------------------------------------------------------------------
# LES-resolved frequency limits and resolved-band variances
# ---------------------------------------------------------------------------

sample_duration = float(time_steps[-1] - time_steps[0])
resolved_f_min = 1.0 / sample_duration if RESOLVED_F_MIN_OVERRIDE is None else float(RESOLVED_F_MIN_OVERRIDE)
resolved_f_nyquist = 1.0 / (2.0 * time_step)

U_target_for_cutoff = target_profile_array[:, 0]
int_length_scales_for_cutoff = target_profile_array[:, -3:].T
sigmas_for_cutoff = np.sqrt(target_profile_array[:, 1:4]).T
mesh_cutoff_freqs = LES._profileAnalysis.get_mesh_cutoff_frequencies(
    mesh_size,
    U_target_for_cutoff,
    int_length_scales_for_cutoff,
    sigmas_for_cutoff,
)
if RESOLVED_F_MAX_OVERRIDE is not None:
    resolved_f_max = float(RESOLVED_F_MAX_OVERRIDE)
elif USE_HEIGHT_DEPENDENT_RESOLVED_FMAX:
    resolved_f_max = np.minimum(mesh_cutoff_freqs, resolved_f_nyquist)
else:
    resolved_f_max = float(min(np.nanmin(mesh_cutoff_freqs), resolved_f_nyquist))

inlet_sigma2 = integrate_spectra_area(freq_array, inlet_spectra_array, f_min=resolved_f_min, f_max=resolved_f_max)
target_sigma2 = integrate_spectra_area(freq_array, target_spectra_array, f_min=resolved_f_min, f_max=resolved_f_max)
if DOWNSTREAM_VARIANCE_SOURCE == "time_series":
    downstream_sigma2 = downstream_profile_array[:, 1:4].T
else:
    downstream_sigma2 = integrate_spectra_area(freq_array, downstream_spectra_smoothed_array, f_min=resolved_f_min, f_max=resolved_f_max)

updated_sigma2 = wong_update_variance(
    inlet_sigma2,
    target_sigma2,
    downstream_sigma2,
    relaxation_factor=VARIANCE_RELAXATION_FACTOR,
    floor=FLOOR,
)

if RENORMALISE_SPECTRAL_BASELINE_TO_UPDATED_VARIANCE:
    spectral_baseline_array = renormalise_spectra_to_variance(
        freq_array,
        spectral_baseline_array,
        updated_sigma2,
        floor=FLOOR,
        f_min=resolved_f_min,
        f_max=resolved_f_max,
    )

print("\nHybrid calibration settings:")
print(f"  resolved f_min = {resolved_f_min:.4g} Hz")
if np.ndim(resolved_f_max) == 0:
    print(f"  resolved f_max = {resolved_f_max:.4g} Hz")
else:
    print(f"  resolved f_max min/max = {np.nanmin(resolved_f_max):.4g} / {np.nanmax(resolved_f_max):.4g} Hz")
print(f"  first spline/bin frequency min/max = {np.nanmin(first_knot_freqs):.4g} / {np.nanmax(first_knot_freqs):.4g} Hz")
print(f"  downstream variance source = {DOWNSTREAM_VARIANCE_SOURCE}")


#%% --------------------------------------------------------------------------
# Autocorrelation-derived update spectrum used only to inform the low-frequency hybrid branch
# ---------------------------------------------------------------------------

tau_array = make_tau_array_from_target_lengths(
    target_profile_array,
    time_step,
    factor=TAU_MAX_FACTOR_OF_MAX_TARGET_T,
    tau_min=TAU_MAX_MIN_SECONDS,
    tau_max=TAU_MAX_MAX_SECONDS,
)
print("\nAutocorrelation branch:")
print(f"  n_tau = {len(tau_array)}")
print(f"  tau_max = {tau_array[-1]:.4g} s")
print(f"  dt = {time_step:.4g} s")

rho_downstream = autocorrelation_array_from_velocity(vel_array_3d, max_lags=len(tau_array))
R_inlet = autocovariance_from_spectrum_array(freq_array, inlet_spectra_array, tau_array, floor=FLOOR, f_min=resolved_f_min, f_max=resolved_f_max)
rho_inlet = normalise_autocovariance_to_rho(R_inlet)
R_target = autocovariance_from_spectrum_array(freq_array, target_spectra_array, tau_array, floor=FLOOR, f_min=resolved_f_min, f_max=resolved_f_max)
rho_target = normalise_autocovariance_to_rho(R_target)

rho_raw_update = rho_inlet + AUTOCORR_RELAXATION_FACTOR * (rho_target - rho_downstream)
rho_raw_update = np.clip(rho_raw_update, -1.0, 1.0)
rho_raw_update[:, :, 0] = 1.0

if USE_EXPONENTIAL_RHO_FIT:
    rho_updated, autocorr_diag = build_exponential_fitted_updated_rho(
        tau_array=tau_array,
        rho_raw=rho_raw_update,
        rho_target=rho_target,
        rho_downstream=rho_downstream,
        dt=time_step,
        zero_tol=EXP_ZERO_TOL,
        zero_persistence_points=AUTOCORR_ZERO_PERSISTENCE_POINTS,
        zero_lookahead_points=AUTOCORR_ZERO_LOOKAHEAD_POINTS,
        rho_min=EXP_FIT_RHO_MIN,
        rho_max=EXP_FIT_RHO_MAX,
        min_points=EXP_FIT_MIN_POINTS,
        p_bounds=EXP_FIT_P_BOUNDS,
        T_bounds_factor=EXP_FIT_T_BOUNDS_FACTOR,
        accept_raw_zero_factor=EXP_ACCEPT_RAW_ZERO_FACTOR,
        accept_raw_zero_abs_pad=EXP_ACCEPT_RAW_ZERO_ABS_PAD,
        zero_ref_factor=EXP_ZERO_REF_FACTOR,
        min_zero_tau_factor=EXP_MIN_ZERO_TAU_FACTOR,
        max_zero_tau_factor=EXP_MAX_ZERO_TAU_FACTOR,
    )
else:
    rho_updated, autocorr_diag = residual_update_autocorrelation_full_raw(
        rho_inlet,
        rho_target,
        rho_downstream,
        tau_array=tau_array,
        relaxation_factor=AUTOCORR_RELAXATION_FACTOR,
        clip=True,
        zero_tol=AUTOCORR_ZERO_TOL,
        zero_persistence_points=AUTOCORR_ZERO_PERSISTENCE_POINTS,
        zero_lookahead_points=AUTOCORR_ZERO_LOOKAHEAD_POINTS,
    )

R_updated = updated_sigma2[:, :, np.newaxis] * rho_updated
autocorr_spectra_array, negative_fraction = spectrum_from_autocovariance_array(
    tau_array=tau_array,
    autocovariance_array=R_updated,
    freq_array=freq_array,
    floor=FLOOR,
    apply_taper=APPLY_AUTOCOVARIANCE_TAPER,
    taper_start_fraction=TAPER_START_FRACTION,
    clip_negative=CLIP_NEGATIVE_SPECTRUM_TO_FLOOR,
)
autocorr_spectra_array = renormalise_spectra_to_variance(
    freq_array,
    autocorr_spectra_array,
    updated_sigma2,
    floor=FLOOR,
    f_min=resolved_f_min,
    f_max=resolved_f_max,
)
print(f"  autocorr spectrum negative fraction min/max = {np.nanmin(negative_fraction):.4g} / {np.nanmax(negative_fraction):.4g}")


#%% --------------------------------------------------------------------------
# u-w co-spectrum branch: high-frequency Cuw + low-frequency cross-covariance
# ---------------------------------------------------------------------------

if INCLUDE_UW_COSPECTRAL_CALIBRATION:
    print("\nu-w co-spectral / cross-covariance branch:")

    target_uw_stress_profile, target_uw_source = get_uw_stress_from_profile_df_or_fallback(
        target_profile_df,
        target_profile_array,
        fallback_rho=UW_FALLBACK_RHO,
    )
    # Current inlet profile files in legacy DFSR normally do not contain uwStress.
    # Prefer spectraProfile:uwStress if present; otherwise use the rho fallback.
    inlet_uw_stress_profile, inlet_uw_source = get_uw_stress_from_profile_df_or_fallback(
        None,
        inlet_profile_array,
        fallback_rho=UW_FALLBACK_RHO,
    )
    if target_uw_from_spectra is not None:
        target_uw_stress_profile = np.asarray(target_uw_from_spectra, dtype=float)
        target_uw_source = "targetSpectraProfile:uwStress"
    if inlet_uw_from_spectra is not None:
        inlet_uw_stress_profile = np.asarray(inlet_uw_from_spectra, dtype=float)
        inlet_uw_source = "spectraProfile:uwStress"

    if target_cuw_from_spectra is None:
        if not UW_USE_KAIMAL_IF_PROFILE_COSPECTRUM_MISSING:
            raise ValueError("No target Cuw profile found and UW_USE_KAIMAL_IF_PROFILE_COSPECTRUM_MISSING is False.")
        target_cuw_shape = kaimal_uw_cospectrum_shape(freq_array, z_array, target_profile_array[:, 0])
        target_uw_cospectrum_array = normalise_cospectrum_to_stress(freq_array, target_cuw_shape, target_uw_stress_profile)
        target_cuw_source = "KaimalShape_normalised_to_target_uwStress"
    else:
        target_uw_cospectrum_array = np.asarray(target_cuw_from_spectra, dtype=float)
        target_cuw_source = "targetSpectraProfile_or_targetUWCoSpectrumProfile:Cuw"

    if inlet_cuw_from_spectra is None:
        inlet_cuw_shape = kaimal_uw_cospectrum_shape(freq_array, z_array, inlet_profile_array[:, 0])
        inlet_uw_cospectrum_array = normalise_cospectrum_to_stress(freq_array, inlet_cuw_shape, inlet_uw_stress_profile)
        inlet_cuw_source = "KaimalShape_normalised_to_inlet_uwStress"
    else:
        inlet_uw_cospectrum_array = np.asarray(inlet_cuw_from_spectra, dtype=float)
        inlet_cuw_source = "spectraProfile_or_uwCoSpectrumProfile:Cuw"

    if UW_ENFORCE_NEGATIVE_COSPECTRUM:
        target_uw_cospectrum_array = -np.maximum(-target_uw_cospectrum_array, UW_MAGNITUDE_FLOOR)
        inlet_uw_cospectrum_array = -np.maximum(-inlet_uw_cospectrum_array, UW_MAGNITUDE_FLOOR)

    downstream_uw_cospectrum_array_rawfit, downstream_uw_raw, downstream_uw_binned = get_downstream_uw_cospectrum_array(
        fMax,
        nFreq,
        vel_array_3d,
        time_step,
        time_bandwidth=4.0,
        num_tapers=None,
    )
    downstream_uw_cospectrum_smoothed_array = smooth_cospectrum_height_kernel(
        z_array,
        downstream_uw_cospectrum_array_rawfit,
        kernel_weights=UW_HEIGHT_KERNEL,
    )

    uw_first_knot_freqs = get_first_uw_knot_frequency_array(
        downstream_uw_binned,
        n_heights=len(z_array),
        fallback=UW_HYBRID_FIRST_KNOT_FALLBACK,
    )
    uw_spectral_knot_freqs = get_uw_knot_frequency_lists(
        downstream_uw_binned,
        n_heights=len(z_array),
        fallback=UW_HYBRID_FIRST_KNOT_FALLBACK,
    )

    Ruw_downstream = cross_covariance_array_from_velocity(vel_array_3d, max_lags=len(tau_array))
    downstream_uw_stress_time_series = Ruw_downstream[:, 0]
    updated_uw_stress_profile = wong_update_uw_stress(
        inlet_uw_stress_profile,
        target_uw_stress_profile,
        downstream_uw_stress_time_series,
        relaxation_factor=UW_STRESS_RELAXATION_FACTOR,
    )

    # High-frequency spectral residual update for Cuw(f).
    uw_spectral_baseline_array = inlet_uw_cospectrum_array + UW_COSPECTRAL_RELAXATION_FACTOR * (
        target_uw_cospectrum_array - downstream_uw_cospectrum_smoothed_array
    )
    if UW_ENFORCE_NEGATIVE_COSPECTRUM:
        uw_spectral_baseline_array = -np.maximum(-uw_spectral_baseline_array, UW_MAGNITUDE_FLOOR)

    # Low-frequency cross-covariance branch.
    Ruw_inlet = cross_covariance_from_cospectrum(
        freq_array,
        inlet_uw_cospectrum_array,
        tau_array,
        f_min=resolved_f_min,
        f_max=resolved_f_max,
    )
    Ruw_target = cross_covariance_from_cospectrum(
        freq_array,
        target_uw_cospectrum_array,
        tau_array,
        f_min=resolved_f_min,
        f_max=resolved_f_max,
    )
    Ruw_updated = Ruw_inlet + UW_CROSSCOV_RELAXATION_FACTOR * (Ruw_target - Ruw_downstream)
    Ruw_updated[:, 0] = updated_uw_stress_profile

    uw_crosscov_cospectrum_array = cospectrum_from_cross_covariance(
        tau_array,
        Ruw_updated,
        freq_array,
        apply_taper=APPLY_AUTOCOVARIANCE_TAPER,
        taper_start_fraction=TAPER_START_FRACTION,
    )
    if UW_ENFORCE_NEGATIVE_COSPECTRUM:
        uw_crosscov_cospectrum_array = -np.maximum(-uw_crosscov_cospectrum_array, UW_MAGNITUDE_FLOOR)

    hybrid_uw_cospectrum_preclip_array, uw_knot_diag_df, uw_knot_store, uw_crosscov_matched_array = build_join_matched_signed_cospectrum(
        freq_array=freq_array,
        spectral_baseline_C=uw_spectral_baseline_array,
        crosscov_C=uw_crosscov_cospectrum_array,
        spectral_knot_freqs=uw_spectral_knot_freqs,
        first_knot_freqs=uw_first_knot_freqs,
        floor=UW_MAGNITUDE_FLOOR,
        low_n_knots=UW_HYBRID_LOW_FREQ_N_KNOTS,
        low_max_fraction_of_first_knot=UW_HYBRID_LOW_FREQ_MAX_FRACTION_OF_FIRST_KNOT,
        ratio_min=UW_HYBRID_LOW_FREQ_RATIO_MIN,
        ratio_max=UW_HYBRID_LOW_FREQ_RATIO_MAX,
        f_min_for_low=resolved_f_min,
        add_endpoint_anchors=UW_HYBRID_ADD_ENDPOINT_ANCHORS,
        join_scale_min=UW_HYBRID_JOIN_SCALE_MIN,
        join_scale_max=UW_HYBRID_JOIN_SCALE_MAX,
    )

    if RENORMALISE_UW_COSPECTRUM_TO_UPDATED_STRESS:
        hybrid_uw_cospectrum_preclip_array = normalise_cospectrum_to_stress(
            freq_array,
            hybrid_uw_cospectrum_preclip_array,
            updated_uw_stress_profile,
            floor=UW_MAGNITUDE_FLOOR,
        )

    print(f"  target uw source = {target_uw_source}")
    print(f"  inlet uw source = {inlet_uw_source}")
    print(f"  target Cuw source = {target_cuw_source}")
    print(f"  inlet Cuw source = {inlet_cuw_source}")
    print(f"  downstream uw stress min/max = {np.nanmin(downstream_uw_stress_time_series):.4g} / {np.nanmax(downstream_uw_stress_time_series):.4g}")
    print(f"  updated uw stress min/max = {np.nanmin(updated_uw_stress_profile):.4g} / {np.nanmax(updated_uw_stress_profile):.4g}")
    if len(uw_knot_diag_df) > 0:
        print(f"  uw join scale applied min/max = {uw_knot_diag_df['applied_join_scale'].min():.4g} / {uw_knot_diag_df['applied_join_scale'].max():.4g}")
else:
    target_uw_stress_profile = None
    inlet_uw_stress_profile = None
    downstream_uw_stress_time_series = None
    updated_uw_stress_profile = None
    target_uw_cospectrum_array = None
    inlet_uw_cospectrum_array = None
    downstream_uw_cospectrum_smoothed_array = None
    uw_spectral_baseline_array = None
    uw_crosscov_cospectrum_array = None
    uw_crosscov_matched_array = None
    hybrid_uw_cospectrum_preclip_array = None
    hybrid_uw_cospectrum_array = None
    uw_knot_diag_df = pd.DataFrame()
    uw_first_knot_freqs = None


#%% --------------------------------------------------------------------------
# Join-matched combined-knot hybrid spectrum
# ---------------------------------------------------------------------------

# Intended autocorrelation update length scale from the full raw rho branch, retained for final diagnostics.
L_acorr_update = integral_length_array_from_rho(tau_array, rho_updated, new_inlet_profile_array[:, 0])

hybrid_spectra_array, knot_diag_df, knot_store, autocorr_spectra_matched_array = build_join_matched_combined_knot_spectrum(
    freq_array=freq_array,
    spectral_baseline_array=spectral_baseline_array,
    autocorr_spectra_array=autocorr_spectra_array,
    spectral_knot_freqs=spectral_knot_freqs,
    first_knot_freqs=first_knot_freqs,
    floor=FLOOR,
    low_n_knots=HYBRID_LOW_FREQ_N_KNOTS,
    low_max_fraction_of_first_knot=HYBRID_LOW_FREQ_MAX_FRACTION_OF_FIRST_KNOT,
    low_min_points_per_bin=HYBRID_LOW_FREQ_MIN_POINTS_PER_BIN,
    ratio_min=HYBRID_LOW_FREQ_RATIO_MIN,
    ratio_max=HYBRID_LOW_FREQ_RATIO_MAX,
    f_min_for_low=resolved_f_min,
    add_endpoint_anchors=HYBRID_ADD_ENDPOINT_ANCHORS,
    match_low_freq_to_spectral_join=HYBRID_MATCH_LOW_FREQ_TO_SPECTRAL_JOIN,
    join_scale_min=HYBRID_JOIN_SCALE_MIN,
    join_scale_max=HYBRID_JOIN_SCALE_MAX,
)

# Compatibility arrays for the existing low-frequency-ratio diagnostic plot.
# The applied ratio is now simply the join-matched autocorrelation branch divided
# by the spectral branch; no beta search is used.
ratio_correction = np.maximum(autocorr_spectra_matched_array, FLOOR) / np.maximum(spectral_baseline_array, FLOOR)
beta_selected = np.ones(spectral_baseline_array.shape[:2], dtype=float)

if len(knot_diag_df) > 0:
    print("\nJoin-matched combined-knot diagnostics:")
    print(
        "  low-frequency autocorr knots requested / actual min-max: "
        f"{HYBRID_LOW_FREQ_N_KNOTS} / "
        f"{int(knot_diag_df['n_low_knots'].min())}-{int(knot_diag_df['n_low_knots'].max())}"
    )
    print(
        "  join scale raw min/max: "
        f"{knot_diag_df['raw_join_scale'].min():.4g} / {knot_diag_df['raw_join_scale'].max():.4g}"
    )
    print(
        "  join scale applied min/max: "
        f"{knot_diag_df['applied_join_scale'].min():.4g} / {knot_diag_df['applied_join_scale'].max():.4g}"
    )
    n_clipped = int(knot_diag_df['join_scale_was_clipped'].sum())
    if n_clipped > 0:
        print(f"  warning: join scale clipped for {n_clipped} component-height pairs.")

if RENORMALISE_HYBRID_TO_UPDATED_VARIANCE:
    hybrid_spectra_array = renormalise_spectra_to_variance(
        freq_array,
        hybrid_spectra_array,
        updated_sigma2,
        floor=FLOOR,
        f_min=resolved_f_min,
        f_max=resolved_f_max,
    )


if APPLY_POWER_LAW_TAIL:
    U_tail = target_profile_array[:, 0]
    L_tail = target_profile_array[:, -3:].T
    sigmas_tail = np.sqrt(target_profile_array[:, 1:4]).T
    mesh_cutoff_freqs_tail = LES._profileAnalysis.get_mesh_cutoff_frequencies(mesh_size, U_tail, L_tail, sigmas_tail)
    mesh_cutoff_freqs_3d = np.broadcast_to(mesh_cutoff_freqs_tail[np.newaxis, :], (3, len(mesh_cutoff_freqs_tail)))
    effective_cutoff_freqs_3d = POWER_LAW_TAIL_CUTOFF_FACTOR * mesh_cutoff_freqs_3d
    hybrid_spectra_array = LES._profileCalibration.apply_power_law_tail(
        freq_array,
        hybrid_spectra_array,
        effective_cutoff_freqs_3d,
        slope=POWER_LAW_TAIL_SLOPE,
        floor=1e-20,
    )
    if RENORMALISE_HYBRID_TO_UPDATED_VARIANCE:
        hybrid_spectra_array = renormalise_spectra_to_variance(
            freq_array,
            hybrid_spectra_array,
            updated_sigma2,
            floor=FLOOR,
            f_min=resolved_f_min,
            f_max=resolved_f_max,
        )


#%% --------------------------------------------------------------------------
# Final u-w co-spectrum realizability against the final auto-spectra
# ---------------------------------------------------------------------------

if INCLUDE_UW_COSPECTRAL_CALIBRATION:
    hybrid_uw_cospectrum_array, uw_realisability_clipped_fraction, uw_realisability_bound = clip_cospectrum_to_realisability(
        hybrid_uw_cospectrum_preclip_array,
        hybrid_spectra_array[0, :, :],
        hybrid_spectra_array[2, :, :],
        rho_max=UW_RHO_MAX,
        floor=FLOOR,
    )
    final_uw_stress_resolved = integrate_cospectrum_area(
        freq_array,
        hybrid_uw_cospectrum_array,
        f_min=resolved_f_min,
        f_max=resolved_f_max,
    )
    preclip_uw_stress_resolved = integrate_cospectrum_area(
        freq_array,
        hybrid_uw_cospectrum_preclip_array,
        f_min=resolved_f_min,
        f_max=resolved_f_max,
    )
    uw_stress_rel_error = (final_uw_stress_resolved - updated_uw_stress_profile) / np.maximum(np.abs(updated_uw_stress_profile), FLOOR)
    print("\nFinal u-w co-spectrum realizability diagnostics:")
    print(f"  Cuw clipping fraction = {uw_realisability_clipped_fraction:.4g}")
    print(f"  final uw stress rel error min/max = {np.nanmin(uw_stress_rel_error):.4g} / {np.nanmax(uw_stress_rel_error):.4g}")
else:
    final_uw_stress_resolved = None
    preclip_uw_stress_resolved = None
    uw_stress_rel_error = None
    uw_realisability_clipped_fraction = None
    uw_realisability_bound = None


#%% --------------------------------------------------------------------------
# Final diagnostics
# ---------------------------------------------------------------------------

rho_inlet_from_spectrum = reconstruct_rho_from_spectra_resolved_band(freq_array, inlet_spectra_array, tau_array, f_min=resolved_f_min, f_max=resolved_f_max, floor=FLOOR)
rho_target_from_spectrum = reconstruct_rho_from_spectra_resolved_band(freq_array, target_spectra_array, tau_array, f_min=resolved_f_min, f_max=resolved_f_max, floor=FLOOR)
rho_spectral_from_spectrum = reconstruct_rho_from_spectra_resolved_band(freq_array, spectral_baseline_array, tau_array, f_min=resolved_f_min, f_max=resolved_f_max, floor=FLOOR)
rho_hybrid_from_spectrum = reconstruct_rho_from_spectra_resolved_band(freq_array, hybrid_spectra_array, tau_array, f_min=resolved_f_min, f_max=resolved_f_max, floor=FLOOR)

L_inlet = integral_length_array_from_rho(tau_array, rho_inlet_from_spectrum, inlet_profile_array[:, 0])
L_target = integral_length_array_from_rho(tau_array, rho_target_from_spectrum, target_profile_array[:, 0])
L_downstream = integral_length_array_from_rho(tau_array, rho_downstream, downstream_profile_array[:, 0])
L_spectral = integral_length_array_from_rho(tau_array, rho_spectral_from_spectrum, new_inlet_profile_array[:, 0])
L_hybrid = integral_length_array_from_rho(tau_array, rho_hybrid_from_spectrum, new_inlet_profile_array[:, 0])

final_sigma2_resolved = integrate_spectra_area(freq_array, hybrid_spectra_array, f_min=resolved_f_min, f_max=resolved_f_max)
final_sigma2_rel_error = (final_sigma2_resolved - updated_sigma2) / np.maximum(updated_sigma2, FLOOR)

rows = []
for h_id, z in enumerate(z_array):
    for comp_id, comp in enumerate(COMPONENT_NAMES):
        rows.append({
            "height_id": h_id,
            "z": z,
            "z_over_H": z / building_height,
            "component": comp,
            "resolved_f_min": resolved_f_min,
            "resolved_f_max": resolved_f_max if np.ndim(resolved_f_max) == 0 else resolved_f_max[h_id],
            "first_spline_freq": first_knot_freqs[comp_id, h_id],
            "join_scale_applied": knot_diag_df.loc[(knot_diag_df["component"] == comp) & (knot_diag_df["height_id"] == h_id), "applied_join_scale"].iloc[0] if len(knot_diag_df) > 0 else np.nan,
            "n_low_freq_knots": knot_diag_df.loc[(knot_diag_df["component"] == comp) & (knot_diag_df["height_id"] == h_id), "n_low_knots"].iloc[0] if len(knot_diag_df) > 0 else np.nan,
            "sigma2_inlet": inlet_sigma2[comp_id, h_id],
            "sigma2_downstream": downstream_sigma2[comp_id, h_id],
            "sigma2_target": target_sigma2[comp_id, h_id],
            "sigma2_updated_wong": updated_sigma2[comp_id, h_id],
            "sigma2_final_hybrid": final_sigma2_resolved[comp_id, h_id],
            "sigma2_final_rel_error": final_sigma2_rel_error[comp_id, h_id],
            "L_profile_target": target_profile_array[h_id, 4 + comp_id],
            "L_profile_downstream": downstream_profile_array[h_id, 4 + comp_id],
            "L_inlet_resolved": L_inlet[comp_id, h_id],
            "L_downstream_time_series": L_downstream[comp_id, h_id],
            "L_target_resolved": L_target[comp_id, h_id],
            "L_acorr_intended": L_acorr_update[comp_id, h_id],
            "L_spectral_baseline": L_spectral[comp_id, h_id],
            "L_hybrid_final": L_hybrid[comp_id, h_id],
            "rho_zero_source": autocorr_diag["zero_source"][comp_id, h_id],
            "rho_selected_zero_tau": autocorr_diag["selected_zero_tau"][comp_id, h_id],
            "rho_fit_T": autocorr_diag["fit_T"][comp_id, h_id],
            "rho_fit_p": autocorr_diag["fit_p"][comp_id, h_id],
        })
summary_df = pd.DataFrame(rows)
summary_df.to_csv(os.path.join(fig_root, "summary.csv"), index=False)
if len(knot_diag_df) > 0:
    knot_diag_df.to_csv(os.path.join(fig_root, "join_matched_knot_summary.csv"), index=False)

if INCLUDE_UW_COSPECTRAL_CALIBRATION:
    uw_rows = []
    inlet_uw_area = integrate_cospectrum_area(freq_array, inlet_uw_cospectrum_array, f_min=resolved_f_min, f_max=resolved_f_max)
    target_uw_area = integrate_cospectrum_area(freq_array, target_uw_cospectrum_array, f_min=resolved_f_min, f_max=resolved_f_max)
    downstream_uw_area = integrate_cospectrum_area(freq_array, downstream_uw_cospectrum_smoothed_array, f_min=resolved_f_min, f_max=resolved_f_max)
    spectral_uw_area = integrate_cospectrum_area(freq_array, uw_spectral_baseline_array, f_min=resolved_f_min, f_max=resolved_f_max)
    crosscov_uw_area = integrate_cospectrum_area(freq_array, uw_crosscov_matched_array, f_min=resolved_f_min, f_max=resolved_f_max)
    for h_id, z in enumerate(z_array):
        join_row = uw_knot_diag_df.loc[uw_knot_diag_df["height_id"] == h_id] if len(uw_knot_diag_df) > 0 else pd.DataFrame()
        uw_rows.append({
            "height_id": h_id,
            "z": z,
            "z_over_H": z / building_height,
            "resolved_f_min": resolved_f_min,
            "resolved_f_max": resolved_f_max if np.ndim(resolved_f_max) == 0 else resolved_f_max[h_id],
            "first_spline_freq": uw_first_knot_freqs[h_id],
            "join_scale_applied": join_row["applied_join_scale"].iloc[0] if len(join_row) > 0 else np.nan,
            "n_low_freq_knots": join_row["n_low_knots"].iloc[0] if len(join_row) > 0 else np.nan,
            "uw_stress_inlet_profile": inlet_uw_stress_profile[h_id],
            "uw_stress_downstream_time_series": downstream_uw_stress_time_series[h_id],
            "uw_stress_target_profile": target_uw_stress_profile[h_id],
            "uw_stress_updated_diagnostic": updated_uw_stress_profile[h_id],
            "uw_area_inlet_cuw": inlet_uw_area[h_id],
            "uw_area_target_cuw": target_uw_area[h_id],
            "uw_area_downstream_cuw": downstream_uw_area[h_id],
            "uw_area_spectral_baseline": spectral_uw_area[h_id],
            "uw_area_crosscov_matched": crosscov_uw_area[h_id],
            "uw_area_preclip_hybrid": preclip_uw_stress_resolved[h_id],
            "uw_area_final_hybrid": final_uw_stress_resolved[h_id],
            "uw_final_rel_error_vs_updated": uw_stress_rel_error[h_id],
        })
    uw_summary_df = pd.DataFrame(uw_rows)
    uw_summary_df.to_csv(os.path.join(fig_root, "uw_cospectral_summary.csv"), index=False)
    if len(uw_knot_diag_df) > 0:
        uw_knot_diag_df.to_csv(os.path.join(fig_root, "uw_join_matched_knot_summary.csv"), index=False)

print("\nFinal hybrid self-consistency diagnostics:")
print(f"  sigma2 rel error min/max = {np.nanmin(final_sigma2_rel_error):.4g} / {np.nanmax(final_sigma2_rel_error):.4g}")
print(f"  join scale applied min/max = {knot_diag_df['applied_join_scale'].min():.4g} / {knot_diag_df['applied_join_scale'].max():.4g}" if len(knot_diag_df) > 0 else "  join scale applied min/max = n/a")
print(f"  L_hybrid - L_acorr_intended min/max = {np.nanmin(L_hybrid - L_acorr_update):.4g} / {np.nanmax(L_hybrid - L_acorr_update):.4g}")


#%% --------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

# Standard windLespy spectral diagnostics from the spectral branch.
try:
    LES._plot.plot_multitaper_spline_spectra(
        os.path.join(fig_root, "01_mt_spline"),
        z_array,
        freq_array,
        downstream_raw_multitaper,
        downstream_binned_multitaper,
        downstream_spectra_spline_array,
        inlet_spectra_array,
        target_spectra_array,
        building_height,
        z_min=0.0,
        z_max_factor=Z_MAX_FACTOR_PLOTS,
    )
    LES._plot.plot_vertical_smoothing_spectra(
        os.path.join(fig_root, "02_vert_smooth"),
        z_array,
        freq_array,
        downstream_spectra_spline_array,
        downstream_spectra_smoothed_array,
        building_height,
        z_min=0.0,
        z_max_factor=Z_MAX_FACTOR_PLOTS,
    )
    LES._plot.plot_transfer_function(
        os.path.join(fig_root, "03_inv_transfer"),
        z_array,
        freq_array,
        inverse_transfer_raw,
        inverse_transfer_smoothed,
        building_height,
        z_min=0.0,
        z_max_factor=Z_MAX_FACTOR_PLOTS,
    )
except Exception as exc:
    print(f"Warning: one or more windLespy diagnostic plots failed: {exc}")

plot_autocorrelation_comparison(
    os.path.join(fig_root, "04_rho"),
    z_array,
    tau_array,
    rho_inlet,
    rho_downstream,
    rho_target,
    rho_raw_update,
    rho_updated,
    building_height,
    z_max_factor=Z_MAX_FACTOR_PLOTS,
    n_heights=N_HEIGHTS_TO_PLOT,
    components=COMPONENT_NAMES,
)

plot_low_frequency_ratio(
    os.path.join(fig_root, "05_low_ratio"),
    z_array,
    freq_array,
    ratio_correction,
    beta_selected,
    first_knot_freqs,
    building_height,
    z_max_factor=Z_MAX_FACTOR_PLOTS,
    n_heights=N_HEIGHTS_TO_PLOT,
    components=COMPONENT_NAMES,
)

plot_hybrid_spectra(
    os.path.join(fig_root, "06_hybrid_spectra"),
    z_array,
    freq_array,
    inlet_spectra_array,
    target_spectra_array,
    spectral_baseline_array,
    autocorr_spectra_matched_array,
    hybrid_spectra_array,
    first_knot_freqs,
    building_height,
    z_max_factor=Z_MAX_FACTOR_FINAL_SPECTRA,
    n_heights=N_HEIGHTS_TO_PLOT,
    components=COMPONENT_NAMES,
    floor=FLOOR,
)

plot_length_profiles(
    os.path.join(fig_root, "07_L_profiles"),
    z_array,
    building_height,
    L_inlet,
    L_downstream,
    L_target,
    L_acorr_update,
    L_spectral,
    L_hybrid,
    components=COMPONENT_NAMES,
)

if INCLUDE_UW_COSPECTRAL_CALIBRATION:
    plot_uw_cospectra(
        os.path.join(fig_root, "09_uw_cospectra"),
        z_array,
        freq_array,
        inlet_uw_cospectrum_array,
        target_uw_cospectrum_array,
        downstream_uw_cospectrum_smoothed_array,
        uw_spectral_baseline_array,
        uw_crosscov_matched_array,
        hybrid_uw_cospectrum_array,
        uw_first_knot_freqs,
        building_height,
        z_max_factor=Z_MAX_FACTOR_FINAL_SPECTRA,
        n_heights=N_HEIGHTS_TO_PLOT,
    )
    plot_uw_stress_profiles(
        os.path.join(fig_root, "10_uw_profiles"),
        z_array,
        building_height,
        inlet_uw_stress_profile,
        downstream_uw_stress_time_series,
        target_uw_stress_profile,
        updated_uw_stress_profile,
        final_uw_stress_resolved,
    )

try:
    LES._plot.plot_spectral_calibration(
        os.path.join(fig_root, "08_final_overview"),
        z_array,
        freq_array,
        downstream_spectra_smoothed_array,
        hybrid_spectra_array,
        inlet_spectra_array,
        target_spectra_array,
        cutoff_freqs=None,
        z_min=0.0,
        z_max=Z_MAX_FACTOR_FINAL_SPECTRA * building_height,
    )
except Exception as exc:
    print(f"Warning: final overview plot failed: {exc}")


#%% --------------------------------------------------------------------------
# Write-back to case and iteration archive
# ---------------------------------------------------------------------------

if WRITE_RESULTS and (not converged) and (not stagnated):
    dfsr_input_spectra_path = os.path.join(case_path, "constant", "boundaryData", "windProfile", "spectraProfile")
    if INCLUDE_UW_COSPECTRAL_CALIBRATION and WRITE_LEGACY_3COMP_SPECTRA_BACKUP:
        backup_path = os.path.join(case_path, "constant", "boundaryData", "windProfile", "spectraProfile_legacy3comp")
        write_spectra_profile_with_optional_uw(
            hybrid_spectra_array,
            z_array,
            backup_path,
            uw_stress=None,
            cuw_array=None,
            clip_min=SPECTRUM_FLOOR_FOR_WRITE,
        )
    if INCLUDE_UW_COSPECTRAL_CALIBRATION and WRITE_AUGMENTED_SPECTRA_PROFILE_WITH_UWSTRESS:
        write_spectra_profile_with_optional_uw(
            hybrid_spectra_array,
            z_array,
            dfsr_input_spectra_path,
            uw_stress=final_uw_stress_resolved,
            cuw_array=None,
            clip_min=SPECTRUM_FLOOR_FOR_WRITE,
        )
    else:
        write_spectra_profile_with_optional_uw(
            hybrid_spectra_array,
            z_array,
            dfsr_input_spectra_path,
            uw_stress=None,
            cuw_array=None,
            clip_min=SPECTRUM_FLOOR_FOR_WRITE,
        )

    if INCLUDE_UW_COSPECTRAL_CALIBRATION and WRITE_UW_COSPECTRA_PROFILE:
        uw_output_path = os.path.join(case_path, "constant", "boundaryData", "windProfile", SPECTRA_PROFILE_UW_FILENAME)
        write_uw_cospectrum_profile(
            hybrid_uw_cospectrum_array,
            z_array,
            final_uw_stress_resolved,
            uw_output_path,
        )

    if INCLUDE_UW_COSPECTRAL_CALIBRATION:
        write_new_dfsr_inlet_profile_with_uw(new_inlet_profile_array, target_profile_df, case_path, uw_stress=final_uw_stress_resolved)
    else:
        LES._caseFiles.write_new_dfsr_inlet_profile(new_inlet_profile_array, target_profile_df, case_path)

if WRITE_ITER_SPECTRA:
    LES._caseFiles.write_dfsr_iter_spectra(
        case_path,
        iter_status,
        z_array,
        freq_array,
        inlet_spectra_array,
        downstream_spectra_smoothed_array,
        inlet_or_downstream="downstream",
        new_inlet_spectra_array=hybrid_spectra_array if (not converged and not stagnated) else None,
        cutoff_freqs=None,
        clip_min=SPECTRUM_FLOOR_FOR_WRITE,
    )
    if INCLUDE_UW_COSPECTRAL_CALIBRATION:
        iter_dir = os.path.join(case_path, "log", "downstreamCalibration", f"iteration{iteration}")
        safe_makedirs(iter_dir)
        write_uw_cospectrum_profile(inlet_uw_cospectrum_array, z_array, inlet_uw_stress_profile, os.path.join(iter_dir, "inletUWCoSpectrumProfile"))
        write_uw_cospectrum_profile(downstream_uw_cospectrum_smoothed_array, z_array, downstream_uw_stress_time_series, os.path.join(iter_dir, "downstreamUWCoSpectrumProfile"))
        write_uw_cospectrum_profile(target_uw_cospectrum_array, z_array, target_uw_stress_profile, os.path.join(iter_dir, "targetUWCoSpectrumProfile"))
        if not converged and not stagnated:
            write_uw_cospectrum_profile(hybrid_uw_cospectrum_array, z_array, final_uw_stress_resolved, os.path.join(iter_dir, "newInletUWCoSpectrumProfile"))


#%% --------------------------------------------------------------------------
# Exit behaviour
# ---------------------------------------------------------------------------

if converged:
    sys.exit(0)
elif stagnated:
    sys.exit(0)
else:
    sys.exit(1)
