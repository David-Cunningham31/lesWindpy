# -*- coding: utf-8 -*-
"""
Standalone downstream DFSR autocorrelation calibration recipe.

This script is intentionally self-contained: it does not require editing windLespy.
It reuses windLespy for case IO, profile processing, spectra reading/writing, and
standard profile plots, but implements the autocorrelation-calibration experiment
locally in this recipe.

Concept:
    1. Read current inlet and target spectra.
    2. Compute downstream autocorrelation directly from velocity time series.
    3. Compute inlet/target autocorrelation from their spectra over the LES-resolved frequency band.
    4. Apply a Wong-style resolved-band variance update and a first-zero-controlled autocorrelation update.
    5. Convert the updated autocovariance back to a one-sided spectrum, then renormalise over the resolved band.
    6. Plot autocorrelation and spectra diagnostics.
    7. Write the updated spectraProfile and inlet profile when the case has not converged/stagnated.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _windows_long_path(path):
    """Return a Windows long-path-safe absolute path for deep OneDrive folders."""
    path = os.path.abspath(path)
    if os.name != "nt":
        return path
    if path.startswith("\\\\?\\"):
        return path
    if path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path[2:]
    return "\\\\?\\" + path


def safe_makedirs(path):
    """Create directories robustly, including long Windows paths."""
    os.makedirs(_windows_long_path(path), exist_ok=True)


def safe_savefig(fig, path, dpi=300, bbox_inches="tight"):
    """Save a matplotlib figure robustly, including long Windows paths."""
    path = os.path.abspath(path)
    safe_makedirs(os.path.dirname(path))
    fig.savefig(_windows_long_path(path), dpi=dpi, bbox_inches=bbox_inches)


try:
    from scipy.optimize import curve_fit
except Exception:
    curve_fit = None

cwd = os.path.dirname(os.path.abspath(__file__))
windlespy_path = os.path.abspath(os.path.join(cwd, "..", ".."))
sys.path.append(windlespy_path)
import windlespy as LES
sys.path.remove(windlespy_path)


#%% --------------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------------

case_path = os.environ["CASE_DIR"]
downstream_probes_folder = os.path.join(case_path, "postProcessing", "probes2")

# Production recipe behaviour, matching the downstream spectral calibration script:
# update the case only while the DFSR loop is still active, and always write
# iteration diagnostics for the current downstream calibration step.
WRITE_RESULTS = True
WRITE_ITER_SPECTRA = True

AUTOCORR_RELAXATION_FACTOR = 0.9
VARIANCE_RELAXATION_FACTOR = 0.9
MEAN_PROFILE_RELAXATION_FACTOR = 0.9

# Tau-domain settings.
TAU_MAX_FACTOR_OF_MAX_TARGET_T = 20.0
TAU_MAX_MIN_SECONDS = None
TAU_MAX_MAX_SECONDS = None

# Resolved-band settings.
# f_min is usually 1 / post-burn-in sample duration.
# f_max is min(mesh cutoff, Nyquist), height-dependent by default.
USE_HEIGHT_DEPENDENT_RESOLVED_FMAX = True
RESOLVED_F_MIN_OVERRIDE = None
RESOLVED_F_MAX_OVERRIDE = None

# Autocorrelation update control.
# The residual update is used only until the UPDATED autocorrelation crosses zero.
# After that point, the target autocorrelation is used as a controlled continuation.
AUTOCORR_POST_ZERO_BLEND_POINTS = 0

# Smooth the updated autocorrelation before transforming back to a spectrum.
# This removes near-zero wiggles that otherwise create high-frequency spectral
# oscillations after the cosine transform.
USE_EXPONENTIAL_RHO_FIT = True
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

# Practical zero-crossing detection.
AUTOCORR_ZERO_TOL = EXP_ZERO_TOL
AUTOCORR_ZERO_PERSISTENCE_POINTS = 3
AUTOCORR_ZERO_LOOKAHEAD_POINTS = 8

# Convert updated autocovariance back to spectrum.
APPLY_AUTOCOVARIANCE_TAPER = True
TAPER_START_FRACTION = 0.75

FLOOR = 1e-16
SPECTRUM_FLOOR_FOR_WRITE = 1e-16
CLIP_NEGATIVE_SPECTRUM_TO_FLOOR = True
RENORMALISE_UPDATED_SPECTRUM_AREA = True

APPLY_POWER_LAW_TAIL = True
POWER_LAW_TAIL_CUTOFF_FACTOR = 0.5
POWER_LAW_TAIL_SLOPE = -5.0 / 3.0

Z_MAX_FACTOR_PLOTS = 1.5
Z_MAX_FACTOR_FINAL_SPECTRA = 3.0
N_HEIGHTS_TO_PLOT = 8
COMPONENT_NAMES = ("u", "v", "w")


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

# The array is already burn-in masked, so use inlet_or_downstream="inlet" to
# avoid double filtering inside windLespy.
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

iter_status = LES._profileCalibration.dfsr_iter_status(
    case_path,
    rmse_array,
    rmse_threshold,
    "downstream",
)

LES._caseFiles.write_dfsr_iter_json(case_path, iter_status, "downstream")

iteration = iter_status["iteration"]
converged = iter_status["converged"]
stagnated = iter_status["stagnated"]

freq_array = LES._profileCalibration.get_freq_array(fMax, nFreq)
z_array = target_profile_df["z"].to_numpy(dtype=float)

target_spectra_array = LES._profileCalibration.read_spectra_profile_file(
    case_path,
    "targetSpectraProfile",
)
inlet_spectra_array = LES._profileCalibration.read_spectra_profile_file(
    case_path,
    "spectraProfile",
)


#%% --------------------------------------------------------------------------
# Standard profile diagnostic plots
# ---------------------------------------------------------------------------

fig_folder = os.path.join(
    case_path,
    "log",
    "downstreamCalibration",
    f"iteration{iteration}",
    "plots",
    "profiles",
)
safe_makedirs(fig_folder)

height_mask = (target_profile_df["z"] <= (3.0 * building_height)).to_numpy()
norm_heights = target_profile_df["z"].to_numpy(dtype=float) / building_height
norm_heights = norm_heights[height_mask]

for col_index, x_axis_desc in enumerate(target_profile_df.columns[1:]):
    profile_list = []
    plot_descs = []

    if "I" in x_axis_desc:
        downstream_profile = (
            np.sqrt(downstream_profile_array[height_mask, col_index])
            / downstream_profile_array[height_mask, 0]
        )
        target_profile = (
            np.sqrt(target_profile_array[height_mask, col_index])
            / target_profile_array[height_mask, 0]
        )
    else:
        downstream_profile = downstream_profile_array[height_mask, col_index]
        target_profile = target_profile_array[height_mask, col_index]

    profile_list.append(downstream_profile)
    plot_descs.append("Downstream Profile")

    profile_list.append(target_profile)
    plot_descs.append("Target Profile")

    profiles_array = np.stack(profile_list, axis=0)

    fig = LES._plot.plot_profile(
        profiles_array,
        norm_heights,
        x_axis_desc,
        "z/H",
        xlims=None,
        ylims=None,
        several=True,
        descs=plot_descs,
    )

    filename = f"{x_axis_desc}_profiles.png"
    safe_savefig(fig, os.path.join(fig_folder, filename))
    plt.close(fig)


#%% --------------------------------------------------------------------------
# Autocorrelation helpers
# ---------------------------------------------------------------------------

def _trapz(y, x, axis=-1):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    return np.trapz(y, x, axis=axis)


def make_tau_array_from_target_lengths(
    target_profile_array,
    dt,
    factor=20.0,
    tau_min=None,
    tau_max=None,
):
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
    """Biased, normalised autocorrelation estimate using FFT."""
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


def autocovariance_from_spectrum_array(freq_array, spectra_array, tau_array, floor=1e-16, f_min=None, f_max=None):
    """R(tau) = integral S(f) cos(2*pi*f*tau) df for one-sided spectra.

    If f_min/f_max are supplied, only the LES-resolved frequency band is used.
    f_max can be either a scalar or one value per height.
    """
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


def wong_update_variance(inlet_sigma2, target_sigma2, downstream_sigma2, relaxation_factor=0.9, floor=1e-16):
    inlet = np.maximum(np.asarray(inlet_sigma2, dtype=float), floor)
    target = np.maximum(np.asarray(target_sigma2, dtype=float), floor)
    downstream = np.maximum(np.asarray(downstream_sigma2, dtype=float), floor)
    updated = inlet + relaxation_factor * (inlet / downstream) * (target - downstream)
    return np.maximum(updated, floor)


def first_zero_index(
    rho_1d,
    start_index=1,
    zero_tol=0.0,
    persistence_points=1,
    lookahead_points=0,
):
    """Return first strict or practical near-zero crossing index."""
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


def smoothstep(x):
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)



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
    """Stretched exponential forced to rho(0)=1 and rho(tau_zero)=0."""
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
    """Fit the reliable positive part of the raw rho update.

    The near-zero tail is deliberately excluded because it is the part that
    generates high-frequency ripples when transformed to spectral space.
    """
    tau = np.asarray(tau_array, dtype=float)
    rho = np.asarray(rho_raw_1d, dtype=float)
    dt = float(dt)

    finite = np.isfinite(tau) & np.isfinite(rho)
    tau_fit_limit = max(float(raw_zero_tau), float(reference_zero_tau), 8.0 * dt)

    rho_smooth = _moving_average_1d(rho, window=5)
    fit_mask = (
        finite
        & (tau > 0.0)
        & (tau <= tau_fit_limit)
        & (rho > float(rho_min))
        & (rho < float(rho_max))
    )

    # Stop the fit after the first practical decay below rho_min. This prevents
    # fitting late noisy positive humps, such as the z/H≈0.33 behaviour.
    low_ids = np.where(finite & (tau > 0.0) & (rho_smooth < float(rho_min)))[0]
    if len(low_ids) > 0:
        fit_mask[int(low_ids[0]) + 1 :] = False

    ids = np.where(fit_mask)[0]
    if len(ids) < int(min_points):
        fit_mask = (
            finite
            & (tau > 0.0)
            & (tau <= tau_fit_limit)
            & (rho > max(0.02, 0.5 * float(rho_min)))
            & (rho < float(rho_max))
        )
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


def build_exponential_fitted_updated_rho(
    tau_array,
    rho_raw,
    rho_target,
    rho_downstream,
    dt,
    zero_tol=1e-2,
    zero_persistence_points=3,
    zero_lookahead_points=8,
    **fit_kwargs,
):
    """Construct a smooth fitted updated rho and append shifted target tail."""
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

            fit = fit_stretched_exponential_decay(
                tau, raw_1d, raw_zero_tau, reference_zero_tau, dt,
                zero_tol=zero_tol, **fit_kwargs,
            )

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

    return rho_out, diagnostics

def residual_update_autocorrelation_first_zero(
    rho_inlet,
    rho_target,
    rho_downstream,
    relaxation_factor=0.9,
    clip=True,
    post_zero_blend_points=0,
    zero_tol=0.0,
    zero_persistence_points=1,
    zero_lookahead_points=0,
):
    """
    Option-A autocorrelation update.

    First form the direct relaxed residual update:

        rho_raw = rho_inlet + alpha * (rho_target - rho_downstream)

    Then, for each component/height:
      1. Use rho_raw only until its own first zero crossing.
      2. Set the zero-crossing point to exactly zero.
      3. After that crossing, use the target autocorrelation as the controlled tail.

    This makes the updated first-zero point belong to the updated curve itself,
    not the downstream curve. It also prevents post-zero downstream noise from
    contaminating the spectrum reconstruction.
    """
    rho_inlet = np.asarray(rho_inlet, dtype=float)
    rho_target = np.asarray(rho_target, dtype=float)
    rho_downstream = np.asarray(rho_downstream, dtype=float)

    if rho_inlet.shape != rho_target.shape or rho_inlet.shape != rho_downstream.shape:
        raise ValueError("rho_inlet, rho_target, and rho_downstream must have the same shape.")

    rho_raw = rho_inlet + relaxation_factor * (rho_target - rho_downstream)
    if clip:
        rho_raw = np.clip(rho_raw, -1.0, 1.0)
    rho_raw[:, :, 0] = 1.0

    rho_updated = np.empty_like(rho_raw)
    diagnostics = {
        "updated_zero_id": np.zeros(rho_raw.shape[:2], dtype=int),
        "target_zero_id": np.zeros(rho_raw.shape[:2], dtype=int),
        "downstream_zero_id": np.zeros(rho_raw.shape[:2], dtype=int),
        "tail_start_id": np.zeros(rho_raw.shape[:2], dtype=int),
        "blend_end_id": np.zeros(rho_raw.shape[:2], dtype=int),
    }

    n_tau = rho_raw.shape[2]
    for comp_id in range(rho_raw.shape[0]):
        for h_id in range(rho_raw.shape[1]):
            raw_1d = rho_raw[comp_id, h_id, :]
            target_1d = rho_target[comp_id, h_id, :]

            updated_zero_id = first_zero_index(raw_1d, start_index=1, zero_tol=zero_tol, persistence_points=zero_persistence_points, lookahead_points=zero_lookahead_points)
            target_zero_id = first_zero_index(target_1d, start_index=1, zero_tol=zero_tol, persistence_points=zero_persistence_points, lookahead_points=zero_lookahead_points)
            downstream_zero_id = first_zero_index(rho_downstream[comp_id, h_id, :], start_index=1, zero_tol=zero_tol, persistence_points=zero_persistence_points, lookahead_points=zero_lookahead_points)

            updated_1d = target_1d.copy()
            updated_1d[: updated_zero_id + 1] = raw_1d[: updated_zero_id + 1]
            updated_1d[0] = 1.0
            updated_1d[updated_zero_id] = 0.0

            blend_end_id = updated_zero_id
            if post_zero_blend_points and post_zero_blend_points > 0 and updated_zero_id < n_tau - 1:
                blend_end_id = min(n_tau - 1, updated_zero_id + int(post_zero_blend_points))
                ids = np.arange(updated_zero_id, blend_end_id + 1)
                w = smoothstep((ids - updated_zero_id) / max(blend_end_id - updated_zero_id, 1))
                # Blend from zero at updated crossing to target continuation.
                zero_branch = np.zeros_like(ids, dtype=float)
                updated_1d[ids] = (1.0 - w) * zero_branch + w * target_1d[ids]

            if clip:
                updated_1d = np.clip(updated_1d, -1.0, 1.0)
                updated_1d[0] = 1.0

            rho_updated[comp_id, h_id, :] = updated_1d
            diagnostics["updated_zero_id"][comp_id, h_id] = updated_zero_id
            diagnostics["target_zero_id"][comp_id, h_id] = target_zero_id
            diagnostics["downstream_zero_id"][comp_id, h_id] = downstream_zero_id
            diagnostics["tail_start_id"][comp_id, h_id] = updated_zero_id + 1
            diagnostics["blend_end_id"][comp_id, h_id] = blend_end_id

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


def spectrum_from_autocovariance_array(
    tau_array,
    autocovariance_array,
    freq_array,
    floor=1e-16,
    apply_taper=True,
    taper_start_fraction=0.75,
    clip_negative=True,
):
    """
    Inverse cosine transform for one-sided PSD:
        S(f) = 4 * integral_0^inf R(tau) cos(2*pi*f*tau) d tau
    """
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


def reconstruct_rho_from_spectra_resolved_band(freq_array, spectra_array, tau_array, f_min=None, f_max=None, floor=1e-16):
    R = autocovariance_from_spectrum_array(
        freq_array,
        spectra_array,
        tau_array,
        floor=floor,
        f_min=f_min,
        f_max=f_max,
    )
    return normalise_autocovariance_to_rho(R, floor=1e-30)


#%% --------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _selected_height_ids(z_array, body_height, z_min=0.0, z_max_factor=1.5, n_heights=8):
    z = np.asarray(z_array, dtype=float)
    mask = (z >= z_min) & (z <= z_max_factor * body_height)
    ids = np.where(mask)[0]
    if len(ids) == 0:
        return np.array([], dtype=int)
    n = min(n_heights, len(ids))
    return np.unique(np.round(np.linspace(ids[0], ids[-1], n)).astype(int))


def plot_autocorrelation_comparison(
    fig_dir,
    z_array,
    tau_array,
    rho_inlet,
    rho_downstream,
    rho_target,
    rho_updated,
    body_height,
    z_max_factor=1.5,
    n_heights=8,
    components=("u", "v", "w"),
    rho_raw_update=None,
):
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
            if rho_raw_update is not None:
                ax.plot(
                    tau_array,
                    rho_raw_update[comp_id, h_id, :],
                    linestyle="--",
                    linewidth=1.4,
                    alpha=0.8,
                    label="Raw Wong-updated rho",
                )
            ax.plot(tau_array, rho_updated[comp_id, h_id, :], label="Fitted updated rho")
            ax.axhline(0.0, linestyle="--", linewidth=1.0, color="k", alpha=0.5)
            ax.set_xlabel(r"$\tau$ [s]")
            ax.set_ylabel(r"$\rho(\tau)$")
            ax.set_title(f"{comp}-component autocorrelation, z/H = {z_array[h_id] / body_height:.2f}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fname = f"rho_{comp}_zH_{z_array[h_id] / body_height:.2f}.png"
            safe_savefig(fig, os.path.join(comp_dir, fname))
            plt.close(fig)


def plot_autocorrelation_spectra_update(
    fig_dir,
    z_array,
    freq_array,
    inlet_spectra_array,
    target_spectra_array,
    updated_spectra_array,
    body_height,
    z_max_factor=3.0,
    n_heights=10,
    components=("u", "v", "w"),
    floor=1e-16,
):
    safe_makedirs(fig_dir)

    z_mask = z_array <= z_max_factor * body_height
    h_ids_all = np.where(z_mask)[0]
    h_ids = np.unique(
        np.round(np.linspace(h_ids_all[0], h_ids_all[-1], min(n_heights, len(h_ids_all)))).astype(int)
    )

    for comp_id, comp in enumerate(components):
        comp_dir = os.path.join(fig_dir, f"S_{comp}")
        safe_makedirs(comp_dir)   # <-- add this

        for h_id in h_ids:
            fig, ax = plt.subplots(figsize=(9, 6))

            ax.loglog(freq_array, np.maximum(inlet_spectra_array[comp_id, h_id, :], floor), label="Current inlet")
            ax.loglog(freq_array, np.maximum(target_spectra_array[comp_id, h_id, :], floor), label="Target")
            ax.loglog(freq_array, np.maximum(updated_spectra_array[comp_id, h_id, :], floor), label="Autocorr-updated inlet")

            ax.set_xlabel("f [Hz]")
            ax.set_ylabel(fr"$S_{{{comp}{comp}}}(f)$")
            ax.set_title(f"{comp}-component autocorrelation-updated spectrum, z/H = {z_array[h_id] / body_height:.2f}")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()

            fname = f"S_{comp}_zH_{z_array[h_id] / body_height:.2f}.png"
            safe_savefig(fig, os.path.join(comp_dir, fname))
            plt.close(fig)


def plot_autocorrelation_length_profiles(fig_dir, z_array, body_height, L_inlet, L_downstream, L_target, L_updated, components=("u", "v", "w")):
    safe_makedirs(fig_dir)
    y = z_array / body_height
    for comp_id, comp in enumerate(components):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(L_inlet[comp_id, :], y, label="Inlet from spectra")
        ax.plot(L_downstream[comp_id, :], y, label="Downstream from time series")
        ax.plot(L_target[comp_id, :], y, label="Target from spectra")
        ax.plot(L_updated[comp_id, :], y, label="Updated rho")
        ax.set_xlabel(f"L_{comp} [m]")
        ax.set_ylabel("z/H")
        ax.set_title(f"{comp}-component autocorrelation-derived length scale")
        ax.grid(True, alpha=0.3)
        ax.legend()
        safe_savefig(fig, os.path.join(fig_dir, f"L_{comp}.png"))
        plt.close(fig)


#%% --------------------------------------------------------------------------
# Build autocorrelations
# ---------------------------------------------------------------------------

# Keep plot paths short on Windows/OneDrive. The full case_path is already long,
# so avoid deeply nested plot folders here.
fig_dir = os.path.join(
    case_path,
    "log",
    f"it{iteration}",
    "ac",
)
safe_makedirs(fig_dir)

# LES-resolved frequency limits.
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

tau_array = make_tau_array_from_target_lengths(
    target_profile_array,
    time_step,
    factor=TAU_MAX_FACTOR_OF_MAX_TARGET_T,
    tau_min=TAU_MAX_MIN_SECONDS,
    tau_max=TAU_MAX_MAX_SECONDS,
)

print("\nAutocorrelation calibration settings:")
print(f"  n_tau = {len(tau_array)}")
print(f"  tau_max = {tau_array[-1]:.4g} s")
print(f"  dt = {time_step:.4g} s")
print(f"  resolved f_min = {resolved_f_min:.4g} Hz")
if np.ndim(resolved_f_max) == 0:
    print(f"  resolved f_max = {resolved_f_max:.4g} Hz")
else:
    print(f"  resolved f_max min/max = {np.nanmin(resolved_f_max):.4g} / {np.nanmax(resolved_f_max):.4g} Hz")

rho_downstream = autocorrelation_array_from_velocity(vel_array_3d, max_lags=len(tau_array))

R_inlet = autocovariance_from_spectrum_array(
    freq_array,
    inlet_spectra_array,
    tau_array,
    floor=FLOOR,
    f_min=resolved_f_min,
    f_max=resolved_f_max,
)
rho_inlet = normalise_autocovariance_to_rho(R_inlet)

R_target = autocovariance_from_spectrum_array(
    freq_array,
    target_spectra_array,
    tau_array,
    floor=FLOOR,
    f_min=resolved_f_min,
    f_max=resolved_f_max,
)
rho_target = normalise_autocovariance_to_rho(R_target)

# Resolved-band spectral variances for inlet and target.
inlet_sigma2 = integrate_spectra_area(
    freq_array,
    inlet_spectra_array,
    f_min=resolved_f_min,
    f_max=resolved_f_max,
)
target_sigma2 = integrate_spectra_area(
    freq_array,
    target_spectra_array,
    f_min=resolved_f_min,
    f_max=resolved_f_max,
)

# Downstream variance is taken from the time series/profile because this is the
# statistic used by the LES post-processing. The LES signal is already resolved
# by the mesh and time step.
downstream_sigma2 = downstream_profile_array[:, 1:4].T

updated_sigma2 = wong_update_variance(
    inlet_sigma2,
    target_sigma2,
    downstream_sigma2,
    relaxation_factor=VARIANCE_RELAXATION_FACTOR,
    floor=FLOOR,
)

# First form the Wong-style raw autocorrelation residual update. This is used
# only as the quantity to be fitted; the fitted version is what is transformed
# back to spectral space.
rho_raw_update = rho_inlet + AUTOCORR_RELAXATION_FACTOR * (rho_target - rho_downstream)
rho_raw_update = np.clip(rho_raw_update, -1.0, 1.0)
rho_raw_update[:, :, 0] = 1.0

if USE_EXPONENTIAL_RHO_FIT:
    rho_updated, autocorr_update_diagnostics = build_exponential_fitted_updated_rho(
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

    # Compatibility aliases used by the diagnostics section below.
    autocorr_update_diagnostics["updated_zero_id"] = autocorr_update_diagnostics["selected_zero_id"]
    autocorr_update_diagnostics["tail_start_id"] = np.minimum(
        autocorr_update_diagnostics["selected_zero_id"] + 1,
        len(tau_array) - 1,
    )
    autocorr_update_diagnostics["blend_end_id"] = autocorr_update_diagnostics["selected_zero_id"]
    autocorr_update_diagnostics["updated_zero_rho_value"] = np.zeros_like(
        autocorr_update_diagnostics["selected_zero_tau"], dtype=float
    )
    autocorr_update_diagnostics["target_zero_rho_value"] = np.zeros_like(
        autocorr_update_diagnostics["selected_zero_tau"], dtype=float
    )
    autocorr_update_diagnostics["downstream_zero_rho_value"] = np.zeros_like(
        autocorr_update_diagnostics["selected_zero_tau"], dtype=float
    )

    print("\nExponential rho-fit diagnostics:")
    zero_source_values = autocorr_update_diagnostics["zero_source"].ravel()
    n_raw = int(np.sum(zero_source_values == "raw"))
    n_fit = int(np.sum(zero_source_values == "fit"))
    print(f"  selected zero from raw / fit: {n_raw} / {n_fit}")
    print(f"  fitted T min/max: {np.nanmin(autocorr_update_diagnostics['fit_T']):.4g} / {np.nanmax(autocorr_update_diagnostics['fit_T']):.4g} s")
    print(f"  fitted p min/max: {np.nanmin(autocorr_update_diagnostics['fit_p']):.4g} / {np.nanmax(autocorr_update_diagnostics['fit_p']):.4g}")
else:
    rho_updated, autocorr_update_diagnostics = residual_update_autocorrelation_first_zero(
        rho_inlet,
        rho_target,
        rho_downstream,
        relaxation_factor=AUTOCORR_RELAXATION_FACTOR,
        clip=True,
        post_zero_blend_points=AUTOCORR_POST_ZERO_BLEND_POINTS,
        zero_tol=AUTOCORR_ZERO_TOL,
        zero_persistence_points=AUTOCORR_ZERO_PERSISTENCE_POINTS,
        zero_lookahead_points=AUTOCORR_ZERO_LOOKAHEAD_POINTS,
    )

R_updated = updated_sigma2[:, :, np.newaxis] * rho_updated


#%% --------------------------------------------------------------------------
# Diagnostics: length scales from autocorrelation
# ---------------------------------------------------------------------------

L_inlet_from_rho = integral_length_array_from_rho(tau_array, rho_inlet, inlet_profile_array[:, 0])
L_downstream_from_rho = integral_length_array_from_rho(tau_array, rho_downstream, downstream_profile_array[:, 0])
L_target_from_rho = integral_length_array_from_rho(tau_array, rho_target, target_profile_array[:, 0])
L_updated_from_rho = integral_length_array_from_rho(tau_array, rho_updated, new_inlet_profile_array[:, 0])

rows = []
for h_id, z in enumerate(z_array):
    for comp_id, comp in enumerate(COMPONENT_NAMES):
        rows.append(
            {
                "height_id": h_id,
                "z": z,
                "z_over_H": z / building_height,
                "component": comp,
                "sigma2_inlet": inlet_sigma2[comp_id, h_id],
                "sigma2_downstream": downstream_sigma2[comp_id, h_id],
                "sigma2_target": target_sigma2[comp_id, h_id],
                "sigma2_updated": updated_sigma2[comp_id, h_id],
                "L_profile_target": target_profile_array[h_id, 4 + comp_id],
                "L_profile_downstream": downstream_profile_array[h_id, 4 + comp_id],
                "L_rho_inlet": L_inlet_from_rho[comp_id, h_id],
                "L_rho_downstream": L_downstream_from_rho[comp_id, h_id],
                "L_rho_target": L_target_from_rho[comp_id, h_id],
                "L_rho_updated": L_updated_from_rho[comp_id, h_id],
                "updated_zero_tau": tau_array[autocorr_update_diagnostics["updated_zero_id"][comp_id, h_id]],
                "target_zero_tau": tau_array[autocorr_update_diagnostics["target_zero_id"][comp_id, h_id]],
                "downstream_zero_tau": tau_array[autocorr_update_diagnostics["downstream_zero_id"][comp_id, h_id]],
                "tail_start_tau": tau_array[min(autocorr_update_diagnostics["tail_start_id"][comp_id, h_id], len(tau_array) - 1)],
                "rho_fit_T": autocorr_update_diagnostics.get("fit_T", np.full_like(updated_sigma2, np.nan))[comp_id, h_id],
                "rho_fit_p": autocorr_update_diagnostics.get("fit_p", np.full_like(updated_sigma2, np.nan))[comp_id, h_id],
                "rho_fitted_zero_tau": autocorr_update_diagnostics.get("fitted_zero_tau", np.full_like(updated_sigma2, np.nan))[comp_id, h_id],
                "rho_selected_zero_tau": autocorr_update_diagnostics.get("selected_zero_tau", np.full_like(updated_sigma2, np.nan))[comp_id, h_id],
                "rho_zero_source": autocorr_update_diagnostics.get("zero_source", np.full(updated_sigma2.shape, "raw", dtype=object))[comp_id, h_id],
                "rho_fit_n_points": autocorr_update_diagnostics.get("n_fit_points", np.zeros(updated_sigma2.shape, dtype=int))[comp_id, h_id],
                "rho_fit_used_fallback": autocorr_update_diagnostics.get("fit_used_fallback", np.zeros(updated_sigma2.shape, dtype=bool))[comp_id, h_id],
            }
        )

summary_df = pd.DataFrame(rows)
summary_path = os.path.join(fig_dir, "summary.csv")
summary_df.to_csv(summary_path, index=False)


#%% --------------------------------------------------------------------------
# Convert updated autocovariance to spectrum and renormalise variance
# ---------------------------------------------------------------------------

updated_spectra_array, negative_fraction = spectrum_from_autocovariance_array(
    tau_array=tau_array,
    autocovariance_array=R_updated,
    freq_array=freq_array,
    floor=FLOOR,
    apply_taper=APPLY_AUTOCOVARIANCE_TAPER,
    taper_start_fraction=TAPER_START_FRACTION,
    clip_negative=CLIP_NEGATIVE_SPECTRUM_TO_FLOOR,
)

if RENORMALISE_UPDATED_SPECTRUM_AREA:
    updated_spectra_array = renormalise_spectra_to_variance(
        freq_array,
        updated_spectra_array,
        updated_sigma2,
        floor=FLOOR,
        f_min=resolved_f_min,
        f_max=resolved_f_max,
    )

print("\nSpectrum-from-autocorrelation diagnostics:")
print(f"  negative spectrum fraction min/max: {np.nanmin(negative_fraction):.4g} / {np.nanmax(negative_fraction):.4g}")

if APPLY_POWER_LAW_TAIL:
    U_target = target_profile_array[:, 0]
    int_length_scales = target_profile_array[:, -3:].T
    sigmas = np.sqrt(target_profile_array[:, 1:4]).T
    mesh_cutoff_freqs = LES._profileAnalysis.get_mesh_cutoff_frequencies(mesh_size, U_target, int_length_scales, sigmas)
    mesh_cutoff_freqs_3d = np.broadcast_to(mesh_cutoff_freqs[np.newaxis, :], (3, len(mesh_cutoff_freqs)))
    effective_cutoff_freqs_3d = POWER_LAW_TAIL_CUTOFF_FACTOR * mesh_cutoff_freqs_3d
    updated_spectra_array = LES._profileCalibration.apply_power_law_tail(
        freq_array,
        updated_spectra_array,
        effective_cutoff_freqs_3d,
        slope=POWER_LAW_TAIL_SLOPE,
        floor=1e-20,
    )

    # Tail treatment can alter area, so enforce the resolved-band variance again.
    if RENORMALISE_UPDATED_SPECTRUM_AREA:
        updated_spectra_array = renormalise_spectra_to_variance(
            freq_array,
            updated_spectra_array,
            updated_sigma2,
            floor=FLOOR,
            f_min=resolved_f_min,
            f_max=resolved_f_max,
        )

# Final self-consistency diagnostics from the final spectrum over the resolved band.
final_sigma2_resolved = integrate_spectra_area(
    freq_array,
    updated_spectra_array,
    f_min=resolved_f_min,
    f_max=resolved_f_max,
)
rho_final_from_spectrum = reconstruct_rho_from_spectra_resolved_band(
    freq_array,
    updated_spectra_array,
    tau_array,
    f_min=resolved_f_min,
    f_max=resolved_f_max,
    floor=FLOOR,
)
L_final_from_spectrum = integral_length_array_from_rho(
    tau_array,
    rho_final_from_spectrum,
    new_inlet_profile_array[:, 0],
)

# Add final spectrum diagnostics to the CSV after all post-processing.
for row in rows:
    h_id = int(row["height_id"])
    comp_id = COMPONENT_NAMES.index(row["component"])
    row["final_sigma2_resolved"] = final_sigma2_resolved[comp_id, h_id]
    row["final_sigma2_resolved_rel_error"] = (
        final_sigma2_resolved[comp_id, h_id] - updated_sigma2[comp_id, h_id]
    ) / max(updated_sigma2[comp_id, h_id], FLOOR)
    row["L_final_from_spectrum"] = L_final_from_spectrum[comp_id, h_id]
    row["L_final_minus_L_updated_rho"] = (
        L_final_from_spectrum[comp_id, h_id] - L_updated_from_rho[comp_id, h_id]
    )

summary_df = pd.DataFrame(rows)
summary_df.to_csv(summary_path, index=False)

print("\nFinal resolved-band self-consistency diagnostics:")
print(f"  sigma2 rel error min/max: {np.nanmin(summary_df['final_sigma2_resolved_rel_error']):.4g} / {np.nanmax(summary_df['final_sigma2_resolved_rel_error']):.4g}")
print(f"  L_final - L_updated_rho min/max: {np.nanmin(summary_df['L_final_minus_L_updated_rho']):.4g} / {np.nanmax(summary_df['L_final_minus_L_updated_rho']):.4g}")


# Build a downstream diagnostic spectrum from the downstream time-series
# autocorrelation. This is not used for the update itself; it gives the normal
# windLespy iteration log a downstream-spectrum object analogous to the smoothed
# downstream spectrum used by the pure spectral calibration recipe.
downstream_autocovariance_diagnostic = downstream_sigma2[:, :, np.newaxis] * rho_downstream
downstream_spectra_diagnostic_array, downstream_negative_fraction = spectrum_from_autocovariance_array(
    tau_array=tau_array,
    autocovariance_array=downstream_autocovariance_diagnostic,
    freq_array=freq_array,
    floor=FLOOR,
    apply_taper=APPLY_AUTOCOVARIANCE_TAPER,
    taper_start_fraction=TAPER_START_FRACTION,
    clip_negative=CLIP_NEGATIVE_SPECTRUM_TO_FLOOR,
)
downstream_spectra_diagnostic_array = renormalise_spectra_to_variance(
    freq_array,
    downstream_spectra_diagnostic_array,
    downstream_sigma2,
    floor=FLOOR,
    f_min=resolved_f_min,
    f_max=resolved_f_max,
)

print("\nDownstream diagnostic spectrum-from-autocorrelation:")
print(f"  negative spectrum fraction min/max: {np.nanmin(downstream_negative_fraction):.4g} / {np.nanmax(downstream_negative_fraction):.4g}")

#%% --------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

plot_autocorrelation_comparison(
    os.path.join(fig_dir, "rho"),
    z_array,
    tau_array,
    rho_inlet,
    rho_downstream,
    rho_target,
    rho_updated,
    building_height,
    z_max_factor=Z_MAX_FACTOR_PLOTS,
    n_heights=N_HEIGHTS_TO_PLOT,
    components=COMPONENT_NAMES,
    rho_raw_update=rho_raw_update,
)

plot_autocorrelation_spectra_update(
    os.path.join(fig_dir, "spec"),
    z_array,
    freq_array,
    inlet_spectra_array,
    target_spectra_array,
    updated_spectra_array,
    building_height,
    z_max_factor=Z_MAX_FACTOR_FINAL_SPECTRA,
    n_heights=10,
    components=COMPONENT_NAMES,
    floor=FLOOR,
)

plot_autocorrelation_length_profiles(
    os.path.join(fig_dir, "L"),
    z_array,
    building_height,
    L_inlet_from_rho,
    L_downstream_from_rho,
    L_target_from_rho,
    L_updated_from_rho,
    components=COMPONENT_NAMES,
)

LES._plot.plot_spectral_calibration(
    os.path.join(fig_dir, "ov"),
    z_array,
    freq_array,
    downstream_spectra_diagnostic_array,
    updated_spectra_array,
    inlet_spectra_array,
    target_spectra_array,
    cutoff_freqs=None,
    z_min=0.0,
    z_max=Z_MAX_FACTOR_FINAL_SPECTRA * building_height,
)


#%% --------------------------------------------------------------------------
# Optional write-back to case
# ---------------------------------------------------------------------------

if (not converged) and (not stagnated):
    if WRITE_RESULTS:
        dfsr_input_spectra_path = os.path.join(
            case_path,
            "constant",
            "boundaryData",
            "windProfile",
            "spectraProfile",
        )
        LES._caseFiles.write_spectra_profile(
            updated_spectra_array,
            z_array,
            dfsr_input_spectra_path,
            clip_min=SPECTRUM_FLOOR_FOR_WRITE,
        )
        LES._caseFiles.write_new_dfsr_inlet_profile(
            new_inlet_profile_array,
            target_profile_df,
            case_path,
        )

    if WRITE_ITER_SPECTRA:
        LES._caseFiles.write_dfsr_iter_spectra(
            case_path,
            iter_status,
            z_array,
            freq_array,
            inlet_spectra_array,
            downstream_spectra_diagnostic_array,
            inlet_or_downstream="downstream",
            new_inlet_spectra_array=updated_spectra_array,
            cutoff_freqs=None,
            clip_min=SPECTRUM_FLOOR_FOR_WRITE,
        )
else:
    if WRITE_ITER_SPECTRA:
        LES._caseFiles.write_dfsr_iter_spectra(
            case_path,
            iter_status,
            z_array,
            freq_array,
            inlet_spectra_array,
            downstream_spectra_diagnostic_array,
            inlet_or_downstream="downstream",
            new_inlet_spectra_array=None,
            cutoff_freqs=None,
            clip_min=SPECTRUM_FLOOR_FOR_WRITE,
        )


#%% --------------------------------------------------------------------------
# Exit behaviour
# ---------------------------------------------------------------------------

if converged:
    sys.exit(0)
elif stagnated:
    sys.exit(0)
else:
    sys.exit(1)
