# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 11:40:10 2026

@author: David Cunningham
"""

import logging
import numpy as np
import pandas as pd
import os
import sys
import shutil

from scipy.signal.windows import dpss
from scipy.fft import rfft, rfftfreq
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
from scipy.optimize import root_scalar

cwd = os.path.dirname(os.path.abspath(__file__))
windlespy_path = os.path.abspath(os.path.join(cwd, "..", ".."))
sys.path.append(windlespy_path)
import windlespy as LES
sys.path.remove(windlespy_path)

#%%

def get_freq_array(fMax, nFreq):
    freq_array = np.linspace(fMax/nFreq,fMax,nFreq)
    
    return freq_array

#%%

def get_vk_spectra_array(fMax, nFreq, profile_array):
    
    freq_array = get_freq_array(fMax, nFreq)
    
    int_length_scales = profile_array[:,-3:]
    
    U = profile_array[:,0]
    
    red_fs_all = LES._profileAnalysis.get_von_karman_red_fs(freq_array, int_length_scales, U)
    
    spectra_array = np.zeros((3,np.shape(profile_array)[0], nFreq))

    for height_id in range(np.shape(profile_array)[0]):
        
        for vel_comp_id,vel_comp in enumerate(["u","v","w"]):
            
            red_fs = red_fs_all[height_id, vel_comp_id, :]
            
            sigma_2 = profile_array[height_id, 1+vel_comp_id]
            
            von_karman_spectrum = LES._profileAnalysis.von_karman_spectra(red_fs, vel_comp)
            
            S_n = (von_karman_spectrum*sigma_2)/freq_array
            
            spectra_array[vel_comp_id, height_id,:] = S_n

    return spectra_array

#%%

def get_welch_nperseg_from_dfsr_grid(time_step, fMax, nFreq):
    """
    Choose nperseg so that Welch frequency spacing matches the DFSR grid:

        delta_f_welch = fs / nperseg
        delta_f_dfsr  = fMax / nFreq

    with fs = 1 / time_step.
    """
    fs = 1.0 / time_step
    delta_f = fMax / nFreq

    nperseg = int(round(fs / delta_f))

    # prefer even nperseg so one-sided Welch gives:
    # [0, delta_f, 2*delta_f, ..., fMax]
    if nperseg % 2 != 0:
        nperseg += 1

    return nperseg

#%%

def get_downstream_spectra_array(
    fMax,
    nFreq,
    vel_array_3d,
    time_step,
    body_dim,
    downstream_profile_array,
    inlet_or_downstream="inlet",
    burn_in_time=None,
    time_steps=None,
    method="multitaper",
    psd_fit_min=0.5,
    psd_fit_max=None,
    psd_band_config=None,
    min_points_per_bin=4,
    psd_left_mode="band_median",
    psd_right_mode="slope",
    psd_right_slope_clip=(-8.0, 0.0),
    psd_low_plateau_band=(0.20, 1.00),
    psd_smooth_knots=True,
    psd_knot_smooth_kernel=(1, 2, 1),
    time_bandwidth=3.5,
    num_tapers=None,
    return_raw=False,
    floor=1e-20,
):
    """
    Estimate downstream spectra on the DFSR frequency grid.

    method='multitaper' returns median-binned log-log spline fits evaluated at nFreq
    DFSR frequencies. Set return_raw=True to also retrieve raw multitaper and bin data.
    """
    
    if psd_band_config is None:
        psd_band_config = [
            (1.0, 2.0, 3),
            (2.0, 5.0, 4),
            (5.0, 10.0, 4),
            (10.0, None, 10),
        ]
    
    if inlet_or_downstream == "downstream":
        if burn_in_time is not None and time_steps is not None:
            mask = np.asarray(time_steps) > burn_in_time
            vel_array_3d = vel_array_3d[:, mask, :]

    fs = 1.0 / time_step
    downstream_spectra_array = np.zeros((3, np.shape(vel_array_3d)[2], nFreq))
    raw_psd = [[None for _ in range(np.shape(vel_array_3d)[2])] for _ in range(3)]
    binned_psd = [[None for _ in range(np.shape(vel_array_3d)[2])] for _ in range(3)]

    for height_id in range(np.shape(vel_array_3d)[2]):
        for vel_comp_id in range(3):
            vel_time_series = vel_array_3d[vel_comp_id, :, height_id]
            if method == "multitaper":
                raw_f, raw_S = _multitaper_psd_1d(
                    vel_time_series,
                    fs,
                    time_bandwidth=time_bandwidth,
                    num_tapers=num_tapers,
                )
                
                target_f = get_freq_array(fMax, nFreq)
                
                raw_on_grid = _interp_loglog(
                    raw_f[(raw_f > 0.0) & (raw_f <= fMax)],
                    raw_S[(raw_f > 0.0) & (raw_f <= fMax)],
                    target_f,
                    floor=floor,
                )
                
                S_fit, knot_dict, _ = smooth_log_spectrum_1d_binned_pchip(
                    target_f,
                    raw_on_grid,
                    band_config=psd_band_config,
                    floor=floor,
                    f_fit_min=psd_fit_min,
                    f_fit_max=psd_fit_max,
                    min_points_per_bin=min_points_per_bin,
                    left_mode=psd_left_mode,
                    right_mode=psd_right_mode,
                    right_slope_clip=psd_right_slope_clip,
                    low_plateau_band=psd_low_plateau_band,
                    smooth_knots=psd_smooth_knots,
                    knot_smooth_kernel=psd_knot_smooth_kernel,
                )
                
                downstream_spectra_array[vel_comp_id, height_id, :] = S_fit
                raw_psd[vel_comp_id][height_id] = (target_f, raw_on_grid)
                
                if knot_dict is None:
                    binned_psd[vel_comp_id][height_id] = (np.array([]), np.array([]))
                else:
                    binned_psd[vel_comp_id][height_id] = (
                        knot_dict["f_knots"],
                        knot_dict["S_knots"],
                    )

    if return_raw:
        return downstream_spectra_array, raw_psd, binned_psd
    return downstream_spectra_array

#%%

def smooth_downstream_spectra_array(
    freq_array,
    downstream_spectra_array,
    window_length=41,
    polyorder=2,
    floor=1e-16,
    n_log_points=None,
):
    """
    Smooth downstream PSDs in log(S)-log(f) space.

    Parameters
    ----------
    freq_array : (nFreq,) array
        Positive frequency array.
    downstream_spectra_array : (3, nHeights, nFreq) array
    window_length : int
        Savitzky-Golay window length on the uniform log-frequency grid.
    polyorder : int
        Savitzky-Golay polynomial order.
    floor : float
        Minimum positive PSD value.
    n_log_points : int or None
        Number of points in uniform log-frequency grid.
        If None, use len(freq_array).

    Returns
    -------
    smoothed_downstream_spectra_array : array
        Same shape as input.
    """

    f = np.asarray(freq_array, dtype=float)

    if np.any(f <= 0.0):
        raise ValueError("freq_array must contain only positive frequencies.")

    if n_log_points is None:
        n_log_points = len(f)

    logf = np.log10(f)
    logf_uniform = np.linspace(logf.min(), logf.max(), n_log_points)

    # Ensure valid SG parameters
    if window_length >= n_log_points:
        window_length = n_log_points - 1
    if window_length % 2 == 0:
        window_length -= 1
    if window_length <= polyorder:
        window_length = polyorder + 3
        if window_length % 2 == 0:
            window_length += 1

    smoothed_downstream_spectra_array = downstream_spectra_array.copy()

    for height_id in range(downstream_spectra_array.shape[1]):
        for vel_comp_id in range(3):
            S = np.asarray(
                smoothed_downstream_spectra_array[vel_comp_id, height_id, :],
                dtype=float
            )
            S = np.maximum(S, floor)

            logS = np.log10(S)

            # interpolate logS onto uniform logf grid
            interp_to_log_grid = interp1d(
                logf,
                logS,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate"
            )
            logS_uniform = interp_to_log_grid(logf_uniform)

            # smooth on uniform log-frequency grid
            logS_uniform_smooth = savgol_filter(
                logS_uniform,
                window_length=window_length,
                polyorder=polyorder,
                mode="interp"
            )

            # interpolate back to original grid
            interp_back = interp1d(
                logf_uniform,
                logS_uniform_smooth,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate"
            )
            logS_smooth = interp_back(logf)

            smoothed_downstream_spectra_array[vel_comp_id, height_id, :] = 10.0**logS_smooth

    return smoothed_downstream_spectra_array

#%%

def smooth_spectral_ratio_array(freq_array, ratio_array, window_length=41, polyorder=2, floor=1e-16, n_log_points=None):

    f = np.asarray(freq_array, dtype=float)
    R = np.maximum(np.asarray(ratio_array, dtype=float), floor).copy()
    if n_log_points is None:
        n_log_points = len(f)
    logf = np.log10(f)
    logf_u = np.linspace(logf.min(), logf.max(), n_log_points)
    win = _valid_savgol_window(n_log_points, window_length, polyorder)
    if win is None:
        return R

    for comp in range(R.shape[0]):
        for h in range(R.shape[1]):
            interp_to = interp1d(logf, np.log10(R[comp, h]), bounds_error=False, fill_value="extrapolate")
            y = interp_to(logf_u)
            y = savgol_filter(y, win, polyorder, mode="interp")
            interp_back = interp1d(logf_u, y, bounds_error=False, fill_value="extrapolate")
            R[comp, h] = 10.0 ** interp_back(logf)
    return np.maximum(R, floor)
#%%

def get_downstream_cutoff_freq(
    freq_array,
    target_spectrum,
    downstream_spectrum,
    mesh_cutoff_freq=None,
    expected_slope=-5/3,
    inertial_tol=0.35,
    slope_excess=0.35,
    relative_slope_excess=0.25,
    sustain=6,
    floor=1e-20
):
    def _first_sustained_true(mask, start_idx=0, sustain=6):
        mask = np.asarray(mask, dtype=bool)

        if sustain <= 1:
            idx = np.where(mask[start_idx:])[0]
            return None if len(idx) == 0 else start_idx + idx[0]

        last_start = len(mask) - sustain
        if last_start < start_idx:
            return None

        for i in range(start_idx, last_start + 1):
            if np.all(mask[i:i+sustain]):
                return i

        return None

    f = np.asarray(freq_array, dtype=float)
    S_t = np.asarray(target_spectrum, dtype=float)
    S_d = np.asarray(downstream_spectrum, dtype=float)

    valid = np.isfinite(f) & np.isfinite(S_t) & np.isfinite(S_d) & (f > 0.0)
    f = f[valid]
    S_t = np.maximum(S_t[valid], floor)
    S_d = np.maximum(S_d[valid], floor)

    if len(f) == 0:
        raise ValueError("No valid frequency/spectral values after filtering.")
    if len(f) < sustain:
        raise ValueError("Spectrum too short for chosen sustain value.")

    logf = np.log(f)
    logS_t = np.log(S_t)
    logS_d = np.log(S_d)

    target_slope = np.gradient(logS_t, logf)
    downstream_slope = np.gradient(logS_d, logf)

    peak_idx = int(np.argmax(S_t))

    inertial_like = np.abs(target_slope - expected_slope) <= inertial_tol
    inertial_idx = _first_sustained_true(
        inertial_like,
        start_idx=peak_idx,
        sustain=max(3, sustain // 2)
    )
    search_start = inertial_idx if inertial_idx is not None else peak_idx

    slope_mask = (
        (downstream_slope < (expected_slope - slope_excess))
        & (downstream_slope < (target_slope - relative_slope_excess))
    )

    slope_idx = _first_sustained_true(
        slope_mask,
        start_idx=search_start,
        sustain=sustain
    )

    spectral_cutoff = f[slope_idx] if slope_idx is not None else f[-1]

    effective_cutoff = spectral_cutoff
    if mesh_cutoff_freq is not None:
        effective_cutoff = min(mesh_cutoff_freq, spectral_cutoff)

    return {
        "effective_cutoff": effective_cutoff,
        "spectral_cutoff": spectral_cutoff,
        "mesh_cutoff": mesh_cutoff_freq,
        "search_start_freq": f[search_start],
        "freq": f,
        "target_slope": target_slope,
        "downstream_slope": downstream_slope,
        "slope_mask": slope_mask,
    }


#%%

def get_all_cutoff_frequencies(
    freq_array,
    target_spectra_array,
    downstream_spectra_array,
    mesh_size,
    target_profile_array,
    expected_slope=-5/3,
    inertial_tol=0.35,
    slope_excess=0.35,
    relative_slope_excess=0.25,
    sustain=6,
    floor=1e-20,
):
    """
    Determine cutoff frequencies for all components and heights
    using:
        1) slope-based spectral cutoff
        2) mesh cutoff
    with the effective cutoff taken as the minimum of the two.
    """

    target_spectra_array = np.asarray(target_spectra_array, dtype=float)
    downstream_spectra_array = np.asarray(downstream_spectra_array, dtype=float)
    target_profile_array = np.asarray(target_profile_array, dtype=float)

    if target_spectra_array.shape != downstream_spectra_array.shape:
        raise ValueError(
            "target_spectra_array and downstream_spectra_array must have the same shape."
        )

    if target_spectra_array.ndim != 3 or target_spectra_array.shape[0] != 3:
        raise ValueError(
            "Expected spectral arrays of shape (3, nHeights, nFreq)."
        )

    n_comp, n_heights, _ = target_spectra_array.shape

    # Target-profile quantities for mesh cutoff calculation
    U = target_profile_array[:, 0]                                  # (nHeights,)
    int_length_scales = target_profile_array[:, -3:].T              # (3, nHeights)
    sigmas = np.sqrt(target_profile_array[:, 1:4]).T  # (3, nHeights)

    mesh_cutoff_freqs = LES._profileAnalysis.get_mesh_cutoff_frequencies(
        mesh_size,
        U,
        int_length_scales,
        sigmas
    )

    effective_cutoffs = np.full((n_comp, n_heights), np.nan)
    spectral_cutoffs = np.full((n_comp, n_heights), np.nan)

    diagnostics = [[None for _ in range(n_heights)] for _ in range(n_comp)]

    for comp in range(n_comp):
        for h in range(n_heights):
            mesh_fc = mesh_cutoff_freqs[h]

            res = get_downstream_cutoff_freq(
                freq_array=freq_array,
                target_spectrum=target_spectra_array[comp, h, :],
                downstream_spectrum=downstream_spectra_array[comp, h, :],
                mesh_cutoff_freq=mesh_fc,
                expected_slope=expected_slope,
                inertial_tol=inertial_tol,
                slope_excess=slope_excess,
                relative_slope_excess=relative_slope_excess,
                sustain=sustain,
                floor=floor,
            )

            effective_cutoffs[comp, h] = res["effective_cutoff"]
            spectral_cutoffs[comp, h] = res["spectral_cutoff"]
            diagnostics[comp][h] = res

    return {
        "effective_cutoffs": effective_cutoffs,   # (3, nHeights)
        "spectral_cutoffs": spectral_cutoffs,     # (3, nHeights)
        "mesh_cutoffs": mesh_cutoff_freqs,        # (nHeights,)
        "diagnostics": diagnostics,
    }
    
#%%
def apply_power_law_tail(freq_array, spectra_array, mesh_cutoff_freqs_3d, slope=-5/3, floor=1e-20):
    f = np.asarray(freq_array, dtype=float)
    S = np.maximum(np.asarray(spectra_array, dtype=float).copy(), floor)
    if np.asarray(mesh_cutoff_freqs_3d).ndim == 1:
        fc_array = np.broadcast_to(np.asarray(mesh_cutoff_freqs_3d)[np.newaxis, :], (S.shape[0], S.shape[1]))
    else:
        fc_array = np.asarray(mesh_cutoff_freqs_3d, dtype=float)
    logf = np.log(np.maximum(f, floor))
    for comp in range(S.shape[0]):
        for h in range(S.shape[1]):
            fc = float(np.clip(fc_array[comp, h], f[0], f[-1]))
            spec = np.maximum(S[comp, h], floor)
            S_fc = np.exp(np.interp(np.log(fc), logf, np.log(spec)))
            mask = f > fc
            S[comp, h, mask] = np.maximum(S_fc * (f[mask] / fc) ** slope, floor)
    return S
#%%

def get_convective_spectral_function(inlet_spectra_array, downstream_spectra_array, floor=1e-16):
    inlet_spectra_array = np.maximum(inlet_spectra_array, floor)
    downstream_spectra_array = np.maximum(downstream_spectra_array, floor)
    convective_spectral_function = inlet_spectra_array / downstream_spectra_array
    return convective_spectral_function

#%%

def get_updated_spectra_array(target_spectra_array, conv_spectral_func, floor=1e-16):
    updated_spectra_array = target_spectra_array * conv_spectral_func
    updated_spectra_array = np.maximum(updated_spectra_array, floor)
    return updated_spectra_array

#%%

def update_mean_profile_only(current_inlet_profile_array, target_profile_array, downstream_profile_array, relaxation_factor=0.9):
    """
    Update only the mean wind speed column U using a Wong-style correction.
    
    Arrays are assumed to have columns:
    [U, R11, R22, R33, Lu, Lv, Lw]
    """
    adaptive_relaxation_factor = relaxation_factor * (current_inlet_profile_array[:,0] / downstream_profile_array[:,0])
    
    conditions = [adaptive_relaxation_factor < 0.5,
                  (adaptive_relaxation_factor >= 0.5) & (adaptive_relaxation_factor <= 5),
                  adaptive_relaxation_factor > 5]
    choices = [0.5, adaptive_relaxation_factor, 5]
     
    adaptive_relaxation_factor = np.select(conditions, choices)
    
    new_mean_profile = current_inlet_profile_array[:,0] + adaptive_relaxation_factor * (target_profile_array[:,0] - downstream_profile_array[:,0])
    
    new_inlet_profile = current_inlet_profile_array.copy()
    
    new_inlet_profile[:,0] = new_mean_profile
    
    new_inlet_profile[:, 0] = np.clip(new_inlet_profile[:, 0], 0.01, None)    # U > 0
    
    return new_inlet_profile

#%%

def read_spectra_profile_file(case_path,filename):
    """
    Read a DFSR spectraProfile-style file.

    Format:
        first line: nHeights nFreq
        each following row:
            z Su[0:nFreq] Sv[0:nFreq] Sw[0:nFreq]

    Returns
    -------
    spectra_array : (3, nHeights, nFreq) ndarray
    """
    filepath = os.path.join(case_path, "constant","boundaryData", "windProfile", filename)

    with open(filepath, "r") as f:
        header = f.readline().split()

    if len(header) != 2:
        raise ValueError(f"Invalid spectra profile header in {filepath}")

    n_heights = int(header[0])
    n_freq = int(header[1])

    data = np.loadtxt(filepath, skiprows=1)

    if data.ndim == 1:
        data = data[np.newaxis, :]

    expected_cols = 1 + 3 * n_freq
    if data.shape[0] != n_heights:
        raise ValueError(
            f"Expected {n_heights} rows in {filepath}, found {data.shape[0]}"
        )
    if data.shape[1] != expected_cols:
        raise ValueError(
            f"Expected {expected_cols} columns in {filepath}, found {data.shape[1]}"
        )

    z_array = data[:, 0]

    spectra_array = np.zeros((3, n_heights, n_freq), dtype=float)
    spectra_array[0, :, :] = data[:, 1 : 1 + n_freq]
    spectra_array[1, :, :] = data[:, 1 + n_freq : 1 + 2 * n_freq]
    spectra_array[2, :, :] = data[:, 1 + 2 * n_freq : 1 + 3 * n_freq]

    return spectra_array

#%%

def _valid_savgol_window(n, window_length, polyorder):
    window_length = int(window_length)
    if n <= polyorder + 2:
        return None
    if window_length > n:
        window_length = n if n % 2 == 1 else n - 1
    if window_length % 2 == 0:
        window_length -= 1
    if window_length <= polyorder:
        window_length = polyorder + 3
        if window_length % 2 == 0:
            window_length += 1
    if window_length > n:
        return None
    return max(window_length, 3)

#%%

def _multitaper_psd_1d(x, fs, time_bandwidth=3.5, num_tapers=None, detrend=True):

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        raise ValueError("Time series too short for multitaper PSD.")
    if detrend:
        x = x - np.mean(x)

    n = x.size
    if num_tapers is None:
        num_tapers = max(1, int(2 * time_bandwidth) - 1)

    tapers, eigvals = dpss(n, time_bandwidth, Kmax=num_tapers, return_ratios=True)
    spectra = []
    for taper in tapers:
        xw = x * taper
        fft_vals = rfft(xw)
        scale = fs * np.sum(taper ** 2)
        S = (np.abs(fft_vals) ** 2) / scale
        if n % 2 == 0:
            S[1:-1] *= 2.0
        else:
            S[1:] *= 2.0
        spectra.append(S)
    S_mt = np.average(np.vstack(spectra), axis=0, weights=eigvals)
    f = rfftfreq(n, d=1.0 / fs)
    return f, S_mt

#%%
def _smooth_1d_with_kernel(values, kernel_weights, pad_mode="reflect"):
    v = np.asarray(values, dtype=float)
    kernel = np.asarray(kernel_weights, dtype=float)
    kernel = kernel / np.sum(kernel)
    pad = len(kernel) // 2
    padded = np.pad(v, pad_width=pad, mode=pad_mode)
    return np.convolve(padded, kernel, mode="valid")

#%%

def _merge_duplicate_x(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    xu = []
    yu = []

    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and np.isclose(x[j], x[i]):
            j += 1
        xu.append(np.mean(x[i:j]))
        yu.append(np.mean(y[i:j]))
        i = j

    return np.asarray(xu), np.asarray(yu)

#%%

def _interp_loglog(x_old, y_old, x_new, floor=1e-16):
    x_old = np.asarray(x_old, dtype=float)
    y_old = np.asarray(y_old, dtype=float)
    x_new = np.asarray(x_new, dtype=float)

    mask = (
        np.isfinite(x_old) & np.isfinite(y_old)
        & (x_old > 0.0) & (y_old > 0.0)
    )

    if np.count_nonzero(mask) < 2:
        return np.full_like(x_new, floor, dtype=float)

    x_old = x_old[mask]
    y_old = np.maximum(y_old[mask], floor)

    order = np.argsort(x_old)
    x_old = x_old[order]
    y_old = y_old[order]

    logx_old = np.log10(x_old)
    logy_old = np.log10(y_old)
    logx_new = np.log10(x_new)

    interp = interp1d(
        logx_old,
        logy_old,
        kind="linear",
        bounds_error=False,
        fill_value=(logy_old[0], logy_old[-1]),
        assume_sorted=True,
    )

    return np.maximum(10.0 ** interp(logx_new), floor)
#%% --------------------------------------------------------------------------
# Robust PSD smoothing: piecewise log-binned median + PCHIP
# ---------------------------------------------------------------------------
def get_low_frequency_plateau_value(
    freq_array,
    spectrum_1d,
    plateau_band=(0.2, 1.0),
    floor=1e-16,
):
    """
    Estimate a low-frequency plateau from the raw spectrum using the log-median
    over a specified low-frequency band.
    """
    f = np.asarray(freq_array, dtype=float)
    S = np.maximum(np.asarray(spectrum_1d, dtype=float), floor)

    fmin, fmax = plateau_band
    mask = (
        np.isfinite(f) & np.isfinite(S) &
        (f >= fmin) & (f <= fmax) &
        (f > 0.0) & (S > 0.0)
    )

    if np.count_nonzero(mask) < 2:
        return None

    return 10.0**np.median(np.log10(S[mask]))


def get_log_binned_median_knots(
    freq_array,
    spectrum_1d,
    band_config,
    floor=1e-16,
    f_fit_min=1.0,
    f_fit_max=None,
    min_points_per_bin=6,
    smooth_knots=True,
    knot_smooth_kernel=(1, 2, 1),
):
    """
    Build robust knots from piecewise log-frequency bins.

    Each band in band_config has format:
        (f_low, f_high, n_bins)

    x-knot = median(log10(f)) in the bin
    y-knot = median(log10(S)) in the bin
    """
    f = np.asarray(freq_array, dtype=float)
    S = np.maximum(np.asarray(spectrum_1d, dtype=float), floor)

    global_mask = np.isfinite(f) & np.isfinite(S) & (f > 0.0) & (S > 0.0)

    if f_fit_min is not None:
        global_mask &= (f >= f_fit_min)

    if f_fit_max is not None:
        global_mask &= (f <= f_fit_max)

    f_fit = f[global_mask]
    S_fit = S[global_mask]

    if len(f_fit) < max(8, 2 * min_points_per_bin):
        return None, None

    xk_all = []
    yk_all = []

    for band in band_config:
        f_lo, f_hi, n_bins_band = band

        band_mask = (f_fit >= f_lo)
        if f_hi is not None:
            band_mask &= (f_fit <= f_hi)

        f_band = f_fit[band_mask]
        S_band = S_fit[band_mask]

        if len(f_band) < max(4, min_points_per_bin):
            continue

        logf = np.log10(f_band)
        logS = np.log10(S_band)

        if np.isclose(logf.min(), logf.max()):
            continue

        edges = np.linspace(logf.min(), logf.max(), n_bins_band + 1)

        for i in range(n_bins_band):
            if i < n_bins_band - 1:
                m = (logf >= edges[i]) & (logf < edges[i + 1])
            else:
                m = (logf >= edges[i]) & (logf <= edges[i + 1])

            if np.count_nonzero(m) < min_points_per_bin:
                continue

            xk_all.append(np.median(logf[m]))
            yk_all.append(np.median(logS[m]))

    if len(xk_all) < 4:
        return None, None

    xk = np.asarray(xk_all, dtype=float)
    yk = np.asarray(yk_all, dtype=float)

    order = np.argsort(xk)
    xk = xk[order]
    yk = yk[order]

    xk, yk = _merge_duplicate_x(xk, yk)

    if len(xk) < 4:
        return None, None

    if smooth_knots and len(yk) >= 3:
        yk = _smooth_1d_with_kernel(yk, knot_smooth_kernel, pad_mode="reflect")

    return 10.0**xk, 10.0**yk


def smooth_log_spectrum_1d_binned_pchip(
    freq_array,
    spectrum_1d,
    band_config,
    floor=1e-16,
    f_fit_min=1.0,
    f_fit_max=None,
    min_points_per_bin=6,
    left_mode="band_median",
    right_mode="slope",
    right_slope_clip=(-8.0, 0.0),
    low_plateau_band=(0.2, 1.0),
    smooth_knots=True,
    knot_smooth_kernel=(1, 2, 1),
):
    """
    Smooth a 1D spectrum by:
    1) building robust knots from piecewise log-binned medians
    2) fitting log10(S) vs log10(f) with a PCHIP interpolant
    3) extending below the first reliable knot using a raw-band plateau if requested
    """
    f = np.asarray(freq_array, dtype=float)
    S_raw = np.maximum(np.asarray(spectrum_1d, dtype=float), floor)

    f_knots, S_knots = get_log_binned_median_knots(
        f,
        S_raw,
        band_config=band_config,
        floor=floor,
        f_fit_min=f_fit_min,
        f_fit_max=f_fit_max,
        min_points_per_bin=min_points_per_bin,
        smooth_knots=smooth_knots,
        knot_smooth_kernel=knot_smooth_kernel,
    )

    if f_knots is None:
        return S_raw.copy(), None, None

    x = np.log10(f)
    xk = np.log10(f_knots)
    yk = np.log10(np.maximum(S_knots, floor))

    pchip = PchipInterpolator(xk, yk, extrapolate=False)

    y = np.empty_like(x)

    mid = (x >= xk[0]) & (x <= xk[-1])
    y[mid] = pchip(x[mid])

    left = x < xk[0]
    if np.any(left):
        if left_mode == "constant":
            y[left] = yk[0]
        elif left_mode == "band_median":
            plateau_val = get_low_frequency_plateau_value(
                f,
                S_raw,
                plateau_band=low_plateau_band,
                floor=floor,
            )
            if plateau_val is None:
                y[left] = yk[0]
            else:
                y[left] = np.log10(np.maximum(plateau_val, floor))
        else:
            if len(xk) < 2:
                y[left] = yk[0]
            else:
                left_slope = (yk[1] - yk[0]) / (xk[1] - xk[0])
                left_slope = np.clip(left_slope, -2.0, 2.0)
                y[left] = yk[0] + left_slope * (x[left] - xk[0])

    right = x > xk[-1]
    if np.any(right):
        if right_mode == "constant" or len(xk) < 2:
            y[right] = yk[-1]
        else:
            right_slope = (yk[-1] - yk[-2]) / (xk[-1] - xk[-2])
            right_slope = np.clip(right_slope, right_slope_clip[0], right_slope_clip[1])
            y[right] = yk[-1] + right_slope * (x[right] - xk[-1])

    S_smooth = np.maximum(10.0**y, floor)

    knot_dict = {
        "f_knots": f_knots,
        "S_knots": np.maximum(S_knots, floor),
    }

    return S_smooth, knot_dict, pchip


def smooth_spectra_array_log_binned_median(
    freq_array,
    spectra_array,
    band_config,
    floor=1e-16,
    f_fit_min=1.0,
    f_fit_max=None,
    min_points_per_bin=6,
    left_mode="band_median",
    right_mode="slope",
    right_slope_clip=(-8.0, 0.0),
    low_plateau_band=(0.2, 1.0),
    smooth_knots=True,
    knot_smooth_kernel=(1, 2, 1),
):
    spectra = np.asarray(spectra_array, dtype=float)
    out = np.zeros_like(spectra)

    for comp_id in range(spectra.shape[0]):
        for z_id in range(spectra.shape[1]):
            out[comp_id, z_id, :], _, _ = smooth_log_spectrum_1d_binned_pchip(
                freq_array,
                spectra[comp_id, z_id, :],
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

    return np.maximum(out, floor)


def smooth_spectra_array_height_kernel(
    z_array,
    spectra_array,
    kernel_weights,
    floor=1e-16,
    pad_mode="reflect",
):
    """
    Smooth spectra in height at each frequency using a user-defined kernel.
    Applied to log(S).
    """
    z = np.asarray(z_array, dtype=float)
    spectra = np.asarray(spectra_array, dtype=float)

    if spectra.ndim != 3 or spectra.shape[0] != 3:
        raise ValueError("spectra_array must have shape (3, nHeights, nFreq).")

    if spectra.shape[1] != len(z):
        raise ValueError("Height dimension of spectra_array must match len(z_array).")

    kernel = np.asarray(kernel_weights, dtype=float)
    if kernel.ndim != 1 or len(kernel) < 1:
        raise ValueError("kernel_weights must be a 1D array-like with at least one value.")
    kernel = kernel / np.sum(kernel)

    pad = len(kernel) // 2
    out = spectra.copy()

    for comp_id in range(spectra.shape[0]):
        for f_id in range(spectra.shape[2]):
            Sz = np.maximum(spectra[comp_id, :, f_id], floor)
            logSz = np.log10(Sz)

            padded = np.pad(logSz, pad_width=pad, mode=pad_mode)
            logSz_smooth = np.convolve(padded, kernel, mode="valid")

            out[comp_id, :, f_id] = 10.0**logSz_smooth

    return np.maximum(out, floor)

#%%

def smooth_spectra_over_height(z_array, spectra_array, window_length=3, polyorder=2, floor=1e-20):
    """Gently smooth log(PSD) vertically at each frequency."""
    from scipy.signal import savgol_filter

    z_array = np.asarray(z_array, dtype=float)
    S = np.maximum(np.asarray(spectra_array, dtype=float), floor).copy()
    if S.ndim != 3 or S.shape[1] != len(z_array):
        raise ValueError("spectra_array must have shape (3, nHeights, nFreq).")

    n_heights = S.shape[1]
    win = _valid_savgol_window(n_heights, window_length, polyorder)
    if win is None:
        return S

    logS = np.log10(S)
    for comp in range(S.shape[0]):
        for fi in range(S.shape[2]):
            y = logS[comp, :, fi]
            finite = np.isfinite(y)
            if np.count_nonzero(finite) >= win:
                y_filled = np.interp(z_array, z_array[finite], y[finite])
                logS[comp, :, fi] = savgol_filter(y_filled, win, polyorder, mode="interp")
    return np.maximum(10.0 ** logS, floor)

#%%
def get_inverse_transfer_function(inlet_spectra_array, downstream_spectra_array, floor=1e-16):
    return np.maximum(inlet_spectra_array, floor) / np.maximum(downstream_spectra_array, floor)

#%%
def get_updated_spectra_array_wong(
    inlet_spectra_array,
    target_spectra_array,
    downstream_spectra_array,
    inverse_transfer_function=None,
    relaxation_factor=0.9,
    gain_min=None,
    gain_max=None,
    floor=1e-16,
):
    """
    Wong-style residual update for direct spectral calibration.

    S_new = S_inlet + alpha * (S_inlet / S_downstream) * (S_target - S_downstream)

    No gain clipping is applied. The gain_min and gain_max arguments are kept
    only for backwards compatibility with older recipe calls and are ignored.
    If inverse_transfer_function is supplied, it is used directly without
    clipping; otherwise S_inlet/S_downstream is computed internally.
    """
    inlet = np.maximum(np.asarray(inlet_spectra_array, dtype=float), floor)
    target = np.maximum(np.asarray(target_spectra_array, dtype=float), floor)
    downstream = np.maximum(np.asarray(downstream_spectra_array, dtype=float), floor)

    if inverse_transfer_function is None:
        inverse_transfer = inlet / downstream
    else:
        inverse_transfer = np.maximum(np.asarray(inverse_transfer_function, dtype=float), floor)

    updated = inlet + relaxation_factor * inverse_transfer * (target - downstream)
    return np.maximum(updated, floor)

#%%

def integrate_spectrum_variance(freq_array, spectrum):
    """Integrate a one-sided PSD over the supplied frequency band."""
    f = np.asarray(freq_array, dtype=float)
    S = np.asarray(spectrum, dtype=float)
    if hasattr(np, "trapezoid"):
        return np.trapezoid(S, f)
    else:
        return np.trapz(S, f)

#%%
def get_resolved_frequency_limits(time_steps, mesh_cutoff_freq=None):
    """
    Estimate frequency limits represented by the post-processing record.

    f_min is the lowest non-zero Fourier frequency of the finite record,
    f_nyquist is the sampling Nyquist frequency, and f_max is optionally
    limited by the mesh cutoff.
    """
    t = np.asarray(time_steps, dtype=float)
    t = t[np.isfinite(t)]
    if len(t) < 2:
        raise ValueError("Need at least two time steps.")

    dt = float(np.mean(np.diff(t)))
    T_sample = float(t[-1] - t[0])
    if T_sample <= 0.0:
        raise ValueError("Invalid sample duration.")

    f_min = 1.0 / T_sample
    f_nyquist = 1.0 / (2.0 * dt)
    f_max = f_nyquist if mesh_cutoff_freq is None else min(float(mesh_cutoff_freq), f_nyquist)

    return {
        "dt": dt,
        "T_sample": T_sample,
        "n_time_steps": len(t),
        "f_min": f_min,
        "f_nyquist": f_nyquist,
        "f_max": f_max,
    }

#%%

def make_resolved_frequency_grid(f_min, f_max, n_freq=256, spacing="log"):
    """Build a frequency grid over the resolved band used for solving."""
    if f_min <= 0.0 or f_max <= f_min:
        raise ValueError("Need 0 < f_min < f_max.")
    if spacing == "log":
        return np.geomspace(f_min, f_max, int(n_freq))
    if spacing == "linear":
        return np.linspace(f_min, f_max, int(n_freq))
    raise ValueError("spacing must be 'log' or 'linear'.")

#%%
def get_von_karman_spectrum_1d(freq_array, U, L, sigma2, component, floor=1e-30):
    """
    Construct a one-sided von Karman PSD using windLespy's non-dimensional
    spectrum definition.
    """
    f = np.asarray(freq_array, dtype=float)
    U = float(U)
    L = float(L)
    sigma2 = float(sigma2)
    if U <= 0.0 or L <= 0.0 or sigma2 < 0.0:
        return np.full_like(f, floor, dtype=float)

    red_f = f * L / U
    non_dim = LES._profileAnalysis.von_karman_spectra(red_f, component)
    S = non_dim * sigma2 / np.maximum(f, floor)
    return np.maximum(S, floor)

#%%

def get_unit_von_karman_shape(freq_array, U, L_eff, component, floor=1e-30):
    """Unit-variance von Karman spectral shape for a trial L_eff."""
    return get_von_karman_spectrum_1d(
        freq_array=freq_array,
        U=U,
        L=L_eff,
        sigma2=1.0,
        component=component,
        floor=floor,
    )
#%%

def build_resolved_variance_normalised_vk_spectrum(
    freq_array,
    U,
    L_eff,
    sigma2_target,
    component,
    floor=1e-30,
):
    """
    Construct a VK spectrum whose integral over freq_array equals sigma2_target.
    For each L_eff, the shape changes and the amplitude is recomputed.
    """
    phi = get_unit_von_karman_shape(freq_array, U, L_eff, component, floor=floor)
    unit_area = integrate_spectrum_variance(freq_array, phi)
    if not np.isfinite(unit_area) or unit_area <= 0.0:
        raise ValueError("Unit spectral shape has invalid resolved-band area.")
    amplitude = float(sigma2_target) / unit_area
    S = amplitude * phi
    return np.maximum(S, floor), amplitude, unit_area


#%%

def make_spectral_autocorrelation_operator(freq_array, dt, tau_max):
    """
    Precompute the cosine-transform operator used to convert a one-sided PSD
    to autocorrelation. This is the main speed-up versus rebuilding the cosine
    matrix during every L_eff trial.
    """
    f = np.asarray(freq_array, dtype=float)
    if len(f) < 2:
        raise ValueError("freq_array must contain at least two frequencies.")
    if tau_max <= 0.0:
        raise ValueError("tau_max must be positive.")

    tau_array = np.arange(0.0, tau_max + 0.5 * float(dt), float(dt))
    if len(tau_array) < 2:
        tau_array = np.array([0.0, float(dt)], dtype=float)

    w = np.zeros_like(f)
    w[1:-1] = 0.5 * (f[2:] - f[:-2])
    w[0] = 0.5 * (f[1] - f[0])
    w[-1] = 0.5 * (f[-1] - f[-2])

    cos_matrix = np.cos(2.0 * np.pi * tau_array[:, None] * f[None, :])

    return {
        "freq_array": f,
        "tau_array": tau_array,
        "weights": w,
        "cos_matrix": cos_matrix,
    }

#%%

def autocorrelation_from_spectrum_operator(spectrum, operator, floor=1e-30):
    """Fast autocorrelation calculation using a precomputed operator."""
    S = np.maximum(np.asarray(spectrum, dtype=float), floor)
    R = operator["cos_matrix"] @ (S * operator["weights"])
    if not np.isfinite(R[0]) or abs(R[0]) <= floor:
        return np.full_like(R, np.nan, dtype=float)
    return R / R[0]
#%%

def autocorrelation_from_one_sided_spectrum(
    freq_array,
    spectrum,
    tau_array,
    normalise=True,
    floor=1e-30,
):
    """
    Reconstruct the autocorrelation implied by a one-sided resolved-band PSD.

        R(tau) = integral S(f) cos(2 pi f tau) df
    """
    f = np.asarray(freq_array, dtype=float)
    S = np.maximum(np.asarray(spectrum, dtype=float), floor)
    tau = np.asarray(tau_array, dtype=float)

    cos_matrix = np.cos(2.0 * np.pi * tau[:, None] * f[None, :])
    integrand = cos_matrix * S[None, :]

    if hasattr(np, "trapezoid"):
        R = np.trapezoid(integrand, f, axis=1)
    else:
        R = np.trapz(integrand, f, axis=1)

    if not normalise:
        return R

    if not np.isfinite(R[0]) or abs(R[0]) <= floor:
        return np.full_like(R, np.nan, dtype=float)

    return R / R[0]


#%%

def integral_time_scale_from_rho(tau_array, rho_array, first_zero=True):
    """Integrate rho from tau=0 to the first zero crossing, matching windLespy."""
    tau = np.asarray(tau_array, dtype=float)
    rho = np.asarray(rho_array, dtype=float)
    valid = np.isfinite(tau) & np.isfinite(rho)
    tau = tau[valid]
    rho = rho[valid]
    if len(tau) < 2:
        return np.nan

    if first_zero:
        zero_crossings = np.where(rho <= 0.0)[0]
        i_end = int(zero_crossings[0]) if len(zero_crossings) > 0 else len(rho) - 1
    else:
        i_end = len(rho) - 1
    if i_end < 1:
        return 0.0

    if hasattr(np, "trapezoid"):
        return np.trapezoid(rho[: i_end + 1], tau[: i_end + 1])
    return np.trapz(rho[: i_end + 1], tau[: i_end + 1])


#%%

def resolved_integral_length_from_spectrum_operator(
    spectrum,
    U,
    operator,
    first_zero=True,
    floor=1e-30,
):
    """Compute resolved integral length from a PSD using a precomputed operator."""
    rho = autocorrelation_from_spectrum_operator(spectrum, operator, floor=floor)
    tau = operator["tau_array"]
    T_resolved = integral_time_scale_from_rho(tau, rho, first_zero=first_zero)
    return float(U) * T_resolved, T_resolved, tau, rho

#%%

def evaluate_resolved_vk_trial_fast(
    L_eff,
    freq_array,
    U,
    sigma2_target,
    component,
    operator,
    floor=1e-30,
):
    """Evaluate one L_eff trial with resolved variance normalisation."""
    S, amplitude, unit_area = build_resolved_variance_normalised_vk_spectrum(
        freq_array=freq_array,
        U=U,
        L_eff=L_eff,
        sigma2_target=sigma2_target,
        component=component,
        floor=floor,
    )
    L_resolved, T_resolved, tau, rho = resolved_integral_length_from_spectrum_operator(
        spectrum=S,
        U=U,
        operator=operator,
        first_zero=True,
        floor=floor,
    )
    return {
        "L_eff": L_eff,
        "S": S,
        "amplitude": amplitude,
        "unit_area": unit_area,
        "L_resolved": L_resolved,
        "T_resolved": T_resolved,
        "tau": tau,
        "rho": rho,
    }

#%%

def solve_effective_L_for_resolved_target_fast(
    freq_array,
    U,
    sigma2_target,
    L_target,
    component,
    operator,
    L_bounds=None,
    floor=1e-30,
    max_expand=6,
    verbose=False,
):
    """
    Solve for L_eff such that the resolved-band VK spectrum gives L_target
    when processed through the autocorrelation / first-zero-crossing method.
    """
    U = float(U)
    sigma2_target = float(sigma2_target)
    L_target = float(L_target)

    if U <= 0.0 or sigma2_target <= 0.0 or L_target <= 0.0:
        S = np.full_like(freq_array, floor, dtype=float)
        return {
            "L_eff": max(L_target, floor),
            "S": S,
            "amplitude": np.nan,
            "unit_area": np.nan,
            "L_resolved": np.nan,
            "T_resolved": np.nan,
            "tau": np.array([]),
            "rho": np.array([]),
            "converged": False,
            "message": "Invalid U, sigma2_target or L_target.",
            "bracket": None,
            "residual_bracket": None,
        }

    if L_bounds is None:
        L_low = max(L_target / 50.0, 1e-6)
        L_high = max(L_target * 50.0, L_low * 10.0)
    else:
        L_low, L_high = L_bounds

    def residual(log_L_eff):
        trial = evaluate_resolved_vk_trial_fast(
            L_eff=10.0 ** log_L_eff,
            freq_array=freq_array,
            U=U,
            sigma2_target=sigma2_target,
            component=component,
            operator=operator,
            floor=floor,
        )
        return trial["L_resolved"] - L_target

    log_low = np.log10(L_low)
    log_high = np.log10(L_high)
    r_low = residual(log_low)
    r_high = residual(log_high)

    expand_count = 0
    while (
        np.isfinite(r_low)
        and np.isfinite(r_high)
        and r_low * r_high > 0.0
        and expand_count < max_expand
    ):
        expand_count += 1
        log_low -= 0.5
        log_high += 0.5
        r_low = residual(log_low)
        r_high = residual(log_high)
        if verbose:
            print(
                f"Expand {expand_count}: "
                f"L_low={10.0**log_low:.3e}, r_low={r_low:.3e}, "
                f"L_high={10.0**log_high:.3e}, r_high={r_high:.3e}"
            )

    if not np.isfinite(r_low) or not np.isfinite(r_high) or r_low * r_high > 0.0:
        candidates = []
        for log_L in [log_low, log_high]:
            candidates.append(
                evaluate_resolved_vk_trial_fast(
                    L_eff=10.0 ** log_L,
                    freq_array=freq_array,
                    U=U,
                    sigma2_target=sigma2_target,
                    component=component,
                    operator=operator,
                    floor=floor,
                )
            )
        best = min(candidates, key=lambda d: abs(d["L_resolved"] - L_target))
        best["converged"] = False
        best["message"] = "Could not bracket target resolved length scale."
        best["bracket"] = (10.0 ** log_low, 10.0 ** log_high)
        best["residual_bracket"] = (r_low, r_high)
        return best

    sol = root_scalar(residual, bracket=[log_low, log_high], method="brentq", xtol=1e-4, rtol=1e-4)
    L_eff = 10.0 ** sol.root
    result = evaluate_resolved_vk_trial_fast(
        L_eff=L_eff,
        freq_array=freq_array,
        U=U,
        sigma2_target=sigma2_target,
        component=component,
        operator=operator,
        floor=floor,
    )
    result["converged"] = sol.converged
    result["message"] = "OK" if sol.converged else "Root solve did not converge."
    result["bracket"] = (10.0 ** log_low, 10.0 ** log_high)
    result["residual_bracket"] = (r_low, r_high)
    return result


#%%

def make_resolved_consistent_target_spectra_profile_dfsr(
    target_profile_array,
    time_steps,
    fMax,
    nFreq,
    mesh_cutoff_freqs=None,
    n_freq_solve=256,
    spacing="log",
    tau_factor=20.0,
    components=("u", "v", "w"),
    floor=1e-30,
    verbose=False,
):
    """
    Build DFSR-compatible targetSpectraProfile arrays using resolved-band
    variance and length-scale correction.

    The L_eff solve is performed on a cheap resolved-frequency grid. The final
    spectra are evaluated on windLespy/DFSR's required common grid:
        get_freq_array(fMax, nFreq)
    so the output shape is exactly (3, nHeights, nFreq).

    Returns
    -------
    corrected_dfsr_spectra : ndarray
        Corrected target spectra on the DFSR grid.
    uncorrected_dfsr_spectra : ndarray
        Standard VK spectra on the DFSR grid using the original target I,L.
    corrected_diagnostics : list[list[dict]]
        Per-component/per-height diagnostics from the solve.
    uncorrected_diagnostics : list[list[dict]]
        Per-component/per-height diagnostics for the original VK spectrum.
    summary_df : pandas.DataFrame
        Profile-level diagnostics suitable for saving to CSV.
    """
    profile = np.asarray(target_profile_array, dtype=float)
    U_array = profile[:, 0]
    sigma2_array = profile[:, 1:4]
    L_array = profile[:, 4:7]
    n_heights = profile.shape[0]

    if mesh_cutoff_freqs is None:
        mesh_cutoff_freqs = [None] * n_heights

    freq_dfsr = get_freq_array(fMax, nFreq)
    corrected_dfsr = np.full((3, n_heights, nFreq), floor, dtype=float)
    uncorrected_dfsr = np.full((3, n_heights, nFreq), floor, dtype=float)
    corrected_diagnostics = [[None for _ in range(n_heights)] for _ in range(3)]
    uncorrected_diagnostics = [[None for _ in range(n_heights)] for _ in range(3)]
    rows = []

    for h in range(n_heights):
        U = U_array[h]
        limits = get_resolved_frequency_limits(time_steps, mesh_cutoff_freq=mesh_cutoff_freqs[h])
        freq_solve = make_resolved_frequency_grid(
            limits["f_min"], limits["f_max"], n_freq=n_freq_solve, spacing=spacing
        )
        L_max = np.nanmax(L_array[h, :])
        tau_max = min(limits["T_sample"], tau_factor * L_max / max(float(U), 1e-12))
        operator = make_spectral_autocorrelation_operator(freq_solve, limits["dt"], tau_max)

        for comp_id, comp in enumerate(components):
            sigma2_target = sigma2_array[h, comp_id]
            L_target = L_array[h, comp_id]

            S_uncorr_solve = get_von_karman_spectrum_1d(
                freq_solve, U, L_target, sigma2_target, comp, floor=floor
            )
            L_uncorr, T_uncorr, tau_uncorr, rho_uncorr = resolved_integral_length_from_spectrum_operator(
                S_uncorr_solve, U, operator, first_zero=True, floor=floor
            )
            var_uncorr = integrate_spectrum_variance(freq_solve, S_uncorr_solve)

            solved = solve_effective_L_for_resolved_target_fast(
                freq_array=freq_solve,
                U=U,
                sigma2_target=sigma2_target,
                L_target=L_target,
                component=comp,
                operator=operator,
                floor=floor,
                verbose=False,
            )

            # Evaluate final spectra on the DFSR grid. The solved amplitude is
            # defined by the resolved-band variance constraint on freq_solve.
            unit_dfsr = get_unit_von_karman_shape(freq_dfsr, U, solved["L_eff"], comp, floor=floor)
            corrected_dfsr[comp_id, h, :] = np.maximum(solved["amplitude"] * unit_dfsr, floor)
            uncorrected_dfsr[comp_id, h, :] = get_von_karman_spectrum_1d(
                freq_dfsr, U, L_target, sigma2_target, comp, floor=floor
            )

            corrected_diagnostics[comp_id][h] = {
                **solved,
                "freq_array": freq_solve,
                "freq_dfsr": freq_dfsr,
                "component": comp,
                "height_id": h,
                "U": U,
                "sigma2_target": sigma2_target,
                "L_target": L_target,
                "limits": limits,
                "tau_max": tau_max,
            }
            uncorrected_diagnostics[comp_id][h] = {
                "S": S_uncorr_solve,
                "freq_array": freq_solve,
                "freq_dfsr": freq_dfsr,
                "component": comp,
                "height_id": h,
                "U": U,
                "sigma2_target": sigma2_target,
                "L_target": L_target,
                "L_resolved": L_uncorr,
                "T_resolved": T_uncorr,
                "tau": tau_uncorr,
                "rho": rho_uncorr,
                "resolved_variance": var_uncorr,
                "limits": limits,
                "tau_max": tau_max,
            }

            rows.append({
                "height_id": h,
                "component": comp,
                "U": U,
                "sigma2_target": sigma2_target,
                "L_target": L_target,
                "L_eff": solved["L_eff"],
                "L_eff_over_L_target": solved["L_eff"] / L_target if L_target > 0 else np.nan,
                "L_resolved_corrected": solved["L_resolved"],
                "L_resolved_uncorrected": L_uncorr,
                "resolved_variance_corrected": integrate_spectrum_variance(freq_solve, solved["S"]),
                "resolved_variance_uncorrected": var_uncorr,
                "resolved_variance_uncorrected_over_target": var_uncorr / sigma2_target if sigma2_target > 0 else np.nan,
                "amplitude": solved["amplitude"],
                "f_min": limits["f_min"],
                "f_max_resolved": limits["f_max"],
                "fMax_dfsr": fMax,
                "nFreq_dfsr": nFreq,
                "f_nyquist": limits["f_nyquist"],
                "T_sample": limits["T_sample"],
                "tau_max": tau_max,
                "converged": solved["converged"],
                "message": solved["message"],
            })

            if verbose:
                print(
                    f"h={h:03d}, comp={comp}, "
                    f"L_target={L_target:.4g}, L_eff={solved['L_eff']:.4g}, "
                    f"L_corr={solved['L_resolved']:.4g}, L_uncorr={L_uncorr:.4g}, "
                    f"A={solved['amplitude']:.4g}, converged={solved['converged']}"
                )

    summary_df = pd.DataFrame(rows)
    return corrected_dfsr, uncorrected_dfsr, corrected_diagnostics, uncorrected_diagnostics, summary_df

#%%

def get_kaimal_uw_cospectrum_shape(
    freq_array,
    z_array,
    U_array,
    floor=1e-30,
):
    """
    Return a positive Kaimal uw co-spectrum magnitude shape phi_uw(z,f).

    Kaimal neutral form:

        - n C_uw(n) / u_*^2 = 14 eta / (1 + 9.6 eta)^2.4

    where eta = n z / U.

    This function returns only the positive shape. The sign and final area
    normalisation are applied later.
    """
    f = np.asarray(freq_array, dtype=float).reshape(-1)
    z = np.asarray(z_array, dtype=float).reshape(-1)
    U = np.asarray(U_array, dtype=float).reshape(-1)

    if np.any(f <= 0.0):
        raise ValueError("freq_array must be strictly positive.")
    if len(z) != len(U):
        raise ValueError("z_array and U_array must have the same length.")

    n_heights = len(z)
    n_freq = len(f)
    phi = np.zeros((n_heights, n_freq), dtype=float)

    for h in range(n_heights):
        Uh = max(float(U[h]), floor)
        zh = max(float(z[h]), floor)

        eta = f * zh / Uh

        # Since -n Cuw/u_*^2 = F(eta),
        # |Cuw| shape is F(eta)/n.
        F_eta = 14.0 * eta / np.power(1.0 + 9.6 * eta, 2.4)
        phi[h, :] = F_eta / np.maximum(f, floor)

    return np.maximum(phi, floor)

#%%

def normalise_uw_cospectrum_to_stress(
    freq_array,
    shape_array,
    uw_stress_array,
    enforce_negative=True,
    floor=1e-30,
):
    """
    Scale a positive co-spectral shape so that:

        integral Cuw(f,z) df = uw_stress(z)

    shape_array has shape (nHeights, nFreq).
    uw_stress_array has shape (nHeights,).
    """
    f = np.asarray(freq_array, dtype=float).reshape(-1)
    shape = np.asarray(shape_array, dtype=float)
    uw = np.asarray(uw_stress_array, dtype=float).reshape(-1)

    if shape.ndim != 2:
        raise ValueError("shape_array must have shape (nHeights, nFreq).")
    if shape.shape[0] != len(uw):
        raise ValueError("uw_stress_array length must match nHeights.")
    if shape.shape[1] != len(f):
        raise ValueError("shape_array frequency dimension must match freq_array.")

    out = np.zeros_like(shape)

    for h in range(shape.shape[0]):
        area = np.trapz(np.maximum(shape[h, :], floor), f)
        if not np.isfinite(area) or area <= 0.0:
            out[h, :] = 0.0
            continue

        target_area = float(uw[h])

        if enforce_negative:
            target_area = -abs(target_area)

        out[h, :] = target_area * shape[h, :] / area

    return out

#%%

def estimate_uw_stress_profile(
    target_profile_array,
    target_profile_df=None,
    uw_column_candidates=("uw", "uwStress", "u'w'", "uPrimeWPrime", "Ruw"),
    fallback_rho=-0.30,
):
    """
    Get uw stress profile from a dataframe if available; otherwise use:

        uw = rho_uw * sigma_u * sigma_w

    target_profile_array convention used by DFSR:
        column 0: U
        columns 1:4: variances uu, vv, ww
    """
    n_heights = target_profile_array.shape[0]

    if target_profile_df is not None:
        for col in uw_column_candidates:
            if col in target_profile_df.columns:
                uw = target_profile_df[col].to_numpy(dtype=float)
                if len(uw) == n_heights:
                    return uw

    sigma_u = np.sqrt(np.maximum(target_profile_array[:, 1], 0.0))
    sigma_w = np.sqrt(np.maximum(target_profile_array[:, 3], 0.0))

    return fallback_rho * sigma_u * sigma_w

#%%

def clip_uw_cospectrum_realizable(
    c_uw_array,
    spectra_array,
    rho_max=0.95,
):
    """
    Enforce local one-point CPSD realizability:

        |Cuw(z,f)| <= rho_max * sqrt(Suu(z,f) Sww(z,f))

    spectra_array shape: (3, nHeights, nFreq)
    c_uw_array shape:   (nHeights, nFreq)
    """
    C = np.asarray(c_uw_array, dtype=float).copy()
    S = np.asarray(spectra_array, dtype=float)

    if S.shape[0] != 3:
        raise ValueError("spectra_array must have shape (3, nHeights, nFreq).")
    if C.shape != S[0].shape:
        raise ValueError("c_uw_array must have shape (nHeights, nFreq).")

    bound = rho_max * np.sqrt(
        np.maximum(S[0, :, :], 0.0) *
        np.maximum(S[2, :, :], 0.0)
    )

    C_clipped = np.clip(C, -bound, bound)

    diagnostics = {
        "n_clipped": int(np.count_nonzero(C_clipped != C)),
        "max_abs_rho_before": float(
            np.nanmax(
                np.abs(C) / np.maximum(
                    np.sqrt(np.maximum(S[0, :, :], 0.0) * np.maximum(S[2, :, :], 0.0)),
                    1e-300,
                )
            )
        ),
        "max_abs_rho_after": float(
            np.nanmax(
                np.abs(C_clipped) / np.maximum(
                    np.sqrt(np.maximum(S[0, :, :], 0.0) * np.maximum(S[2, :, :], 0.0)),
                    1e-300,
                )
            )
        ),
    }

    return C_clipped, diagnostics

#%%

def make_target_cospectral_dfsr_profiles(
    target_profile_array,
    z_array,
    freq_array,
    spectra_array,
    uw_stress_array=None,
    target_profile_df=None,
    fallback_rho=-0.30,
    rho_max=0.95,
    enforce_negative=True,
    floor=1e-30,
    resolved_fmin_array=None,
    resolved_fmax_array=None,
):
    """
    Build target uwStress(z) and Cuw(z,f) for CoSpectralDFSRTurb.
    """
    z_array = np.asarray(z_array, dtype=float).reshape(-1)
    freq_array = np.asarray(freq_array, dtype=float).reshape(-1)
    target_profile_array = np.asarray(target_profile_array, dtype=float)
    spectra_array = np.asarray(spectra_array, dtype=float)

    U = target_profile_array[:, 0]

    if uw_stress_array is not None:
        uw_stress = np.asarray(uw_stress_array, dtype=float).reshape(-1)

    if len(uw_stress) != target_profile_array.shape[0]:
        raise ValueError(
            "uw_stress_array length must match the number of target profile heights."
        )
    else:
        uw_stress = estimate_uw_stress_profile(
            target_profile_array=target_profile_array,
            target_profile_df=target_profile_df,
            fallback_rho=fallback_rho,
        )

    shape = get_kaimal_uw_cospectrum_shape(
        freq_array=freq_array,
        z_array=z_array,
        U_array=U,
        floor=floor,
    )

    if resolved_fmin_array is not None and resolved_fmax_array is not None:
        c_uw = normalise_uw_cospectrum_to_resolved_stress(
            freq_array=freq_array,
            shape_array=shape,
            uw_stress_array=uw_stress,
            resolved_fmin_array=resolved_fmin_array,
            resolved_fmax_array=resolved_fmax_array,
            enforce_negative=enforce_negative,
            floor=floor,
        )
    else:
        c_uw = normalise_uw_cospectrum_to_stress(
            freq_array=freq_array,
            shape_array=shape,
            uw_stress_array=uw_stress,
            enforce_negative=enforce_negative,
            floor=floor,
        )

    c_uw_clipped, diagnostics = clip_uw_cospectrum_realizable(
        c_uw_array=c_uw,
        spectra_array=spectra_array,
        rho_max=rho_max,
    )

    if hasattr(np, "trapezoid"):
        diagnostics["uw_area_before_clip"] = np.trapezoid(c_uw, freq_array, axis=1)
        diagnostics["uw_area_after_clip"] = np.trapezoid(c_uw_clipped, freq_array, axis=1)
    else:
        diagnostics["uw_area_before_clip"] = np.trapz(c_uw, freq_array, axis=1)
        diagnostics["uw_area_after_clip"] = np.trapz(c_uw_clipped, freq_array, axis=1)
    diagnostics["uw_stress_target"] = uw_stress

    return uw_stress, c_uw_clipped, diagnostics

#%%

def normalise_uw_cospectrum_to_resolved_stress(
    freq_array,
    shape_array,
    uw_stress_array,
    resolved_fmin_array,
    resolved_fmax_array,
    enforce_negative=True,
    floor=1e-30,
):
    """
    Scale positive Cuw shape so that the integral over the resolved frequency
    band equals uw_stress(z).

        integral_{fmin(z)}^{fmax(z)} Cuw(z,f) df = uwStress(z)

    shape_array has shape (nHeights, nFreq).
    """
    f = np.asarray(freq_array, dtype=float).reshape(-1)
    shape = np.asarray(shape_array, dtype=float)
    uw = np.asarray(uw_stress_array, dtype=float).reshape(-1)
    fmin = np.asarray(resolved_fmin_array, dtype=float).reshape(-1)
    fmax = np.asarray(resolved_fmax_array, dtype=float).reshape(-1)

    if shape.shape != (len(uw), len(f)):
        raise ValueError("shape_array must have shape (nHeights, nFreq).")

    out = np.zeros_like(shape)

    for h in range(shape.shape[0]):
        mask = (f >= fmin[h]) & (f <= fmax[h])
        if np.count_nonzero(mask) < 2:
            mask = np.ones_like(f, dtype=bool)

        if hasattr(np, "trapezoid"):
            area = np.trapezoid(np.maximum(shape[h, mask], floor), f[mask])
        else:
            area = np.trapz(np.maximum(shape[h, mask], floor), f[mask])

        if not np.isfinite(area) or area <= 0.0:
            out[h, :] = 0.0
            continue

        target_area = float(uw[h])
        if enforce_negative:
            target_area = -abs(target_area)

        out[h, :] = target_area * shape[h, :] / area

    return out


#%%

def make_cospectral_target_profile_from_nheri(
    approach_flow_data,
    z_inlet,
    z_top,
    structure_height,
    smoothing=True,
    ti_smooth_num=5,
    l_smooth_num=7,
    uw_extension="constant_correlation",
):
    """
    Build target profile data for CoSpectralDFSRTurb from NHERI approach-flow data.

    The final profile contains exactly one Reynolds shear-stress column:
        uwStress

    The returned array has columns:
        U, uu, vv, ww, Lu, Lv, Lw, uwStress
    """

    z_inlet = np.asarray(z_inlet, dtype=float).reshape(-1)

    # ------------------------------------------------------------------
    # 1. Load measured NHERI velocity/profile data
    # ------------------------------------------------------------------
    vel_array_3d = LES._windTunnel.get_nheri_vel_time_series(approach_flow_data)

    measured_df = LES._windTunnel.get_nheri_profile_df(approach_flow_data)

    int_length_scales = LES._windTunnel.calc_nheri_int_length_scales(vel_array_3d)

    measured_df = LES._windTunnel.add_nheri_int_length_scales(
            measured_df,
            int_length_scales,
        )
    
    measured_df = LES._windTunnel.add_nheri_reynolds_stresses(
        measured_df,
        vel_array_3d,
        ddof=0,
    )
    
    experimental_profile_df = measured_df.copy()
    
    extended_df = LES._windTunnel.extend_nheri_profiles_with_reynolds_stress(
        measured_df,
        z_top,
        fit_zmin=None,
        fit_zmax=None,
        uw_extension=uw_extension,
    )
    
    if "uwStress" not in extended_df.columns and "uw" in extended_df.columns:
        extended_df["uwStress"] = extended_df["uw"].to_numpy(dtype=float)
    
    if "uw" in extended_df.columns:
        extended_df = extended_df.drop(columns=["uw"])
    
    if smoothing:
        smoothed_df = smooth_cospectral_target_profile_df(
        extended_df,
        ti_smooth_num=ti_smooth_num,
        l_smooth_num=l_smooth_num,
    )
    else:
        smoothed_df = extended_df.copy()

    # ------------------------------------------------------------------
    # 4. Map to inlet cell-centre heights
    # ------------------------------------------------------------------
    mapped_df = map_cospectral_target_profile_to_z(
        smoothed_df,
        z_inlet,
    )

    # ------------------------------------------------------------------
    # 5. Build direct coSpectral profile array
    # ------------------------------------------------------------------
    cospectral_target_profile_array = cospectral_target_profile_df_to_array(
        mapped_df
    )

    return (
        experimental_profile_df,
        extended_df,
        smoothed_df,
        mapped_df,
        cospectral_target_profile_array,
    )

#%%

def smooth_cospectral_target_profile_df(
    profile_df,
    window_length=7,
    polyorder=2,
    columns=("U", "Iu", "Iv", "Iw", "Lu", "Lv", "Lw", "uwStress"),
):
    """
    Smooth all CoSpectralDFSRTurb target-profile quantities consistently.

    For positive quantities, smoothing is done in linear space and clipped
    positive afterward.

    For uwStress, smoothing is done in linear space and the dominant sign is
    preserved.
    """

    import numpy as np
    from scipy.signal import savgol_filter

    df = profile_df.copy()

    if "uw" in df.columns:
        if "uwStress" not in df.columns:
            df["uwStress"] = df["uw"]
        df = df.drop(columns=["uw"])

    z = df["z"].to_numpy(dtype=float)

    def _smooth_1d(y, preserve_sign=False, positive=False):
        y = np.asarray(y, dtype=float)
        valid = np.isfinite(z) & np.isfinite(y)

        if np.count_nonzero(valid) < 3:
            return y

        y_fill = np.interp(z, z[valid], y[valid])

        n = len(y_fill)
        win = int(window_length)

        if win > n:
            win = n if n % 2 == 1 else n - 1
        if win % 2 == 0:
            win -= 1
        if win <= polyorder or win < 3:
            y_smooth = y_fill
        else:
            y_smooth = savgol_filter(
                y_fill,
                window_length=win,
                polyorder=polyorder,
                mode="interp",
            )

        if positive:
            y_smooth = np.maximum(y_smooth, 1e-30)

        if preserve_sign:
            dominant_sign = np.sign(np.nanmedian(y_fill[valid]))
            if dominant_sign != 0.0:
                y_smooth = dominant_sign * np.abs(y_smooth)

        return y_smooth

    positive_cols = {"U", "Iu", "Iv", "Iw", "Lu", "Lv", "Lw"}

    for col in columns:
        if col not in df.columns:
            continue

        if col == "uwStress":
            df[col] = _smooth_1d(
                df[col].to_numpy(dtype=float),
                preserve_sign=True,
                positive=False,
            )
        else:
            df[col] = _smooth_1d(
                df[col].to_numpy(dtype=float),
                preserve_sign=False,
                positive=(col in positive_cols),
            )

    return df

#%%

def map_cospectral_target_profile_to_z(
    profile_df,
    z_target,
    columns=("U", "Iu", "Iv", "Iw", "Lu", "Lv", "Lw", "uwStress"),
):
    """
    Map a CoSpectralDFSRTurb target profile dataframe to inlet cell-centre heights.
    """

    import numpy as np
    import pandas as pd

    z_source = profile_df["z"].to_numpy(dtype=float)
    z_target = np.asarray(z_target, dtype=float).reshape(-1)

    mapped = {"z": z_target}

    for col in columns:
        if col not in profile_df.columns:
            raise ValueError(f"Required coSpectral target column missing: {col}")

        mapped[col] = np.interp(
            z_target,
            z_source,
            profile_df[col].to_numpy(dtype=float),
        )

    return pd.DataFrame(mapped)

#%%

def cospectral_target_profile_df_to_array(profile_df):
    """
    Convert coSpectralDFSR target profile dataframe to array.

    Output columns:
        U, uu, vv, ww, Lu, Lv, Lw, uwStress

    where:
        uu = (Iu U)^2
        vv = (Iv U)^2
        ww = (Iw U)^2
    """

    import numpy as np

    required = ["U", "Iu", "Iv", "Iw", "Lu", "Lv", "Lw", "uwStress"]

    missing = [c for c in required if c not in profile_df.columns]
    if missing:
        raise ValueError(f"Missing required coSpectral target columns: {missing}")

    U = profile_df["U"].to_numpy(dtype=float)
    Iu = profile_df["Iu"].to_numpy(dtype=float)
    Iv = profile_df["Iv"].to_numpy(dtype=float)
    Iw = profile_df["Iw"].to_numpy(dtype=float)

    uu = (Iu * U) ** 2
    vv = (Iv * U) ** 2
    ww = (Iw * U) ** 2

    Lu = profile_df["Lu"].to_numpy(dtype=float)
    Lv = profile_df["Lv"].to_numpy(dtype=float)
    Lw = profile_df["Lw"].to_numpy(dtype=float)
    uw = profile_df["uwStress"].to_numpy(dtype=float)

    return np.column_stack([U, uu, vv, ww, Lu, Lv, Lw, uw])

#%%

def smooth_cospectral_target_profile_df(
    profile_df,
    ti_smooth_num=5,
    l_smooth_num=7,
    columns=("U", "Iu", "Iv", "Iw", "Lu", "Lv", "Lw", "uwStress"),
):
    """
    Smooth CoSpectralDFSRTurb target profiles using the same moving-average
    kernel convention as LES._profileAnalysis.smooth_profiles(), but without
    plotting side effects.

    U, Iu, Iv, Iw, uwStress use ti_smooth_num.
    Lu, Lv, Lw use l_smooth_num.
    """

    df = profile_df.copy()

    if "uwStress" not in df.columns and "uw" in df.columns:
        df["uwStress"] = df["uw"].to_numpy(dtype=float)

    if "uw" in df.columns:
        df = df.drop(columns=["uw"])

    out = {"z": df["z"].to_numpy(dtype=float)}

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Required coSpectral target column missing: {col}")

        y = df[col].to_numpy(dtype=float)

        finite = np.isfinite(y)
        if np.count_nonzero(finite) < 2:
            raise ValueError(
                f"Column {col} has fewer than two finite values before smoothing."
            )

        if not np.all(finite):
            z = df["z"].to_numpy(dtype=float)
            y = np.interp(z, z[finite], y[finite])

        n_smooth = l_smooth_num if col in ("Lu", "Lv", "Lw") else ti_smooth_num

        out[col] = LES._profileAnalysis.wong_kernel_smooth(
            y,
            n=n_smooth,
        )

    return pd.DataFrame(out)