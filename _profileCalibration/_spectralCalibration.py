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
from scipy.signal import savgol_filter

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
    nperseg=None,
    noverlap=None,
    ):
    
    if inlet_or_downstream == "downstream":
        if burn_in_time is None or time_steps is None:
            raise ValueError(
                "For inlet_or_downstream='downstream', both burn_in_time and time_steps must be provided."
            )
        mask = time_steps > burn_in_time
        vel_array_3d = vel_array_3d[:, mask, :]

    freq_array = get_freq_array(fMax, nFreq)
    
    if nperseg is None:
        nperseg = get_welch_nperseg_from_dfsr_grid(time_step, fMax, nFreq)

    if noverlap is None:
        noverlap = nperseg // 2
    
    downstream_spectra_array = np.zeros((3, np.shape(vel_array_3d)[2], nFreq))

    for height_id in range(np.shape(vel_array_3d)[2]):
        for vel_comp_id, vel_comp in enumerate(["u", "v", "w"]):
            
            U_ref = downstream_profile_array[height_id, 0]
            vel_time_series = vel_array_3d[vel_comp_id, :, height_id]
            
            f, S_n, _, _ = LES._profileAnalysis.welch_psd(
                vel_time_series,
                1 / time_step,
                body_dim,
                U_ref,
                nperseg=nperseg,
                noverlap=noverlap,
                detrend="constant",
                window="hann"
            )
            
            # Drop the DC bin so Welch frequencies become:
            # [fMax/nFreq, 2*fMax/nFreq, ..., fMax]
            f = f[1:]
            S_n = S_n[1:]
            
            downstream_spectra_array[vel_comp_id, height_id, :] = S_n
 
    return downstream_spectra_array

#%%

def smooth_downstream_spectra_array(freq_array, downstream_spectra_array):
    
    def smooth_psd(freq_array, S_ii, window_length=31, polyorder=2, floor=1e-16):
        """
        Smooth PSD in log10(S) on a log-frequency grid, then interpolate
        back to the original frequency array.
        """
        f = np.asarray(freq_array, dtype=float)
        S_ii = np.maximum(np.asarray(S_ii, dtype=float), floor)

        valid = np.isfinite(f) & np.isfinite(S_ii) & (f > 0.0)
        f_valid = f[valid]
        S_valid = S_ii[valid]

        if len(f_valid) < window_length:
            # shrink window if needed
            window_length_local = len(f_valid) if len(f_valid) % 2 == 1 else len(f_valid) - 1
            if window_length_local < 5:
                return S_ii.copy()
        else:
            window_length_local = window_length

        # create log-spaced frequency grid
        f_log = np.geomspace(f_valid[0], f_valid[-1], len(f_valid))

        # interpolate original PSD onto log-frequency grid
        S_loggrid = np.interp(f_log, f_valid, S_valid)

        # smooth in log10(PSD)
        logS = np.log10(S_loggrid)
        logS_s = savgol_filter(
            logS,
            window_length=window_length_local,
            polyorder=polyorder,
            mode="interp"
        )

        S_s_loggrid = 10**logS_s

        # interpolate back onto original grid
        S_smoothed = np.interp(f, f_log, S_s_loggrid)

        return S_smoothed
    
    smoothed_downstream_spectra_array = downstream_spectra_array.copy()
    
    for height_id in range(smoothed_downstream_spectra_array.shape[1]):
        for vel_comp_id in range(3):
            smoothed_downstream_spectra_array[vel_comp_id, height_id, :] = smooth_psd(
                freq_array,
                smoothed_downstream_spectra_array[vel_comp_id, height_id, :],
                window_length=31,
                polyorder=2
            )
            
    return smoothed_downstream_spectra_array

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
    """
    For each spectrum, overwrite values above cutoff with a continuous
    power-law tail:
        S(f) = S(fc) * (f/fc)^slope,   for f > fc

    Parameters
    ----------
    freq_array : (nFreq,)
    spectra_array : (3, nHeights, nFreq)
    cutoff_freqs : (nHeights,) or (3, nHeights)
    slope : float
        Use -5/3 for PSDs.
    """
    f = np.asarray(freq_array, dtype=float)
    S = np.maximum(np.asarray(spectra_array, dtype=float).copy(), floor)

    if S.ndim != 3 or S.shape[0] != 3:
        raise ValueError("spectra_array must have shape (3, nHeights, nFreq)")

    n_comp, n_heights, n_freq = S.shape

    if f.ndim != 1 or f.shape[0] != n_freq:
        raise ValueError("freq_array must have length matching spectra_array.shape[2]")

    logf = np.log(np.maximum(f, floor))

    for comp in range(n_comp):
        for h in range(n_heights):
            fc = float(mesh_cutoff_freqs_3d[comp, h])

            if not np.isfinite(fc):
                continue

            fc = np.clip(fc, f[0], f[-1])

            spec = np.maximum(S[comp, h, :], floor)
            logS = np.log(spec)

            # Interpolate S(fc) in log-log space for continuity
            logS_fc = np.interp(np.log(fc), logf, logS)
            S_fc = np.exp(logS_fc)

            mask = f > fc
            if np.any(mask):
                S[comp, h, mask] = np.maximum(S_fc * (f[mask] / fc) ** slope, floor)

    return S
#%%

def get_convective_spectral_function(inlet_spectra_array, downstream_spectra_array):
    
    convective_spectral_function = inlet_spectra_array / downstream_spectra_array 
    
    return convective_spectral_function

#%%

def get_updated_spectra_array(target_spectra_array, conv_spectral_func, inlet_spectra_array, downstream_spectra_array):
    
    updated_spectra_array = inlet_spectra_array + conv_spectral_func*(target_spectra_array - downstream_spectra_array)
    
    return updated_spectra_array

