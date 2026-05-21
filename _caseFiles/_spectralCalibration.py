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

cwd = os.path.dirname(os.path.abspath(__file__))
windlespy_path = os.path.abspath(os.path.join(cwd, "..", ".."))
sys.path.append(windlespy_path)
import windlespy as LES
sys.path.remove(windlespy_path)

#%%

def write_spectra_profile(spectra_array, z_array, filepath, clip_min=1e-16):
    
    """
    Write spectraProfile for modified DFSR.

    Parameters
    ----------
    filepath : str
        Output file path, e.g. .../constant/boundaryData/windProfile/spectraProfile
    z_array : (nHeights,) array
        Heights corresponding to each row
    spectra_array : (3, nHeights, nFreq) array
        PSD array in order [u/v/w, height, frequency]
    clip_min : float
        Minimum PSD value written to file
    """

    z_array = np.asarray(z_array, dtype=float).reshape(-1)
    spectra_array = np.asarray(spectra_array, dtype=float)

    if spectra_array.ndim != 3:
        raise ValueError(f"spectra_array must be 3D, got shape {spectra_array.shape}")

    if spectra_array.shape[0] != 3:
        raise ValueError(
            f"spectra_array first dimension must be 3 for [u,v,w], got {spectra_array.shape[0]}"
        )

    _, n_heights, n_freq = spectra_array.shape

    if len(z_array) != n_heights:
        raise ValueError(
            f"Length of z_array ({len(z_array)}) does not match spectra heights ({n_heights})"
        )

    # Clean spectra
    spectra_clean = spectra_array.copy()

    # Replace non-finite values
    bad = ~np.isfinite(spectra_clean)
    if np.any(bad):
        print(f"Warning: replacing {bad.sum()} non-finite spectral values")
        spectra_clean[bad] = np.nan

    # Fallback for NaNs/Infs: use target/inlet later if you want, but here clip safely
    spectra_clean = np.nan_to_num(
        spectra_clean,
        nan=clip_min,
        posinf=clip_min,
        neginf=clip_min
    )

    # PSD must be non-negative
    spectra_clean = np.maximum(spectra_clean, clip_min)

    with open(filepath, "w") as f:
        f.write(f"{n_heights} {n_freq}\n")

        for h in range(n_heights):
            row = np.concatenate([
                [z_array[h]],
                spectra_clean[0, h, :],   # Su
                spectra_clean[1, h, :],   # Sv
                spectra_clean[2, h, :]    # Sw
            ])

            f.write(" ".join(f"{val:.12e}" for val in row) + "\n")
            

#%%

def write_dfsr_iter_spectra(
    case_path,
    iter_status,
    z_array,
    freq_array,
    current_spectra_array,
    downstream_spectra_array,
    inlet_or_downstream="inlet",
    new_inlet_spectra_array=None,
    cutoff_freqs=None,
    clip_min=1e-16,
):
    """
    Write archived spectra profiles for a DFSR calibration iteration.

    Files written in iteration folder:
        inletSpectraProfile
        postCorrectionSpectraProfile
        newInletSpectraProfile      (optional)
        targetSpectraProfile        (optional)
        freqArray                   (always)
        cutoffFrequencies           (optional)
        spectra_arrays.npz          (optional convenience archive)
    """

    if inlet_or_downstream == "inlet":
        dfsr_iter_path = os.path.join(case_path, "log", "inletCalibration")
    elif inlet_or_downstream == "downstream":
        dfsr_iter_path = os.path.join(case_path, "log", "downstreamCalibration")
    else:
        raise ValueError("inlet_or_downstream must be 'inlet' or 'downstream'")

    os.makedirs(dfsr_iter_path, exist_ok=True)

    iteration = iter_status["iteration"]
    iteration_path = os.path.join(dfsr_iter_path, f"iteration{iteration}")
    os.makedirs(iteration_path, exist_ok=True)

    z_array = np.asarray(z_array, dtype=float).reshape(-1)
    freq_array = np.asarray(freq_array, dtype=float).reshape(-1)

    current_spectra_array = np.asarray(current_spectra_array, dtype=float)
    downstream_spectra_array = np.asarray(downstream_spectra_array, dtype=float)

    if current_spectra_array.ndim != 3 or current_spectra_array.shape[0] != 3:
        raise ValueError("current_spectra_array must have shape (3, nHeights, nFreq)")
    if downstream_spectra_array.ndim != 3 or downstream_spectra_array.shape[0] != 3:
        raise ValueError("downstream_spectra_array must have shape (3, nHeights, nFreq)")

    n_heights = current_spectra_array.shape[1]
    n_freq = current_spectra_array.shape[2]

    if downstream_spectra_array.shape != current_spectra_array.shape:
        raise ValueError("current_spectra_array and downstream_spectra_array must have the same shape")
    if len(z_array) != n_heights:
        raise ValueError(f"len(z_array)={len(z_array)} does not match nHeights={n_heights}")
    if len(freq_array) != n_freq:
        raise ValueError(f"len(freq_array)={len(freq_array)} does not match nFreq={n_freq}")

    # Main spectra profile files
    write_spectra_profile(
        current_spectra_array,
        z_array,
        os.path.join(iteration_path,"inletSpectraProfile"),
        clip_min=clip_min,
    )

    write_spectra_profile(
        downstream_spectra_array,
        z_array,
        os.path.join(iteration_path,"postCorrectionSpectraProfile"),
        clip_min=clip_min,
    )

    if new_inlet_spectra_array is not None:
        new_inlet_spectra_array = np.asarray(new_inlet_spectra_array, dtype=float)
        if new_inlet_spectra_array.shape != current_spectra_array.shape:
            raise ValueError("new_inlet_spectra_array must have same shape as current_spectra_array")

        write_spectra_profile(
            new_inlet_spectra_array,
            z_array,
            os.path.join(iteration_path,"newInletSpectraProfile"),
            clip_min=clip_min,
        )


    # Save frequency array for reference
    np.savetxt(
        os.path.join(iteration_path, "freqArray"),
        freq_array[:, np.newaxis],
        fmt="%.12e",
        delimiter="\t",
        header="f_Hz",
        comments="",
    )

    # Save cutoff frequencies if provided
    if cutoff_freqs is not None:
        cutoff_freqs = np.asarray(cutoff_freqs, dtype=float)

        if cutoff_freqs.ndim == 1:
            if cutoff_freqs.shape[0] != n_heights:
                raise ValueError("1D cutoff_freqs must have length nHeights")
            cutoff_df = pd.DataFrame({
                "z": z_array,
                "fc": cutoff_freqs,
            })
        elif cutoff_freqs.ndim == 2:
            if cutoff_freqs.shape != (3, n_heights):
                raise ValueError("2D cutoff_freqs must have shape (3, nHeights)")
            cutoff_df = pd.DataFrame({
                "z": z_array,
                "fc_u": cutoff_freqs[0, :],
                "fc_v": cutoff_freqs[1, :],
                "fc_w": cutoff_freqs[2, :],
            })
        else:
            raise ValueError("cutoff_freqs must be 1D or 2D")

        cutoff_df.to_csv(
            os.path.join(iteration_path, "cutoffFrequencies"),
            sep="\t",
            index=False,
            float_format="%.12e",
        )

    # Optional convenience bundle for Python reloading
    savez_dict = {
        "z_array": z_array,
        "freq_array": freq_array,
        "current_spectra_array": current_spectra_array,
        "downstream_spectra_array": downstream_spectra_array,
    }

    if new_inlet_spectra_array is not None:
        savez_dict["new_inlet_spectra_array"] = new_inlet_spectra_array
    if cutoff_freqs is not None:
        savez_dict["cutoff_freqs"] = cutoff_freqs

    np.savez_compressed(
        os.path.join(iteration_path, "spectra_arrays.npz"),
        **savez_dict,
    )
    

#%%

def write_cospectral_spectra_profile(
    spectra_array,
    z_array,
    uw_stress_array,
    filepath,
    clip_min=1e-16,
):
    """
    Write augmented coSpectralDFSR spectra profile:

        nHeights nFreq
        z uwStress Su[0:nFreq] Sv[0:nFreq] Sw[0:nFreq]

    spectra_array shape: (3, nHeights, nFreq)
    """
    z_array = np.asarray(z_array, dtype=float).reshape(-1)
    uw_stress_array = np.asarray(uw_stress_array, dtype=float).reshape(-1)
    spectra_array = np.asarray(spectra_array, dtype=float)

    if spectra_array.ndim != 3 or spectra_array.shape[0] != 3:
        raise ValueError("spectra_array must have shape (3, nHeights, nFreq).")

    _, n_heights, n_freq = spectra_array.shape

    if len(z_array) != n_heights:
        raise ValueError("z_array length does not match spectra height dimension.")
    if len(uw_stress_array) != n_heights:
        raise ValueError("uw_stress_array length does not match nHeights.")

    spectra_clean = np.nan_to_num(
        spectra_array.copy(),
        nan=clip_min,
        posinf=clip_min,
        neginf=clip_min,
    )
    spectra_clean = np.maximum(spectra_clean, clip_min)

    with open(filepath, "w") as f:
        f.write(f"{n_heights} {n_freq}\n")

        for h in range(n_heights):
            row = np.concatenate(
                [
                    [z_array[h], uw_stress_array[h]],
                    spectra_clean[0, h, :],
                    spectra_clean[1, h, :],
                    spectra_clean[2, h, :],
                ]
            )
            f.write(" ".join(f"{val:.12e}" for val in row) + "\n")
            

#%%

def write_uw_cospectrum_profile(
    c_uw_array,
    z_array,
    uw_stress_array,
    filepath,
):
    """
    Write uw co-spectrum profile:

        nHeights nFreq
        z uwStress Cuw[0:nFreq]

    c_uw_array shape: (nHeights, nFreq)
    """
    z_array = np.asarray(z_array, dtype=float).reshape(-1)
    uw_stress_array = np.asarray(uw_stress_array, dtype=float).reshape(-1)
    c_uw_array = np.asarray(c_uw_array, dtype=float)

    if c_uw_array.ndim != 2:
        raise ValueError("c_uw_array must have shape (nHeights, nFreq).")

    n_heights, n_freq = c_uw_array.shape

    if len(z_array) != n_heights:
        raise ValueError("z_array length does not match c_uw_array.")
    if len(uw_stress_array) != n_heights:
        raise ValueError("uw_stress_array length does not match nHeights.")

    c_clean = np.nan_to_num(
        c_uw_array.copy(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    with open(filepath, "w") as f:
        f.write(f"{n_heights} {n_freq}\n")

        for h in range(n_heights):
            row = np.concatenate(
                [
                    [z_array[h], uw_stress_array[h]],
                    c_clean[h, :],
                ]
            )
            f.write(" ".join(f"{val:.12e}" for val in row) + "\n")
            

#%%

