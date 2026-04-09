# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:53:34 2026

Module to calculate wind profiles obtained from OpenFOAM simulations.

@author: David Cunningham
"""

import logging
import numpy as np
from scipy.signal import correlate
from scipy.signal import welch
from scipy.optimize import root_scalar
import os
import sys

logger = logging.getLogger(__name__)

cwd = os.path.dirname(os.path.abspath(__file__))
windlespy_path = os.path.abspath(os.path.join(cwd, "..", ".."))
sys.path.append(windlespy_path)
import windlespy as LES
sys.path.remove(windlespy_path)

#%%

def t_avg_start(min_avg_period, time_steps):
    """
    Function returns the start time step from which the final turbulent stats will be calculated.

    Parameters
    ----------
    min_avg_period : This is the averaging period for which to calculate the turbulent statistics - minimum of 40t_star - Lamberti et. al 2018 & Kim et. al 2013
    time_steps : 1D Numpy array of time steps.

    Returns
    -------
    t_cutoff : The time-step from which calculating of turbulence statistics should start.

    """
    
    t_start = time_steps[-1] - (min_avg_period)

    t_diff = time_steps - t_start
    
    t_cutoff = time_steps[t_diff==max(t_diff[t_diff<=0])][0]
    
    return t_cutoff
    
#%%

def vel_avg_window(t_cutoff, time_steps, vel_array_3d):
    """

    Parameters
    ----------
    t_cutoff : Time step from which turbulence statistics are calculated.
    time_steps : 1D array of time steps.
    vel_array_3d : 3D velocity array from simulation.

    Returns
    -------
    vel_array_3d : 3D velocity array containing the values from the time-steps for turbulence statistic calculations only.

    """
    mask = time_steps >= t_cutoff
    vel_array_3d = vel_array_3d[:, mask, :]
    
    return vel_array_3d


#%%

def mean_vel(vel_array_3d):
    """
    

    Parameters
    ----------
    vel_array_3d : 3D Numpy velocity array containing only time steps for turbulent statistics calculation.

    Returns
    -------
    2D Numpy array of mean velocities.

    """
    
    mean_vel_array_2d = np.mean(vel_array_3d, axis=1)
    
    return mean_vel_array_2d


#%%

def fluc_vel(vel_array_3d, mean_vel_array_2d):
    """

    Parameters
    ----------
    vel_array_3d : 3D Numpy array of velocities.
    mean_vel_array_2d : 2D Numpy array of mean velocities.

    Returns
    -------
    fluc_vel_array_3d : 3D Numpy array of fluctuating components of velocity.

    """
    
    fluc_vel_array_3d = vel_array_3d - mean_vel_array_2d[:, None, :]
    
    return fluc_vel_array_3d
    
#%%

def turb_int(vel_array_3d, mean_vel_array_2d):
    """
    

    Parameters
    ----------
    vel_array_3d : 3D Numpy array of instantaneous velocity values.
    mean_vel_array_2d : 2D Numpy array of mean velocity values.

    Returns
    -------
    ti_array_3d : 3D Numpy array of turbulence intensities. 

    """
    
    # Finding the std for each time block:
    std_array_2d = np.std(vel_array_3d, axis=1)
    
    # Finding TI for each component of velocity:
    ti_array_3d = std_array_2d / mean_vel_array_2d[0]
    
    return ti_array_3d

#%%

def re_stresses(fluc_vel_array_3d):
    
    R_11 = np.mean((fluc_vel_array_3d[0] * fluc_vel_array_3d[0]), axis=0)
    R_22 = np.mean((fluc_vel_array_3d[1] * fluc_vel_array_3d[1]), axis=0)
    R_33 = np.mean((fluc_vel_array_3d[2] * fluc_vel_array_3d[2]), axis=0)
    R_31 = np.mean((fluc_vel_array_3d[0] * fluc_vel_array_3d[2]), axis=0)
    
    re_stresses = np.stack([R_11, R_22, R_33, R_31])
    
    return re_stresses

#%%

def int_time_scale(fluc_vel_array_3d, time_step):
    """
    

    Parameters
    ----------
    fluc_vel_array_3d : 3D Numpy array of shape (3, no. time steps, no. probes)
    time_step : a 1d array of the time steps in the simulation.

    Returns
    -------
    int_time_scales : 2D Numpy array (3, no. probes)

    """

    int_time_scales = np.full((3,np.shape(fluc_vel_array_3d[0])[1]), np.nan)
    
    for vel_comp, array in zip(["u","v","w"],fluc_vel_array_3d):
        
        for probe in range(0,np.shape(array)[1]):
    
            vel_time_series = array[:,probe]
            vel_time_series = vel_time_series[np.isfinite(vel_time_series)]
        
            n = len(vel_time_series)
            
            # Full autocorrelation
            acf_full = correlate(vel_time_series, vel_time_series, mode='full')
            acf = acf_full[n-1:]   # keep non-negative lags only
            
            # Unbiased normalization by number of overlapping samples
            lags = np.arange(n)
            acf = acf / (n - lags)
            
            # Normalize so rho(0) = 1
            rho = acf / acf[0]
            
            tau = lags * time_step
            
            # Find first zero crossing
            zero_crossings = np.where(rho <= 0)[0]
            if len(zero_crossings) > 0:
                i_end = zero_crossings[0]
            else:
                i_end = len(rho) - 1
            
            # Integrate autocorrelation
            T_int = np.trapezoid(rho[:i_end+1], tau[:i_end+1])
    
            int_time_scales[ ["u","v","w"].index(vel_comp), probe ] = T_int
        
    return int_time_scales


#%%

def int_length_scales(int_time_scales, mean_vel_array_2d):
    
    # Calculate integral length scales using Taylor's Hypothesis: L=TU:
    
    int_length_scales = int_time_scales * mean_vel_array_2d[0,:]
    
    return int_length_scales

#%%

def welch_psd(vel_time_series, fs, body_dim, U_ref, nperseg=None, noverlap=None, detrend="constant", window="hann"):
    """
    

    Parameters
    ----------
    vel_time_series : 1D velocity time series
    fs : Sampling frequency: 1 / time step
    body_dim : Body dimension of structure for calculating the reduced frequency.
    U_ref : Reference wind speed at the height the spectrum is calculated.
    nperseg : TYPE, optional
        DESCRIPTION. The default is None.
    noverlap : TYPE, optional
        DESCRIPTION. The default is None.
    detrend : TYPE, optional
        DESCRIPTION. The default is "constant".
    window : TYPE, optional
        DESCRIPTION. The default is "hann".

    Returns
    -------
    f : a 1D array of frequencies of the spectrum.
    S : 1D array containing the power spectrum values.
    reduced_f : 1D array of reduced frequencies.
    norm_power_spectrum : 1D array of normalised power spectrum values.

    """

    vel_time_series = vel_time_series[np.isfinite(vel_time_series)]
    sigma2 = np.var(vel_time_series, ddof=0)
    
    f, S = welch(
        vel_time_series,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        scaling="density",
        return_onesided=True,
    )
    
    reduced_f = (f * body_dim) / U_ref
    norm_power_spectrum = (reduced_f * S) / sigma2
    
    return f, S, reduced_f, norm_power_spectrum


#%%

def von_karman_spectra(red_fs, vel_comp):
    """
    

    Parameters
    ----------
    red_fs : 1D array of reduced frequencies
    vel_comp : TYPE
        DESCRIPTION.

    Returns
    -------
    von_karman_spectrum : TYPE
        DESCRIPTION.

    """
    
    if vel_comp=="u":
        
        numerator = 4*red_fs
        denominator = ( ( 1 + 70.8*(red_fs**2) ) )**(5/6)
        von_karman_spectrum = numerator / denominator
        
        return von_karman_spectrum
        
    elif vel_comp=="v":

        numerator = 4*red_fs*( 1 + 755.2*(red_fs**2) )
        denominator = ( ( 1 + 283.2*(red_fs**2) ) )**(11/6)
        von_karman_spectrum = numerator / denominator
        
        return von_karman_spectrum

    elif vel_comp=="w":

        numerator = 4*red_fs*( 1 + 755.2*(red_fs**2) )
        denominator = ( ( 1 + 283.2*(red_fs**2) ) )**(11/6)
        von_karman_spectrum = numerator / denominator
        
        return von_karman_spectrum
    
#%%

def target_profile_re_stresses(target_profile_df):
    
    target_profile_df["R_11"] = (target_profile_df["Iu"] * target_profile_df["U"])**2
    target_profile_df["R_22"] = (target_profile_df["Iv"] * target_profile_df["U"])**2
    target_profile_df["R_33"] = (target_profile_df["Iw"] * target_profile_df["U"])**2

    return target_profile_df

#%%

def get_n_c(delta, frequency_energy):
    n_c = (np.pi * frequency_energy)/(delta**2)
    
    return n_c

#%%

def get_von_karman_red_fs(n_c, int_length_scales, U):
    von_karman_red_fs = (n_c*int_length_scales)/U
    
    return von_karman_red_fs

#%%

def get_frequency_energy(n_c, L, U_z, sigmas_z):
    von_karman_red_fs = get_von_karman_red_fs(n_c, L, U_z)
    
    von_karman_spectra_arrays = []
    
    for red_fs, vel_comp, sigma in zip(von_karman_red_fs,["u","v","w"], sigmas_z):
        non_dim_von_karman_spectrum = LES._profileAnalysis.von_karman_spectra(red_fs, vel_comp)
        von_karman_spectrum = (non_dim_von_karman_spectrum * (sigma**2)) / n_c
        von_karman_spectra_arrays.append(von_karman_spectrum)
        
    von_karman_spectra = np.stack(von_karman_spectra_arrays, axis = 0)
    
    frequency_energy = np.sum(von_karman_spectra)

    return frequency_energy

#%%

def get_delta(frequency_energy, n_c):
    delta = np.sqrt((np.pi / n_c) * frequency_energy)
    
    return delta

#%%

def residual(n_c, delta, L, U_z, sigmas_z):
    
    frequency_energy = get_frequency_energy(n_c, L, U_z, sigmas_z)
    
    return ((delta**2) * n_c) - (np.pi * frequency_energy)

#%%

def get_mesh_cutoff_frequencies(delta, z_array, U, int_length_scales, sigmas):
    
    n_c = np.full(len(z_array), np.nan)
    
    for i in range(len(z_array)):
        
        U_z = U[i]
        
        L = int_length_scales[:,i]
        
        sigmas_z = sigmas[:,i]
        
        a = 1e-6
        b = 1e6
        
        sol = root_scalar(
            residual,
            args=(delta, L, U_z, sigmas_z),
            bracket=[a, b],
            method="brentq"
        )
        
        n_c[i] = sol.root
    
    return n_c