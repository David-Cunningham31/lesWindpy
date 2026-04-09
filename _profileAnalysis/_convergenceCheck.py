# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:57:11 2026

Module for checking whether the turbulence statistics from an OpenFOAM simulation have converged.

@author: David Cunningham
"""

from scipy.optimize import curve_fit
import logging
import numpy as np
import pandas as pd
import re
import os

logger = logging.getLogger(__name__)

#%%

def fit_log_law(U, z, zmin = None, zmax = None):  
    """
    This function is intended to fit a log-law profile from a velocity and height series.

    Parameters
    ----------
    U : Array of mean wind speed values
    z : Array of heights
    zmin : A minimum height from which to fit the log law profile.
    zmax : A maximum height to which to fit the log law profile - log-law tends to be inaccurate for z>200 m in ABL.

    Returns
    -------
    TYPE: float
        u_star - friction velocity value
    TYPE: float
        DESCRIPTION: a z_0 value - surface roughness length.

    """
        
    kappa = 0.41
    
    def loglaw(z, u_star, z0):
        return (u_star/kappa) * np.log((z + z0)/z0)
    
    if (zmin != None) and (zmax != None):
        mask = np.isfinite(z) & np.isfinite(U) & (z > zmin) & (z < zmax)
    else:
        mask = np.isfinite(z) & np.isfinite(U) & (z > 0)
        
        
    zf, Uf = z[mask], U[mask]
    
    p0 = (1, 0.1)  # initial guesses (u*, z0)
    (u_star_fit, z0_fit), cov = curve_fit(loglaw, zf, Uf, p0=p0, bounds=(0, np.inf))
    
    return u_star_fit, z0_fit
    
#%%

def calculate_t_star(u_star, domain_height):
    """
    Function calculates t_star - time for assessing convergence of Reynolds stresses.

    Parameters
    ----------
    u_star : Friction velocity
    domain_height : Height of LES domain

    Returns
    -------
    t_star

    """
    return (domain_height*0.5) / u_star

#%%

def split_vel_to_blocks(time_steps, vel_array_3d, block_length):
    """
    
    Function splits velocity time series into specified lengths of time.

    Parameters
    ----------
    time_steps : Array of time steps from simulation.
    vel_array_3d : 3D velocity array.
    block_length : Length of time to bin the velocity values - Lamberti et. al (2018) suggest 4t_star

    Returns
    -------
    4D velocity array of shape(3, no. of time blocks, no. of time-steps per block, no. of probes)
    An array of the end-times for each block - can be used for plotting the convergence of Re stresses later.
    """
    
    # An array putting each time step into bins based on time chosen to check convergence of statistics:
    block_from_end = np.floor((time_steps.max() - time_steps) / (block_length)).astype(int)
    block = block_from_end.max() - block_from_end
    
    # Finding the different block numbers or time step categories and the number of time steps in each block (counts)
    blocks, counts = np.unique(block, return_counts=True)
    
    block_end_times = np.array([time_steps[block == b][-1] for b in np.unique(block)])
    
    # The number of blocks, maximum number of rows in a block and the number of columns (or probes):
    n_blocks = len(blocks)
    max_rows = counts.max()
    n_cols = np.shape(vel_array_3d)[-1]
    
    # Initializing variables to contain a list of 3D velocity arrays that we will turn into 4D arrays:
    vel_array_3d_list=[]
    
    # Iterating through each of the velocity vector component arrays:
    for vel_array in vel_array_3d:
        # Creating a 3D array from the 2D array based on the number of time blocks:
        vel_array_3d = np.full((n_blocks, max_rows, n_cols), np.nan)
    
        # Iterating through each of the time blocks and the number of time steps (c) in each block to create a 3D array (extra dimension being number of blocks):
        start = 0
        for i, c in enumerate(counts):
            vel_array_3d[i, :c, :] = vel_array[start:start+c]
            start += c
    
        # Adding the 3D velocity array to a list so that we can create a 4D array later:
        vel_array_3d_list.append(vel_array_3d)
    
    vel_array_4d = np.stack(vel_array_3d_list)
    
    return vel_array_4d, block_end_times


#%%

def mean_block_vel(vel_array_4d):
    """
    Function averages the velocity values over each time block.

    Parameters
    ----------
    vel_array_4d : 4d velocity array split into time blocks.

    Returns
    -------
    3d velocity array containing the averaged velocities over each time block.

    """
    
    # Finding the mean velocities for each time block:
    mean_vel_array_3d = np.nanmean(vel_array_4d, axis=2)
    
    return mean_vel_array_3d

#%%

def fluc_block_vel(vel_array_4d, mean_vel_array_3d):
    """
    

    Parameters
    ----------
    vel_array_4d : TYPE
        DESCRIPTION.
    mean_vel_array_3d : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    fluc_vel_array_4d = vel_array_4d - mean_vel_array_3d[:, :, None, :]
    
    return fluc_vel_array_4d

#%%

def TI_block(vel_array_4d, mean_vel_array_3d):
    """
    

    Parameters
    ----------
    vel_array_4d : TYPE
        DESCRIPTION.
    mean_vel_array_3d : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    # Finding the std for each time block:
    std_array_3d = np.nanstd(vel_array_4d, axis=2)
    
    # Finding TI for each component of velocity:
    TI_array_3d = std_array_3d / mean_vel_array_3d[0]
    
    return TI_array_3d

#%%

def re_stresses_block(fluc_vel_array_4d):
    """
    

    Parameters
    ----------
    fluc_vel_array_4d : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    # Calculating Reynolds Stresses:
    R_11 = np.nanmean((fluc_vel_array_4d[0] * fluc_vel_array_4d[0]), axis=1)
    R_22 = np.nanmean((fluc_vel_array_4d[1] * fluc_vel_array_4d[1]), axis=1)
    R_33 = np.nanmean((fluc_vel_array_4d[2] * fluc_vel_array_4d[2]), axis=1)
    R_31 = np.nanmean((fluc_vel_array_4d[0] * fluc_vel_array_4d[2]), axis=1)
    
    re_stresses = np.stack([R_11, R_22, R_33, R_31])
    
    return re_stresses

    
#%%

def max_percent_block_errors(re_stresses, probe_id):
    """
    Compute percentage difference between consecutive blocks at a particular height.
    Array must be 2D: (num_of_blocks, num_of_probes)
    

    Parameters
    ----------
    array : TYPE
        DESCRIPTION.

    Returns
    -------
    An array of length (number of Re stresses, number of time blocks - 1), containing the percentage error between time blocks.
    The maximum error among all Re stresses for the latest time block - governing parameter for convergence.

    """
    
    errs_array = np.full((np.shape(re_stresses)[0], np.shape(re_stresses)[1] -1), np.nan)
    latest_errors = []
    
    for index, re_stress in enumerate(re_stresses):
    
        re_stress = re_stress[:,probe_id]
        numerator = np.abs(re_stress[1:] - re_stress[:-1])
        denominator = np.maximum(np.abs(re_stress[1:]), 1e-12)
        errs = 100.0 * numerator / denominator
        
        errs_array[index,:] = errs
        latest_errors.append(errs[-1])
        
    max_err = max(latest_errors)
    
    return errs_array, max_err

#%%

def can_LES_finish(time_steps, max_re_stress_err, initializing_time, stat_avg_time):
    """
    

    Parameters
    ----------
    time_steps : array of time-steps
    max_re_stress_err : maximum error between time blocks for the Re stresses at a particular height.
    initializing_time : Initialization time for simulation (portion discarded): 12.5t_star to 20t_star (Lamberti et. al 2018, Kim et. al 2013)
    stat_avg_time : Minimum time to average the statistics: 40t_star - (Lamberti et. al 2018, Kim et. al 2013)

    Returns
    -------
    bool
        DESCRIPTION.

    """
    
    sim_duration = time_steps[-1] - time_steps[0]
    
    if (max_re_stress_err <= 3) and (sim_duration >= ( (initializing_time) + (stat_avg_time) ) ):
        return True
    else:
        return False

#%%

def get_cfl_df(case_path):
    
    logfile = os.path.join(case_path,"pisFoam_burn_in.log")

    time_pattern = re.compile(r"^Time =\s+([0-9Ee+\-\.]+)")
    co_pattern = re.compile(
        r"Courant Number mean:\s*([0-9Ee+\-\.]+)\s*max:\s*([0-9Ee+\-\.]+)"
    )

    rows = []
    current_time = None

    with open(logfile, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            m_time = time_pattern.match(line)
            if m_time:
                current_time = float(m_time.group(1))
                continue

            m_co = co_pattern.search(line)
            if m_co and current_time is not None:
                co_mean = float(m_co.group(1))
                co_max = float(m_co.group(2))
                rows.append((current_time, co_mean, co_max))

    cfl_df = pd.DataFrame(rows, columns=["time", "Co_mean", "Co_max"])
    cfl_df["deltaT"] = cfl_df["time"].diff()

    return cfl_df

#%%

def get_cfl_time_step_dict(cfl_df, cfl_percentile, deltaT_percentile):

    deltaT_10th_percentile = cfl_df["deltaT"].quantile(0.10)
    deltaT_5th_percentile = cfl_df["deltaT"].quantile(0.05)
    deltaT_median = cfl_df["deltaT"].median()
    deltaT_sim = cfl_df["deltaT"].quantile(deltaT_percentile/100)
    
    max_cfl_mean = cfl_df["Co_max"].mean()
    max_cfl_95th_percentile = cfl_df["Co_max"].quantile(0.95)
    max_cfl_max = cfl_df["Co_max"].quantile(0.95)
    cfl_sim = cfl_df["Co_max"].quantile(cfl_percentile/100)
    
    cfl_time_step_dict = {
        "deltaT_10th_percentile": deltaT_10th_percentile,
        "deltaT_5th_percentile": deltaT_5th_percentile,
        "deltaT_median": deltaT_median,
        "deltaT_sim": deltaT_sim,
        "max_cfl_mean": max_cfl_mean,
        "max_cfl_95th_percentile": max_cfl_95th_percentile,
        "max_cfl_max": max_cfl_max,
        "cfl_sim": cfl_sim,
        }
    
    return cfl_time_step_dict
