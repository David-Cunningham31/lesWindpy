# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 12:25:54 2026

@author: David Cunningham
"""

import logging
import numpy as np
import pandas as pd
import json
import os
import sys
import shutil
import matplotlib.pyplot as plt

cwd = os.path.dirname(os.path.abspath(__file__))
windlespy_path = os.path.abspath(os.path.join(cwd, "..", ".."))
sys.path.append(windlespy_path)
import windlespy as LES
sys.path.remove(windlespy_path)

#%%
case_path = r"C:\Users\david\OneDrive\Documents\PhD\Year 1\Spectral Calibration Method"
downstream_probes_folder = os.path.join(case_path, "postProcessing", "probes2")
#%%
variable_dict = LES._caseFiles.parse_setup_file(case_path)

building_height = variable_dict['buildingHeight']
lower_z_threshold = variable_dict['lowerZThreshold']
upper_z_thresold = variable_dict['upperZThreshold']
rmse_threshold = variable_dict['rmseThreshold']
mesh_size = variable_dict['meshSize']
fMax = variable_dict['fMax']
nFreq = variable_dict['nFreq']

json_path = os.path.join(case_path, "log", "downstreamCalibration", "sim_init.json")

with open(json_path, "r") as f:
    dfsr_les_init_dict = json.load(f)

burn_in_time = dfsr_les_init_dict["burn_in_time"]

#%%

target_profile_df = LES._profileCalibration.get_dfsr_target_profile_df(case_path)

target_profile_array = LES._profileCalibration.get_dfsr_target_profile_array(case_path)

inlet_profile_array = LES._profileCalibration.get_current_dfsr_inlet_profile_array(case_path)

vel_array_3d = LES._profileAnalysis.get_velocity_components(downstream_probes_folder)

time_steps = LES._profileAnalysis.get_time_steps_probe_data(downstream_probes_folder)

mask = time_steps > burn_in_time

vel_array_3d = vel_array_3d[:, mask, :]

time_steps = time_steps[mask]

time_step = np.mean(np.diff(time_steps))

downstream_profile_array = LES._profileCalibration.get_downstream_dfsr_profile_array(vel_array_3d, time_step, inlet_or_downstream="downstream", burn_in_time=burn_in_time, time_steps=time_steps)

int_length_scales = downstream_profile_array[:,-3:]

U = downstream_profile_array[:,0]

#%%

lower_z_threshold_id, upper_z_threshold_id = LES._profileCalibration.get_avg_z_thresolds_ids(target_profile_df, lower_z_threshold, upper_z_thresold)

rmse_array = LES._profileCalibration.get_rmse(downstream_profile_array, target_profile_array, lower_z_threshold_id, upper_z_threshold_id)

#%%

iter_status = LES._profileCalibration.dfsr_iter_status(case_path, rmse_array, rmse_threshold, "downstream")

LES._caseFiles.write_dfsr_iter_json(case_path, iter_status, "downstream")

#%%

iteration = iter_status['iteration']
fig_folder = os.path.join(case_path,"log", "downstreamCalibration", f"iteration{iteration}","plots", "profiles")
os.makedirs(fig_folder, exist_ok=True)

height_mask = (target_profile_df["z"]<=(3*building_height)).to_numpy()
norm_heights = target_profile_df["z"].to_numpy()/building_height
norm_heights = norm_heights[height_mask]

for col_index, x_axis_desc in enumerate(target_profile_df.columns[1:]):
    profile_list = []
    plot_descs=[]
    
    if "I" in x_axis_desc:
        downstream_profile = np.sqrt(downstream_profile_array[height_mask, col_index])/downstream_profile_array[height_mask, 0]
        
        target_profile = np.sqrt(target_profile_array[height_mask, col_index])/target_profile_array[height_mask, 0]
    else:
        downstream_profile = downstream_profile_array[height_mask, col_index]
        
        target_profile = target_profile_array[height_mask, col_index]
        
    profile_list.append(downstream_profile)
    plot_descs.append("Downstream Profile")
    
    profile_list.append(target_profile)
    plot_descs.append("Target Profile")
    
    profiles_array = np.stack(profile_list, axis=0)
    
    fig = LES._plot.plot_profile(profiles_array, norm_heights, x_axis_desc, "z/H", xlims=None, ylims=None, several=True, descs=plot_descs)
        
    filename = f"{x_axis_desc}_profiles.png"
    fig.savefig(os.path.join(fig_folder, filename), dpi=300, bbox_inches="tight")
    
    plt.close(fig)

#%%

converged = iter_status["converged"]
stagnated = iter_status["stagnated"]

#%%    
freq_array = LES._profileCalibration.get_freq_array(fMax, nFreq)

#%%

target_spectra_array = LES._profileCalibration.get_vk_spectra_array(fMax, nFreq, target_profile_array)
inlet_spectra_array = LES._profileCalibration.get_vk_spectra_array(fMax, nFreq, inlet_profile_array)
downstream_spectra_array = LES._profileCalibration.get_downstream_spectra_array(
    fMax,
    nFreq,
    vel_array_3d,
    time_step,
    building_height,
    downstream_profile_array,
    inlet_or_downstream="downstream",
    burn_in_time=burn_in_time,
    time_steps=time_steps
)

#%%
smooth_downstream_spectra_array = LES._profileCalibration.smooth_downstream_spectra_array(freq_array, downstream_spectra_array)
#%%
U = target_profile_array[:,0]
int_length_scales = target_profile_array[:,-3:].transpose()
sigmas = np.sqrt(target_profile_array[:,1:4]).transpose()
mesh_cutoff_freqs = LES._profileAnalysis.get_mesh_cutoff_frequencies(mesh_size, U, int_length_scales, sigmas)
mesh_cutoff_freqs_3d = np.broadcast_to(mesh_cutoff_freqs[np.newaxis, :], (3, len(mesh_cutoff_freqs)))

#%%
smooth_downstream_spectra_array = LES._profileCalibration.apply_power_law_tail(freq_array, smooth_downstream_spectra_array, mesh_cutoff_freqs_3d*0.5, slope=-5/3, floor=1e-20)

#%%
conv_spectral_func = LES._profileCalibration.get_convective_spectral_function(inlet_spectra_array, smooth_downstream_spectra_array)
updated_spectra_array = LES._profileCalibration.get_updated_spectra_array(target_spectra_array, conv_spectral_func, inlet_spectra_array, smooth_downstream_spectra_array)

#%%
z_array = target_profile_df["z"]

fig_dir = os.path.join(case_path,"log", "downstreamCalibration", f"iteration{iteration}","plots", "spectra")
os.makedirs(fig_dir, exist_ok=True)

LES._plot.plot_spectral_calibration(
    fig_dir,
    z_array,
    freq_array,
    downstream_spectra_array,
    updated_spectra_array,
    inlet_spectra_array,
    target_spectra_array,
    cutoff_freqs=None,          # shape (3, nHeights) or None
    z_min=None,
    z_max=None,
)

#%%

if (not converged) and (not stagnated):
    dfsr_input_spectra_path = os.path.join(case_path, "constant", "boundaryData", "windProfile", "spectraProfile")
    LES._caseFiles.write_spectra_profile(updated_spectra_array, z_array, dfsr_input_spectra_path, clip_min=1e-16)
           
    LES._caseFiles.write_dfsr_iter_spectra(
        case_path,
        iter_status,
        z_array,
        freq_array,
        inlet_spectra_array,
        downstream_spectra_array,
        inlet_or_downstream="downstream",
        new_inlet_spectra_array=updated_spectra_array,
        cutoff_freqs=None,
        clip_min=1e-16,
    )
    
else:
    
    LES._caseFiles.write_dfsr_iter_spectra(
        case_path,
        iter_status,
        z_array,
        freq_array,
        inlet_spectra_array,
        downstream_spectra_array,
        inlet_or_downstream="downstream",
        new_inlet_spectra_array=None,
        cutoff_freqs=None,
        clip_min=1e-16,
    )
    
#%%

if converged:
    sys.exit(0)
elif stagnated:
    sys.exit(0)
else:
    sys.exit(1)