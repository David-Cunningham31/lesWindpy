# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 17:32:56 2026

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
# USER INPUTS:

case_path = r"/home/people/20397873/LES/NHERI_Tall_Building/empty_domain_2"
downstream_probes_folder = os.path.join(case_path, "postProcessing", "probes2")

#%%
variable_dict = LES._caseFiles.parse_setup_file(case_path)

building_height = variable_dict['buildingHeight']
lower_z_threshold = variable_dict['lowerZThreshold']
upper_z_thresold = variable_dict['upperZThreshold']
rmse_threshold = variable_dict['rmseThreshold']

json_path = os.path.join(case_path, "log", "downstreamCalibration", "sim_init.json")

with open(json_path, "r") as f:
    dfsr_les_init_dict = json.load(f)

burn_in_time = dfsr_les_init_dict["burn_in_time"]

#%%

target_profile_df = LES._profileCalibration.get_dfsr_target_profile_df(case_path)

z_array = target_profile_df["z"].to_numpy()

target_profile_array = LES._profileCalibration.get_dfsr_target_profile_array(case_path)

current_profile_array = LES._profileCalibration.get_current_dfsr_inlet_profile_array(case_path)

vel_array_3d = LES._profileAnalysis.get_velocity_components(downstream_probes_folder)

time_steps = LES._profileAnalysis.get_time_steps_probe_data(downstream_probes_folder)

time_step = np.mean(np.diff(time_steps))

downstream_profile_array = LES._profileCalibration.get_downstream_dfsr_profile_array(vel_array_3d, time_step, inlet_or_downstream="downstream", burn_in_time=burn_in_time, time_steps=time_steps)

#%%

lower_z_threshold_id, upper_z_threshold_id = LES._profileCalibration.get_avg_z_thresolds_ids(target_profile_df, lower_z_threshold, upper_z_thresold)

rmse_array = LES._profileCalibration.get_rmse(downstream_profile_array, target_profile_array, lower_z_threshold_id, upper_z_threshold_id)

#%%

iter_status = LES._profileCalibration.dfsr_iter_status(case_path, rmse_array, rmse_threshold, "downstream")

LES._caseFiles.write_dfsr_iter_json(case_path, iter_status, "downstream")


#%%

iteration = iter_status['iteration']
fig_folder = os.path.join(case_path,"log", "downstreamCalibration", f"iteration{iteration}","plots")
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

if (not converged) and (not stagnated): 
    
    new_inlet_profile_array = LES._profileCalibration.new_dfsr_profile_array(current_profile_array, target_profile_array, downstream_profile_array, relaxation_factor=0.9)
    
    smoothed_new_profiles = LES._profileAnalysis.smooth_profiles(new_inlet_profile_array, z_array, 3, 3, building_height)
    
    LES._caseFiles.write_new_dfsr_inlet_profile(smoothed_new_profiles, target_profile_df, case_path)
    
    LES._caseFiles.write_dfsr_inlet_iter_profiles(case_path, iter_status, target_profile_df, current_profile_array, downstream_profile_array, "downstream", smoothed_new_profiles)

else:
    
    LES._caseFiles.write_dfsr_inlet_iter_profiles(case_path, iter_status, target_profile_df, current_profile_array, downstream_profile_array, inlet_or_downstream="downstream")

#%%

if converged:
    sys.exit(0)
elif stagnated:
    sys.exit(0)
else:
    sys.exit(1)