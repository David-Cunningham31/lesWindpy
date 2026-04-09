# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:40:12 2026

@author: David Cunningham
"""

import logging
import numpy as np
import pandas as pd
import os
import sys
import shutil
import json

cwd = os.path.dirname(os.path.abspath(__file__))
windlespy_path = os.path.abspath(os.path.join(cwd, "..", ".."))
sys.path.append(windlespy_path)
import windlespy as LES
sys.path.remove(windlespy_path)


#%%

def get_iter_cal_profiles(case_path, inlet_or_downstream="inlet"):
    
    iter_profile_dict = {}
    
    if inlet_or_downstream=="inlet":
        iter_profiles_folder = os.path.join(case_path,"log", "inletCalibration")
    else:
        iter_profiles_folder = os.path.join(case_path,"log", "downstreamCalibration")
    
    for iter_desc in os.listdir(iter_profiles_folder):
        
        iter_profile_dict[iter_desc] = {}
        
        inlet_profile_path = os.path.join(iter_profiles_folder, iter_desc, "inletProfile")
        post_corr_inlet_profile_path = os.path.join(iter_profiles_folder, iter_desc, "postCorrectionProfile")
        new_inlet_profile_path = os.path.join(iter_profiles_folder, iter_desc, "newInletProfile")
        
        columns=["z", "U", "Iu", "Iv", "Iw", "Lu", "Lv", "Lw"]
        
        inlet_profile_df=pd.read_csv(inlet_profile_path, sep=r"\s+", header=None)
        inlet_profile_df.columns = columns
        post_corr_inlet_profile_df=pd.read_csv(post_corr_inlet_profile_path, sep=r"\s+", header=None)
        post_corr_inlet_profile_df.columns = columns
        if "newInletProfile" in os.listdir(os.path.join(iter_profiles_folder, iter_desc)):
            new_inlet_profile_df=pd.read_csv(new_inlet_profile_path, sep=r"\s+", header=None)
            new_inlet_profile_df.columns = columns
            iter_profile_dict[iter_desc]["new_inlet_profile"] = new_inlet_profile_df
        else:
            iter_profile_dict[iter_desc]["new_inlet_profile"] = "NA"
        
        iter_profile_dict[iter_desc]["inlet_profile"] = inlet_profile_df
        iter_profile_dict[iter_desc]["post_corr_profile"] = post_corr_inlet_profile_df
        
    return iter_profile_dict
        
#%%

def get_dfsr_target_profile_df(case_path):
    
    profile_path = os.path.join(case_path,'constant','boundaryData','windProfile')

    folder_list = os.listdir(profile_path)
    
    if "targetProfile" not in folder_list:
        shutil.copy(os.path.join(profile_path,'profile'), os.path.join(profile_path,'targetProfile'))
        
    target_profile_path = os.path.join(profile_path,'targetProfile')
    
    target_profile_df=pd.read_csv(target_profile_path, sep=r"\s+", header=None)
    
    columns=["z", "U", "Iu", "Iv", "Iw", "Lu", "Lv", "Lw"]
    target_profile_df.columns = columns
    
    return target_profile_df

#%%

def get_dfsr_target_profile_array(case_path):
    
    profile_path = os.path.join(case_path,'constant','boundaryData','windProfile')

    folder_list = os.listdir(profile_path)
    
    if "targetProfile" not in folder_list:
        shutil.copy(os.path.join(profile_path,'profile'), os.path.join(profile_path,'targetProfile'))
        
    target_profile_path = os.path.join(profile_path,'targetProfile')
    
    target_profile_df=pd.read_csv(target_profile_path, sep=r"\s+", header=None)
    
    columns = ["z", "U", "Iu", "Iv", "Iw", "Lu", "Lv", "Lw"]
    target_profile_df.columns = columns
    
    target_profile_df["R_11"] = (target_profile_df["Iu"] * target_profile_df["U"])**2
    target_profile_df["R_22"] = (target_profile_df["Iv"] * target_profile_df["U"])**2
    target_profile_df["R_33"] = (target_profile_df["Iw"] * target_profile_df["U"])**2
    
    target_profile_df = target_profile_df[["U", "R_11", "R_22", "R_33", "Lu", "Lv", "Lw"]]

    target_profile_array = target_profile_df.to_numpy()
    
    return target_profile_array

#%%

def get_current_dfsr_inlet_profile_array(case_path):
    
    profile_path = os.path.join(case_path,'constant','boundaryData','windProfile')
        
    current_profile_path = os.path.join(profile_path,'profile')
    
    current_profile_df=pd.read_csv(current_profile_path, sep=r"\s+", header=None)
    
    columns = ["z", "U", "Iu", "Iv", "Iw", "Lu", "Lv", "Lw"]
    current_profile_df.columns = columns

    current_profile_df["R_11"] = (current_profile_df["Iu"] * current_profile_df["U"])**2
    current_profile_df["R_22"] = (current_profile_df["Iv"] * current_profile_df["U"])**2
    current_profile_df["R_33"] = (current_profile_df["Iw"] * current_profile_df["U"])**2
    
    current_profile_df = current_profile_df[["U", "R_11", "R_22", "R_33", "Lu", "Lv", "Lw"]]

    current_profile_array = current_profile_df.to_numpy()
    
    return current_profile_array

#%%

def dfsr_vel_array(case_path):

    data_path = os.path.join(case_path,'constant','boundaryData','windProfile','sampledData')

    vel_array_list=[]
    for vel_comp in ['Ux', 'Uy', 'Uz']:
        
        vel_array=pd.read_csv(os.path.join(data_path,vel_comp), sep=r"\s+", comment="#", header=None).iloc[:,1:].to_numpy().astype(float)
        vel_array_list.append(vel_array)
        
    vel_array_3d = np.stack(vel_array_list, axis = 0)
    
    return vel_array_3d


#%%

def get_time_steps_dfsr_data(case_path):
    
    data_path = os.path.join(case_path,'constant','boundaryData','windProfile','sampledData')
    
    time_steps = pd.read_csv(os.path.join(data_path,"Ux"), sep=r"\s+", comment="#", header=None).iloc[:,0].to_numpy().astype(float)
    
    return time_steps

#%%

def get_downstream_dfsr_profile_array(vel_array_3d, time_step, inlet_or_downstream="inlet", burn_in_time=None, time_steps=None):
    
    if inlet_or_downstream=="inlet":
        mean_vel_array_2d = LES._profileAnalysis.mean_vel(vel_array_3d)
        fluc_vel_array_3d = LES._profileAnalysis.fluc_vel(vel_array_3d, mean_vel_array_2d)
        re_stresses = LES._profileAnalysis.re_stresses(fluc_vel_array_3d)

        int_time_scales = LES._profileAnalysis.int_time_scale(fluc_vel_array_3d, time_step)
        int_length_scales = LES._profileAnalysis.int_length_scales(int_time_scales, mean_vel_array_2d)
            
        downstream_profile = np.stack( [mean_vel_array_2d[0], re_stresses[0],
                                        re_stresses[1], re_stresses[2], int_length_scales[0],
                                        int_length_scales[1], int_length_scales[2] ], axis = 1)

    else:
        
        mask = (time_steps>burn_in_time)

        vel_array_3d = vel_array_3d[:,mask,:]

        mean_vel_array_2d = LES._profileAnalysis.mean_vel(vel_array_3d)
        fluc_vel_array_3d = LES._profileAnalysis.fluc_vel(vel_array_3d, mean_vel_array_2d)
        re_stresses = LES._profileAnalysis.re_stresses(fluc_vel_array_3d)

        int_time_scales = LES._profileAnalysis.int_time_scale(fluc_vel_array_3d, time_step)
        int_length_scales = LES._profileAnalysis.int_length_scales(int_time_scales, mean_vel_array_2d)
            
        downstream_profile = np.stack( [mean_vel_array_2d[0], re_stresses[0],
                                        re_stresses[1], re_stresses[2], int_length_scales[0],
                                        int_length_scales[1], int_length_scales[2] ], axis = 1)
    
    return downstream_profile


#%%

def new_dfsr_profile_array(current_inlet_profile_array, target_profile_array, downstream_profile_array, relaxation_factor=0.9):
    
    adaptive_relaxation_factor = relaxation_factor * (current_inlet_profile_array / downstream_profile_array)
    
    conditions = [adaptive_relaxation_factor < 0.5,
                  (adaptive_relaxation_factor >= 0.5) & (adaptive_relaxation_factor <= 5),
                  adaptive_relaxation_factor > 5]
    choices = [0.5, adaptive_relaxation_factor, 5]
     
    adaptive_relaxation_factor = np.select(conditions, choices)
    
    new_inlet_profile = current_inlet_profile_array + adaptive_relaxation_factor * (target_profile_array - downstream_profile_array)
    
    new_inlet_profile[:, 0] = np.clip(new_inlet_profile[:, 0], 0.01, None)    # U > 0
    new_inlet_profile[:, 1:4] = np.clip(new_inlet_profile[:, 1:4], 1e-8, None)  # Reynolds stresses >= 0
    new_inlet_profile[:, 4:7] = np.clip(new_inlet_profile[:, 4:7], 0.01, None)  # length scales > 0

    return new_inlet_profile


#%%

def get_avg_z_thresolds_ids(target_profile_df, lower_z_threshold, upper_z_thresold):
    
    lower_z_thresold_diffs = abs(target_profile_df["z"] - lower_z_threshold)
    upper_z_thresold_diffs = abs(target_profile_df["z"] - upper_z_thresold)

    lower_z_threshold_id = lower_z_thresold_diffs[lower_z_thresold_diffs==np.min(lower_z_thresold_diffs)].index[0]
    upper_z_threshold_id = upper_z_thresold_diffs[upper_z_thresold_diffs==np.min(upper_z_thresold_diffs)].index[0]
    
    return lower_z_threshold_id, upper_z_threshold_id

#%% 

def get_rmse(downstream_profile, target_profile, lower_z_threshold_id, upper_z_threshold_id):
    
    downstream_profile_filtered = downstream_profile[lower_z_threshold_id : upper_z_threshold_id+1, :]
    target_profile_filtered = target_profile[lower_z_threshold_id : upper_z_threshold_id+1, :]
    
    rmse_array = np.sqrt( (np.sum((downstream_profile_filtered - target_profile_filtered)**2, axis=0) / np.shape(target_profile_filtered)[0]) )
    
    return rmse_array

#%% 

def dfsr_iter_status(case_path, rmse_array, rmse_threshold, inlet_or_downstream="inlet"):
    
    if inlet_or_downstream == "inlet":
        dfsr_iter_path = os.path.join(case_path, "log", "inletCalibration")
    elif inlet_or_downstream == "downstream":
        dfsr_iter_path = os.path.join(case_path, "log", "downstreamCalibration")
    else:
        raise ValueError("inlet_or_downstream must be 'inlet' or 'downstream'")
    
    os.makedirs(dfsr_iter_path, exist_ok=True)
    
    iteration_list = [
        d for d in os.listdir(dfsr_iter_path)
        if d.startswith("iteration") and os.path.isdir(os.path.join(dfsr_iter_path, d))
    ]
    
    worst_rmse = float(np.max(rmse_array))
    
    if len(iteration_list) == 0:
        iteration = 1
        improvement_ratio = None
        stagnated = False
    else:
        iter_nums = []
        for iteration_str in iteration_list:
            try:
                iter_nums.append(int(iteration_str.replace("iteration", "")))
            except ValueError:
                pass

        if len(iter_nums) == 0:
            iteration = 1
            improvement_ratio = None
            stagnated = False
        else:
            max_iter = max(iter_nums)
            iteration = max_iter + 1

            prev_iter_json = os.path.join(
                dfsr_iter_path,
                f"iteration{max_iter}",
                f"iteration{max_iter}.json"
            )

            with open(prev_iter_json, "r") as f:
                prev_rmse = json.load(f).get("worst_rmse")

            if prev_rmse is None or prev_rmse == 0:
                improvement_ratio = None
                stagnated = False
            else:
                improvement_ratio = float(worst_rmse / prev_rmse)
                stagnated = bool(improvement_ratio > 0.98)

    converged = bool(worst_rmse <= rmse_threshold)

    iter_status = {
        "iteration": int(iteration),
        "converged": bool(converged),
        "stagnated": bool(stagnated),
        "worst_rmse": float(worst_rmse),
        "improvement_ratio": None if improvement_ratio is None else float(improvement_ratio)
    }

    return iter_status
    