# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:39:25 2026

@author: David Cunningham
"""

import os
import pandas as pd
import numpy as np
import json

#%%

def write_dfsr_samp_pts(x_coord, y_coord, case_path, target_profile_df):
    
    z_coord_strs = (target_profile_df["z"].astype(str) + ")").to_numpy()
    
    probe_df = pd.DataFrame({"x":np.full((len(z_coord_strs),),f"({x_coord}"),
                             "y":np.full((len(z_coord_strs),),f"{x_coord}"),
                             "z":z_coord_strs})
    
    output_path = os.path.join(case_path, "constant", "boundaryData","windProfile","sampledData","samplingPoints")
    
    with open(output_path, "w") as f:
        # custom lines at the top
        f.write(str(len(probe_df))+"\n")
        f.write("( \n")
    
        # dataframe content
        probe_df.to_csv(f, sep="\t", index=False, header=False)
    
        # custom lines at the bottom
        f.write(");")
        
#%%

def write_new_dfsr_inlet_profile(new_inlet_profile_array, target_profile_df, case_path):
        
    output_path = os.path.join(case_path,'constant','boundaryData','windProfile','profile')
            
    new_inlet_df = pd.DataFrame({"z" : target_profile_df["z"],
                                 "U" : new_inlet_profile_array[:,0],
                                 "Iu" : np.sqrt(new_inlet_profile_array[:,1]) / new_inlet_profile_array[:,0] ,
                                 "Iv" : np.sqrt(new_inlet_profile_array[:,2]) / new_inlet_profile_array[:,0] ,
                                 "Iw" : np.sqrt(new_inlet_profile_array[:,3]) / new_inlet_profile_array[:,0] ,
                                 "Lu" : new_inlet_profile_array[:,4],
                                 "Lv" : new_inlet_profile_array[:,5],
                                 "Lw" : new_inlet_profile_array[:,6]})
    
    np.savetxt(
        output_path,
        new_inlet_df.to_numpy(),
        fmt="%.6f",
        delimiter="\t")

#%%

def write_dfsr_iter_json(case_path, iter_status, inlet_or_downstream = "inlet"):
    
    if inlet_or_downstream == "inlet":
        
        dfsr_iter_path = os.path.join(case_path, "log", "inletCalibration")
        
    elif inlet_or_downstream == "downstream":
        
        dfsr_iter_path = os.path.join(case_path, "log", "downstreamCalibration")
    
    os.makedirs(dfsr_iter_path, exist_ok=True)
    
    iteration = iter_status["iteration"]
    
    iteration_path = os.path.join(dfsr_iter_path,f"iteration{iteration}")
    
    os.makedirs(iteration_path, exist_ok=True)
    
    json_path = os.path.join(iteration_path, f"iteration{iteration}.json")
    
    with open(json_path, "w") as f:
        json.dump(iter_status, f, indent=2)
    
    
#%%

def write_dfsr_inlet_iter_profiles(case_path, iter_status, target_profile_df, current_profile_array, inlet_profile_array, inlet_or_downstream = "inlet", new_inlet_profile_array = None):
    
    if inlet_or_downstream == "inlet":
        
        dfsr_iter_path = os.path.join(case_path, "log", "inletCalibration")
        
    elif inlet_or_downstream == "downstream":
        
        dfsr_iter_path = os.path.join(case_path, "log", "downstreamCalibration")
    
    os.makedirs(dfsr_iter_path, exist_ok=True)
    
    iteration = iter_status["iteration"]
    
    iteration_path = os.path.join(dfsr_iter_path,f"iteration{iteration}")
            
    current_profile_array_path = os.path.join(iteration_path, "inletProfile")
    inlet_profile_array_path = os.path.join(iteration_path, "postCorrectionProfile")
    
    inlet_profile_df = pd.DataFrame({"z" : target_profile_df["z"],
                                 "U" : current_profile_array[:,0],
                                 "Iu" : np.sqrt(current_profile_array[:,1]) / current_profile_array[:,0] ,
                                 "Iv" : np.sqrt(current_profile_array[:,2]) / current_profile_array[:,0] ,
                                 "Iw" : np.sqrt(current_profile_array[:,3]) / current_profile_array[:,0] ,
                                 "Lu" : current_profile_array[:,4],
                                 "Lv" : current_profile_array[:,5],
                                 "Lw" : current_profile_array[:,6]})
    
    post_correction_profile_df = pd.DataFrame({"z" : target_profile_df["z"],
                                 "U" : inlet_profile_array[:,0],
                                 "Iu" : np.sqrt(inlet_profile_array[:,1]) / inlet_profile_array[:,0] ,
                                 "Iv" : np.sqrt(inlet_profile_array[:,2]) / inlet_profile_array[:,0] ,
                                 "Iw" : np.sqrt(inlet_profile_array[:,3]) / inlet_profile_array[:,0] ,
                                 "Lu" : inlet_profile_array[:,4],
                                 "Lv" : inlet_profile_array[:,5],
                                 "Lw" : inlet_profile_array[:,6]})
    
    np.savetxt(
        current_profile_array_path,
        inlet_profile_df.to_numpy(),
        fmt="%.6f",
        delimiter="\t")
    
    np.savetxt(
        inlet_profile_array_path,
        post_correction_profile_df.to_numpy(),
        fmt="%.6f",
        delimiter="\t")
    
    if new_inlet_profile_array is not None:
        
        new_profile_array_path = os.path.join(iteration_path, "newInletProfile")
        
        new_profile_df = pd.DataFrame({"z" : target_profile_df["z"],
                                     "U" : new_inlet_profile_array[:,0],
                                     "Iu" : np.sqrt(new_inlet_profile_array[:,1]) / new_inlet_profile_array[:,0] ,
                                     "Iv" : np.sqrt(new_inlet_profile_array[:,2]) / new_inlet_profile_array[:,0] ,
                                     "Iw" : np.sqrt(new_inlet_profile_array[:,3]) / new_inlet_profile_array[:,0] ,
                                     "Lu" : new_inlet_profile_array[:,4],
                                     "Lv" : new_inlet_profile_array[:,5],
                                     "Lw" : new_inlet_profile_array[:,6]})
        
        np.savetxt(
            new_profile_array_path,
            new_profile_df.to_numpy(),
            fmt="%.6f",
            delimiter="\t")
        
#%%

def write_dfsr_les_init_json(case_path, dfsr_les_init_dict):
    dir_path = os.path.join(case_path, "log", "downstreamCalibration")    
    
    os.makedirs(dir_path, exist_ok=True)
    
    filepath = os.path.join(dir_path, "sim_init.json")
    
    with open(filepath, "w") as f:
        json.dump(dfsr_les_init_dict, f, indent=2)
        