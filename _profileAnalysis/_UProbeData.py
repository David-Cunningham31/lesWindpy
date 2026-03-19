# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:59:39 2026

Module to process probe velocity vector data obtained from OpenFOAM simulations.

@author: David Cunningham
"""

import logging
import numpy as np
import pandas as pd
import os

logger = logging.getLogger(__name__)


#%%

def get_probe_coordinates(filepath):
    """
    Parameters
    ----------
    filepath : Filepath to OpenFOAM probes U file.

    Returns
    -------
    DataFrame containing (x,y,z) coordinates of probe points.

    """
    
    # Read in data again without filtering out the rows starting with "#":
    df=pd.read_csv(filepath, header=None)
    
    # Define masks which allow for extraction of the probe coordinates:
    mask1 = df.iloc[:,0].str.startswith("#")
    mask2 = df.iloc[:,0].str.endswith(")")
    
    # Extract DF of coordinates in string format
    coord_df=df[(mask1) & (mask2)]
    
    # Using regex expression, extract the coordinates we need and convert to numeric format
    coord_df = coord_df.iloc[:,0].str.extract(r"\(\s*(?P<x>[-\d\.eE+]+)\s+(?P<y>[-\d\.eE+]+)\s+(?P<z>[-\d\.eE+]+)\s*\)").astype(float) 
    
    return coord_df

#%%

def get_closest_probe_id(probe_heights, target_height):
    
    height_diffs = abs(probe_heights - target_height)
    
    closest_idx = np.argmin(height_diffs)
    
    probe_id = closest_idx
    
    return probe_id
#%%

def get_velocity_components(filepath):
    """
    

    Parameters
    ----------
    filepath : Filepath to OpenFOAM probes folder (N.B. - to folder, will concatenate data from several time-steps if necessary - if simulation was stopped and restarted).

    Returns
    
    Array of time-steps and 3D numpy array of velocity components

    """
    
    # Iterate through time steps in probe folder and add DF's to a list:
    df_list=[]
    for folder in os.listdir(filepath):
        df_list.append(pd.read_csv(os.path.join(filepath,folder,"U"), sep=r"\s+", comment="#", header=None))
    
    # Concatenating the list of DF's:
    u_df=pd.concat(df_list, axis=0)
    
    # Create Series with time-steps:
    time = pd.to_numeric(u_df.iloc[:, 0], errors="coerce")
    
    # Extract all U data and convert to a numpy array of strings:
    data = u_df.iloc[:, 1:].astype(str).to_numpy()
    
    # The number of time steps is one dimension and the number of cols is the number of probes:
    nsteps, ncols = data.shape
    
    # Number of probes: Velocity vector (3 cols) for each probe:
    nprobes = ncols // 3
    
    # Reshaping the data to a 3D numpy array.
    tok = data.reshape(nsteps, nprobes, 3).astype(str)
    # We strip any "(" and ")" non-numeric characters from the velocity values:
    tok[:,:,0] = np.char.lstrip(tok[:,:,0], "(")
    tok[:,:,2] = np.char.rstrip(tok[:,:,2], ")")
    
    # Convert array to numeric values:
    arr = tok.astype(float) 
    
    # Create velocity arrays for each of the velocity components:
    Ux = pd.DataFrame(arr[:, :, 0], index=time)
    Uy = pd.DataFrame(arr[:, :, 1], index=time)
    Uz = pd.DataFrame(arr[:, :, 2], index=time)
    
    vel_array_3d = np.stack([Ux.to_numpy(), Uy.to_numpy(), Uz.to_numpy()])
        
    return vel_array_3d

#%%

def get_time_steps_probe_data(filepath):
    
    # Iterate through time steps in probe folder and add DF's to a list:
    df_list=[]
    for folder in os.listdir(filepath):
        df_list.append(pd.read_csv(os.path.join(filepath,folder,"U"), sep=r"\s+", comment="#", header=None))
    
    # Concatenating the list of DF's:
    u_df=pd.concat(df_list, axis=0)
    
    # Create Series with time-steps:
    time = pd.to_numeric(u_df.iloc[:, 0], errors="coerce")
    
    # Get an array of the time steps:
    time_steps = time.to_numpy()
    
    return time_steps


    
    
    
    
    
    