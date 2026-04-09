# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 15:24:28 2026

@author: David Cunningham
"""

import os
import pandas as pd
import numpy as np
import json
import math

#%%

def write_probes_from_target_profile(x_coord, y_coord, case_path, target_profile_df, filename):
    
    z_coord_strs = (target_profile_df["z"].astype(str) + ")").to_numpy()
    
    probe_df = pd.DataFrame({"x":np.full((len(z_coord_strs),),f"({x_coord}"),
                             "y":np.full((len(z_coord_strs),),f"{x_coord}"),
                             "z":z_coord_strs})
    
    output_path = os.path.join(case_path, "system", filename)
    
    header_str = f"""/*--------------------------------*- C++ -*----------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Web:      www.OpenFOAM.org
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Description
    Writes out values of fields from cells nearest to specified locations.

\*---------------------------------------------------------------------------*/

#includeEtc "caseDicts/postProcessing/probes/probes.cfg"

type            probes;
libs            ("libsampling.so");
writeControl    timeStep;
writeInterval   1;

fields
(
    U
);

probeLocations
(
"""
             
    with open(output_path, "w", newline="") as f:
        f.write(header_str)
        for z in target_profile_df["z"]:
            f.write(f"({x_coord}\t{y_coord}\t{z})\n")
        f.write(");\n")
        f.write("\n")
        f.write("// ************************************************************************* //")
        f.write("\n")
        
        
#%%

def write_cfl_time_step_json(case_path, cfl_time_step_dict):
    
    filepath = os.path.join(case_path, "log", "cfl_time_step.json")
    
    with open(filepath, "w") as f:
        json.dump(cfl_time_step_dict, f, indent=2)
        
#%%

def return_block_mesh_cell_numbers(base_mesh_size, domain_x, domain_y, domain_z):
    
    nx = math.ceil(domain_x / base_mesh_size)
    ny = math.ceil(domain_y / base_mesh_size)
    nz = math.ceil(domain_z / base_mesh_size)
    
    num_cell_dict = {"nx":nx,
                     "ny":ny,
                     "nz":nz,
                     }
    
    return num_cell_dict
    
#%%

def write_surf_presure_probes(field, coords_df, patch_name, case_path, filename):
    
    x_coord_strs = ("(" + coords_df["x"].astype(str)).to_numpy()
    y_coord_strs = (coords_df["y"].astype(str)).to_numpy()
    z_coord_strs = (coords_df["z"].astype(str) + ")").to_numpy()
    
    probe_df = pd.DataFrame({"x":x_coord_strs,
                             "y":y_coord_strs,
                             "z":z_coord_strs})
    
    output_path = os.path.join(case_path, "system", filename)
    
    header_str = f"""/*--------------------------------*- C++ -*----------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Web:      www.OpenFOAM.org
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Description
    Writes out values of fields from specified patch.

\*---------------------------------------------------------------------------*/

#includeEtc "caseDicts/postProcessing/probes/probes.cfg"

type            patchProbes;
libs            ("libsampling.so");
writeControl    timeStep;
writeInterval   1;
patchName       {patch_name};

fields
(
    {field}
);

probeLocations
(
"""
             
    with open(output_path, "w", newline="") as f:
        f.write(header_str)
        for i in range(0,len(probe_df)):
            x_coord = probe_df["x"].iloc[i]
            y_coord = probe_df["y"].iloc[i]
            z_coord = probe_df["z"].iloc[i]
            f.write(f"{x_coord}\t{y_coord}\t{z_coord}\n")
        f.write(");\n")
        f.write("\n")
        f.write("// ************************************************************************* //")
        f.write("\n")