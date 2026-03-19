# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 15:24:28 2026

@author: David Cunningham
"""

import os
import pandas as pd
import numpy as np
import json

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
