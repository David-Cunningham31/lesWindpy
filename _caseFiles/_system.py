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
import pyvista as pv

#%%

def write_probes_from_target_profile(x_coord, y_coord, case_path, target_profile_df, filename):
    
    z_coord_strs = (target_profile_df["z"].astype(str) + ")").to_numpy()
    
    probe_df = pd.DataFrame({"x":np.full((len(z_coord_strs),),f"({x_coord}"),
                             "y":np.full((len(z_coord_strs),),f"{y_coord}"),
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

def _format_openfoam_scalar(value):
    """Format numeric values cleanly for OpenFOAM dictionaries."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.12g}"
    return str(value)


#%%

def write_surf_presure_probes(
    field,
    coords_df,
    patch_name,
    case_path,
    filename,
    write_control="timeStep",
    write_interval=None,
):
    """
    Write an OpenFOAM patchProbes functionObject dictionary for surface pressure probes.

    Parameters
    ----------
    field : str or list[str]
        Field or fields to sample, e.g. "p" or ["p"].
    coords_df : pandas.DataFrame
        DataFrame containing columns x, y, z in OpenFOAM/LES coordinates [m].
    patch_name : str
        OpenFOAM boundary patch to sample. Written as the required `patch` entry.
    case_path : str
        Path to OpenFOAM case. The file is written to <case_path>/system/<filename>.
    filename : str
        Name of the functionObject dictionary file to write.
    write_control : str, optional
        OpenFOAM writeControl. Defaults to "timeStep".
    write_interval : int, float, or None, optional
        OpenFOAM writeInterval. If None, defaults to 1.

    Notes
    -----
    This function intentionally keeps the historical misspelling
    `write_surf_presure_probes` so existing scripts do not break.
    """

    required_cols = ["x", "y", "z"]
    missing = [c for c in required_cols if c not in coords_df.columns]
    if missing:
        raise ValueError(f"coords_df is missing required coordinate columns: {missing}")

    if write_interval is None:
        write_interval = 1

    if isinstance(write_interval, (int, float, np.integer, np.floating)) and write_interval <= 0:
        raise ValueError("write_interval must be positive or None.")

    if isinstance(field, str):
        field_names = [field]
    else:
        field_names = list(field)

    if not field_names:
        raise ValueError("At least one field must be supplied.")

    coords = coords_df[required_cols].to_numpy(dtype=float)
    fields_block = "\n".join(f"    {field_name}" for field_name in field_names)
    write_interval_str = _format_openfoam_scalar(write_interval)

    output_path = os.path.join(case_path, "system", filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    header_str = f"""/*--------------------------------*- C++ -*----------------------------------*\
  =========                 |
  \      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \    /   O peration     |
    \  /    A nd           | Web:      www.OpenFOAM.org
     \/     M anipulation  |
-------------------------------------------------------------------------------
Description
    Writes out values of fields from specified patch.

\*---------------------------------------------------------------------------*/

#includeEtc "caseDicts/postProcessing/probes/probes.cfg"

type            patchProbes;
libs            ("libsampling.so");
writeControl    {write_control};
writeInterval   {write_interval_str};
patch           {patch_name};

fields
(
{fields_block}
);

probeLocations
(
"""

    with open(output_path, "w", newline="\n") as f:
        f.write(header_str)
        for x, y, z in coords:
            f.write(f"    ({x:.10g} {y:.10g} {z:.10g})\n")
        f.write(");\n\n")
        f.write("// ************************************************************************* //\n")


#%%

def get_inlet_cell_centres(case_path, foamFilename, inlet_patch_name):

    mesh=pv.read(os.path.join(case_path,foamFilename))

    inlet_mesh = mesh["boundary"][inlet_patch_name]

    inlet_face_centres = inlet_mesh.cell_centers()

    cell_ctr_array = inlet_face_centres.points

    cell_ctrs_df = pd.DataFrame({"x":cell_ctr_array[:,0], "y":cell_ctr_array[:,1], "z":cell_ctr_array[:,2]})

    z_centres = np.sort(cell_ctrs_df["z"].round(10).unique())

    return z_centres


#%%

def write_vel_probes_from_z_array(
    x_coord,
    y_coord,
    case_path,
    z_array,
    filename,
    sampling_rate=None,
    field_names=None,
):
    """
    Write an OpenFOAM probes functionObject dictionary for velocity probes
    at fixed x, y and multiple z values.

    Parameters
    ----------
    x_coord : float
        x coordinate of all probe locations.
    y_coord : float
        y coordinate of all probe locations.
    case_path : str
        Path to OpenFOAM case.
    z_array : array-like
        Iterable of z coordinates.
    filename : str
        Output filename inside case/system, e.g. "probes_U".
    sampling_rate : float or None, optional
        Sampling frequency in Hz.
        If provided:
            writeControl adjustableRunTime
            writeInterval = sampling_rate
        If None:
            writeControl timeStep
            writeInterval 1
    field_names : list[str] or None, optional
        Fields to sample. Defaults to ["U"].
    """

    z_array = np.asarray(z_array, dtype=float)

    if z_array.ndim != 1:
        raise ValueError("z_array must be a 1D array-like object.")

    if field_names is None:
        field_names = ["U"]

    if sampling_rate is not None:
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive.")
        write_control = "adjustableRunTime"
        write_interval = sampling_rate
    else:
        write_control = "timeStep"
        write_interval = 1

    output_path = os.path.join(case_path, "system", filename)

    fields_block = "\n".join(f"    {field}" for field in field_names)

    header_str = f"""/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     |
    \\\\  /    A nd           |
     \\\\/     M anipulation  |
-------------------------------------------------------------------------------
Description
    Writes out values of fields from cells nearest to specified locations.

\\*---------------------------------------------------------------------------*/

#includeEtc "caseDicts/postProcessing/probes/probes.cfg"

type            probes;
libs            ("libsampling.so");
writeControl    {write_control};
writeInterval   {write_interval};

fields
(
{fields_block}
);

probeLocations
(
"""

    with open(output_path, "w", newline="\n") as f:
        f.write(header_str)
        for z in z_array:
            f.write(f"    ({x_coord} {y_coord} {z})\n")
        f.write(");\n\n")
        f.write("// ************************************************************************* //\n")