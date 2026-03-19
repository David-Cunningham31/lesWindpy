# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 14:49:42 2026

@author: David Cunningham
"""

import logging
import numpy as np
import pandas as pd
import os
import sys

cwd = os.path.dirname(os.path.abspath(__file__))
windlespy_path = os.path.abspath(os.path.join(cwd, "..", ".."))
sys.path.append(windlespy_path)
import windlespy as LES
sys.path.remove(windlespy_path)

#%%
case_path = r"/home/people/20397873/LES/DFSR_testing/empty_domain"

variable_dict = LES._caseFiles.parse_setup_file(case_path)

#%%

target_profile_df = LES._profileCalibration.get_dfsr_target_profile_df(case_path)
                    
U = target_profile_df["U"].to_numpy()

z = target_profile_df["z"].to_numpy()

u_star_fit, z0_fit = LES._profileAnalysis.fit_log_law(U, z, zmin = 0.1, zmax = 1.5)

domain_height = variable_dict["zMax"] - variable_dict["zMin"]

t_star = LES._profileAnalysis.calculate_t_star(u_star_fit, domain_height)


#%%
probe_path = os.path.join(case_path,"postProcessing","probes2")

vel_array_3d = LES._profileAnalysis.get_velocity_components(probe_path)

time_steps = LES._profileAnalysis.get_time_steps_probe_data(probe_path)

vel_array_4d, block_end_times = LES._profileAnalysis.split_vel_to_blocks(time_steps, vel_array_3d, 4*t_star)

mean_vel_array_3d = LES._profileAnalysis.mean_block_vel(vel_array_4d)

fluc_vel_array_4d = LES._profileAnalysis.fluc_block_vel(vel_array_4d, mean_vel_array_3d)

re_stresses = LES._profileAnalysis.re_stresses_block(fluc_vel_array_4d)

building_height = variable_dict['buildingHeight']

probe_id = LES._profileAnalysis.get_closest_probe_id(z, building_height)

errs_array, max_re_stress_err = LES._profileAnalysis.max_percent_block_errors(re_stresses, probe_id)

initializing_time = 12.5*t_star
stat_avg_time = 40*t_star

can_les_finish = LES._profileAnalysis.can_LES_finish(time_steps, max_re_stress_err, initializing_time, stat_avg_time)

#%%

if can_les_finish:
    sys.exit(0)
else:
    sys.exit(1)
