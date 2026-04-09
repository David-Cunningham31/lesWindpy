# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 12:52:03 2026

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
case_path = r"/home/people/20397873/LES/NHERI_Tall_Building/empty_domain_2"

variable_dict = LES._caseFiles.parse_setup_file(case_path)

#%%

target_profile_df = LES._profileCalibration.get_dfsr_target_profile_df(case_path)
                    
U = target_profile_df["U"].to_numpy()

z = target_profile_df["z"].to_numpy()

#%%

u_star_fit, z0_fit = LES._profileAnalysis.fit_log_law(U, z, zmin = 0.1, zmax = 1.5)

domain_height = variable_dict["zMax"] - variable_dict["zMin"]

t_star = LES._profileAnalysis.calculate_t_star(u_star_fit, domain_height)

#%%

dfsr_les_init_dict = {"t_star" : t_star,
                      "burn_in_time" : 20*t_star,
                      "min_avg_time" : 40*t_star,
                      "initial_sim_duration" : (40*t_star + 20*t_star)}

#%%

LES._caseFiles.write_dfsr_les_init_json(case_path, dfsr_les_init_dict)
