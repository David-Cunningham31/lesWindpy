# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:00:18 2026

@author: David Cunningham
"""

import os
import sys

cwd = os.path.dirname(os.path.abspath(__file__))
windlespy_path = os.path.abspath(os.path.join(cwd, "..", ".."))
sys.path.append(windlespy_path)
import windlespy as LES
sys.path.remove(windlespy_path)


#%%

case_path = r"/home/people/20397873/LES/NHERI_Tall_Building/empty_domain_2"

cfl_df = LES._profileAnalysis.get_cfl_df(case_path)

cfl_time_step_dict = LES._profileAnalysis.get_cfl_time_step_dict(cfl_df, 90, 10)

LES._caseFiles.write_cfl_time_step_json(case_path, cfl_time_step_dict)

