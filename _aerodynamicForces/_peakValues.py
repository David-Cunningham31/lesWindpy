# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 10:41:30 2026

@author: david
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

def get_blue_coeff_df():
    
    blue_coeff_dict = {}
    blue_coeff_dict["n"]=[]
    blue_coeff_dict["i"]=[]
    blue_coeff_dict["a_i"]=[]
    blue_coeff_dict["b_i"]=[]
                
    
    blue_coeff_dict["n"] =[2]*2 + [3]*3 + [4]*4 + [5]*5 + [6]*6 + [7]*7 + [8]*8 + [9]*9 + [10]*10 + [11]*11 + [12]*12 + [13]*13 + [14]*14 + [15]*15 + [16]*16
    
    blue_coeff_dict["i"] = list(range(1,3)) + list(range(1,4)) + list(range(1,5)) + list(range(1,6)) + list(range(1,7)) + list(range(1,8)) + list(range(1,9)) + list(range(1,10)) + list(range(1,11)) + list(range(1,12)) \
    + list(range(1,13)) + list(range(1,14)) + list(range(1,15)) + list(range(1,16)) + list(range(1,17))
    
    blue_coeff_dict["a_i"]+=[0.916373, 
                             0.083627, 
                             0.656320, 
                             0.255714, 
                             0.087966, 
                             0.510998, 
                             0.263943, 
                             0.153680, 
                             0.071380,
                             0.418934,
                             0.246282,
                             0.167609,
                             0.108824,
                             0.058350,
                             0.355450,
                             0.225488,
                             0.165620,
                             0.121054,
                             0.083522,
                             0.048867,
                             0.309008,
                             0.206260,
                             0.158590,
                             0.123223,
                             0.093747,
                             0.067331,
                             0.041841,
                             0.273535,
                             0.189428,
                             0.150200,
                             0.121174,
                             0.097142,
                             0.075904,
                             0.056132,
                             0.036485,
                             0.245539,
                             0.174882,
                             0.141789,
                             0.117357,
                             0.097218,
                             0.079569,
                             0.063400,
                             0.047957,
                             0.032291,
                             0.222867,
                             0.162308,
                             0.133845,
                             0.112868,
                             0.095636,
                             0.080618,
                             0.066988,
                             0.054193,
                             0.041748,
                             0.028929,
                             0.204123,
                             0.151384,
                             0.126522,
                             0.108226,
                             0.093234,
                             0.080222,
                             0.068485,
                             0.057578,
                             0.047159,
                             0.036886,
                             0.026180,
                             0.188361,
                             0.141833,
                             0.119838,
                             0.103673,
                             0.090455,
                             0.079018,
                             0.068747,
                             0.059266,
                             0.050303,
                             0.041628,
                             0.032984,
                             0.023894,
                             0.174916,
                             0.133422,
                             0.113759,
                             0.099323,
                             0.087540,
                             0.077368,
                             0.068264,
                             0.059900,
                             0.052047,
                             0.044528,
                             0.037177,
                             0.029790,
                             0.021965,
                             0.163309,
                             0.125966,
                             0.108230,
                             0.095223,
                             0.084619,
                             0.075484,
                             0.067331,
                             0.059866,
                             0.052891,
                             0.046260,
                             0.039847,
                             0.033526,
                             0.027131,
                             0.020317,
                             0.153184,
                             0.119314,
                             0.103196,
                             0.091384,
                             0.081767,
                             0.073495,
                             0.066128,
                             0.059401,
                             0.053140,
                             0.047217,
                             0.041529,
                             0.035984,
                             0.030484,
                             0.024887,
                             0.018894,
                             0.144271,
                             0.113346,
                             0.098600,
                             0.087801,
                             0.079021,
                             0.071476,
                             0.064771,
                             0.058660,
                             0.052989,
                             0.047646,
                             0.042539,
                             0.037597,
                             0.032748,
                             0.027911,
                             0.022969,
                             0.017653]
    
    blue_coeff_dict["b_i"]+=[ -0.721348, 
                             0.721348,
                             -0.630541, 
                             0.255816, 
                             0.374725,
                             -0.558619,
                             0.085903, 
                             0.223919, 
                             0.248797,
                             -0.503127,
                             0.006534,
                             0.130455,
                             0.181656,
                             0.184483,
                             -0.459273,
                             -0.035992,
                             0.073199,
                             0.126724,
                             0.149534,
                             0.145807,
                             -0.423700,
                             -0.060698,
                             0.036192,
                             0.087339,
                             0.114868,
                             0.125859,
                             0.120141,
                             -0.394187,
                             -0.075767,
                             0.011124,
                             0.058928,
                             0.087162,
                             0.102728,
                             0.108074,
                             0.101936,
                             -0.369242,
                             -0.085203,
                             -0.006486,
                             0.037977,
                             0.065574,
                             0.082654,
                             0.091965,
                             0.094369,
                             0.088391,
                             -0.347830,
                             -0.091158,
                             -0.019210,
                             0.022179,
                             0.048671,
                             0.066064,
                             0.077021,
                             0.082771,
                             0.083552,
                             0.077940,
                             -0.329210,
                             -0.094869,
                             -0.028604,
                             0.010032,
                             0.035284,
                             0.052464,
                             0.064071,
                             0.071381,
                             0.074977,
                             0.074830,
                             0.069644,
                             -0.312840,
                             -0.097086,
                             -0.035655,
                             0.000534,
                             0.024548,
                             0.041278,
                             0.053053,
                             0.061112,
                             0.066122,
                             0.068357,
                             0.067671,
                             0.062906,
                             -0.298313,
                             -0.098284,
                             -0.041013,
                             -0.006997,
                             0.015836,
                             0.032014,
                             0.043710,
                             0.052101,
                             0.057862,
                             0.061355,
                             0.062699,
                             0.061699,
                             0.057330,
                             -0.285316,
                             -0.098775,
                             -0.045120,
                             -0.013039,
                             0.008690,
                             0.024282,
                             0.035768,
                             0.044262,
                             0.050418,
                             0.054624,
                             0.057083,
                             0.057829,
                             0.056652,
                             0.052642,
                             -0.273606,
                             -0.098768,
                             -0.048285,
                             -0.017934,
                             0.002773,
                             0.017779,
                             0.028988,
                             0.037452,
                             0.043798,
                             0.048415,
                             0.051534,
                             0.053267,
                             0.053603,
                             0.052334,
                             0.048648,
                             -0.262990,
                             -0.098406,
                             -0.050731,
                             -0.021933,
                             -0.002167,
                             0.012270,
                             0.023168,
                             0.031528,
                             0.037939,
                             0.042787,
                             0.046308,
                             0.048646,
                             0.049860,
                             0.049912,
                             0.048602,
                             0.045207
                             ]
                                     
    return pd.DataFrame(blue_coeff_dict)
    
#%%

def create_epochal_windows(values, time_steps, n_windows):
    values = np.asarray(values, dtype=float)
    time_steps = np.asarray(time_steps, dtype=float)

    if len(values) != len(time_steps):
        raise ValueError(
            f"values and time_steps must have the same length. "
            f"Got {len(values)} and {len(time_steps)}."
        )

    if n_windows < 2:
        raise ValueError("n_windows must be at least 2.")

    if len(values) < n_windows:
        raise ValueError(
            f"Cannot split {len(values)} samples into {n_windows} non-empty windows."
        )

    epochal_values = np.array_split(values, n_windows)
    epochal_time_steps = np.array_split(time_steps, n_windows)

    return epochal_values, epochal_time_steps

#%%

def get_epochal_peaks(epochal_values):
    positive_peaks = np.zeros(len(epochal_values), dtype=float)
    negative_peaks = np.zeros(len(epochal_values), dtype=float)

    for i, epoch in enumerate(epochal_values):
        epoch = np.asarray(epoch, dtype=float)

        if epoch.size == 0:
            raise ValueError(f"Epoch {i} is empty.")

        positive_peaks[i] = np.nanmax(epoch)
        negative_peaks[i] = np.nanmin(epoch)

    return positive_peaks, negative_peaks

#%%

def get_gumbel_fit_parameters(positive_peaks, negative_peaks, epochal_time_steps):
    
    ordered_positive_peaks = np.sort(positive_peaks,axis=0)
    ordered_negative_peaks = np.sort(negative_peaks*-1,axis=0)
        
    n_positive_samples = len(ordered_positive_peaks)
    n_negative_samples = len(ordered_negative_peaks)

    blue_coeff_df = get_blue_coeff_df()
    
    pos_blue_a_i = blue_coeff_df[blue_coeff_df["n"]==n_positive_samples]["a_i"].to_numpy()
    
    u_pos = np.sum(pos_blue_a_i * ordered_positive_peaks)
    
    pos_blue_b_i = blue_coeff_df[blue_coeff_df["n"]==n_positive_samples]["b_i"].to_numpy()
    
    b_pos = np.sum(pos_blue_b_i * ordered_positive_peaks)
    
    neg_blue_a_i = blue_coeff_df[blue_coeff_df["n"]==n_negative_samples]["a_i"].to_numpy()
    
    u_neg = np.sum(neg_blue_a_i * ordered_negative_peaks)
    
    neg_blue_b_i = blue_coeff_df[blue_coeff_df["n"]==n_negative_samples]["b_i"].to_numpy()
    
    b_neg = np.sum(neg_blue_b_i * ordered_negative_peaks)
    
    a_pos = 1/b_pos
    U_pos = u_pos
    
    a_neg = 1/b_neg
    U_neg = u_neg
        
    r = len(epochal_time_steps)
    
    U_pos_r = U_pos + np.log(r)/a_pos
    
    U_neg_r = U_neg + np.log(r)/a_neg
    
    return a_pos, U_pos_r, a_neg, U_neg_r

#%%
    
def get_peak_values(a_pos, U_pos_r, a_neg, U_neg_r, percentile=78):
    
    reduced_variate = -np.log(-np.log(percentile/100))
    
    pos_peak = U_pos_r + (1/a_pos)*reduced_variate
    
    neg_peak_mag = U_neg_r + (1/a_neg)*reduced_variate
    
    neg_peak_signed = -neg_peak_mag
    
    return pos_peak, neg_peak_signed
    
    
    
    
    
    

