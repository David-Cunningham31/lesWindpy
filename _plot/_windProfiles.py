# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 11:11:22 2026

@author: David Cunningham
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

#%%

def set_plot_format(markers = True):
    
    # -----------------------------
    # Global plotting style
    # -----------------------------
    plt.rcParams.update({
        # Figure
        "figure.figsize": (7.0, 4.5),
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    
        # Fonts
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    
        # Axes
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
    
        # Lines
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
    
        # Ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
    
        # Legend
        "legend.frameon": False,
    
        # Math text
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
    })
    
    if markers:
        
        plt.rcParams["axes.prop_cycle"] = (
            cycler(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']) +
            cycler(linestyle=['-', '--', '-.', ':', (0, (1, 10))]) +
            cycler(marker=['o', 's', '^', 'd', "*"]))
        
    else:
    
        plt.rcParams["axes.prop_cycle"] = (
            cycler(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']) +
            cycler(linestyle=['-', '--', '-.', ':', (0, (1, 10))]))
            #cycler(marker=['o', 's', '^', 'd', "*"]))
            
#%%

def plot_re_stress_errors(errs_array,block_end_times):
    
    fig, ax = plt.subplots()
    
    re_stress_err_dict = {
    r'$\overline{u^\prime u^\prime}$': errs_array[0],
    r'$\overline{v^\prime v^\prime}$': errs_array[1],
    r'$\overline{w^\prime w^\prime}$': errs_array[2],
    r'$\overline{u^\prime w^\prime}$': errs_array[3]}
    
    for re_stress_err_desc in re_stress_err_dict.keys():
        ax.plot(
        block_end_times[1:], re_stress_err_dict[re_stress_err_desc],
        label=re_stress_err_desc)
    
    ax.axhline(
        3.0,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=r'$3\%$ threshold'
    )
    
    ax.set_xlim(0, block_end_times.max()+1)
    ax.set_ylim(0, None)
    ax.set_xlabel(r'Time [s]')
    ax.set_ylabel(r'Residual [%]')
    
    ax.legend(ncol=2)
    ax.minorticks_on()
    
    return fig

#%% 

def norm_heights(z, body_dim):
    
    norm_heights = z / body_dim
    
    return norm_heights

#%%

def plot_ti_u(ti_array_3d, norm_heights):
        
    fig, ax = plt.subplots()

    ax.plot(ti_array_3d[0], norm_heights)

    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_xlabel(r'$I_{u}$ [%]')
    ax.set_ylabel(r'z/H')
    
    ax.minorticks_on()
    
    return fig

#%%

def plot_ti_v(ti_array_3d, norm_heights):
        
    fig, ax = plt.subplots()

    ax.plot(ti_array_3d[1], norm_heights)

    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_xlabel(r'$I_{v}$ [%]')
    ax.set_ylabel(r'z/H')
    
    ax.minorticks_on()
    
    return fig

#%%

def plot_ti_w(ti_array_3d, norm_heights):
        
    fig, ax = plt.subplots()

    ax.plot(ti_array_3d[2], norm_heights)

    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_xlabel(r'$I_{w}$ [%]')
    ax.set_ylabel(r'z/H')
    
    ax.minorticks_on()
    
    return fig

#%%

def plot_re_stress_11(re_stresses, norm_heights):
        
    fig, ax = plt.subplots()

    ax.plot(re_stresses[0], norm_heights)

    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_xlabel(r'$\overline{u^\prime u^\prime} \quad [m^{2}/s^{2}]$')
    ax.set_ylabel(r'z/H')
    
    ax.minorticks_on()
    
    return fig

#%%

def plot_re_stress_22(re_stresses, norm_heights):
        
    fig, ax = plt.subplots()

    ax.plot(re_stresses[1], norm_heights)

    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_xlabel(r'$\overline{v^\prime v^\prime} \quad [m^{2}/s^{2}]$')
    ax.set_ylabel(r'z/H')
    
    ax.minorticks_on()
    
    return fig

#%%

def plot_re_stress_33(re_stresses, norm_heights):
        
    fig, ax = plt.subplots()

    ax.plot(re_stresses[2], norm_heights)

    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_xlabel(r'$\overline{w^\prime w^\prime} \quad [m^{2}/s^{2}]$')
    ax.set_ylabel(r'z/H')
    
    ax.minorticks_on()
    
    return fig

#%%

def plot_re_stress_31(re_stresses, norm_heights):
        
    fig, ax = plt.subplots()

    ax.plot(re_stresses[3], norm_heights)

    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_xlabel(r'$\overline{u^\prime w^\prime} \quad [m^{2}/s^{2}]$')
    ax.set_ylabel(r'z/H')
    
    ax.minorticks_on()
    
    return fig

#%%

def plot_t_u(int_time_scales, norm_heights):
        
    fig, ax = plt.subplots()

    ax.plot(int_time_scales[0], norm_heights)

    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_xlabel(r'$T_{u}^{x}$ [s]')
    ax.set_ylabel(r'z/H')
    
    ax.minorticks_on()
    
    return fig

#%%

def plot_t_v(int_time_scales, norm_heights):
        
    fig, ax = plt.subplots()

    ax.plot(int_time_scales[1], norm_heights)

    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_xlabel(r'$T_{v}^{x}$ [s]')
    ax.set_ylabel(r'z/H')
    
    ax.minorticks_on()
    
    return fig

#%%

def plot_t_w(int_time_scales, norm_heights):
        
    fig, ax = plt.subplots()

    ax.plot(int_time_scales[2], norm_heights)

    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_xlabel(r'$T_{v}^{x}$ [s]')
    ax.set_ylabel(r'z/H')
    
    ax.minorticks_on()
    
    return fig

#%%

def plot_l_u(int_len_scales, norm_heights):
        
    fig, ax = plt.subplots()

    ax.plot(int_len_scales[0], norm_heights)

    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_xlabel(r'$L_{u}^{x}$ [m]')
    ax.set_ylabel(r'z/H')
    
    ax.minorticks_on()
    
    return fig

#%%

def plot_l_v(int_len_scales, norm_heights):
        
    fig, ax = plt.subplots()

    ax.plot(int_len_scales[1], norm_heights)

    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_xlabel(r'$L_{v}^{x}$ [m]')
    ax.set_ylabel(r'z/H')
    
    ax.minorticks_on()
    
    return fig

#%%

def plot_l_w(int_len_scales, norm_heights):
        
    fig, ax = plt.subplots()

    ax.plot(int_len_scales[2], norm_heights)

    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_xlabel(r'$L_{w}^{x}$ [m]')
    ax.set_ylabel(r'z/H')
    
    ax.minorticks_on()
    
    return fig

#%%

def plot_s_uu(LES_reduced_fs, LES_spectrum, vk_red_fs=None, vk_spectrum=None):
    
    if (vk_spectrum==None) and (vk_red_fs==None):
        
        fig, ax = plt.subplots()
    
        ax.plot(LES_reduced_fs, LES_spectrum)
    
        ax.set_xlabel(r'$f^{*}=fB/U_{ref}$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(r'$f^{*}S_{uu}(f)/ \sigma_{uu}^{2}$')
        
        ax.minorticks_on()
    
    elif (vk_spectrum!=None) and (vk_red_fs==None):
        
        fig, ax = plt.subplots()
    
        ax.plot(LES_reduced_fs, LES_spectrum, label = "LES")
        ax.plot(LES_reduced_fs, vk_spectrum, label = "von Karman")
    
        ax.set_xlabel(r'$f^{*}=fB/U_{ref}$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(r'$f^{*}S_{uu}(f)/ \sigma_{uu}^{2}$')
        ax.legend()
        ax.minorticks_on()
        
    elif (vk_spectrum!=None) and (vk_red_fs!=None):
        
        fig, ax = plt.subplots()
    
        ax.plot(LES_reduced_fs, LES_spectrum, label = "LES")
        ax.plot(vk_red_fs, vk_spectrum, label = "von Karman")
    
        ax.set_xlabel(r'$f^{*}=fB/U_{ref}$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(r'$f^{*}S_{uu}(f)/ \sigma_{uu}^{2}$')
        ax.legend()
        ax.minorticks_on()
        
    
    return fig

#%%

def plot_s_vv(LES_reduced_fs, LES_spectrum, vk_red_fs=None, vk_spectrum=None):
    
    if (vk_spectrum==None) and (vk_red_fs==None):
        
        fig, ax = plt.subplots()
    
        ax.plot(LES_reduced_fs, LES_spectrum)
    
        ax.set_xlabel(r'$f^{*}=fB/U_{ref}$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(r'$f^{*}S_{vv}(f)/ \sigma_{vv}^{2}$')
        
        ax.minorticks_on()
    
    elif (vk_spectrum!=None) and (vk_red_fs==None):
        
        fig, ax = plt.subplots()
    
        ax.plot(LES_reduced_fs, LES_spectrum, label = "LES")
        ax.plot(LES_reduced_fs, vk_spectrum, label = "von Karman")
    
        ax.set_xlabel(r'$f^{*}=fB/U_{ref}$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(r'$f^{*}S_{vv}(f)/ \sigma_{vv}^{2}$')
        ax.legend()
        ax.minorticks_on()
        
    elif (vk_spectrum!=None) and (vk_red_fs!=None):
        
        fig, ax = plt.subplots()
    
        ax.plot(LES_reduced_fs, LES_spectrum, label = "LES")
        ax.plot(vk_red_fs, vk_spectrum, label = "von Karman")
    
        ax.set_xlabel(r'$f^{*}=fB/U_{ref}$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(r'$f^{*}S_{vv}(f)/ \sigma_{vv}^{2}$')
        ax.legend()
        ax.minorticks_on()
        
    
    return fig


#%%

def plot_s_ww(LES_reduced_fs, LES_spectrum, vk_red_fs=None, vk_spectrum=None):
    
    if (vk_spectrum==None) and (vk_red_fs==None):
        
        fig, ax = plt.subplots()
    
        ax.plot(LES_reduced_fs, LES_spectrum)
    
        ax.set_xlabel(r'$f^{*}=fB/U_{ref}$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(r'$f^{*}S_{ww}(f)/ \sigma_{ww}^{2}$')
        
        ax.minorticks_on()
    
    elif (vk_spectrum!=None) and (vk_red_fs==None):
        
        fig, ax = plt.subplots()
    
        ax.plot(LES_reduced_fs, LES_spectrum, label = "LES")
        ax.plot(LES_reduced_fs, vk_spectrum, label = "von Karman")
    
        ax.set_xlabel(r'$f^{*}=fB/U_{ref}$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(r'$f^{*}S_{ww}(f)/ \sigma_{ww}^{2}$')
        ax.legend()
        ax.minorticks_on()
        
    elif (vk_spectrum!=None) and (vk_red_fs!=None):
        
        fig, ax = plt.subplots()
    
        ax.plot(LES_reduced_fs, LES_spectrum, label = "LES")
        ax.plot(vk_red_fs, vk_spectrum, label = "von Karman")
    
        ax.set_xlabel(r'$f^{*}=fB/U_{ref}$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(r'$f^{*}S_{ww}(f)/ \sigma_{ww}^{2}$')
        ax.legend()
        ax.minorticks_on()
        
    
    return fig
        


    