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

def plot_profile(array, norm_heights, x_label, y_label, xlims=None, ylims=None, several=False, descs=None):
    fig, ax = plt.subplots()
    
    if not several:
        ax.plot(array, norm_heights)
    else:        
        for profile,desc in zip(array,descs):
            ax.plot(profile, norm_heights, label=desc)
        ax.legend()
    
    if xlims==None:
        ax.set_xlim(0, None)
    else:
        ax.set_xlim(xlims[0], xlims[1])
    
    if ylims==None:
        ax.set_ylim(0, None)
    else:
        ax.set_ylim(ylims[0], ylims[1])

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    ax.minorticks_on()
    
    return fig

#%%

def plot_spectra(fs, spectra, x_label, y_label, descs):
    
    fig, ax = plt.subplots()
    
    for spectrum, desc in zip(spectra, descs):
        ax.plot(fs, spectrum, label=desc)
        ax.legend()
        
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
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
        

#%%

def plot_cospectral_target_profiles_0_to_3H(
    experimental_profile_df,
    smoothed_profile_df,
    building_height,
    fig_dir,
    columns=("U", "Iu", "Iv", "Iw", "Lu", "Lv", "Lw", "uwStress"),
):
    """
    Plot measured NHERI points and smoothed/extended CoSpectralDFSRTurb target
    profiles from 0 to 3H.
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)

    labels = {
        "U": "Mean velocity, U",
        "Iu": "Longitudinal turbulence intensity, Iu",
        "Iv": "Lateral turbulence intensity, Iv",
        "Iw": "Vertical turbulence intensity, Iw",
        "Lu": "Longitudinal integral length scale, Lu",
        "Lv": "Lateral integral length scale, Lv",
        "Lw": "Vertical integral length scale, Lw",
        "uwStress": "Reynolds shear stress, u'w'",
    }

    z_max = 3.0 * float(building_height)

    for col in columns:
        if col not in smoothed_profile_df.columns:
            print(f"Skipping {col}: not found in smoothed profile.")
            continue

        fig, ax = plt.subplots(figsize=(5.5, 6.0))

        z_s = smoothed_profile_df["z"].to_numpy(dtype=float)
        y_s = smoothed_profile_df[col].to_numpy(dtype=float)
        m_s = np.isfinite(z_s) & np.isfinite(y_s) & (z_s <= z_max)

        ax.plot(
            y_s[m_s],
            z_s[m_s] / building_height,
            linewidth=2.0,
            label="Smoothed/extended target",
        )

        if col in experimental_profile_df.columns:
            z_e = experimental_profile_df["z"].to_numpy(dtype=float)
            y_e = experimental_profile_df[col].to_numpy(dtype=float)
            m_e = np.isfinite(z_e) & np.isfinite(y_e) & (z_e <= z_max)

            ax.scatter(
                y_e[m_e],
                z_e[m_e] / building_height,
                s=28,
                marker="o",
                label="Experimental",
                zorder=3,
            )

        ax.set_ylim(0.0, 3.0)
        ax.set_ylabel(r"$z/H$")
        ax.set_xlabel(labels.get(col, col))
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(labels.get(col, col))

        fig.tight_layout()
        fig.savefig(
            os.path.join(fig_dir, f"cospectral_target_profile_{col}.png"),
            dpi=300,
        )
        plt.close(fig)
    