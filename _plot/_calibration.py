# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 12:09:19 2026

@author: David Cunningham
"""

import logging
import numpy as np
import pandas as pd
import os
import sys
import shutil
import matplotlib.pyplot as plt

cwd = os.path.dirname(os.path.abspath(__file__))
windlespy_path = os.path.abspath(os.path.join(cwd, "..", ".."))
sys.path.append(windlespy_path)
import windlespy as LES
sys.path.remove(windlespy_path)

#%%

def plot_spectral_calibration(
    fig_dir,
    z_array,
    freq_array,
    downstream_spectra_array,
    updated_spectra_array,
    inlet_spectra_array,
    target_spectra_array,
    cutoff_freqs=None,          # shape (3, nHeights) or None
    z_min=None,
    z_max=None,
):
    """
    Plot spectra for selected heights only.

    Parameters
    ----------
    cutoff_freqs : array or None
        If provided, should have shape (3, nHeights). A vertical line is
        plotted for each component/height cutoff.
    z_min, z_max : float or None
        Height range to plot. If None, no limit is applied.
    """

    descs = ["Downstream", "Updated", "Inlet", "Target"]
    z_array = np.asarray(z_array)

    # Height mask
    height_mask = np.ones(len(z_array), dtype=bool)
    if z_min is not None:
        height_mask &= z_array >= z_min
    if z_max is not None:
        height_mask &= z_array <= z_max

    height_ids = np.where(height_mask)[0]

    for vel_comp_id, vel_comp in enumerate(["u", "v", "w"]):

        fig_dir = os.path.join(fig_dir, f"S_{vel_comp}{vel_comp}")
        os.makedirs(fig_dir, exist_ok=True)

        for height_id in height_ids:
            height = z_array[height_id]

            comp_downstream_spectra = downstream_spectra_array[vel_comp_id, height_id, :]
            comp_updated_spectra = updated_spectra_array[vel_comp_id, height_id, :]
            comp_inlet_spectra = inlet_spectra_array[vel_comp_id, height_id, :]
            comp_target_spectra = target_spectra_array[vel_comp_id, height_id, :]

            spectra = np.stack(
                [
                    comp_downstream_spectra,
                    comp_updated_spectra,
                    comp_inlet_spectra,
                    comp_target_spectra,
                ],
                axis=0,
            )

            fig = LES._plot.plot_spectra(
                freq_array,
                spectra,
                "f [Hz]",
                fr"$S_{{{vel_comp}{vel_comp}}}(f)$",
                descs,
            )

            # Assume plot_spectra returns a figure with one main axis
            ax = fig.axes[0]

            # Add cutoff line if available
            if cutoff_freqs is not None:
                fc = cutoff_freqs[vel_comp_id, height_id]
                if np.isfinite(fc):
                    ax.axvline(fc, linestyle="--", linewidth=1.2, color="k", label=f"$f_c$ = {fc:.2f} Hz")
                    ax.legend()

            # Add title with height
            ax.set_title(fr"{vel_comp}-component spectra at $z$ = {height:.3f} m")

            filename = f"Height_{height:.3f}m_S_{vel_comp}{vel_comp}.png"
            fig.savefig(os.path.join(fig_dir, filename), dpi=300, bbox_inches="tight")

            plt.close(fig)
            
