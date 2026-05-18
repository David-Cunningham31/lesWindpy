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
            

#%% --------------------------------------------------------------------------
# Shared spectral diagnostic helpers
# ---------------------------------------------------------------------------

def _selected_height_ids(z_array, z_min=None, z_max=None, n=6):
    z = np.asarray(z_array, dtype=float)
    mask = np.ones_like(z, dtype=bool)
    if z_min is not None:
        mask &= z >= z_min
    if z_max is not None:
        mask &= z <= z_max
    ids = np.where(mask)[0]
    if ids.size == 0:
        return np.array([], dtype=int)
    if ids.size <= n:
        return ids
    return np.unique(np.round(np.linspace(ids[0], ids[-1], n)).astype(int))


def _selected_freq_ids(freq_array, n=6, freq_targets=None):
    f = np.asarray(freq_array, dtype=float)
    if freq_targets is not None:
        ids = [int(np.argmin(np.abs(f - ft))) for ft in freq_targets]
        return np.unique(ids)
    if len(f) <= n:
        return np.arange(len(f))
    return np.unique(np.round(np.geomspace(1, len(f), n)).astype(int) - 1)


def _safe_spectrum(values, floor=1e-30):
    v = np.asarray(values, dtype=float)
    v = np.nan_to_num(v, nan=floor, posinf=floor, neginf=floor)
    return np.maximum(v, floor)


#%% --------------------------------------------------------------------------
# Downstream calibration diagnostic plots
# ---------------------------------------------------------------------------

def plot_multitaper_spline_spectra(
    fig_dir,
    z_array,
    freq_array,
    raw_psd,
    binned_psd,
    spline_downstream_spectra_array,
    inlet_spectra_array,
    target_spectra_array,
    body_height,
    z_min=0.0,
    z_max_factor=1.5,
):
    """Plot raw multitaper PSD, median-bin points, spline fit, inlet and target spectra."""
    os.makedirs(fig_dir, exist_ok=True)
    z = np.asarray(z_array, dtype=float)
    height_ids = _selected_height_ids(z, z_min, z_max_factor * body_height, n=6)
    for comp_id, comp in enumerate(["u", "v", "w"]):
        comp_dir = os.path.join(fig_dir, f"S_{comp}{comp}")
        os.makedirs(comp_dir, exist_ok=True)
        for h in height_ids:
            fig, ax = plt.subplots(figsize=(9, 6))
            if raw_psd is not None and raw_psd[comp_id][h] is not None:
                rf, rS = raw_psd[comp_id][h]
                rf = np.asarray(rf, dtype=float)
                rS = np.asarray(rS, dtype=float)
                m = (rf > 0) & (rf <= np.max(freq_array)) & np.isfinite(rS) & (rS > 0)
                ax.loglog(rf[m], rS[m], alpha=0.35, linewidth=0.8, label="Raw multitaper")
            if binned_psd is not None and binned_psd[comp_id][h] is not None:
                bf, bS = binned_psd[comp_id][h]
                if len(bf) > 0:
                    ax.loglog(bf, bS, "o", markersize=4, label="Median bins")
            ax.loglog(freq_array, _safe_spectrum(spline_downstream_spectra_array[comp_id, h]), label="Downstream spline")
            ax.loglog(freq_array, _safe_spectrum(inlet_spectra_array[comp_id, h]), label="Previous inlet")
            ax.loglog(freq_array, _safe_spectrum(target_spectra_array[comp_id, h]), label="Target")
            ax.set_xlabel("f [Hz]")
            ax.set_ylabel(fr"$S_{{{comp}{comp}}}(f)$")
            ax.set_title(fr"{comp}-component at $z/H$ = {z[h] / body_height:.2f}")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
            fig.savefig(os.path.join(comp_dir, f"zH_{z[h]/body_height:.2f}_multitaper_spline.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)


def plot_vertical_smoothing_spectra(
    fig_dir,
    z_array,
    freq_array,
    before_smoothing_array,
    after_smoothing_array,
    body_height,
    z_min=0.0,
    z_max_factor=1.5,
):
    os.makedirs(fig_dir, exist_ok=True)
    z = np.asarray(z_array, dtype=float)
    height_ids = _selected_height_ids(z, z_min, z_max_factor * body_height, n=6)
    for comp_id, comp in enumerate(["u", "v", "w"]):
        comp_dir = os.path.join(fig_dir, f"S_{comp}{comp}")
        os.makedirs(comp_dir, exist_ok=True)
        for h in height_ids:
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.loglog(freq_array, _safe_spectrum(before_smoothing_array[comp_id, h]), label="Before height smoothing")
            ax.loglog(freq_array, _safe_spectrum(after_smoothing_array[comp_id, h]), label="After height smoothing")
            ax.set_xlabel("f [Hz]")
            ax.set_ylabel(fr"$S_{{{comp}{comp}}}(f)$")
            ax.set_title(fr"Vertical smoothing check at $z/H$ = {z[h] / body_height:.2f}")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
            fig.savefig(os.path.join(comp_dir, f"zH_{z[h]/body_height:.2f}_height_smoothing.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)


def plot_frequency_vertical_profiles(
    fig_dir,
    z_array,
    freq_array,
    downstream_before_array,
    downstream_after_array,
    inlet_spectra_array,
    target_spectra_array,
    body_height,
    z_max_factor=1.5,
):
    os.makedirs(fig_dir, exist_ok=True)
    z = np.asarray(z_array, dtype=float)
    mask = z <= z_max_factor * body_height
    zH = z[mask] / body_height
    freq_ids = _selected_freq_ids(freq_array, n=6)
    for comp_id, comp in enumerate(["u", "v", "w"]):
        comp_dir = os.path.join(fig_dir, f"S_{comp}{comp}")
        os.makedirs(comp_dir, exist_ok=True)
        for fi in freq_ids:
            fig, ax = plt.subplots(figsize=(7, 8))
            ax.semilogx(_safe_spectrum(downstream_before_array[comp_id, mask, fi]), zH, label="Downstream before height smoothing")
            ax.semilogx(_safe_spectrum(downstream_after_array[comp_id, mask, fi]), zH, label="Downstream after height smoothing")
            ax.semilogx(_safe_spectrum(inlet_spectra_array[comp_id, mask, fi]), zH, label="Previous inlet")
            ax.semilogx(_safe_spectrum(target_spectra_array[comp_id, mask, fi]), zH, label="Target")
            ax.set_xlabel(fr"$S_{{{comp}{comp}}}(f={freq_array[fi]:.3g}\,Hz)$")
            ax.set_ylabel("z/H")
            ax.set_title(fr"Vertical spectral variation, {comp}-component")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
            fig.savefig(os.path.join(comp_dir, f"f_{freq_array[fi]:.4g}_vertical_profile.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)


def plot_transfer_function(
    fig_dir,
    z_array,
    freq_array,
    inverse_transfer_raw,
    inverse_transfer_smoothed,
    body_height,
    z_min=0.0,
    z_max_factor=1.5,
    plot_clip=(0.0, 3.0),
):
    """Plot raw and smoothed inverse transfer. Values are clipped only for display."""
    os.makedirs(fig_dir, exist_ok=True)
    z = np.asarray(z_array, dtype=float)
    height_ids = _selected_height_ids(z, z_min, z_max_factor * body_height, n=6)
    for comp_id, comp in enumerate(["u", "v", "w"]):
        comp_dir = os.path.join(fig_dir, f"S_{comp}{comp}")
        os.makedirs(comp_dir, exist_ok=True)
        for h in height_ids:
            fig, ax = plt.subplots(figsize=(9, 6))
            raw_plot = np.asarray(inverse_transfer_raw[comp_id, h], dtype=float)
            smooth_plot = np.asarray(inverse_transfer_smoothed[comp_id, h], dtype=float)
            if plot_clip is not None:
                raw_plot = np.clip(raw_plot, plot_clip[0], plot_clip[1])
                smooth_plot = np.clip(smooth_plot, plot_clip[0], plot_clip[1])
            ax.semilogx(freq_array, raw_plot, alpha=0.5, label="Raw inverse transfer, display clipped")
            ax.semilogx(freq_array, smooth_plot, label="Smoothed inverse transfer")
            ax.axhline(1.0, linestyle="--", linewidth=1.0, color="k", alpha=0.5)
            if plot_clip is not None:
                ax.set_ylim(plot_clip[0], plot_clip[1])
            ax.set_xlabel("f [Hz]")
            ax.set_ylabel(r"$S_{inlet}/S_{downstream}$")
            ax.set_title(fr"Inverse transfer, {comp}-component, $z/H$ = {z[h] / body_height:.2f}")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
            fig.savefig(os.path.join(comp_dir, f"zH_{z[h]/body_height:.2f}_inverse_transfer.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

#%% --------------------------------------------------------------------------
# Resolved target spectra diagnostic plots
# ---------------------------------------------------------------------------

def _summary_row(summary_df, height_id, component):
    if summary_df is None:
        return None
    df = summary_df[(summary_df["height_id"] == height_id) & (summary_df["component"] == component)]
    if len(df) == 0:
        return None
    return df.iloc[0]


def plot_resolved_target_spectra_comparison_dfsr(
    fig_dir,
    z_array,
    freq_array,
    corrected_spectra_array,
    uncorrected_spectra_array,
    summary_df=None,
    body_height=None,
    z_max=None,
    floor=1e-30,
):
    """Compare corrected target spectra against original VK target spectra on the DFSR grid."""
    os.makedirs(fig_dir, exist_ok=True)
    z = np.asarray(z_array, dtype=float)
    height_ids = np.arange(len(z))
    if z_max is not None:
        height_ids = height_ids[z <= z_max]

    for comp_id, comp in enumerate(["u", "v", "w"]):
        comp_dir = os.path.join(fig_dir, f"S_{comp}{comp}")
        os.makedirs(comp_dir, exist_ok=True)
        for h in height_ids:
            row = _summary_row(summary_df, h, comp)
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.loglog(freq_array, _safe_spectrum(uncorrected_spectra_array[comp_id, h], floor), label="Original VK target")
            ax.loglog(freq_array, _safe_spectrum(corrected_spectra_array[comp_id, h], floor), label="Resolved-band corrected target")
            if row is not None and np.isfinite(row.get("f_max_resolved", np.nan)):
                ax.axvline(row["f_max_resolved"], linestyle="--", linewidth=1.0, color="k", alpha=0.5, label="$f_{max,res}$")
            ax.set_xlabel("f [Hz]")
            ax.set_ylabel(fr"$S_{{{comp}{comp}}}(f)$")
            if row is not None:
                title = (
                    f"{comp}-component target spectra at z={z[h]:.3f} m"
                    f"\nL_target={row['L_target']:.3g}, L_eff={row['L_eff']:.3g}, "
                    f"L_res,corr={row['L_resolved_corrected']:.3g}, "
                    f"L_res,orig={row['L_resolved_uncorrected']:.3g}"
                )
            else:
                title = f"{comp}-component target spectra at z={z[h]:.3f} m"
            ax.set_title(title)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
            fig.savefig(os.path.join(comp_dir, f"target_spectra_{comp}_z_{z[h]:.3f}m.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)


def plot_resolved_target_frequency_vertical_profiles_dfsr(
    fig_dir,
    z_array,
    freq_array,
    corrected_spectra_array,
    uncorrected_spectra_array,
    body_height,
    freq_targets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
    z_max=None,
    floor=1e-30,
):
    """Plot target spectral variation with height at selected frequencies."""
    os.makedirs(fig_dir, exist_ok=True)
    z = np.asarray(z_array, dtype=float)
    mask = np.ones_like(z, dtype=bool)
    if z_max is not None:
        mask &= z <= z_max
    zH = z[mask] / body_height
    freq_ids = _selected_freq_ids(freq_array, freq_targets=freq_targets)

    for comp_id, comp in enumerate(["u", "v", "w"]):
        comp_dir = os.path.join(fig_dir, f"S_{comp}{comp}")
        os.makedirs(comp_dir, exist_ok=True)
        for fi in freq_ids:
            fig, ax = plt.subplots(figsize=(7, 8))
            ax.semilogx(_safe_spectrum(uncorrected_spectra_array[comp_id, mask, fi], floor), zH, label="Original VK target")
            ax.semilogx(_safe_spectrum(corrected_spectra_array[comp_id, mask, fi], floor), zH, label="Resolved-band corrected target")
            ax.set_xlabel(fr"$S_{{{comp}{comp}}}(f={freq_array[fi]:.3g}\,Hz)$")
            ax.set_ylabel("z/H")
            ax.set_title(fr"Target spectra vertical variation, {comp}-component")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
            fig.savefig(os.path.join(comp_dir, f"f_{freq_array[fi]:.4g}_target_vertical_profile.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)


def plot_resolved_target_height_spectra_stack_dfsr(
    fig_dir,
    z_array,
    freq_array,
    corrected_spectra_array,
    uncorrected_spectra_array,
    body_height,
    z_min=0.0,
    z_max=None,
    n_heights=8,
    floor=1e-30,
):
    """Plot spectra at a handful of heights together to inspect vertical trends."""
    os.makedirs(fig_dir, exist_ok=True)
    z = np.asarray(z_array, dtype=float)
    height_ids = _selected_height_ids(z, z_min=z_min, z_max=z_max, n=n_heights)

    for comp_id, comp in enumerate(["u", "v", "w"]):
        comp_dir = os.path.join(fig_dir, f"S_{comp}{comp}")
        os.makedirs(comp_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 7))
        for h in height_ids:
            ax.loglog(freq_array, _safe_spectrum(uncorrected_spectra_array[comp_id, h], floor), linestyle="--", alpha=0.65, label=f"Original z/H={z[h]/body_height:.2f}")
            ax.loglog(freq_array, _safe_spectrum(corrected_spectra_array[comp_id, h], floor), alpha=0.9, label=f"Corrected z/H={z[h]/body_height:.2f}")
        ax.set_xlabel("f [Hz]")
        ax.set_ylabel(fr"$S_{{{comp}{comp}}}(f)$")
        ax.set_title(fr"Target spectra at selected heights, {comp}-component")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        fig.savefig(os.path.join(comp_dir, f"target_spectra_height_stack_{comp}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_resolved_target_autocorrelation_comparison(
    fig_dir,
    z_array,
    corrected_diagnostics,
    uncorrected_diagnostics,
    body_height=None,
    z_max=None,
):
    """Plot autocorrelation implied by the resolved-band original and corrected spectra."""
    os.makedirs(fig_dir, exist_ok=True)
    z = np.asarray(z_array, dtype=float)
    height_ids = np.arange(len(z))
    if z_max is not None:
        height_ids = height_ids[z <= z_max]

    for comp_id, comp in enumerate(["u", "v", "w"]):
        comp_dir = os.path.join(fig_dir, f"S_{comp}{comp}")
        os.makedirs(comp_dir, exist_ok=True)
        for h in height_ids:
            corr = corrected_diagnostics[comp_id][h]
            uncorr = uncorrected_diagnostics[comp_id][h]
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.plot(uncorr["tau"], uncorr["rho"], label=f"Original VK, L_res={uncorr['L_resolved']:.3g} m")
            ax.plot(corr["tau"], corr["rho"], label=f"Corrected VK, L_res={corr['L_resolved']:.3g} m")
            ax.axhline(0.0, linestyle="--", linewidth=1.0, color="k", alpha=0.5)
            ax.set_xlabel(r"$\tau$ [s]")
            ax.set_ylabel(r"$\rho(\tau)$")
            ax.set_title(f"{comp}-component implied autocorrelation at z={z[h]:.3f} m")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.savefig(os.path.join(comp_dir, f"target_rho_{comp}_z_{z[h]:.3f}m.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)


def plot_resolved_target_summary_profiles(fig_dir, z_array, summary_df, body_height=None):
    """Plot profile-level diagnostics for the resolved target spectra construction."""
    os.makedirs(fig_dir, exist_ok=True)
    z = np.asarray(z_array, dtype=float)
    y = z / body_height if body_height is not None else z
    y_label = "z/H" if body_height is not None else "z [m]"

    for comp in ["u", "v", "w"]:
        df = summary_df[summary_df["component"] == comp].copy()
        h_ids = df["height_id"].to_numpy(dtype=int)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(df["L_eff"] / df["L_target"], y[h_ids], label=r"$L_{eff}/L_{target}$")
        ax.axvline(1.0, linestyle="--", linewidth=1.0, color="k", alpha=0.5)
        ax.set_xlabel(r"$L_{eff}/L_{target}$")
        ax.set_ylabel(y_label)
        ax.set_title(f"{comp}-component effective VK shape parameter")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(os.path.join(fig_dir, f"L_eff_ratio_{comp}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(df["L_target"], y[h_ids], label="Target L")
        ax.plot(df["L_resolved_uncorrected"], y[h_ids], label="Resolved L from original VK")
        ax.plot(df["L_resolved_corrected"], y[h_ids], label="Resolved L from corrected VK")
        ax.set_xlabel("L [m]")
        ax.set_ylabel(y_label)
        ax.set_title(f"{comp}-component resolved length-scale comparison")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(os.path.join(fig_dir, f"L_resolved_compare_{comp}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(df["resolved_variance_uncorrected"] / df["sigma2_target"], y[h_ids], label="Original VK resolved variance / target")
        ax.plot(df["resolved_variance_corrected"] / df["sigma2_target"], y[h_ids], label="Corrected VK resolved variance / target")
        ax.axvline(1.0, linestyle="--", linewidth=1.0, color="k", alpha=0.5)
        ax.set_xlabel(r"$\sigma^2_{resolved}/\sigma^2_{target}$")
        ax.set_ylabel(y_label)
        ax.set_title(f"{comp}-component resolved variance comparison")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(os.path.join(fig_dir, f"variance_resolved_ratio_{comp}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
