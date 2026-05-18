# -*- coding: utf-8 -*-
"""
Downstream DFSR spectral calibration with explicit resolved-band variance correction.

This is the classic spectral-calibration recipe with one important correction:
after the pointwise Wong spectral update, and again after the power-law tail is
applied, the updated spectrum is scaled component-by-component and
height-by-height so that its area over the LES-resolved frequency band matches
the intended Wong-updated Reynolds stress.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cwd = os.path.dirname(os.path.abspath(__file__))
windlespy_path = os.path.abspath(os.path.join(cwd, "..", ".."))
sys.path.append(windlespy_path)
import windlespy as LES
sys.path.remove(windlespy_path)

#%% --------------------------------------------------------------------------
# Case setup
# ---------------------------------------------------------------------------

case_path = os.environ["CASE_DIR"]
downstream_probes_folder = os.path.join(case_path, "postProcessing", "probes2")

variable_dict = LES._caseFiles.parse_setup_file(case_path)

building_height = variable_dict["buildingHeight"]
lower_z_threshold = variable_dict["lowerZThreshold"]
upper_z_thresold = variable_dict["upperZThreshold"]
rmse_threshold = variable_dict["rmseThreshold"]
mesh_size = variable_dict["meshSize"]
fMax = variable_dict["fMax"]
nFreq = variable_dict["nFreq"]

json_path = os.path.join(case_path, "log", "downstreamCalibration", "sim_init.json")
with open(json_path, "r") as f:
    dfsr_les_init_dict = json.load(f)

burn_in_time = dfsr_les_init_dict["burn_in_time"]

#%% --------------------------------------------------------------------------
# User/calibration settings
# ---------------------------------------------------------------------------

PSD_FIT_MIN = 0.5
PSD_FIT_MAX = None
PSD_MIN_POINTS_PER_BIN = 4

PSD_LOW_PLATEAU_BAND = (0.20, 1.00)
PSD_LEFT_MODE = "band_median"
PSD_RIGHT_MODE = "slope"
PSD_RIGHT_SLOPE_CLIP = (-8.0, 0.0)

PSD_BAND_CONFIG = [
    (1.0, 2.0, 3),
    (2.0, 5.0, 4),
    (5.0, 10.0, 4),
    (10.0, None, 10),
]

PSD_SMOOTH_KNOTS = True
PSD_KNOT_SMOOTH_KERNEL = [1, 2, 1]
PSD_HEIGHT_KERNEL = [1, 2, 1]

SPECTRAL_RELAXATION_FACTOR = 0.9
TRANSFER_SAVGOL_WINDOW = 121
TRANSFER_SAVGOL_POLYORDER = 2

# Variance correction. The spectral update defines the shape; this rescales the
# final spectrum so the LES-resolved band area matches the desired Reynolds stress.
RENORMALISE_UPDATED_SPECTRA_VARIANCE = True
USE_HEIGHT_DEPENDENT_RESOLVED_FMAX = True
RESOLVED_F_MIN_OVERRIDE = None
RESOLVED_F_MAX_OVERRIDE = None
WRITE_VARIANCE_DIAGNOSTICS = True

APPLY_POWER_LAW_TAIL = True
POWER_LAW_TAIL_CUTOFF_FACTOR = 0.5
POWER_LAW_TAIL_SLOPE = -5.0 / 3.0

floor = 1e-16
SPECTRUM_FLOOR_FOR_WRITE = floor

#%% --------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _trapz(y, x, axis=-1):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    return np.trapz(y, x, axis=axis)


def _band_mask_for_height(freq_array, h_id, f_min=None, f_max=None):
    f = np.asarray(freq_array, dtype=float)
    mask = np.isfinite(f) & (f > 0.0)
    if f_min is not None:
        mask &= f >= float(f_min)
    if f_max is not None:
        if np.ndim(f_max) == 0:
            f_hi = float(f_max)
        else:
            f_hi = float(np.asarray(f_max, dtype=float)[h_id])
        mask &= f <= f_hi
    return mask


def integrate_spectra_area(freq_array, spectra_array, f_min=None, f_max=None):
    f = np.asarray(freq_array, dtype=float)
    spectra = np.asarray(spectra_array, dtype=float)
    area = np.zeros(spectra.shape[:2], dtype=float)
    for h_id in range(spectra.shape[1]):
        mask = _band_mask_for_height(f, h_id, f_min=f_min, f_max=f_max)
        if np.count_nonzero(mask) < 2:
            raise ValueError(f"Not enough frequency points in integration band for height index {h_id}.")
        for comp_id in range(spectra.shape[0]):
            area[comp_id, h_id] = _trapz(spectra[comp_id, h_id, mask], f[mask], axis=0)
    return area


def wong_update_variance(inlet_sigma2, target_sigma2, downstream_sigma2, relaxation_factor=0.9, floor=1e-16):
    inlet = np.maximum(np.asarray(inlet_sigma2, dtype=float), floor)
    target = np.maximum(np.asarray(target_sigma2, dtype=float), floor)
    downstream = np.maximum(np.asarray(downstream_sigma2, dtype=float), floor)
    updated = inlet + relaxation_factor * (inlet / downstream) * (target - downstream)
    return np.maximum(updated, floor)


def renormalise_spectra_to_variance(freq_array, spectra_array, target_variance, floor=1e-16, f_min=None, f_max=None):
    spectra = np.maximum(np.asarray(spectra_array, dtype=float), floor)
    target = np.maximum(np.asarray(target_variance, dtype=float), floor)
    current_area = np.maximum(integrate_spectra_area(freq_array, spectra, f_min=f_min, f_max=f_max), floor)
    scale = target / current_area
    out = spectra.copy()
    for comp_id in range(out.shape[0]):
        for h_id in range(out.shape[1]):
            out[comp_id, h_id, :] *= scale[comp_id, h_id]
    return np.maximum(out, floor), scale, current_area


def get_resolved_frequency_limits(time_steps, time_step, mesh_size, target_profile_array):
    sample_duration = float(time_steps[-1] - time_steps[0])
    f_min = 1.0 / sample_duration if RESOLVED_F_MIN_OVERRIDE is None else float(RESOLVED_F_MIN_OVERRIDE)
    f_nyquist = 1.0 / (2.0 * float(time_step))

    if RESOLVED_F_MAX_OVERRIDE is not None:
        f_max = float(RESOLVED_F_MAX_OVERRIDE)
    else:
        U = target_profile_array[:, 0]
        L = target_profile_array[:, -3:].T
        sigmas = np.sqrt(target_profile_array[:, 1:4]).T
        mesh_cutoff_freqs = LES._profileAnalysis.get_mesh_cutoff_frequencies(mesh_size, U, L, sigmas)
        if USE_HEIGHT_DEPENDENT_RESOLVED_FMAX:
            f_max = np.minimum(mesh_cutoff_freqs, f_nyquist)
        else:
            f_max = float(min(np.nanmin(mesh_cutoff_freqs), f_nyquist))
    return f_min, f_max


def write_variance_diagnostics(fig_dir, z_array, building_height, resolved_f_min, resolved_f_max,
                               inlet_sigma2, target_sigma2, downstream_sigma2, updated_sigma2,
                               final_sigma2, scale_pre_tail, scale_post_tail):
    rows = []
    components = ("u", "v", "w")
    rel_error = (final_sigma2 - updated_sigma2) / np.maximum(updated_sigma2, floor)
    for comp_id, comp in enumerate(components):
        for h_id, z in enumerate(z_array):
            rows.append({
                "component": comp,
                "height_id": h_id,
                "z": z,
                "z_over_H": z / building_height,
                "resolved_f_min": resolved_f_min,
                "resolved_f_max": resolved_f_max if np.ndim(resolved_f_max) == 0 else resolved_f_max[h_id],
                "inlet_sigma2_resolved": inlet_sigma2[comp_id, h_id],
                "target_sigma2_resolved": target_sigma2[comp_id, h_id],
                "downstream_sigma2_profile": downstream_sigma2[comp_id, h_id],
                "updated_sigma2_resolved_target": updated_sigma2[comp_id, h_id],
                "updated_sigma2_resolved_final": final_sigma2[comp_id, h_id],
                "final_sigma2_rel_error": rel_error[comp_id, h_id],
                "variance_scale_pre_tail": scale_pre_tail[comp_id, h_id],
                "variance_scale_post_tail": scale_post_tail[comp_id, h_id],
            })
    pd.DataFrame(rows).to_csv(os.path.join(fig_dir, "resolved_variance_correction_diagnostics.csv"), index=False)
    return rel_error

#%% --------------------------------------------------------------------------
# Read profiles and probe data
# ---------------------------------------------------------------------------

target_profile_df = LES._profileCalibration.get_dfsr_target_profile_df(case_path)
target_profile_array = LES._profileCalibration.get_dfsr_target_profile_array(case_path)
inlet_profile_array = LES._profileCalibration.get_current_dfsr_inlet_profile_array(case_path)

vel_array_3d = LES._profileAnalysis.get_velocity_components(downstream_probes_folder)
time_steps = LES._profileAnalysis.get_time_steps_probe_data(downstream_probes_folder)

mask = time_steps > burn_in_time
vel_array_3d = vel_array_3d[:, mask, :]
time_steps = time_steps[mask]
time_step = float(np.mean(np.diff(time_steps)))

# The probe array has already been burn-in masked, so use inlet_or_downstream="inlet"
# to avoid any possible double filtering inside windLespy.
downstream_profile_array = LES._profileCalibration.get_downstream_dfsr_profile_array(
    vel_array_3d,
    time_step,
    inlet_or_downstream="inlet",
    burn_in_time=None,
    time_steps=None,
)

new_inlet_profile_array = LES._profileCalibration.update_mean_profile_only(
    inlet_profile_array,
    target_profile_array,
    downstream_profile_array,
    relaxation_factor=0.9,
)

lower_z_threshold_id, upper_z_threshold_id = LES._profileCalibration.get_avg_z_thresolds_ids(
    target_profile_df, lower_z_threshold, upper_z_thresold
)
rmse_array = LES._profileCalibration.get_rmse(
    downstream_profile_array,
    target_profile_array,
    lower_z_threshold_id,
    upper_z_threshold_id,
)

iter_status = LES._profileCalibration.dfsr_iter_status(case_path, rmse_array, rmse_threshold, "downstream")
LES._caseFiles.write_dfsr_iter_json(case_path, iter_status, "downstream")

iteration = iter_status["iteration"]
converged = iter_status["converged"]
stagnated = iter_status["stagnated"]

freq_array = LES._profileCalibration.get_freq_array(fMax, nFreq)
z_array = target_profile_df["z"].to_numpy(dtype=float)

resolved_f_min, resolved_f_max = get_resolved_frequency_limits(
    time_steps=time_steps,
    time_step=time_step,
    mesh_size=mesh_size,
    target_profile_array=target_profile_array,
)

print("\nSpectral variance correction settings:")
print(f"  resolved f_min = {resolved_f_min:.4g} Hz")
if np.ndim(resolved_f_max) == 0:
    print(f"  resolved f_max = {resolved_f_max:.4g} Hz")
else:
    print(f"  resolved f_max min/max = {np.nanmin(resolved_f_max):.4g} / {np.nanmax(resolved_f_max):.4g} Hz")

#%% --------------------------------------------------------------------------
# Profile plots
# ---------------------------------------------------------------------------

fig_folder = os.path.join(case_path, "log", "downstreamCalibration", f"iteration{iteration}", "plots", "profiles")
os.makedirs(fig_folder, exist_ok=True)

height_mask = (target_profile_df["z"] <= (3.0 * building_height)).to_numpy()
norm_heights = target_profile_df["z"].to_numpy(dtype=float) / building_height
norm_heights = norm_heights[height_mask]

for col_index, x_axis_desc in enumerate(target_profile_df.columns[1:]):
    if "I" in x_axis_desc:
        downstream_profile = np.sqrt(downstream_profile_array[height_mask, col_index]) / downstream_profile_array[height_mask, 0]
        target_profile = np.sqrt(target_profile_array[height_mask, col_index]) / target_profile_array[height_mask, 0]
    else:
        downstream_profile = downstream_profile_array[height_mask, col_index]
        target_profile = target_profile_array[height_mask, col_index]

    profiles_array = np.stack([downstream_profile, target_profile], axis=0)
    fig = LES._plot.plot_profile(
        profiles_array,
        norm_heights,
        x_axis_desc,
        "z/H",
        xlims=None,
        ylims=None,
        several=True,
        descs=["Downstream Profile", "Target Profile"],
    )
    fig.savefig(os.path.join(fig_folder, f"{x_axis_desc}_profiles.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

#%% --------------------------------------------------------------------------
# Spectral calibration
# ---------------------------------------------------------------------------

target_spectra_array = LES._profileCalibration.read_spectra_profile_file(case_path, "targetSpectraProfile")
inlet_spectra_array = LES._profileCalibration.read_spectra_profile_file(case_path, "spectraProfile")

(
    downstream_spectra_spline_array,
    downstream_raw_multitaper,
    downstream_binned_multitaper,
) = LES._profileCalibration.get_downstream_spectra_array(
    fMax,
    nFreq,
    vel_array_3d,
    time_step,
    building_height,
    downstream_profile_array,
    inlet_or_downstream="inlet",
    burn_in_time=None,
    time_steps=None,
    method="multitaper",
    psd_fit_min=PSD_FIT_MIN,
    psd_fit_max=PSD_FIT_MAX,
    psd_band_config=PSD_BAND_CONFIG,
    min_points_per_bin=PSD_MIN_POINTS_PER_BIN,
    psd_left_mode=PSD_LEFT_MODE,
    psd_right_mode=PSD_RIGHT_MODE,
    psd_right_slope_clip=PSD_RIGHT_SLOPE_CLIP,
    psd_low_plateau_band=PSD_LOW_PLATEAU_BAND,
    psd_smooth_knots=PSD_SMOOTH_KNOTS,
    psd_knot_smooth_kernel=PSD_KNOT_SMOOTH_KERNEL,
    time_bandwidth=4.0,
    num_tapers=None,
    return_raw=True,
    floor=floor,
)

fig_dir = os.path.join(case_path, "log", "downstreamCalibration", f"iteration{iteration}", "plots", "spectra")
os.makedirs(fig_dir, exist_ok=True)

LES._plot.plot_multitaper_spline_spectra(
    os.path.join(fig_dir, "01_multitaper_spline"),
    z_array,
    freq_array,
    downstream_raw_multitaper,
    downstream_binned_multitaper,
    downstream_spectra_spline_array,
    inlet_spectra_array,
    target_spectra_array,
    building_height,
    z_min=0.0,
    z_max_factor=1.5,
)

downstream_spectra_smoothed_array = LES._profileCalibration.smooth_spectra_array_height_kernel(
    z_array,
    downstream_spectra_spline_array,
    kernel_weights=PSD_HEIGHT_KERNEL,
    floor=floor,
)

LES._plot.plot_vertical_smoothing_spectra(
    os.path.join(fig_dir, "02_vertical_smoothing_spectra"),
    z_array,
    freq_array,
    downstream_spectra_spline_array,
    downstream_spectra_smoothed_array,
    building_height,
    z_min=0.0,
    z_max_factor=1.5,
)

LES._plot.plot_frequency_vertical_profiles(
    os.path.join(fig_dir, "03_frequency_vertical_profiles"),
    z_array,
    freq_array,
    downstream_spectra_spline_array,
    downstream_spectra_smoothed_array,
    inlet_spectra_array,
    target_spectra_array,
    building_height,
    z_max_factor=1.5,
)

inverse_transfer_raw = LES._profileCalibration.get_inverse_transfer_function(
    inlet_spectra_array,
    downstream_spectra_smoothed_array,
    floor=floor,
)
inverse_transfer_smoothed = LES._profileCalibration.smooth_spectral_ratio_array(
    freq_array,
    inverse_transfer_raw,
    window_length=TRANSFER_SAVGOL_WINDOW,
    polyorder=TRANSFER_SAVGOL_POLYORDER,
    floor=floor,
)

LES._plot.plot_transfer_function(
    os.path.join(fig_dir, "04_inverse_transfer_function"),
    z_array,
    freq_array,
    inverse_transfer_raw,
    inverse_transfer_smoothed,
    building_height,
    z_min=0.0,
    z_max_factor=1.5,
)

updated_spectra_array = LES._profileCalibration.get_updated_spectra_array_wong(
    inlet_spectra_array,
    target_spectra_array,
    downstream_spectra_smoothed_array,
    inverse_transfer_function=None,
    relaxation_factor=SPECTRAL_RELAXATION_FACTOR,
    floor=floor,
)

# Wong-updated resolved-band Reynolds stresses. These are the target areas for
# the written updated spectra.
inlet_sigma2_resolved = integrate_spectra_area(freq_array, inlet_spectra_array, f_min=resolved_f_min, f_max=resolved_f_max)
target_sigma2_resolved = integrate_spectra_area(freq_array, target_spectra_array, f_min=resolved_f_min, f_max=resolved_f_max)
downstream_sigma2_resolved = np.maximum(downstream_profile_array[:, 1:4].T, floor)
updated_sigma2_resolved = wong_update_variance(
    inlet_sigma2_resolved,
    target_sigma2_resolved,
    downstream_sigma2_resolved,
    relaxation_factor=SPECTRAL_RELAXATION_FACTOR,
    floor=floor,
)

if RENORMALISE_UPDATED_SPECTRA_VARIANCE:
    updated_spectra_array, variance_scale_pre_tail, _ = renormalise_spectra_to_variance(
        freq_array,
        updated_spectra_array,
        updated_sigma2_resolved,
        floor=floor,
        f_min=resolved_f_min,
        f_max=resolved_f_max,
    )
else:
    variance_scale_pre_tail = np.ones_like(updated_sigma2_resolved)

if APPLY_POWER_LAW_TAIL:
    U_tail = target_profile_array[:, 0]
    L_tail = target_profile_array[:, -3:].T
    sigmas_tail = np.sqrt(target_profile_array[:, 1:4]).T
    mesh_cutoff_freqs = LES._profileAnalysis.get_mesh_cutoff_frequencies(mesh_size, U_tail, L_tail, sigmas_tail)
    mesh_cutoff_freqs_3d = np.broadcast_to(mesh_cutoff_freqs[np.newaxis, :], (3, len(mesh_cutoff_freqs)))
    effective_cutoff_freqs_3d = POWER_LAW_TAIL_CUTOFF_FACTOR * mesh_cutoff_freqs_3d
    updated_spectra_array = LES._profileCalibration.apply_power_law_tail(
        freq_array,
        updated_spectra_array,
        effective_cutoff_freqs_3d,
        slope=POWER_LAW_TAIL_SLOPE,
        floor=1e-20,
    )

# Enforce resolved-band variance again after the tail treatment.
if RENORMALISE_UPDATED_SPECTRA_VARIANCE:
    updated_spectra_array, variance_scale_post_tail, _ = renormalise_spectra_to_variance(
        freq_array,
        updated_spectra_array,
        updated_sigma2_resolved,
        floor=floor,
        f_min=resolved_f_min,
        f_max=resolved_f_max,
    )
else:
    variance_scale_post_tail = np.ones_like(updated_sigma2_resolved)

final_sigma2_resolved = integrate_spectra_area(freq_array, updated_spectra_array, f_min=resolved_f_min, f_max=resolved_f_max)
final_sigma2_rel_error = (final_sigma2_resolved - updated_sigma2_resolved) / np.maximum(updated_sigma2_resolved, floor)

print("\nResolved-band variance correction diagnostics:")
print(f"  final sigma2 rel error min/max = {np.nanmin(final_sigma2_rel_error):.4g} / {np.nanmax(final_sigma2_rel_error):.4g}")
print(f"  pre-tail scale min/max = {np.nanmin(variance_scale_pre_tail):.4g} / {np.nanmax(variance_scale_pre_tail):.4g}")
print(f"  post-tail scale min/max = {np.nanmin(variance_scale_post_tail):.4g} / {np.nanmax(variance_scale_post_tail):.4g}")

if WRITE_VARIANCE_DIAGNOSTICS:
    write_variance_diagnostics(
        fig_dir,
        z_array,
        building_height,
        resolved_f_min,
        resolved_f_max,
        inlet_sigma2_resolved,
        target_sigma2_resolved,
        downstream_sigma2_resolved,
        updated_sigma2_resolved,
        final_sigma2_resolved,
        variance_scale_pre_tail,
        variance_scale_post_tail,
    )

LES._plot.plot_spectral_calibration(
    os.path.join(fig_dir, "05_final_update_overview"),
    z_array,
    freq_array,
    downstream_spectra_smoothed_array,
    updated_spectra_array,
    inlet_spectra_array,
    target_spectra_array,
    cutoff_freqs=None,
    z_min=0.0,
    z_max=3.0 * building_height,
)

#%% --------------------------------------------------------------------------
# Write results
# ---------------------------------------------------------------------------

if (not converged) and (not stagnated):
    dfsr_input_spectra_path = os.path.join(case_path, "constant", "boundaryData", "windProfile", "spectraProfile")
    LES._caseFiles.write_spectra_profile(
        updated_spectra_array,
        z_array,
        dfsr_input_spectra_path,
        clip_min=SPECTRUM_FLOOR_FOR_WRITE,
    )
    LES._caseFiles.write_new_dfsr_inlet_profile(new_inlet_profile_array, target_profile_df, case_path)

    LES._caseFiles.write_dfsr_iter_spectra(
        case_path,
        iter_status,
        z_array,
        freq_array,
        inlet_spectra_array,
        downstream_spectra_smoothed_array,
        inlet_or_downstream="downstream",
        new_inlet_spectra_array=updated_spectra_array,
        cutoff_freqs=None,
        clip_min=SPECTRUM_FLOOR_FOR_WRITE,
    )
else:
    LES._caseFiles.write_dfsr_iter_spectra(
        case_path,
        iter_status,
        z_array,
        freq_array,
        inlet_spectra_array,
        downstream_spectra_smoothed_array,
        inlet_or_downstream="downstream",
        new_inlet_spectra_array=None,
        cutoff_freqs=None,
        clip_min=SPECTRUM_FLOOR_FOR_WRITE,
    )

#%% --------------------------------------------------------------------------
# Exit behaviour
# ---------------------------------------------------------------------------

if converged:
    sys.exit(0)
elif stagnated:
    sys.exit(0)
else:
    sys.exit(1)
