# -*- coding: utf-8 -*-
"""
Create MannHybridTurb-compatible theoretical von Karman / Kaimal target input files.

Minimal outputs in constant/boundaryData/windProfile/:
    profile
    targetProfile
    spectraProfile
    targetSpectraProfile
    uwCoSpectrumProfile
    targetUWCoSpectrumProfile
    targetExperimentalProfile

The first six files are the active/target inputs needed by MannHybridTurb and
the downstream spectral calibration scripts. targetExperimentalProfile is kept
for Melaku-style diagnostic plots.
"""

import json
import os
import sys
import shutil
import numpy as np
import pandas as pd


def _trapz(y, x=None, dx=1.0, axis=-1):
    """Version-safe trapezoidal integration for NumPy 1.x/2.x."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x, dx=dx, axis=axis)
    return np.trapz(y, x=x, dx=dx, axis=axis)

cwd = os.path.dirname(os.path.abspath(__file__))
windlespy_path = r"C:\Users\david\OneDrive\Documents\PhD\Year 1"
sys.path.append(windlespy_path)
import windlespy as LES
sys.path.remove(windlespy_path)

#%%

# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------

case_path = r"C:\Users\david\OneDrive\Documents\PhD\Year 1\Co-Spectrum DFSR\OpenFoamTestCase\div_free_corr"
case_path = r"C:\Users\david\OneDrive\Documents\PhD\Year 1\spctral_tilt_cailbration\classic_wong_dfsr_new_mesh"

approach_flow_data = (
    r"C:\Users\david\OneDrive\Documents\PhD\Year 1\Wind Tunnel Test Data"
    r"\NHERI BLWT Tall Building\Approach Flow"
    r"\Approach Flow - EH160 - Marine Spires - 1200 RPM - 091721_1028.mat"
)

foam_file_for_inlet_centres = "case.foam"
inlet_patch_name = "inlet"

N_FREQ_SOLVE = 256
TAU_FACTOR = 20.0
SOLVE_SPACING = "log"
FLOOR = 1e-30

FALLBACK_RHO_UW = -0.30
UW_RHO_MAX = 0.999
ENFORCE_NEGATIVE_UW = False

WRITE_RESULTS = True
PLOT_RESULTS = True

# Keep the windProfile folder clean.  With MINIMAL_OUTPUTS=True the script
# writes only the six MannHybridTurb/calibration files plus targetExperimentalProfile.
MINIMAL_OUTPUTS = True
WRITE_TARGET_EXPERIMENTAL_PROFILE = True
WRITE_LEGACY_3COMP_SPECTRA = False
WRITE_VERBOSE_PROFILE_DIAGNOSTICS = False
WRITE_NUMPY_DIAGNOSTICS = False

# ---------------------------------------------------------------------------
# Local robust co-spectral helpers
# ---------------------------------------------------------------------------

def _kaimal_uw_shape(freq_array, z, U, floor=1e-30):
    """
    Positive Kaimal u-w co-spectrum shape in one-sided cyclic-frequency form.

    shape(f) = [14 * (f*z/U) / (1 + 9.6*f*z/U)^2.4] / f

    Sign and stress normalization are applied separately.
    """
    f = np.asarray(freq_array, dtype=float)
    ff = np.maximum(f, floor)
    zz = max(float(z), floor)
    UU = max(float(U), floor)
    n = ff * zz / UU
    return 14.0 * n / ((1.0 + 9.6 * n) ** 2.4) / ff


def _bounded_shape_normalise_to_stress(
    freq_array,
    shape,
    target_stress,
    rho_limit,
    band_mask,
    floor=1e-30,
    rtol=1e-8,
    max_iter=80,
):
    """
    Build Cuw(f) = sign(target_stress)*min(a*shape(f), rho_limit(f))
    inside band_mask and zero outside. The scalar a is selected so that
    integral Cuw df equals target_stress whenever feasible under the rho cap.
    """
    f = np.asarray(freq_array, dtype=float)
    shape = np.asarray(shape, dtype=float)
    rho_limit = np.asarray(rho_limit, dtype=float)
    band_mask = np.asarray(band_mask, dtype=bool)

    c = np.zeros_like(f, dtype=float)
    target = float(target_stress)

    if abs(target) <= floor or not np.any(band_mask):
        return c, 0.0, 0.0, False

    sign = 1.0 if target > 0.0 else -1.0
    target_abs = abs(target)

    fb = f[band_mask]
    sh = np.maximum(shape[band_mask], 0.0)
    lim = np.maximum(rho_limit[band_mask], 0.0)

    if not np.any(sh > floor):
        return c, 0.0, 0.0, False

    max_area = _trapz(lim, fb)

    if target_abs >= max_area * (1.0 - rtol):
        c[band_mask] = sign * lim
        return c, sign * _trapz(lim, fb), max_area, True

    def area_for(scale):
        return _trapz(np.minimum(scale * sh, lim), fb)

    lo = 0.0
    hi = max(target_abs / max(_trapz(sh, fb), floor), floor)

    while area_for(hi) < target_abs and hi < 1e300:
        hi *= 2.0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if area_for(mid) < target_abs:
            lo = mid
        else:
            hi = mid

    vals = np.minimum(hi * sh, lim)
    c[band_mask] = sign * vals
    return c, sign * _trapz(vals, fb), max_area, False


def make_bounded_resolved_kaimal_uw_cospectra(
    target_profile_array,
    z_array,
    freq_array,
    spectra_array,
    uw_stress_array,
    resolved_fmin_array,
    resolved_fmax_array,
    rho_max=0.95,
    enforce_negative=False,
    floor=1e-30,
):
    """
    Build resolved-band, locally bounded Kaimal u-w co-spectra.

    The returned uw_stress_realised is the actual area under the written Cuw(f),
    so the uwStress column in the OpenFOAM files remains internally consistent.
    """
    z = np.asarray(z_array, dtype=float).reshape(-1)
    f = np.asarray(freq_array, dtype=float).reshape(-1)

    target_profile_array = np.asarray(target_profile_array, dtype=float)
    spectra_array = np.asarray(spectra_array, dtype=float)

    uw_target = np.asarray(uw_stress_array, dtype=float).reshape(-1)
    if enforce_negative:
        uw_target = -np.abs(uw_target)

    resolved_fmin_array = np.asarray(resolved_fmin_array, dtype=float).reshape(-1)
    resolved_fmax_array = np.asarray(resolved_fmax_array, dtype=float).reshape(-1)

    n_heights = len(z)
    n_freq = len(f)

    if target_profile_array.shape[0] != n_heights:
        raise ValueError(
            f"target_profile_array height mismatch: expected {n_heights}, "
            f"got {target_profile_array.shape[0]}"
        )

    if spectra_array.shape == (3, n_heights, n_freq):
        # windlespy convention: component, height, frequency
        spectra_array = np.transpose(spectra_array, (1, 0, 2))
    elif spectra_array.shape == (n_heights, 3, n_freq):
        # local helper convention: height, component, frequency
        pass
    else:
        raise ValueError(
            "spectra_array must have shape (3, nHeights, nFreq) or "
            "(nHeights, 3, nFreq). "
            f"Got {spectra_array.shape}"
        )

    if uw_target.shape[0] != n_heights:
        raise ValueError(
            f"uw_stress_array length mismatch: expected {n_heights}, "
            f"got {uw_target.shape[0]}"
        )

    if resolved_fmin_array.shape[0] != n_heights:
        raise ValueError(
            f"resolved_fmin_array length mismatch: expected {n_heights}, "
            f"got {resolved_fmin_array.shape[0]}"
        )

    if resolved_fmax_array.shape[0] != n_heights:
        raise ValueError(
            f"resolved_fmax_array length mismatch: expected {n_heights}, "
            f"got {resolved_fmax_array.shape[0]}"
        )

    c_uw = np.zeros((n_heights, n_freq), dtype=float)

    uw_area_before_clip = np.zeros(n_heights)
    uw_area_after_clip = np.zeros(n_heights)
    uw_max_feasible_area_abs = np.zeros(n_heights)
    uw_infeasible_under_rho_cap = np.zeros(n_heights, dtype=bool)
    uw_clipped_fraction = np.zeros(n_heights)
    rho_uw_abs_max = np.zeros(n_heights)

    U = target_profile_array[:, 0]

    for h in range(n_heights):
        fmin_h = float(resolved_fmin_array[h])
        fmax_h = float(resolved_fmax_array[h])

        band = (
            (f >= fmin_h)
            & (f <= fmax_h)
            & np.isfinite(f)
            & (f > 0.0)
        )

        if np.count_nonzero(band) < 2:
            continue

        shape = _kaimal_uw_shape(
            freq_array=f,
            z=float(z[h]),
            U=float(U[h]),
            floor=floor,
        )

        Su = np.maximum(spectra_array[h, 0, :], floor)
        Sw = np.maximum(spectra_array[h, 2, :], floor)
        rho_limit = rho_max * np.sqrt(Su * Sw)

        # Unbounded diagnostic version: area should equal the requested target
        # before the local rho cap is applied.
        c_raw = np.zeros_like(f)
        shape_area = _trapz(np.maximum(shape[band], 0.0), f[band])

        if shape_area > floor:
            c_raw[band] = float(uw_target[h]) * np.maximum(shape[band], 0.0) / shape_area

        c_final, realised_area, max_feasible_area_abs, infeasible = (
            _bounded_shape_normalise_to_stress(
                freq_array=f,
                shape=shape,
                target_stress=float(uw_target[h]),
                rho_limit=rho_limit,
                band_mask=band,
                floor=floor,
            )
        )

        c_uw[h, :] = c_final

        denom = np.maximum(np.sqrt(Su * Sw), floor)
        rho_abs = np.abs(c_final) / denom

        rho_uw_abs_max[h] = np.nanmax(rho_abs[band])
        uw_clipped_fraction[h] = np.mean(
            np.abs(c_final[band]) >= 0.999 * rho_limit[band]
        )

        uw_area_before_clip[h] = _trapz(c_raw, f)
        uw_area_after_clip[h] = _trapz(c_final, f)
        uw_max_feasible_area_abs[h] = max_feasible_area_abs
        uw_infeasible_under_rho_cap[h] = bool(infeasible)

    uw_stress_realised = uw_area_after_clip.copy()

    diagnostics = {
        "uw_stress_input": uw_target,
        "uw_stress_realised": uw_stress_realised,
        "uw_area_before_clip": uw_area_before_clip,
        "uw_area_after_clip": uw_area_after_clip,
        "uw_max_feasible_area_abs": uw_max_feasible_area_abs,
        "uw_infeasible_under_rho_cap": uw_infeasible_under_rho_cap,
        "uw_clipped_fraction": uw_clipped_fraction,
        "rho_uw_abs_max": rho_uw_abs_max,
        "resolved_fmin_array": resolved_fmin_array,
        "resolved_fmax_array": resolved_fmax_array,
    }

    return uw_stress_realised, c_uw, diagnostics


def validate_cospectral_target_files(freq_array, spectra_array, c_uw_array, uw_stress_array, rho_max):
    """Raise a clear error if the generated co-spectral targets are inconsistent."""
    f = np.asarray(freq_array, dtype=float).reshape(-1)
    spectra_array = np.asarray(spectra_array, dtype=float)
    c_uw_array = np.asarray(c_uw_array, dtype=float)
    uw_stress_array = np.asarray(uw_stress_array, dtype=float).reshape(-1)
    
    n_heights = c_uw_array.shape[0]
    n_freq = c_uw_array.shape[1]
    
    if spectra_array.shape == (3, n_heights, n_freq):
        spectra_array = np.transpose(spectra_array, (1, 0, 2))
    elif spectra_array.shape == (n_heights, 3, n_freq):
        pass
    else:
        raise ValueError(
            "spectra_array must have shape (3, nHeights, nFreq) or "
            "(nHeights, 3, nFreq). "
            f"Got {spectra_array.shape}"
        )
    
    Su = np.maximum(spectra_array[:, 0, :], FLOOR)
    Sw = np.maximum(spectra_array[:, 2, :], FLOOR)

    rho = c_uw_array / np.maximum(np.sqrt(Su * Sw), FLOOR)
    area = _trapz(c_uw_array, f, axis=1)

    max_abs_rho = np.nanmax(np.abs(rho))
    max_area_error = np.nanmax(np.abs(area - uw_stress_array))

    if max_abs_rho > rho_max * (1.0 + 1e-8):
        raise RuntimeError(f"Cuw violates local rho cap: max |rho_uw| = {max_abs_rho:.6g}")

    if max_area_error > 1e-6 * max(1.0, np.nanmax(np.abs(uw_stress_array))):
        raise RuntimeError(
            "Cuw integral does not match written uwStress: "
            f"max abs error = {max_area_error:.6g}"
        )

    return {
        "max_abs_rho": float(max_abs_rho),
        "max_area_error": float(max_area_error),
        "fraction_abs_rho_gt_0p9": float(np.mean(np.abs(rho) > 0.9)),
    }


#%%

# ---------------------------------------------------------------------------
# Case setup
# ---------------------------------------------------------------------------

variable_dict = LES._caseFiles.parse_setup_file(case_path)

building_height = variable_dict["buildingHeight"]
mesh_size = variable_dict["meshSize"]
fMax = variable_dict["fMax"]
nFreq = variable_dict["nFreq"]

json_path = os.path.join(case_path, "log", "downstreamCalibration", "sim_init.json")
with open(json_path, "r") as f:
    dfsr_les_init_dict = json.load(f)

burn_in_time = dfsr_les_init_dict["burn_in_time"]

#%%

# ---------------------------------------------------------------------------
# Load target wind-tunnel profiles
# ---------------------------------------------------------------------------

vel_array_3d = LES._windTunnel.get_nheri_vel_time_series(approach_flow_data)

target_profile_df = LES._windTunnel.get_nheri_profile_df(approach_flow_data)

# Keep a copy of the raw experimental profile exactly as loaded from the
# NHERI/BLWT data file. This is useful for plotting measured points against
# smoothed/mapped targets later on MeluXina.
raw_experimental_profile_df = target_profile_df.copy()

int_length_scales = LES._windTunnel.calc_nheri_int_length_scales(vel_array_3d)

target_profile_df = LES._windTunnel.add_nheri_int_length_scales(
    target_profile_df,
    int_length_scales,
)

target_profile_df = LES._windTunnel.add_nheri_reynolds_stresses(
    target_profile_df,
    vel_array_3d,
    ddof=0,
)

# Measured experimental profile with turbulence statistics added, before any
# vertical extension. This is the preferred file for plotting experimental
# data points.
measured_experimental_profile_df = target_profile_df.copy()

target_profile_df = LES._windTunnel.extend_nheri_profiles_with_reynolds_stress(
    target_profile_df,
    6,
    fit_zmin=None,
    fit_zmax=None,
    uw_extension="constant_correlation",
)

# Extended profile on the experimental z-grid, before smoothing/mapping.
extended_experimental_profile_df = target_profile_df.copy()

z_old = target_profile_df["z"].to_numpy(dtype=float)

z_centres = LES._caseFiles.get_inlet_cell_centres(
    case_path,
    foam_file_for_inlet_centres,
    inlet_patch_name,
)

smoothed_target_profile_df = LES._profileAnalysis.smooth_profiles(
    target_profile_df,
    z_old,
    5,
    7,
    0.5,
)

# Ensure uwStress survives smoothing.
if "uwStress" not in smoothed_target_profile_df.columns:
    smoothed_target_profile_df["uwStress"] = target_profile_df["uwStress"].to_numpy(dtype=float)
if "uw" not in smoothed_target_profile_df.columns:
    smoothed_target_profile_df["uw"] = target_profile_df["uw"].to_numpy(dtype=float)

# Smoothed/extended target profile on the experimental z-grid.
smoothed_experimental_profile_df = smoothed_target_profile_df.copy()

uw_stress_mapped = np.interp(
    z_centres,
    smoothed_target_profile_df["z"].to_numpy(dtype=float),
    smoothed_target_profile_df["uwStress"].to_numpy(dtype=float),
)

# Convert to the internal profile array.  The first seven columns are the
# classic DFSR layout: U, uu, vv, ww, Lu, Lv, Lw.  The optional eighth column
# is signed uwStress.  map_profile_to_inlet_z was written for the classic
# seven-column array, so map those seven columns only and map uwStress
# separately by direct interpolation.
target_profile_array_full = LES._profileCalibration.convert_target_profile_df_to_array(
    smoothed_target_profile_df
)
target_profile_array_classic = target_profile_array_full[:, :7]

mapped_target_profile_array_classic = LES._profileAnalysis.map_profile_to_inlet_z(
    target_profile_array_classic,
    z_old,
    z_centres,
)

mapped_target_profile_df = LES._profileCalibration.convert_target_profile_array_to_df(
    z_centres,
    mapped_target_profile_array_classic,
)

mapped_target_profile_df["uwStress"] = uw_stress_mapped

target_profile_df = mapped_target_profile_df.copy()
# Full array for co-spectral profile writing and Kaimal stress targets.
target_profile_array = np.column_stack([mapped_target_profile_array_classic[:, :7], uw_stress_mapped])
# Classic seven-column array for resolved von Karman auto-spectrum construction.
target_profile_array_classic = mapped_target_profile_array_classic[:, :7].copy()
z_array = target_profile_df["z"].to_numpy(dtype=float)

#%%

# ---------------------------------------------------------------------------
# LES resolved-band setup
# ---------------------------------------------------------------------------

DT_STATS = 0.0025
T_SAMPLE_STATS = 45.0

time_steps = np.arange(
    0.0,
    T_SAMPLE_STATS + 0.5 * DT_STATS,
    DT_STATS,
)

U = target_profile_array[:, 0]
int_length_scales = target_profile_array[:, 4:7].T
sigmas = np.sqrt(target_profile_array[:, 1:4]).T

mesh_cutoff_freqs = LES._profileAnalysis.get_mesh_cutoff_frequencies(
    mesh_size,
    U,
    int_length_scales,
    sigmas,
)

freq_array_dfsr = LES._profileCalibration.get_freq_array(fMax, nFreq)

resolved_fmin_array = np.zeros(len(z_array), dtype=float)
resolved_fmax_array = np.zeros(len(z_array), dtype=float)

for h in range(len(z_array)):
    limits = LES._profileCalibration.get_resolved_frequency_limits(
        time_steps,
        mesh_cutoff_freq=float(mesh_cutoff_freqs[h]),
    )

    resolved_fmin_array[h] = float(np.asarray(limits["f_min"]).reshape(-1)[0])
    resolved_fmax_array[h] = float(np.asarray(limits["f_max"]).reshape(-1)[0])


#%%

# ---------------------------------------------------------------------------
# Auto-spectra: use existing resolved-band-consistent DFSR method
# ---------------------------------------------------------------------------

resolved_target_dir = os.path.join(
    case_path,
    "log",
    "coSpectralResolvedBandTargetSpectra",
)
os.makedirs(resolved_target_dir, exist_ok=True)

(
    target_spectra_array_resolved,
    target_spectra_array_original_vk,
    corrected_diagnostics,
    uncorrected_diagnostics,
    resolved_summary_df,
) = LES._profileCalibration.make_resolved_consistent_target_spectra_profile_dfsr(
    target_profile_array=target_profile_array_classic,
    time_steps=time_steps,
    fMax=fMax,
    nFreq=nFreq,
    mesh_cutoff_freqs=mesh_cutoff_freqs,
    n_freq_solve=N_FREQ_SOLVE,
    spacing=SOLVE_SPACING,
    tau_factor=TAU_FACTOR,
    floor=FLOOR,
    verbose=False,
)

#%%
    
# ---------------------------------------------------------------------------
# Co-spectrum: Kaimal shape, resolved-band normalised to measured uwStress(z)
# ---------------------------------------------------------------------------
#
# This local robust version deliberately replaces the older windlespy helper
# because the old output could:
#   - integrate to a different stress from the written uwStress column;
#   - force negative Cuw even where the target profile was positive;
#   - clip to rho_max without a consistent re-normalisation.
# ---------------------------------------------------------------------------

uw_stress_input = target_profile_df["uwStress"].to_numpy(dtype=float)

uw_stress_array, target_cuw_array, uw_diag = make_bounded_resolved_kaimal_uw_cospectra(
    target_profile_array=target_profile_array,
    z_array=z_array,
    freq_array=freq_array_dfsr,
    spectra_array=target_spectra_array_resolved,
    uw_stress_array=uw_stress_input,
    resolved_fmin_array=resolved_fmin_array,
    resolved_fmax_array=resolved_fmax_array,
    rho_max=UW_RHO_MAX,
    enforce_negative=ENFORCE_NEGATIVE_UW,
    floor=FLOOR,
)

# Use the actually realised/resolved-band bounded uwStress everywhere from here on.
# Important: files consumed by OpenFOAM/windlespy must contain only one
# Reynolds shear-stress column. Keep only "uwStress" in those files; do not
# also write a duplicate "uw" column.
target_profile_df["uwStress"] = uw_stress_array

dfsr_profile_columns = [
    "z", "U", "Iu", "Iv", "Iw", "Lu", "Lv", "Lw", "uwStress"
]
target_profile_for_dfsr_df = target_profile_df[dfsr_profile_columns].copy()

# Write ordinary DFSR/profile files only after the realised co-spectral
# uwStress is known. These files must have 8 columns, or 9 columns with a
# single uwStress column.
LES._caseFiles.write_dfsr_inlet_profile(target_profile_for_dfsr_df, case_path)
LES._caseFiles.write_target_dfsr_inlet_profile(target_profile_for_dfsr_df, case_path)

preflight = validate_cospectral_target_files(
    freq_array=freq_array_dfsr,
    spectra_array=target_spectra_array_resolved,
    c_uw_array=target_cuw_array,
    uw_stress_array=uw_stress_array,
    rho_max=UW_RHO_MAX,
)

print("Co-spectral preflight:")
print(f"  max |rho_uw|              = {preflight['max_abs_rho']:.6g}")
print(f"  max |integral(Cuw)-uw|    = {preflight['max_area_error']:.6g}")
print(f"  fraction |rho_uw| > 0.9   = {preflight['fraction_abs_rho_gt_0p9']:.6g}")
print(f"  infeasible heights count  = {np.sum(uw_diag['uw_infeasible_under_rho_cap'])}")

uw_summary_df = pd.DataFrame(
    {
        "z": z_array,
        "uwStress_input": uw_diag["uw_stress_input"],
        "uwStress_written": uw_diag["uw_stress_realised"],
        "uwStress_area_after_clip": uw_diag["uw_area_after_clip"],
        "uw_max_feasible_area_abs": uw_diag["uw_max_feasible_area_abs"],
        "uw_infeasible_under_rho_cap": uw_diag["uw_infeasible_under_rho_cap"],
        "uw_clipped_fraction": uw_diag["uw_clipped_fraction"],
        "rho_uw_abs_max": uw_diag["rho_uw_abs_max"],
    }
)

uw_summary_df.to_csv(
    os.path.join(resolved_target_dir, "uw_cospectrum_summary.csv"),
    index=False,
)

# Optional verbose diagnostics outside windProfile.  Disabled by default to keep
# the case folder clean; the required generator/calibration inputs are written
# below in windProfile.
if WRITE_VERBOSE_PROFILE_DIAGNOSTICS and not MINIMAL_OUTPUTS:
    raw_experimental_profile_df.to_csv(
        os.path.join(resolved_target_dir, "targetExperimentalProfile_raw.tsv"),
        sep="\t", header=True, index=False, float_format="%.12e",
    )
    measured_experimental_profile_df.to_csv(
        os.path.join(resolved_target_dir, "targetExperimentalProfile_withStats.tsv"),
        sep="\t", header=True, index=False, float_format="%.12e",
    )
    extended_experimental_profile_df.to_csv(
        os.path.join(resolved_target_dir, "targetExperimentalProfile_extended.tsv"),
        sep="\t", header=True, index=False, float_format="%.12e",
    )
    smoothed_experimental_profile_df.to_csv(
        os.path.join(resolved_target_dir, "targetSmoothedProfile.tsv"),
        sep="\t", header=True, index=False, float_format="%.12e",
    )
    target_profile_for_dfsr_df.to_csv(
        os.path.join(resolved_target_dir, "targetMappedProfile.tsv"),
        sep="\t", header=True, index=False, float_format="%.12e",
    )

#%%

# ---------------------------------------------------------------------------
# Write MannHybridTurb input files

#%%

# ---------------------------------------------------------------------------
# Write coSpectralDFSR input files
# ---------------------------------------------------------------------------

if WRITE_RESULTS:
    wind_profile_dir = os.path.join(
        case_path,
        "constant",
        "boundaryData",
        "windProfile",
    )
    os.makedirs(wind_profile_dir, exist_ok=True)

    spectra_path = os.path.join(wind_profile_dir, "spectraProfile")
    target_spectra_path = os.path.join(wind_profile_dir, "targetSpectraProfile")
    uw_path = os.path.join(wind_profile_dir, "uwCoSpectrumProfile")
    target_uw_path = os.path.join(wind_profile_dir, "targetUWCoSpectrumProfile")

    # Required active MannHybridTurb input and matching calibration target:
    # z uwStress Su Sv Sw
    for path in (spectra_path, target_spectra_path):
        LES._caseFiles.write_cospectral_spectra_profile(
            spectra_array=target_spectra_array_resolved,
            z_array=z_array,
            uw_stress_array=uw_stress_array,
            filepath=path,
            clip_min=1e-16,
        )

    # Required active MannHybridTurb input and matching calibration target:
    # z uwStress Cuw
    for path in (uw_path, target_uw_path):
        LES._caseFiles.write_uw_cospectrum_profile(
            c_uw_array=target_cuw_array,
            z_array=z_array,
            uw_stress_array=uw_stress_array,
            filepath=path,
        )

    # Optional legacy output for old DFSR inspection/rollback. Disabled by default.
    if WRITE_LEGACY_3COMP_SPECTRA and not MINIMAL_OUTPUTS:
        LES._caseFiles.write_spectra_profile(
            target_spectra_array_resolved,
            z_array,
            os.path.join(wind_profile_dir, "targetSpectraProfile_legacy3comp"),
            clip_min=1e-16,
        )

    # Keep one experimental profile for Melaku-style profile plots in the
    # downstream calibration scripts. This is measured data before extension,
    # smoothing and mapping.
    if WRITE_TARGET_EXPERIMENTAL_PROFILE:
        measured_experimental_profile_df.to_csv(
            os.path.join(wind_profile_dir, "targetExperimentalProfile"),
            sep="\t",
            header=True,
            index=False,
            float_format="%.12e",
        )

    # Optional verbose profile diagnostics. Disabled by default because they
    # clutter windProfile and are not required by MannHybridTurb.
    if WRITE_VERBOSE_PROFILE_DIAGNOSTICS and not MINIMAL_OUTPUTS:
        raw_experimental_profile_df.to_csv(
            os.path.join(wind_profile_dir, "targetExperimentalProfile_raw"),
            sep="\t", header=True, index=False, float_format="%.12e",
        )
        measured_experimental_profile_df.to_csv(
            os.path.join(wind_profile_dir, "targetExperimentalProfile_withStats"),
            sep="\t", header=True, index=False, float_format="%.12e",
        )
        extended_experimental_profile_df.to_csv(
            os.path.join(wind_profile_dir, "targetExperimentalProfile_extended"),
            sep="\t", header=True, index=False, float_format="%.12e",
        )
        smoothed_experimental_profile_df.to_csv(
            os.path.join(wind_profile_dir, "targetSmoothedProfile"),
            sep="\t", header=True, index=False, float_format="%.12e",
        )
        target_profile_for_dfsr_df.to_csv(
            os.path.join(wind_profile_dir, "targetMappedProfile"),
            sep="\t", header=True, index=False, float_format="%.12e",
        )

#%%

if WRITE_NUMPY_DIAGNOSTICS and not MINIMAL_OUTPUTS:
    np.savez_compressed(
        os.path.join(resolved_target_dir, "coSpectral_target_arrays.npz"),
        z_array=z_array,
        freq_array_dfsr=freq_array_dfsr,
        target_spectra_array_resolved=target_spectra_array_resolved,
        target_spectra_array_original_vk=target_spectra_array_original_vk,
        target_cuw_array=target_cuw_array,
        uw_stress_array=uw_stress_array,
        mesh_cutoff_freqs=mesh_cutoff_freqs,
    )

print("MannHybrid active and target spectra written to:")
print(os.path.join(case_path, "constant", "boundaryData", "windProfile", "spectraProfile"))
print(os.path.join(case_path, "constant", "boundaryData", "windProfile", "targetSpectraProfile"))

print("MannHybrid active and target uw co-spectrum written to:")
print(os.path.join(case_path, "constant", "boundaryData", "windProfile", "uwCoSpectrumProfile"))
print(os.path.join(case_path, "constant", "boundaryData", "windProfile", "targetUWCoSpectrumProfile"))

if not MINIMAL_OUTPUTS:
    print("Diagnostics written to:")
    print(resolved_target_dir)

#%%

def validate_written_profile_file(filepath):
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        n_cols = data.size
    else:
        n_cols = data.shape[1]

    print("Written profile validation:")
    print(f"  file = {filepath}")
    print(f"  columns = {n_cols}")

    if n_cols not in (8, 9):
        raise RuntimeError(
            f"Written profile file must have 8 columns, or 9 with uwStress. Found {n_cols}."
        )


def validate_written_uw_file(filepath, fMax, nFreq, tol=1e-6):
    data = np.loadtxt(filepath, skiprows=1)
    z = data[:, 0]
    uw_written = data[:, 1]
    c_uw = data[:, 2:]

    freq = np.arange(1, nFreq + 1, dtype=float) * float(fMax) / float(nFreq)

    area = _trapz(c_uw, freq, axis=1)
    err = area - uw_written

    print("Written UW file validation:")
    print(f"  max |area - uw| = {np.max(np.abs(err)):.6e}")
    print(f"  mean |area - uw| = {np.mean(np.abs(err)):.6e}")

    bad = np.argmax(np.abs(err))
    print(
        f"  worst row: i={bad}, z={z[bad]:.6g}, "
        f"uw={uw_written[bad]:.6g}, area={area[bad]:.6g}, "
        f"err={err[bad]:.6g}"
    )

    if np.max(np.abs(err)) > tol * max(1.0, np.max(np.abs(uw_written))):
        raise RuntimeError("Written targetUWCoSpectrumProfile is inconsistent.")
        
#%%

validate_written_profile_file(
    os.path.join(case_path, "constant", "boundaryData", "windProfile", "profile")
)

validate_written_profile_file(
    os.path.join(case_path, "constant", "boundaryData", "windProfile", "targetProfile")
)

validate_written_uw_file(
    target_uw_path,
    fMax=fMax,
    nFreq=nFreq,
)

#%%

LES._caseFiles.write_probes_from_target_profile(2.5, 0, case_path, target_profile_df, "probes2")
