# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:49:26 2026

@author: David Cunningham
"""

import pandas as pd 
import os 
import numpy as np 
from scipy.io import loadmat 
import sys

cwd = os.path.dirname(os.path.abspath(__file__))
windlespy_path = os.path.abspath(os.path.join(cwd, "..", ".."))
sys.path.append(windlespy_path)
import windlespy as LES
sys.path.remove(windlespy_path)

#%%

def get_nheri_profile_heights(data_path):
    
    data = loadmat(data_path) 
    
    data = data["Profile"]
    
    z_array = data["Position"][0,0]['Z_mm'][0,0]
    
    z_array = z_array[:,0] / 1000
    
    return z_array

#%%

def get_nheri_vel_time_series(data_path):
    
    data = loadmat(data_path) 
    
    data = data["Profile"]
    
    vel_time_series = data["TimeSeries"][0,0]
    
    u = vel_time_series["u"][0,0][:,0]
    v = vel_time_series["v"][0,0][:,0]
    w = vel_time_series["w"][0,0][:,0]
    
    vel_array_dict = {}
    vel_array_dict["u"] = []
    vel_array_dict["v"] = []
    vel_array_dict["w"] = []
    
    for height in range(0,np.shape(u)[0]):
        
        vel_array_dict["u"].append(u[height].squeeze())
        vel_array_dict["v"].append(v[height].squeeze())
        vel_array_dict["w"].append(w[height].squeeze())
        
    u_array = np.stack(vel_array_dict["u"], axis =0)
    v_array = np.stack(vel_array_dict["v"], axis =0)
    w_array = np.stack(vel_array_dict["w"], axis =0)
    
    vel_array_3d = np.stack([u_array,v_array,w_array], axis= 0)
                            
    vel_array_3d = np.transpose(vel_array_3d, (0, 2, 1))
    
    return vel_array_3d
    
#%%

def get_nheri_profile_df(data_path):
    
    data = loadmat(data_path) 
    
    data = data["Profile"]
    
    stats = data["Statistics"][0,0]
    
    stat_array_dict = {"z" : get_nheri_profile_heights(data_path),
                       "U" : stats["U"][0,0].squeeze(),
                       "Iu" : stats["Iu"][0,0].squeeze(), 
                       "Iv" : stats["Iv"][0,0].squeeze(), 
                       "Iw" : stats["Iw"][0,0].squeeze(), 
                       "Lu" : stats["Lux"][0,0].squeeze(), 
                       "Lv" : np.full(np.shape(stats["Iu"][0,0].squeeze()), np.nan), 
                       "Lw" : np.full(np.shape(stats["Iu"][0,0].squeeze()), np.nan)}
    
    profile_df = pd.DataFrame(stat_array_dict)
    
    return profile_df


#%%

def calc_nheri_int_length_scales(vel_array_3d, time_step=1/1250):
    
    mean_vel_array_2d = LES._profileAnalysis.mean_vel(vel_array_3d)
    
    fluc_vel_array_3d = LES._profileAnalysis.fluc_vel(vel_array_3d, mean_vel_array_2d)
    
    int_time_scales = LES._profileAnalysis.int_time_scale(fluc_vel_array_3d, time_step)
    
    int_length_scales = LES._profileAnalysis.int_length_scales(int_time_scales, mean_vel_array_2d)
    
    return int_length_scales
    
#%%

def add_nheri_int_length_scales(profile_df, int_length_scales):
    
    if "Lu" not in profile_df.columns:
        Lu = int_length_scales[0,:]
        profile_df["Lu"] = Lu
            
    Lv = int_length_scales[1,:]
    Lw = int_length_scales[2,:]
    
    profile_df["Lv"] = Lv
    profile_df["Lw"] = Lw
    
    return profile_df
    
#%%

def extend_nheri_profiles(profile_df, z_top, fit_zmin=None, fit_zmax=None):
    
    z_meas = profile_df["z"].to_numpy(dtype=float)
    z_m = np.nanmax(z_meas)
    
    # fit log law only on chosen upper region
    if (fit_zmin!=None) and (fit_zmax!=None):
        u_star, z0 = LES._profileAnalysis.fit_log_law(
            profile_df["U"].to_numpy(),
            profile_df["z"].to_numpy(),
            zmin=fit_zmin,
            zmax=fit_zmax,
        )
    else:
        u_star, z0 = LES._profileAnalysis.fit_log_law(
            profile_df["U"].to_numpy(),
            profile_df["z"].to_numpy(),
            zmin=0.8*z_m,
            zmax=z_m,
        )
        
    # create extra z values using same spacing as the existing profile if possible
    dzs = np.diff(z_meas[np.isfinite(z_meas)])
    dz = np.nanmedian(dzs) if len(dzs) > 0 else 0.02
    
    z_extra = np.arange(z_m + dz, z_top + 0.5 * dz, dz)
    
    # top measured values
    top = profile_df.loc[profile_df["z"].idxmax()]
    U_m  = float(top["U"])
    Iu_m = float(top["Iu"])
    Iv_m = float(top["Iv"])
    Iw_m = float(top["Iw"])
    Lu_m = float(top["Lu"])
    Lv_m = float(top["Lv"])
    Lw_m = float(top["Lw"])
    
    # log-law extension for U
    U_extra = (u_star / 0.41) * np.log(z_extra / z0)
    
    # optional: rescale so extension matches exactly at z_m
    U_fit_at_zm = (u_star / 0.41) * np.log(z_m / z0)
    U_extra *= U_m / U_fit_at_zm
    
    # Dyrbye-Hansen style extension for Iu
    # Iu(z) = Iu(z_m) * ln(z_m/z0) / ln(z/z0)
    log_m = np.log(z_m / z0)
    log_z = np.log(z_extra / z0)
    Iu_extra = Iu_m * (log_m / log_z)
    
    # preserve anisotropy ratios at top measurement
    eps = 1e-12
    rv = Iv_m / max(Iu_m, eps)
    rw = Iw_m / max(Iu_m, eps)
    
    Iv_extra = rv * Iu_extra
    Iw_extra = rw * Iu_extra
    
    # extend length scales
    Lu_extra = np.full_like(z_extra, Lu_m, dtype=float)
    Lv_extra = np.full_like(z_extra, Lv_m, dtype=float)
    Lw_extra = np.full_like(z_extra, Lw_m, dtype=float)
    
    extra = pd.DataFrame({
        "z":  z_extra,
        "U":  U_extra,
        "Iu": Iu_extra,
        "Iv": Iv_extra,
        "Iw": Iw_extra,
        "Lu": Lu_extra,
        "Lv": Lv_extra,
        "Lw": Lw_extra,
    })
    
    profile_df = pd.concat([profile_df, extra], ignore_index=True)
    
    return profile_df


#%%

def get_tap_coord_df(excel_path, coordinate_units="mm"):
    tap_coord_df = pd.read_excel(excel_path, sheet_name="Taps")

    tap_coord_df = tap_coord_df[
        ~(
            pd.isna(tap_coord_df["X"])
            | pd.isna(tap_coord_df["Y"])
            | pd.isna(tap_coord_df["Z"])
        )
    ].copy()

    tap_coord_df.rename(columns={"X": "x", "Y": "y", "Z": "z"}, inplace=True)

    if coordinate_units.lower() in ("mm", "millimetres", "millimeters"):
        tap_coord_df[["x", "y", "z"]] = tap_coord_df[["x", "y", "z"]] / 1000.0
    elif coordinate_units.lower() in ("m", "metres", "meters"):
        pass
    else:
        raise ValueError("coordinate_units must be 'mm' or 'm'.")

    return tap_coord_df
    
 #%%

def write_surf_pressure_probe_files(case_path, tap_coord_df, patch_name):

    field = "p"
    
    for surface_num in tap_coord_df["Surface"].unique():
        surf_tap_df = tap_coord_df[tap_coord_df["Surface"]==surface_num]
        surf_str = "Surface"+str(surface_num)[0]
        LES._caseFiles.write_surf_presure_probes(field, surf_tap_df, patch_name, case_path, f"probes{surf_str}")
        
#%%

def calc_nheri_reynolds_stresses(vel_array_3d, ddof=0):
    """
    Calculate Reynolds stress profiles from NHERI velocity time series.

    Parameters
    ----------
    vel_array_3d : ndarray
        Velocity array with shape (3, nTime, nHeights), as returned by
        get_nheri_vel_time_series().
    ddof : int
        Delta degrees of freedom for covariance averaging.
        ddof=0 gives population mean, ddof=1 gives sample covariance.

    Returns
    -------
    reynolds_stress_df : pandas.DataFrame
        Columns:
            uu, vv, ww, uv, uw, vw
    """

    vel = np.asarray(vel_array_3d, dtype=float)

    if vel.ndim != 3 or vel.shape[0] != 3:
        raise ValueError(
            "vel_array_3d must have shape (3, nTime, nHeights)."
        )

    mean_vel = np.nanmean(vel, axis=1, keepdims=True)
    fluc = vel - mean_vel

    u = fluc[0, :, :]
    v = fluc[1, :, :]
    w = fluc[2, :, :]

    if ddof == 0:
        norm = np.sum(np.isfinite(u), axis=0)
    else:
        norm = np.maximum(np.sum(np.isfinite(u), axis=0) - ddof, 1)

    def cov(a, b):
        good = np.isfinite(a) & np.isfinite(b)
        num = np.nansum(np.where(good, a * b, 0.0), axis=0)
        den = np.maximum(np.sum(good, axis=0) - ddof, 1)
        return num / den

    reynolds_stress_df = pd.DataFrame(
        {
            "uu": cov(u, u),
            "vv": cov(v, v),
            "ww": cov(w, w),
            "uv": cov(u, v),
            "uw": cov(u, w),
            "vw": cov(v, w),
        }
    )

    return reynolds_stress_df

#%%

def add_nheri_reynolds_stresses(profile_df, vel_array_3d, ddof=0):
    """
    Add Reynolds stress columns to an NHERI profile dataframe.

    Adds:
        uu, vv, ww, uv, uw, vw, uwStress

    uwStress is duplicated from uw because CoSpectralDFSRTurb uses that name.
    """

    profile_df = profile_df.copy()
    rs_df = calc_nheri_reynolds_stresses(vel_array_3d, ddof=ddof)

    if len(rs_df) != len(profile_df):
        raise ValueError(
            f"Reynolds stress length ({len(rs_df)}) does not match "
            f"profile length ({len(profile_df)})."
        )

    for col in rs_df.columns:
        profile_df[col] = rs_df[col].to_numpy(dtype=float)

    profile_df["uwStress"] = profile_df["uw"]

    return profile_df

#%%

def extend_nheri_profiles_with_reynolds_stress(
    profile_df,
    z_top,
    fit_zmin=None,
    fit_zmax=None,
    uw_extension="constant_correlation",
):
    """
    Extend NHERI mean/TI/length-scale profiles and preserve Reynolds stresses.

    This wraps extend_nheri_profiles(), then fills uv/uw/vw above the measured
    region.

    uw_extension options
    --------------------
    "constant_correlation":
        Preserve the top measured correlation coefficient:
            rho_uw = uw / sqrt(uu * ww)
        and apply it to the extended uu and ww.

    "constant_stress":
        Hold uw, uv, vw fixed at their top measured values.

    Notes
    -----
    extend_nheri_profiles() currently extends U, Iu, Iv, Iw, Lu, Lv, Lw.
    Since uu/vv/ww are implied by (I * U)^2, this function recomputes
    uu/vv/ww after extension.
    """

    input_df = profile_df.copy()
    extended = extend_nheri_profiles(
        input_df,
        z_top=z_top,
        fit_zmin=fit_zmin,
        fit_zmax=fit_zmax,
    )

    # Recompute normal stresses from extended U and turbulence intensities.
    extended["uu"] = (extended["Iu"] * extended["U"]) ** 2
    extended["vv"] = (extended["Iv"] * extended["U"]) ** 2
    extended["ww"] = (extended["Iw"] * extended["U"]) ** 2

    measured = input_df.copy()
    top = measured.loc[measured["z"].idxmax()]

    has_cross = all(col in measured.columns for col in ["uv", "uw", "vw"])

    if not has_cross:
        extended["uv"] = np.nan
        extended["uw"] = np.nan
        extended["vw"] = np.nan
        extended["uwStress"] = np.nan
        return extended

    z_m = float(top["z"])
    measured_mask = extended["z"] <= z_m + 1e-12
    extra_mask = ~measured_mask

    # Preserve measured stresses over the measured region by interpolation,
    # because extend_nheri_profiles concatenates the original measured rows.
    for col in ["uv", "uw", "vw"]:
        extended.loc[measured_mask, col] = np.interp(
            extended.loc[measured_mask, "z"].to_numpy(dtype=float),
            measured["z"].to_numpy(dtype=float),
            measured[col].to_numpy(dtype=float),
        )

    if uw_extension == "constant_stress":
        for col in ["uv", "uw", "vw"]:
            extended.loc[extra_mask, col] = float(top[col])

    elif uw_extension == "constant_correlation":
        eps = 1e-30

        rho_uv_top = float(top["uv"]) / np.sqrt(
            max(float(top["uu"]) * float(top["vv"]), eps)
        )
        rho_uw_top = float(top["uw"]) / np.sqrt(
            max(float(top["uu"]) * float(top["ww"]), eps)
        )
        rho_vw_top = float(top["vw"]) / np.sqrt(
            max(float(top["vv"]) * float(top["ww"]), eps)
        )

        extended.loc[extra_mask, "uv"] = rho_uv_top * np.sqrt(
            extended.loc[extra_mask, "uu"] *
            extended.loc[extra_mask, "vv"]
        )
        extended.loc[extra_mask, "uw"] = rho_uw_top * np.sqrt(
            extended.loc[extra_mask, "uu"] *
            extended.loc[extra_mask, "ww"]
        )
        extended.loc[extra_mask, "vw"] = rho_vw_top * np.sqrt(
            extended.loc[extra_mask, "vv"] *
            extended.loc[extra_mask, "ww"]
        )

    else:
        raise ValueError(
            "uw_extension must be 'constant_correlation' or 'constant_stress'."
        )

    extended["uwStress"] = extended["uw"]

    return extended

#%%

