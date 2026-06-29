# -*- coding: utf-8 -*-
"""
Build MannHybridTurb-compatible windProfile files from NHERI approach-flow data.

This version intentionally produces *multiple* low-frequency target variants
(plateau/raw/blendRaw/free) so they can be compared and copied into the active
MannHybridTurb input folder.  The active main profile/spectra files are written
using WT_LOW_FREQ_MODE, but all variants are also written under

    constant/boundaryData/windProfile/lowFrequencyVariants/<mode>/

and plotted under

    Figures/targetSpectra/windTunnelFitted/<mode>/

The profile smoothing follows David's original NHERI target-spectra workflow:

    extend_nheri_profiles(..., 3, ...)
    smooth_profiles(target_profile_df, z_array, 5, 7, H)
    map_profile_to_inlet_z(...)

unless overridden by environment variables.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mann_calibration_common import (
    write_auto_spectra,
    write_uw_cospectrum,
    write_profile,
    profile_to_internal_array,
    internal_array_to_profile,
    estimate_downstream_spectra,
    smooth_spectra_array,
    smooth_cospectrum_array,
    freq_array_from_fmax,
    safe_makedirs,
    multitaper_psd,
    multitaper_csd_xy,
    apply_low_frequency_shape,
    _trapz,
)

# =============================================================================
# User / environment settings
# =============================================================================

CASE_DIR = os.environ.get(
    "CASE_DIR",
    r"C:\Users\david\OneDrive\Documents\PhD\Year 1\NHERI LES Case\OpenFOAM Cases\spectalCalibrationWTSpectra",
)
APPROACH_FLOW_MAT = os.environ.get(
    "NHERI_APPROACH_FLOW_MAT",
    r"C:\Users\david\OneDrive\Documents\PhD\Year 1\Wind Tunnel Test Data\NHERI BLWT Tall Building\Approach Flow\Approach Flow - EH160 - Marine Spires - 1200 RPM - 091721_1028.mat",
)
WINDLESPY_ROOT = os.environ.get(
    "WINDLESPY_ROOT",
    r"C:\Users\david\OneDrive\Documents\PhD\Year 1",
)

FMAX = float(os.environ.get("MANN_TARGET_FMAX", "200.0"))
NFREQ = int(os.environ.get("MANN_TARGET_NFREQ", "4096"))
WT_DT = float(os.environ.get("NHERI_DT", str(1.0 / 1250.0)))
BODY_HEIGHT = float(os.environ.get("BUILDING_HEIGHT", "0.5"))

# Preserve the tuned smoothing from the user's original NHERI target script:
# smoothed_target_profile_df = LES._profileAnalysis.smooth_profiles(target_profile_df, z_array, 5, 7, 0.5)
PROFILE_SMOOTH_WINDOW = int(os.environ.get("PROFILE_SMOOTH_WINDOW", "5"))
PROFILE_SMOOTH_POLYORDER = int(os.environ.get("PROFILE_SMOOTH_POLYORDER", "7"))
PROFILE_SMOOTH_H = float(os.environ.get("PROFILE_SMOOTH_H", str(BODY_HEIGHT)))
EXTEND_PROFILE_FACTOR = float(os.environ.get("EXTEND_PROFILE_FACTOR", "3"))
MAP_PROFILE_TO_INLET_Z = os.environ.get("MAP_PROFILE_TO_INLET_Z", "true").lower() in ("1", "true", "yes", "on")
INLET_FOAM_FILE = os.environ.get("INLET_FOAM_FILE", "spectralDFSR.foam")
INLET_PATCH_NAME = os.environ.get("INLET_PATCH_NAME", "inlet")

INTERPOLATE_MEASURED_SPECTRA_TO_ALL_HEIGHTS = os.environ.get(
    "INTERPOLATE_MEASURED_SPECTRA_TO_ALL_HEIGHTS", "true"
).lower() in ("1", "true", "yes", "on")
SMOOTH_TARGET_SPECTRA = os.environ.get("SMOOTH_TARGET_SPECTRA", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
SMOOTH_BINS = int(os.environ.get("SMOOTH_BINS", "40"))
LOW_PLATEAU_HZ = float(os.environ.get("WT_LOW_PLATEAU_HZ", "0.20"))
WT_LOW_FREQ_MODE = os.environ.get("WT_LOW_FREQ_MODE", os.environ.get("WT_LOW_FREQUENCY_MODE", "blendRaw"))
TAIL_AFTER_HZ = float(os.environ.get("WT_TAIL_AFTER_HZ", str(0.65 * FMAX)))

# Generate all these variants every run.  The selected WT_LOW_FREQ_MODE is copied
# to the active windProfile files, but the others are also saved for testing.
WT_LOW_FREQ_VARIANTS = os.environ.get(
    "WT_LOW_FREQ_VARIANTS",
    os.environ.get("WT_LOW_FREQ_MODES", "plateau,raw,blendRaw,free"),
)
WRITE_VARIANT_SUBFOLDERS = os.environ.get("WRITE_VARIANT_SUBFOLDERS", "true").lower() in ("1", "true", "yes", "on")

# Limit Cuw by realizability before writing. This avoids impossible co-spectra
# if the measured/smoothed Cuw briefly exceeds sqrt(Suu*Sww).
MAX_UW_COHERENCE = float(os.environ.get("MAX_UW_COHERENCE", "0.98"))

# Diagnostic plots
WRITE_PLOTS = os.environ.get("WRITE_WT_SPECTRA_PLOTS", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
PLOT_HEIGHTS = os.environ.get("WT_PLOT_HEIGHTS", "0.1,0.25,0.5,1.0,1.5")
WRITE_PROFILE_8PANEL = os.environ.get("WRITE_WT_PROFILE_8PANEL", "true").lower() in ("1", "true", "yes", "on")

sys.path.append(WINDLESPY_ROOT)
try:
    import windlespy as LES  # noqa: N806
finally:
    try:
        sys.path.remove(WINDLESPY_ROOT)
    except ValueError:
        pass


# =============================================================================
# Generic helpers
# =============================================================================

def _assert_finite(name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr, float)
    if not np.all(np.isfinite(arr)):
        n_bad = int(np.size(arr) - np.count_nonzero(np.isfinite(arr)))
        raise FloatingPointError(f"{name} contains {n_bad} non-finite values")


def parse_variants(s: str) -> list[str]:
    aliases = {
        "flat": "plateau",
        "constant": "plateau",
        "measured": "raw",
        "native": "raw",
        "blend": "blendRaw",
        "rawblend": "blendRaw",
        "none": "free",
        "off": "free",
        "pchip": "free",
    }
    out = []
    for token in s.replace(";", ",").split(","):
        t = token.strip()
        if not t:
            continue
        key = aliases.get(t.lower(), t)
        # Normalize canonical case.
        if key.lower() == "blendraw":
            key = "blendRaw"
        elif key.lower() == "plateau":
            key = "plateau"
        elif key.lower() == "raw":
            key = "raw"
        elif key.lower() == "free":
            key = "free"
        if key not in out:
            out.append(key)
    if not out:
        out = ["plateau", "raw", "blendRaw", "free"]
    return out




def _finite_interpolate_column(z: np.ndarray, y: np.ndarray, z_out: np.ndarray) -> np.ndarray:
    """Finite, monotone-safe 1D interpolation with endpoint hold."""
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    z_out = np.asarray(z_out, float)
    m = np.isfinite(z) & np.isfinite(y)
    if np.count_nonzero(m) == 0:
        return np.zeros_like(z_out, dtype=float)
    z_m = z[m]
    y_m = y[m]
    order = np.argsort(z_m)
    z_m = z_m[order]
    y_m = y_m[order]
    # Remove duplicate z values by median.
    uniq = []
    vals = []
    for zu in np.unique(z_m):
        uniq.append(zu)
        vals.append(float(np.median(y_m[np.isclose(z_m, zu)])))
    z_m = np.asarray(uniq, float)
    y_m = np.asarray(vals, float)
    if len(z_m) == 1:
        return np.full_like(z_out, y_m[0], dtype=float)
    return np.interp(z_out, z_m, y_m, left=y_m[0], right=y_m[-1])


def _sanitize_profile_df(df: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    """Remove inf/NaN from columns before calling windlespy smoothing/plotting."""
    out = df.copy()
    z = out["z"].to_numpy(float)
    for col in required_cols:
        arr = pd.to_numeric(out[col], errors="coerce").to_numpy(float)
        if not np.all(np.isfinite(arr)):
            print(f"Warning: replacing non-finite values in profile column {col!r} before smoothing")
            arr = _finite_interpolate_column(z, arr, z)
        # Conservative physical floors for classic profile quantities.
        if col in ("U", "Iu", "Iv", "Iw", "Lu", "Lv", "Lw"):
            floor = 1e-12 if col == "U" else 1e-8
            arr = np.maximum(arr, floor)
        out[col] = arr
    return out


def _smooth_uw_stress_profile(z_raw: np.ndarray, uw_raw: np.ndarray, z_out: np.ndarray, window: int = 5) -> np.ndarray:
    """Create a smooth signed u'w' profile without passing it to smooth_profiles.

    David's tuned smooth_profiles call is retained exactly for U, I and L.  The
    signed shear-stress column is handled separately because windlespy's profile
    plotter/smoother was written around positive classic DFSR profile quantities
    and can fail if a signed or extrapolated column creates invalid axis limits.
    """
    y = _finite_interpolate_column(z_raw, uw_raw, z_out)
    if len(y) < 3 or window <= 1:
        return y
    window = int(max(3, window))
    if window % 2 == 0:
        window += 1
    pad = window // 2
    yp = np.pad(y, pad, mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    ys = np.convolve(yp, kernel, mode="valid")
    # Preserve sign convention and avoid artificial sign flips due to smoothing.
    if np.nanmedian(uw_raw) <= 0:
        ys = np.minimum(ys, 0.0)
    return ys

def interp_spectra_in_height(z_meas: np.ndarray, S_meas: np.ndarray, z_target: np.ndarray) -> np.ndarray:
    """Interpolate positive spectra in height, independently for component/frequency."""
    S_out = np.zeros((S_meas.shape[0], len(z_target), S_meas.shape[2]), dtype=float)
    order = np.argsort(z_meas)
    z_m = np.asarray(z_meas, float)[order]
    S_m = np.asarray(S_meas, float)[:, order, :]
    for c in range(S_meas.shape[0]):
        for k in range(S_meas.shape[2]):
            vals = np.maximum(S_m[c, :, k], 1e-16)
            S_out[c, :, k] = np.exp(
                np.interp(
                    z_target,
                    z_m,
                    np.log(vals),
                    left=np.log(vals[0]),
                    right=np.log(vals[-1]),
                )
            )
    return S_out


def interp_signed_in_height(z_meas: np.ndarray, C_meas: np.ndarray, z_target: np.ndarray) -> np.ndarray:
    order = np.argsort(z_meas)
    z_m = np.asarray(z_meas, float)[order]
    C_m = np.asarray(C_meas, float)[order, :]
    C_out = np.zeros((len(z_target), C_m.shape[1]), dtype=float)
    for k in range(C_m.shape[1]):
        C_out[:, k] = np.interp(z_target, z_m, C_m[:, k], left=C_m[0, k], right=C_m[-1, k])
    return C_out


def enforce_uw_realizability(Cuw: np.ndarray, S: np.ndarray, max_rho: float = 0.98) -> np.ndarray:
    Su = np.maximum(S[0], 1e-16)
    Sw = np.maximum(S[2], 1e-16)
    bound = max_rho * np.sqrt(Su * Sw)
    return np.clip(Cuw, -bound, bound)


def scale_auto_spectra_to_variance(S: np.ndarray, freq: np.ndarray, var_target: np.ndarray) -> np.ndarray:
    """Scale each spectrum to match the smoothed/input profile variances.

    This preserves David's tuned profile smoothing and makes the MannHybridTurb
    spectra consistent with the input Iu/Iv/Iw values, instead of letting noisy WT
    spectral integrals overwrite the smoothed profile.
    """
    out = np.asarray(S, float).copy()
    var_target = np.asarray(var_target, float)
    for c in range(out.shape[0]):
        for h in range(out.shape[1]):
            area = _trapz(np.maximum(out[c, h], 1e-30), freq)
            target = max(float(var_target[h, c]), 1e-16)
            if np.isfinite(area) and area > 1e-30:
                out[c, h] *= target / area
            else:
                out[c, h] = target / max(freq[-1] - freq[0], 1e-12)
    return np.maximum(out, 1e-16)


def scale_cuw_to_profile(Cuw: np.ndarray, freq: np.ndarray, uw_target: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Rescale Cuw shape to match target integral, but preserve realizability."""
    out = np.asarray(Cuw, float).copy()
    for i in range(out.shape[0]):
        ci = out[i]
        integral = _trapz(ci, freq)
        target = float(uw_target[i])
        if not np.isfinite(integral) or abs(integral) < 1e-14 or not np.isfinite(target):
            # Use a bounded constant-correlation fallback based on target sign.
            sign = -1.0 if target <= 0 else 1.0
            shape = sign * np.sqrt(np.maximum(S[0, i, :], 1e-16) * np.maximum(S[2, i, :], 1e-16))
            shape = enforce_uw_realizability(shape[None, :], S[:, i:i+1, :], MAX_UW_COHERENCE)[0]
            integral = _trapz(shape, freq)
            ci = shape
        if abs(integral) > 1e-14:
            ci = ci * (target / integral)
        out[i] = ci
    out = enforce_uw_realizability(out, S, MAX_UW_COHERENCE)
    return out


def select_plot_indices(z: np.ndarray) -> list[int]:
    vals = []
    for token in PLOT_HEIGHTS.replace(";", ",").split(","):
        token = token.strip()
        if token:
            vals.append(float(token))
    if not vals:
        vals = [float(z[0]), float(z[len(z)//2]), float(z[-1])]
    idx = []
    for zh in vals:
        idx.append(int(np.argmin(np.abs(z - zh))))
    return sorted(set(idx))


# =============================================================================
# Diagnostic plotting
# =============================================================================

def plot_spectra_diagnostics(fig_dir: Path, freq: np.ndarray, z_meas: np.ndarray, S_raw: np.ndarray, C_raw: np.ndarray, z_final: np.ndarray, S_final: np.ndarray, C_final: np.ndarray, title_suffix: str = "") -> None:
    safe_makedirs(fig_dir)
    comp_names = ["u", "v", "w"]
    idx_final = select_plot_indices(z_final)

    for c, name in enumerate(comp_names):
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        for j in idx_final:
            z0 = z_final[j]
            jm = int(np.argmin(np.abs(z_meas - z0)))
            ax.loglog(freq, np.maximum(S_final[c, j], 1e-16), lw=2.0, label=f"final z={z0:.3g} m")
            ax.loglog(freq, np.maximum(S_raw[c, jm], 1e-16), lw=0.8, alpha=0.45, ls="--", label=f"raw-grid z≈{z_meas[jm]:.3g} m")
        ax.axvline(LOW_PLATEAU_HZ, color="0.35", lw=0.8, ls=":", label="LF join" if c == 0 else None)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(f"S{name}{name} [m²/s]")
        ax.set_title(f"NHERI spectra used by MannHybridTurb: {name}-component{title_suffix}")
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(fig_dir / f"raw_vs_final_spectrum_{name}.png", dpi=250)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for j in idx_final:
        z0 = z_final[j]
        jm = int(np.argmin(np.abs(z_meas - z0)))
        ax.semilogx(freq, C_final[j], lw=2.0, label=f"final z={z0:.3g} m")
        ax.semilogx(freq, C_raw[jm], lw=0.8, alpha=0.45, ls="--", label=f"raw-grid z≈{z_meas[jm]:.3g} m")
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.axvline(LOW_PLATEAU_HZ, color="0.35", lw=0.8, ls=":", label="LF join")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Cuw [m²/s]")
    ax.set_title(f"NHERI u-w co-spectrum used by MannHybridTurb{title_suffix}")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "raw_vs_final_uw_cospectrum.png", dpi=250)
    plt.close(fig)

    # Integral/profile consistency plot.
    var = _trapz(S_final, freq, axis=-1).T
    uw = _trapz(C_final, freq, axis=-1)
    fig, axes = plt.subplots(1, 4, figsize=(13, 4), sharey=True)
    labels = ["var(u)", "var(v)", "var(w)", "int Cuw"]
    data = [var[:, 0], var[:, 1], var[:, 2], uw]
    for ax, lab, x in zip(axes, labels, data):
        ax.plot(x, z_final, "-o", ms=2.5)
        if lab == "int Cuw":
            ax.axvline(0, color="0.3", lw=0.8)
        ax.set_xlabel(lab)
        ax.grid(True, ls="--", alpha=0.35)
    axes[0].set_ylabel("z [m]")
    fig.suptitle(f"Resolved spectral integrals written to MannHybridTurb files{title_suffix}")
    fig.tight_layout()
    fig.savefig(fig_dir / "spectral_integrals_profile.png", dpi=250)
    plt.close(fig)


def collect_native_raw_spectra_for_plots(vel_array_3d: np.ndarray, fs: float, plot_height_indices: list[int]) -> dict:
    raw = {"auto": {}, "uw": {}}
    for h in plot_height_indices:
        raw["auto"][h] = []
        for c in range(3):
            f, p = multitaper_psd(vel_array_3d[c, :, h], fs=fs, time_bandwidth=3.5)
            raw["auto"][h].append((f, p))
        f, cxy = multitaper_csd_xy(vel_array_3d[0, :, h], vel_array_3d[2, :, h], fs=fs, time_bandwidth=3.5)
        raw["uw"][h] = (f, cxy)
    return raw


def plot_spectra_diagnostics_native(fig_dir: Path, freq: np.ndarray, z_meas: np.ndarray, vel_array_3d: np.ndarray, z_final: np.ndarray, S_final: np.ndarray, C_final: np.ndarray, title_suffix: str = "") -> None:
    safe_makedirs(fig_dir)
    comp_names = ["u", "v", "w"]
    idx_final = select_plot_indices(z_final)
    idx_meas = sorted(set(int(np.argmin(np.abs(z_meas - z_final[j]))) for j in idx_final))
    raw_native = collect_native_raw_spectra_for_plots(vel_array_3d, fs=1.0/WT_DT, plot_height_indices=idx_meas)

    for c, name in enumerate(comp_names):
        fig, ax = plt.subplots(figsize=(8.8, 5.5))
        for j in idx_final:
            z0 = z_final[j]
            jm = int(np.argmin(np.abs(z_meas - z0)))
            f_raw, s_raw = raw_native["auto"][jm][c]
            ax.loglog(f_raw, np.maximum(s_raw, 1e-16), lw=0.75, alpha=0.45, ls="--", label=f"raw native z≈{z_meas[jm]:.3g} m")
            ax.loglog(freq, np.maximum(S_final[c, j], 1e-16), lw=2.2, label=f"final input z={z0:.3g} m")
        ax.axvline(LOW_PLATEAU_HZ, color="0.35", lw=0.8, ls=":", label="LF join" if c == 0 else None)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(f"S{name}{name} [m²/s]")
        ax.set_title(f"Native NHERI raw vs final MannHybridTurb input: {name}-component{title_suffix}")
        ax.grid(True, which="both", ls="--", alpha=0.35)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(fig_dir / f"native_raw_vs_final_spectrum_{name}.png", dpi=250)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    for j in idx_final:
        z0 = z_final[j]
        jm = int(np.argmin(np.abs(z_meas - z0)))
        f_raw, c_raw = raw_native["uw"][jm]
        ax.semilogx(f_raw, c_raw, lw=0.75, alpha=0.45, ls="--", label=f"raw native z≈{z_meas[jm]:.3g} m")
        ax.semilogx(freq, C_final[j], lw=2.2, label=f"final input z={z0:.3g} m")
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.axvline(LOW_PLATEAU_HZ, color="0.35", lw=0.8, ls=":", label="LF join")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Cuw [m²/s]")
    ax.set_title(f"Native NHERI raw vs final MannHybridTurb input: u-w co-spectrum{title_suffix}")
    ax.grid(True, which="both", ls="--", alpha=0.35)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "native_raw_vs_final_uw_cospectrum.png", dpi=250)
    plt.close(fig)


def plot_wt_profile_8panel(fig_dir: Path, raw_df: pd.DataFrame, final_df: pd.DataFrame, H: float, filename: str = "raw_experimental_vs_mannhybrid_input_profiles_8panel.png", title_suffix: str = "") -> None:
    safe_makedirs(fig_dir)
    U_H = float(np.interp(H, final_df["z"].to_numpy(float), final_df["U"].to_numpy(float)))
    fig, axes = plt.subplots(2, 4, figsize=(12.4, 8.3), sharey=True)
    axes = axes.ravel()
    panel_defs = [
        ("U/U_H", lambda df: df["U"].to_numpy(float)/U_H, (0, 1.5), r"$U/U_H$"),
        ("Iu", lambda df: df["Iu"].to_numpy(float), (0, 0.35), r"$I_u$"),
        ("Iv", lambda df: df["Iv"].to_numpy(float), (0, 0.35), r"$I_v$"),
        ("Iw", lambda df: df["Iw"].to_numpy(float), (0, 0.35), r"$I_w$"),
        ("uw", lambda df: df["uwStress"].to_numpy(float)/(U_H**2) if "uwStress" in df.columns else np.full(len(df), np.nan), (-0.08, 0.02), r"$\overline{u'w'}/U_H^2$"),
        ("Lu/H", lambda df: df["Lu"].to_numpy(float)/H, (0, 4.0), r"$L_u/H$"),
        ("Lv/H", lambda df: df["Lv"].to_numpy(float)/H, (0, 2.0), r"$L_v/H$"),
        ("Lw/H", lambda df: df["Lw"].to_numpy(float)/H, (0, 2.0), r"$L_w/H$"),
    ]
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
    z_raw = raw_df["z"].to_numpy(float)/H
    z_fin = final_df["z"].to_numpy(float)/H
    for ax, (name, func, xlim, xlabel), lab in zip(axes, panel_defs, labels):
        xr = func(raw_df)
        xf = func(final_df)
        mr = np.isfinite(xr) & np.isfinite(z_raw)
        mf = np.isfinite(xf) & np.isfinite(z_fin)
        ax.scatter(xr[mr], z_raw[mr], s=20, facecolors="none", edgecolors="black", linewidths=0.9, label="raw experiment", zorder=5)
        ax.plot(xf[mf], z_fin[mf], color="#d62728", lw=2.0, label="smoothed/final input", zorder=4)
        if name == "uw":
            ax.axvline(0, color="0.35", lw=0.8)
        ax.set_xlim(*xlim)
        ax.set_ylim(0, max(3.0, np.nanmax(z_fin[np.isfinite(z_fin)])))
        ax.set_xlabel(xlabel)
        ax.set_title(lab)
        ax.grid(True, ls="--", alpha=0.35)
        ax.tick_params(direction="in")
    axes[0].set_ylabel(r"$z/H$")
    axes[4].set_ylabel(r"$z/H$")
    axes[0].legend(fontsize=9, loc="upper left", frameon=True)
    fig.suptitle(f"NHERI raw experimental profiles vs MannHybridTurb input profiles{title_suffix}", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig_dir / filename, dpi=300)
    plt.close(fig)


def plot_variant_comparison(fig_dir: Path, freq: np.ndarray, z_final: np.ndarray, variants: dict[str, dict], H: float) -> None:
    """Compact plots comparing low-frequency choices at the roof-height-ish point."""
    safe_makedirs(fig_dir)
    j = int(np.argmin(np.abs(z_final - H)))
    for c, name in enumerate(["u", "v", "w"]):
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        for mode, data in variants.items():
            ax.loglog(freq, np.maximum(data["S"][c, j], 1e-16), lw=1.8, label=mode)
        ax.axvline(LOW_PLATEAU_HZ, color="0.35", lw=0.8, ls=":", label="LF join")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(f"S{name}{name} [m²/s]")
        ax.set_title(f"Low-frequency target variants at z≈{z_final[j]:.3g} m: {name}")
        ax.grid(True, which="both", ls="--", alpha=0.35)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"low_frequency_variant_comparison_{name}.png", dpi=250)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for mode, data in variants.items():
        ax.semilogx(freq, data["Cuw"][j], lw=1.8, label=mode)
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.axvline(LOW_PLATEAU_HZ, color="0.35", lw=0.8, ls=":", label="LF join")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Cuw [m²/s]")
    ax.set_title(f"Low-frequency Cuw variants at z≈{z_final[j]:.3g} m")
    ax.grid(True, which="both", ls="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "low_frequency_variant_comparison_uw_cospectrum.png", dpi=250)
    plt.close(fig)


# =============================================================================
# Core processing
# =============================================================================

def build_profile_data(case_dir: Path, vel_array_3d: np.ndarray):
    print("Building measured profile and restoring tuned NHERI profile smoothing...")
    profile_df = LES._windTunnel.get_nheri_profile_df(APPROACH_FLOW_MAT)
    int_length_scales = LES._windTunnel.calc_nheri_int_length_scales(vel_array_3d)
    profile_df = LES._windTunnel.add_nheri_int_length_scales(profile_df, int_length_scales)

    # Correct Reynolds shear-stress calculation. Shape is (3,nTime,nHeight).
    fluc = vel_array_3d - np.mean(vel_array_3d, axis=1, keepdims=True)
    uw_meas = np.mean(fluc[0] * fluc[2], axis=0)
    profile_df = profile_df.copy()
    profile_df["uwStress"] = uw_meas
    measured_experimental_profile_df = profile_df.copy()

    # IMPORTANT: David's original tuned smoothing was applied to the classic
    # NHERI profile columns only, before any signed u'w' column was introduced.
    # Passing the signed/extrapolated uwStress column through windlespy's
    # smooth_profiles can trigger invalid plot limits and can over-smooth or
    # corrupt the tuned I(z)/L(z) behaviour.  Therefore: smooth exactly the
    # classic columns with smooth_profiles(..., 5, 7, 0.5), then append a
    # separately smoothed signed uwStress profile.
    classic_cols = ["z", "U", "Iu", "Iv", "Iw", "Lu", "Lv", "Lw"]
    classic_profile_df = profile_df.loc[:, classic_cols].copy()

    extended_classic_df = LES._windTunnel.extend_nheri_profiles(
        classic_profile_df,
        EXTEND_PROFILE_FACTOR,
        fit_zmin=None,
        fit_zmax=None,
    )
    extended_classic_df = _sanitize_profile_df(extended_classic_df, classic_cols)
    z_extended = extended_classic_df["z"].to_numpy(float)

    print(
        "Applying tuned NHERI profile smoothing exactly for U/I/L: "
        f"smooth_profiles(classic_df, z, {PROFILE_SMOOTH_WINDOW}, {PROFILE_SMOOTH_POLYORDER}, {PROFILE_SMOOTH_H})"
    )
    try:
        smoothed_extended_df = LES._profileAnalysis.smooth_profiles(
            extended_classic_df,
            z_extended,
            PROFILE_SMOOTH_WINDOW,
            PROFILE_SMOOTH_POLYORDER,
            PROFILE_SMOOTH_H,
        )
    finally:
        # smooth_profiles shows diagnostic figures internally; close them so
        # batch/Spyder runs do not accumulate stale Matplotlib state.
        plt.close("all")

    smoothed_extended_df = smoothed_extended_df.copy()
    smoothed_extended_df = _sanitize_profile_df(smoothed_extended_df, classic_cols)

    # Append signed u'w' on the smoothed/extended grid.  This is intentionally
    # separate from the tuned U/I/L smoothing because u'w' is signed and was not
    # part of David's original smooth_profiles call.
    uw_smooth = _smooth_uw_stress_profile(
        measured_experimental_profile_df["z"].to_numpy(float),
        measured_experimental_profile_df["uwStress"].to_numpy(float),
        smoothed_extended_df["z"].to_numpy(float),
        window=max(PROFILE_SMOOTH_WINDOW, 5),
    )
    smoothed_extended_df["uwStress"] = uw_smooth

    z_final = smoothed_extended_df["z"].to_numpy(float)

    # Build the internal array on the smoothed/extended grid.  IMPORTANT: the
    # windlespy map_profile_to_inlet_z helper was written for the classic DFSR
    # 7-column internal profile only:
    #     U, uu, vv, ww, Lu, Lv, Lw
    # If an 8th signed uwStress column is passed through it, some versions drop
    # or zero that column.  That was the source of the zero Reynolds shear-stress
    # profile and all-zero uwCoSpectrumProfile.  Therefore the classic U/I/L
    # fields are mapped with windlespy, while uwStress is interpolated separately.
    profile_array_full = profile_to_internal_array(smoothed_extended_df)
    profile_array_classic = profile_array_full[:, :7]
    uw_smooth_extended = profile_array_full[:, 7] if profile_array_full.shape[1] >= 8 else np.zeros(len(z_final))

    if MAP_PROFILE_TO_INLET_Z:
        try:
            z_centres = LES._caseFiles.get_inlet_cell_centres(str(case_dir), INLET_FOAM_FILE, INLET_PATCH_NAME)
            z_centres = np.asarray(z_centres, dtype=float)
            mapped_classic = LES._profileAnalysis.map_profile_to_inlet_z(profile_array_classic, z_final, z_centres)
            mapped_uw = _finite_interpolate_column(z_final, uw_smooth_extended, z_centres)
            mapped_array = np.column_stack([mapped_classic[:, :7], mapped_uw])
            final_df = internal_array_to_profile(z_centres, mapped_array, include_uw=True)
            z_final = final_df["z"].to_numpy(float)
            profile_array = profile_to_internal_array(final_df)
            print(
                f"Mapped smoothed classic profile to {len(z_final)} inlet face-centre heights using "
                f"{INLET_FOAM_FILE}:{INLET_PATCH_NAME}; mapped uwStress separately."
            )
            print(
                "Mapped uwStress range: "
                f"min={float(np.min(profile_array[:, 7])):.6e}, "
                f"max={float(np.max(profile_array[:, 7])):.6e}, "
                f"mean={float(np.mean(profile_array[:, 7])):.6e}"
            )
        except Exception as exc:
            print(f"Warning: profile mapping to inlet z failed; using extended smoothed grid. Reason: {exc}")
            profile_array = profile_array_full
            final_df = internal_array_to_profile(z_final, profile_array, include_uw=True)
    else:
        profile_array = profile_array_full
        final_df = internal_array_to_profile(z_final, profile_array, include_uw=True)

    _assert_finite("measured experimental profile", measured_experimental_profile_df.to_numpy(float))
    _assert_finite("smoothed extended profile", smoothed_extended_df.to_numpy(float))
    _assert_finite("final smoothed/mapped profile", final_df.to_numpy(float))

    if profile_array.shape[1] >= 8:
        raw_uw = measured_experimental_profile_df["uwStress"].to_numpy(float)
        mapped_uw = profile_array[:, 7]
        if np.nanmax(np.abs(raw_uw)) > 1e-8 and np.nanmax(np.abs(mapped_uw)) < 1e-12:
            raise RuntimeError(
                "Mapped uwStress is essentially zero although the measured uwStress is not. "
                "This indicates that the signed shear-stress column was lost during profile mapping."
            )
    return measured_experimental_profile_df, smoothed_extended_df, final_df, profile_array

def make_variant(mode: str, freq: np.ndarray, z_measured: np.ndarray, z_final: np.ndarray, S_raw: np.ndarray, Cuw_raw: np.ndarray, S_fit: np.ndarray, Cuw_fit: np.ndarray, profile_array: np.ndarray) -> dict:
    S_low = apply_low_frequency_shape(
        freq,
        S_fit,
        raw_reference=S_raw,
        mode=mode,
        join_freq=LOW_PLATEAU_HZ,
        signed=False,
    )
    C_low = apply_low_frequency_shape(
        freq,
        Cuw_fit,
        raw_reference=Cuw_raw,
        mode=mode,
        join_freq=LOW_PLATEAU_HZ,
        signed=True,
    )

    if INTERPOLATE_MEASURED_SPECTRA_TO_ALL_HEIGHTS:
        S_final = interp_spectra_in_height(z_measured, S_low, z_final)
        Cuw_final = interp_signed_in_height(z_measured, C_low, z_final)
    else:
        # Fallback: place measured heights where available; interpolate Cuw.
        S_final = interp_spectra_in_height(z_measured, S_low, z_final)
        Cuw_final = interp_signed_in_height(z_measured, C_low, z_final)

    # Keep the carefully smoothed I profiles.  Shape changes between variants,
    # but each variant has the same target resolved variance.
    var_target = profile_array[:, 1:4]
    S_final = scale_auto_spectra_to_variance(S_final, freq, var_target)

    uw_target = profile_array[:, 7] if profile_array.shape[1] >= 8 else np.zeros(profile_array.shape[0])
    Cuw_final = scale_cuw_to_profile(Cuw_final, freq, uw_target, S_final)

    _assert_finite(f"{mode} auto-spectra", S_final)
    _assert_finite(f"{mode} uw co-spectrum", Cuw_final)

    # Profile remains the smoothed/mapped target profile; update variances/Cuw
    # to exactly match the written spectra integrals for full consistency.
    prof = profile_array.copy()
    prof[:, 1:4] = _trapz(S_final, freq, axis=-1).T
    if prof.shape[1] >= 8:
        prof[:, 7] = _trapz(Cuw_final, freq, axis=-1)

    return {"mode": mode, "S": S_final, "Cuw": Cuw_final, "profile_array": prof}


def write_variant_files(out_dir: Path, z: np.ndarray, freq: np.ndarray, variant: dict) -> pd.DataFrame:
    safe_makedirs(out_dir)
    profile_df = internal_array_to_profile(z, variant["profile_array"], include_uw=True)

    write_profile(out_dir / "profile", profile_df)
    write_profile(out_dir / "targetProfile", profile_df)
    write_profile(out_dir / "targetSmoothedProfile", profile_df)

    uw_stress = profile_df["uwStress"].to_numpy(float) if "uwStress" in profile_df.columns else None
    write_auto_spectra(out_dir / "spectraProfile", z, variant["S"], uw_stress=uw_stress)
    write_auto_spectra(out_dir / "targetSpectraProfile", z, variant["S"], uw_stress=uw_stress)
    write_uw_cospectrum(out_dir / "uwCoSpectrumProfile", z, variant["Cuw"], uw_stress=uw_stress)
    write_uw_cospectrum(out_dir / "targetUWCoSpectrumProfile", z, variant["Cuw"], uw_stress=uw_stress)

    return profile_df


def main() -> None:
    case_dir = Path(CASE_DIR)
    wind_profile_dir = case_dir / "constant" / "boundaryData" / "windProfile"
    safe_makedirs(wind_profile_dir)

    print(f"Reading NHERI approach flow data:\n  {APPROACH_FLOW_MAT}")
    vel_array_3d = LES._windTunnel.get_nheri_vel_time_series(APPROACH_FLOW_MAT)
    vel_array_3d = np.asarray(vel_array_3d, float)
    if vel_array_3d.ndim != 3 or vel_array_3d.shape[0] != 3:
        raise ValueError(f"Expected velocity array shape (3,nTime,nHeight); got {vel_array_3d.shape}")

    measured_profile_df, smoothed_extended_df, final_profile_df, final_profile_array = build_profile_data(case_dir, vel_array_3d)

    z_measured = measured_profile_df["z"].to_numpy(float)
    z_final = final_profile_df["z"].to_numpy(float)
    freq = freq_array_from_fmax(FMAX, NFREQ)

    print("Estimating raw measured auto-spectra and u-w co-spectrum using multitaper...")
    S_raw, Cuw_raw = estimate_downstream_spectra(
        vel_array_3d,
        fs=1.0 / WT_DT,
        f_target=freq,
        method="multitaper",
        time_bandwidth=3.5,
    )
    _assert_finite("raw auto-spectra", S_raw)
    _assert_finite("raw uw co-spectrum", Cuw_raw)

    if SMOOTH_TARGET_SPECTRA:
        print(
            "Smoothing spectra using log-bin median + PCHIP, "
            f"n_bins={SMOOTH_BINS}, join={LOW_PLATEAU_HZ:g} Hz, and -5/3 tail..."
        )
        S_fit = smooth_spectra_array(
            freq,
            S_raw,
            low_plateau_max=None,
            tail_slope_after=TAIL_AFTER_HZ,
            n_bins=SMOOTH_BINS,
        )
        Cuw_fit = smooth_cospectrum_array(freq, Cuw_raw, n_bins=SMOOTH_BINS)
    else:
        S_fit = S_raw.copy()
        Cuw_fit = Cuw_raw.copy()
    _assert_finite("base fitted auto-spectra", S_fit)
    _assert_finite("base fitted uw co-spectrum", Cuw_fit)

    variants_requested = parse_variants(WT_LOW_FREQ_VARIANTS)
    if WT_LOW_FREQ_MODE not in variants_requested:
        variants_requested.insert(0, WT_LOW_FREQ_MODE)

    print(f"Building low-frequency variants: {variants_requested}")
    variants: dict[str, dict] = {}
    summary_rows = []
    for mode in variants_requested:
        v = make_variant(mode, freq, z_measured, z_final, S_raw, Cuw_raw, S_fit, Cuw_fit, final_profile_array)
        variants[mode] = v
        prof_df = internal_array_to_profile(z_final, v["profile_array"], include_uw=True)
        var = _trapz(v["S"], freq, axis=-1).T
        uw = _trapz(v["Cuw"], freq, axis=-1)
        summary_rows.append({
            "mode": mode,
            "nHeights": len(z_final),
            "nFreq": len(freq),
            "meanVarU": float(np.mean(var[:, 0])),
            "meanVarV": float(np.mean(var[:, 1])),
            "meanVarW": float(np.mean(var[:, 2])),
            "meanUwStress": float(np.mean(uw)),
            "minUwStress": float(np.min(uw)),
            "maxUwStress": float(np.max(uw)),
            "meanIu": float(np.mean(prof_df["Iu"])),
            "meanIv": float(np.mean(prof_df["Iv"])),
            "meanIw": float(np.mean(prof_df["Iw"])),
        })

    variant_root = wind_profile_dir / "lowFrequencyVariants"
    safe_makedirs(variant_root)

    print("Writing all low-frequency variant folders...")
    variant_profile_dfs: dict[str, pd.DataFrame] = {}
    for mode, v in variants.items():
        out_dir = variant_root / mode
        variant_profile_dfs[mode] = write_variant_files(out_dir, z_final, freq, v)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(variant_root / "low_frequency_variant_summary.csv", index=False)

    # Write active main files using WT_LOW_FREQ_MODE.
    selected_mode = WT_LOW_FREQ_MODE
    if selected_mode not in variants:
        selected_mode = variants_requested[0]
    print(f"Writing active MannHybridTurb input files using WT_LOW_FREQ_MODE={selected_mode!r}")
    active_profile_df = write_variant_files(wind_profile_dir, z_final, freq, variants[selected_mode])

    # Preserve experimental and smoothed diagnostic profiles.  These are not the
    # active generator inputs but are used by plots/calibration diagnostics.
    write_profile(wind_profile_dir / "targetExperimentalProfile", measured_profile_df)
    write_profile(wind_profile_dir / "targetSmoothedProfile_experimentalGrid", smoothed_extended_df)

    if WRITE_PLOTS:
        base_fig_dir = case_dir / "Figures" / "targetSpectra" / "windTunnelFitted"
        print(f"Writing variant spectra/profile diagnostic plots to {base_fig_dir}")
        for mode, v in variants.items():
            fig_dir = base_fig_dir / mode
            profile_df_mode = variant_profile_dfs[mode]
            plot_spectra_diagnostics(fig_dir, freq, z_measured, S_raw, Cuw_raw, z_final, v["S"], v["Cuw"], title_suffix=f" ({mode})")
            plot_spectra_diagnostics_native(fig_dir, freq, z_measured, vel_array_3d, z_final, v["S"], v["Cuw"], title_suffix=f" ({mode})")
            if WRITE_PROFILE_8PANEL:
                plot_wt_profile_8panel(fig_dir, measured_profile_df, profile_df_mode, BODY_HEIGHT, title_suffix=f" ({mode})")

        # Backward-compatible plots in the old location use the active selected mode.
        sel = variants[selected_mode]
        plot_spectra_diagnostics(base_fig_dir, freq, z_measured, S_raw, Cuw_raw, z_final, sel["S"], sel["Cuw"], title_suffix=f" ({selected_mode})")
        plot_spectra_diagnostics_native(base_fig_dir, freq, z_measured, vel_array_3d, z_final, sel["S"], sel["Cuw"], title_suffix=f" ({selected_mode})")
        if WRITE_PROFILE_8PANEL:
            plot_wt_profile_8panel(base_fig_dir, measured_profile_df, active_profile_df, BODY_HEIGHT, title_suffix=f" ({selected_mode})")
        plot_variant_comparison(base_fig_dir / "lowFrequencyVariantComparison", freq, z_final, variants, BODY_HEIGHT)

    try:
        LES._caseFiles.write_dfsr_samp_pts(0, 0, str(case_dir), active_profile_df)
        LES._caseFiles.write_probes_from_target_profile(2.5, 0, str(case_dir), active_profile_df, "probes2")
    except Exception as exc:
        print(f"Warning: could not write windlespy sampling/probes helpers: {exc}")

    print("Done.")
    print(f"  active low-frequency mode: {selected_mode}")
    print(f"  all variants written under: {variant_root}")
    print(f"  active files written under: {wind_profile_dir}")
    print(f"  heights: {len(z_final)}")
    print(f"  frequencies: {NFREQ}, fMax={FMAX}")
    print(f"  profile smoothing: smooth_profiles(..., {PROFILE_SMOOTH_WINDOW}, {PROFILE_SMOOTH_POLYORDER}, {PROFILE_SMOOTH_H})")
    print(f"  low-frequency join: {LOW_PLATEAU_HZ:g} Hz")


if __name__ == "__main__":
    main()
