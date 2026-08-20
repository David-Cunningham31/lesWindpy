# -*- coding: utf-8 -*-
"""
Plot MannHybrid spectral-tilt calibration iteration profiles against raw NHERI
wind-tunnel approach-flow data, using a Melaku/Bitsuamlak-style profile layout.

This is the spectral-tilt equivalent of the Wong/downstream calibration review
script. It reads iteration data from

    case_path/log/spectralTiltCalibration/iteration*/data/

where each iteration contains, at minimum,

    target_profile.csv
    current_profile.csv
    downstream_profile.csv
    updated_profile.csv

The profile CSVs are expected to contain:

    z, U, Iu, Iv, Iw, Lu, Lv, Lw, uwStress

where Iu/Iv/Iw are turbulence intensities and uwStress is dimensional
<u'w'> in m^2/s^2. The script derives the normal Reynolds stresses as

    <u'u'> = (Iu U)^2
    <v'v'> = (Iv U)^2
    <w'w'> = (Iw U)^2

and plots them normalised by U_H^2.

Outputs
-------
For each iteration, the script writes:

  1. a Melaku-style profile figure with
       U/U_H, Iu, Iv, Iw, Lu/H, Lv/H, Lw/H, -<u'w'>/U_H^2

  2. a Reynolds-stress figure with
       <u'u'>/U_H^2, <v'v'>/U_H^2, <w'w'>/U_H^2, -<u'w'>/U_H^2

  3. QA CSVs containing exactly what was plotted.

It can also create GIF animations over iterations.

Important notes
---------------
1. EXP points are computed directly from the raw NHERI approach-flow .mat file.
   They are not digitised Melaku points and are not smoothed/extended.
2. The EXP length scales are calculated through windLespy's NHERI helper
   functions, matching the original Wong plotting script.
3. The spectral-tilt target profile is plotted as a dashed line because it is
   useful to distinguish the raw measured experimental points from the smooth/
   extended calibration target.
4. By default the downstream LES curve is the main solid curve. The current and
   updated inlet curves can be toggled below if you want to see how the inlet is
   evolving.
"""

from __future__ import annotations

import os
import re
import sys
import json
import zipfile
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------------

# Case directory containing setUp and log/spectralTiltCalibration.
# On the cluster this is normally the OpenFOAM case root.
case_path = r"C:\Users\david\OneDrive\Documents\PhD\Year 1\spctral_tilt_cailbration\classic_wong_dfsr_new_mesh"

# Raw NHERI approach-flow .mat file.
approach_flow_data = r"C:\Users\david\OneDrive\Documents\PhD\Year 1\Wind Tunnel Test Data\NHERI BLWT Tall Building\Approach Flow\Approach Flow - EH160 - Marine Spires - 1200 RPM - 091721_1028.mat"

# Optional: read iteration data directly from a ZIP of the spectralTiltCalibration
# folder. Leave as None for normal case-folder use.
SPECTRAL_TILT_RESULTS_ZIP = None
# Example:
# SPECTRAL_TILT_RESULTS_ZIP = r"C:\path\to\spectral_tilt_cal_results.zip"

# The calibration log directory relative to case_path.
CALIBRATION_LOG_REL = os.path.join("log", "downstreamCalibration")

# Output folder name under the spectralTiltCalibration folder, or beside the ZIP.
OUTPUT_SUBDIR = "profile_review_nheri_wong_melaku_style"

# Plot vertical ranges. Set to [] and use one manually if you prefer.
# None means auto/full-height based on available data.
VERTICAL_RANGES = [
    (0.0, 3.0, "z0_to_3H"),
    (0.0, None, "fullHeight"),
]

# Which spectral-tilt profiles to plot.
PLOT_TARGET_PROFILE = True       # dashed smooth target from target_profile.csv
PLOT_CURRENT_INLET = False       # active inlet before update for that iteration
PLOT_UPDATED_INLET = False       # inlet after update from that iteration
PLOT_DOWNSTREAM_LES = True       # downstream LES profile; main curve

# Optional CSV output for checking exactly what was plotted.
WRITE_QA_CSV = True

# Figure outputs.
FIG_DPI = 300
SAVE_PDF = False
DARK_STYLE = False
CREATE_ANIMATION = True
ANIMATION_FPS = 2

# If True, stress x-limits are chosen automatically from EXP/TARGET/LES.
AUTO_STRESS_XLIMS = True

# Hard fallback limits used if AUTO_STRESS_XLIMS=False or if auto fails.
DEFAULT_STRESS_XLIMS = {
    "uu_over_UH2": (0.0, 0.12),
    "vv_over_UH2": (0.0, 0.08),
    "ww_over_UH2": (0.0, 0.04),
    "neg_uw_over_UH2": (0.0, 0.04),
}

# Marker/line styling similar to the Melaku paper figure.
EXP_MARKER_SIZE = 26
EXP_MARKER_LINEWIDTH = 1.1
TARGET_LINEWIDTH = 1.25
CURRENT_LINEWIDTH = 1.15
UPDATED_LINEWIDTH = 1.35
LES_LINEWIDTH = 2.0

# ---------------------------------------------------------------------------
# windLespy import
# ---------------------------------------------------------------------------

cwd = os.path.dirname(os.path.abspath(__file__))
windlespy_path = os.path.abspath(os.path.join(cwd, "..", ".."))
if windlespy_path not in sys.path:
    sys.path.append(windlespy_path)
try:
    import windlespy as LES
except Exception as exc:
    raise RuntimeError(
        "Could not import windlespy. Put this script in windlespy/_recipes, "
        "or add the parent directory containing windlespy to PYTHONPATH."
    ) from exc
finally:
    try:
        sys.path.remove(windlespy_path)
    except ValueError:
        pass

# ---------------------------------------------------------------------------
# Plot configuration
# ---------------------------------------------------------------------------

PROFILE_KEYS = [
    "U_over_UH",
    "Iu",
    "Iv",
    "Iw",
    "Lu_over_H",
    "Lv_over_H",
    "Lw_over_H",
    "neg_uw_over_UH2",
]

STRESS_KEYS = [
    "uu_over_UH2",
    "vv_over_UH2",
    "ww_over_UH2",
    "neg_uw_over_UH2",
]

AXIS_CONFIG = {
    "U_over_UH": {
        "xlim": (0.0, 1.5),
        "xticks": [0.0, 0.5, 1.0, 1.5],
        "xlabel": r"$U_{av}/U_H$",
        "panel": "(a)",
    },
    "Iu": {
        "xlim": (0.0, 0.35),
        "xticks": [0.0, 0.1, 0.2, 0.3],
        "xlabel": r"$I_u$",
        "panel": "(b)",
    },
    "Iv": {
        "xlim": (0.0, 0.35),
        "xticks": [0.0, 0.1, 0.2, 0.3],
        "xlabel": r"$I_v$",
        "panel": "(c)",
    },
    "Iw": {
        "xlim": (0.0, 0.35),
        "xticks": [0.0, 0.1, 0.2, 0.3],
        "xlabel": r"$I_w$",
        "panel": "(d)",
    },
    "Lu_over_H": {
        "xlim": (0.0, 4.0),
        "xticks": [0.0, 1.0, 2.0, 3.0, 4.0],
        "xlabel": r"$L_u/H$",
        "panel": "(e)",
    },
    "Lv_over_H": {
        "xlim": (0.0, 2.0),
        "xticks": [0.0, 0.5, 1.0, 1.5, 2.0],
        "xlabel": r"$L_v/H$",
        "panel": "(f)",
    },
    "Lw_over_H": {
        "xlim": (0.0, 2.0),
        "xticks": [0.0, 0.5, 1.0, 1.5, 2.0],
        "xlabel": r"$L_w/H$",
        "panel": "(g)",
    },
    "uu_over_UH2": {
        "xlabel": r"$\overline{u'u'}/U_H^2$",
        "panel": "(a)",
    },
    "vv_over_UH2": {
        "xlabel": r"$\overline{v'v'}/U_H^2$",
        "panel": "(b)",
    },
    "ww_over_UH2": {
        "xlabel": r"$\overline{w'w'}/U_H^2$",
        "panel": "(c)",
    },
    "neg_uw_over_UH2": {
        "xlim": (0.0, 0.04),
        "xticks": [0.0, 0.01, 0.02, 0.03, 0.04],
        "xlabel": r"$-\overline{u'w'}/U_H^2$",
        "panel": "(h)",
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def natural_iteration_key(name_or_path) -> int:
    base = os.path.basename(str(name_or_path).rstrip("/"))
    m = re.search(r"iteration(\d+)$", base)
    return int(m.group(1)) if m else 10**9


def column_lookup(df: pd.DataFrame, candidates: Iterable[str]):
    norm = {str(c).strip().lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower().replace(" ", "").replace("_", "")
        if key in norm:
            return norm[key]
    return None


def safe_interp(x, y, x0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(m) < 2:
        return np.nan
    order = np.argsort(x[m])
    return float(np.interp(x0, x[m][order], y[m][order]))


def robust_profile_df(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Return a clean spectral-tilt profile dataframe with standard columns."""
    required = ["z", "U", "Iu", "Iv", "Iw", "Lu", "Lv", "Lw"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing required columns {missing}; got {list(df.columns)}")
    out = df.copy()
    if "uwStress" not in out.columns:
        warnings.warn(f"{label}: no uwStress column found. Filling with NaN.")
        out["uwStress"] = np.nan
    for c in required + ["uwStress"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out[np.isfinite(out["z"])].copy()
    out = out.sort_values("z").reset_index(drop=True)
    return out


def profile_df_to_quantities(df: pd.DataFrame, U_H_ref: float, H: float, source: str) -> Dict[str, np.ndarray]:
    df = robust_profile_df(df, source)
    z = df["z"].to_numpy(float)
    U = df["U"].to_numpy(float)
    Iu = df["Iu"].to_numpy(float)
    Iv = df["Iv"].to_numpy(float)
    Iw = df["Iw"].to_numpy(float)
    Lu = df["Lu"].to_numpy(float)
    Lv = df["Lv"].to_numpy(float)
    Lw = df["Lw"].to_numpy(float)
    uw = df["uwStress"].to_numpy(float)

    UHref2 = max(float(U_H_ref) ** 2, 1e-24)
    uu = (Iu * U) ** 2
    vv = (Iv * U) ** 2
    ww = (Iw * U) ** 2

    return {
        "source": source,
        "z": z,
        "z_over_H": z / H,
        "U": U,
        "U_over_UH": U / max(abs(float(U_H_ref)), 1e-12),
        "Iu": Iu,
        "Iv": Iv,
        "Iw": Iw,
        "Lu": Lu,
        "Lv": Lv,
        "Lw": Lw,
        "Lu_over_H": Lu / H,
        "Lv_over_H": Lv / H,
        "Lw_over_H": Lw / H,
        "uu": uu,
        "vv": vv,
        "ww": ww,
        "uw": uw,
        "uu_over_UH2": uu / UHref2,
        "vv_over_UH2": vv / UHref2,
        "ww_over_UH2": ww / UHref2,
        "uw_over_UH2": uw / UHref2,
        "neg_uw_over_UH2": -uw / UHref2,
    }


def build_nheri_experimental_quantities(approach_flow_mat: str, H: float):
    """Build raw NHERI experimental profile quantities from the .mat data."""
    vel_array_3d = LES._windTunnel.get_nheri_vel_time_series(approach_flow_mat)
    target_profile_df = LES._windTunnel.get_nheri_profile_df(approach_flow_mat).copy()

    # Add integral length scales calculated from the raw NHERI time series.
    try:
        int_length_scales = LES._windTunnel.calc_nheri_int_length_scales(vel_array_3d)
        target_profile_df = LES._windTunnel.add_nheri_int_length_scales(target_profile_df, int_length_scales)
    except Exception as exc:
        warnings.warn(f"Could not calculate/add NHERI integral length scales through windLespy: {exc}")

    z_col = column_lookup(target_profile_df, ["z", "Z", "height", "Height"])
    U_col = column_lookup(target_profile_df, ["U", "Uav", "U_av", "meanU", "Ux", "Umean"])
    Iu_col = column_lookup(target_profile_df, ["Iu", "I_u", "uIntensity", "Iu_exp"])
    Iv_col = column_lookup(target_profile_df, ["Iv", "I_v", "vIntensity", "Iv_exp"])
    Iw_col = column_lookup(target_profile_df, ["Iw", "I_w", "wIntensity", "Iw_exp"])
    Lu_col = column_lookup(target_profile_df, ["Lu", "L_u", "xLu", "x_Lu", "LengthScaleU"])
    Lv_col = column_lookup(target_profile_df, ["Lv", "L_v", "xLv", "x_Lv", "LengthScaleV"])
    Lw_col = column_lookup(target_profile_df, ["Lw", "L_w", "xLw", "x_Lw", "LengthScaleW"])

    if z_col is None:
        raise ValueError(f"Could not identify z column in NHERI profile dataframe: {list(target_profile_df.columns)}")

    z = target_profile_df[z_col].to_numpy(dtype=float)
    v = np.asarray(vel_array_3d, dtype=float)
    if v.ndim != 3:
        raise ValueError(f"Unexpected NHERI velocity array shape {v.shape}; expected 3D array.")

    # Expected shape is (3, n_time, n_heights). Handle a common transposed variant.
    if v.shape[0] == 3:
        vel = v
    elif v.shape[-1] == 3:
        vel = np.moveaxis(v, -1, 0)
    else:
        raise ValueError(f"Cannot infer component axis from NHERI velocity array shape {v.shape}")

    n_heights = min(len(z), vel.shape[2])
    z = z[:n_heights]
    vel = vel[:, :, :n_heights]

    U_raw = np.nanmean(vel[0], axis=0)
    if U_col is not None:
        U_profile = target_profile_df[U_col].to_numpy(dtype=float)[:n_heights]
        # Prefer the profile dataframe if it is finite; otherwise fall back to raw time-series mean.
        U = np.where(np.isfinite(U_profile), U_profile, U_raw)
    else:
        U = U_raw

    fluc = vel - np.nanmean(vel, axis=1, keepdims=True)
    uu = np.nanmean(fluc[0] * fluc[0], axis=0)
    vv = np.nanmean(fluc[1] * fluc[1], axis=0)
    ww = np.nanmean(fluc[2] * fluc[2], axis=0)
    uw = np.nanmean(fluc[0] * fluc[2], axis=0)

    U_safe = np.maximum(np.abs(U), 1e-12)
    Iu_raw = np.sqrt(np.maximum(uu, 0.0)) / U_safe
    Iv_raw = np.sqrt(np.maximum(vv, 0.0)) / U_safe
    Iw_raw = np.sqrt(np.maximum(ww, 0.0)) / U_safe

    def use_profile_or_raw(col, raw, name):
        if col is None:
            return raw
        vals = target_profile_df[col].to_numpy(dtype=float)[:n_heights]
        if np.count_nonzero(np.isfinite(vals)) < max(2, n_heights // 4):
            warnings.warn(f"NHERI {name} column mostly invalid; using raw time-series estimate.")
            return raw
        return np.where(np.isfinite(vals), vals, raw)

    Iu = use_profile_or_raw(Iu_col, Iu_raw, "Iu")
    Iv = use_profile_or_raw(Iv_col, Iv_raw, "Iv")
    Iw = use_profile_or_raw(Iw_col, Iw_raw, "Iw")

    def maybe_col(col, name):
        if col is None:
            warnings.warn(f"NHERI {name} column not found. Filling with NaN.")
            return np.full_like(z, np.nan, dtype=float)
        vals = target_profile_df[col].to_numpy(dtype=float)[:n_heights]
        return vals

    Lu = maybe_col(Lu_col, "Lu")
    Lv = maybe_col(Lv_col, "Lv")
    Lw = maybe_col(Lw_col, "Lw")

    m = np.isfinite(z) & np.isfinite(U)
    order = np.argsort(z[m])
    z = z[m][order]
    U = U[m][order]
    Iu = Iu[m][order]
    Iv = Iv[m][order]
    Iw = Iw[m][order]
    Lu = Lu[m][order]
    Lv = Lv[m][order]
    Lw = Lw[m][order]
    uu = uu[m][order]
    vv = vv[m][order]
    ww = ww[m][order]
    uw = uw[m][order]

    U_H_ref = safe_interp(z, U, H)
    if not np.isfinite(U_H_ref):
        raise ValueError("Could not interpolate NHERI U at z=H.")

    UHref2 = max(float(U_H_ref) ** 2, 1e-24)
    q = {
        "source": "EXP_NHERI_raw",
        "z": z,
        "z_over_H": z / H,
        "U": U,
        "U_over_UH": U / max(abs(float(U_H_ref)), 1e-12),
        "Iu": Iu,
        "Iv": Iv,
        "Iw": Iw,
        "Lu": Lu,
        "Lv": Lv,
        "Lw": Lw,
        "Lu_over_H": Lu / H,
        "Lv_over_H": Lv / H,
        "Lw_over_H": Lw / H,
        "uu": uu,
        "vv": vv,
        "ww": ww,
        "uw": uw,
        "uu_over_UH2": uu / UHref2,
        "vv_over_UH2": vv / UHref2,
        "ww_over_UH2": ww / UHref2,
        "uw_over_UH2": uw / UHref2,
        "neg_uw_over_UH2": -uw / UHref2,
    }
    return target_profile_df, q, float(U_H_ref)


class SpectralTiltResultReader:
    def __init__(self, case_dir: Path, zip_path: Optional[Path] = None):
        self.case_dir = Path(case_dir).resolve()
        self.zip_path = Path(zip_path).resolve() if zip_path else None
        self._zip = None
        if self.zip_path:
            self._zip = zipfile.ZipFile(self.zip_path, "r")

    def close(self):
        if self._zip is not None:
            self._zip.close()

    def calibration_dir(self) -> Path:
        return self.case_dir / CALIBRATION_LOG_REL

    def output_base(self) -> Path:
        if self.zip_path:
            return self.zip_path.parent / OUTPUT_SUBDIR
        return self.calibration_dir() / OUTPUT_SUBDIR

    def iteration_names(self) -> List[str]:
        if self._zip is not None:
            names = self._zip.namelist()
            iters = sorted({n.split("/")[0] for n in names if re.match(r"iteration\d+(/|$)", n)}, key=natural_iteration_key)
            if not iters:
                # Some archives may include log/spectralTiltCalibration/iterationNN/...
                iters = sorted(
                    {"/".join(n.split("/")[:3]) for n in names if re.search(r"iteration\d+", n)},
                    key=natural_iteration_key,
                )
            return iters

        calib_dir = self.calibration_dir()
        if not calib_dir.exists():
            # Also allow case_path itself to be the spectralTiltCalibration folder.
            if (self.case_dir / "iteration01").exists() or any(self.case_dir.glob("iteration*")):
                calib_dir = self.case_dir
            else:
                raise FileNotFoundError(f"Could not find spectral tilt calibration directory: {calib_dir}")
        return [p.name for p in sorted(calib_dir.glob("iteration*"), key=natural_iteration_key) if p.is_dir()]

    def read_profile(self, iteration_name: str, profile_csv: str) -> pd.DataFrame:
        if self._zip is not None:
            candidates = [
                f"{iteration_name}/data/{profile_csv}",
                f"log/spectralTiltCalibration/{iteration_name}/data/{profile_csv}",
            ]
            for cand in candidates:
                try:
                    with self._zip.open(cand) as f:
                        return pd.read_csv(f)
                except KeyError:
                    continue
            # Search fallback.
            suffix = f"{iteration_name}/data/{profile_csv}"
            matches = [n for n in self._zip.namelist() if n.endswith(suffix)]
            if matches:
                with self._zip.open(matches[0]) as f:
                    return pd.read_csv(f)
            raise FileNotFoundError(f"Could not find {profile_csv} for {iteration_name} in {self.zip_path}")

        calib_dir = self.calibration_dir()
        if not (calib_dir / iteration_name).exists() and (self.case_dir / iteration_name).exists():
            calib_dir = self.case_dir
        path = calib_dir / iteration_name / "data" / profile_csv
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_csv(path)


def apply_plot_style():
    if not DARK_STYLE:
        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "text.color": "black",
            "font.family": "serif",
            "mathtext.fontset": "cm",
        })
        return {
            "bg": "white",
            "fg": "black",
            "grid": "0.70",
            "exp_edge": "black",
            "exp_face": "none",
            "target": "0.25",
            "current": "tab:purple",
            "updated": "tab:blue",
            "les": "tab:red",
        }
    plt.rcParams.update({
        "figure.facecolor": "#222b38",
        "axes.facecolor": "#222b38",
        "savefig.facecolor": "#222b38",
        "axes.edgecolor": "#d5dce8",
        "axes.labelcolor": "#d5dce8",
        "xtick.color": "#d5dce8",
        "ytick.color": "#d5dce8",
        "text.color": "#d5dce8",
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })
    return {
        "bg": "#222b38",
        "fg": "#d5dce8",
        "grid": "#7d8795",
        "exp_edge": "#d5dce8",
        "exp_face": "none",
        "target": "#c8ccd4",
        "current": "#bb6bd9",
        "updated": "#2d9cdb",
        "les": "#ff3b30",
    }


def finite_concat(qs: Iterable[Dict[str, np.ndarray]], key: str, y_max: Optional[float] = None) -> np.ndarray:
    vals = []
    for q in qs:
        if q is None or key not in q:
            continue
        x = np.asarray(q[key], dtype=float)
        y = np.asarray(q.get("z_over_H", np.full_like(x, np.nan)), dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if y_max is not None:
            m &= y <= y_max
        vals.append(x[m])
    if not vals:
        return np.array([], dtype=float)
    return np.concatenate(vals)


def auto_xlim_for_key(qs: Iterable[Dict[str, np.ndarray]], key: str, y_max: Optional[float], positive_zero: bool = True) -> Tuple[float, float]:
    vals = finite_concat(qs, key, y_max)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return DEFAULT_STRESS_XLIMS.get(key, (0.0, 1.0))
    if positive_zero:
        lo = 0.0
        hi = float(np.nanpercentile(vals, 99))
        hi = max(hi, float(np.nanmax(vals)))
        hi *= 1.12
        if hi <= 0 or not np.isfinite(hi):
            hi = DEFAULT_STRESS_XLIMS.get(key, (0.0, 1.0))[1]
        return lo, hi
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    pad = 0.1 * max(abs(hi - lo), 1e-12)
    return lo - pad, hi + pad


def plot_series(ax, key: str, exp_q, target_q, current_q, updated_q, downstream_q, colours, y_max=None):
    # EXP raw points.
    ax.scatter(
        exp_q[key], exp_q["z_over_H"],
        s=EXP_MARKER_SIZE,
        facecolors=colours["exp_face"],
        edgecolors=colours["exp_edge"],
        linewidths=EXP_MARKER_LINEWIDTH,
        label="EXP raw",
        zorder=6,
    )

    if PLOT_TARGET_PROFILE and target_q is not None:
        ax.plot(
            target_q[key], target_q["z_over_H"],
            color=colours["target"], linestyle="--", linewidth=TARGET_LINEWIDTH,
            label="Target", zorder=3,
        )
    if PLOT_CURRENT_INLET and current_q is not None:
        ax.plot(
            current_q[key], current_q["z_over_H"],
            color=colours["current"], linestyle=":", linewidth=CURRENT_LINEWIDTH,
            label="Current inlet", zorder=2,
        )
    if PLOT_UPDATED_INLET and updated_q is not None:
        ax.plot(
            updated_q[key], updated_q["z_over_H"],
            color=colours["updated"], linestyle="-.", linewidth=UPDATED_LINEWIDTH,
            label="Updated inlet", zorder=4,
        )
    if PLOT_DOWNSTREAM_LES and downstream_q is not None:
        ax.plot(
            downstream_q[key], downstream_q["z_over_H"],
            color=colours["les"], linestyle="-", linewidth=LES_LINEWIDTH,
            label="LES downstream", zorder=5,
        )


def configure_axis(ax, key: str, y_lim: Tuple[float, Optional[float]], q_list):
    cfg = AXIS_CONFIG[key]
    y0 = y_lim[0]
    y1 = y_lim[1]
    if y1 is None:
        ymaxs = []
        for q in q_list:
            if q is not None and "z_over_H" in q:
                zoh = np.asarray(q["z_over_H"], dtype=float)
                if np.any(np.isfinite(zoh)):
                    ymaxs.append(float(np.nanmax(zoh)))
        y1 = max(ymaxs) if ymaxs else 3.0
        y1 = float(np.ceil(y1 * 2.0) / 2.0)
    ax.set_ylim(y0, y1)
    yticks = np.arange(y0, y1 + 1e-9, 0.5)
    if len(yticks) > 14:
        yticks = np.arange(y0, y1 + 1e-9, 1.0)
    ax.set_yticks(yticks)

    if key in STRESS_KEYS and AUTO_STRESS_XLIMS:
        ax.set_xlim(*auto_xlim_for_key(q_list, key, y1, positive_zero=(key == "neg_uw_over_UH2" or key.endswith("_over_UH2"))))
    else:
        ax.set_xlim(*cfg.get("xlim", DEFAULT_STRESS_XLIMS.get(key, (0.0, 1.0))))
        if "xticks" in cfg:
            ax.set_xticks(cfg["xticks"])
    ax.set_xlabel(cfg["xlabel"], fontsize=10)
    ax.set_title(cfg["panel"], fontsize=11, pad=2)


def decorate_axis(ax, colours, show_ylabel: bool):
    if show_ylabel:
        ax.set_ylabel(r"$z/H$", fontsize=10)
    else:
        ax.set_ylabel("")
    ax.tick_params(
        axis="both", which="both", direction="in", length=4.0, width=0.8,
        labelbottom=True, labelleft=True, labelsize=8.5, colors=colours["fg"],
    )
    ax.grid(True, which="major", linestyle="--", linewidth=0.50, alpha=0.35, color=colours["grid"])
    for spine in ax.spines.values():
        spine.set_linewidth(0.85)
        spine.set_color(colours["fg"])


def plot_profile_figure(iteration_number: int, exp_q, target_q, current_q, updated_q, downstream_q, output_dir: Path, y_lim, tag: str) -> str:
    colours = apply_plot_style()

    panel_w = 2.15
    panel_h = 3.05
    gap_x = 0.58
    gap_y = 0.58
    left_margin = 0.62
    right_margin = 0.25
    bottom_margin = 0.70
    top_margin = 0.62
    title_extra = 0.18

    fig_w = left_margin + 4 * panel_w + 3 * gap_x + right_margin
    fig_h = bottom_margin + 2 * panel_h + gap_y + top_margin + title_extra
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
    fig.patch.set_facecolor(colours["bg"])

    def rect_in(x, y, w=panel_w, h=panel_h):
        return [x / fig_w, y / fig_h, w / fig_w, h / fig_h]

    y_bottom = bottom_margin
    y_top = bottom_margin + panel_h + gap_y
    x0 = left_margin
    x1 = left_margin + panel_w + gap_x
    x2 = x1 + panel_w + gap_x
    x3 = x2 + panel_w + gap_x

    axes = {
        "U_over_UH":       fig.add_axes(rect_in(x0, y_top)),
        "Iu":              fig.add_axes(rect_in(x1, y_top)),
        "Iv":              fig.add_axes(rect_in(x2, y_top)),
        "Iw":              fig.add_axes(rect_in(x3, y_top)),
        "Lu_over_H":       fig.add_axes(rect_in(x0, y_bottom)),
        "Lv_over_H":       fig.add_axes(rect_in(x1, y_bottom)),
        "Lw_over_H":       fig.add_axes(rect_in(x2, y_bottom)),
        "neg_uw_over_UH2": fig.add_axes(rect_in(x3, y_bottom)),
    }

    q_list = [exp_q, target_q, current_q, updated_q, downstream_q]
    for key in PROFILE_KEYS:
        ax = axes[key]
        plot_series(ax, key, exp_q, target_q, current_q, updated_q, downstream_q, colours, y_max=y_lim[1])
        configure_axis(ax, key, y_lim, q_list)
        decorate_axis(ax, colours, show_ylabel=(key in ("U_over_UH", "Lu_over_H")))

    axes["U_over_UH"].legend(
        loc="upper left", frameon=True, facecolor=colours["bg"], edgecolor=colours["fg"],
        fontsize=8.5, handlelength=1.8, borderpad=0.45, labelspacing=0.35,
    )

    fig.suptitle(
        f"MannHybrid spectral-tilt calibration — iteration {iteration_number} — profiles ({tag})",
        fontsize=13, y=0.982, color=colours["fg"],
    )

    ensure_dir(output_dir)
    png_path = output_dir / f"iteration{iteration_number:02d}_profiles_melaku_style_{tag}.png"
    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(output_dir / f"iteration{iteration_number:02d}_profiles_melaku_style_{tag}.pdf", bbox_inches="tight")
    plt.close(fig)
    return str(png_path)


def plot_reynolds_stress_figure(iteration_number: int, exp_q, target_q, current_q, updated_q, downstream_q, output_dir: Path, y_lim, tag: str) -> str:
    colours = apply_plot_style()
    fig, axes = plt.subplots(1, 4, figsize=(10.0, 5.2), constrained_layout=True)
    fig.patch.set_facecolor(colours["bg"])
    q_list = [exp_q, target_q, current_q, updated_q, downstream_q]

    for ax, key in zip(axes, STRESS_KEYS):
        plot_series(ax, key, exp_q, target_q, current_q, updated_q, downstream_q, colours, y_max=y_lim[1])
        configure_axis(ax, key, y_lim, q_list)
        decorate_axis(ax, colours, show_ylabel=(key == STRESS_KEYS[0]))

    axes[0].legend(
        loc="upper left", frameon=True, facecolor=colours["bg"], edgecolor=colours["fg"],
        fontsize=8.5, handlelength=1.8, borderpad=0.45, labelspacing=0.35,
    )
    fig.suptitle(
        f"MannHybrid spectral-tilt calibration — iteration {iteration_number} — Reynolds stresses ({tag})",
        fontsize=13, color=colours["fg"],
    )
    ensure_dir(output_dir)
    png_path = output_dir / f"iteration{iteration_number:02d}_reynolds_stresses_{tag}.png"
    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(output_dir / f"iteration{iteration_number:02d}_reynolds_stresses_{tag}.pdf", bbox_inches="tight")
    plt.close(fig)
    return str(png_path)


def quantities_to_dataframe(q: Dict[str, np.ndarray], source: str, iteration: Optional[int]) -> pd.DataFrame:
    n = len(q["z_over_H"])
    keys = [
        "z", "z_over_H", "U", "U_over_UH", "Iu", "Iv", "Iw",
        "Lu", "Lv", "Lw", "Lu_over_H", "Lv_over_H", "Lw_over_H",
        "uu", "vv", "ww", "uw", "uu_over_UH2", "vv_over_UH2", "ww_over_UH2", "uw_over_UH2", "neg_uw_over_UH2",
    ]
    data = {"source": source, "iteration": iteration}
    for k in keys:
        data[k] = q.get(k, np.full(n, np.nan))
    return pd.DataFrame(data)


def interpolation_error_metrics(reference_q, test_q, keys: List[str], iteration: int, label: str, y_max: Optional[float]) -> pd.DataFrame:
    rows = []
    z_ref = np.asarray(reference_q["z_over_H"], dtype=float)
    for key in keys:
        x_ref = np.asarray(reference_q[key], dtype=float)
        z_test = np.asarray(test_q["z_over_H"], dtype=float)
        x_test = np.asarray(test_q[key], dtype=float)
        mref = np.isfinite(z_ref) & np.isfinite(x_ref)
        mtest = np.isfinite(z_test) & np.isfinite(x_test)
        if y_max is not None:
            mref &= z_ref <= y_max
            mtest &= z_test <= y_max
        if np.count_nonzero(mref) < 2 or np.count_nonzero(mtest) < 2:
            continue
        order = np.argsort(z_test[mtest])
        pred = np.interp(z_ref[mref], z_test[mtest][order], x_test[mtest][order])
        ref = x_ref[mref]
        denom = np.maximum(np.abs(ref), 1e-12)
        rows.append({
            "iteration": iteration,
            "comparison": label,
            "key": key,
            "zmax_over_H": y_max,
            "bias_mean": float(np.nanmean(pred - ref)),
            "mae": float(np.nanmean(np.abs(pred - ref))),
            "mape_percent": float(100.0 * np.nanmean(np.abs(pred - ref) / denom)),
            "mean_ratio": float(np.nanmean(pred / np.where(np.abs(ref) > 1e-12, ref, np.nan))),
        })
    return pd.DataFrame(rows)


def create_animation(frame_paths: List[str], output_path: Path, fps: float = ANIMATION_FPS):
    frame_paths = [Path(p) for p in frame_paths if Path(p).exists()]
    if not frame_paths:
        return None
    try:
        from PIL import Image
    except Exception as exc:
        warnings.warn(f"Could not import PIL/Pillow; skipping animation: {exc}")
        return None
    duration_ms = int(round(1000.0 / max(float(fps), 1e-9)))
    frames = []
    max_w = max_h = 0
    for path in frame_paths:
        im = Image.open(path).convert("RGB")
        frames.append(im)
        max_w = max(max_w, im.width)
        max_h = max(max_h, im.height)
    bg = (255, 255, 255) if not DARK_STYLE else (34, 43, 56)
    padded = []
    for im in frames:
        canvas = Image.new("RGB", (max_w, max_h), bg)
        canvas.paste(im, ((max_w - im.width) // 2, (max_h - im.height) // 2))
        padded.append(canvas)
    output_path = Path(output_path)
    padded[0].save(output_path, save_all=True, append_images=padded[1:], duration=duration_ms, loop=0, optimize=False)
    return str(output_path)


def parse_building_height(case_dir: Path) -> float:
    try:
        variable_dict = LES._caseFiles.parse_setup_file(str(case_dir))
        return float(variable_dict["buildingHeight"])
    except Exception as exc:
        raw = os.environ.get("MST_BUILDING_HEIGHT") or os.environ.get("BUILDING_HEIGHT")
        if raw is not None:
            return float(raw)
        warnings.warn(
            f"Could not parse buildingHeight from setUp using windLespy: {exc}. "
            "Falling back to H=0.5 m. Set MST_BUILDING_HEIGHT to override."
        )
        return 0.5


def main():
    case_dir = Path(case_path).resolve()
    zip_path = Path(SPECTRAL_TILT_RESULTS_ZIP).resolve() if SPECTRAL_TILT_RESULTS_ZIP else None
    H = parse_building_height(case_dir)

    reader = SpectralTiltResultReader(case_dir, zip_path)
    output_dir = ensure_dir(reader.output_base())

    print("Building experimental target from raw NHERI wind-tunnel data...")
    exp_df, exp_q, U_H_ref = build_nheri_experimental_quantities(approach_flow_data, H)
    print(f"  building height H = {H:.6g} m")
    print(f"  NHERI U_H = {U_H_ref:.6g} m/s")
    print(f"  NHERI measured profile points = {len(exp_q['z_over_H'])}")

    iteration_names = reader.iteration_names()
    if not iteration_names:
        raise RuntimeError("No spectral-tilt iteration directories found.")
    print(f"Found {len(iteration_names)} spectral-tilt iterations.")
    print(f"Writing figures to: {output_dir}")

    if WRITE_QA_CSV:
        exp_df.to_csv(output_dir / "nheri_raw_profile_dataframe_used.csv", index=False)
        quantities_to_dataframe(exp_q, "EXP_NHERI_raw", None).to_csv(output_dir / "nheri_raw_profile_quantities_used.csv", index=False)

    qa_rows = [quantities_to_dataframe(exp_q, "EXP_NHERI_raw", None)] if WRITE_QA_CSV else []
    metrics_rows = []
    profile_frames_by_tag: Dict[str, List[str]] = {}
    stress_frames_by_tag: Dict[str, List[str]] = {}

    try:
        for iter_name in iteration_names:
            iteration_number = natural_iteration_key(iter_name)
            print(f"Processing {iter_name}...")

            target_df = reader.read_profile(iter_name, "target_profile.csv")
            current_df = reader.read_profile(iter_name, "current_profile.csv")
            downstream_df = reader.read_profile(iter_name, "downstream_profile.csv")
            updated_df = reader.read_profile(iter_name, "updated_profile.csv")

            target_q = profile_df_to_quantities(target_df, U_H_ref, H, "TARGET_spectral_tilt")
            current_q = profile_df_to_quantities(current_df, U_H_ref, H, "CURRENT_inlet")
            downstream_q = profile_df_to_quantities(downstream_df, U_H_ref, H, "LES_downstream")
            updated_q = profile_df_to_quantities(updated_df, U_H_ref, H, "UPDATED_inlet")

            if WRITE_QA_CSV:
                qa_rows.extend([
                    quantities_to_dataframe(target_q, "TARGET_spectral_tilt", iteration_number),
                    quantities_to_dataframe(current_q, "CURRENT_inlet", iteration_number),
                    quantities_to_dataframe(downstream_q, "LES_downstream", iteration_number),
                    quantities_to_dataframe(updated_q, "UPDATED_inlet", iteration_number),
                ])

            for y0, y1, tag in VERTICAL_RANGES:
                if y1 is None:
                    ymax_candidates = []
                    for q in [target_q, current_q, downstream_q, updated_q]:
                        ymax_candidates.append(float(np.nanmax(q["z_over_H"])))
                    y_lim = (float(y0), float(np.ceil(max(ymax_candidates) * 2.0) / 2.0))
                else:
                    y_lim = (float(y0), float(y1))

                prof_path = plot_profile_figure(
                    iteration_number, exp_q, target_q, current_q, updated_q, downstream_q,
                    output_dir, y_lim, tag,
                )
                stress_path = plot_reynolds_stress_figure(
                    iteration_number, exp_q, target_q, current_q, updated_q, downstream_q,
                    output_dir, y_lim, tag,
                )
                profile_frames_by_tag.setdefault(tag, []).append(prof_path)
                stress_frames_by_tag.setdefault(tag, []).append(stress_path)

                metrics_rows.append(interpolation_error_metrics(exp_q, downstream_q, PROFILE_KEYS + STRESS_KEYS, iteration_number, f"LES_downstream_vs_EXP_{tag}", y_lim[1]))
                metrics_rows.append(interpolation_error_metrics(target_q, downstream_q, PROFILE_KEYS + STRESS_KEYS, iteration_number, f"LES_downstream_vs_TARGET_{tag}", y_lim[1]))

            print(f"  wrote figures for iteration {iteration_number}")

        if WRITE_QA_CSV and qa_rows:
            pd.concat(qa_rows, ignore_index=True).to_csv(output_dir / "all_plotted_profile_quantities.csv", index=False)
        if metrics_rows:
            pd.concat(metrics_rows, ignore_index=True).to_csv(output_dir / "downstream_error_metrics.csv", index=False)

        if CREATE_ANIMATION:
            for tag, frames in profile_frames_by_tag.items():
                path = create_animation(frames, output_dir / f"profile_iteration_progression_{tag}.gif")
                if path:
                    print(f"  wrote animation {path}")
            for tag, frames in stress_frames_by_tag.items():
                path = create_animation(frames, output_dir / f"reynolds_stress_iteration_progression_{tag}.gif")
                if path:
                    print(f"  wrote animation {path}")

    finally:
        reader.close()

    print("\nDone.")
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    main()
