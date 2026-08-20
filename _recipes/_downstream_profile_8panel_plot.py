# -*- coding: utf-8 -*-
"""
Downstream DFSR profile diagnostic plotter with 8-panel Melaku-style layout.

This script loads the target, current inlet, and downstream LES profiles from an
OpenFOAM/DFSR case and creates an 8-panel overview figure:

    Top row:    U/U_H, Iu, Iv, Iw
    Bottom row: <u'w'>/U_H^2, Lu/H, Lv/H, Lw/H

The Reynolds shear stress panel supports negative values and draws a vertical
zero line for reference.

The script is intentionally self-contained for plotting/diagnostics. It does
not require editing windlespy/lesWindpy. By default it does NOT overwrite the
case inlet profile. Set WRITE_UPDATED_INLET_PROFILE = True only when you want
to write a new profile file.
"""

import json
import os
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# REQUIRED / USER INPUTS - EDIT THESE FIRST
# =============================================================================

# OpenFOAM case folder.
CASE_PATH = r"C:\Users\david\OneDrive\Documents\PhD\Year 1\spctral_tilt_cailbration\classic_wong_dfsr_new_mesh"

WINDLESPY_PATH = r"C:\Users\david\OneDrive\Documents\PhD\Year 1"

# Downstream probes folder. Leave as None for case_path/postProcessing/probes2.
DOWNSTREAM_PROBES_FOLDER = None

# Building height H. Leave as None to read variable_dict['buildingHeight'] from
# LES._caseFiles.parse_setup_file(case_path).
BUILDING_HEIGHT = 0.5

# Burn-in time for downstream statistics. Leave as None to read
# case_path/log/downstreamCalibration/sim_init.json. If that file does not exist,
# FALLBACK_BURN_IN_TIME is used.
BURN_IN_TIME = None
FALLBACK_BURN_IN_TIME = 10.0

# Optional experimental/raw profile. Leave as None to look for:
# case_path/constant/boundaryData/windProfile/targetExperimentalProfile
# The file may contain either 8 classic columns:
#     z U Iu Iv Iw Lu Lv Lw
# or 9 columns including Reynolds shear stress:
#     z U Iu Iv Iw Lu Lv Lw uwStress
# Header rows are allowed but not required.
EXPERIMENTAL_PROFILE_PATH = None

# Optional smoothed target profile. Leave as None to use targetProfile from the
# case via LES._profileCalibration.get_dfsr_target_profile_df(case_path).
SMOOTHED_TARGET_PROFILE_PATH = None

# Output figure root. Leave as None for:
# case_path/log/downstreamCalibration/diagnosticPlots
FIG_ROOT = None

# Figure/control settings.
ITERATION = None                  # None = infer from log folders; used only in title/file name.
Z_MAX_OVER_H = 3.0                # plot up to this nondimensional height.
FIGURE_THEME = "dark"             # "dark" or "white"
DPI = 300

# Updated inlet profile curve/settings.
# This only controls the cyan "Updated inlet" curve and optional file write.
PLOT_UPDATED_INLET = True
WRITE_UPDATED_INLET_PROFILE = False
UPDATED_PROFILE_RELAXATION_FACTOR = 0.9
WRITE_ITERATION_PROFILE_SNAPSHOTS = False

# Safety: signed uw update should not use the original adaptive ratio logic,
# because u'w' is usually negative and can cross zero. This fixed relaxation is
# used only for column 7, when present.
UW_SIGNED_RELAXATION_FACTOR = 0.9


# =============================================================================
# IMPORT WINDLESPY
# =============================================================================


def import_windlespy(windlespy_path=None):
    """Import windlespy from the supplied path or from two folders above script."""
    if windlespy_path is None:
        cwd = os.path.dirname(os.path.abspath(__file__))
        windlespy_path = os.path.abspath(os.path.join(cwd, "..", ".."))

    sys.path.append(windlespy_path)
    try:
        import windlespy as LES  # noqa: N806
    finally:
        try:
            sys.path.remove(windlespy_path)
        except ValueError:
            pass

    return LES


LES = import_windlespy(WINDLESPY_PATH)


# =============================================================================
# PATH / IO HELPERS
# =============================================================================


def windows_long_path(path):
    """Return a Windows long-path-safe version of path when running on Windows."""
    path = os.path.abspath(path)
    if os.name != "nt":
        return path

    bs = chr(92)
    long_prefix = bs * 2 + "?" + bs
    unc_prefix = bs * 2 + "?" + bs + "UNC" + bs

    if path.startswith(long_prefix):
        return path
    if path.startswith(bs * 2):
        return unc_prefix + path[2:]
    return long_prefix + path


def safe_makedirs(path):
    os.makedirs(windows_long_path(path), exist_ok=True)


def safe_savefig(fig, path, dpi=300, bbox_inches="tight"):
    path = os.path.abspath(path)
    safe_makedirs(os.path.dirname(path))
    fig.savefig(windows_long_path(path), dpi=dpi, bbox_inches=bbox_inches)


def is_float_token(token):
    try:
        float(token)
        return True
    except Exception:
        return False


def read_profile_table(path, optional=False):
    """
    Read DFSR-style profile files with or without a header.

    Supported formats:
        8 columns:  z, U, Iu, Iv, Iw, Lu, Lv, Lw
        9 columns:  z, U, Iu, Iv, Iw, Lu, Lv, Lw, uwStress
        15 columns: z, U, Iu, Iv, Iw, Lu, Lv, Lw, uu, vv, ww, uv, uw, vw, uwStress

    Also accepts already-normalised experimental columns such as z_over_H,
    U_over_UH, Lu_over_H, etc., if a header is present.
    """
    if path is None or not os.path.exists(path):
        if optional:
            return None
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        non_comment_lines = [
            line.strip() for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]

    if not non_comment_lines:
        if optional:
            return None
        raise ValueError(f"Profile file is empty: {path}")

    first_tokens = non_comment_lines[0].replace(",", " ").split()
    has_header = not all(is_float_token(tok) for tok in first_tokens)

    if has_header:
        df = pd.read_csv(path, sep=r"[\s,]+", engine="python", comment="#")
    else:
        df = pd.read_csv(path, sep=r"[\s,]+", engine="python", comment="#", header=None)
        n_cols = df.shape[1]

        if n_cols == 8:
            df.columns = ["z", "U", "Iu", "Iv", "Iw", "Lu", "Lv", "Lw"]
        elif n_cols == 9:
            df.columns = ["z", "U", "Iu", "Iv", "Iw", "Lu", "Lv", "Lw", "uwStress"]
        elif n_cols == 15:
            df.columns = [
                "z", "U", "Iu", "Iv", "Iw", "Lu", "Lv", "Lw",
                "uu", "vv", "ww", "uv", "uw", "vw", "uwStress",
            ]
        else:
            raise ValueError(
                f"Unsupported profile format in {path}: found {n_cols} columns. "
                "Expected 8, 9, or 15 columns."
            )

    return df


def get_default_paths(case_path):
    wind_profile_dir = os.path.join(case_path, "constant", "boundaryData", "windProfile")
    return {
        "downstream_probes_folder": os.path.join(case_path, "postProcessing", "probes2"),
        "experimental_profile_path": os.path.join(wind_profile_dir, "targetExperimentalProfile"),
        "smoothed_target_profile_path": os.path.join(wind_profile_dir, "targetSmoothedProfile"),
        "profile_path": os.path.join(wind_profile_dir, "profile"),
        "target_profile_path": os.path.join(wind_profile_dir, "targetProfile"),
        "fig_root": os.path.join(case_path, "log", "downstreamCalibration", "diagnosticPlots"),
        "sim_init_json": os.path.join(case_path, "log", "downstreamCalibration", "sim_init.json"),
        "downstream_calibration_log": os.path.join(case_path, "log", "downstreamCalibration"),
    }


def load_building_height(case_path, override=None):
    if override is not None:
        return float(override)

    try:
        variable_dict = LES._caseFiles.parse_setup_file(case_path)
        return float(variable_dict["buildingHeight"])
    except Exception as exc:
        raise RuntimeError(
            "BUILDING_HEIGHT is None and could not be read from setup file. "
            "Set BUILDING_HEIGHT manually at the top of this script."
        ) from exc


def load_burn_in_time(sim_init_json, override=None, fallback=10.0):
    if override is not None:
        return float(override)

    if os.path.exists(sim_init_json):
        with open(sim_init_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "burn_in_time" in data:
            return float(data["burn_in_time"])

    return float(fallback)


def infer_iteration(case_path, override=None):
    """Infer next iteration number from downstreamCalibration/iteration* folders."""
    if override is not None:
        return int(override)

    iter_root = os.path.join(case_path, "log", "downstreamCalibration")
    if not os.path.isdir(iter_root):
        return 0

    iter_nums = []
    for name in os.listdir(iter_root):
        if name.startswith("iteration"):
            try:
                iter_nums.append(int(name.replace("iteration", "")))
            except ValueError:
                pass

    if not iter_nums:
        return 0
    return max(iter_nums)


# =============================================================================
# PROFILE CONVERSION HELPERS
# =============================================================================


def first_existing_column(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def profile_df_to_array(df):
    """
    Convert a profile dataframe to the internal DFSR array layout used by your repo:

        col 0: U
        col 1: R_11 = uu = (Iu U)^2
        col 2: R_22 = vv = (Iv U)^2
        col 3: R_33 = ww = (Iw U)^2
        col 4: Lu
        col 5: Lv
        col 6: Lw
        col 7: R_31 = uwStress = <u'w'>, optional and signed
    """
    if df is None:
        return None

    if "U" not in df.columns:
        raise KeyError("Profile dataframe must contain dimensional mean velocity column 'U'.")

    U = df["U"].to_numpy(dtype=float)

    if all(col in df.columns for col in ["uu", "vv", "ww"]):
        R11 = df["uu"].to_numpy(dtype=float)
        R22 = df["vv"].to_numpy(dtype=float)
        R33 = df["ww"].to_numpy(dtype=float)
    else:
        for col in ["Iu", "Iv", "Iw"]:
            if col not in df.columns:
                raise KeyError(f"Profile dataframe must contain '{col}' or uu/vv/ww columns.")
        R11 = (df["Iu"].to_numpy(dtype=float) * U) ** 2
        R22 = (df["Iv"].to_numpy(dtype=float) * U) ** 2
        R33 = (df["Iw"].to_numpy(dtype=float) * U) ** 2

    for col in ["Lu", "Lv", "Lw"]:
        if col not in df.columns:
            raise KeyError(f"Profile dataframe must contain '{col}'.")

    arr_cols = [
        U,
        R11,
        R22,
        R33,
        df["Lu"].to_numpy(dtype=float),
        df["Lv"].to_numpy(dtype=float),
        df["Lw"].to_numpy(dtype=float),
    ]

    uw_col = first_existing_column(
        df,
        ["uwStress", "UWStress", "Ruw", "R31", "R_31", "uw", "u'w'", "u_w", "cov_uw"],
    )
    if uw_col is not None:
        arr_cols.append(df[uw_col].to_numpy(dtype=float))

    return np.stack(arr_cols, axis=1)


def load_target_profile(case_path):
    """Load targetProfile robustly, including 8-, 9-, and 15-column cases."""
    target_df = LES._profileCalibration.get_dfsr_target_profile_df(case_path)
    target_array = profile_df_to_array(target_df)
    return target_df, target_array


def load_current_inlet_profile(profile_path):
    """Load current windProfile/profile robustly, including signed uwStress if present."""
    current_df = read_profile_table(profile_path, optional=False)
    current_array = profile_df_to_array(current_df)
    return current_df, current_array


def ensure_uw_column(profile_array, reference_array=None, fallback_value=0.0):
    """Append an optional signed uw column if profile_array has only 7 columns."""
    if profile_array is None:
        return None

    arr = np.asarray(profile_array, dtype=float)
    if arr.shape[1] >= 8:
        return arr

    if reference_array is not None and np.asarray(reference_array).shape[1] >= 8:
        uw = np.asarray(reference_array, dtype=float)[:, 7]
    else:
        uw = np.full(arr.shape[0], float(fallback_value))

    return np.column_stack([arr, uw])


def make_updated_profile_array(current_array, target_array, downstream_array,
                               relaxation_factor=0.9, uw_relaxation_factor=0.9):
    """
    Compute a proposed updated inlet profile.

    For U, variances, and length scales, this mirrors your existing adaptive
    Wong-style update. For signed uwStress, a fixed relaxation is used because
    adaptive ratios are unstable for negative or near-zero quantities.
    """
    current = np.asarray(current_array, dtype=float)
    target = np.asarray(target_array, dtype=float)
    downstream = np.asarray(downstream_array, dtype=float)

    if target.shape[1] >= 8 or downstream.shape[1] >= 8:
        current = ensure_uw_column(current, reference_array=target)
        target = ensure_uw_column(target, reference_array=current)
        downstream = ensure_uw_column(downstream, reference_array=target)

    n_cols = min(current.shape[1], target.shape[1], downstream.shape[1])
    current = current[:, :n_cols]
    target = target[:, :n_cols]
    downstream = downstream[:, :n_cols]

    updated = current.copy()

    # Classic non-signed DFSR columns: U, R11, R22, R33, Lu, Lv, Lw.
    n_classic = min(7, n_cols)
    with np.errstate(divide="ignore", invalid="ignore"):
        adaptive = relaxation_factor * (current[:, :n_classic] / downstream[:, :n_classic])
    adaptive = np.where(np.isfinite(adaptive), adaptive, relaxation_factor)
    adaptive = np.clip(adaptive, 0.5, 5.0)
    updated[:, :n_classic] = current[:, :n_classic] + adaptive * (
        target[:, :n_classic] - downstream[:, :n_classic]
    )

    if n_classic >= 1:
        updated[:, 0] = np.clip(updated[:, 0], 0.01, None)
    if n_classic >= 4:
        updated[:, 1:4] = np.clip(updated[:, 1:4], 1e-8, None)
    if n_classic >= 7:
        updated[:, 4:7] = np.clip(updated[:, 4:7], 0.01, None)

    # Signed shear stress column.
    if n_cols >= 8:
        updated[:, 7] = current[:, 7] + uw_relaxation_factor * (target[:, 7] - downstream[:, 7])

    return updated


def array_to_profile_df(z_array, profile_array):
    """Convert internal array back to physical DFSR dataframe for writing/inspection."""
    arr = np.asarray(profile_array, dtype=float)
    U = arr[:, 0]
    df_dict = {
        "z": z_array,
        "U": U,
        "Iu": np.sqrt(np.maximum(arr[:, 1], 0.0)) / U,
        "Iv": np.sqrt(np.maximum(arr[:, 2], 0.0)) / U,
        "Iw": np.sqrt(np.maximum(arr[:, 3], 0.0)) / U,
        "Lu": arr[:, 4],
        "Lv": arr[:, 5],
        "Lw": arr[:, 6],
    }
    if arr.shape[1] >= 8:
        df_dict["uwStress"] = arr[:, 7]
    return pd.DataFrame(df_dict)


def write_profile_file(path, df):
    safe_makedirs(os.path.dirname(path))
    np.savetxt(windows_long_path(path), df.to_numpy(), fmt="%.6f", delimiter="\t")


def profile_df_to_plot_quantities(df, H, U_H_ref=None):
    """Convert a physical or normalised dataframe to plotting quantities."""
    if df is None:
        return None

    q = {}

    if "z_over_H" in df.columns:
        q["z_over_H"] = df["z_over_H"].to_numpy(dtype=float)
    elif "z" in df.columns:
        q["z_over_H"] = df["z"].to_numpy(dtype=float) / H
    else:
        raise KeyError("Profile dataframe must contain 'z' or 'z_over_H'.")

    if "U_over_UH" in df.columns:
        q["U_over_UH"] = df["U_over_UH"].to_numpy(dtype=float)
    elif "U" in df.columns:
        if U_H_ref is None:
            raise ValueError("U_H_ref is required when dataframe contains dimensional 'U'.")
        q["U_over_UH"] = df["U"].to_numpy(dtype=float) / U_H_ref

    for key in ["Iu", "Iv", "Iw"]:
        if key in df.columns:
            q[key] = df[key].to_numpy(dtype=float)

    if "Lu_over_H" in df.columns:
        q["Lu_over_H"] = df["Lu_over_H"].to_numpy(dtype=float)
    elif "Lu" in df.columns:
        q["Lu_over_H"] = df["Lu"].to_numpy(dtype=float) / H

    if "Lv_over_H" in df.columns:
        q["Lv_over_H"] = df["Lv_over_H"].to_numpy(dtype=float)
    elif "Lv" in df.columns:
        q["Lv_over_H"] = df["Lv"].to_numpy(dtype=float) / H

    if "Lw_over_H" in df.columns:
        q["Lw_over_H"] = df["Lw_over_H"].to_numpy(dtype=float)
    elif "Lw" in df.columns:
        q["Lw_over_H"] = df["Lw"].to_numpy(dtype=float) / H

    uw_col = first_existing_column(
        df,
        ["uwStress_over_UH2", "uw_over_UH2", "R31_over_UH2", "Ruw_over_UH2"],
    )
    if uw_col is not None:
        q["uwStress_over_UH2"] = df[uw_col].to_numpy(dtype=float)
    else:
        uw_col = first_existing_column(
            df,
            ["uwStress", "UWStress", "Ruw", "R31", "R_31", "uw", "u'w'", "u_w", "cov_uw"],
        )
        if uw_col is not None:
            if U_H_ref is None:
                raise ValueError("U_H_ref is required when dataframe contains dimensional uwStress.")
            q["uwStress_over_UH2"] = df[uw_col].to_numpy(dtype=float) / (U_H_ref ** 2)

    return q


def profile_array_to_plot_quantities(z_array, profile_array, H, U_H_ref=None):
    """Convert internal DFSR array to plotting quantities."""
    if profile_array is None:
        return None

    arr = np.asarray(profile_array, dtype=float)
    U = arr[:, 0]

    if U_H_ref is None:
        U_H_ref = np.interp(H, z_array, U)

    q = {
        "z_over_H": np.asarray(z_array, dtype=float) / H,
        "U_over_UH": U / U_H_ref,
    }

    if arr.shape[1] >= 4:
        q["Iu"] = np.sqrt(np.maximum(arr[:, 1], 0.0)) / U
        q["Iv"] = np.sqrt(np.maximum(arr[:, 2], 0.0)) / U
        q["Iw"] = np.sqrt(np.maximum(arr[:, 3], 0.0)) / U

    if arr.shape[1] >= 7:
        q["Lu_over_H"] = arr[:, 4] / H
        q["Lv_over_H"] = arr[:, 5] / H
        q["Lw_over_H"] = arr[:, 6] / H

    if arr.shape[1] >= 8:
        q["uwStress_over_UH2"] = arr[:, 7] / (U_H_ref ** 2)

    return q


# =============================================================================
# PLOTTING
# =============================================================================


def finite_values_from_profiles(profiles, key):
    values = []
    for q in profiles:
        if q is None or key not in q:
            continue
        arr = np.asarray(q[key], dtype=float)
        values.append(arr[np.isfinite(arr)])
    if not values:
        return np.array([])
    return np.concatenate(values)


def padded_xlim(values, fallback, include_zero=False):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return fallback

    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))

    if include_zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)

    if np.isclose(lo, hi):
        pad = max(abs(lo) * 0.1, 0.01)
    else:
        pad = 0.08 * (hi - lo)

    return (lo - pad, hi + pad)


def set_theme(theme):
    if theme.lower() == "white":
        return {
            "bg": "white",
            "fg": "black",
            "grid": "0.65",
            "exp_edge": "black",
            "exp_face": "none",
            "smooth_target": "#b8860b",
            "mapped_target": "0.45",
            "downstream": "#d62728",
            "new_inlet": "#1f77b4",
        }

    return {
        "bg": "#222b38",
        "fg": "#d5dce8",
        "grid": "#7d8795",
        "exp_edge": "#d5dce8",
        "exp_face": "none",
        "smooth_target": "#f2c94c",
        "mapped_target": "#9aa4b2",
        "downstream": "#ff3b30",
        "new_inlet": "#56ccf2",
    }


def plot_downstream_profiles_melaku_style_8panel(
    fig_dir,
    iteration,
    experimental_q,
    smoothed_target_q,
    downstream_q,
    new_inlet_q,
    target_q=None,
    z_max_over_H=3.0,
    theme="light",
    dpi=300,
):
    """
    Create 8-panel Melaku/NHERI-style profile figure.

    Layout:
        top row:    U/UH, Iu, Iv, Iw
        bottom row: u'w'/UH^2, Lu/H, Lv/H, Lw/H

    This version uses a taller figure aspect ratio and fixed axis scales
    similar to the example image.
    """

    old_rc = plt.rcParams.copy()

    # Force white publication-style figure, similar to the figure you showed.
    colours = {
        "bg": "white",
        "fg": "black",
        "grid": "0.85",
        "exp_edge": "black",
        "exp_face": "none",
        "downstream": "#d62728",
        "mapped_target": "0.45",
        "smooth_target": "#ffbf00",
        "new_inlet": "#1f77b4",
    }

    plt.rcParams.update({
        "figure.facecolor": colours["bg"],
        "axes.facecolor": colours["bg"],
        "savefig.facecolor": colours["bg"],
        "axes.edgecolor": colours["fg"],
        "axes.labelcolor": colours["fg"],
        "xtick.color": colours["fg"],
        "ytick.color": colours["fg"],
        "text.color": colours["fg"],
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })

    try:
        axis_config = {
            "U_over_UH": {
                "xlim": (0.0, 1.5),
                "xticks": [0.0, 0.5, 1.0, 1.5],
                "xlabel": r"$U_{av}/U_H$",
                "panel": "(a)",
            },
            "Iu": {
                "xlim": (0.0, 0.3),
                "xticks": [0.0, 0.1, 0.2, 0.3],
                "xlabel": r"$I_u$",
                "panel": "(b)",
            },
            "Iv": {
                "xlim": (0.0, 0.3),
                "xticks": [0.0, 0.1, 0.2, 0.3],
                "xlabel": r"$I_v$",
                "panel": "(c)",
            },
            "Iw": {
                "xlim": (0.0, 0.3),
                "xticks": [0.0, 0.1, 0.2, 0.3],
                "xlabel": r"$I_w$",
                "panel": "(d)",
            },
            "uwStress_over_UH2": {
                "xlim": (-0.08, 0.02),
                "xticks": [-0.08, -0.06, -0.04, -0.02, 0.0, 0.02],
                "xlabel": r"$\overline{u'w'}/U_H^2$",
                "panel": "(e)",
            },
            "Lu_over_H": {
                "xlim": (0.0, 4.0),
                "xticks": [0.0, 1.0, 2.0, 3.0, 4.0],
                "xlabel": r"$L_u/H$",
                "panel": "(f)",
            },
            "Lv_over_H": {
                "xlim": (0.0, 2.0),
                "xticks": [0.0, 0.5, 1.0, 1.5, 2.0],
                "xlabel": r"$L_v/H$",
                "panel": "(g)",
            },
            "Lw_over_H": {
                "xlim": (0.0, 2.0),
                "xticks": [0.0, 0.5, 1.0, 1.5, 2.0],
                "xlabel": r"$L_w/H$",
                "panel": "(h)",
            },
        }

        panel_keys = [
            "U_over_UH", "Iu", "Iv", "Iw",
            "uwStress_over_UH2", "Lu_over_H", "Lv_over_H", "Lw_over_H",
        ]

        # Similar overall aspect ratio to the image you showed:
        # wider than tall, but with relatively tall/narrow panels.
        fig, axes_2d = plt.subplots(
            2,
            4,
            figsize=(12.4, 8.3),
            constrained_layout=False,
        )

        axes = dict(zip(panel_keys, axes_2d.ravel()))
        fig.patch.set_facecolor(colours["bg"])

        def plot_one_profile(ax, q, key, kind, label, zorder):
            if q is None:
                return
            if key not in q:
                return
            if "z_over_H" not in q:
                return

            x = np.asarray(q[key], dtype=float)
            y = np.asarray(q["z_over_H"], dtype=float)

            mask = (
                np.isfinite(x)
                & np.isfinite(y)
                & (y >= 0.0)
                & (y <= z_max_over_H)
            )

            if not np.any(mask):
                return

            if kind == "experimental":
                ax.scatter(
                    x[mask],
                    y[mask],
                    s=26,
                    facecolors=colours["exp_face"],
                    edgecolors=colours["exp_edge"],
                    linewidths=1.05,
                    label=label,
                    zorder=zorder,
                )

            elif kind == "downstream":
                ax.plot(
                    x[mask],
                    y[mask],
                    color=colours["downstream"],
                    linestyle="-",
                    linewidth=2.0,
                    label=label,
                    zorder=zorder,
                )

            elif kind == "target":
                ax.plot(
                    x[mask],
                    y[mask],
                    color=colours["mapped_target"],
                    linestyle=":",
                    linewidth=1.4,
                    label=label,
                    zorder=zorder,
                )

            elif kind == "smoothed":
                ax.plot(
                    x[mask],
                    y[mask],
                    color=colours["smooth_target"],
                    linestyle="--",
                    linewidth=1.5,
                    label=label,
                    zorder=zorder,
                )

            elif kind == "updated":
                ax.plot(
                    x[mask],
                    y[mask],
                    color=colours["new_inlet"],
                    linestyle="-.",
                    linewidth=1.5,
                    label=label,
                    zorder=zorder,
                )

        for key in panel_keys:
            ax = axes[key]
            cfg = axis_config[key]

            plot_one_profile(ax, experimental_q, key, "experimental", "EXP", 8)
            plot_one_profile(ax, downstream_q, key, "downstream", "LES", 7)

            # These are optional. They only appear if you pass them in.
            plot_one_profile(ax, smoothed_target_q, key, "smoothed", "Smoothed target", 5)
            plot_one_profile(ax, target_q, key, "target", "Target", 4)
            plot_one_profile(ax, new_inlet_q, key, "updated", "Updated inlet", 6)

            ax.set_xlim(*cfg["xlim"])
            ax.set_xticks(cfg["xticks"])

            ax.set_ylim(0.0, z_max_over_H)
            ax.set_yticks(np.arange(0.0, z_max_over_H + 0.001, 0.5))

            ax.set_xlabel(cfg["xlabel"], fontsize=12)
            ax.set_title(cfg["panel"], fontsize=13, pad=2)

            if key in ("U_over_UH", "uwStress_over_UH2"):
                ax.set_ylabel(r"$z/H$", fontsize=12)
            else:
                ax.set_ylabel("")

            if key == "uwStress_over_UH2":
                ax.axvline(
                    0.0,
                    color="0.35",
                    linestyle="-",
                    linewidth=0.8,
                    alpha=0.8,
                    zorder=1,
                )

            ax.tick_params(
                axis="both",
                which="both",
                direction="in",
                length=4.0,
                width=0.8,
                labelsize=10,
                colors=colours["fg"],
            )

            ax.grid(
                True,
                which="major",
                linestyle="--",
                linewidth=0.5,
                alpha=0.75,
                color=colours["grid"],
            )

            for spine in ax.spines.values():
                spine.set_linewidth(0.9)
                spine.set_color(colours["fg"])

        # Legend inside panel (a), matching the example image better.
        axes["U_over_UH"].legend(
            loc="upper left",
            frameon=True,
            facecolor="white",
            edgecolor="black",
            fontsize=10,
            handlelength=1.8,
            borderpad=0.45,
            labelspacing=0.35,
        )

        fig.suptitle(
            f"Downstream LES profile comparison — iteration {iteration}",
            fontsize=16,
            y=0.975,
            color=colours["fg"],
        )

        fig.subplots_adjust(
            left=0.065,
            right=0.985,
            bottom=0.085,
            top=0.90,
            wspace=0.30,
            hspace=0.38,
        )

        safe_makedirs(fig_dir)

        png_path = os.path.join(
            fig_dir,
            f"iteration{int(iteration):02d}_profiles_melaku_style_8panel.png",
        )

        safe_savefig(fig, png_path, dpi=dpi)
        plt.close(fig)

    finally:
        plt.rcParams.update(old_rc)

    return png_path

def profile_array_to_profile_quantities(
    z_array,
    profile_array,
    H,
    U_H_ref=None,
    stores_variances=True,
):
    """
    Convert DFSR profile array to plotting quantities.

    Expected profile_array columns:
        0: U
        1: R_11 = uu
        2: R_22 = vv
        3: R_33 = ww
        4: Lu
        5: Lv
        6: Lw
        7: R_31 = u'w' / uwStress, optional
    """

    if profile_array is None:
        return None

    arr = np.asarray(profile_array)

    q = {}
    q["z_over_H"] = np.asarray(z_array) / H

    U = arr[:, 0]

    if U_H_ref is None:
        U_H_ref = np.interp(H, z_array, U)

    q["U_over_UH"] = U / U_H_ref

    if arr.shape[1] >= 4:
        if stores_variances:
            sigma_u = np.sqrt(np.maximum(arr[:, 1], 0.0))
            sigma_v = np.sqrt(np.maximum(arr[:, 2], 0.0))
            sigma_w = np.sqrt(np.maximum(arr[:, 3], 0.0))
        else:
            sigma_u = arr[:, 1]
            sigma_v = arr[:, 2]
            sigma_w = arr[:, 3]

        q["Iu"] = sigma_u / U
        q["Iv"] = sigma_v / U
        q["Iw"] = sigma_w / U

    if arr.shape[1] >= 7:
        q["Lu_over_H"] = arr[:, 4] / H
        q["Lv_over_H"] = arr[:, 5] / H
        q["Lw_over_H"] = arr[:, 6] / H

    if arr.shape[1] >= 8:
        q["uw_over_UH2"] = arr[:, 7] / (U_H_ref ** 2)

    return q






# =============================================================================
# MAIN SCRIPT
# =============================================================================


def main():
    case_path = os.path.abspath(CASE_PATH)
    paths = get_default_paths(case_path)

    downstream_probes_folder = DOWNSTREAM_PROBES_FOLDER or paths["downstream_probes_folder"]
    experimental_profile_path = EXPERIMENTAL_PROFILE_PATH or paths["experimental_profile_path"]
    fig_root = FIG_ROOT or paths["fig_root"]

    building_height = load_building_height(case_path, BUILDING_HEIGHT)
    burn_in_time = load_burn_in_time(paths["sim_init_json"], BURN_IN_TIME, FALLBACK_BURN_IN_TIME)
    iteration = infer_iteration(case_path, ITERATION)

    print("Loading target and current inlet profiles...")
    target_profile_df, target_profile_array = load_target_profile(case_path)
    z_array = target_profile_df["z"].to_numpy(dtype=float)

    current_profile_df, current_profile_array = load_current_inlet_profile(paths["profile_path"])

    print("Loading downstream probe velocities and calculating downstream profile...")
    vel_array_3d = LES._profileAnalysis.get_velocity_components(downstream_probes_folder)
    time_steps = LES._profileAnalysis.get_time_steps_probe_data(downstream_probes_folder)
    time_step = float(np.mean(np.diff(time_steps)))

    downstream_profile_array = LES._profileCalibration.get_downstream_dfsr_profile_array(
        vel_array_3d,
        time_step,
        inlet_or_downstream="downstream",
        burn_in_time=burn_in_time,
        time_steps=time_steps,
    )

    # Ensure all arrays can carry the signed uwStress column when any profile has it.
    if target_profile_array.shape[1] >= 8 or downstream_profile_array.shape[1] >= 8:
        current_profile_array = ensure_uw_column(current_profile_array, reference_array=target_profile_array)
        target_profile_array = ensure_uw_column(target_profile_array, reference_array=current_profile_array)
        downstream_profile_array = ensure_uw_column(downstream_profile_array, reference_array=target_profile_array)

    if PLOT_UPDATED_INLET:
        new_inlet_profile_array = make_updated_profile_array(
            current_profile_array,
            target_profile_array,
            downstream_profile_array,
            relaxation_factor=UPDATED_PROFILE_RELAXATION_FACTOR,
            uw_relaxation_factor=UW_SIGNED_RELAXATION_FACTOR,
        )
    else:
        new_inlet_profile_array = None

    U_H_ref = float(np.interp(building_height, z_array, target_profile_array[:, 0]))

    print("Loading optional experimental/smoothed profile files...")
    experimental_df = read_profile_table(experimental_profile_path, optional=True)

    if SMOOTHED_TARGET_PROFILE_PATH is not None:
        smoothed_df = read_profile_table(SMOOTHED_TARGET_PROFILE_PATH, optional=True)
    elif os.path.exists(paths["smoothed_target_profile_path"]):
        smoothed_df = read_profile_table(paths["smoothed_target_profile_path"], optional=True)
    else:
        smoothed_df = target_profile_df

    experimental_q = profile_df_to_plot_quantities(experimental_df, building_height, U_H_ref=U_H_ref)
    smoothed_q = profile_df_to_plot_quantities(smoothed_df, building_height, U_H_ref=U_H_ref)
    target_q = profile_array_to_plot_quantities(z_array, target_profile_array, building_height, U_H_ref=U_H_ref)
    downstream_q = profile_array_to_plot_quantities(z_array, downstream_profile_array, building_height, U_H_ref=U_H_ref)
    new_inlet_q = profile_array_to_plot_quantities(z_array, new_inlet_profile_array, building_height, U_H_ref=U_H_ref)

    print("Creating 8-panel profile figure...")
    fig_dir = os.path.join(fig_root, "profiles")
    png_path = plot_downstream_profiles_melaku_style_8panel(
        fig_dir=fig_dir,
        iteration=iteration,
        experimental_q=experimental_q,
        smoothed_target_q=smoothed_q,
        downstream_q=downstream_q,
        new_inlet_q=new_inlet_q,
        target_q=target_q,
        z_max_over_H=Z_MAX_OVER_H,
        theme=FIGURE_THEME,
        dpi=DPI,
    )

    print(f"Saved 8-panel profile plot to:\n{png_path}")

    if WRITE_UPDATED_INLET_PROFILE and new_inlet_profile_array is not None:
        new_profile_df = array_to_profile_df(z_array, new_inlet_profile_array)
        output_profile_path = paths["profile_path"]

        backup_path = output_profile_path + ".backup_before_8panel_script"
        if os.path.exists(output_profile_path) and not os.path.exists(backup_path):
            shutil.copy2(output_profile_path, backup_path)
            print(f"Backed up existing inlet profile to:\n{backup_path}")

        write_profile_file(output_profile_path, new_profile_df)
        print(f"Wrote updated inlet profile to:\n{output_profile_path}")

    if WRITE_ITERATION_PROFILE_SNAPSHOTS and new_inlet_profile_array is not None:
        iter_dir = os.path.join(paths["downstream_calibration_log"], f"iteration{iteration}", "profileSnapshots8Panel")
        safe_makedirs(iter_dir)
        write_profile_file(os.path.join(iter_dir, "currentInletProfile"), array_to_profile_df(z_array, current_profile_array))
        write_profile_file(os.path.join(iter_dir, "targetProfile"), array_to_profile_df(z_array, target_profile_array))
        write_profile_file(os.path.join(iter_dir, "downstreamProfile"), array_to_profile_df(z_array, downstream_profile_array))
        write_profile_file(os.path.join(iter_dir, "newInletProfile"), array_to_profile_df(z_array, new_inlet_profile_array))
        print(f"Wrote profile snapshots to:\n{iter_dir}")


if __name__ == "__main__":
    main()
