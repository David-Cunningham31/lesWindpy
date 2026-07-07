#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classic native windlespy/Wong downstream calibration for the DFSR utility.

The active DFSR profile is required to have 8 columns:
    z U Iu Iv Iw Lu Lv Lw

The downstream LES statistics are still computed with the 8-column Wong/statistics
array returned by windlespy:
    U R11 R22 R33 Lu Lv Lw uwStress

Only the first seven Wong columns are calibrated for DFSR, because uwStress is not
an input to DFSRTurb.  The shear stress is nevertheless retained in all diagnostic
CSV files and in the 8-panel plot.  The actual update is performed by the native
windlespy function LES._profileCalibration.new_dfsr_profile_array(...,
relaxation_factor=0.9).
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROFILE_COLS = ["z", "U", "Iu", "Iv", "Iw", "Lu", "Lv", "Lw"]
PROFILE_COLS_UW = PROFILE_COLS + ["uwStress"]
WONG_COLS_7 = ["U", "R_11", "R_22", "R_33", "Lu", "Lv", "Lw"]
WONG_COLS_8 = WONG_COLS_7 + ["uwStress"]


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return float(default)
    return float(raw)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def import_windlespy():
    """Import the local windlespy package in the same style as the native recipes."""
    candidates: List[Path] = []
    for key in ["WINDLESPY_ROOT", "LES_ROOT"]:
        val = os.environ.get(key)
        if val:
            candidates.append(Path(val).expanduser().resolve())

    here = Path(__file__).resolve()
    # If script is in .../LES/windlespy/_recipes, parents[2] is .../LES.
    for parent in list(here.parents)[:5]:
        candidates.append(parent)
        if (parent / "LES").is_dir():
            candidates.append(parent / "LES")

    candidates.append(Path("/home/people/20397873/LES"))

    tried = []
    for root in candidates:
        tried.append(str(root))
        if (root / "windlespy").is_dir():
            sys.path.insert(0, str(root))
            try:
                import windlespy as LES  # type: ignore
                return LES
            finally:
                try:
                    sys.path.remove(str(root))
                except ValueError:
                    pass

    # Fall back to any importable installation.
    try:
        import windlespy as LES  # type: ignore
        return LES
    except Exception as exc:
        raise ImportError(
            "Could not import windlespy. Set WINDLESPY_ROOT to the folder containing the windlespy package. "
            f"Tried: {tried}"
        ) from exc


LES = import_windlespy()


def safe_float(text: str, default: float = 0.0) -> float:
    try:
        return float(text)
    except Exception:
        return default


def is_number_token(token: str) -> bool:
    try:
        float(token)
        return True
    except Exception:
        return False


def read_profile_file(path: Path, allow_8: bool = True, allow_9: bool = True, allow_15: bool = True) -> pd.DataFrame:
    """Read profile-like files with 8, 9, or 15 columns.

    The returned dataframe always contains at least z, U, Iu, Iv, Iw, Lu, Lv, Lw,
    and contains uwStress if available.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    lines = [ln for ln in path.read_text(errors="ignore").splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"Profile file is empty: {path}")

    first = lines[0].split()
    has_header = first and not is_number_token(first[0])

    if has_header:
        df = pd.read_csv(path, sep=r"\s+", comment="#", engine="python")
        rename = {
            "height": "z", "Z": "z", "u": "U", "Uav": "U",
            "I_U": "Iu", "I_V": "Iv", "I_W": "Iw",
            "L_u": "Lu", "L_v": "Lv", "L_w": "Lw",
            "uw": "uwStress", "u'w'": "uwStress", "R13": "uwStress", "R_13": "uwStress",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        keep = [c for c in PROFILE_COLS_UW if c in df.columns]
        df = df.loc[:, keep]
    else:
        arr = np.loadtxt(path)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        ncols = arr.shape[1]
        if ncols == 8 and allow_8:
            cols = PROFILE_COLS
        elif ncols == 9 and allow_9:
            cols = PROFILE_COLS_UW
        elif ncols == 15 and allow_15:
            cols = [
                "z", "U", "Iu", "Iv", "Iw", "Lu", "Lv", "Lw",
                "uu", "vv", "ww", "uv", "uw", "vw", "uwStress",
            ]
        else:
            raise ValueError(
                f"Unsupported number of columns in {path}: {ncols}. "
                "Expected 8 DFSR columns or 9 MannHybrid columns."
            )
        df = pd.DataFrame(arr, columns=cols)
        df = df[[c for c in PROFILE_COLS_UW if c in df.columns]]

    missing = [c for c in PROFILE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Profile file {path} is missing columns: {missing}")

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if not np.isfinite(df[[c for c in df.columns if c != "uwStress"]].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite values in required profile columns: {path}")

    # Convert obvious percentage intensities to decimal.
    for c in ["Iu", "Iv", "Iw"]:
        if df[c].max() > 2.0:
            print(f"WARNING: {path}:{c} looks like percent, dividing by 100", flush=True)
            df[c] = df[c] / 100.0

    df = df.sort_values("z").drop_duplicates("z", keep="last").reset_index(drop=True)
    return df


def profile_to_wong_array(df: pd.DataFrame, include_uw: bool) -> np.ndarray:
    out = pd.DataFrame()
    out["U"] = df["U"].to_numpy(dtype=float)
    out["R_11"] = (df["Iu"].to_numpy(dtype=float) * out["U"].to_numpy(dtype=float)) ** 2
    out["R_22"] = (df["Iv"].to_numpy(dtype=float) * out["U"].to_numpy(dtype=float)) ** 2
    out["R_33"] = (df["Iw"].to_numpy(dtype=float) * out["U"].to_numpy(dtype=float)) ** 2
    out["Lu"] = df["Lu"].to_numpy(dtype=float)
    out["Lv"] = df["Lv"].to_numpy(dtype=float)
    out["Lw"] = df["Lw"].to_numpy(dtype=float)
    if include_uw:
        if "uwStress" not in df.columns:
            raise ValueError("include_uw=True but dataframe has no uwStress column")
        out["uwStress"] = df["uwStress"].to_numpy(dtype=float)
    return out.to_numpy(dtype=float)


def wong_array_to_profile_df(arr: np.ndarray, z: np.ndarray, include_uw_col: bool = True) -> pd.DataFrame:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] not in (7, 8):
        raise ValueError(f"Expected Wong array with 7 or 8 columns; got shape {arr.shape}")
    U = arr[:, 0]
    df = pd.DataFrame({
        "z": np.asarray(z, dtype=float),
        "U": U,
        "Iu": np.sqrt(np.maximum(arr[:, 1], 0.0)) / np.maximum(np.abs(U), 1e-12),
        "Iv": np.sqrt(np.maximum(arr[:, 2], 0.0)) / np.maximum(np.abs(U), 1e-12),
        "Iw": np.sqrt(np.maximum(arr[:, 3], 0.0)) / np.maximum(np.abs(U), 1e-12),
        "Lu": arr[:, 4],
        "Lv": arr[:, 5],
        "Lw": arr[:, 6],
    })
    if include_uw_col:
        if arr.shape[1] >= 8:
            df["uwStress"] = arr[:, 7]
        else:
            df["uwStress"] = np.nan
    return df


def interp_profile_to_z(df: pd.DataFrame, z_new: np.ndarray) -> pd.DataFrame:
    z_old = df["z"].to_numpy(dtype=float)
    out = pd.DataFrame({"z": z_new})
    for c in df.columns:
        if c == "z":
            continue
        vals = df[c].to_numpy(dtype=float)
        finite = np.isfinite(vals)
        if finite.sum() < 2:
            out[c] = np.nan
        else:
            out[c] = np.interp(z_new, z_old[finite], vals[finite])
    return out


def merge_target_uw_if_available(target_display: pd.DataFrame, profile_dir: Path, case_dir: Path) -> pd.DataFrame:
    """For DFSR, active targetProfile is normally 8 columns. This function looks for
    a 9-column diagnostic target profile that contains uwStress for plotting/reporting.
    """
    if "uwStress" in target_display.columns and np.isfinite(target_display["uwStress"]).any():
        return target_display

    explicit = os.environ.get("TARGET_STATS_PROFILE") or os.environ.get("DFSR_TARGET_STATS_PROFILE")
    candidates: List[Path] = []
    if explicit:
        p = Path(explicit).expanduser()
        candidates.append(p if p.is_absolute() else case_dir / p)
        candidates.append(profile_dir / explicit)

    for name in [
        "targetProfileStats",
        "targetProfile_withUW",
        "targetProfile_with_uw",
        "targetProfile9",
        "targetProfile_9col",
        "targetExperimentalProfile",
        "targetExperimentalProfile_withStats",
        "targetSmoothedProfile",
    ]:
        candidates.append(profile_dir / name)

    for p in candidates:
        try:
            if p.exists():
                df = read_profile_file(p, allow_8=True, allow_9=True, allow_15=True)
                if "uwStress" in df.columns and np.isfinite(df["uwStress"].to_numpy(dtype=float)).any():
                    out = target_display.copy()
                    uw = interp_profile_to_z(df[["z", "uwStress"]], target_display["z"].to_numpy(dtype=float))
                    out["uwStress"] = uw["uwStress"].to_numpy(dtype=float)
                    print(f"Using target uwStress for diagnostics from: {p}", flush=True)
                    return out
        except Exception as exc:
            print(f"WARNING: could not read target stats candidate {p}: {exc}", flush=True)

    out = target_display.copy()
    out["uwStress"] = np.nan
    print("WARNING: no target uwStress file found; DFSR shear-stress target curve will be omitted.", flush=True)
    return out


def write_active_profile(df: pd.DataFrame, path: Path, include_uw: bool) -> None:
    cols = PROFILE_COLS_UW if include_uw else PROFILE_COLS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot write active profile {path}; missing columns {missing}")
    ensure_dir(path.parent)
    df.loc[:, cols].to_csv(path, sep="\t", header=False, index=False, float_format="%.12e")


def write_diag_profile(df: pd.DataFrame, path: Path, header: bool = False) -> None:
    ensure_dir(path.parent)
    out = df.copy()
    for c in PROFILE_COLS_UW:
        if c not in out.columns:
            out[c] = np.nan
    out.loc[:, PROFILE_COLS_UW].to_csv(path, sep="\t", header=header, index=False, float_format="%.12e")


def write_diag_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    out = df.copy()
    for c in PROFILE_COLS_UW:
        if c not in out.columns:
            out[c] = np.nan
    out.loc[:, PROFILE_COLS_UW].to_csv(path, index=False, float_format="%.12e")


def get_iteration_number(log_root: Path) -> int:
    ensure_dir(log_root)
    nums = []
    for p in log_root.glob("iteration*"):
        if p.is_dir():
            m = re.match(r"iteration(\d+)$", p.name)
            if m:
                nums.append(int(m.group(1)))
    return max(nums) + 1 if nums else 1


def get_iter_status(case_path: Path, rmse_array: np.ndarray, rmse_threshold: float, names: Sequence[str]) -> Dict[str, object]:
    log_root = case_path / "log" / "downstreamCalibration"
    ensure_dir(log_root)
    previous_nums = []
    for d in log_root.glob("iteration*"):
        if d.is_dir():
            m = re.match(r"iteration(\d+)$", d.name)
            if m:
                previous_nums.append(int(m.group(1)))
    if previous_nums:
        last = max(previous_nums)
        iteration = last + 1
        prev_json = log_root / f"iteration{last}" / f"iteration{last}.json"
        improvement_ratio = None
        stagnated = False
        if prev_json.exists():
            try:
                prev = json.loads(prev_json.read_text())
                prev_worst = prev.get("worst_rmse")
                worst = float(np.nanmax(rmse_array))
                if prev_worst not in (None, 0):
                    improvement_ratio = worst / float(prev_worst)
                    stagnated = bool(0.98 <= improvement_ratio <= 1.02)
            except Exception:
                improvement_ratio = None
                stagnated = False
    else:
        iteration = 1
        improvement_ratio = None
        stagnated = False

    worst_rmse = float(np.nanmax(rmse_array))
    converged = bool(worst_rmse <= rmse_threshold)
    status: Dict[str, object] = {
        "iteration": int(iteration),
        "converged": converged,
        "stagnated": stagnated,
        "worst_rmse": worst_rmse,
        "improvement_ratio": None if improvement_ratio is None else float(improvement_ratio),
    }
    for name, val in zip(names, rmse_array):
        status[f"{name}_rmse"] = float(val)
    return status


def rmse_absolute(down: np.ndarray, target: np.ndarray, i0: int, i1: int) -> np.ndarray:
    return np.sqrt(np.mean((down[i0:i1 + 1, :] - target[i0:i1 + 1, :]) ** 2, axis=0))


def smooth_wong_array(new_arr: np.ndarray, z_array: np.ndarray, building_height: float, include_uw: bool) -> np.ndarray:
    if not env_bool("CLASSIC_WONG_SMOOTH", True):
        return new_arr
    try:
        if new_arr.shape[1] == 7:
            return LES._profileAnalysis.smooth_profiles(new_arr, z_array, 3, 3, building_height)
        # Preserve native smoothing on the classic 7 Wong columns. Smooth uwStress lightly with a 3-point moving average.
        smooth7 = LES._profileAnalysis.smooth_profiles(new_arr[:, :7], z_array, 3, 3, building_height)
        uw = new_arr[:, 7].copy()
        if uw.size >= 3:
            uw_pad = np.pad(uw, 1, mode="edge")
            uw = np.convolve(uw_pad, np.ones(3) / 3.0, mode="valid")
        return np.column_stack([smooth7, uw])
    except Exception as exc:
        print(f"WARNING: profile smoothing failed; using unsmoothed Wong update. Error: {exc}", flush=True)
        return new_arr


def interpolate_at(z: np.ndarray, y: np.ndarray, z0: float) -> float:
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(z) & np.isfinite(y)
    if finite.sum() < 2:
        return float(np.nanmax(y))
    return float(np.interp(float(z0), z[finite], y[finite]))


def plot_melaku_8panel(
    target: pd.DataFrame,
    downstream: pd.DataFrame,
    current: pd.DataFrame,
    updated: Optional[pd.DataFrame],
    building_height: float,
    output_path: Path,
    title: str,
) -> None:
    H = float(building_height)
    UH = interpolate_at(target["z"].to_numpy(), target["U"].to_numpy(), H)
    if not np.isfinite(UH) or abs(UH) < 1e-12:
        UH = float(np.nanmax(target["U"].to_numpy(dtype=float)))

    panels = [
        ("U", r"$U/U_H$"),
        ("Iu", r"$I_u$"),
        ("Iv", r"$I_v$"),
        ("Iw", r"$I_w$"),
        ("uwStress", r"$u'w'/U_H^2$"),
        ("Lu", r"$L_u/H$"),
        ("Lv", r"$L_v/H$"),
        ("Lw", r"$L_w/H$"),
    ]
    curves = [(target, "Target", "k", "--", 1.7), (downstream, "Downstream", "tab:red", "-", 1.9), (current, "Current inlet", "0.45", ":", 1.5)]
    if updated is not None:
        curves.append((updated, "Updated inlet", "tab:blue", "-.", 1.8))

    fig, axes = plt.subplots(2, 4, figsize=(17, 9), sharey=True)
    axes = axes.ravel()
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

    for ax, lab, (col, xlabel) in zip(axes, labels, panels):
        for df, desc, color, ls, lw in curves:
            if col not in df.columns:
                continue
            z = df["z"].to_numpy(dtype=float) / H
            vals = df[col].to_numpy(dtype=float)
            if col == "U":
                x = vals / UH
            elif col == "uwStress":
                x = vals / max(UH * UH, 1e-12)
            elif col in ["Lu", "Lv", "Lw"]:
                x = vals / H
            else:
                x = vals
            finite = np.isfinite(x) & np.isfinite(z)
            if finite.any():
                ax.plot(x[finite], z[finite], color=color, linestyle=ls, lw=lw, label=desc)
        if col == "uwStress":
            ax.axvline(0.0, color="0.3", lw=0.8)
        ax.set_title(lab)
        ax.set_xlabel(xlabel)
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[0].set_ylabel(r"$z/H$")
    axes[4].set_ylabel(r"$z/H$")
    zmax = float(np.nanmax(target["z"].to_numpy(dtype=float) / H))
    for ax in axes:
        ax.set_ylim(0.0, min(max(3.0, zmax * 1.02), zmax * 1.05 if zmax > 3.0 else 3.0))
    axes[0].legend(fontsize=9, loc="best")
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def read_config(case_path: Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    try:
        variable_dict = LES._caseFiles.parse_setup_file(str(case_path))
    except Exception as exc:
        print(f"WARNING: windlespy parse_setup_file failed: {exc}; using empty setup dict.", flush=True)
        variable_dict = {}
    json_path = case_path / "log" / "downstreamCalibration" / "sim_init.json"
    sim_init: Dict[str, float] = {}
    if json_path.exists():
        with json_path.open("r") as f:
            sim_init = json.load(f)
    return variable_dict, sim_init


def get_threshold_indices(target_profile_df: pd.DataFrame, lower_z: float, upper_z: float) -> Tuple[int, int]:
    try:
        return LES._profileCalibration.get_avg_z_thresolds_ids(target_profile_df, lower_z, upper_z)
    except Exception:
        z = target_profile_df["z"].to_numpy(dtype=float)
        i0 = int(np.argmin(np.abs(z - lower_z)))
        i1 = int(np.argmin(np.abs(z - upper_z)))
        return min(i0, i1), max(i0, i1)


MODE = "DFSR"
CALIBRATES_UW = False
ACTIVE_PROFILE_HAS_UW = False
SCRIPT_LABEL = "DFSR classic Wong downstream calibration"


def run() -> int:
    case_path = Path(os.environ.get("CASE_DIR", os.getcwd())).resolve()
    profile_dir = case_path / os.environ.get("PROFILE_DIR", "constant/boundaryData/windProfile")
    probes_name = os.environ.get("PROBES_NAME", "probes2")
    downstream_probes_folder = case_path / "postProcessing" / probes_name

    variable_dict, sim_init = read_config(case_path)
    building_height = env_float("BUILDING_HEIGHT", float(variable_dict.get("buildingHeight", 1.0)))
    lower_z_threshold = env_float("LOWER_Z_THRESHOLD", float(variable_dict.get("lowerZThreshold", -1e30)))
    upper_z_threshold = env_float("UPPER_Z_THRESHOLD", float(variable_dict.get("upperZThreshold", 1e30)))
    rmse_threshold = env_float("RMSE_THRESHOLD", float(variable_dict.get("rmseThreshold", 0.05)))
    burn_in_time = env_float("BURN_IN_TIME", float(sim_init.get("burn_in_time", 0.0)))
    relaxation_factor = env_float("WONG_RELAXATION_FACTOR", 0.9)

    print(f"=== {SCRIPT_LABEL} ===", flush=True)
    print(f"case_path = {case_path}", flush=True)
    print(f"profile_dir = {profile_dir}", flush=True)
    print(f"downstream_probes_folder = {downstream_probes_folder}", flush=True)
    print(f"building_height = {building_height}", flush=True)
    print(f"height range = {lower_z_threshold} to {upper_z_threshold}", flush=True)
    print(f"rmse_threshold = {rmse_threshold}", flush=True)
    print(f"burn_in_time = {burn_in_time}", flush=True)
    print(f"Wong relaxation_factor = {relaxation_factor}", flush=True)
    print(f"calibrates uwStress = {CALIBRATES_UW}", flush=True)

    target_profile_df = LES._profileCalibration.get_dfsr_target_profile_df(str(case_path))
    z_array = target_profile_df["z"].to_numpy(dtype=float)

    active_current_path = profile_dir / "profile"
    active_target_path = profile_dir / "targetProfile"
    active_current_profile = read_profile_file(active_current_path, allow_8=True, allow_9=True)
    active_target_profile = read_profile_file(active_target_path, allow_8=True, allow_9=True)

    if MODE == "DFSR":
        if active_current_profile.shape[1] != 8 or active_target_profile.shape[1] != 8:
            raise RuntimeError(
                "DFSR classic calibration expects active profile and targetProfile to have exactly 8 columns: "
                "z U Iu Iv Iw Lu Lv Lw. Do not include uwStress in the active DFSR input profile."
            )
    elif MODE == "MANNHYBRID":
        if "uwStress" not in active_current_profile.columns or "uwStress" not in active_target_profile.columns:
            raise RuntimeError(
                "MannHybrid classic calibration expects active profile and targetProfile to have 9 columns, "
                "including uwStress."
            )

    target_update_array = LES._profileCalibration.get_dfsr_target_profile_array(str(case_path))
    current_update_array = LES._profileCalibration.get_current_dfsr_inlet_profile_array(str(case_path))

    expected_update_cols = 8 if CALIBRATES_UW else 7
    if target_update_array.shape[1] != expected_update_cols:
        raise RuntimeError(f"target_update_array should have {expected_update_cols} columns for {MODE}, got {target_update_array.shape}")
    if current_update_array.shape[1] != expected_update_cols:
        raise RuntimeError(f"current_update_array should have {expected_update_cols} columns for {MODE}, got {current_update_array.shape}")

    vel_array_3d = LES._profileAnalysis.get_velocity_components(str(downstream_probes_folder))
    time_steps = LES._profileAnalysis.get_time_steps_probe_data(str(downstream_probes_folder))
    time_step = float(np.mean(np.diff(time_steps)))
    downstream_full_array = LES._profileCalibration.get_downstream_dfsr_profile_array(
        vel_array_3d,
        time_step,
        inlet_or_downstream="downstream",
        burn_in_time=burn_in_time,
        time_steps=time_steps,
    )

    if downstream_full_array.shape[1] != 8:
        raise RuntimeError(
            "windlespy downstream profile should return 8 columns: "
            "U R11 R22 R33 Lu Lv Lw uwStress. Got " + str(downstream_full_array.shape)
        )
    if downstream_full_array.shape[0] != target_update_array.shape[0]:
        raise RuntimeError(
            "Downstream profile height count does not match target profile height count. "
            f"downstream={downstream_full_array.shape}, target={target_update_array.shape}"
        )

    downstream_update_array = downstream_full_array if CALIBRATES_UW else downstream_full_array[:, :7]

    lower_id, upper_id = get_threshold_indices(target_profile_df, lower_z_threshold, upper_z_threshold)
    rmse_names = WONG_COLS_8 if CALIBRATES_UW else WONG_COLS_7
    rmse_array = rmse_absolute(downstream_update_array, target_update_array, lower_id, upper_id)
    iter_status = get_iter_status(case_path, rmse_array, rmse_threshold, rmse_names)

    iteration = int(iter_status["iteration"])
    iteration_path = case_path / "log" / "downstreamCalibration" / f"iteration{iteration}"
    data_dir = ensure_dir(iteration_path / "data")
    plots_dir = ensure_dir(iteration_path / "plots")

    converged = bool(iter_status["converged"])
    stagnated = bool(iter_status["stagnated"])

    print("Validation checks:", flush=True)
    print(f"  target_update_array shape      = {target_update_array.shape}", flush=True)
    print(f"  current_update_array shape     = {current_update_array.shape}", flush=True)
    print(f"  downstream_full_array shape    = {downstream_full_array.shape}", flush=True)
    print(f"  downstream_update_array shape  = {downstream_update_array.shape}", flush=True)
    print(f"  update uses native windlespy LES._profileCalibration.new_dfsr_profile_array", flush=True)

    # Profile dataframes for identical diagnostics.
    target_display = wong_array_to_profile_df(target_update_array, z_array, include_uw_col=True)
    if not CALIBRATES_UW:
        target_display = merge_target_uw_if_available(target_display, profile_dir, case_path)

    current_display = wong_array_to_profile_df(current_update_array, z_array, include_uw_col=True)
    if not CALIBRATES_UW:
        # DFSR has no active uwStress input.  Keep the diagnostic column present
        # and explicitly show the controllable inlet shear stress as zero.
        current_display["uwStress"] = 0.0
    downstream_display = wong_array_to_profile_df(downstream_full_array, z_array, include_uw_col=True)

    updated_display = None
    new_update_array = None
    if (not converged) and (not stagnated):
        new_update_array = LES._profileCalibration.new_dfsr_profile_array(
            current_update_array,
            target_update_array,
            downstream_update_array,
            relaxation_factor=relaxation_factor,
        )
        new_update_array = smooth_wong_array(new_update_array, z_array, building_height, include_uw=CALIBRATES_UW)
        updated_display = wong_array_to_profile_df(new_update_array, z_array, include_uw_col=True)
        if not CALIBRATES_UW:
            updated_display["uwStress"] = 0.0
    else:
        updated_display = current_display.copy()

    # Save iteration status in the same location as the classic script.
    ensure_dir(iteration_path)
    with (iteration_path / f"iteration{iteration}.json").open("w") as f:
        json.dump(iter_status, f, indent=2)

    # Identical diagnostic profile formats for both DFSR and MannHybrid.
    write_diag_csv(target_display, data_dir / "target_profile.csv")
    write_diag_csv(current_display, data_dir / "current_profile.csv")
    write_diag_csv(downstream_display, data_dir / "downstream_profile.csv")
    if updated_display is not None:
        write_diag_csv(updated_display, data_dir / "updated_profile.csv")

    # Legacy-style headerless snapshots, but with an identical 9-column diagnostic layout.
    write_diag_profile(current_display, iteration_path / "inletProfile", header=False)
    write_diag_profile(downstream_display, iteration_path / "postCorrectionProfile", header=False)
    if updated_display is not None:
        write_diag_profile(updated_display, iteration_path / "newInletProfile", header=False)

    rmse_df = pd.DataFrame({"quantity": list(rmse_names), "rmse": rmse_array})
    rmse_df.to_csv(data_dir / "rmse.csv", index=False, float_format="%.12e")

    validation = {
        "mode": MODE,
        "active_profile_has_uwStress": bool(ACTIVE_PROFILE_HAS_UW),
        "calibrates_uwStress": bool(CALIBRATES_UW),
        "target_update_array_shape": list(target_update_array.shape),
        "current_update_array_shape": list(current_update_array.shape),
        "downstream_full_array_shape": list(downstream_full_array.shape),
        "downstream_update_array_shape": list(downstream_update_array.shape),
        "wong_update_function": "LES._profileCalibration.new_dfsr_profile_array",
        "wong_relaxation_factor": relaxation_factor,
        "active_profile_written_columns": PROFILE_COLS_UW if ACTIVE_PROFILE_HAS_UW else PROFILE_COLS,
        "diagnostic_profile_written_columns": PROFILE_COLS_UW,
    }
    with (data_dir / "validation_checks.json").open("w") as f:
        json.dump(validation, f, indent=2)

    plot_melaku_8panel(
        target=target_display,
        downstream=downstream_display,
        current=current_display,
        updated=updated_display,
        building_height=building_height,
        output_path=plots_dir / f"iteration{iteration}_melaku_8panel.png",
        title=SCRIPT_LABEL,
    )

    print("RMSE summary:", flush=True)
    print(rmse_df.to_string(index=False), flush=True)
    print(f"Converged={converged}, stagnated={stagnated}", flush=True)
    print(f"Iteration output written to: {iteration_path}", flush=True)

    if converged or stagnated:
        print("No active profile update written.", flush=True)
        return 0

    # Write active profile with the correct utility-specific column count.
    backup = profile_dir / f"profile_before_iteration{iteration}"
    if active_current_path.exists() and not backup.exists():
        shutil.copy(active_current_path, backup)

    if MODE == "DFSR":
        # DFSR active input remains 8 columns: z U Iu Iv Iw Lu Lv Lw.
        active_updated = wong_array_to_profile_df(new_update_array, z_array, include_uw_col=False)
        write_active_profile(active_updated, active_current_path, include_uw=False)
        print(f"Updated DFSR active 8-column profile written to: {active_current_path}", flush=True)
    else:
        # MannHybrid active input remains 9 columns: z U Iu Iv Iw Lu Lv Lw uwStress.
        active_updated = wong_array_to_profile_df(new_update_array, z_array, include_uw_col=True)
        write_active_profile(active_updated, active_current_path, include_uw=True)
        print(f"Updated MannHybrid active 9-column profile written to: {active_current_path}", flush=True)

    return 1


def main() -> None:
    try:
        code = run()
    except Exception as exc:
        print(f"ERROR in {SCRIPT_LABEL}:", file=sys.stderr)
        print(f"  {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
