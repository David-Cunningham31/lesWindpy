#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MannHybridWongCalibration.py

Profile-level Wong-style downstream calibration for MannHybridTurb.

This recipe intentionally does NOT perform frequency-by-frequency spectral
calibration and it does NOT require spectraProfile / uwCoSpectrumProfile files.
It is intended for MannHybridTurb dictionaries using, for example:

    targetSpectraSource       vonKarman;
    uwCoSpectrumSource        kaimal;
    uwStressSource            profile;

Each calibration iteration:
  1. Reads current inlet profile and fixed targetProfile.
  2. Reads all OpenFOAM probe U files under postProcessing/probes2/*/U.
  3. Concatenates probe segments, filters by burn_in_time, removes duplicate times.
  4. Computes downstream U, Iu/Iv/Iw, Lu/Lv/Lw, and <u'w'> directly from time histories.
  5. Applies a damped Wong-style profile update, including a signed update for <u'w'>.
  6. Writes the updated active profile only.
  7. Saves iteration snapshots and a Melaku-style 8-panel profile plot.

Exit code convention:
    0  converged
    1  update written; rerun MannHybridTurb + LES
    2  runtime/input error

Environment variables commonly used:
    CASE_DIR                         case directory, default cwd
    MHW_PROBES_NAME                  probes folder name, default probes2
    MHW_PROFILE_DIR                  windProfile relative directory,
                                     default constant/boundaryData/windProfile
    MHW_CALIBRATE_UW                 true/false, default true when target has uwStress
    MHW_RELAX_U                      default 0.20
    MHW_RELAX_I                      default 0.35
    MHW_RELAX_L                      default 0.25
    MHW_RELAX_UW                     default 0.20
    MHW_RHO_UW_LIMIT                 default 0.999
    MHW_MIN_FACTOR_U                 default 0.90
    MHW_MAX_FACTOR_U                 default 1.10
    MHW_MIN_FACTOR_I                 default 0.60
    MHW_MAX_FACTOR_I                 default 1.80
    MHW_MIN_FACTOR_L                 default 0.70
    MHW_MAX_FACTOR_L                 default 1.50
    MHW_MIN_FACTOR_UW                default 0.70
    MHW_MAX_FACTOR_UW                default 1.40
    MHW_SMOOTH_WINDOW                odd integer; 0 disables, default 0
    MHW_UPDATE_TARGET_HEIGHT_RANGE   true/false, default true
    MHW_L_METHOD                     firstZero or expFit, default firstZero
    MHW_MAX_LAG_FRACTION             default 0.5
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Use a non-interactive backend on clusters.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROFILE_COLS = ["z", "U", "Iu", "Iv", "Iw", "Lu", "Lv", "Lw"]
PROFILE_COLS_UW = PROFILE_COLS + ["uwStress"]


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return float(default)
    return float(raw)


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return int(default)
    return int(raw)


def trapz(y, x=None, dx=1.0, axis=-1):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x, dx=dx, axis=axis)
    return np.trapz(y, x=x, dx=dx, axis=axis)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_number_token(token: str) -> bool:
    try:
        float(token)
        return True
    except Exception:
        return False


def sanitise_numeric_df(df: pd.DataFrame, context: str) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    bad = ~np.isfinite(out.to_numpy(dtype=float))
    if bad.any():
        rows = np.unique(np.argwhere(bad)[:, 0])[:10]
        raise ValueError(f"Non-finite numeric values in {context}; example rows {rows.tolist()}")
    return out


# -----------------------------------------------------------------------------
# OpenFOAM setup/dictionary parsing
# -----------------------------------------------------------------------------


def parse_setup_file(case_dir: Path) -> Dict[str, float]:
    """Parse simple OpenFOAM-like setUp files with entries such as `deltaT 0.001;`."""
    candidates = [case_dir / "setUp", case_dir / "setup", case_dir / "system" / "setUp"]
    data: Dict[str, float] = {}
    pattern = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:=)?\s*([-+0-9.eE]+)\s*;?.*$")
    for p in candidates:
        if not p.exists():
            continue
        for line in p.read_text(errors="ignore").splitlines():
            line = line.split("//", 1)[0].strip()
            if not line:
                continue
            m = pattern.match(line)
            if m:
                key, val = m.group(1), m.group(2)
                try:
                    data[key] = float(val)
                except ValueError:
                    pass
    return data


def read_sim_init(case_dir: Path) -> Dict[str, float]:
    p = case_dir / "log" / "downstreamCalibration" / "sim_init.json"
    if not p.exists():
        return {}
    with p.open("r") as f:
        return json.load(f)


def dictionary_value(text: str, key: str) -> Optional[str]:
    m = re.search(rf"\b{re.escape(key)}\s+([^;]+);", text)
    if not m:
        return None
    return m.group(1).strip().strip('"')


def warn_if_mann_dict_inconsistent(case_dir: Path) -> None:
    """Warn if MannHybridTurbDict still expects tabulated spectra/co-spectra."""
    candidates = [case_dir / "constant" / "MannHybridTurbDict", case_dir / "system" / "MannHybridTurbDict"]
    for p in candidates:
        if not p.exists():
            continue
        txt = p.read_text(errors="ignore")
        target_src = dictionary_value(txt, "targetSpectraSource")
        uw_src = dictionary_value(txt, "uwCoSpectrumSource")
        uw_stress_src = dictionary_value(txt, "uwStressSource")
        if target_src and target_src != "vonKarman":
            print(
                f"WARNING: {p} has targetSpectraSource={target_src!r}. "
                "This Wong-profile calibration script writes only `profile`; "
                "use `targetSpectraSource vonKarman;` if you do not want spectraProfile input.",
                flush=True,
            )
        if uw_src and uw_src != "kaimal":
            print(
                f"WARNING: {p} has uwCoSpectrumSource={uw_src!r}. "
                "Use `uwCoSpectrumSource kaimal;` if you want Kaimal co-spectrum from profile uwStress.",
                flush=True,
            )
        if uw_stress_src and uw_stress_src != "profile":
            print(
                f"WARNING: {p} has uwStressSource={uw_stress_src!r}. "
                "Use `uwStressSource profile;` if the calibrated profile contains uwStress.",
                flush=True,
            )
        return


# -----------------------------------------------------------------------------
# Profile IO
# -----------------------------------------------------------------------------


def read_profile_file(path: Path, allow_missing_uw: bool = True) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")

    lines = [ln for ln in path.read_text(errors="ignore").splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"Profile file is empty: {path}")

    first = lines[0].split()
    has_header = not is_number_token(first[0])

    if has_header:
        df = pd.read_csv(path, sep=r"\s+", comment="#", engine="python")
        # Tolerate alternative naming.
        rename = {
            "height": "z", "Z": "z",
            "u": "U", "Uav": "U",
            "I_U": "Iu", "I_V": "Iv", "I_W": "Iw",
            "L_u": "Lu", "L_v": "Lv", "L_w": "Lw",
            "uw": "uwStress", "uv": "uwStress", "u'w'": "uwStress",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        keep = [c for c in PROFILE_COLS_UW if c in df.columns]
        df = df.loc[:, keep]
    else:
        data = np.loadtxt(path)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        ncols = data.shape[1]
        if ncols == 8:
            cols = PROFILE_COLS
        elif ncols == 9:
            cols = PROFILE_COLS_UW
        else:
            raise ValueError(
                f"Profile file {path} must have 8 or 9 columns; found {ncols}. "
                "Expected z U Iu Iv Iw Lu Lv Lw [uwStress]."
            )
        df = pd.DataFrame(data, columns=cols)

    missing = [c for c in PROFILE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Profile {path} is missing required columns: {missing}")
    if "uwStress" not in df.columns and not allow_missing_uw:
        raise ValueError(f"Profile {path} has no uwStress column")

    df = sanitise_numeric_df(df, f"profile {path}")

    # If an experimental table accidentally stores turbulence intensity in percent,
    # convert it to decimal. Do this only for clearly percentage-like values.
    for c in ["Iu", "Iv", "Iw"]:
        if df[c].max() > 2.0:
            print(f"WARNING: {path.name}:{c} appears to be in percent; dividing by 100.", flush=True)
            df[c] = df[c] / 100.0

    df = df.sort_values("z").drop_duplicates("z", keep="last").reset_index(drop=True)
    return df


def write_profile_file(df: pd.DataFrame, path: Path, include_uw: bool = True, header: bool = False) -> None:
    cols = PROFILE_COLS_UW if include_uw and "uwStress" in df.columns else PROFILE_COLS
    out = df.loc[:, cols].copy()
    ensure_dir(path.parent)
    out.to_csv(path, sep="\t", header=header, index=False, float_format="%.12e")


def write_profile_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, float_format="%.12e")


def interp_profile_to_z(df: pd.DataFrame, z_new: np.ndarray) -> pd.DataFrame:
    z_old = df["z"].to_numpy(dtype=float)
    out = pd.DataFrame({"z": z_new})
    for c in df.columns:
        if c == "z":
            continue
        out[c] = np.interp(z_new, z_old, df[c].to_numpy(dtype=float))
    return out


# -----------------------------------------------------------------------------
# Probe reader
# -----------------------------------------------------------------------------


VEC_RE = re.compile(
    r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)"
)
PROBE_RE = re.compile(
    r"^\s*#\s*Probe\s+(\d+)\s+\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)"
)


def parse_probe_file(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Return times, velocity array (3,nTimes,nProbes), positions if in header."""
    times: List[float] = []
    rows: List[np.ndarray] = []
    positions: Dict[int, Tuple[float, float, float]] = {}

    with path.open("r", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
                m = PROBE_RE.match(line)
                if m:
                    positions[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
                continue
            parts = line.split(maxsplit=1)
            if not parts or not is_number_token(parts[0]):
                continue
            t = float(parts[0])
            triples = VEC_RE.findall(line)
            if not triples:
                continue
            vals = np.asarray([[float(a), float(b), float(c)] for a, b, c in triples], dtype=float)
            times.append(t)
            rows.append(vals)

    if not rows:
        raise ValueError(f"No vector rows found in probe file: {path}")

    nprobes = rows[0].shape[0]
    for i, arr in enumerate(rows):
        if arr.shape[0] != nprobes:
            raise ValueError(f"Inconsistent probe count in {path} at data row {i}: {arr.shape[0]} vs {nprobes}")

    data = np.stack(rows, axis=0)  # nTimes,nProbes,3
    vel = np.transpose(data, (2, 0, 1))  # 3,nTimes,nProbes

    pos_arr = None
    if positions:
        pos_arr = np.full((nprobes, 3), np.nan, dtype=float)
        for i, xyz in positions.items():
            if 0 <= i < nprobes:
                pos_arr[i, :] = xyz
        if not np.isfinite(pos_arr).all():
            pos_arr = None

    return np.asarray(times, dtype=float), vel, pos_arr


def read_all_probe_segments(case_dir: Path, probes_name: str = "probes2") -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Path]]:
    root = case_dir / "postProcessing" / probes_name
    if not root.exists():
        raise FileNotFoundError(f"Probe directory not found: {root}")

    files = sorted(root.glob("*/U"), key=lambda p: (safe_float(p.parent.name, 0.0), str(p)))
    if (root / "U").exists():
        files.insert(0, root / "U")
    if not files:
        raise FileNotFoundError(f"No U probe files found under {root}")

    all_t: List[np.ndarray] = []
    all_v: List[np.ndarray] = []
    positions = None
    nprobes = None
    for p in files:
        t, v, pos = parse_probe_file(p)
        if nprobes is None:
            nprobes = v.shape[2]
        elif v.shape[2] != nprobes:
            raise ValueError(f"Probe count mismatch in {p}: {v.shape[2]} vs {nprobes}")
        if positions is None and pos is not None:
            positions = pos
        all_t.append(t)
        all_v.append(v)

    time = np.concatenate(all_t, axis=0)
    vel = np.concatenate(all_v, axis=1)

    if positions is None:
        positions = read_probe_positions_from_system(case_dir, probes_name, nprobes)
    if positions is None:
        raise RuntimeError(
            "Could not determine probe locations from U-file headers or system/probes file. "
            "Add '# Probe i (x y z)' headers or provide system/probes2 with probeLocations."
        )

    return time, vel, positions, files


def safe_float(text: str, default: float) -> float:
    try:
        return float(text)
    except Exception:
        return default


def read_probe_positions_from_system(case_dir: Path, probes_name: str, nprobes: int) -> Optional[np.ndarray]:
    candidates = [case_dir / "system" / probes_name, case_dir / "system" / "probes"]
    for p in candidates:
        if not p.exists():
            continue
        txt = p.read_text(errors="ignore")
        # Prefer probeLocations block. This simple regex is sufficient for normal OpenFOAM probe dictionaries.
        triples = VEC_RE.findall(txt)
        if len(triples) >= nprobes:
            vals = np.asarray([[float(a), float(b), float(c)] for a, b, c in triples[:nprobes]], dtype=float)
            return vals
    return None


def clean_time_history(
    time_full: np.ndarray,
    vel_full: np.ndarray,
    burn_in_time: float,
    duplicate_tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    t = np.asarray(time_full, dtype=float)
    v = np.asarray(vel_full, dtype=float)
    finite = np.isfinite(t)
    finite &= np.all(np.isfinite(v), axis=(0, 2))
    mask = finite & (t > burn_in_time)
    if np.count_nonzero(mask) < 8:
        raise RuntimeError(f"Not enough finite probe samples after burn_in_time={burn_in_time}")
    t = t[mask]
    v = v[:, mask, :]

    order = np.argsort(t, kind="mergesort")
    t = t[order]
    v = v[:, order, :]

    # Keep last occurrence of duplicated times.
    keep = np.r_[np.diff(t) > duplicate_tol, True]
    duplicate_count = int(t.size - np.count_nonzero(keep))
    t = t[keep]
    v = v[:, keep, :]

    diffs = np.diff(t)
    pos = diffs[np.isfinite(diffs) & (diffs > duplicate_tol)]
    if pos.size == 0:
        raise RuntimeError("Could not infer positive time step from post-burn-in probe data")
    dt = float(np.median(pos))
    fs = 1.0 / dt

    info = {
        "n_samples": int(t.size),
        "t_min": float(t.min()),
        "t_max": float(t.max()),
        "dt": dt,
        "fs": fs,
        "duplicate_count_removed": duplicate_count,
    }
    return t, v, info


# -----------------------------------------------------------------------------
# Time-domain turbulence statistics
# -----------------------------------------------------------------------------


def autocorr_fft(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = x.size
    var = np.mean(x * x)
    if not np.isfinite(var) or var <= 0.0:
        return np.ones(1)
    nfft = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, n=nfft)
    ac = np.fft.irfft(fx * np.conjugate(fx), n=nfft)[:n]
    # Biased normalization is less noisy at large lags than unbiased normalization.
    ac = ac / ac[0]
    return np.asarray(ac, dtype=float)


def integral_time_scale_first_zero(x: np.ndarray, dt: float, max_lag_fraction: float = 0.5) -> float:
    ac = autocorr_fft(x)
    if ac.size < 3:
        return 0.0
    nmax = max(3, min(ac.size, int(max_lag_fraction * ac.size)))
    ac = ac[:nmax]
    nonpos = np.where(ac <= 0.0)[0]
    if nonpos.size:
        iz = int(nonpos[0])
        if iz <= 1:
            return 0.0
        # Linear interpolation to zero crossing for cleaner area.
        y0, y1 = ac[iz - 1], ac[iz]
        frac = y0 / (y0 - y1) if (y0 - y1) != 0 else 0.0
        t_vals = np.arange(iz, dtype=float) * dt
        ac_vals = ac[:iz].copy()
        t_zero = (iz - 1 + frac) * dt
        t_aug = np.r_[t_vals, t_zero]
        ac_aug = np.r_[ac_vals, 0.0]
        return float(max(trapz(ac_aug, t_aug), 0.0))
    else:
        t_vals = np.arange(ac.size, dtype=float) * dt
        return float(max(trapz(ac, t_vals), 0.0))


def integral_time_scale_expfit(x: np.ndarray, dt: float, max_lag_fraction: float = 0.25) -> float:
    """Simple exponential fit R(t)=exp(-t/T). Falls back to first-zero if fit is poor."""
    ac = autocorr_fft(x)
    nmax = max(10, min(ac.size, int(max_lag_fraction * ac.size)))
    ac = ac[:nmax]
    mask = (ac > 0.05) & (ac < 0.95)
    if np.count_nonzero(mask) < 5:
        return integral_time_scale_first_zero(x, dt, max_lag_fraction=0.5)
    tau = np.arange(ac.size, dtype=float) * dt
    slope, intercept = np.polyfit(tau[mask], np.log(ac[mask]), deg=1)
    if slope >= 0 or not np.isfinite(slope):
        return integral_time_scale_first_zero(x, dt, max_lag_fraction=0.5)
    return float(-1.0 / slope)


def grouped_profile_from_velocity(
    vel: np.ndarray,
    positions: np.ndarray,
    dt: float,
    z_tol: float = 1e-7,
    l_method: str = "firstZero",
    max_lag_fraction: float = 0.5,
) -> pd.DataFrame:
    """Compute profile statistics from velocity array shaped (3,nTimes,nProbes)."""
    z = np.asarray(positions[:, 2], dtype=float)
    rounded = np.round(z / z_tol).astype(np.int64) if z_tol > 0 else np.arange(z.size)
    groups: Dict[int, List[int]] = {}
    for i, key in enumerate(rounded):
        groups.setdefault(int(key), []).append(i)

    rows = []
    for key, inds in sorted(groups.items(), key=lambda kv: np.mean(z[kv[1]])):
        inds_arr = np.asarray(inds, dtype=int)
        per_probe = []
        for j in inds_arr:
            u = vel[0, :, j]
            v = vel[1, :, j]
            w = vel[2, :, j]
            Umean = float(np.mean(u))
            Vmean = float(np.mean(v))
            Wmean = float(np.mean(w))
            up = u - Umean
            vp = v - Vmean
            wp = w - Wmean
            uu = float(np.mean(up * up))
            vv = float(np.mean(vp * vp))
            ww = float(np.mean(wp * wp))
            uw = float(np.mean(up * wp))
            if l_method.lower() == "expfit":
                Tu = integral_time_scale_expfit(up, dt, max_lag_fraction=max_lag_fraction)
                Tv = integral_time_scale_expfit(vp, dt, max_lag_fraction=max_lag_fraction)
                Tw = integral_time_scale_expfit(wp, dt, max_lag_fraction=max_lag_fraction)
            else:
                Tu = integral_time_scale_first_zero(up, dt, max_lag_fraction=max_lag_fraction)
                Tv = integral_time_scale_first_zero(vp, dt, max_lag_fraction=max_lag_fraction)
                Tw = integral_time_scale_first_zero(wp, dt, max_lag_fraction=max_lag_fraction)
            U_for_L = max(abs(Umean), 1e-12)
            per_probe.append(
                {
                    "z": float(z[j]),
                    "U": Umean,
                    "uu": uu,
                    "vv": vv,
                    "ww": ww,
                    "uwStress": uw,
                    "Lu": U_for_L * Tu,
                    "Lv": U_for_L * Tv,
                    "Lw": U_for_L * Tw,
                }
            )
        pp = pd.DataFrame(per_probe)
        Ubar = float(pp["U"].mean())
        row = {
            "z": float(pp["z"].mean()),
            "U": Ubar,
            "Iu": math.sqrt(max(float(pp["uu"].mean()), 0.0)) / max(abs(Ubar), 1e-12),
            "Iv": math.sqrt(max(float(pp["vv"].mean()), 0.0)) / max(abs(Ubar), 1e-12),
            "Iw": math.sqrt(max(float(pp["ww"].mean()), 0.0)) / max(abs(Ubar), 1e-12),
            "Lu": float(pp["Lu"].mean()),
            "Lv": float(pp["Lv"].mean()),
            "Lw": float(pp["Lw"].mean()),
            "uwStress": float(pp["uwStress"].mean()),
            "nProbesAveraged": int(len(inds_arr)),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("z").reset_index(drop=True)


# -----------------------------------------------------------------------------
# Wong update
# -----------------------------------------------------------------------------


@dataclass
class CalibrationConfig:
    case_dir: Path
    probes_name: str
    profile_dir: Path
    building_height: float
    lower_z: float
    upper_z: float
    rmse_threshold: float
    burn_in_time: float
    calibrate_u: bool
    calibrate_i: bool
    calibrate_l: bool
    calibrate_uw: bool
    relax_u: float
    relax_i: float
    relax_l: float
    relax_uw: float
    min_factor_u: float
    max_factor_u: float
    min_factor_i: float
    max_factor_i: float
    min_factor_l: float
    max_factor_l: float
    min_factor_uw: float
    max_factor_uw: float
    rho_uw_limit: float
    smooth_window: int
    update_target_height_range: bool
    l_method: str
    max_lag_fraction: float
    z_group_tol: float


def load_config() -> CalibrationConfig:
    case_dir = Path(os.environ.get("CASE_DIR", os.getcwd())).resolve()
    setup = parse_setup_file(case_dir)
    sim_init = read_sim_init(case_dir)

    building_height = env_float("MHW_BUILDING_HEIGHT", setup.get("buildingHeight", 1.0))
    lower_z = env_float("MHW_LOWER_Z_THRESHOLD", setup.get("lowerZThreshold", 0.25 * building_height))
    upper_z = env_float("MHW_UPPER_Z_THRESHOLD", setup.get("upperZThreshold", 1.5 * building_height))
    rmse_threshold = env_float("MHW_RMSE_THRESHOLD", setup.get("rmseThreshold", 0.05))
    burn = env_float("MHW_BURN_IN_TIME", sim_init.get("burn_in_time", 0.0))

    rel_profile_dir = os.environ.get("MHW_PROFILE_DIR", "constant/boundaryData/windProfile")

    smooth_window = env_int("MHW_SMOOTH_WINDOW", 0)
    if smooth_window < 0:
        smooth_window = 0
    if smooth_window > 0 and smooth_window % 2 == 0:
        smooth_window += 1

    return CalibrationConfig(
        case_dir=case_dir,
        probes_name=os.environ.get("MHW_PROBES_NAME", "probes2"),
        profile_dir=case_dir / rel_profile_dir,
        building_height=building_height,
        lower_z=lower_z,
        upper_z=upper_z,
        rmse_threshold=rmse_threshold,
        burn_in_time=burn,
        calibrate_u=env_bool("MHW_CALIBRATE_U", True),
        calibrate_i=env_bool("MHW_CALIBRATE_I", True),
        calibrate_l=env_bool("MHW_CALIBRATE_L", True),
        calibrate_uw=env_bool("MHW_CALIBRATE_UW", True),
        relax_u=env_float("MHW_RELAX_U", 0.20),
        relax_i=env_float("MHW_RELAX_I", 0.35),
        relax_l=env_float("MHW_RELAX_L", 0.25),
        relax_uw=env_float("MHW_RELAX_UW", 0.20),
        min_factor_u=env_float("MHW_MIN_FACTOR_U", 0.90),
        max_factor_u=env_float("MHW_MAX_FACTOR_U", 1.10),
        min_factor_i=env_float("MHW_MIN_FACTOR_I", 0.60),
        max_factor_i=env_float("MHW_MAX_FACTOR_I", 1.80),
        min_factor_l=env_float("MHW_MIN_FACTOR_L", 0.70),
        max_factor_l=env_float("MHW_MAX_FACTOR_L", 1.50),
        min_factor_uw=env_float("MHW_MIN_FACTOR_UW", 0.70),
        max_factor_uw=env_float("MHW_MAX_FACTOR_UW", 1.40),
        rho_uw_limit=env_float("MHW_RHO_UW_LIMIT", 0.999),
        smooth_window=smooth_window,
        update_target_height_range=env_bool("MHW_UPDATE_TARGET_HEIGHT_RANGE", True),
        l_method=os.environ.get("MHW_L_METHOD", "firstZero"),
        max_lag_fraction=env_float("MHW_MAX_LAG_FRACTION", 0.5),
        z_group_tol=env_float("MHW_Z_GROUP_TOL", 1e-7),
    )


def positive_wong_update(
    current: np.ndarray,
    target: np.ndarray,
    downstream: np.ndarray,
    relax: float,
    min_factor: float,
    max_factor: float,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    downstream = np.asarray(downstream, dtype=float)
    ratio = np.maximum(target, eps) / np.maximum(downstream, eps)
    factor = np.clip(ratio ** relax, min_factor, max_factor)
    return current * factor, factor


def signed_uw_wong_update(
    current: np.ndarray,
    target: np.ndarray,
    downstream: np.ndarray,
    relax: float,
    min_factor: float,
    max_factor: float,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    downstream = np.asarray(downstream, dtype=float)

    mag_target = np.abs(target)
    mag_down = np.abs(downstream)
    mag_current = np.abs(current)

    # If current is zero but target is not, start from the target magnitude rather
    # than remaining permanently zero.
    mag_current = np.maximum(mag_current, 0.25 * mag_target)
    mag_current = np.maximum(mag_current, eps)

    ratio = np.maximum(mag_target, eps) / np.maximum(mag_down, eps)
    factor = np.clip(ratio ** relax, min_factor, max_factor)

    sign = np.sign(target)
    zero = sign == 0.0
    sign[zero] = np.sign(current[zero])
    sign[sign == 0.0] = -1.0

    updated = sign * mag_current * factor
    updated[mag_target <= eps] = 0.0
    return updated, factor


def moving_average_smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(y, dtype=float)
    if window % 2 == 0:
        window += 1
    y = np.asarray(y, dtype=float)
    if y.size < window:
        return y
    pad = window // 2
    yp = np.pad(y, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(yp, kernel, mode="valid")


def smooth_updated_profile(updated: pd.DataFrame, target: pd.DataFrame, cfg: CalibrationConfig) -> pd.DataFrame:
    if cfg.smooth_window <= 1:
        return updated
    out = updated.copy()
    # Keep U less smoothed by default; profile smoothness usually comes from target generation.
    for c in ["Iu", "Iv", "Iw", "Lu", "Lv", "Lw", "uwStress"]:
        if c in out.columns:
            out[c] = moving_average_smooth(out[c].to_numpy(dtype=float), cfg.smooth_window)
    return out


def enforce_profile_bounds(df: pd.DataFrame, cfg: CalibrationConfig) -> pd.DataFrame:
    out = df.copy()
    out["U"] = np.maximum(out["U"].to_numpy(dtype=float), 1e-6)
    for c in ["Iu", "Iv", "Iw"]:
        out[c] = np.clip(out[c].to_numpy(dtype=float), 1e-5, 2.0)
    for c in ["Lu", "Lv", "Lw"]:
        out[c] = np.maximum(out[c].to_numpy(dtype=float), 1e-6)
    if "uwStress" in out.columns:
        sig_u = out["Iu"].to_numpy(dtype=float) * out["U"].to_numpy(dtype=float)
        sig_w = out["Iw"].to_numpy(dtype=float) * out["U"].to_numpy(dtype=float)
        lim = cfg.rho_uw_limit * sig_u * sig_w
        out["uwStress"] = np.clip(out["uwStress"].to_numpy(dtype=float), -lim, lim)
    return out


def wong_update_profile(
    current: pd.DataFrame,
    target: pd.DataFrame,
    downstream: pd.DataFrame,
    cfg: CalibrationConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    z = current["z"].to_numpy(dtype=float)
    target_i = interp_profile_to_z(target, z)
    down_i = interp_profile_to_z(downstream, z)

    updated = current.copy()
    factors = pd.DataFrame({"z": z})
    active = np.ones(z.size, dtype=bool)
    if cfg.update_target_height_range:
        active = (z >= cfg.lower_z) & (z <= cfg.upper_z)

    def assign_active(col: str, vals: np.ndarray, fac: np.ndarray):
        old = updated[col].to_numpy(dtype=float).copy()
        old[active] = vals[active]
        updated[col] = old
        f = np.ones_like(old)
        f[active] = fac[active]
        factors[f"factor_{col}"] = f

    if cfg.calibrate_u:
        vals, fac = positive_wong_update(
            current["U"].to_numpy(dtype=float),
            target_i["U"].to_numpy(dtype=float),
            down_i["U"].to_numpy(dtype=float),
            cfg.relax_u,
            cfg.min_factor_u,
            cfg.max_factor_u,
        )
        assign_active("U", vals, fac)

    if cfg.calibrate_i:
        for c in ["Iu", "Iv", "Iw"]:
            vals, fac = positive_wong_update(
                current[c].to_numpy(dtype=float),
                target_i[c].to_numpy(dtype=float),
                down_i[c].to_numpy(dtype=float),
                cfg.relax_i,
                cfg.min_factor_i,
                cfg.max_factor_i,
            )
            assign_active(c, vals, fac)

    if cfg.calibrate_l:
        for c in ["Lu", "Lv", "Lw"]:
            vals, fac = positive_wong_update(
                current[c].to_numpy(dtype=float),
                target_i[c].to_numpy(dtype=float),
                down_i[c].to_numpy(dtype=float),
                cfg.relax_l,
                cfg.min_factor_l,
                cfg.max_factor_l,
            )
            assign_active(c, vals, fac)

    if cfg.calibrate_uw and "uwStress" in current.columns and "uwStress" in target_i.columns and "uwStress" in down_i.columns:
        vals, fac = signed_uw_wong_update(
            current["uwStress"].to_numpy(dtype=float),
            target_i["uwStress"].to_numpy(dtype=float),
            down_i["uwStress"].to_numpy(dtype=float),
            cfg.relax_uw,
            cfg.min_factor_uw,
            cfg.max_factor_uw,
        )
        assign_active("uwStress", vals, fac)

    updated = smooth_updated_profile(updated, target_i, cfg)
    updated = enforce_profile_bounds(updated, cfg)
    return updated, factors


# -----------------------------------------------------------------------------
# Diagnostics, plotting, convergence
# -----------------------------------------------------------------------------


def interpolate_at(z: np.ndarray, y: np.ndarray, z0: float) -> float:
    return float(np.interp(float(z0), np.asarray(z, dtype=float), np.asarray(y, dtype=float)))


def profile_rmse(target: pd.DataFrame, downstream: pd.DataFrame, cfg: CalibrationConfig) -> pd.DataFrame:
    z = target["z"].to_numpy(dtype=float)
    down = interp_profile_to_z(downstream, z)
    mask = (z >= cfg.lower_z) & (z <= cfg.upper_z)
    if np.count_nonzero(mask) < 2:
        mask[:] = True

    rows = []
    for c in ["U", "Iu", "Iv", "Iw", "Lu", "Lv", "Lw"]:
        t = target[c].to_numpy(dtype=float)[mask]
        d = down[c].to_numpy(dtype=float)[mask]
        rel = (d - t) / np.maximum(np.abs(t), 1e-12)
        rows.append({"quantity": c, "rmse": float(np.sqrt(np.mean(rel * rel))), "normalisation": "relative"})

    if "uwStress" in target.columns and "uwStress" in down.columns:
        t = target["uwStress"].to_numpy(dtype=float)[mask]
        d = down["uwStress"].to_numpy(dtype=float)[mask]
        sig_u = target["Iu"].to_numpy(dtype=float)[mask] * target["U"].to_numpy(dtype=float)[mask]
        sig_w = target["Iw"].to_numpy(dtype=float)[mask] * target["U"].to_numpy(dtype=float)[mask]
        norm = np.maximum(sig_u * sig_w, 1e-12)
        err = (d - t) / norm
        rows.append({"quantity": "uwStress", "rmse": float(np.sqrt(np.mean(err * err))), "normalisation": "sigma_u_sigma_w"})
        UH = interpolate_at(target["z"].to_numpy(), target["U"].to_numpy(), cfg.building_height)
        rows.append({"quantity": "uwStress_over_UH2", "rmse": float(np.sqrt(np.mean(((d - t) / max(UH * UH, 1e-12)) ** 2))), "normalisation": "UH2"})
    return pd.DataFrame(rows)


def converged_from_rmse(rmse_df: pd.DataFrame, cfg: CalibrationConfig) -> bool:
    # Do not let the reporting-only UH2 row control convergence.
    control = rmse_df[rmse_df["quantity"] != "uwStress_over_UH2"].copy()
    return bool((control["rmse"] <= cfg.rmse_threshold).all())


def read_optional_experimental_profile(profile_dir: Path) -> Optional[pd.DataFrame]:
    for name in ["targetExperimentalProfile", "targetExperimentalProfile_withStats", "targetSmoothedProfile"]:
        p = profile_dir / name
        if p.exists():
            try:
                return read_profile_file(p, allow_missing_uw=True)
            except Exception:
                # Some targetExperimentalProfile files may not have L columns; ignore if incompatible.
                pass
    return None


def plot_melaku_8panel(
    target: pd.DataFrame,
    downstream: pd.DataFrame,
    current: pd.DataFrame,
    updated: pd.DataFrame,
    cfg: CalibrationConfig,
    output_path: Path,
    experimental: Optional[pd.DataFrame] = None,
) -> None:
    H = cfg.building_height
    zt = target["z"].to_numpy(dtype=float)
    UH = interpolate_at(zt, target["U"].to_numpy(dtype=float), H)
    if not np.isfinite(UH) or abs(UH) < 1e-12:
        UH = float(np.nanmax(target["U"].to_numpy(dtype=float)))

    curves = [
        (target, "Target", "k", "--", None, 1.6),
        (downstream, "LES downstream", "tab:red", "-", None, 1.8),
        (current, "Current inlet", "0.45", ":", None, 1.4),
        (updated, "Updated inlet", "tab:blue", "-.", None, 1.8),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(17, 9), sharey=True)
    axes = axes.ravel()
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
    xlabels = [r"$U/U_H$", r"$I_u$", r"$I_v$", r"$I_w$", r"$u'w'/U_H^2$", r"$L_u/H$", r"$L_v/H$", r"$L_w/H$"]
    cols = ["U", "Iu", "Iv", "Iw", "uwStress", "Lu", "Lv", "Lw"]

    def xy_for(df: pd.DataFrame, col: str):
        z = df["z"].to_numpy(dtype=float)
        y = z / H
        if col == "U":
            x = df[col].to_numpy(dtype=float) / UH
        elif col == "uwStress":
            if col not in df.columns:
                x = np.full_like(y, np.nan)
            else:
                x = df[col].to_numpy(dtype=float) / max(UH * UH, 1e-12)
        elif col in ["Lu", "Lv", "Lw"]:
            x = df[col].to_numpy(dtype=float) / H
        else:
            x = df[col].to_numpy(dtype=float)
        return x, y

    for ax, lab, xlabel, col in zip(axes, panel_labels, xlabels, cols):
        if experimental is not None and col in experimental.columns:
            try:
                x, y = xy_for(experimental, col)
                finite = np.isfinite(x) & np.isfinite(y)
                ax.plot(x[finite], y[finite], "o", ms=4, mfc="none", mec="black", mew=0.8, label="EXP")
            except Exception:
                pass
        for df, label, color, ls, marker, lw in curves:
            if col not in df.columns:
                continue
            x, y = xy_for(df, col)
            finite = np.isfinite(x) & np.isfinite(y)
            ax.plot(x[finite], y[finite], linestyle=ls, marker=marker, color=color, lw=lw, label=label)
        ax.set_xlabel(xlabel)
        ax.set_title(lab, fontsize=12)
        ax.grid(True, alpha=0.35, linestyle="--")
        ax.set_ylim(0.0, max(3.0, 1.05 * np.nanmax(zt / H)))
        if col == "uwStress":
            ax.axvline(0.0, color="0.2", lw=0.8)
    axes[0].set_ylabel(r"$z/H$")
    axes[4].set_ylabel(r"$z/H$")
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle("MannHybrid Wong downstream calibration profiles", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def next_iteration_dir(log_root: Path) -> Tuple[int, Path]:
    ensure_dir(log_root)
    nums = []
    for p in log_root.glob("iteration*"):
        m = re.match(r"iteration(\d+)$", p.name)
        if m:
            nums.append(int(m.group(1)))
    n = (max(nums) + 1) if nums else 1
    return n, log_root / f"iteration{n:02d}"


def save_iteration_snapshot(
    iteration_dir: Path,
    cfg: CalibrationConfig,
    current: pd.DataFrame,
    target: pd.DataFrame,
    downstream: pd.DataFrame,
    updated: pd.DataFrame,
    factors: pd.DataFrame,
    rmse: pd.DataFrame,
    time_info: Dict[str, float],
    experimental: Optional[pd.DataFrame],
) -> None:
    inputs = ensure_dir(iteration_dir / "inputs")
    data = ensure_dir(iteration_dir / "data")
    profiles = ensure_dir(iteration_dir / "profiles")

    # Headered CSV data for analysis.
    write_profile_csv(current, data / "current_profile.csv")
    write_profile_csv(target, data / "target_profile.csv")
    write_profile_csv(downstream, data / "downstream_profile_time_series.csv")
    write_profile_csv(updated, data / "updated_profile.csv")
    factors.to_csv(data / "calibration_factors.csv", index=False, float_format="%.12e")
    rmse.to_csv(data / "rmse.csv", index=False, float_format="%.12e")
    with (data / "probe_time_info.json").open("w") as f:
        json.dump(time_info, f, indent=2)

    # Headerless OpenFOAM-style profile snapshots, ready to copy back if a given
    # iteration is chosen for a building simulation.
    write_profile_file(current, inputs / "profile_current_before_update", include_uw="uwStress" in current.columns, header=False)
    write_profile_file(target, inputs / "targetProfile", include_uw="uwStress" in target.columns, header=False)
    write_profile_file(downstream, inputs / "profile_downstream_time_series", include_uw="uwStress" in downstream.columns, header=False)
    write_profile_file(updated, inputs / "profile_updated", include_uw="uwStress" in updated.columns, header=False)
    if experimental is not None:
        write_profile_csv(experimental, data / "target_experimental_profile.csv")

    plot_melaku_8panel(
        target=target,
        downstream=downstream,
        current=current,
        updated=updated,
        cfg=cfg,
        output_path=profiles / f"{iteration_dir.name}_profiles_melaku_8panel.png",
        experimental=experimental,
    )


# -----------------------------------------------------------------------------
# Main procedure
# -----------------------------------------------------------------------------


def run() -> int:
    cfg = load_config()
    print("=== MannHybrid Wong profile calibration ===", flush=True)
    print(f"case_dir       = {cfg.case_dir}", flush=True)
    print(f"profile_dir    = {cfg.profile_dir}", flush=True)
    print(f"probes_name    = {cfg.probes_name}", flush=True)
    print(f"burn_in_time   = {cfg.burn_in_time}", flush=True)
    print(f"height range   = {cfg.lower_z} to {cfg.upper_z}", flush=True)
    print(f"rmse threshold = {cfg.rmse_threshold}", flush=True)

    warn_if_mann_dict_inconsistent(cfg.case_dir)

    current_path = cfg.profile_dir / "profile"
    target_path = cfg.profile_dir / "targetProfile"
    current = read_profile_file(current_path, allow_missing_uw=True)
    target = read_profile_file(target_path, allow_missing_uw=True)

    if "uwStress" not in current.columns and "uwStress" in target.columns:
        print("Current profile has no uwStress; initializing from target uwStress.", flush=True)
        current = interp_profile_to_z(target, current["z"].to_numpy(dtype=float))
    if "uwStress" not in target.columns:
        cfg.calibrate_uw = False
        print("Target profile has no uwStress; uw calibration disabled.", flush=True)
    if "uwStress" not in current.columns:
        cfg.calibrate_uw = False
        print("Current profile has no uwStress; uw calibration disabled.", flush=True)

    # Probe reading and downstream statistics.
    time_full, vel_full, positions, files = read_all_probe_segments(cfg.case_dir, cfg.probes_name)
    time, vel, time_info = clean_time_history(time_full, vel_full, cfg.burn_in_time)
    print("Probe files:", flush=True)
    for f in files:
        print(f"  {f}", flush=True)
    print(f"Cleaned probe samples = {time_info['n_samples']}", flush=True)
    print(f"Time range            = {time_info['t_min']} -> {time_info['t_max']}", flush=True)
    print(f"dt, fs                = {time_info['dt']}, {time_info['fs']}", flush=True)
    print(f"duplicates removed    = {time_info['duplicate_count_removed']}", flush=True)

    downstream_raw = grouped_profile_from_velocity(
        vel,
        positions,
        dt=float(time_info["dt"]),
        z_tol=cfg.z_group_tol,
        l_method=cfg.l_method,
        max_lag_fraction=cfg.max_lag_fraction,
    )
    downstream = interp_profile_to_z(downstream_raw, target["z"].to_numpy(dtype=float))

    updated, factors = wong_update_profile(current, target, downstream, cfg)
    rmse = profile_rmse(target, downstream, cfg)
    is_converged = converged_from_rmse(rmse, cfg)

    log_root = cfg.case_dir / "log" / "downstreamCalibration"
    iteration_num, iteration_dir = next_iteration_dir(log_root)
    experimental = read_optional_experimental_profile(cfg.profile_dir)
    save_iteration_snapshot(
        iteration_dir=iteration_dir,
        cfg=cfg,
        current=current,
        target=target,
        downstream=downstream,
        updated=updated,
        factors=factors,
        rmse=rmse,
        time_info=time_info,
        experimental=experimental,
    )

    print("RMSE summary:", flush=True)
    print(rmse.to_string(index=False), flush=True)
    print(f"Converged = {is_converged}", flush=True)
    print(f"Iteration data written to: {iteration_dir}", flush=True)

    if is_converged:
        print("No profile update written because convergence criterion is satisfied.", flush=True)
        return 0

    # Only active profile is overwritten. Spectra/co-spectra are not written by
    # this recipe; MannHybridTurb should construct von Karman / Kaimal internally.
    backup = cfg.profile_dir / f"profile_before_{iteration_dir.name}"
    if current_path.exists() and not backup.exists():
        shutil.copy(current_path, backup)
    write_profile_file(updated, current_path, include_uw="uwStress" in updated.columns, header=False)
    print(f"Updated active profile written to: {current_path}", flush=True)
    print("No spectraProfile or uwCoSpectrumProfile files were written by this script.", flush=True)
    return 1


def main() -> None:
    try:
        code = run()
    except Exception as exc:
        print("ERROR in MannHybridWongCalibration.py:", file=sys.stderr)
        print(f"  {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
