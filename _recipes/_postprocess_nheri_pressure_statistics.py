#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NHERI LES pressure-statistics post-processing with robust WT tap/column alignment.

Main fixes in this version
--------------------------
1. The wind-tunnel tap number is NOT used blindly as the Cp matrix column.
2. Cp columns are assigned from the valid tap-layout row order:
       cp_col = 0, 1, ..., nCp-1
   after obvious invalid rows are removed.
3. If the layout has one extra row and that row is tap 0 / blank / invalid, it is
   removed automatically. This resolves the common 0..510 vs 0..509 error.
4. The upstream reference velocity and pressure are taken from upstreamProbe.
5. Restarted OpenFOAM postProcessing folders are concatenated and duplicate times
   are removed.
6. Comparisons are written for:
       - full WT record, all repetitions combined
       - full WT record, REP1..REP5 separately
       - LES-duration WT record, all repetitions combined
       - LES-duration WT record, REP1..REP5 separately
7. Plots use mean Cp and RMS Cp only, with fixed presentation-style axes.
8. Roof/Surface 1 is excluded from plots, while remaining faces are labelled.

Before running
--------------
Check the CONFIG section below. The paths are written from your current folder
layout as far as possible, but you may need to adjust WT_CP_DIR.
"""

from __future__ import annotations

import os
import re
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

try:
    from scipy.io import loadmat
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script requires scipy: pip/conda install scipy") from exc


# =============================================================================
# CONFIG
# =============================================================================

CASE_DIR = Path(
    r"C:\Users\david\OneDrive\Documents\PhD\Year 1\NHERI LES Case\OpenFOAM Cases\Building Case\building_case_meluxina"
)

POSTPROCESSING_DIR = CASE_DIR / "postProcessing"

# Surface pressure probe folders. Non-existing folders are skipped.
# Add/remove names here to match your case.
LES_PRESSURE_PROBE_FOLDERS = [
    "probesSurface1",
    "probesSurface2",
    "probesSurface3",
    "probesSurface4",
    "probesSurface5",
    "probesSurface6",
]

UPSTREAM_PROBE_NAME = "upstreamProbe"

# Wind tunnel data.
WT_ROOT = Path(
    r"C:\Users\david\OneDrive\Documents\PhD\Year 1\Wind Tunnel Test Data\NHERI BLWT Tall Building"
)

TAP_LAYOUT_FILE = WT_ROOT / r"Pressure Taps\Tap Layout - FSU - Mid-Rise-Model.xlsx"

# The script searches recursively under this folder for Cp_PHASE1_ANG000_REP*.mat.
# Set this to the exact folder containing the .mat files if recursive search is slow.
WT_CP_DIR = WT_ROOT

WT_FILE_GLOB = "Cp_PHASE1_ANG000_REP*.mat"

# Analysis window in LES time.
LES_START_TIME = 25.0      # burn-in removal: use t >= 25 s
LES_END_TIME = None        # None means latest available LES sample

# WT matched-record window. Usually start at beginning of each WT repetition.
WT_MATCH_START_TIME = 0.0

# Moving averages are deliberately disabled. The comparisons use raw WT Cp records:
#   1) each full repetition, and
#   2) each repetition clipped to the same duration as the LES window.
WT_MOVING_AVERAGE_WINDOWS_SECONDS: List[float] = []

# Pressure coefficient definition.
# OpenFOAM incompressible p is kinematic pressure, so Cp = (p - p_ref)/(0.5*U_ref^2).
CP_SIGN = 1.0
U_REF_MODE = "magnitude"   # "magnitude", "Ux", "Uy", or "Uz"

# Coordinate handling.
# If tap layout coordinates appear to be in mm and LES coordinates in m, the script
# automatically scales WT coordinates by 0.001.
AUTO_SCALE_WT_COORDS = True

# If the WT tap layout uses a different origin from the LES probe coordinates,
# translate the scaled WT coordinate cloud so its minimum x/y/z bounds match
# the LES probe coordinate bounds. Your previous run had mapping distances
# around 2.3 m, which strongly suggests an origin offset.
AUTO_TRANSLATE_WT_COORDS_TO_LES_BOUNDS = True

# Map LES probes only to WT taps on the same surface when the surface number can
# be inferred from the LES folder name, e.g. probesSurface3 -> surface 3.
SURFACE_RESTRICTED_MAPPING = True

# Chunk size used for memory-safe statistics on large WT records.
WT_STATS_CHUNK_ROWS = 100_000

# If known bad/unused taps exist, list their tap labels here.
# They will be removed before assigning Cp columns by row order.
INVALID_TAP_IDS: List[int] = []

# If the layout has more rows than the Cp file has columns and the automatic invalid-row
# removal is still insufficient, this controls the fallback.
# True: keep first nCp valid layout rows and write a warning/diagnostic.
# False: stop with a clear error.
ALLOW_TRUNCATE_LAYOUT_TO_N_CP_COLUMNS = True

# Plot styling.
FIG_DPI = 220
SCATTER_SIZE = 72
SCATTER_LINEWIDTH = 1.6
AXIS_LABEL_FONTSIZE = 16
TICK_FONTSIZE = 13
TITLE_FONTSIZE = 15

# Plot controls requested for presentation-style comparison figures.
PLOT_STATS = ["mean", "rms"]
SCATTER_AXIS_LIMITS = {
    "mean": (-2.0, 2.0),
    "rms": (0.0, 0.6),
}

# Contour plot controls.
# Mean Cp: -2..2 split into 20 colour bins (0.2 increments).
# RMS Cp: 0..1 with 0.1-spaced colour levels.
CONTOUR_VALUE_LIMITS = {
    "mean": (-2.0, 2.0),
    "rms": (0.0, 1.0),
}
CONTOUR_VALUE_STEP = {
    "mean": 0.2,
    "rms": 0.1,
}

# Percent error is 100*(LES - WT)/abs(WT), with a small denominator floor
# to avoid meaningless blow-ups where WT is very close to zero.
PERCENT_ERROR_DENOM_FLOOR = 0.05
PERCENT_ERROR_LEVEL_STEP = 5.0
PERCENT_ERROR_CLIP_PERCENTILE = 98.0
PERCENT_ERROR_MIN_LIMIT = 25.0
PERCENT_ERROR_MAX_LIMIT = 200.0

# Surface 1 is the roof in your layout; keep it in CSV diagnostics but exclude it from figures.
EXCLUDE_SURFACES_FROM_PLOTS = {"1", "1.0", "surface 1", "roof"}

# Adjust these labels if your tap-layout surface numbering differs.
FACE_LABEL_BY_SURFACE = {
    "2": "Windward face",
    "2.0": "Windward face",
    "3": "Leeward face",
    "3.0": "Leeward face",
    "4": "Side face",
    "4.0": "Side face",
    "5": "Side face",
    "5.0": "Side face",
    "6": "Side face",
    "6.0": "Side face",
}

# If the Scanivalve Cp column ordering for a side face is opposite to the tap-layout
# coordinate ordering, the LES-to-WT mapping will look spatially mirrored even when
# the coordinates themselves appear to align. These surfaces are mirrored in the
# WT tap-coordinate table before nearest-neighbour mapping so Cp columns are paired
# with the opposite horizontal coordinate on that face. Leave empty to disable.
MIRROR_WT_MAPPING_SURFACES = {"4", "5"}

# Output.
OUT_DIR = CASE_DIR / "postProcessing_nheri_pressure_statistics_revised"


# =============================================================================
# General helpers
# =============================================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_float_from_name(name: str) -> float:
    try:
        return float(name)
    except Exception:
        return float("inf")


def natural_time_sort(paths: Sequence[Path]) -> List[Path]:
    return sorted(paths, key=lambda p: (safe_float_from_name(p.parent.name), str(p)))


def as_1d(a) -> np.ndarray:
    return np.asarray(a).reshape(-1)


def parse_vector_token(tok: str) -> Tuple[float, float, float]:
    tok = tok.strip()
    tok = tok.strip("()")
    parts = tok.split()
    if len(parts) != 3:
        raise ValueError(f"Could not parse OpenFOAM vector token: {tok!r}")
    return float(parts[0]), float(parts[1]), float(parts[2])


def parse_openfoam_data_line(line: str) -> List[str]:
    """Split OpenFOAM probe line while keeping parenthesised vectors together."""
    line = line.strip()
    if not line or line.startswith("#"):
        return []
    return re.findall(r"\([^)]+\)|\S+", line)


def read_probe_coordinates_from_file(path: Path) -> Optional[np.ndarray]:
    coords = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("#"):
                break
            # Typical:
            # # Probe 0 (1 -1.2 0.5)
            m = re.search(r"Probe\s+\d+\s+\(([^)]+)\)", line)
            if m:
                vals = [float(x) for x in m.group(1).split()]
                if len(vals) == 3:
                    coords.append(vals)
    if coords:
        return np.asarray(coords, dtype=float)
    return None


def find_field_files(probe_folder: Path, field_name: str) -> List[Path]:
    """
    OpenFOAM postProcessing usually uses:
        postProcessing/folder/startTime/U
        postProcessing/folder/startTime/p
    Restarted runs create multiple startTime folders. This returns all matching files.
    """
    probe_folder = Path(probe_folder)
    if not probe_folder.exists():
        return []
    files = [p for p in probe_folder.rglob(field_name) if p.is_file()]
    return natural_time_sort(files)


def concat_and_deduplicate(times: List[np.ndarray], values: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if not times:
        raise ValueError("No time arrays provided.")
    t = np.concatenate(times)
    x = np.concatenate(values, axis=0)

    finite = np.isfinite(t)
    t = t[finite]
    x = x[finite]

    order = np.argsort(t, kind="mergesort")
    t = t[order]
    x = x[order]

    # Keep last occurrence of duplicate times, useful after restarts.
    _, last_idx_reversed = np.unique(t[::-1], return_index=True)
    keep = len(t) - 1 - last_idx_reversed
    keep = np.sort(keep)

    return t[keep], x[keep]


def read_openfoam_scalar_probe_folder(probe_folder: Path, field_name: str = "p") -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    files = find_field_files(probe_folder, field_name)
    if not files:
        raise FileNotFoundError(f"No OpenFOAM field file {field_name!r} found under {probe_folder}")

    coords = read_probe_coordinates_from_file(files[0])
    times, vals = [], []

    for path in files:
        local_t = []
        local_v = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                toks = parse_openfoam_data_line(line)
                if not toks:
                    continue
                local_t.append(float(toks[0]))
                local_v.append([float(x) for x in toks[1:]])
        if local_t:
            times.append(np.asarray(local_t, dtype=float))
            vals.append(np.asarray(local_v, dtype=float))

    t, v = concat_and_deduplicate(times, vals)
    return t, v, coords


def read_openfoam_vector_probe_folder(probe_folder: Path, field_name: str = "U") -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    files = find_field_files(probe_folder, field_name)
    if not files:
        raise FileNotFoundError(f"No OpenFOAM field file {field_name!r} found under {probe_folder}")

    coords = read_probe_coordinates_from_file(files[0])
    times, vals = [], []

    for path in files:
        local_t = []
        local_v = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                toks = parse_openfoam_data_line(line)
                if not toks:
                    continue
                local_t.append(float(toks[0]))
                local_v.append([parse_vector_token(tok) for tok in toks[1:]])
        if local_t:
            times.append(np.asarray(local_t, dtype=float))
            vals.append(np.asarray(local_v, dtype=float))

    t, v = concat_and_deduplicate(times, vals)
    return t, v, coords


def time_mask(t: np.ndarray, start: Optional[float], end: Optional[float]) -> np.ndarray:
    m = np.ones_like(t, dtype=bool)
    if start is not None:
        m &= t >= start
    if end is not None:
        m &= t <= end
    return m


def init_stats_acc(n: int) -> Dict[str, np.ndarray]:
    return {
        "count": np.zeros(n, dtype=np.int64),
        "sum": np.zeros(n, dtype=np.float64),
        "sumsq": np.zeros(n, dtype=np.float64),
        "min": np.full(n, np.inf, dtype=np.float64),
        "max": np.full(n, -np.inf, dtype=np.float64),
    }


def update_stats_acc(acc: Dict[str, np.ndarray], x: np.ndarray) -> None:
    """Memory-safe axis-0 update for mean/std/min/max, ignoring NaNs/Infs."""
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2-D array, got shape {x.shape}")

    finite = np.isfinite(x)
    acc["count"] += finite.sum(axis=0)

    # np.sum(..., where=finite) avoids allocating a full NaN-replaced float64 copy.
    acc["sum"] += np.sum(x, axis=0, where=finite, dtype=np.float64)

    # Process square in-place on a chunk-sized float64 array, not the full WT record.
    x64 = np.asarray(x, dtype=np.float64)
    np.square(x64, out=x64)
    acc["sumsq"] += np.sum(x64, axis=0, where=finite, dtype=np.float64)

    with np.errstate(all="ignore"):
        acc["min"] = np.minimum(acc["min"], np.min(x, axis=0, where=finite, initial=np.inf))
        acc["max"] = np.maximum(acc["max"], np.max(x, axis=0, where=finite, initial=-np.inf))


def finalize_stats_acc(acc: Dict[str, np.ndarray]) -> pd.DataFrame:
    count = acc["count"].astype(np.float64)
    mean = np.divide(acc["sum"], count, out=np.full_like(acc["sum"], np.nan), where=count > 0)

    # RMS of pressure-coefficient fluctuations about the record mean.
    # This is the population RMS, sqrt(E[(Cp - mean(Cp))^2]).
    rms_var = np.divide(acc["sumsq"], count, out=np.full_like(acc["sumsq"], np.nan), where=count > 0) - mean ** 2
    rms_var[rms_var < 0] = 0.0
    rms = np.sqrt(rms_var)

    # Sample standard deviation is still saved to CSV for diagnostics, but it is not plotted.
    var_num = acc["sumsq"] - (acc["sum"] ** 2) / np.maximum(count, 1.0)
    var = np.divide(var_num, count - 1.0, out=np.full_like(var_num, np.nan), where=count > 1)
    var[var < 0] = 0.0

    mn = acc["min"].copy()
    mx = acc["max"].copy()
    mn[~np.isfinite(mn)] = np.nan
    mx[~np.isfinite(mx)] = np.nan

    return pd.DataFrame({
        "mean": mean,
        "rms": rms,
        "std": np.sqrt(var),
        "min": mn,
        "max": mx,
    })


def stats_matrix(x: np.ndarray) -> pd.DataFrame:
    """
    x shape: [samples, points]. Returns mean/rms/std/min/max without building
    a large float64 copy of the full input array.
    """
    x = np.asarray(x)
    acc = init_stats_acc(x.shape[1])
    for i0 in range(0, x.shape[0], WT_STATS_CHUNK_ROWS):
        update_stats_acc(acc, x[i0:i0 + WT_STATS_CHUNK_ROWS, :])
    return finalize_stats_acc(acc)


def read_les_pressure_and_reference() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], np.ndarray]:
    print("Reading LES pressure probes and upstream reference...")

    # Read surface pressure probe folders and concatenate probe columns.
    p_blocks = []
    coord_blocks = []
    source_blocks = []
    common_t = None

    for folder_name in LES_PRESSURE_PROBE_FOLDERS:
        folder = POSTPROCESSING_DIR / folder_name
        if not folder.exists():
            continue

        try:
            t, p, coords = read_openfoam_scalar_probe_folder(folder, "p")
        except FileNotFoundError:
            continue

        if common_t is None:
            common_t = t
        else:
            # Interpolate if the folders are not perfectly identical in time.
            if len(t) != len(common_t) or np.nanmax(np.abs(t - common_t)) > 1e-10:
                p_interp = np.empty((len(common_t), p.shape[1]), dtype=float)
                for j in range(p.shape[1]):
                    p_interp[:, j] = np.interp(common_t, t, p[:, j])
                p = p_interp
                t = common_t

        p_blocks.append(p)
        if coords is not None:
            coord_blocks.append(coords)
        else:
            coord_blocks.append(np.full((p.shape[1], 3), np.nan))
        source_blocks.extend([folder_name] * p.shape[1])

    if common_t is None or not p_blocks:
        raise FileNotFoundError(
            f"No LES pressure probe files found. Checked folders under {POSTPROCESSING_DIR}: "
            f"{LES_PRESSURE_PROBE_FOLDERS}"
        )

    p_all = np.concatenate(p_blocks, axis=1)
    coords_all = np.concatenate(coord_blocks, axis=0)

    m = time_mask(common_t, LES_START_TIME, LES_END_TIME)
    if not np.any(m):
        raise ValueError("LES analysis window contains no samples.")

    t_used = common_t[m]
    p_used = p_all[m, :]

    # Upstream reference.
    up_folder = POSTPROCESSING_DIR / UPSTREAM_PROBE_NAME
    tU, U, up_coords = read_openfoam_vector_probe_folder(up_folder, "U")
    tp, pref, _ = read_openfoam_scalar_probe_folder(up_folder, "p")

    mU = time_mask(tU, LES_START_TIME, LES_END_TIME)
    mp = time_mask(tp, LES_START_TIME, LES_END_TIME)

    if U_REF_MODE.lower() == "magnitude":
        U_series = np.linalg.norm(U[mU, 0, :], axis=1)
    else:
        comp_map = {"ux": 0, "uy": 1, "uz": 2}
        idx = comp_map[U_REF_MODE.lower()]
        U_series = U[mU, 0, idx]

    p_ref_series = pref[mp, 0]

    U_ref = float(np.nanmean(U_series))
    p_ref = float(np.nanmean(p_ref_series))
    q_ref = 0.5 * U_ref ** 2

    if not np.isfinite(q_ref) or q_ref <= 0:
        raise ValueError(f"Invalid q_ref={q_ref}; check upstreamProbe U.")

    cp_les = CP_SIGN * (p_used - p_ref) / q_ref
    les_stats = stats_matrix(cp_les)
    les_stats.insert(0, "les_index", np.arange(cp_les.shape[1]))
    les_stats["probe_source"] = source_blocks
    les_stats["x_les"] = coords_all[:, 0]
    les_stats["y_les"] = coords_all[:, 1]
    les_stats["z_les"] = coords_all[:, 2]

    ref_info = {
        "les_start_used": float(t_used[0]),
        "les_end_used": float(t_used[-1]),
        "les_duration": float(t_used[-1] - t_used[0]),
        "les_samples": int(len(t_used)),
        "U_ref": U_ref,
        "p_ref": p_ref,
        "q_ref": q_ref,
    }

    print(f"LES analysis window: {t_used[0]:.6g} to {t_used[-1]:.6g} s")
    print(f"LES duration used for matched WT records: {ref_info['les_duration']:.6g} s")
    print(f"U_ref={U_ref:.6g}, p_ref={p_ref:.6g}, q_ref={q_ref:.6g}")

    return les_stats, pd.DataFrame(cp_les), ref_info, coords_all


# =============================================================================
# Wind tunnel reader
# =============================================================================

def mat_public_items(raw: Dict) -> Dict:
    return {k: v for k, v in raw.items() if not k.startswith("__")}


def field_from_obj(obj, name: str, default=None):
    """Robust field getter for scipy MATLAB structs/dicts."""
    if obj is None:
        return default

    if isinstance(obj, dict):
        if name in obj:
            return obj[name]
        for k in obj.keys():
            if str(k).lower() == name.lower():
                return obj[k]
        return default

    if hasattr(obj, name):
        return getattr(obj, name)

    arr = np.asarray(obj)
    if arr.dtype.names:
        names = list(arr.dtype.names)
        for n in names:
            if n.lower() == name.lower():
                return arr[n]
    if arr.size == 1:
        item = arr.reshape(-1)[0]
        if hasattr(item, name):
            return getattr(item, name)
        if isinstance(item, np.void) and item.dtype.names:
            for n in item.dtype.names:
                if n.lower() == name.lower():
                    return item[n]

    return default


def unwrap_mat(x):
    """Remove common 1x1 object wrappers."""
    for _ in range(20):
        if isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
            x = x.reshape(-1)[0]
            continue
        if isinstance(x, np.ndarray) and x.size == 1 and x.dtype.names:
            x = x.reshape(-1)[0]
            continue
        break
    return x


def maybe_float(x, default=np.nan) -> float:
    try:
        x = unwrap_mat(x)
        arr = np.asarray(x).astype(float).reshape(-1)
        if arr.size:
            return float(arr[0])
    except Exception:
        pass
    return float(default)


def find_numeric_2d_arrays(obj, prefix="") -> List[Tuple[str, np.ndarray]]:
    """Fallback scanner inside Scanivalve if field names differ."""
    out = []
    obj = unwrap_mat(obj)

    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(find_numeric_2d_arrays(v, f"{prefix}.{k}" if prefix else str(k)))
        return out

    if hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            if not k.startswith("_"):
                out.extend(find_numeric_2d_arrays(v, f"{prefix}.{k}" if prefix else str(k)))
        return out

    arr = np.asarray(obj)
    if arr.dtype.names:
        for n in arr.dtype.names:
            out.extend(find_numeric_2d_arrays(arr[n], f"{prefix}.{n}" if prefix else str(n)))
        return out

    if np.issubdtype(arr.dtype, np.number) and arr.ndim == 2 and min(arr.shape) > 10:
        out.append((prefix, arr))

    return out


def get_scanivalve_struct(raw: Dict, path: Path):
    public = mat_public_items(raw)
    if "Scanivalve" not in public:
        keys = ", ".join(public.keys())
        raise KeyError(f"{path.name}: expected top-level variable 'Scanivalve'. Available variables: {keys}")
    return unwrap_mat(public["Scanivalve"])


def extract_cp_from_scanivalve(scan, path: Path) -> np.ndarray:
    # Known/common names first.
    cp_names = [
        "Cp", "CP", "cp",
        "PressureCoefficient", "PressureCoefficients",
        "pressureCoefficient", "pressureCoefficients",
        "Coeff", "Coefficients",
        "Data", "data",
    ]

    for name in cp_names:
        val = field_from_obj(scan, name, default=None)
        if val is None:
            continue
        arr = np.asarray(unwrap_mat(val))
        if np.issubdtype(arr.dtype, np.number) and arr.ndim == 2 and min(arr.shape) > 10:
            cp = np.asarray(arr, dtype=np.float32)
            # Cp should be [samples, taps]. If transposed, fix it.
            if cp.shape[0] < cp.shape[1]:
                cp = cp.T
            return cp

    # Fallback: largest numeric 2-D array inside Scanivalve.
    candidates = find_numeric_2d_arrays(scan, "Scanivalve")
    if not candidates:
        raise ValueError(f"{path.name}: could not identify Cp array inside Scanivalve.")

    candidates.sort(key=lambda kv: np.asarray(kv[1]).size, reverse=True)
    selected_name, selected_arr = candidates[0]
    warnings.warn(
        f"{path.name}: Cp field name not found directly; using largest numeric 2-D array: {selected_name}",
        RuntimeWarning,
    )
    cp = np.asarray(selected_arr, dtype=np.float32)
    if cp.shape[0] < cp.shape[1]:
        cp = cp.T
    return cp


@dataclass
class WTData:
    path: Path
    rep_label: str
    cp: np.ndarray
    fs: float
    duration: float


def read_scanivalve_cp_file(path: Path) -> WTData:
    path = Path(path)

    # scipy handles normal MAT files. The previous "Expecting miMATRIX" can happen
    # when a file is not actually a standard MAT file or is partially corrupted.
    raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    scan = get_scanivalve_struct(raw, path)

    cp = extract_cp_from_scanivalve(scan, path)

    fs = maybe_float(field_from_obj(scan, "SamplingFreq", None), default=np.nan)
    if not np.isfinite(fs):
        fs = maybe_float(field_from_obj(scan, "SamplingFrequency", None), default=np.nan)
    if not np.isfinite(fs):
        fs = maybe_float(field_from_obj(scan, "fs", None), default=np.nan)

    duration = maybe_float(field_from_obj(scan, "Duration", None), default=np.nan)
    if not np.isfinite(duration) and np.isfinite(fs) and fs > 0:
        duration = cp.shape[0] / fs

    if not np.isfinite(fs) or fs <= 0:
        if np.isfinite(duration) and duration > 0:
            fs = cp.shape[0] / duration
        else:
            raise ValueError(
                f"{path.name}: could not determine SamplingFreq or Duration from Scanivalve."
            )

    m = re.search(r"(REP\d+)", path.stem, flags=re.IGNORECASE)
    rep_label = m.group(1).upper() if m else path.stem

    return WTData(path=path, rep_label=rep_label, cp=cp, fs=float(fs), duration=float(duration))


def list_cp_files() -> List[Path]:
    files = sorted(Path(WT_CP_DIR).rglob(WT_FILE_GLOB))
    # Keep only REP1..REP5 if present.
    rep_re = re.compile(r"REP([1-5])", re.IGNORECASE)
    files = [f for f in files if rep_re.search(f.name)]
    if not files:
        raise FileNotFoundError(f"No files matching {WT_FILE_GLOB!r} under {WT_CP_DIR}")
    print("Wind-tunnel Cp files selected:")
    for f in files:
        print(f"  {f.name}")
    return files


# =============================================================================
# Tap layout and mapping
# =============================================================================

def detect_column(df: pd.DataFrame, candidates: Sequence[str], required=True) -> Optional[str]:
    norm = {re.sub(r"[^a-z0-9]+", "", str(c).lower()): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]+", "", cand.lower())
        if key in norm:
            return norm[key]

    # partial contains fallback
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]+", "", cand.lower())
        for nk, original in norm.items():
            if key in nk or nk in key:
                return original

    if required:
        raise KeyError(f"Could not detect any of columns {candidates}. Available: {list(df.columns)}")
    return None


def read_tap_layout() -> pd.DataFrame:
    print("Reading tap layout and building coordinate mapping...")
    print(f"Using tap layout: {TAP_LAYOUT_FILE}")

    df = pd.read_excel(TAP_LAYOUT_FILE)

    surf_col = detect_column(df, ["Surface", "surface", "Face", "face"])
    x_col = detect_column(df, ["X", "x", "Xcoord", "xcoord"])
    y_col = detect_column(df, ["Y", "y", "Ycoord", "ycoord"])
    z_col = detect_column(df, ["Z", "z", "Zcoord", "zcoord"])
    tap_col = detect_column(df, ["Tap", "tap", "TapID", "tap_id", "ID", "Channel"], required=False)

    print(f"Detected columns: surface={surf_col}, x={x_col}, y={y_col}, z={z_col}, tap={tap_col}")

    out = pd.DataFrame({
        "layout_row": np.arange(len(df)),
        "surface": df[surf_col],
        "x_wt_raw": pd.to_numeric(df[x_col], errors="coerce"),
        "y_wt_raw": pd.to_numeric(df[y_col], errors="coerce"),
        "z_wt_raw": pd.to_numeric(df[z_col], errors="coerce"),
    })

    if tap_col is not None:
        out["tap"] = pd.to_numeric(df[tap_col], errors="coerce")
    else:
        out["tap"] = np.nan

    out["surface"] = out["surface"].astype(str).str.strip()
    out = out.dropna(subset=["x_wt_raw", "y_wt_raw", "z_wt_raw"]).copy()

    return out


def prepare_layout_for_cp_columns(layout: pd.DataFrame, n_cp_cols: int, diag_dir: Path) -> pd.DataFrame:
    """
    Critical fix:
    Use row order of valid tap layout rows to assign Cp columns.
    Do not use tap labels as Python column indices.
    """
    out = layout.copy()

    # Remove explicitly invalid taps.
    if INVALID_TAP_IDS:
        out = out[~out["tap"].isin(INVALID_TAP_IDS)].copy()

    # Remove obvious non-working/placeholder rows if present.
    tap_numeric = pd.to_numeric(out["tap"], errors="coerce")
    obvious_invalid = tap_numeric.isna() | (tap_numeric <= 0)
    if len(out) > n_cp_cols and obvious_invalid.any():
        removed = out[obvious_invalid].copy()
        removed.to_csv(diag_dir / "tap_layout_removed_invalid_or_nonpositive_taps.csv", index=False)
        out = out[~obvious_invalid].copy()

    # If there is still exactly one too many row, try removing tap 0 if it exists
    # even if not caught above due string formatting.
    if len(out) == n_cp_cols + 1:
        tap_numeric = pd.to_numeric(out["tap"], errors="coerce")
        zero_rows = out[tap_numeric == 0]
        if len(zero_rows) == 1:
            zero_rows.to_csv(diag_dir / "tap_layout_removed_single_tap0_row.csv", index=False)
            out = out[tap_numeric != 0].copy()

    # Final fallback.
    if len(out) > n_cp_cols:
        msg = (
            f"Tap layout still has {len(out)} valid coordinate rows, but Cp file has "
            f"{n_cp_cols} columns. "
        )
        extra = out.iloc[n_cp_cols:].copy()
        extra.to_csv(diag_dir / "tap_layout_rows_beyond_cp_columns.csv", index=False)

        if ALLOW_TRUNCATE_LAYOUT_TO_N_CP_COLUMNS:
            warnings.warn(
                msg + f"Keeping first {n_cp_cols} rows by layout order. "
                "Check tap_layout_rows_beyond_cp_columns.csv.",
                RuntimeWarning,
            )
            out = out.iloc[:n_cp_cols].copy()
        else:
            raise ValueError(
                msg + "Set ALLOW_TRUNCATE_LAYOUT_TO_N_CP_COLUMNS=True only if row order is confirmed."
            )

    if len(out) < n_cp_cols:
        warnings.warn(
            f"Tap layout has only {len(out)} rows after filtering, but Cp has {n_cp_cols} columns. "
            "Only mapped layout rows will be compared.",
            RuntimeWarning,
        )

    out = out.reset_index(drop=True)
    out["cp_col"] = np.arange(len(out), dtype=int)

    raw_min = int(out["cp_col"].min()) if len(out) else -1
    raw_max = int(out["cp_col"].max()) if len(out) else -1
    if raw_min < 0 or raw_max >= n_cp_cols:
        raise ValueError(
            f"Internal mapping error: assigned cp_col range {raw_min}..{raw_max}, "
            f"but Cp columns are 0..{n_cp_cols - 1}."
        )

    out.to_csv(diag_dir / "tap_layout_valid_with_assigned_cp_columns.csv", index=False)
    return out


def maybe_scale_wt_coords(layout: pd.DataFrame, les_coords: np.ndarray) -> Tuple[pd.DataFrame, float, np.ndarray]:
    out = layout.copy()
    wt_xyz_raw = out[["x_wt_raw", "y_wt_raw", "z_wt_raw"]].to_numpy(float)
    les_xyz = np.asarray(les_coords, dtype=float)

    scale = 1.0
    if AUTO_SCALE_WT_COORDS:
        wt_range = np.nanmax(wt_xyz_raw, axis=0) - np.nanmin(wt_xyz_raw, axis=0)
        les_range = np.nanmax(les_xyz, axis=0) - np.nanmin(les_xyz, axis=0)
        wt_span = np.nanmax(wt_range)
        les_span = np.nanmax(les_range)

        if np.isfinite(wt_span) and np.isfinite(les_span) and les_span > 0:
            ratio = wt_span / les_span
            if ratio > 50:
                scale = 0.001
            elif ratio < 0.02:
                scale = 1000.0

    wt_xyz = wt_xyz_raw * scale
    translation = np.zeros(3, dtype=float)
    if AUTO_TRANSLATE_WT_COORDS_TO_LES_BOUNDS:
        # Align lower bounds in x/y/z. This fixes a pure origin offset without
        # changing the WT tap spacings. If axes are permuted/sign-flipped, the
        # diagnostics will still show large distances and you should disable this
        # and provide the coordinate transform explicitly.
        translation = np.nanmin(les_xyz, axis=0) - np.nanmin(wt_xyz, axis=0)
        wt_xyz = wt_xyz + translation[None, :]

    out["x_wt"] = wt_xyz[:, 0]
    out["y_wt"] = wt_xyz[:, 1]
    out["z_wt"] = wt_xyz[:, 2]
    out["wt_coord_scale_applied"] = scale
    out["wt_x_translation_applied"] = translation[0]
    out["wt_y_translation_applied"] = translation[1]
    out["wt_z_translation_applied"] = translation[2]
    return out, scale, translation




def _surface_key_for_mapping(value) -> str:
    m = re.search(r"(\d+)", str(value))
    return m.group(1) if m else str(value).strip().lower()


def apply_wt_mapping_mirrors(layout_scaled: pd.DataFrame, diag_dir: Path) -> pd.DataFrame:
    """Mirror selected WT surfaces before nearest-neighbour mapping.

    This is intended for cases where the tap-layout coordinates are correct as a
    face geometry, but the Scanivalve Cp columns for that face are ordered in the
    opposite horizontal direction. For each selected surface, the function finds
    the horizontal coordinate with the largest spread (x or y) and reflects it
    about the face midline. The z coordinate is never mirrored.
    """
    out = layout_scaled.copy()
    mirror_keys = {str(k) for k in MIRROR_WT_MAPPING_SURFACES}
    if not mirror_keys:
        out["wt_mapping_mirrored"] = False
        out["wt_mapping_mirror_axis"] = ""
        return out

    out["wt_mapping_mirrored"] = False
    out["wt_mapping_mirror_axis"] = ""
    surf_keys = out["surface"].map(_surface_key_for_mapping).astype(str)

    rows = []
    for surf in sorted(mirror_keys):
        m = (surf_keys == surf).to_numpy()
        if not np.any(m):
            continue
        ranges = {}
        for col in ["x_wt", "y_wt"]:
            vals = pd.to_numeric(out.loc[m, col], errors="coerce").to_numpy(float)
            ranges[col] = np.nanmax(vals) - np.nanmin(vals) if np.any(np.isfinite(vals)) else 0.0
        axis = max(ranges, key=ranges.get)
        if ranges[axis] <= 1e-12:
            continue
        vals = pd.to_numeric(out.loc[m, axis], errors="coerce").to_numpy(float)
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
        out.loc[m, axis] = lo + hi - vals
        out.loc[m, "wt_mapping_mirrored"] = True
        out.loc[m, "wt_mapping_mirror_axis"] = axis
        rows.append({"surface": surf, "mirror_axis": axis, "min_before": lo, "max_before": hi})

    if rows:
        pd.DataFrame(rows).to_csv(diag_dir / "wt_mapping_mirrored_surfaces.csv", index=False)
        print("WT Cp-column mapping mirror applied to surfaces: " + ", ".join(f"{r['surface']} about {r['mirror_axis']}" for r in rows))
    return out


def build_les_to_wt_mapping(
    les_stats: pd.DataFrame,
    layout_valid: pd.DataFrame,
    diag_dir: Path,
) -> pd.DataFrame:
    """
    Map each LES probe point to nearest WT tap coordinate.
    This uses coordinates for spatial matching and cp_col for Cp data indexing.
    """
    layout_scaled, scale, translation = maybe_scale_wt_coords(
        layout_valid,
        les_stats[["x_les", "y_les", "z_les"]].to_numpy(float),
    )

    layout_scaled = apply_wt_mapping_mirrors(layout_scaled, diag_dir)

    wt_xyz_all = layout_scaled[["x_wt", "y_wt", "z_wt"]].to_numpy(float)
    les_xyz = les_stats[["x_les", "y_les", "z_les"]].to_numpy(float)

    layout_surface_key = layout_scaled["surface"].map(_surface_key_for_mapping).to_numpy()
    les_surface_key = les_stats["probe_source"].map(_surface_key_for_mapping).to_numpy()

    rows = []
    for i, xyz in enumerate(les_xyz):
        if not np.all(np.isfinite(xyz)):
            rows.append({
                "les_index": i,
                "cp_col": np.nan,
                "tap": np.nan,
                "surface": "",
                "distance": np.nan,
                "x_les": xyz[0],
                "y_les": xyz[1],
                "z_les": xyz[2],
                "x_wt": np.nan,
                "y_wt": np.nan,
                "z_wt": np.nan,
            })
            continue

        candidates = np.arange(len(layout_scaled))
        if SURFACE_RESTRICTED_MAPPING:
            same_surface = candidates[layout_surface_key == les_surface_key[i]]
            if len(same_surface) > 0:
                candidates = same_surface

        wt_xyz = wt_xyz_all[candidates, :]
        d = np.linalg.norm(wt_xyz - xyz[None, :], axis=1)
        j_local = int(np.nanargmin(d))
        j = int(candidates[j_local])
        tap_row = layout_scaled.iloc[j]

        rows.append({
            "les_index": i,
            "cp_col": int(tap_row["cp_col"]),
            "tap": tap_row["tap"],
            "surface": tap_row["surface"],
            "distance": float(d[j_local]),
            "x_les": xyz[0],
            "y_les": xyz[1],
            "z_les": xyz[2],
            "x_wt": tap_row["x_wt"],
            "y_wt": tap_row["y_wt"],
            "z_wt": tap_row["z_wt"],
            "probe_source": les_stats.loc[i, "probe_source"],
            "wt_mapping_mirrored": bool(tap_row.get("wt_mapping_mirrored", False)),
            "wt_mapping_mirror_axis": tap_row.get("wt_mapping_mirror_axis", ""),
        })

    mapping = pd.DataFrame(rows)

    mapping["wt_coord_scale_applied"] = scale
    mapping["wt_x_translation_applied"] = translation[0]
    mapping["wt_y_translation_applied"] = translation[1]
    mapping["wt_z_translation_applied"] = translation[2]
    mapping.to_csv(diag_dir / "les_to_wt_tap_mapping.csv", index=False)

    # Diagnostics to catch duplicated mapping and bad alignment.
    dup = mapping[mapping.duplicated("cp_col", keep=False)].sort_values("cp_col")
    dup.to_csv(diag_dir / "diagnostic_duplicate_cp_columns_in_mapping.csv", index=False)

    far = mapping.sort_values("distance", ascending=False).head(50)
    far.to_csv(diag_dir / "diagnostic_largest_mapping_distances.csv", index=False)

    print(f"WT coordinate scale applied: {scale}")
    print(f"WT coordinate translation applied: dx={translation[0]:.6g}, dy={translation[1]:.6g}, dz={translation[2]:.6g}")
    print(f"Mapping distance: min={mapping['distance'].min():.6g}, "
          f"median={mapping['distance'].median():.6g}, max={mapping['distance'].max():.6g}")
    if len(dup):
        print(f"WARNING: {dup['cp_col'].nunique()} WT Cp columns are mapped by more than one LES probe.")
        print("         See diagnostic_duplicate_cp_columns_in_mapping.csv")

    return mapping


# =============================================================================
# WT statistics, moving average, comparisons
# =============================================================================

def select_time_record(cp: np.ndarray, fs: float, start: float, duration: float) -> np.ndarray:
    i0 = max(0, int(round(start * fs)))
    n = max(1, int(round(duration * fs)))
    i1 = min(cp.shape[0], i0 + n)
    return cp[i0:i1, :]


def moving_average_2d(x: np.ndarray, window_samples: int) -> np.ndarray:
    if window_samples <= 1:
        return x
    if window_samples >= x.shape[0]:
        # Return one average sample if window is longer than record.
        return np.nanmean(x, axis=0, keepdims=True)

    # Efficient trailing moving average with valid length n-window+1.
    x64 = np.asarray(x, dtype=np.float64)
    cs = np.cumsum(np.vstack([np.zeros((1, x64.shape[1])), x64]), axis=0)
    return (cs[window_samples:] - cs[:-window_samples]) / float(window_samples)


def cp_stats_for_columns(cp: np.ndarray, cols: np.ndarray) -> pd.DataFrame:
    """
    Memory-safe statistics for selected Cp columns. Duplicate requested columns
    are handled by computing each unique WT column once and expanding back to
    the requested LES-probe order.
    """
    cols = np.asarray(cols, dtype=int)
    unique_cols, inverse = np.unique(cols, return_inverse=True)
    acc = init_stats_acc(len(unique_cols))
    n = cp.shape[0]
    for i0 in range(0, n, WT_STATS_CHUNK_ROWS):
        # Fancy indexing only creates a chunk-sized temporary, not the full file.
        chunk = cp[i0:i0 + WT_STATS_CHUNK_ROWS, :][:, unique_cols]
        update_stats_acc(acc, chunk)
    unique_stats = finalize_stats_acc(acc)
    return unique_stats.iloc[inverse].reset_index(drop=True)


def update_cp_stats_acc_for_columns(acc: Dict[str, np.ndarray], cp: np.ndarray, cols: np.ndarray) -> None:
    """Update an existing accumulator for one Cp record and selected columns."""
    cols = np.asarray(cols, dtype=int)
    n = cp.shape[0]
    for i0 in range(0, n, WT_STATS_CHUNK_ROWS):
        chunk = cp[i0:i0 + WT_STATS_CHUNK_ROWS, :][:, cols]
        update_stats_acc(acc, chunk)


def assemble_pointwise_comparison(
    les_stats: pd.DataFrame,
    mapping: pd.DataFrame,
    wt_stats: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    li = mapping["les_index"].to_numpy(int)
    point = pd.DataFrame({
        "comparison": label,
        "les_index": li,
        "cp_col": mapping["cp_col"].to_numpy(int),
        "tap": mapping["tap"].to_numpy(),
        "surface": mapping["surface"].astype(str).to_numpy(),
        "face_label": mapping["surface"].astype(str).map(face_label).to_numpy(),
        "map_distance": mapping["distance"].to_numpy(float),

        # Keep both coordinate systems in the pointwise table.
        # Contour plots use LES coordinates by default, with WT coordinates
        # available as a cross-check in the CSV output.
        "x_les": mapping["x_les"].to_numpy(float),
        "y_les": mapping["y_les"].to_numpy(float),
        "z_les": mapping["z_les"].to_numpy(float),
        "x_wt": mapping["x_wt"].to_numpy(float),
        "y_wt": mapping["y_wt"].to_numpy(float),
        "z_wt": mapping["z_wt"].to_numpy(float),

        "Cp_mean_LES": les_stats.loc[li, "mean"].to_numpy(float),
        "Cp_rms_LES": les_stats.loc[li, "rms"].to_numpy(float),
        "Cp_std_LES": les_stats.loc[li, "std"].to_numpy(float),
        "Cp_min_LES": les_stats.loc[li, "min"].to_numpy(float),
        "Cp_max_LES": les_stats.loc[li, "max"].to_numpy(float),

        "Cp_mean_WT": wt_stats["mean"].to_numpy(float),
        "Cp_rms_WT": wt_stats["rms"].to_numpy(float),
        "Cp_std_WT": wt_stats["std"].to_numpy(float),
        "Cp_min_WT": wt_stats["min"].to_numpy(float),
        "Cp_max_WT": wt_stats["max"].to_numpy(float),
    })

    for stat in ["mean", "rms", "std", "min", "max"]:
        point[f"Cp_{stat}_diff_LES_minus_WT"] = point[f"Cp_{stat}_LES"] - point[f"Cp_{stat}_WT"]

    return point


def metrics_from_pointwise(point: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for stat in ["mean", "rms", "std", "min", "max"]:
        x = point[f"Cp_{stat}_WT"].to_numpy(float)
        y = point[f"Cp_{stat}_LES"].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        if not np.any(m):
            continue
        err = y[m] - x[m]
        corr = np.corrcoef(x[m], y[m])[0, 1] if np.count_nonzero(m) > 1 else np.nan
        rows.append({
            "comparison": label,
            "stat": stat,
            "n": int(np.count_nonzero(m)),
            "bias_LES_minus_WT": float(np.nanmean(err)),
            "mae": float(np.nanmean(np.abs(err))),
            "rmse": float(np.sqrt(np.nanmean(err ** 2))),
            "corr": float(corr),
            "LES_mean_of_stat": float(np.nanmean(y[m])),
            "WT_mean_of_stat": float(np.nanmean(x[m])),
        })
    return pd.DataFrame(rows)


def canonical_surface(value) -> str:
    text = str(value).strip()
    try:
        f = float(text)
        if abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return str(f)
    except Exception:
        m = re.search(r"(\d+(?:\.\d+)?)", text)
        if m:
            return canonical_surface(m.group(1))
        return text.lower()


def face_label(surface) -> str:
    key = canonical_surface(surface)
    return FACE_LABEL_BY_SURFACE.get(key, FACE_LABEL_BY_SURFACE.get(str(surface), "Face"))


def plot_surface_mask(point: pd.DataFrame) -> np.ndarray:
    keys = point["surface"].map(canonical_surface).astype(str).str.lower()
    excluded = {canonical_surface(s).lower() for s in EXCLUDE_SURFACES_FROM_PLOTS}
    return ~keys.isin(excluded).to_numpy()


def surface_color_map(surfaces: Sequence[str]) -> Dict[str, str]:
    unique = list(pd.Series(surfaces).astype(str).map(canonical_surface).dropna().unique())
    cmap = plt.get_cmap("tab10")
    return {s: cmap(i % 10) for i, s in enumerate(unique)}




def equal_axis_limits(x, y, pad_fraction: float = 0.05) -> Tuple[float, float]:
    """Return symmetric/equal plot limits for scatter diagnostics when no fixed limits are configured."""
    vals = np.concatenate([np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -1.0, 1.0
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return -1.0, 1.0
    if abs(hi - lo) < 1e-12:
        pad = max(abs(hi), 1.0) * 0.1
        return lo - pad, hi + pad
    pad = (hi - lo) * pad_fraction
    return lo - pad, hi + pad

def scatter_plot(point: pd.DataFrame, stat: str, label: str, out_dir: Path) -> None:
    mask = plot_surface_mask(point)
    point_plot = point.loc[mask].copy()
    if point_plot.empty:
        return

    x = point_plot[f"Cp_{stat}_WT"].to_numpy(float)
    y = point_plot[f"Cp_{stat}_LES"].to_numpy(float)
    surfaces = point_plot["surface"].map(canonical_surface).astype(str).to_numpy()
    colors = surface_color_map(surfaces)

    fig, ax = plt.subplots(figsize=(8.4, 7.6))

    for surf, color in colors.items():
        idx = surfaces == surf
        label_text = f"Surface {surf} ({face_label(surf)})"
        ax.scatter(
            x[idx],
            y[idx],
            s=SCATTER_SIZE,
            facecolors="white",
            edgecolors=color,
            linewidths=SCATTER_LINEWIDTH,
            label=label_text,
            alpha=0.95,
        )

    # Do not use dict.get(..., equal_axis_limits(...)) here because Python
    # evaluates the default argument even when the fixed limit exists.
    if stat in SCATTER_AXIS_LIMITS:
        lo, hi = SCATTER_AXIS_LIMITS[stat]
    else:
        lo, hi = equal_axis_limits(x, y)
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.4, label="1:1")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.28)

    stat_label = {
        "mean": r"$\overline{C_p}$",
        "rms": r"$C_{p,\mathrm{RMS}}$",
        "std": r"$\sigma_{C_p}$",
        "min": r"$C_{p,\min}$",
        "max": r"$C_{p,\max}$",
    }[stat]

    ax.set_xlabel(f"Wind tunnel {stat_label}", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(f"LES {stat_label}", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(label, fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.legend(fontsize=8.5, loc="best", frameon=True)

    fig.tight_layout()
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    fig.savefig(out_dir / f"scatter_{stat}_{safe_label}.png", dpi=FIG_DPI)
    plt.close(fig)



def _coord_columns_available(df: pd.DataFrame) -> Tuple[str, str, str, str]:
    """Return coordinate prefix and x/y/z column names, preferring LES coordinates."""
    if all(c in df.columns for c in ["x_les", "y_les", "z_les"]):
        return "les", "x_les", "y_les", "z_les"
    if all(c in df.columns for c in ["x_wt", "y_wt", "z_wt"]):
        return "wt", "x_wt", "y_wt", "z_wt"
    raise KeyError(
        "Contour plotting requires coordinate columns. Expected either "
        "x_les/y_les/z_les or x_wt/y_wt/z_wt in the pointwise table."
    )


def _choose_vertical_face_axes(df: pd.DataFrame) -> Tuple[str, str]:
    """Choose face plotting axes with z on the vertical plot axis whenever possible.

    For wall faces this makes the plot orientation intuitive: horizontal axis is the
    varying plan coordinate and vertical axis is physical height z. If z has almost
    no range, fall back to the two most-varying coordinates.
    """
    _, xcol, ycol, zcol = _coord_columns_available(df)
    ranges = {}
    for c in [xcol, ycol, zcol]:
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        ranges[c] = np.nanmax(vals) - np.nanmin(vals) if np.any(np.isfinite(vals)) else 0.0

    if ranges.get(zcol, 0.0) > 1e-12:
        horizontal_candidates = [xcol, ycol]
        horizontal = max(horizontal_candidates, key=lambda c: ranges.get(c, 0.0))
        return horizontal, zcol

    axes = sorted([xcol, ycol, zcol], key=lambda c: ranges.get(c, 0.0), reverse=True)[:2]
    return axes[0], axes[1]


def _levels_for_stat(stat: str) -> np.ndarray:
    vmin, vmax = CONTOUR_VALUE_LIMITS.get(stat, SCATTER_AXIS_LIMITS.get(stat, (-1.0, 1.0)))
    step = CONTOUR_VALUE_STEP.get(stat, 0.1)
    return np.round(np.arange(vmin, vmax + 0.5 * step, step), 10)


def _discrete_norm(levels: np.ndarray) -> BoundaryNorm:
    return BoundaryNorm(levels, ncolors=256, clip=True)


def _percent_error(les: np.ndarray, wt: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.abs(wt), PERCENT_ERROR_DENOM_FLOOR)
    return 100.0 * (les - wt) / denom


def _percent_error_levels(values: Sequence[np.ndarray]) -> np.ndarray:
    finite_chunks = []
    for arr in values:
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if a.size:
            finite_chunks.append(a)
    if not finite_chunks:
        lim = PERCENT_ERROR_MIN_LIMIT
    else:
        all_vals = np.concatenate(finite_chunks)
        lim = float(np.nanpercentile(np.abs(all_vals), PERCENT_ERROR_CLIP_PERCENTILE))
        lim = max(PERCENT_ERROR_MIN_LIMIT, lim)
        lim = min(PERCENT_ERROR_MAX_LIMIT, lim)
    lim = math.ceil(lim / PERCENT_ERROR_LEVEL_STEP) * PERCENT_ERROR_LEVEL_STEP
    return np.arange(-lim, lim + 0.5 * PERCENT_ERROR_LEVEL_STEP, PERCENT_ERROR_LEVEL_STEP)


def _plot_one_contour_panel(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: str,
    levels: np.ndarray,
    cmap: str,
):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    ax.set_title(title, fontsize=11)
    if len(z) < 3:
        ax.text(0.5, 0.5, "Too few points", ha="center", va="center", transform=ax.transAxes)
        return None

    norm = _discrete_norm(levels)
    try:
        cf = ax.tricontourf(x, y, z, levels=levels, cmap=cmap, norm=norm, extend="both")
        ax.tricontour(x, y, z, levels=levels[::max(1, len(levels)//10)], colors="k", linewidths=0.22, alpha=0.28)
    except Exception:
        cf = ax.scatter(x, y, c=z, s=34, cmap=cmap, norm=norm)

    ax.scatter(x, y, s=8, c="k", alpha=0.25, linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", labelsize=8)
    return cf


def contour_plots_all_faces(point: pd.DataFrame, stat: str, label: str, out_dir: Path) -> None:
    """Write one large contour figure per comparison/stat with all non-roof faces.

    Rows are building faces/surfaces; columns are WT, LES, and percentage error
    relative to WT. The physical z coordinate is always used as the plot vertical
    axis for wall faces.
    """
    point_plot = point.loc[plot_surface_mask(point)].copy()
    if point_plot.empty:
        return

    stat_label = {
        "mean": r"$\overline{C_p}$",
        "rms": r"$C_{p,\mathrm{RMS}}$",
    }.get(stat, stat)

    surf_keys = list(point_plot["surface"].map(canonical_surface).dropna().unique())
    surf_keys = sorted(surf_keys, key=lambda s: float(s) if str(s).replace('.', '', 1).isdigit() else str(s))
    if not surf_keys:
        return

    value_levels = _levels_for_stat(stat)
    value_cmap = "RdBu_r" if stat == "mean" else "viridis"

    # Precompute percent errors so every face in this figure uses the same symmetric scale.
    pe_by_surface: Dict[str, np.ndarray] = {}
    for surf_key in surf_keys:
        face_df = point_plot.loc[point_plot["surface"].map(canonical_surface) == surf_key]
        wt = face_df[f"Cp_{stat}_WT"].to_numpy(float)
        les = face_df[f"Cp_{stat}_LES"].to_numpy(float)
        pe_by_surface[surf_key] = _percent_error(les, wt)
    pe_levels = _percent_error_levels(list(pe_by_surface.values()))

    nrows = len(surf_keys)
    fig_h = max(4.0, 3.2 * nrows)
    fig, axs = plt.subplots(nrows, 3, figsize=(16.5, fig_h), squeeze=False, constrained_layout=True)

    value_mappable = None
    pe_mappable = None

    for row, surf_key in enumerate(surf_keys):
        face_df = point_plot.loc[point_plot["surface"].map(canonical_surface) == surf_key].copy()
        ax1_name, ax2_name = _choose_vertical_face_axes(face_df)
        x = pd.to_numeric(face_df[ax1_name], errors="coerce").to_numpy(float)
        y = pd.to_numeric(face_df[ax2_name], errors="coerce").to_numpy(float)
        wt = face_df[f"Cp_{stat}_WT"].to_numpy(float)
        les = face_df[f"Cp_{stat}_LES"].to_numpy(float)
        pe = pe_by_surface[surf_key]

        row_label = f"Surface {surf_key}\n{face_label(surf_key)}"
        titles = [f"{row_label}\nWT", f"{row_label}\nLES", f"{row_label}\n% error vs WT"]

        cf0 = _plot_one_contour_panel(axs[row, 0], x, y, wt, titles[0], value_levels, value_cmap)
        cf1 = _plot_one_contour_panel(axs[row, 1], x, y, les, titles[1], value_levels, value_cmap)
        cf2 = _plot_one_contour_panel(axs[row, 2], x, y, pe, titles[2], pe_levels, "RdBu_r")

        value_mappable = cf1 or cf0 or value_mappable
        pe_mappable = cf2 or pe_mappable

        xlab = ax1_name.replace("_les", "").replace("_wt", "")
        ylab = ax2_name.replace("_les", "").replace("_wt", "")
        for col in range(3):
            axs[row, col].set_xlabel(xlab, fontsize=9)
            axs[row, col].set_ylabel(ylab, fontsize=9)

    # Discrete colour bars. Mean/RMS share one scale for WT and LES; error has its own symmetric scale.
    if value_mappable is not None:
        cbar = fig.colorbar(value_mappable, ax=axs[:, :2], shrink=0.92, pad=0.012)
        cbar.set_label(stat_label, fontsize=11)
        # Avoid overcrowding mean colourbar labels while preserving 0.1 bins.
        tick_step = 0.5 if stat == "mean" else 0.1
        vmin, vmax = value_levels[0], value_levels[-1]
        cbar.set_ticks(np.arange(vmin, vmax + 0.5 * tick_step, tick_step))
    if pe_mappable is not None:
        cbar = fig.colorbar(pe_mappable, ax=axs[:, 2], shrink=0.92, pad=0.012)
        cbar.set_label(r"$100(LES-WT)/|WT|$ [%]", fontsize=11)
        lim = max(abs(pe_levels[0]), abs(pe_levels[-1]))
        tick_step = max(PERCENT_ERROR_LEVEL_STEP, 25.0 if lim > 75 else 10.0)
        cbar.set_ticks(np.arange(-lim, lim + 0.5 * tick_step, tick_step))

    fig.suptitle(f"{label} | {stat_label} contours, all non-roof faces", fontsize=14)
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    fig.savefig(out_dir / f"contours_all_faces_{stat}_{safe_label}.png", dpi=FIG_DPI)
    plt.close(fig)



def contour_plots_by_face(point: pd.DataFrame, stat: str, label: str, out_dir: Path) -> None:
    """Write one contour figure per face/surface.

    Each figure has WT, LES, and percentage-error panels. This is easier to
    inspect than a single very large all-face figure and keeps the physical z
    axis vertical for wall surfaces.
    """
    point_plot = point.loc[plot_surface_mask(point)].copy()
    if point_plot.empty:
        return

    stat_label = {
        "mean": r"$\overline{C_p}$",
        "rms": r"$C_{p,\mathrm{RMS}}$",
    }.get(stat, stat)

    surf_keys = list(point_plot["surface"].map(canonical_surface).dropna().unique())
    surf_keys = sorted(surf_keys, key=lambda s: float(s) if str(s).replace('.', '', 1).isdigit() else str(s))
    if not surf_keys:
        return

    value_levels = _levels_for_stat(stat)
    value_cmap = "RdBu_r" if stat == "mean" else "viridis"

    for surf_key in surf_keys:
        face_df = point_plot.loc[point_plot["surface"].map(canonical_surface) == surf_key].copy()
        if face_df.empty:
            continue

        ax1_name, ax2_name = _choose_vertical_face_axes(face_df)
        x = pd.to_numeric(face_df[ax1_name], errors="coerce").to_numpy(float)
        y = pd.to_numeric(face_df[ax2_name], errors="coerce").to_numpy(float)
        wt = face_df[f"Cp_{stat}_WT"].to_numpy(float)
        les = face_df[f"Cp_{stat}_LES"].to_numpy(float)
        pe = _percent_error(les, wt)
        pe_levels = _percent_error_levels([pe])

        fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.8), squeeze=False, constrained_layout=True)
        axs = axs[0]

        row_label = f"Surface {surf_key} — {face_label(surf_key)}"
        cf0 = _plot_one_contour_panel(axs[0], x, y, wt, f"WT\n{row_label}", value_levels, value_cmap)
        cf1 = _plot_one_contour_panel(axs[1], x, y, les, f"LES\n{row_label}", value_levels, value_cmap)
        cf2 = _plot_one_contour_panel(axs[2], x, y, pe, f"% error vs WT\n{row_label}", pe_levels, "RdBu_r")

        xlab = ax1_name.replace("_les", "").replace("_wt", "")
        ylab = ax2_name.replace("_les", "").replace("_wt", "")
        for ax in axs:
            ax.set_xlabel(xlab, fontsize=10)
            ax.set_ylabel(ylab, fontsize=10)

        value_mappable = cf1 or cf0
        if value_mappable is not None:
            cbar = fig.colorbar(value_mappable, ax=axs[:2], shrink=0.92, pad=0.012)
            cbar.set_label(stat_label, fontsize=11)
            # Mean has 20 bins over -2..2; show readable 0.4 tick labels. RMS keeps 0.1 ticks.
            tick_step = 0.4 if stat == "mean" else 0.1
            vmin, vmax = value_levels[0], value_levels[-1]
            cbar.set_ticks(np.arange(vmin, vmax + 0.5 * tick_step, tick_step))

        if cf2 is not None:
            cbar = fig.colorbar(cf2, ax=axs[2], shrink=0.92, pad=0.012)
            cbar.set_label(r"$100(LES-WT)/|WT|$ [%]", fontsize=11)
            lim = max(abs(pe_levels[0]), abs(pe_levels[-1]))
            tick_step = max(PERCENT_ERROR_LEVEL_STEP, 25.0 if lim > 75 else 10.0)
            cbar.set_ticks(np.arange(-lim, lim + 0.5 * tick_step, tick_step))

        fig.suptitle(f"{label} | {stat_label} contours | {row_label}", fontsize=14)
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
        safe_surf = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(surf_key))
        fig.savefig(out_dir / f"contours_surface{safe_surf}_{stat}_{safe_label}.png", dpi=FIG_DPI)
        plt.close(fig)


def write_all_plots(point: pd.DataFrame, label: str, fig_dir: Path) -> None:
    # Scatter plots: mean and RMS only, with fixed axes requested by the user.
    for stat in PLOT_STATS:
        scatter_plot(point, stat, label, fig_dir)
        contour_plots_by_face(point, stat, label, fig_dir)


# =============================================================================
# Main
# =============================================================================

def run() -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dir(OUT_DIR)
    csv_dir = ensure_dir(OUT_DIR / "csv")
    fig_dir = ensure_dir(OUT_DIR / "figures")
    diag_dir = ensure_dir(OUT_DIR / "diagnostics")

    les_stats, les_cp_df, ref_info, les_coords = read_les_pressure_and_reference()

    # Save LES diagnostics.
    les_stats.to_csv(csv_dir / "les_cp_statistics.csv", index=False)
    pd.DataFrame([ref_info]).to_csv(csv_dir / "les_upstream_reference.csv", index=False)

    files = list_cp_files()

    # Read first file to determine Cp column count and WT sampling rate.
    first = read_scanivalve_cp_file(files[0])
    n_cp_cols = first.cp.shape[1]
    first_fs = first.fs

    layout = read_tap_layout()
    layout_valid = prepare_layout_for_cp_columns(layout, n_cp_cols=n_cp_cols, diag_dir=diag_dir)
    mapping = build_les_to_wt_mapping(les_stats, layout_valid, diag_dir=diag_dir)

    # Drop any LES probes that could not be mapped.
    mapping = mapping.dropna(subset=["cp_col"]).copy()
    mapping["cp_col"] = mapping["cp_col"].astype(int)
    needed_cols = mapping["cp_col"].to_numpy(int)

    if needed_cols.min() < 0 or needed_cols.max() >= n_cp_cols:
        bad = mapping[(mapping["cp_col"] < 0) | (mapping["cp_col"] >= n_cp_cols)]
        bad.to_csv(diag_dir / "bad_mapping_rows_out_of_cp_range.csv", index=False)
        raise ValueError(
            f"Mapping cp_col range {needed_cols.min()}..{needed_cols.max()} is outside "
            f"Cp column range 0..{n_cp_cols - 1}. See diagnostics."
        )

    mapping.to_csv(csv_dir / "les_to_wt_tap_mapping.csv", index=False)

    all_points = []
    all_metrics = []
    meta_rows = []

    les_duration = ref_info["les_duration"]

    # Helper that processes and writes one comparison.
    def process_comparison(label: str, cp_record: np.ndarray, fs: float):
        print(f"Processing comparison: {label}  [samples={cp_record.shape[0]}, taps={cp_record.shape[1]}]")
        wt_stats = cp_stats_for_columns(cp_record, needed_cols)
        point = assemble_pointwise_comparison(les_stats, mapping, wt_stats, label)
        metrics = metrics_from_pointwise(point, label)

        point.to_csv(csv_dir / f"pointwise_{re.sub(r'[^A-Za-z0-9_.-]+', '_', label)}.csv", index=False)
        write_all_plots(point, label, fig_dir)

        all_points.append(point)
        all_metrics.append(metrics)

    # Full all-repetition aggregate. This is now streaming/memory-safe: do not
    # concatenate all five full records into one huge array.
    print("Processing full WT record, all repetitions combined...")

    full_all_acc = init_stats_acc(len(needed_cols))
    matched_all_acc = init_stats_acc(len(needed_cols))

    # Use already-loaded first file to avoid loading it twice.
    loaded_first_used = False

    for f in files:
        if not loaded_first_used and f == first.path:
            wt = first
            loaded_first_used = True
        else:
            wt = read_scanivalve_cp_file(f)

        if wt.cp.shape[1] != n_cp_cols:
            raise ValueError(f"{wt.path.name}: expected {n_cp_cols} Cp columns, found {wt.cp.shape[1]}.")

        meta_rows.append({
            "file": wt.path.name,
            "rep": wt.rep_label,
            "samples": wt.cp.shape[0],
            "n_taps": wt.cp.shape[1],
            "sampling_freq": wt.fs,
            "duration": wt.duration,
        })

        update_cp_stats_acc_for_columns(full_all_acc, wt.cp, needed_cols)
        matched = select_time_record(wt.cp, wt.fs, WT_MATCH_START_TIME, les_duration)
        update_cp_stats_acc_for_columns(matched_all_acc, matched, needed_cols)

    wt_meta = pd.DataFrame(meta_rows)
    wt_meta.to_csv(csv_dir / "wind_tunnel_file_metadata.csv", index=False)

    label = "WT_full_all_REP1-REP5"
    print(f"Processing comparison: {label}  [streamed all full records]")
    wt_stats = finalize_stats_acc(full_all_acc)
    point = assemble_pointwise_comparison(les_stats, mapping, wt_stats, label)
    metrics = metrics_from_pointwise(point, label)
    point.to_csv(csv_dir / "pointwise_WT_full_all_REP1-REP5.csv", index=False)
    write_all_plots(point, label, fig_dir)
    all_points.append(point)
    all_metrics.append(metrics)

    label = "WT_matchedDuration_all_REP1-REP5"
    print(f"Processing comparison: {label}  [streamed matched-duration records]")
    wt_stats = finalize_stats_acc(matched_all_acc)
    point = assemble_pointwise_comparison(les_stats, mapping, wt_stats, label)
    metrics = metrics_from_pointwise(point, label)
    point.to_csv(csv_dir / "pointwise_WT_matchedDuration_all_REP1-REP5.csv", index=False)
    write_all_plots(point, label, fig_dir)
    all_points.append(point)
    all_metrics.append(metrics)

    # Per repetition: full duration and LES-matched duration only. No moving averages.
    loaded_first_used = False
    for f in files:
        if not loaded_first_used and f == first.path:
            wt = first
            loaded_first_used = True
        else:
            wt = read_scanivalve_cp_file(f)

        rep = wt.rep_label

        # Full WT repetition.
        process_comparison(f"WT_full_{rep}", wt.cp, wt.fs)

        # Same-duration WT record as the LES window.
        matched = select_time_record(wt.cp, wt.fs, WT_MATCH_START_TIME, les_duration)
        process_comparison(f"WT_matchedDuration_{rep}", matched, wt.fs)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    point_df = pd.concat(all_points, ignore_index=True)

    metrics_df.to_csv(csv_dir / "summary_metrics_all_comparisons.csv", index=False)
    point_df.to_csv(csv_dir / "pointwise_all_comparisons.csv", index=False)

    print("\nDone.")
    print(f"Outputs written to: {OUT_DIR}")
    print(f"Diagnostics written to: {diag_dir}")
    print("Most important diagnostic files:")
    print(f"  {diag_dir / 'tap_layout_valid_with_assigned_cp_columns.csv'}")
    print(f"  {diag_dir / 'les_to_wt_tap_mapping.csv'}")
    print(f"  {diag_dir / 'diagnostic_largest_mapping_distances.csv'}")
    print(f"  {csv_dir / 'summary_metrics_all_comparisons.csv'}")

    return metrics_df, point_df


if __name__ == "__main__":
    metrics, pointwise = run()
