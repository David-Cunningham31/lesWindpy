# -*- coding: utf-8 -*-
"""
Common utilities for MannHybridTurb downstream spectral calibration recipes.

This module is intentionally self-contained enough to run from a windlespy
_recipes folder on the cluster. It expects CASE_DIR to point to an OpenFOAM case.

Supported MannHybridTurb/windProfile file formats
-------------------------------------------------
profile / targetProfile:
    8 columns:  z U Iu Iv Iw Lu Lv Lw
    9 columns:  z U Iu Iv Iw Lu Lv Lw uwStress

auto spectra profile:
    first line: nHeights nFreq
    each row:   z [uwStress] Su_1 ... Su_N Sv_1 ... Sv_N Sw_1 ... Sw_N

uw co-spectrum profile:
    first line: nHeights nFreq
    each row:   z [uwStress] Cuw_1 ... Cuw_N

The optional uwStress column is preserved/written because the Mann-hybrid
utility and diagnostic scripts use it for consistency with the local co-spectrum.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.interpolate import PchipInterpolator, interp1d
    from scipy.signal import welch, csd
    from scipy.signal.windows import dpss
except Exception as exc:  # pragma: no cover - handled at runtime on cluster
    raise RuntimeError("SciPy is required for the Mann-hybrid calibration scripts") from exc


# -----------------------------------------------------------------------------
# Generic IO helpers
# -----------------------------------------------------------------------------

FLOOR = 1.0e-16


def _trapz(y, x=None, dx=1.0, axis=-1):
    """Version-safe trapezoidal integration for NumPy 1.x/2.x.

    NumPy 2.x introduced ``np.trapezoid``; older cluster/Anaconda
    environments may only have ``np.trapz``. This helper keeps all recipes
    portable. Do not call ``_trapz`` recursively from here.
    """
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x, dx=dx, axis=axis)
    return np.trapz(y, x=x, dx=dx, axis=axis)


def _windows_long_path(path: str | os.PathLike) -> str:
    path = os.path.abspath(os.fspath(path))
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


def safe_makedirs(path: str | os.PathLike) -> None:
    os.makedirs(_windows_long_path(path), exist_ok=True)


def safe_savefig(fig: plt.Figure, path: str | os.PathLike, dpi: int = 300) -> None:
    path = os.path.abspath(os.fspath(path))
    safe_makedirs(os.path.dirname(path))
    fig.savefig(_windows_long_path(path), dpi=dpi, bbox_inches="tight")


def parse_openfoam_scalar(path: str | os.PathLike, entry: str, default: Optional[float] = None) -> Optional[float]:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return default
    # Remove // and /* */ comments in a lightweight way.
    text = re.sub(r"//.*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    m = re.search(r"(^|\s)" + re.escape(entry) + r"\s+([-+0-9.eE]+)\s*;", text)
    if not m:
        return default
    try:
        return float(m.group(2))
    except Exception:
        return default


def parse_case_setup(case_dir: str | os.PathLike) -> Dict[str, float]:
    """Try windlespy setup parsing first; fall back to control/dict entries."""
    case_dir = os.path.abspath(os.fspath(case_dir))
    out: Dict[str, float] = {}

    try:
        import windlespy as LES  # type: ignore
        d = LES._caseFiles.parse_setup_file(case_dir)
        for k, v in d.items():
            if isinstance(v, (int, float, np.number)):
                out[k] = float(v)
            else:
                out[k] = v
    except Exception:
        pass

    mh = os.path.join(case_dir, "constant", "MannHybridTurbDict")
    ctrl = os.path.join(case_dir, "system", "controlDict")
    out.setdefault("buildingHeight", parse_openfoam_scalar(mh, "buildingHeight", None) or 0.5)
    out.setdefault("meshSize", parse_openfoam_scalar(mh, "meshSize", None) or 0.03)
    out.setdefault("fMax", parse_openfoam_scalar(mh, "fMax", None) or 200.0)
    # nFreq is the number of positive temporal frequencies in spectra files, not always in dict.
    out.setdefault("lowerZThreshold", 0.2 * float(out.get("buildingHeight", 0.5)))
    out.setdefault("upperZThreshold", 1.5 * float(out.get("buildingHeight", 0.5)))
    out.setdefault("rmseThreshold", 0.05)
    out.setdefault("deltaT_sim", parse_openfoam_scalar(ctrl, "deltaT", None) or 5e-4)
    return out


def load_sim_init(case_dir: str | os.PathLike) -> Dict[str, float]:
    p = Path(case_dir) / "log" / "downstreamCalibration" / "sim_init.json"
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"burn_in_time": 0.0, "initial_sim_duration": parse_openfoam_scalar(Path(case_dir)/"system"/"controlDict", "endTime", 0.0) or 0.0}


def case_paths(case_dir: str | os.PathLike) -> Dict[str, str]:
    case_dir = os.path.abspath(os.fspath(case_dir))
    w = os.path.join(case_dir, "constant", "boundaryData", "windProfile")
    return {
        "case": case_dir,
        "windProfile": w,
        "profile": os.path.join(w, "profile"),
        "targetProfile": os.path.join(w, "targetProfile"),
        "targetSmoothedProfile": os.path.join(w, "targetSmoothedProfile"),
        "targetExperimentalProfile": os.path.join(w, "targetExperimentalProfile"),
        "spectraProfile": os.path.join(w, "spectraProfile"),
        "targetSpectraProfile": os.path.join(w, "targetSpectraProfile"),
        "uwCoSpectrumProfile": os.path.join(w, "uwCoSpectrumProfile"),
        "targetUWCoSpectrumProfile": os.path.join(w, "targetUWCoSpectrumProfile"),
        "probes2": os.path.join(case_dir, "postProcessing", "probes2"),
        "calLog": os.path.join(case_dir, "log", "downstreamCalibration"),
    }



# -----------------------------------------------------------------------------
# OpenFOAM probe reader
# -----------------------------------------------------------------------------

_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _numeric_dir_key(path: Path):
    """Sort OpenFOAM time directories numerically when possible."""
    try:
        return (0, float(path.name), path.name)
    except Exception:
        return (1, 0.0, path.name)


def _probe_vector_files(probe_dir: str | os.PathLike, field_name: str = "U") -> List[Path]:
    """Return OpenFOAM probe vector files from a probes function-object folder.

    Handles the usual layouts:

        postProcessing/probes2/0/U
        postProcessing/probes2/<restartTime>/U
        postProcessing/probes2/U

    The files are sorted by their parent time directory.  The actual time column
    in each file is still used later, so this sorting is only for deterministic
    diagnostics and concatenation before duplicate removal.
    """
    root = Path(probe_dir)
    files: List[Path] = []
    direct = root / field_name
    if direct.is_file():
        files.append(direct)
    files.extend(sorted(root.glob(f"*/{field_name}"), key=lambda q: _numeric_dir_key(q.parent)))
    # Remove accidental duplicates while preserving order.
    seen = set()
    unique: List[Path] = []
    for f in files:
        key = str(f.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def _count_probe_headers(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("# Probe"):
                n += 1
            elif line and not line.startswith("#") and line.strip():
                # Stop once data starts.
                break
    return n


def _read_openfoam_probe_vector_file(path: str | os.PathLike, expected_n_probes: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, int]:
    """Read one OpenFOAM vector probe file.

    Returns
    -------
    vel : ndarray, shape (3, nTime, nProbes)
    time : ndarray, shape (nTime,)
    n_probes : int
    """
    path = Path(path)
    header_n = _count_probe_headers(path)
    n_probes = expected_n_probes or header_n or None

    times: List[float] = []
    rows: List[np.ndarray] = []

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line_no, line in enumerate(fh, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            nums = [float(x) for x in _FLOAT_RE.findall(s)]
            if len(nums) < 4:
                continue
            t = nums[0]
            vals = nums[1:]
            if n_probes is None:
                if len(vals) % 3 != 0:
                    raise ValueError(f"Cannot infer probe count from {path}:{line_no}; found {len(vals)} vector values")
                n_probes = len(vals) // 3
            expected_vals = 3 * int(n_probes)
            if len(vals) != expected_vals:
                raise ValueError(
                    f"Malformed probe row in {path}:{line_no}: expected {expected_vals} vector values "
                    f"for {n_probes} probes, found {len(vals)}"
                )
            arr = np.asarray(vals, dtype=float).reshape(int(n_probes), 3).T
            times.append(float(t))
            rows.append(arr)

    if n_probes is None:
        raise ValueError(f"Could not determine number of probes in {path}")
    if not rows:
        raise ValueError(f"No data rows found in {path}")

    vel = np.stack(rows, axis=1)  # (3, nTime, nProbe)
    time = np.asarray(times, dtype=float)
    return vel, time, int(n_probes)


def read_openfoam_probe_vector_directory(
    probe_dir: str | os.PathLike,
    field_name: str = "U",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Concatenate all OpenFOAM probe vector files for one function object.

    This intentionally mirrors the behaviour you used in the older calibration
    scripts: all probe-file segments are concatenated first, and the downstream
    calibration later filters by burn-in time.  The returned time array can still
    contain duplicates at restarts; ``clean_time_series_for_spectra`` removes
    duplicates after burn-in.
    """
    files = _probe_vector_files(probe_dir, field_name=field_name)
    if not files:
        raise FileNotFoundError(f"No {field_name!r} probe files found below {probe_dir}")

    vel_parts: List[np.ndarray] = []
    time_parts: List[np.ndarray] = []
    n_probes: Optional[int] = None
    file_rows = []

    for f in files:
        vel_i, t_i, n_i = _read_openfoam_probe_vector_file(f, expected_n_probes=n_probes)
        if n_probes is None:
            n_probes = n_i
        elif n_i != n_probes:
            raise ValueError(f"Probe count changed between files: expected {n_probes}, got {n_i} in {f}")
        vel_parts.append(vel_i)
        time_parts.append(t_i)
        file_rows.append(
            {
                "file": str(f),
                "rows": int(t_i.size),
                "time_min": float(np.nanmin(t_i)),
                "time_max": float(np.nanmax(t_i)),
            }
        )

    vel = np.concatenate(vel_parts, axis=1)
    t = np.concatenate(time_parts, axis=0)

    info: Dict[str, object] = {
        "reader": "native-openfoam-probes",
        "probe_dir": str(probe_dir),
        "field": field_name,
        "n_files": len(files),
        "n_probes": int(n_probes or 0),
        "n_rows_total": int(t.size),
        "time_min": float(np.nanmin(t)),
        "time_max": float(np.nanmax(t)),
        "files": file_rows,
    }

    if verbose:
        print("OpenFOAM probe reader summary:")
        print(f"  directory      : {probe_dir}")
        print(f"  field          : {field_name}")
        print(f"  files          : {len(files)}")
        print(f"  probes         : {n_probes}")
        print(f"  rows total     : {t.size}")
        print(f"  time range     : {np.nanmin(t):.12g} -> {np.nanmax(t):.12g}")
        for row in file_rows:
            print(f"    {row['file']} | rows={row['rows']} | t={row['time_min']:.12g}->{row['time_max']:.12g}")

    return vel, t, info


def load_downstream_probe_velocity(probe_dir: str | os.PathLike) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Load downstream probe velocity data.

    Default is the native OpenFOAM parser, because it robustly handles multiple
    postProcessing/probes2/<time>/U files.  Set ``MANN_CAL_PROBE_READER=windlespy``
    to force the old windlespy reader, or ``auto`` to try native then windlespy.
    """
    mode = os.environ.get("MANN_CAL_PROBE_READER", "native").strip().lower()
    if mode in ("native", "openfoam", "foam", "auto"):
        try:
            return read_openfoam_probe_vector_directory(probe_dir, field_name="U", verbose=True)
        except Exception as exc:
            if mode != "auto":
                raise
            print(f"WARNING: native OpenFOAM probe reader failed, falling back to windlespy: {exc}")

    if mode in ("windlespy", "legacy", "auto"):
        import windlespy as LES  # type: ignore
        vel = LES._profileAnalysis.get_velocity_components(probe_dir)
        t = LES._profileAnalysis.get_time_steps_probe_data(probe_dir)
        info = {
            "reader": "windlespy",
            "probe_dir": str(probe_dir),
            "n_rows_total": int(np.asarray(t).size),
        }
        print("Windlespy probe reader summary:")
        print(f"  directory      : {probe_dir}")
        print(f"  rows total     : {np.asarray(t).size}")
        return np.asarray(vel, dtype=float), np.asarray(t, dtype=float), info

    raise ValueError(
        f"Unknown MANN_CAL_PROBE_READER={mode!r}. Use 'native', 'auto', or 'windlespy'."
    )


# -----------------------------------------------------------------------------
# Profile files and arrays
# -----------------------------------------------------------------------------

PROFILE_COLUMNS_8 = ["z", "U", "Iu", "Iv", "Iw", "Lu", "Lv", "Lw"]
PROFILE_COLUMNS_9 = PROFILE_COLUMNS_8 + ["uwStress"]


def _first_data_line_tokens(path: str | os.PathLike) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            return re.split(r"[\s,]+", s)
    return []


def _all_float(tokens: Sequence[str]) -> bool:
    for t in tokens:
        try:
            float(t)
        except Exception:
            return False
    return True


def read_profile(path: str | os.PathLike, optional: bool = False) -> Optional[pd.DataFrame]:
    path = os.fspath(path)
    if not os.path.exists(path):
        if optional:
            return None
        raise FileNotFoundError(path)
    tokens = _first_data_line_tokens(path)
    if not tokens:
        if optional:
            return None
        raise ValueError(f"Profile file is empty: {path}")
    has_header = not _all_float(tokens)
    if has_header:
        df = pd.read_csv(path, sep=r"[\s,]+", engine="python", comment="#")
    else:
        df = pd.read_csv(path, sep=r"[\s,]+", engine="python", comment="#", header=None)
        if df.shape[1] == 8:
            df.columns = PROFILE_COLUMNS_8
        elif df.shape[1] == 9:
            df.columns = PROFILE_COLUMNS_9
        else:
            raise ValueError(f"Unsupported profile format in {path}: {df.shape[1]} columns")
    # Normalize common column variants.
    rename = {}
    for c in df.columns:
        cl = c.strip()
        if cl.lower() in ("uav", "meanu", "ux"):
            rename[c] = "U"
        elif cl.lower() in ("i_u",):
            rename[c] = "Iu"
        elif cl.lower() in ("i_v",):
            rename[c] = "Iv"
        elif cl.lower() in ("i_w",):
            rename[c] = "Iw"
        elif cl.lower() in ("l_u",):
            rename[c] = "Lu"
        elif cl.lower() in ("l_v",):
            rename[c] = "Lv"
        elif cl.lower() in ("l_w",):
            rename[c] = "Lw"
        elif cl.lower() in ("uw", "ruw", "r31", "r_31", "uwstress", "u_w", "cov_uw"):
            rename[c] = "uwStress"
    if rename:
        df = df.rename(columns=rename)
    return df


def profile_to_internal_array(df: pd.DataFrame) -> np.ndarray:
    """Return columns U, uu, vv, ww, Lu, Lv, Lw, optional uwStress."""
    U = df["U"].to_numpy(float)
    if all(c in df.columns for c in ["uu", "vv", "ww"]):
        uu = df["uu"].to_numpy(float)
        vv = df["vv"].to_numpy(float)
        ww = df["ww"].to_numpy(float)
    else:
        uu = (df["Iu"].to_numpy(float) * U) ** 2
        vv = (df["Iv"].to_numpy(float) * U) ** 2
        ww = (df["Iw"].to_numpy(float) * U) ** 2
    cols = [U, uu, vv, ww, df["Lu"].to_numpy(float), df["Lv"].to_numpy(float), df["Lw"].to_numpy(float)]
    if "uwStress" in df.columns:
        cols.append(df["uwStress"].to_numpy(float))
    return np.column_stack(cols)


def internal_array_to_profile(z: np.ndarray, arr: np.ndarray, include_uw: bool = True) -> pd.DataFrame:
    arr = np.asarray(arr, float)
    U = np.maximum(arr[:, 0], 1e-12)
    data = {
        "z": z,
        "U": U,
        "Iu": np.sqrt(np.maximum(arr[:, 1], 0.0)) / U,
        "Iv": np.sqrt(np.maximum(arr[:, 2], 0.0)) / U,
        "Iw": np.sqrt(np.maximum(arr[:, 3], 0.0)) / U,
        "Lu": arr[:, 4],
        "Lv": arr[:, 5],
        "Lw": arr[:, 6],
    }
    if include_uw and arr.shape[1] >= 8:
        data["uwStress"] = arr[:, 7]
    return pd.DataFrame(data)


def write_profile(path: str | os.PathLike, df: pd.DataFrame) -> None:
    safe_makedirs(Path(path).parent)
    cols = PROFILE_COLUMNS_9 if "uwStress" in df.columns else PROFILE_COLUMNS_8
    df2 = df.loc[:, cols]
    np.savetxt(_windows_long_path(path), df2.to_numpy(float), fmt="%.12g", delimiter="\t")


def interpolate_profile_array(z_src: np.ndarray, arr_src: np.ndarray, z_dst: np.ndarray) -> np.ndarray:
    out = np.zeros((len(z_dst), arr_src.shape[1]), dtype=float)
    for j in range(arr_src.shape[1]):
        out[:, j] = np.interp(z_dst, z_src, arr_src[:, j], left=arr_src[0, j], right=arr_src[-1, j])
    return out


# -----------------------------------------------------------------------------
# Spectra/co-spectrum profile files
# -----------------------------------------------------------------------------

@dataclass
class AutoSpectraProfile:
    z: np.ndarray
    spectra: np.ndarray  # shape (3, nH, nF)
    uw_stress: Optional[np.ndarray] = None

@dataclass
class UWCoSpectrumProfile:
    z: np.ndarray
    cospectrum: np.ndarray  # shape (nH, nF)
    uw_stress: Optional[np.ndarray] = None


def read_auto_spectra(path: str | os.PathLike) -> AutoSpectraProfile:
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    header = lines[0].split()
    nH, nF = int(header[0]), int(header[1])
    z = np.zeros(nH)
    S = np.zeros((3, nH, nF), dtype=float)
    uw = None
    for i in range(nH):
        vals = np.fromstring(lines[i+1], sep=" ")
        if vals.size == 1 + 3*nF:
            z[i] = vals[0]
            off = 1
        elif vals.size == 2 + 3*nF:
            if uw is None:
                uw = np.zeros(nH)
            z[i] = vals[0]
            uw[i] = vals[1]
            off = 2
        else:
            raise ValueError(f"Unexpected row length in {path}, row {i}: {vals.size}, expected {1+3*nF} or {2+3*nF}")
        S[0, i, :] = vals[off:off+nF]
        S[1, i, :] = vals[off+nF:off+2*nF]
        S[2, i, :] = vals[off+2*nF:off+3*nF]
    return AutoSpectraProfile(z=z, spectra=np.maximum(S, FLOOR), uw_stress=uw)


def write_auto_spectra(path: str | os.PathLike, z: np.ndarray, spectra: np.ndarray, uw_stress: Optional[np.ndarray] = None, floor: float = FLOOR) -> None:
    z = np.asarray(z, float)
    S = np.maximum(np.asarray(spectra, float), floor)
    assert S.shape[0] == 3 and S.shape[1] == len(z)
    nH, nF = len(z), S.shape[2]
    safe_makedirs(Path(path).parent)
    with open(_windows_long_path(path), "w", encoding="utf-8") as f:
        f.write(f"{nH} {nF}\n")
        for i in range(nH):
            row = [z[i]]
            if uw_stress is not None:
                row.append(float(uw_stress[i]))
            row.extend(S[0, i, :])
            row.extend(S[1, i, :])
            row.extend(S[2, i, :])
            f.write(" ".join(f"{float(v):.12e}" for v in row) + "\n")


def read_uw_cospectrum(path: str | os.PathLike) -> UWCoSpectrumProfile:
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    header = lines[0].split()
    nH, nF = int(header[0]), int(header[1])
    z = np.zeros(nH)
    C = np.zeros((nH, nF), dtype=float)
    uw = None
    for i in range(nH):
        vals = np.fromstring(lines[i+1], sep=" ")
        if vals.size == 1 + nF:
            z[i] = vals[0]
            off = 1
        elif vals.size == 2 + nF:
            if uw is None:
                uw = np.zeros(nH)
            z[i] = vals[0]
            uw[i] = vals[1]
            off = 2
        else:
            raise ValueError(f"Unexpected row length in {path}, row {i}: {vals.size}, expected {1+nF} or {2+nF}")
        C[i, :] = vals[off:off+nF]
    return UWCoSpectrumProfile(z=z, cospectrum=C, uw_stress=uw)


def write_uw_cospectrum(path: str | os.PathLike, z: np.ndarray, cospectrum: np.ndarray, uw_stress: Optional[np.ndarray] = None) -> None:
    z = np.asarray(z, float)
    C = np.asarray(cospectrum, float)
    assert C.shape[0] == len(z)
    nH, nF = len(z), C.shape[1]
    safe_makedirs(Path(path).parent)
    with open(_windows_long_path(path), "w", encoding="utf-8") as f:
        f.write(f"{nH} {nF}\n")
        for i in range(nH):
            row = [z[i]]
            if uw_stress is not None:
                row.append(float(uw_stress[i]))
            row.extend(C[i, :])
            f.write(" ".join(f"{float(v):.12e}" for v in row) + "\n")


def freq_array_from_fmax(fmax: float, nfreq: int) -> np.ndarray:
    return np.linspace(fmax/nfreq, fmax, nfreq)


def infer_fmax_and_freq(case_dir: str | os.PathLike, nfreq: int) -> Tuple[float, np.ndarray]:
    mh = Path(case_dir) / "constant" / "MannHybridTurbDict"
    fmax = parse_openfoam_scalar(mh, "fMax", None)
    if fmax is None:
        # Fall back to nFreq*df from dict time settings.
        dt = parse_openfoam_scalar(mh, "timeStep", 0.0025) or 0.0025
        fmax = 1.0/(2.0*dt)
    return float(fmax), freq_array_from_fmax(float(fmax), nfreq)


# -----------------------------------------------------------------------------
# Spectral estimation and smoothing
# -----------------------------------------------------------------------------

def _safe_interp_log_spectrum(f_src: np.ndarray, y_src: np.ndarray, f_dst: np.ndarray, floor: float = FLOOR) -> np.ndarray:
    """Robust log-log interpolation for positive spectra.

    Important: this deliberately avoids PCHIP extrapolation. The earlier
    implementation used ``extrapolate=True`` in log space; when the first/last
    few spectral ordinates were noisy this could produce astronomically large
    values or ``inf`` at the target-grid edges. Outside the fitted range we use
    constant endpoint values; any high-frequency inertial tail is imposed later
    by ``log_bin_smooth_spectrum(..., tail_slope_after=...)``.
    """
    f_src = np.asarray(f_src, float)
    y_src = np.asarray(y_src, float)
    f_dst = np.asarray(f_dst, float)
    mask = np.isfinite(f_src) & np.isfinite(y_src) & (f_src > 0) & (y_src > floor)
    if np.count_nonzero(mask) < 2:
        return np.full_like(f_dst, floor, dtype=float)

    x = np.log(f_src[mask])
    y = np.log(np.maximum(y_src[mask], floor))
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    order = np.argsort(x)
    x, y = x[order], y[order]

    # Frequencies are normally already unique. Avoid an O(n^2) duplicate
    # median loop here; that made full 4096-bin calibration impractically slow.
    ux, idx = np.unique(x, return_index=True)
    uy = y[idx]

    if len(ux) < 2:
        val = np.exp(np.clip(uy[0], -700.0, 700.0))
        return np.full_like(f_dst, max(val, floor), dtype=float)

    xq = np.log(f_dst)
    if len(ux) >= 3:
        interp = PchipInterpolator(ux, uy, extrapolate=False)
        yq = interp(xq)
        left = xq < ux[0]
        right = xq > ux[-1]
        yq[left] = uy[0]
        yq[right] = uy[-1]
    else:
        yq = np.interp(xq, ux, uy, left=uy[0], right=uy[-1])

    yq = np.asarray(yq, float)
    yq[~np.isfinite(yq)] = np.nanmedian(uy)
    return np.maximum(np.exp(np.clip(yq, -700.0, 700.0)), floor)


_DPSS_CACHE = {}


def _get_dpss_cached(n: int, time_bandwidth: float, n_tapers: int) -> np.ndarray:
    """Return DPSS tapers, caching by record length and NW.

    DPSS generation is expensive.  The calibration loop applies the same
    multitaper settings to every height/component, so recomputing the tapers
    hundreds of times can dominate the Python runtime.
    """
    key = (int(n), float(time_bandwidth), int(n_tapers))
    tapers = _DPSS_CACHE.get(key)
    if tapers is None:
        tapers = dpss(int(n), float(time_bandwidth), Kmax=int(n_tapers), sym=False)
        _DPSS_CACHE[key] = tapers
    return tapers


def multitaper_psd(x: np.ndarray, fs: float, time_bandwidth: float = 3.5, n_tapers: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    x = x - np.mean(x)
    n = len(x)
    if n < 8:
        raise ValueError("Not enough samples for PSD")
    if n_tapers is None:
        n_tapers = max(3, int(2*time_bandwidth) - 1)
    tapers = _get_dpss_cached(n, time_bandwidth, n_tapers)
    X = np.fft.rfft(tapers * x[None, :], axis=1)
    scale = fs * np.sum(tapers*tapers, axis=1)[:, None]
    P = (np.abs(X)**2) / scale
    if n > 1 and P.shape[1] > 2:
        P[:, 1:-1] *= 2.0
    Pm = np.mean(P, axis=0)
    f = np.fft.rfftfreq(n, d=1.0/fs)
    return f[1:], Pm[1:]


def multitaper_csd_xy(x: np.ndarray, y: np.ndarray, fs: float, time_bandwidth: float = 3.5, n_tapers: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask] - np.mean(x[mask])
    y = y[mask] - np.mean(y[mask])
    n = min(len(x), len(y))
    x = x[:n]; y = y[:n]
    if n < 8:
        raise ValueError("Not enough samples for CSD")
    if n_tapers is None:
        n_tapers = max(3, int(2*time_bandwidth) - 1)
    tapers = _get_dpss_cached(n, time_bandwidth, n_tapers)
    X = np.fft.rfft(tapers * x[None, :], axis=1)
    Y = np.fft.rfft(tapers * y[None, :], axis=1)
    scale = fs * np.sum(tapers*tapers, axis=1)[:, None]
    C = (X * np.conj(Y)) / scale
    if n > 1 and C.shape[1] > 2:
        C[:, 1:-1] *= 2.0
    Cm = np.mean(C, axis=0)
    f = np.fft.rfftfreq(n, d=1.0/fs)
    return f[1:], np.real(Cm[1:])


def estimate_downstream_spectra(vel: np.ndarray, fs: float, f_target: np.ndarray, method: str = "multitaper", time_bandwidth: float = 3.5) -> Tuple[np.ndarray, np.ndarray]:
    """Return auto spectra (3,nH,nF) and uw co-spectrum (nH,nF).

    Multitaper is computed in height chunks with cached DPSS tapers. This keeps
    memory modest and avoids recomputing the expensive DPSS basis for every
    probe/component.
    """
    vel = np.asarray(vel, dtype=float)
    n_comp, n_time, nH = vel.shape
    nF = len(f_target)
    S = np.zeros((3, nH, nF), float)
    Cuw = np.zeros((nH, nF), float)

    if method.lower() == "welch":
        for h in range(nH):
            for c in range(3):
                x = vel[c, :, h]
                nperseg = min(len(x), max(256, len(x)//4))
                f, p = welch(x - np.mean(x), fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg//2, detrend="constant", scaling="density")
                f, p = f[1:], p[1:]
                S[c, h, :] = _safe_interp_log_spectrum(f, p, f_target)
            nperseg = min(n_time, max(256, n_time//4))
            f, cxy = csd(vel[0, :, h] - np.mean(vel[0, :, h]), vel[2, :, h] - np.mean(vel[2, :, h]), fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg//2, detrend="constant", scaling="density")
            f, cxy = f[1:], np.real(cxy[1:])
            mask = np.isfinite(f) & np.isfinite(cxy) & (f > 0)
            if np.count_nonzero(mask) >= 2:
                xlog = np.log(f[mask]); y = cxy[mask]
                order = np.argsort(xlog)
                ux, idx = np.unique(xlog[order], return_index=True)
                yy = y[order][idx]
                Cuw[h, :] = np.interp(np.log(f_target), ux, yy, left=yy[0], right=yy[-1])
        return S, Cuw

    n_tapers = max(3, int(2*time_bandwidth) - 1)
    tapers = _get_dpss_cached(n_time, time_bandwidth, n_tapers)
    taper_power = np.sum(tapers*tapers, axis=1)[None, :, None]
    f_full = np.fft.rfftfreq(n_time, d=1.0/fs)
    f = f_full[1:]
    chunk = int(os.environ.get("MANN_CAL_MULTITAPER_HEIGHT_CHUNK", "16"))
    chunk = max(1, min(chunk, nH))

    def _prep_block(c, i0, i1):
        X = vel[c, :, i0:i1].T.copy()  # (chunk,n)
        # Fill rare non-finite values with the height-wise mean.
        for hh in range(X.shape[0]):
            m = np.isfinite(X[hh])
            if not np.all(m):
                fill = np.nanmean(X[hh, m]) if np.any(m) else 0.0
                X[hh, ~m] = fill
        X -= np.mean(X, axis=1, keepdims=True)
        return X

    for i0 in range(0, nH, chunk):
        i1 = min(nH, i0 + chunk)
        Xtrans = {}
        for c in range(3):
            X = _prep_block(c, i0, i1)
            Xt = np.fft.rfft(X[:, None, :] * tapers[None, :, :], axis=2)
            P = (np.abs(Xt)**2) / (fs * taper_power)
            if n_time > 1 and P.shape[2] > 2:
                P[:, :, 1:-1] *= 2.0
            Pm = np.mean(P, axis=1)[:, 1:]
            for local_h, h in enumerate(range(i0, i1)):
                S[c, h, :] = _safe_interp_log_spectrum(f, Pm[local_h], f_target)
            if c in (0, 2):
                Xtrans[c] = Xt
        C = (Xtrans[0] * np.conj(Xtrans[2])) / (fs * taper_power)
        if n_time > 1 and C.shape[2] > 2:
            C[:, :, 1:-1] *= 2.0
        Cm = np.real(np.mean(C, axis=1))[:, 1:]
        for local_h, h in enumerate(range(i0, i1)):
            cxy = Cm[local_h]
            mask = np.isfinite(f) & np.isfinite(cxy) & (f > 0)
            if np.count_nonzero(mask) < 2:
                Cuw[h, :] = 0.0
            else:
                xlog = np.log(f[mask]); y = cxy[mask]
                order = np.argsort(xlog)
                xlog = xlog[order]; y = y[order]
                ux, idx = np.unique(xlog, return_index=True)
                y = y[idx]
                Cuw[h, :] = np.interp(np.log(f_target), ux, y, left=y[0], right=y[-1])
        del Xtrans
    return S, Cuw


def log_bin_smooth_spectrum(freq: np.ndarray, spectrum: np.ndarray, min_points: int = 4, n_bins: int = 56, floor: float = FLOOR, low_plateau_max: Optional[float] = None, tail_slope_after: Optional[float] = None, tail_slope: float = -5.0/3.0) -> np.ndarray:
    """Smooth a 1D spectrum using median log-bin knots and monotone PCHIP.

    This is closer to the spline/median fitting workflow you found useful in
    calibration, not a Savitzky-Golay filter. It is deliberately conservative:
    non-finite values are removed, knots are medians in logarithmic bins, PCHIP
    is used only between knots, and optional low-frequency plateau / high-frequency
    -5/3 tail constraints are applied afterwards.
    """
    f = np.asarray(freq, float)
    s_raw = np.asarray(spectrum, float)
    mask = np.isfinite(f) & np.isfinite(s_raw) & (f > 0) & (s_raw > floor)
    if np.count_nonzero(mask) < 3:
        return np.full_like(f, floor, dtype=float)

    f2 = f[mask]
    s2 = np.maximum(s_raw[mask], floor)

    # Remove pathological isolated ordinates before fitting. This prevents one
    # bad multitaper/CSD bin from exploding the PCHIP curve.
    log_s = np.log(s2)
    med = np.nanmedian(log_s)
    mad = np.nanmedian(np.abs(log_s - med)) + 1e-12
    robust = np.abs(log_s - med) < 12.0 * 1.4826 * mad
    if np.count_nonzero(robust) >= 3:
        f2, s2 = f2[robust], s2[robust]

    fmin, fmax = np.min(f2), np.max(f2)
    edges = np.geomspace(fmin, fmax, n_bins + 1)
    knots_f, knots_s = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (f2 >= a) & (f2 < b)
        if np.count_nonzero(m) >= min_points:
            knots_f.append(np.exp(np.mean(np.log(f2[m]))))
            knots_s.append(np.exp(np.median(np.log(np.maximum(s2[m], floor)))))

    if len(knots_f) < 3:
        y = _safe_interp_log_spectrum(f2, s2, f, floor=floor)
    else:
        y = _safe_interp_log_spectrum(np.asarray(knots_f), np.asarray(knots_s), f, floor=floor)

    if low_plateau_max is not None:
        mlow = f <= low_plateau_max
        mj = f > low_plateau_max
        if np.any(mlow) and np.any(mj):
            join_val = y[np.argmax(mj)]
            y[mlow] = join_val

    if tail_slope_after is not None:
        mtail = f > tail_slope_after
        if np.any(mtail) and np.any(~mtail):
            idx = np.where(~mtail)[0][-1]
            y[mtail] = y[idx] * (f[mtail] / max(f[idx], floor)) ** tail_slope

    y = np.asarray(y, float)
    y[~np.isfinite(y)] = floor
    return np.maximum(y, floor)


def smooth_spectra_array(freq: np.ndarray, S: np.ndarray, low_plateau_max: Optional[float] = None, tail_slope_after: Optional[float] = None, n_bins: int = 56) -> np.ndarray:
    out = np.zeros_like(S)
    for c in range(S.shape[0]):
        for h in range(S.shape[1]):
            out[c, h, :] = log_bin_smooth_spectrum(freq, S[c, h, :], low_plateau_max=low_plateau_max, tail_slope_after=tail_slope_after, n_bins=n_bins)
    return out


def smooth_cospectrum_array(freq: np.ndarray, C: np.ndarray, n_bins: int = 56) -> np.ndarray:
    out = np.zeros_like(C)
    for h in range(C.shape[0]):
        y = C[h, :]
        sign = np.sign(np.nanmedian(y[np.isfinite(y)]))
        if sign == 0:
            sign = -1.0
        mag = log_bin_smooth_spectrum(freq, np.abs(y), n_bins=n_bins)
        out[h, :] = sign * mag
    return out



# -----------------------------------------------------------------------------
# Low-frequency modelling helpers
# -----------------------------------------------------------------------------

def _blend_weight_log_frequency(freq: np.ndarray, join_freq: float) -> np.ndarray:
    """Return 0 near f_min and 1 near join_freq on a log-frequency scale."""
    f = np.asarray(freq, float)
    fmin = np.nanmin(f[f > 0]) if np.any(f > 0) else 1e-12
    fj = max(float(join_freq), fmin * (1.0 + 1e-9))
    w = (np.log(np.maximum(f, fmin)) - np.log(fmin)) / max(np.log(fj) - np.log(fmin), 1e-12)
    return np.clip(w, 0.0, 1.0)


def apply_low_frequency_shape(
    freq: np.ndarray,
    fitted: np.ndarray,
    raw_reference: Optional[np.ndarray] = None,
    mode: str = "plateau",
    join_freq: Optional[float] = None,
    floor: float = FLOOR,
    signed: bool = False,
) -> np.ndarray:
    """Apply one of several low-frequency models to an already smoothed spectrum.

    Parameters
    ----------
    freq:
        1D frequency array.
    fitted:
        Spectrum/co-spectrum already smoothed onto ``freq``. Can be any shape
        with the last axis corresponding to frequency.
    raw_reference:
        Optional less-smoothed spectrum/co-spectrum on the same frequency grid.
        Used by ``raw`` and ``blendRaw`` modes.
    mode:
        - ``plateau``: replace f <= join with a flat value equal to the fitted
          value at the first bin above join. This is the conservative von-Karman
          style low-frequency plateau used previously.
        - ``raw``: keep the raw/reference low-frequency ordinates below join.
          This deliberately trusts the measured finite-duration low-frequency
          estimate.
        - ``blendRaw``: use raw near the first bin and gradually transition to
          the fitted curve at join.
        - ``free`` or ``none``: leave the fitted curve untouched.
    join_freq:
        Frequency below which the low-frequency rule is active. If None, the
        input is returned unchanged.
    signed:
        If True, treat the values as signed co-spectra and blend linearly rather
        than in log-magnitude space.
    """
    y = np.asarray(fitted, float).copy()
    f = np.asarray(freq, float)
    if join_freq is None:
        return y
    mode_l = str(mode or "plateau").strip().lower()
    if mode_l in ("none", "free", "off", "pchip"):
        return y
    mlow = f <= float(join_freq)
    mhi = f > float(join_freq)
    if not np.any(mlow) or not np.any(mhi):
        return y

    # Shape-safe index of the join value.
    join_idx = int(np.where(mhi)[0][0])
    join_val = y[..., join_idx][..., None]

    if mode_l in ("plateau", "flat", "constant"):
        y[..., mlow] = join_val
        return np.where(np.isfinite(y), y, floor if not signed else 0.0)

    if raw_reference is None:
        # Fall back to plateau if raw is unavailable.
        y[..., mlow] = join_val
        return np.where(np.isfinite(y), y, floor if not signed else 0.0)

    raw = np.asarray(raw_reference, float)
    if raw.shape != y.shape:
        raise ValueError(f"raw_reference shape {raw.shape} does not match fitted shape {y.shape}")
    raw_low = raw[..., mlow]

    if mode_l in ("raw", "measured", "native"):
        if signed:
            y[..., mlow] = np.where(np.isfinite(raw_low), raw_low, y[..., mlow])
        else:
            y[..., mlow] = np.maximum(np.where(np.isfinite(raw_low), raw_low, y[..., mlow]), floor)
        return np.where(np.isfinite(y), y, floor if not signed else 0.0)

    if mode_l in ("blendraw", "blend", "rawblend"):
        w = _blend_weight_log_frequency(f[mlow], float(join_freq))
        # w=0 at f_min -> raw; w=1 near join -> fitted.
        if signed:
            y[..., mlow] = (1.0 - w) * np.where(np.isfinite(raw_low), raw_low, y[..., mlow]) + w * y[..., mlow]
        else:
            raw_pos = np.maximum(np.where(np.isfinite(raw_low), raw_low, y[..., mlow]), floor)
            fit_pos = np.maximum(y[..., mlow], floor)
            y[..., mlow] = np.exp((1.0 - w) * np.log(raw_pos) + w * np.log(fit_pos))
        return np.where(np.isfinite(y), y, floor if not signed else 0.0)

    raise ValueError(f"Unknown low-frequency mode: {mode!r}")

def integrate_spectra(freq: np.ndarray, S: np.ndarray, f_min: Optional[float] = None, f_max: Optional[float] = None) -> np.ndarray:
    f = np.asarray(freq, float)
    m = np.isfinite(f) & (f > 0)
    if f_min is not None:
        m &= f >= f_min
    if f_max is not None:
        m &= f <= f_max
    if np.count_nonzero(m) < 2:
        raise ValueError("Not enough frequency points for integration")
    return _trapz(S[..., m], f[m], axis=-1)


def autocorr_first_zero_integral_length(x: np.ndarray, dt: float, U: float, max_lag: Optional[int] = None) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    x = x - np.mean(x)
    n = len(x)
    if max_lag is None:
        max_lag = n//2
    max_lag = max(2, min(max_lag, n-1))
    nfft = 1 << int(np.ceil(np.log2(2*n-1)))
    X = np.fft.rfft(x, n=nfft)
    ac = np.fft.irfft(X*np.conj(X), n=nfft)[:n]
    ac = ac / max(abs(ac[0]), 1e-30)
    rho = ac[:max_lag]
    zero = np.where(rho <= 0)[0]
    end = int(zero[0]) if len(zero) else len(rho)-1
    if end < 2:
        return 0.0
    tau = np.arange(end+1)*dt
    T = _trapz(rho[:end+1], tau)
    return float(max(U, 0.0)*T)


def downstream_profile_from_velocity(vel: np.ndarray, dt: float, z: np.ndarray, max_lag_factor: float = 0.5) -> np.ndarray:
    """Return internal array U,uu,vv,ww,Lu,Lv,Lw,uwStress for probes."""
    n_comp, n_time, nH = vel.shape
    out = np.zeros((nH, 8), float)
    max_lag = max(4, int(n_time*max_lag_factor))
    U = np.mean(vel[0], axis=0)
    out[:, 0] = U
    fl = vel - np.mean(vel, axis=1, keepdims=True)
    out[:, 1] = np.var(fl[0], axis=0, ddof=0)
    out[:, 2] = np.var(fl[1], axis=0, ddof=0)
    out[:, 3] = np.var(fl[2], axis=0, ddof=0)
    out[:, 7] = np.mean(fl[0]*fl[2], axis=0)
    for h in range(nH):
        for c in range(3):
            out[h, 4+c] = autocorr_first_zero_integral_length(fl[c, :, h], dt, U[h], max_lag=max_lag)
    return out


# -----------------------------------------------------------------------------
# Update rules
# -----------------------------------------------------------------------------

@dataclass
class CalibrationConfig:
    method_name: str = "wong"
    spectral_relaxation: float = 0.45
    mean_relaxation: float = 0.45
    uw_relaxation: float = 0.35
    variance_relaxation: float = 0.35
    max_factor: float = 2.0
    min_factor: float = 0.5
    psd_estimator: str = "multitaper"
    multitaper_time_bandwidth: float = 3.5
    smooth_bins: int = 56
    low_plateau_hz: Optional[float] = None
    low_frequency_mode: str = "plateau"
    tail_slope_after_hz: Optional[float] = None
    renormalise_variance: bool = True
    update_cospectrum: bool = True
    update_profile: bool = True
    resolved_fmin: Optional[float] = None
    resolved_fmax: Optional[float] = None
    acf_low_frequency: bool = False
    lengthscale_low_frequency_gain: bool = False
    low_frequency_gain_max_hz: float = 1.0
    low_frequency_gain_relaxation: float = 0.25
    low_frequency_gain_bounds: Tuple[float, float] = (0.5, 2.0)


def bounded_factor(factor: np.ndarray, min_factor: float, max_factor: float) -> np.ndarray:
    return np.clip(np.where(np.isfinite(factor), factor, 1.0), min_factor, max_factor)


def update_auto_spectra(freq: np.ndarray, S_in: np.ndarray, S_target: np.ndarray, S_down: np.ndarray, cfg: CalibrationConfig) -> np.ndarray:
    S_in = np.maximum(S_in, FLOOR)
    S_target = np.maximum(S_target, FLOOR)
    S_down = np.maximum(S_down, FLOOR)
    method = cfg.method_name.lower()
    if method in ("log", "multiplicative"):
        factor = (S_target / S_down) ** cfg.spectral_relaxation
        factor = bounded_factor(factor, cfg.min_factor, cfg.max_factor)
        S_new = S_in * factor
    else:
        # Wong/adaptive additive residual on each frequency.
        S_new = S_in + cfg.spectral_relaxation * (S_in / S_down) * (S_target - S_down)
        S_new = np.maximum(S_new, FLOOR)
        factor = bounded_factor(S_new / S_in, cfg.min_factor, cfg.max_factor)
        S_new = S_in * factor
    # Smooth the updated spectrum to avoid bin-to-bin overfitting.
    S_new_raw_update = np.maximum(S_new, FLOOR).copy()
    S_new = smooth_spectra_array(freq, S_new, low_plateau_max=None, tail_slope_after=cfg.tail_slope_after_hz, n_bins=cfg.smooth_bins)
    S_new = apply_low_frequency_shape(
        freq, S_new, raw_reference=S_new_raw_update,
        mode=cfg.low_frequency_mode, join_freq=cfg.low_plateau_hz, signed=False
    )
    if cfg.renormalise_variance:
        fmin = cfg.resolved_fmin
        fmax = cfg.resolved_fmax
        var_in = integrate_spectra(freq, S_in, f_min=fmin, f_max=fmax)
        var_t = integrate_spectra(freq, S_target, f_min=fmin, f_max=fmax)
        var_d = integrate_spectra(freq, S_down, f_min=fmin, f_max=fmax)
        var_wong = var_in + cfg.variance_relaxation * (var_in / np.maximum(var_d, FLOOR)) * (var_t - var_d)
        var_wong = np.maximum(var_wong, FLOOR)
        var_new = np.maximum(integrate_spectra(freq, S_new, f_min=fmin, f_max=fmax), FLOOR)
        scale = var_wong / var_new
        S_new = S_new * scale[:, :, None]
    return np.maximum(S_new, FLOOR)


def update_cospectrum(freq: np.ndarray, C_in: np.ndarray, C_target: np.ndarray, C_down: np.ndarray, S_new: np.ndarray, S_down: np.ndarray, cfg: CalibrationConfig) -> np.ndarray:
    if not cfg.update_cospectrum:
        return C_in.copy()
    denom_new = np.sqrt(np.maximum(S_new[0], FLOOR) * np.maximum(S_new[2], FLOOR))
    denom_down = np.sqrt(np.maximum(S_down[0], FLOOR) * np.maximum(S_down[2], FLOOR))
    rho_in = np.clip(C_in / np.maximum(np.sqrt(np.maximum(S_new[0], FLOOR)*np.maximum(S_new[2], FLOOR)), FLOOR), -0.95, 0.95)
    rho_t = np.clip(C_target / np.maximum(denom_new, FLOOR), -0.95, 0.95)
    rho_d = np.clip(C_down / np.maximum(denom_down, FLOOR), -0.95, 0.95)
    rho_new = rho_in + cfg.uw_relaxation * (rho_t - rho_d)
    rho_new = np.clip(rho_new, -0.95, 0.95)
    C_new = rho_new * denom_new
    C_raw_update = C_new.copy()
    C_smooth = smooth_cospectrum_array(freq, C_new, n_bins=cfg.smooth_bins)
    C_smooth = apply_low_frequency_shape(
        freq, C_smooth, raw_reference=C_raw_update,
        mode=cfg.low_frequency_mode, join_freq=cfg.low_plateau_hz, signed=True
    )
    return C_smooth


def low_frequency_lengthscale_gain(freq: np.ndarray, S_new: np.ndarray, profile_target: np.ndarray, profile_downstream: np.ndarray, cfg: CalibrationConfig) -> np.ndarray:
    """Simple experimental low-frequency gain based on target/downstream L ratio."""
    if not cfg.lengthscale_low_frequency_gain:
        return S_new
    S2 = S_new.copy()
    f0 = max(float(cfg.low_frequency_gain_max_hz), np.min(freq))
    w = np.exp(-0.5*(freq/f0)**2)
    for c in range(3):
        Lt = np.maximum(profile_target[:, 4+c], 1e-8)
        Ld = np.maximum(profile_downstream[:, 4+c], 1e-8)
        ratio = np.clip((Lt/Ld)**cfg.low_frequency_gain_relaxation, cfg.low_frequency_gain_bounds[0], cfg.low_frequency_gain_bounds[1])
        for h in range(S2.shape[1]):
            S2[c, h, :] *= (1.0 + (ratio[h]-1.0)*w)
    S2_raw = S2.copy()
    S2 = smooth_spectra_array(freq, S2, low_plateau_max=None, tail_slope_after=cfg.tail_slope_after_hz, n_bins=cfg.smooth_bins)
    return apply_low_frequency_shape(
        freq, S2, raw_reference=S2_raw,
        mode=cfg.low_frequency_mode, join_freq=cfg.low_plateau_hz, signed=False
    )


def update_profile_from_spectra(z: np.ndarray, current: np.ndarray, target: np.ndarray, downstream: np.ndarray, S_new: np.ndarray, C_new: np.ndarray, freq: np.ndarray, cfg: CalibrationConfig) -> pd.DataFrame:
    U = current[:, 0] + cfg.mean_relaxation * (target[:, 0] - downstream[:, 0])
    U = np.maximum(U, 1e-6)
    var = integrate_spectra(freq, S_new, f_min=cfg.resolved_fmin, f_max=cfg.resolved_fmax).T  # nH,3
    # Keep length-scale update Wong-style from profile, but damped; spectra shape is doing most of the work.
    L = current[:, 4:7] + cfg.mean_relaxation * (current[:, 4:7]/np.maximum(downstream[:, 4:7], 1e-8)) * (target[:, 4:7] - downstream[:, 4:7])
    L = np.maximum(L, 0.005)
    uw = _trapz(C_new, freq, axis=-1)
    arr = np.column_stack([U, var[:, 0], var[:, 1], var[:, 2], L[:, 0], L[:, 1], L[:, 2], uw])
    return internal_array_to_profile(z, arr, include_uw=True)


# -----------------------------------------------------------------------------
# Diagnostics, plotting, convergence
# -----------------------------------------------------------------------------

def profile_rmse(downstream: np.ndarray, target: np.ndarray, z: np.ndarray, H: float, lower: Optional[float] = None, upper: Optional[float] = None) -> Dict[str, float]:
    if lower is None:
        lower = 0.2*H
    if upper is None:
        upper = 1.5*H
    m = (z >= lower) & (z <= upper)
    if not np.any(m):
        m = np.ones_like(z, dtype=bool)
    names = ["U", "uu", "vv", "ww", "Lu", "Lv", "Lw", "uw"]
    out = {}
    for j, name in enumerate(names):
        if j >= downstream.shape[1] or j >= target.shape[1]:
            continue
        denom = np.maximum(np.nanmean(np.abs(target[m, j])), 1e-12)
        out[name] = float(np.sqrt(np.nanmean((downstream[m, j]-target[m, j])**2))/denom)
    out["overall"] = float(np.nanmean([v for v in out.values() if np.isfinite(v)]))
    return out


def next_iteration(cal_log: str | os.PathLike) -> int:
    p = Path(cal_log)
    p.mkdir(parents=True, exist_ok=True)
    nums = []
    for d in p.glob("iteration*"):
        try:
            nums.append(int(d.name.replace("iteration", "")))
        except Exception:
            pass
    return (max(nums)+1) if nums else 1


def write_status(case_dir: str | os.PathLike, iteration: int, rmse: Dict[str, float], threshold: float, max_iters: int = 30) -> Dict[str, object]:
    converged = rmse.get("overall", 1e9) <= threshold
    stagnated = iteration >= max_iters
    status = {"iteration": int(iteration), "converged": bool(converged), "stagnated": bool(stagnated), "rmse": rmse}
    p = Path(case_dir)/"log"/"downstreamCalibration"/"iter_status_mann_spectral.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def profile_to_plot_quantities(z: np.ndarray, arr: np.ndarray, H: float, U_H_ref: Optional[float] = None) -> Dict[str, np.ndarray]:
    U = arr[:, 0]
    if U_H_ref is None:
        U_H_ref = float(np.interp(H, z, U))
    q = {
        "z_over_H": z/H,
        "U_over_UH": U/ max(abs(U_H_ref), 1e-12),
        "Iu": np.sqrt(np.maximum(arr[:, 1], 0.0))/np.maximum(U, 1e-12),
        "Iv": np.sqrt(np.maximum(arr[:, 2], 0.0))/np.maximum(U, 1e-12),
        "Iw": np.sqrt(np.maximum(arr[:, 3], 0.0))/np.maximum(U, 1e-12),
        "Lu_over_H": arr[:, 4]/H,
        "Lv_over_H": arr[:, 5]/H,
        "Lw_over_H": arr[:, 6]/H,
    }
    if arr.shape[1] >= 8:
        q["uwStress_over_UH2"] = arr[:, 7]/max(U_H_ref**2, 1e-12)
    return q


def profile_df_to_plot_quantities(df: Optional[pd.DataFrame], H: float, U_H_ref: float) -> Optional[Dict[str, np.ndarray]]:
    if df is None:
        return None
    if "z" not in df.columns:
        return None
    return profile_to_plot_quantities(df["z"].to_numpy(float), profile_to_internal_array(df), H, U_H_ref=U_H_ref)


def plot_profiles_8panel(fig_dir: str | os.PathLike, iteration: int, H: float, target: Dict[str, np.ndarray], downstream: Dict[str, np.ndarray], updated: Optional[Dict[str, np.ndarray]] = None, experimental: Optional[Dict[str, np.ndarray]] = None, smoothed: Optional[Dict[str, np.ndarray]] = None, dpi: int = 300) -> str:
    safe_makedirs(fig_dir)
    keys = ["U_over_UH", "Iu", "Iv", "Iw", "uwStress_over_UH2", "Lu_over_H", "Lv_over_H", "Lw_over_H"]
    labels = [r"$U/U_H$", r"$I_u$", r"$I_v$", r"$I_w$", r"$\overline{u'w'}/U_H^2$", r"$L_u/H$", r"$L_v/H$", r"$L_w/H$"]
    panels = list("abcdefgh")
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 8.2))
    for ax, key, lab, pan in zip(axes.ravel(), keys, labels, panels):
        if experimental is not None and key in experimental:
            ax.scatter(experimental[key], experimental["z_over_H"], s=20, facecolors="none", edgecolors="k", label="EXP", zorder=5)
        if smoothed is not None and key in smoothed:
            ax.plot(smoothed[key], smoothed["z_over_H"], "--", color="#ffbf00", lw=1.5, label="Smoothed target")
        if key in target:
            ax.plot(target[key], target["z_over_H"], ":", color="0.35", lw=1.4, label="Target")
        if key in downstream:
            ax.plot(downstream[key], downstream["z_over_H"], "-", color="#d62728", lw=2.0, label="LES downstream")
        if updated is not None and key in updated:
            ax.plot(updated[key], updated["z_over_H"], "-.", color="#1f77b4", lw=1.5, label="Updated inlet")
        ax.set_xlabel(lab)
        ax.set_ylim(0, 3.0)
        ax.set_title(f"({pan})")
        ax.grid(True, ls="--", lw=0.5, alpha=0.55)
        if key in ("U_over_UH", "uwStress_over_UH2"):
            ax.set_ylabel(r"$z/H$")
        if key == "uwStress_over_UH2":
            ax.axvline(0, color="0.4", lw=0.8)
    axes.ravel()[0].legend(loc="upper left", fontsize=8, frameon=True)
    fig.suptitle(f"Mann-hybrid downstream calibration profiles — iteration {iteration}", y=0.98)
    fig.tight_layout(rect=[0,0,1,0.955])
    path = os.path.join(fig_dir, f"iteration{iteration:02d}_profiles_melaku_8panel.png")
    safe_savefig(fig, path, dpi=dpi)
    plt.close(fig)
    return path


def plot_spectra_diagnostics(fig_dir: str | os.PathLike, iteration: int, freq: np.ndarray, z: np.ndarray, H: float, S_target: np.ndarray, S_down: np.ndarray, S_updated: np.ndarray, heights_to_plot: int = 6, dpi: int = 250) -> str:
    safe_makedirs(fig_dir)
    ids = np.unique(np.round(np.linspace(0, len(z)-1, min(heights_to_plot, len(z)))).astype(int))
    comp_names = ["u", "v", "w"]
    for c, name in enumerate(comp_names):
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        for i in ids:
            ax.loglog(freq, S_target[c, i, :], ":", lw=1.2, alpha=0.8, label=f"target z/H={z[i]/H:.2f}" if i == ids[0] else None)
            ax.loglog(freq, S_down[c, i, :], "-", lw=0.9, alpha=0.55)
            ax.loglog(freq, S_updated[c, i, :], "--", lw=1.2, alpha=0.85)
        ax.set_xlabel("f [Hz]")
        ax.set_ylabel(f"S_{name}{name} [m2/s]")
        ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.5)
        ax.set_title(f"{name}-component spectra, iteration {iteration}\nsolid=downstream, dotted=target, dashed=updated inlet")
        out = os.path.join(fig_dir, f"iteration{iteration:02d}_spectra_{name}.png")
        safe_savefig(fig, out, dpi=dpi)
        plt.close(fig)
    return os.fspath(fig_dir)


# -----------------------------------------------------------------------------
# Main calibration runner
# -----------------------------------------------------------------------------


def clean_time_series_for_spectra(vel, time_steps, burn=0.0, min_samples=8):
    """Return post-burn-in velocity/time arrays with strictly increasing unique times.

    Probe segments are allowed to overlap or restart.  This function concatenates
    first, filters by burn-in time, sorts by the actual time column, removes
    duplicate time rows while keeping the last occurrence, then estimates dt from
    the median positive time difference.
    """
    vel = np.asarray(vel, dtype=float)
    t = np.asarray(time_steps, dtype=float).reshape(-1)

    if vel.ndim != 3:
        raise ValueError(f"Expected velocity array shape (3,nTime,nHeight); got {vel.shape}")
    if vel.shape[1] != t.size:
        raise ValueError(
            f"Velocity/time length mismatch: vel has {vel.shape[1]} time samples but time array has {t.size}"
        )

    finite = np.isfinite(t)
    mask = finite & (t > float(burn))
    if np.count_nonzero(mask) < min_samples:
        raise RuntimeError(
            f"Not enough finite probe samples after burn-in time {burn}: {np.count_nonzero(mask)}. "
            f"Raw time range is {np.nanmin(t):.6g} to {np.nanmax(t):.6g}."
        )

    t = t[mask]
    vel = vel[:, mask, :]

    order = np.argsort(t, kind="mergesort")
    t = t[order]
    vel = vel[:, order, :]

    # Keep the last row for every duplicated time value.  This is robust for
    # overlapping OpenFOAM probe files produced after solver restarts.
    if t.size > 1:
        duplicate_next = np.zeros(t.size, dtype=bool)
        duplicate_next[:-1] = np.abs(np.diff(t)) <= 1e-12
        keep_last = ~duplicate_next
        n_dupes = int(np.count_nonzero(~keep_last))
        if n_dupes:
            print(f"Probe time cleaning: removed {n_dupes} duplicate post-burn-in rows, keeping last occurrence.")
        t = t[keep_last]
        vel = vel[:, keep_last, :]

    if t.size < min_samples:
        raise RuntimeError(
            f"Not enough unique post-burn-in probe samples after removing duplicate times: {t.size}"
        )

    diffs = np.diff(t)
    pos = diffs[np.isfinite(diffs) & (diffs > 1e-12)]
    if pos.size == 0:
        raise RuntimeError(
            "Could not infer a positive probe time step. The post-burn-in probe time array is constant or duplicated. "
            f"Unique post-burn time range is {t[0]:.12g} to {t[-1]:.12g} with {t.size} samples."
        )

    dt = float(np.median(pos))
    rel_jitter = float(np.max(np.abs(pos - dt)) / max(dt, 1e-30)) if pos.size else 0.0
    print(
        f"Probe time cleaning summary: post-burn unique samples={t.size}, "
        f"time={t[0]:.12g}->{t[-1]:.12g}, median dt={dt:.12g}, max rel jitter={rel_jitter:.3g}"
    )
    if rel_jitter > 1e-3:
        print(
            f"WARNING: probe time step is not perfectly uniform after duplicate removal: "
            f"median dt={dt:.8g}, max relative jitter={rel_jitter:.3g}"
        )

    return vel, t, dt

def run_calibration(cfg: CalibrationConfig) -> int:
    case_dir = os.environ.get("CASE_DIR", os.getcwd())
    p = case_paths(case_dir)
    setup = parse_case_setup(case_dir)
    sim_init = load_sim_init(case_dir)
    H = float(setup.get("buildingHeight", 0.5))
    lower = float(setup.get("lowerZThreshold", 0.2*H))
    upper = float(setup.get("upperZThreshold", 1.5*H))
    rmse_threshold = float(setup.get("rmseThreshold", 0.05))
    max_cal_iters = int(os.environ.get("MAX_CAL_ITERS", "30"))

    auto_target = read_auto_spectra(p["targetSpectraProfile"])
    auto_inlet = read_auto_spectra(p["spectraProfile"])
    if not np.allclose(auto_target.z, auto_inlet.z, atol=1e-8):
        raise ValueError("spectraProfile and targetSpectraProfile have different heights")
    z = auto_target.z
    nF = auto_target.spectra.shape[2]
    fmax, freq = infer_fmax_and_freq(case_dir, nF)
    cfg.resolved_fmin = cfg.resolved_fmin or float(freq[0])
    cfg.resolved_fmax = cfg.resolved_fmax or fmax
    if cfg.low_plateau_hz is None:
        cfg.low_plateau_hz = max(2.0*freq[0], 0.20)
    if cfg.tail_slope_after_hz is None:
        # Conservative default: start tail treatment at 0.65 fmax unless overridden.
        cfg.tail_slope_after_hz = 0.65*fmax

    profile_target_df = read_profile(p["targetProfile"])
    profile_current_df = read_profile(p["profile"])
    profile_target = interpolate_profile_array(profile_target_df["z"].to_numpy(float), profile_to_internal_array(profile_target_df), z)
    profile_current = interpolate_profile_array(profile_current_df["z"].to_numpy(float), profile_to_internal_array(profile_current_df), z)

    # Read/cohere target and current Cuw. If missing current, initialise with target.
    if os.path.exists(p["targetUWCoSpectrumProfile"]):
        cuw_target = read_uw_cospectrum(p["targetUWCoSpectrumProfile"])
        C_target = cuw_target.cospectrum
        target_uw_stress = cuw_target.uw_stress if cuw_target.uw_stress is not None else _trapz(C_target, freq, axis=-1)
    else:
        C_target = np.zeros((len(z), nF), float)
        target_uw_stress = np.zeros(len(z))
    if os.path.exists(p["uwCoSpectrumProfile"]):
        cuw_inlet = read_uw_cospectrum(p["uwCoSpectrumProfile"])
        C_inlet = cuw_inlet.cospectrum
    else:
        C_inlet = C_target.copy()

    # Load downstream probes.  Use the native OpenFOAM parser by default so
    # multiple postProcessing/probes2/<time>/U files are concatenated before
    # burn-in filtering, matching the behaviour of the earlier calibration scripts.
    try:
        vel_full, time_steps_full, probe_info = load_downstream_probe_velocity(p["probes2"])
    except Exception as exc:
        raise RuntimeError("Could not read downstream probes. Check postProcessing/probes2 and MANN_CAL_PROBE_READER.") from exc
    burn = float(sim_init.get("burn_in_time", 0.0))
    vel, time_steps, dt = clean_time_series_for_spectra(
        vel_full,
        time_steps_full,
        burn=burn,
        min_samples=8,
    )
    if dt <= 0.0 or not np.isfinite(dt):
        raise RuntimeError(f"Invalid probe time step after cleaning: dt={dt}")
    fs = 1.0/dt
    print(f"Case: {case_dir}")
    print(f"Calibration method: {cfg.method_name}")
    print(f"Samples after burn-in: {vel.shape[1]}, dt={dt:.6g}, fs={fs:.6g}")
    print(f"Spectral target grid: nH={len(z)}, nF={nF}, f=[{freq[0]:.6g}, {freq[-1]:.6g}] Hz")

    # Interpolate downstream probe heights to spectra heights if needed.
    # windlespy probes normally match target profile. If not, assume same order until postprocessing fails.
    if vel.shape[2] != len(z):
        print(f"WARNING: downstream probe count {vel.shape[2]} != target heights {len(z)}. Interpolating statistics by index is not possible; using common min count.")
        nmin = min(vel.shape[2], len(z))
        vel = vel[:, :, :nmin]
        z = z[:nmin]
        auto_target.spectra = auto_target.spectra[:, :nmin, :]
        auto_inlet.spectra = auto_inlet.spectra[:, :nmin, :]
        C_target = C_target[:nmin, :]
        C_inlet = C_inlet[:nmin, :]
        profile_target = profile_target[:nmin, :]
        profile_current = profile_current[:nmin, :]
        target_uw_stress = target_uw_stress[:nmin]

    S_down_raw, C_down_raw = estimate_downstream_spectra(vel, fs, freq, method=cfg.psd_estimator, time_bandwidth=cfg.multitaper_time_bandwidth)
    S_down = smooth_spectra_array(freq, S_down_raw, low_plateau_max=None, tail_slope_after=cfg.tail_slope_after_hz, n_bins=cfg.smooth_bins)
    S_down = apply_low_frequency_shape(
        freq, S_down, raw_reference=S_down_raw,
        mode=cfg.low_frequency_mode, join_freq=cfg.low_plateau_hz, signed=False
    )
    C_down = smooth_cospectrum_array(freq, C_down_raw, n_bins=cfg.smooth_bins)
    C_down = apply_low_frequency_shape(
        freq, C_down, raw_reference=C_down_raw,
        mode=cfg.low_frequency_mode, join_freq=cfg.low_plateau_hz, signed=True
    )
    prof_down = downstream_profile_from_velocity(vel, dt, z)
    # Add profile target uw if available.
    if profile_target.shape[1] < 8:
        profile_target = np.column_stack([profile_target, target_uw_stress])
    if profile_current.shape[1] < 8:
        profile_current = np.column_stack([profile_current, _trapz(C_inlet, freq, axis=-1)])

    S_new = update_auto_spectra(freq, auto_inlet.spectra, auto_target.spectra, S_down, cfg)
    S_new = low_frequency_lengthscale_gain(freq, S_new, profile_target, prof_down, cfg)
    C_new = update_cospectrum(freq, C_inlet, C_target, C_down, S_new, S_down, cfg)
    uw_integral = _trapz(C_new, freq, axis=-1)

    # Write updated files.
    if cfg.update_profile:
        profile_new_df = update_profile_from_spectra(z, profile_current, profile_target, prof_down, S_new, C_new, freq, cfg)
        backup = p["profile"] + ".backup_before_mann_spectral_calibration"
        if os.path.exists(p["profile"]) and not os.path.exists(backup):
            shutil.copy2(p["profile"], backup)
        write_profile(p["profile"], profile_new_df)
    write_auto_spectra(p["spectraProfile"], z, S_new, uw_stress=uw_integral)
    write_uw_cospectrum(p["uwCoSpectrumProfile"], z, C_new, uw_stress=uw_integral)

    # Diagnostics.
    iteration = next_iteration(p["calLog"])
    fig_root = Path(p["calLog"]) / f"iteration{iteration}"
    U_H = float(np.interp(H, z, profile_target[:, 0]))
    q_target = profile_to_plot_quantities(z, profile_target, H, U_H)
    q_down = profile_to_plot_quantities(z, prof_down, H, U_H)
    q_updated = profile_df_to_plot_quantities(profile_new_df, H, U_H) if cfg.update_profile else None
    exp_df = read_profile(p["targetExperimentalProfile"], optional=True)
    smooth_df = read_profile(p["targetSmoothedProfile"], optional=True)
    q_exp = profile_df_to_plot_quantities(exp_df, H, U_H) if exp_df is not None else None
    q_smooth = profile_df_to_plot_quantities(smooth_df, H, U_H) if smooth_df is not None else None
    prof_fig = plot_profiles_8panel(fig_root/"profiles", iteration, H, q_target, q_down, q_updated, experimental=q_exp, smoothed=q_smooth)
    plot_spectra_diagnostics(fig_root/"spectra", iteration, freq, z, H, auto_target.spectra, S_down, S_new)
    # Save arrays/metrics.
    safe_makedirs(fig_root/"data")
    rmse = profile_rmse(prof_down, profile_target, z, H, lower=lower, upper=upper)
    status = write_status(case_dir, iteration, rmse, rmse_threshold, max_iters=max_cal_iters)
    pd.DataFrame([rmse]).to_csv(fig_root/"data"/"rmse.csv", index=False)
    pd.DataFrame({"z": z, "uw_target_int": _trapz(C_target, freq, axis=-1), "uw_updated_int": uw_integral, "uw_downstream": prof_down[:, 7]}).to_csv(fig_root/"data"/"uw_integrals.csv", index=False)
    print(f"Profile plot: {prof_fig}")
    print(f"RMSE: {rmse}")
    print(f"Status: {status}")
    return 0 if (status["converged"] or status["stagnated"]) else 1
