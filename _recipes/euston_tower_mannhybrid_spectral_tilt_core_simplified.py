#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Euston Tower MannHybrid spectral-tilt calibration core.

This revision deliberately retains the established NHERI/Euston controller,
while allowing its tilt gain and hard log-tilt limit to be specified separately
for u, v and w.  It does not add a length-error deadband, a length-error cap,
or smoothing over height.  Target auto-spectra remain the tabulated von Karman
spectra used by MannHybridTurb.

It is intended for cases where the MannHybridTurb dictionary uses tabulated
auto spectra, e.g.

    targetSpectraSource       tabulated;
    spectraProfileFile        "spectraProfile";
    uwCoSpectrumSource        tabulated;    // recommended for all-stats calibration
    uwCoSpectrumProfileFile   "uwCoSpectrumProfile";
    uwStressSource            profile;      // profile column is also updated

The script writes:
    constant/boundaryData/windProfile/profile
    constant/boundaryData/windProfile/spectraProfile
    constant/boundaryData/windProfile/uwCoSpectrumProfile   (when enabled)

It reuses windlespy for the unchanged Wong profile update.  The only
post-Wong safeguards are positive mean velocity and an absolute turbulence-
intensity ceiling.  It reads OpenFOAM probe files and MannHybridTurb
profile/spectra files directly.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.signal import welch, csd
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scipy is required for spectral-tilt calibration") from exc

PROFILE_COLS = ["z", "U", "Iu", "Iv", "Iw", "Lu", "Lv", "Lw"]
PROFILE_COLS_UW = PROFILE_COLS + ["uwStress"]
COMPONENTS = ["u", "v", "w"]
I_COL = {"u": "Iu", "v": "Iv", "w": "Iw"}
L_COL = {"u": "Lu", "v": "Lv", "w": "Lw"}
COMP_INDEX = {"u": 0, "v": 1, "w": 2}
FLOOR = 1.0e-16
NUMERICAL_ZERO = 1.0e-12
MINIMUM_MEAN_SPEED = 0.01

ComponentParameter = Union[float, Mapping[str, float]]

VEC_RE = re.compile(r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)")
PROBE_RE = re.compile(r"^\s*#\s*Probe\s+(\d+)\s+\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)")


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


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return int(default)
    return int(raw)


def env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None or raw.strip() == "" else raw.strip()


def component_parameter_value(
    parameter: ComponentParameter,
    component: str,
    parameter_name: str,
) -> float:
    """Return one component's value from a scalar or explicit component map.

    A scalar retains the original common-setting behaviour.  A mapping must
    contain the requested component explicitly, which avoids any hidden
    fallback or precedence between common and component-specific controls.
    """
    if isinstance(parameter, Mapping):
        if component not in parameter:
            raise ValueError(
                f"{parameter_name} is missing component {component!r}; "
                "provide explicit values for every active component"
            )
        value = float(parameter[component])
    else:
        value = float(parameter)

    if not np.isfinite(value) or value < 0.0:
        raise ValueError(
            f"{parameter_name}[{component!r}] must be finite and non-negative; "
            f"got {value!r}"
        )
    return value


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


def safe_float(text: str, default: float = math.nan) -> float:
    try:
        return float(text)
    except Exception:
        return default


def parse_openfoam_scalar(path: Path, key: str, default: Optional[float] = None) -> Optional[float]:
    if not path.exists():
        return default
    txt = path.read_text(errors="ignore")
    # key value; or key = value;
    m = re.search(rf"\b{re.escape(key)}\s*(?:=)?\s*([-+0-9.eE]+)\s*;", txt)
    if not m:
        return default
    try:
        return float(m.group(1))
    except Exception:
        return default


def parse_set_up(case_dir: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p in [case_dir / "setUp", case_dir / "setup", case_dir / "system" / "setUp"]:
        if not p.exists():
            continue
        for line in p.read_text(errors="ignore").splitlines():
            line = line.split("//", 1)[0].strip()
            if not line:
                continue
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(?:=)?\s*([-+0-9.eE]+)\s*;?", line)
            if m:
                try:
                    out[m.group(1)] = float(m.group(2))
                except Exception:
                    pass
    return out


def read_profile_file(path: Path, allow_missing_uw: bool = True) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")
    lines = [ln for ln in path.read_text(errors="ignore").splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"Profile file is empty: {path}")
    first = lines[0].split()
    if not is_number_token(first[0]):
        df = pd.read_csv(path, sep=r"\s+", comment="#", engine="python")
        rename = {
            "height": "z", "Z": "z", "u": "U", "Uav": "U",
            "I_U": "Iu", "I_V": "Iv", "I_W": "Iw",
            "L_u": "Lu", "L_v": "Lv", "L_w": "Lw",
            "uw": "uwStress", "uv": "uwStress", "u'w'": "uwStress",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        cols = [c for c in PROFILE_COLS_UW if c in df.columns]
        df = df.loc[:, cols]
    else:
        arr = np.loadtxt(path, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] == 8:
            cols = PROFILE_COLS
        elif arr.shape[1] == 9:
            cols = PROFILE_COLS_UW
        else:
            raise ValueError(f"Unexpected profile column count in {path}: {arr.shape[1]}, expected 8 or 9")
        df = pd.DataFrame(arr, columns=cols)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if not allow_missing_uw and "uwStress" not in df.columns:
        raise ValueError(f"Profile {path} has no uwStress column")
    if "uwStress" not in df.columns:
        df["uwStress"] = 0.0
    df = df.loc[:, PROFILE_COLS_UW]
    arr = df.to_numpy(float)
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values found in profile {path}")
    df = df.sort_values("z").drop_duplicates("z", keep="last").reset_index(drop=True)
    return df


def write_profile_file(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    cols = [c for c in PROFILE_COLS_UW if c in df.columns]
    arr = df.loc[:, cols].to_numpy(float)
    with path.open("w", encoding="utf-8") as f:
        for row in arr:
            f.write("\t".join(f"{float(v):.12g}" for v in row) + "\n")


@dataclass
class SpectraProfile:
    z: np.ndarray
    S: np.ndarray  # shape (3, nH, nF)
    uw_stress: Optional[np.ndarray] = None


def read_auto_spectra(path: Path) -> SpectraProfile:
    if not path.exists():
        raise FileNotFoundError(f"spectraProfile not found: {path}")
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().split()
        if len(header) < 2:
            raise ValueError(f"Bad spectraProfile header in {path}")
        nH, nF = int(header[0]), int(header[1])
        z = np.zeros(nH, dtype=float)
        S = np.zeros((3, nH, nF), dtype=float)
        uw: Optional[np.ndarray] = None
        for i in range(nH):
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF in spectraProfile {path} at row {i}")
            vals = np.fromstring(line, sep=" ")
            if vals.size == 1 + 3*nF:
                off = 1
            elif vals.size == 2 + 3*nF:
                if uw is None:
                    uw = np.zeros(nH, dtype=float)
                uw[i] = vals[1]
                off = 2
            else:
                raise ValueError(f"Unexpected spectraProfile row length at row {i}: {vals.size}; expected {1+3*nF} or {2+3*nF}")
            z[i] = vals[0]
            S[0, i, :] = vals[off:off+nF]
            S[1, i, :] = vals[off+nF:off+2*nF]
            S[2, i, :] = vals[off+2*nF:off+3*nF]
    S = np.asarray(S, float)
    S[~np.isfinite(S)] = FLOOR
    S = np.maximum(S, FLOOR)
    return SpectraProfile(z=z, S=S, uw_stress=uw)


def write_auto_spectra(path: Path, z: np.ndarray, S: np.ndarray, uw_stress: Optional[np.ndarray] = None) -> None:
    z = np.asarray(z, float)
    S = np.maximum(np.asarray(S, float), FLOOR)
    if S.shape[:2] != (3, len(z)):
        raise ValueError(f"Bad S shape {S.shape}; expected (3,{len(z)},nF)")
    nH, nF = len(z), S.shape[2]
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{nH} {nF}\n")
        for i in range(nH):
            row: List[float] = [float(z[i])]
            if uw_stress is not None:
                row.append(float(uw_stress[i]))
            row.extend(S[0, i, :])
            row.extend(S[1, i, :])
            row.extend(S[2, i, :])
            f.write(" ".join(f"{float(v):.12e}" for v in row) + "\n")


@dataclass
class CospectraProfile:
    z: np.ndarray
    Cuw: np.ndarray  # shape (nH, nF), one-sided real co-spectrum
    uw_stress: Optional[np.ndarray] = None


def read_uw_cospectra(path: Path) -> CospectraProfile:
    """Read a MannHybridTurb u-w co-spectrum profile file.

    Expected format:
        nHeights nFreq
        z uwStress Cuw_1 ... Cuw_nFreq

    A legacy variant without the uwStress column is also accepted.
    """
    if not path.exists():
        raise FileNotFoundError(f"uwCoSpectrumProfile not found: {path}")
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().split()
        if len(header) < 2:
            raise ValueError(f"Bad uwCoSpectrumProfile header in {path}")
        nH, nF = int(header[0]), int(header[1])
        z = np.zeros(nH, dtype=float)
        Cuw = np.zeros((nH, nF), dtype=float)
        uw = None
        for i in range(nH):
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF in uwCoSpectrumProfile {path} at row {i}")
            vals = np.fromstring(line, sep=" ")
            if vals.size == 1 + nF:
                off = 1
            elif vals.size == 2 + nF:
                if uw is None:
                    uw = np.zeros(nH, dtype=float)
                uw[i] = vals[1]
                off = 2
            else:
                raise ValueError(f"Unexpected uwCoSpectrumProfile row length at row {i}: {vals.size}; expected {1+nF} or {2+nF}")
            z[i] = vals[0]
            Cuw[i, :] = vals[off:off+nF]
    Cuw = np.asarray(Cuw, float)
    Cuw[~np.isfinite(Cuw)] = 0.0
    if uw is not None:
        uw[~np.isfinite(uw)] = 0.0
    return CospectraProfile(z=z, Cuw=Cuw, uw_stress=uw)


def write_uw_cospectra(path: Path, z: np.ndarray, Cuw: np.ndarray, uw_stress: Optional[np.ndarray] = None) -> None:
    z = np.asarray(z, float)
    Cuw = np.asarray(Cuw, float)
    Cuw[~np.isfinite(Cuw)] = 0.0
    if Cuw.shape[0] != len(z):
        raise ValueError(f"Bad Cuw shape {Cuw.shape}; expected ({len(z)},nF)")
    nH, nF = len(z), Cuw.shape[1]
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{nH} {nF}\n")
        for i in range(nH):
            row: List[float] = [float(z[i])]
            if uw_stress is not None:
                row.append(float(uw_stress[i]))
            row.extend(Cuw[i, :])
            f.write(" ".join(f"{float(v):.12e}" for v in row) + "\n")


def map_cospectra_to_z(cp: CospectraProfile, z_new: np.ndarray) -> np.ndarray:
    z_new = np.asarray(z_new, float)
    C = np.zeros((len(z_new), cp.Cuw.shape[1]), dtype=float)
    for k in range(cp.Cuw.shape[1]):
        C[:, k] = np.interp(z_new, cp.z, cp.Cuw[:, k])
    C[~np.isfinite(C)] = 0.0
    return C


def make_default_uw_cospectrum(profile: pd.DataFrame, freq: np.ndarray, S_auto: np.ndarray, rho_limit: float = 0.95) -> np.ndarray:
    """Build a smooth fallback Cuw shape and normalize it to profile['uwStress'].

    The shape is proportional to sqrt(Suu*Sww) with the target stress sign.
    It is only a fallback; a measured/target co-spectrum file is preferable.
    """
    z = profile["z"].to_numpy(float)
    C = np.zeros((len(z), len(freq)), dtype=float)
    for j in range(len(z)):
        shape = np.sqrt(np.maximum(S_auto[0, j, :], FLOOR) * np.maximum(S_auto[2, j, :], FLOOR))
        sign = -1.0 if float(profile["uwStress"].iloc[j]) <= 0.0 else 1.0
        C[j, :] = sign * np.maximum(shape, FLOOR)
    return normalize_cospectrum_to_stress(C, freq, profile["uwStress"].to_numpy(float), S_auto[0], S_auto[2], rho_limit)


def normalize_cospectrum_to_stress(C: np.ndarray, freq: np.ndarray, uw_target: np.ndarray, Suu: np.ndarray, Sww: np.ndarray, rho_limit: float = 0.98, n_iter: int = 8) -> np.ndarray:
    """Scale each Cuw(f,z) row so its integral matches uw_target.

    A pointwise realizability limiter |Cuw| <= rho_limit*sqrt(Suu*Sww) is applied.
    If clipping prevents exact area matching, the closest clipped spectrum is kept.
    """
    C = np.asarray(C, float).copy()
    C[~np.isfinite(C)] = 0.0
    uw_target = np.asarray(uw_target, float)
    Suu = np.maximum(np.asarray(Suu, float), FLOOR)
    Sww = np.maximum(np.asarray(Sww, float), FLOOR)
    limit = float(rho_limit) * np.sqrt(Suu * Sww)
    out = np.zeros_like(C)
    for j in range(C.shape[0]):
        target = float(uw_target[j])
        if abs(target) < 1e-14:
            out[j, :] = 0.0
            continue
        sign = -1.0 if target < 0.0 else 1.0
        shape = C[j, :].copy()
        # If the row has the wrong sign or near-zero area, force a sign-consistent shape.
        area = float(trapz(shape, x=freq))
        if (area * target <= 0.0) or abs(area) < 1e-14 or not np.isfinite(area):
            shape = sign * np.maximum(np.abs(shape), FLOOR)
            if float(trapz(shape, x=freq)) * target <= 0.0 or abs(float(trapz(shape, x=freq))) < 1e-14:
                shape = sign * np.maximum(limit[j, :], FLOOR)
        cj = shape.copy()
        for _ in range(max(1, n_iter)):
            area = float(trapz(cj, x=freq))
            if abs(area) < 1e-14 or not np.isfinite(area):
                cj = sign * np.maximum(limit[j, :], FLOOR)
                area = float(trapz(cj, x=freq))
            cj *= target / area
            cj = np.clip(cj, -limit[j, :], limit[j, :])
        out[j, :] = cj
    out[~np.isfinite(out)] = 0.0
    return out


def freq_array_from_fmax(fmax: float, nfreq: int) -> np.ndarray:
    return np.linspace(float(fmax)/int(nfreq), float(fmax), int(nfreq))


def infer_fmax(case_dir: Path, nfreq: int, dt_fallback: Optional[float] = None) -> float:
    mh = case_dir / "constant" / "MannHybridTurbDict"
    fmax = parse_openfoam_scalar(mh, "fMax", None)
    if fmax is not None:
        return float(fmax)
    setup = parse_set_up(case_dir)
    if "fMax" in setup:
        return float(setup["fMax"])
    if dt_fallback:
        return 1.0/(2.0*float(dt_fallback))
    return 200.0


def parse_probe_file(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
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
            triples = VEC_RE.findall(line)
            if not triples:
                continue
            times.append(float(parts[0]))
            rows.append(np.asarray([[float(a), float(b), float(c)] for a, b, c in triples], dtype=float))
    if not rows:
        raise ValueError(f"No vector rows found in probe file: {path}")
    nprobes = rows[0].shape[0]
    for i, row in enumerate(rows):
        if row.shape[0] != nprobes:
            raise ValueError(f"Probe count mismatch in {path} row {i}: {row.shape[0]} vs {nprobes}")
    vel = np.transpose(np.stack(rows, axis=0), (2, 0, 1))
    pos_arr = None
    if positions:
        pos_arr = np.full((nprobes, 3), np.nan)
        for idx, xyz in positions.items():
            if 0 <= idx < nprobes:
                pos_arr[idx, :] = xyz
        if not np.isfinite(pos_arr).all():
            pos_arr = None
    return np.asarray(times, float), vel, pos_arr


def read_probe_positions_from_system(case_dir: Path, probes_name: str, nprobes: int) -> Optional[np.ndarray]:
    for p in [case_dir / "system" / probes_name, case_dir / "system" / "probes"]:
        if not p.exists():
            continue
        triples = VEC_RE.findall(p.read_text(errors="ignore"))
        if len(triples) >= nprobes:
            return np.asarray([[float(a), float(b), float(c)] for a,b,c in triples[:nprobes]], float)
    return None


def read_all_probe_segments(case_dir: Path, probes_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Path]]:
    root = case_dir / "postProcessing" / probes_name
    if not root.exists():
        raise FileNotFoundError(f"Probe directory not found: {root}")
    files = sorted(root.glob("*/U"), key=lambda p: (safe_float(p.parent.name, 0.0), str(p)))
    if (root / "U").exists():
        files.insert(0, root / "U")
    if not files:
        raise FileNotFoundError(f"No U files found under {root}")
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
    t = np.concatenate(all_t)
    v = np.concatenate(all_v, axis=1)
    if positions is None:
        positions = read_probe_positions_from_system(case_dir, probes_name, nprobes or v.shape[2])
    if positions is None:
        raise RuntimeError("Could not determine probe positions from headers or system/probes file")
    return t, v, positions, files


def clean_time_history(time_full: np.ndarray, vel_full: np.ndarray, burn_in_time: float, duplicate_tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    t = np.asarray(time_full, float)
    v = np.asarray(vel_full, float)
    finite = np.isfinite(t) & np.all(np.isfinite(v), axis=(0, 2))
    mask = finite & (t > burn_in_time)
    if np.count_nonzero(mask) < 8:
        raise RuntimeError(f"Not enough samples after burn_in_time={burn_in_time}")
    t = t[mask]
    v = v[:, mask, :]
    order = np.argsort(t, kind="mergesort")
    t = t[order]
    v = v[:, order, :]
    keep = np.r_[np.diff(t) > duplicate_tol, True]
    dup = int(t.size - np.count_nonzero(keep))
    t = t[keep]
    v = v[:, keep, :]
    diffs = np.diff(t)
    pos = diffs[np.isfinite(diffs) & (diffs > duplicate_tol)]
    if pos.size == 0:
        raise RuntimeError("Could not infer positive dt from probe file")
    dt = float(np.median(pos))
    return t, v, {"n_samples": int(t.size), "t_min": float(t.min()), "t_max": float(t.max()), "dt": dt, "fs": 1.0/dt, "duplicates_removed": dup}


def group_probe_indices_by_height(positions: np.ndarray, tol: float = 1e-6) -> Tuple[np.ndarray, List[np.ndarray]]:
    z = np.asarray(positions, float)[:, 2]
    order = np.argsort(z)
    groups: List[List[int]] = []
    heights: List[float] = []
    current: List[int] = []
    current_z: Optional[float] = None
    for idx in order:
        zi = float(z[idx])
        if current_z is None or abs(zi - current_z) <= tol:
            current.append(int(idx))
            current_z = zi if current_z is None else current_z
        else:
            groups.append(current)
            heights.append(float(np.mean(z[current])))
            current = [int(idx)]
            current_z = zi
    if current:
        groups.append(current)
        heights.append(float(np.mean(z[current])))
    return np.asarray(heights, float), [np.asarray(g, dtype=int) for g in groups]


def autocorr_fft(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    x = x - np.mean(x)
    n = x.size
    if n < 4:
        return np.ones(1)
    nfft = 1 << (2*n - 1).bit_length()
    fx = np.fft.rfft(x, n=nfft)
    ac = np.fft.irfft(fx * np.conj(fx), n=nfft)[:n]
    norm = np.arange(n, 0, -1, dtype=float)
    ac = ac / norm
    if ac[0] <= 0 or not np.isfinite(ac[0]):
        return np.ones(n) * np.nan
    return ac / ac[0]


def integral_time_first_zero(
    x: np.ndarray,
    dt: float,
    max_lag_fraction: float = 0.5,
) -> float:
    """Integrate the normalized autocorrelation to its first zero crossing.

    The zero-crossing time is linearly interpolated between the final positive
    autocorrelation ordinate and the first non-positive ordinate.  Returning
    NaN when no crossing is present prevents a truncated positive tail from
    being mislabeled as a first-zero integral time scale.
    """
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be positive and finite; got {dt!r}")

    r = autocorr_fft(x)
    if r.size < 3 or not np.isfinite(r).all():
        return np.nan

    max_index = min(
        r.size - 1,
        max(2, int(math.floor(max_lag_fraction * (r.size - 1)))),
    )
    rr = r[: max_index + 1]
    crossings = np.flatnonzero(rr[1:] <= 0.0) + 1
    if crossings.size == 0:
        return np.nan

    k = int(crossings[0])
    r_left = float(rr[k - 1])
    r_right = float(rr[k])
    denominator = r_left - r_right
    fraction = 0.0 if denominator <= 0.0 else r_left / denominator
    fraction = float(np.clip(fraction, 0.0, 1.0))
    tau_zero = ((k - 1) + fraction) * float(dt)

    tau = np.arange(k, dtype=float) * float(dt)
    values = rr[:k].astype(float, copy=False)
    tau_with_zero = np.append(tau, tau_zero)
    values_with_zero = np.append(values, 0.0)
    return float(trapz(values_with_zero, x=tau_with_zero))


def integral_length_first_zero(
    x: np.ndarray,
    dt: float,
    U: float,
    max_lag_fraction: float = 0.5,
) -> float:
    """Return L = U*T, with T integrated to the first ACF zero crossing."""
    integral_time = integral_time_first_zero(
        x,
        dt,
        max_lag_fraction=max_lag_fraction,
    )
    if not np.isfinite(integral_time):
        return np.nan
    return float(max(float(U), 0.0) * integral_time)



def integral_length_efold(x: np.ndarray, dt: float, U: float, max_lag_fraction: float = 0.5) -> float:
    r = autocorr_fft(x)
    if not np.isfinite(r).all():
        return np.nan
    nmax = max(3, min(len(r)-1, int(max_lag_fraction*len(r))))
    rr = r[:nmax]
    target = math.exp(-1.0)
    idx = np.where(rr <= target)[0]
    if idx.size:
        k = int(idx[0])
        if k == 0:
            tau = 0.0
        else:
            # linear interpolate between k-1 and k
            x0, x1 = rr[k-1], rr[k]
            a = 0.0 if x1 == x0 else (target - x0)/(x1 - x0)
            tau = (k-1 + a) * dt
    else:
        tau = nmax * dt
    return float(U * tau)


def _welch_parameters_from_spectra_grid(
    time: np.ndarray,
    spectra_freq: np.ndarray,
    requested_nperseg: int,
) -> Dict[str, object]:
    """Choose spectral-estimation parameters without losing the record low end.

    A positive ``requested_nperseg`` retains the historical Welch behaviour.
    A non-positive value selects the Euston mode: use the complete retained
    record as one Hann-window segment and, where possible, zero-pad to the
    uniform frequency spacing represented by ``spectraProfile``.  Zero padding
    aligns/eases interpolation onto that table; it does not create information
    below the physical resolution 1/T_record.
    """
    time = np.asarray(time, float)
    spectra_freq = np.asarray(spectra_freq, float)
    if time.size < 16:
        raise ValueError("At least 16 retained samples are required for spectra")
    if spectra_freq.size < 2 or np.any(~np.isfinite(spectra_freq)):
        raise ValueError("spectraProfile must provide at least two finite bins")
    if np.any(spectra_freq <= 0.0) or np.any(np.diff(spectra_freq) <= 0.0):
        raise ValueError("spectraProfile frequencies must be positive and increasing")

    dt = float(np.median(np.diff(time)))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"Could not determine a positive time step; got {dt!r}")
    fs = 1.0 / dt
    n_samples = int(time.size)

    df_values = np.diff(spectra_freq)
    df_profile = float(np.median(df_values))
    if not np.allclose(
        df_values,
        df_profile,
        rtol=1.0e-6,
        atol=max(1.0e-14, 1.0e-10 * abs(df_profile)),
    ):
        raise ValueError(
            "spectraProfile frequency bins are not uniformly spaced; "
            "the current MannHybrid table format is expected to be uniform"
        )

    if int(requested_nperseg) > 0:
        nperseg_eff = int(min(max(16, int(requested_nperseg)), n_samples))
        nfft_eff = nperseg_eff
        noverlap = int(0.5 * nperseg_eff)
        mode = "configured Welch segmentation"
    else:
        # Full-record segment gives the finest physically available low-frequency
        # resolution.  Match the spectraProfile spacing where this only requires
        # zero padding; otherwise retain the full-record FFT and interpolate.
        nperseg_eff = n_samples
        target_nfft = max(16, int(round(fs / df_profile)))
        nfft_eff = max(nperseg_eff, target_nfft)
        noverlap = 0
        mode = "full retained record, spectraProfile-aligned FFT grid"

    return {
        "dt": dt,
        "fs": fs,
        "nperseg": nperseg_eff,
        "noverlap": noverlap,
        "nfft": int(nfft_eff),
        "welch_df": float(fs / nfft_eff),
        "spectra_profile_df": df_profile,
        "record_duration": float(time[-1] - time[0]),
        "mode": mode,
    }


def compute_downstream_profiles_and_spectra(
    time: np.ndarray,
    vel: np.ndarray,
    positions: np.ndarray,
    profile_z: np.ndarray,
    nperseg: int,
    spectra_freq: np.ndarray,
    l_method: str = "efold",
    z_group_tol: float = 1e-6,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, object]]:
    """Return downstream statistics on ``profile_z`` and auto-spectra.

    The simplified Euston recipe selects ``l_method='efold'`` and
    ``nperseg=4096``, using the historical 50%-overlapped Welch path.
    """
    spectral_parameters = _welch_parameters_from_spectra_grid(
        time,
        spectra_freq,
        nperseg,
    )
    dt = float(spectral_parameters["dt"])
    fs = float(spectral_parameters["fs"])
    nperseg_eff = int(spectral_parameters["nperseg"])
    noverlap = int(spectral_parameters["noverlap"])
    nfft_eff = int(spectral_parameters["nfft"])

    z_heights, groups = group_probe_indices_by_height(positions, z_group_tol)
    records = []
    S_h = np.zeros((3, len(z_heights), len(spectra_freq)), float)
    method_key = l_method.lower().replace("-", "_").strip()

    for j, inds in enumerate(groups):
        # Average spectra/statistics across same-height probes, not concatenating them.
        recs_j = []
        S_acc = np.zeros((3, len(spectra_freq)), float)
        for ind in inds:
            u = vel[0, :, ind]
            v = vel[1, :, ind]
            w = vel[2, :, ind]
            Umean = float(np.mean(u))
            means = [Umean, float(np.mean(v)), float(np.mean(w))]
            sig = [
                float(np.std(u - means[0], ddof=0)),
                float(np.std(v - means[1], ddof=0)),
                float(np.std(w - means[2], ddof=0)),
            ]

            Lvals = []
            for component_name, x in zip(("u", "v", "w"), (u, v, w)):
                xp = x - np.mean(x)
                if method_key in {"efold", "exp", "expfit"}:
                    length = integral_length_efold(
                        xp,
                        dt,
                        max(Umean, 1.0e-12),
                    )
                elif method_key in {
                    "first_zero",
                    "firstzero",
                    "integral",
                    "zero_crossing",
                }:
                    integral_time = integral_time_first_zero(xp, dt)
                    if not np.isfinite(integral_time):
                        raise RuntimeError(
                            "The normalized autocorrelation did not cross zero "
                            f"within half the retained record for component "
                            f"{component_name!r}, probe={int(ind)}, "
                            f"z={z_heights[j]:.12g}. A first-zero integral "
                            "length scale cannot be estimated from this record."
                        )
                    length = max(Umean, 1.0e-12) * integral_time
                else:
                    raise ValueError(
                        f"Unknown integral-length method {l_method!r}; use "
                        "'first_zero' or 'efold'."
                    )
                Lvals.append(float(length))

            uw = float(np.mean((u - means[0]) * (w - means[2])))
            recs_j.append(
                [
                    z_heights[j],
                    Umean,
                    sig[0] / max(Umean, 1.0e-12),
                    sig[1] / max(Umean, 1.0e-12),
                    sig[2] / max(Umean, 1.0e-12),
                    Lvals[0],
                    Lvals[1],
                    Lvals[2],
                    uw,
                ]
            )

            for c, x in enumerate(
                (u - means[0], v - means[1], w - means[2])
            ):
                fw, Pw = welch(
                    x,
                    fs=fs,
                    window="hann",
                    nperseg=nperseg_eff,
                    noverlap=noverlap,
                    nfft=nfft_eff,
                    detrend="constant",
                    scaling="density",
                )
                valid = (fw > 0.0) & np.isfinite(Pw) & (Pw > 0.0)
                if np.count_nonzero(valid) < 4:
                    interp = np.full_like(spectra_freq, FLOOR)
                else:
                    interp = np.exp(
                        np.interp(
                            np.log(spectra_freq),
                            np.log(fw[valid]),
                            np.log(Pw[valid]),
                            left=np.log(Pw[valid][0]),
                            right=np.log(Pw[valid][-1]),
                        )
                    )
                S_acc[c, :] += interp

        arrj = np.asarray(recs_j, float)
        records.append(np.nanmean(arrj, axis=0))
        S_h[:, j, :] = S_acc / max(len(inds), 1)

    df_h = pd.DataFrame(np.asarray(records), columns=PROFILE_COLS_UW)
    out = pd.DataFrame({"z": profile_z})
    for col in PROFILE_COLS_UW[1:]:
        out[col] = np.interp(profile_z, df_h["z"], df_h[col])

    S_p = np.zeros((3, len(profile_z), len(spectra_freq)), float)
    for c in range(3):
        for k in range(len(spectra_freq)):
            S_p[c, :, k] = np.interp(
                profile_z,
                z_heights,
                S_h[c, :, k],
            )

    meta = {
        **spectral_parameters,
        "integral_length_method": method_key,
        "unique_probe_heights": z_heights.tolist(),
        "n_height_groups": len(z_heights),
    }
    return out, np.maximum(S_p, FLOOR), meta



def compute_downstream_uw_cospectrum(
    time: np.ndarray,
    vel: np.ndarray,
    positions: np.ndarray,
    profile_z: np.ndarray,
    nperseg: int,
    spectra_freq: np.ndarray,
    z_group_tol: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Estimate the one-sided real u-w co-spectrum on the spectraProfile grid."""
    spectral_parameters = _welch_parameters_from_spectra_grid(
        time,
        spectra_freq,
        nperseg,
    )
    fs = float(spectral_parameters["fs"])
    nperseg_eff = int(spectral_parameters["nperseg"])
    noverlap = int(spectral_parameters["noverlap"])
    nfft_eff = int(spectral_parameters["nfft"])

    z_heights, groups = group_probe_indices_by_height(positions, z_group_tol)
    C_h = np.zeros((len(z_heights), len(spectra_freq)), float)
    for j, inds in enumerate(groups):
        C_acc = np.zeros(len(spectra_freq), float)
        for ind in inds:
            up = vel[0, :, ind] - np.mean(vel[0, :, ind])
            wp = vel[2, :, ind] - np.mean(vel[2, :, ind])
            fw, Puw = csd(
                up,
                wp,
                fs=fs,
                window="hann",
                nperseg=nperseg_eff,
                noverlap=noverlap,
                nfft=nfft_eff,
                detrend="constant",
                scaling="density",
            )
            Cw = np.real(Puw)
            valid = (fw > 0.0) & np.isfinite(Cw)
            if np.count_nonzero(valid) < 4:
                interp = np.zeros_like(spectra_freq)
            else:
                interp = np.interp(
                    np.log(spectra_freq),
                    np.log(fw[valid]),
                    Cw[valid],
                    left=Cw[valid][0],
                    right=Cw[valid][-1],
                )
            C_acc += interp
        C_h[j, :] = C_acc / max(len(inds), 1)

    C_p = np.zeros((len(profile_z), len(spectra_freq)), float)
    for k in range(len(spectra_freq)):
        C_p[:, k] = np.interp(profile_z, z_heights, C_h[:, k])
    C_p[~np.isfinite(C_p)] = 0.0

    meta = {
        **spectral_parameters,
        "unique_probe_heights": z_heights.tolist(),
        "n_height_groups": len(z_heights),
    }
    return C_p, meta



def make_vonkarman_like_spectra(profile: pd.DataFrame, freq: np.ndarray) -> np.ndarray:
    """A robust one-sided von-Karman-like spectrum scaled exactly to sigma^2.

    The exact constants are less important here than stable shape and exact area.
    nS/sigma^2 = 4 x / (1 + 70.8 x^2)^(5/6), x = f L / U.
    """
    z = profile["z"].to_numpy(float)
    S = np.zeros((3, len(z), len(freq)), float)
    U = np.maximum(profile["U"].to_numpy(float), 1e-12)
    for comp in COMPONENTS:
        c = COMP_INDEX[comp]
        I = np.maximum(profile[I_COL[comp]].to_numpy(float), 1e-12)
        L = np.maximum(profile[L_COL[comp]].to_numpy(float), 1e-12)
        sig2 = (I * U)**2
        for j in range(len(z)):
            x = np.maximum(freq * L[j] / U[j], 1e-12)
            Sj = sig2[j] * (4.0 * L[j] / U[j]) / ((1.0 + 70.8*x*x)**(5.0/6.0))
            area = trapz(Sj, x=freq)
            if np.isfinite(area) and area > 0:
                Sj = Sj * sig2[j] / area
            S[c, j, :] = np.maximum(Sj, FLOOR)
    return S


def rescale_to_area(S: np.ndarray, freq: np.ndarray, target_area: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(S, float).copy(), FLOOR)
    area = trapz(out, x=freq, axis=-1)
    factor = np.asarray(target_area, float) / np.maximum(area, FLOOR)
    return np.maximum(out * factor[..., None], FLOOR)


def spectral_centroid_logf(S: np.ndarray, freq: np.ndarray, fmin: Optional[float] = None, fmax: Optional[float] = None) -> np.ndarray:
    mask = np.ones_like(freq, dtype=bool)
    if fmin is not None:
        mask &= freq >= fmin
    if fmax is not None:
        mask &= freq <= fmax
    f = freq[mask]
    lnf = np.log(np.maximum(f, 1e-12))
    Sm = np.maximum(S[..., mask], FLOOR)
    E = trapz(Sm, x=f, axis=-1)
    return trapz(Sm * lnf, x=f, axis=-1) / np.maximum(E, FLOOR)




def spectral_centroid_logf_heightwise(S: np.ndarray, freq: np.ndarray, fmin: np.ndarray, fmax: np.ndarray) -> np.ndarray:
    """Height-specific log-frequency centroid using the resolved band at each height.

    Parameters
    ----------
    S : ndarray
        Spectrum array with shape (..., nH, nF).  For this script the usual
        shape is (3, nH, nF).
    freq : ndarray
        Frequency array.
    fmin, fmax : ndarray
        Per-height resolved lower/upper frequency limits.
    """
    S = np.maximum(np.asarray(S, float), FLOOR)
    freq = np.asarray(freq, float)
    fmin = np.asarray(fmin, float).reshape(-1)
    fmax = np.asarray(fmax, float).reshape(-1)
    if S.shape[-2] != fmin.size:
        raise ValueError(f"S height dimension {S.shape[-2]} does not match fmin/fmax length {fmin.size}")
    out = np.zeros(S.shape[:-1], dtype=float)
    for idx in np.ndindex(out.shape):
        j = idx[-1]
        lo = max(float(freq[0]), float(fmin[j]))
        hi = min(float(freq[-1]), float(fmax[j]))
        mask = (freq >= lo) & (freq <= hi)
        if np.count_nonzero(mask) < 2:
            # Fallback to the full available frequency range only if the
            # resolved range is numerically degenerate.
            mask = np.ones_like(freq, dtype=bool)
        f = freq[mask]
        lnf = np.log(np.maximum(f, 1e-12))
        Sj = np.maximum(S[idx + (slice(None),)][mask], FLOOR)
        E = float(trapz(Sj, x=f))
        out[idx] = float(trapz(Sj * lnf, x=f) / max(E, FLOOR))
    return out

def make_log_bands(freq: np.ndarray, fmin: float, fmax: float, n_bands: int) -> np.ndarray:
    fmin = max(float(fmin), float(freq[0]))
    fmax = min(float(fmax), float(freq[-1]))
    if fmax <= fmin:
        fmin, fmax = float(freq[0]), float(freq[-1])
    return np.geomspace(fmin, fmax, int(n_bands)+1)


def band_energies(S: np.ndarray, freq: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return energies shape S.shape[:-1] + (nBands,)."""
    out_shape = S.shape[:-1] + (len(edges)-1,)
    out = np.zeros(out_shape, float)
    for b in range(len(edges)-1):
        m = (freq >= edges[b]) & (freq <= edges[b+1])
        if np.count_nonzero(m) < 2:
            # include nearest index to band center
            center = math.sqrt(edges[b]*edges[b+1])
            idx = int(np.argmin(np.abs(freq - center)))
            out[..., b] = S[..., idx] * max(edges[b+1] - edges[b], 1e-12)
        else:
            out[..., b] = trapz(S[..., m], x=freq[m], axis=-1)
    return out




def _direction(value: float) -> int:
    """Return the sign of a finite control signal.

    ``NUMERICAL_ZERO`` is only a floating-point guard.  It is not an
    engineering length-error deadband or a tunable controller setting.
    """
    if not np.isfinite(value) or abs(float(value)) <= NUMERICAL_ZERO:
        return 0
    return 1 if float(value) > 0.0 else -1


def _safe_log_ratio(num: float, den: float, default: float = 0.0) -> float:
    """Log(num/den) only when both values are positive and finite."""
    if not (np.isfinite(num) and np.isfinite(den)):
        return float(default)
    if num <= 0.0 or den <= 0.0:
        return float(default)
    return float(math.log(max(num, FLOOR) / max(den, FLOOR)))


def _local_band_log_update(
    S_target_1d: np.ndarray,
    S_down_1d: np.ndarray,
    freq: np.ndarray,
    fmin: float,
    fmax: float,
    n_bands: int,
    relax: float,
    max_log_update: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return local band centres and relaxed log-energy correction.

    This is used only in the optional band-energy variant.  The bands are built
    over the same height-specific resolved frequency interval that will actually
    be modified, avoiding the earlier inconsistency where a global set of bands
    could drive an update inside a narrower local resolved range.
    """
    edges = make_log_bands(freq, fmin, fmax, n_bands)
    centers = np.sqrt(edges[:-1] * edges[1:])
    logs = np.zeros(len(edges) - 1, dtype=float)
    for b in range(len(edges) - 1):
        lo, hi = float(edges[b]), float(edges[b + 1])
        Et = max(integrate_1d_between(freq, S_target_1d, lo, hi), FLOOR)
        Ed = max(integrate_1d_between(freq, S_down_1d, lo, hi), FLOOR)
        logs[b] = float(relax) * np.clip(math.log(Et / Ed), -max_log_update, max_log_update)
    return centers, logs

@dataclass
class TiltConfig:
    case_dir: Path
    profile_dir_rel: str = "constant/boundaryData/windProfile"
    probes_name: str = "probes2"
    target_profile_file: str = "targetProfile"
    active_profile_file: str = "profile"
    target_spectra_file: str = "targetSpectraProfile"
    active_spectra_file: str = "spectraProfile"
    target_uw_cospectrum_file: str = "targetUWCoSpectrumProfile"
    active_uw_cospectrum_file: str = "uwCoSpectrumProfile"
    log_dir_name: str = "spectralTiltCalibration"
    building_height: float = 0.5
    burn_in_time: float = 0.0
    z_cal_min: float = 0.0
    z_cal_max: float = 1.5
    f_min: float = 0.05
    f_max_update: float = 120.0
    n_bands: int = 8
    nperseg: int = 4096
    min_samples: int = 4000
    min_record_duration: float = 10.0
    l_method: str = "efold"
    components: Tuple[str, ...] = ("v", "w")
    # MannHybridTurb generator mode. This changes only whether the existing
    # Wong Reynolds-shear-stress update is active; the established auto-spectrum
    # calibration algorithm and all of its tuned constants remain unchanged.
    inflow_mode: str = "sameComponentCoherence"
    mode: str = "moment"  # moment, bands, twoband, hybrid, diagnostic
    profile_relax_U: float = 0.15
    profile_relax_I: float = 0.30
    profile_relax_L: float = 0.00
    profile_relax_uw: float = 0.20
    variance_relax: float = 0.50
    shape_relax: float = 0.60
    # Each controller setting may be one common scalar (legacy behaviour) or
    # an explicit {"u": ..., "v": ..., "w": ...} component mapping.
    moment_relax: ComponentParameter = 0.50
    # For any band-energy correction, compute correction bands over each
    # height's local resolved range rather than one global frequency range.
    band_correction_uses_resolved_range: bool = True
    max_log_area_update: float = math.log(2.0)
    max_log_band_update: float = math.log(3.0)
    max_log_tilt: ComponentParameter = 0.70
    max_log_uw_update: float = math.log(2.0)
    rho_uw_limit: float = 0.98
    update_profile_intensity: bool = True
    update_profile_length: bool = False
    update_uw_stress: bool = True
    update_uw_cospectrum: bool = True
    freeze_above_zmax: bool = True
    normalize_to_target_variance: bool = False
    use_windlespy_wong_update: bool = True
    wong_relaxation_factor: float = 0.9
    max_turbulence_intensity: float = 0.50
    use_windlespy_resolved_band: bool = True
    uw_cospectrum_resolved_band_only: bool = True
    write_plots: bool = True
    exit_always_continue: bool = True


def infer_burn_in_time(case_dir: Path, setup: Dict[str, float]) -> float:
    """Infer the statistics start time for downstream calibration.

    Important: setUp/minSimDuration is the intended total simulated record
    duration, not the burn-in time.  Earlier versions accidentally used
    minSimDuration as the default burn-in, which left only the last few dozen
    samples in cases with endTime ~= minSimDuration.

    Priority:
      1. explicit MST_BURN_IN_TIME environment variable;
      2. log/downstreamCalibration/sim_init.json burn_in_time;
      3. log/spectralTiltCalibration/sim_init.json burn_in_time;
      4. zero.
    """
    raw = os.environ.get("MST_BURN_IN_TIME")
    if raw is not None and raw.strip() != "":
        return float(raw)

    for rel in [
        "log/downstreamCalibration/sim_init.json",
        "log/spectralTiltCalibration/sim_init.json",
        "downstreamCalibration/sim_init.json",
        "spectralTiltCalibration/sim_init.json",
    ]:
        p = case_dir / rel
        if p.exists():
            try:
                data = json.loads(p.read_text(errors="ignore"))
                if "burn_in_time" in data:
                    return float(data["burn_in_time"])
            except Exception:
                pass

    # Deliberately do not fall back to setup["minSimDuration"].
    return 0.0



# -----------------------------------------------------------------------------
# Calibration-band helpers using the MannHybrid active-frequency ceiling
# -----------------------------------------------------------------------------

def compute_windlespy_resolved_limits(
    case_dir: Path,
    profile: pd.DataFrame,
    time: np.ndarray,
    freq: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the calibration frequency band on the spectraProfile grid.

    The lower physical limit is the reciprocal of the retained time-series
    duration.  The upper physical limit is the user-supplied
    ``maximumFrequency`` scalar in ``setUp``.  This is treated as the active
    MannHybrid generator ceiling, not as a separately fitted Kaimal cutoff.
    The limits returned to the height-wise calibration code are the first and
    last actual spectraProfile bins inside those physical limits.
    """
    try:
        import windlespy as LES  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Frequency-band selection requires windlespy to be importable. "
            "Set PYTHONPATH to the parent directory containing windlespy."
        ) from exc

    variable_dict = LES._caseFiles.parse_setup_file(str(case_dir))
    if "maximumFrequency" not in variable_dict:
        raise ValueError(
            "Could not find the required scalar 'maximumFrequency' in setUp. "
            "Set this to the active MannHybrid maximum frequency in Hz."
        )

    configured_maximum = float(variable_dict["maximumFrequency"])
    if not np.isfinite(configured_maximum) or configured_maximum <= 0.0:
        raise ValueError(
            "setUp:maximumFrequency must be positive and finite; found "
            f"{variable_dict['maximumFrequency']!r}"
        )

    time = np.asarray(time, float)
    freq = np.asarray(freq, float)
    if time.size < 2:
        raise ValueError("At least two retained time samples are required")
    if freq.size < 2 or np.any(~np.isfinite(freq)):
        raise ValueError("spectraProfile must contain at least two finite bins")
    if np.any(freq <= 0.0) or np.any(np.diff(freq) <= 0.0):
        raise ValueError(
            "spectraProfile frequencies must be positive and strictly increasing"
        )

    duration = float(time[-1] - time[0])
    dt = float(np.median(np.diff(time)))
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"Retained record duration is invalid: {duration!r}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"Retained sample interval is invalid: {dt!r}")

    physical_minimum = 1.0 / duration
    nyquist = 1.0 / (2.0 * dt)
    physical_maximum = min(configured_maximum, nyquist, float(freq[-1]))

    active_indices = np.flatnonzero(
        (freq >= physical_minimum) & (freq <= physical_maximum)
    )
    if active_indices.size < 2:
        raise ValueError(
            "The calibration band contains fewer than two spectraProfile "
            f"bins: 1/T={physical_minimum:.12g} Hz, "
            f"setUp:maximumFrequency={configured_maximum:.12g} Hz, "
            f"Nyquist={nyquist:.12g} Hz, table=[{freq[0]:.12g}, "
            f"{freq[-1]:.12g}] Hz."
        )

    selected_minimum = float(freq[int(active_indices[0])])
    selected_maximum = float(freq[int(active_indices[-1])])
    print(
        "Calibration frequency band: "
        f"1/T_record={physical_minimum:.12g} Hz; "
        f"first spectraProfile bin={selected_minimum:.12g} Hz; "
        f"setUp:maximumFrequency={configured_maximum:.12g} Hz; "
        f"last spectraProfile bin={selected_maximum:.12g} Hz",
        flush=True,
    )

    n_heights = len(profile)
    return (
        np.full(n_heights, selected_minimum, dtype=float),
        np.full(n_heights, selected_maximum, dtype=float),
        np.full(n_heights, configured_maximum, dtype=float),
    )



def integrate_1d_between(freq: np.ndarray, y: np.ndarray, fmin: float, fmax: float) -> float:
    mask = (freq >= fmin) & (freq <= fmax)
    if np.count_nonzero(mask) < 2:
        return 0.0
    return float(trapz(y[mask], x=freq[mask]))


def integrate_heightwise_between(freq: np.ndarray, S_2d: np.ndarray, fmin: np.ndarray, fmax: np.ndarray) -> np.ndarray:
    out = np.zeros(S_2d.shape[0], dtype=float)
    for j in range(S_2d.shape[0]):
        out[j] = integrate_1d_between(freq, S_2d[j, :], float(fmin[j]), float(fmax[j]))
    return out


def rescale_heightwise_resolved_area(S_2d: np.ndarray, freq: np.ndarray, target_area: np.ndarray, fmin: np.ndarray, fmax: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(S_2d, float).copy(), FLOOR)
    for j in range(out.shape[0]):
        mask = (freq >= fmin[j]) & (freq <= fmax[j])
        if np.count_nonzero(mask) < 2:
            continue
        area = float(trapz(out[j, mask], x=freq[mask]))
        if area > FLOOR:
            out[j, mask] *= float(max(target_area[j], FLOOR)) / area
    return np.maximum(out, FLOOR)


def normalize_cospectrum_resolved_band(C_base: np.ndarray, freq: np.ndarray, stress: np.ndarray, Suu: np.ndarray, Sww: np.ndarray, rho_lim: float, fmin: np.ndarray, fmax: np.ndarray) -> np.ndarray:
    """Scale Cuw shape so the resolved-band integral equals stress(z).

    No spectral tilting is applied to Cuw here.  The shape is preserved, but the
    magnitude is adjusted in the resolved band and clipped pointwise by
    |Cuw| <= rho_lim*sqrt(Suu*Sww).
    """
    C = np.asarray(C_base, float).copy()
    nH = C.shape[0]
    for j in range(nH):
        mask = (freq >= fmin[j]) & (freq <= fmax[j])
        if np.count_nonzero(mask) < 2:
            C[j, :] = 0.0
            continue
        shape = C[j, :].copy()
        # If the existing shape is unusable, construct a simple signed target from sqrt(Suu*Sww).
        area_shape = float(trapz(shape[mask], x=freq[mask]))
        if abs(area_shape) < 1e-14:
            sign = -1.0 if stress[j] <= 0.0 else 1.0
            shape = sign * np.sqrt(np.maximum(Suu[j, :], FLOOR) * np.maximum(Sww[j, :], FLOOR))
            area_shape = float(trapz(shape[mask], x=freq[mask]))
        if abs(area_shape) < 1e-14:
            C[j, :] = 0.0
            continue
        C[j, :] = shape * (float(stress[j]) / area_shape)
        lim = rho_lim * np.sqrt(np.maximum(Suu[j, :], FLOOR) * np.maximum(Sww[j, :], FLOOR))
        C[j, :] = np.clip(C[j, :], -lim, lim)
        # One additional resolved-band renormalisation after clipping if feasible.
        area2 = float(trapz(C[j, mask], x=freq[mask]))
        if abs(area2) > 1e-14:
            C[j, :] *= float(stress[j]) / area2
            C[j, :] = np.clip(C[j, :], -lim, lim)
    return C

VALID_MANNHYBRID_INFLOW_MODES = (
    "rawMann",
    "sameComponentCoherence",
    "reynoldsShearStress",
)


def enforce_mannhybrid_inflow_mode(cfg: TiltConfig) -> TiltConfig:
    """Apply only the mode gates required by MannHybridTurb v1.3.1.

    The existing profile/auto-spectrum calibration is deliberately left
    unchanged.  Reynolds shear stress is calibrated only in
    ``reynoldsShearStress`` mode.  A separate u-w cospectrum file is never
    iteratively updated because the revised utility constructs the Mann-shaped
    cospectrum internally from the current profile stress.
    """
    if cfg.inflow_mode not in VALID_MANNHYBRID_INFLOW_MODES:
        raise ValueError(
            f"Unknown inflow_mode {cfg.inflow_mode!r}; expected one of "
            f"{VALID_MANNHYBRID_INFLOW_MODES}"
        )
    cfg.update_uw_stress = cfg.inflow_mode == "reynoldsShearStress"
    cfg.update_uw_cospectrum = False
    return cfg


def default_config(mode: str = "moment", components: Sequence[str] = ("v", "w"), **overrides) -> TiltConfig:
    case_dir_override = overrides.pop("case_dir", None)
    case_dir = Path(
        case_dir_override
        if case_dir_override is not None
        else os.environ.get("CASE_DIR", ".")
    ).expanduser().resolve()
    setup = parse_set_up(case_dir)
    H = env_float("MST_BUILDING_HEIGHT", setup.get("buildingHeight", 0.5))
    full_height = env_bool("MST_FULL_HEIGHT", True)
    default_zmin = -1.0e30 if full_height else 0.0
    default_zmax =  1.0e30 if full_height else 3.0*H
    zmax = env_float("MST_Z_CAL_MAX", default_zmax)
    cfg = TiltConfig(
        case_dir=case_dir,
        profile_dir_rel=env_str("MST_PROFILE_DIR", "constant/boundaryData/windProfile"),
        probes_name=env_str("MST_PROBES_NAME", "probes2"),
        log_dir_name=env_str("MST_LOG_DIR_NAME", "spectralTiltCalibration"),
        building_height=H,
        burn_in_time=infer_burn_in_time(case_dir, setup),
        z_cal_min=env_float("MST_Z_CAL_MIN", default_zmin),
        z_cal_max=zmax,
        f_min=env_float("MST_F_MIN", 0.05),
        f_max_update=env_float("MST_F_MAX_UPDATE", setup.get("fMax", 120.0)),
        n_bands=env_int("MST_N_BANDS", 8),
        nperseg=env_int("MST_NPERSEG", 4096),
        min_samples=env_int("MST_MIN_SAMPLES", 4000),
        min_record_duration=env_float("MST_MIN_RECORD_DURATION", 10.0),
        l_method=env_str("MST_L_METHOD", "efold"),
        inflow_mode=env_str("MST_INFLOW_MODE", "sameComponentCoherence"),
        mode=env_str("MST_MODE", mode),
        components=tuple(env_str("MST_COMPONENTS", ",".join(components)).replace(" ", "").split(",")),
        profile_relax_U=env_float("MST_RELAX_U", 0.15),
        profile_relax_I=env_float("MST_RELAX_I", 0.30),
        profile_relax_L=env_float("MST_RELAX_L", 0.00),
        profile_relax_uw=env_float("MST_RELAX_UW", 0.10),
        variance_relax=env_float("MST_VARIANCE_RELAX", 0.50),
        shape_relax=env_float("MST_SHAPE_RELAX", 0.60),
        moment_relax=env_float("MST_MOMENT_RELAX", 0.50),
        band_correction_uses_resolved_range=env_bool("MST_BANDS_LOCAL_RESOLVED", True),
        max_log_area_update=env_float("MST_MAX_LOG_AREA_UPDATE", math.log(2.0)),
        max_log_band_update=env_float("MST_MAX_LOG_BAND_UPDATE", math.log(3.0)),
        max_log_tilt=env_float("MST_MAX_LOG_TILT", 0.70),
        max_log_uw_update=env_float("MST_MAX_LOG_UW_UPDATE", math.log(2.0)),
        rho_uw_limit=env_float("MST_RHO_UW_LIMIT", 0.98),
        update_profile_intensity=env_bool("MST_UPDATE_PROFILE_INTENSITY", True),
        update_profile_length=env_bool("MST_UPDATE_PROFILE_LENGTH", False),
        update_uw_stress=env_bool("MST_UPDATE_UW_STRESS", True),
        update_uw_cospectrum=env_bool("MST_UPDATE_UW_COSPECTRUM", True),
        freeze_above_zmax=env_bool("MST_FREEZE_ABOVE_ZMAX", False if full_height else True),
        normalize_to_target_variance=env_bool("MST_NORMALIZE_TO_TARGET_VARIANCE", False),
        use_windlespy_wong_update=env_bool("MST_USE_WINDLESPY_WONG_UPDATE", True),
        wong_relaxation_factor=env_float("MST_WONG_RELAXATION_FACTOR", 0.9),
        max_turbulence_intensity=env_float("MST_MAX_TURBULENCE_INTENSITY", 0.50),
        use_windlespy_resolved_band=env_bool("MST_USE_WINDLESPY_RESOLVED_BAND", True),
        uw_cospectrum_resolved_band_only=env_bool("MST_UW_RESOLVED_BAND_ONLY", True),
        write_plots=env_bool("MST_WRITE_PLOTS", True),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return enforce_mannhybrid_inflow_mode(cfg)


def next_iteration_dir(case_dir: Path, log_dir_name: str) -> Tuple[int, Path]:
    root = ensure_dir(case_dir / "log" / log_dir_name)
    existing = []
    for p in root.glob("iteration*"):
        m = re.match(r"iteration(\d+)", p.name)
        if m:
            existing.append(int(m.group(1)))
    it = (max(existing) + 1) if existing else 1
    out = ensure_dir(root / f"iteration{it:02d}")
    ensure_dir(out / "data")
    ensure_dir(out / "inputs")
    ensure_dir(out / "plots")
    return it, out



def profile_to_wong_array_from_statistics(profile: pd.DataFrame) -> np.ndarray:
    """Convert a profile dataframe to the classic Wong-calibration array.

    The windlespy Wong routine operates on

        U, R_11, R_22, R_33, L_u, L_v, L_w[, u'w']

    rather than on turbulence intensities.  Here the normal Reynolds stresses
    are always constructed directly from the profile statistics:

        R_ii = (I_i U)^2

    For the downstream LES profile, ``I_i`` and ``u'w'`` were computed directly
    from the post-burn-in velocity time series in
    ``compute_downstream_profiles_and_spectra``.  Therefore this Wong update uses
    the time-domain resolved LES variances/covariance, not spectral integrals of
    Welch estimates.
    """
    U = np.maximum(profile["U"].to_numpy(float), 1e-12)
    R11 = np.maximum((profile["Iu"].to_numpy(float) * U)**2, 1e-12)
    R22 = np.maximum((profile["Iv"].to_numpy(float) * U)**2, 1e-12)
    R33 = np.maximum((profile["Iw"].to_numpy(float) * U)**2, 1e-12)
    Lu = np.maximum(profile["Lu"].to_numpy(float), 1e-12)
    Lv = np.maximum(profile["Lv"].to_numpy(float), 1e-12)
    Lw = np.maximum(profile["Lw"].to_numpy(float), 1e-12)
    uw = profile["uwStress"].to_numpy(float) if "uwStress" in profile.columns else np.zeros_like(U)
    arr = np.stack([U, R11, R22, R33, Lu, Lv, Lw, uw], axis=1)
    arr[~np.isfinite(arr)] = 0.0
    arr[:, 0] = np.clip(arr[:, 0], 1e-12, None)
    arr[:, 1:4] = np.clip(arr[:, 1:4], 1e-12, None)
    arr[:, 4:7] = np.clip(arr[:, 4:7], 1e-12, None)
    return arr


def wong_array_to_profile_df(
    template: pd.DataFrame,
    wong_array: np.ndarray,
    keep_lengths_from: Optional[pd.DataFrame] = None,
    keep_uw_from: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Convert U/R/L/uw Wong array back to the MannHybrid profile dataframe."""
    arr = np.asarray(wong_array, float)
    out = template.copy()
    U = np.clip(arr[:, 0], 1e-12, None)
    out["U"] = U
    out["Iu"] = np.sqrt(np.maximum(arr[:, 1], 0.0)) / U
    out["Iv"] = np.sqrt(np.maximum(arr[:, 2], 0.0)) / U
    out["Iw"] = np.sqrt(np.maximum(arr[:, 3], 0.0)) / U
    out["Lu"] = np.maximum(arr[:, 4], 1e-12)
    out["Lv"] = np.maximum(arr[:, 5], 1e-12)
    out["Lw"] = np.maximum(arr[:, 6], 1e-12)
    if arr.shape[1] >= 8:
        out["uwStress"] = arr[:, 7]
    if keep_lengths_from is not None:
        for col in ["Lu", "Lv", "Lw"]:
            out[col] = keep_lengths_from[col].to_numpy(float)
    if keep_uw_from is not None:
        out["uwStress"] = keep_uw_from["uwStress"].to_numpy(float)
    return out.loc[:, PROFILE_COLS_UW]


def exact_windlespy_wong_update(
    current_array: np.ndarray,
    target_array: np.ndarray,
    downstream_array: np.ndarray,
    relaxation_factor: float = 0.9,
) -> np.ndarray:
    """Call the windlespy Wong updater, with an exact local fallback.

    The fallback is intentionally identical to the current windlespy definition:
        adaptive = relaxation_factor * current/downstream
        adaptive clipped to [0.5, 5]
        new = current + adaptive*(target-downstream)
    followed by the same positivity clips for U, R_ii and L_i.
    """
    try:
        import windlespy as LES  # type: ignore
        return np.asarray(
            LES._profileCalibration.new_dfsr_profile_array(
                np.asarray(current_array, float),
                np.asarray(target_array, float),
                np.asarray(downstream_array, float),
                relaxation_factor=float(relaxation_factor),
            ),
            dtype=float,
        )
    except Exception as exc:
        print(
            "WARNING: could not call windlespy._profileCalibration.new_dfsr_profile_array; "
            f"using an exact local copy of the same update. Original error: {exc}",
            file=sys.stderr,
            flush=True,
        )
        current_array = np.asarray(current_array, float)
        target_array = np.asarray(target_array, float)
        downstream_array = np.asarray(downstream_array, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            adaptive_relaxation_factor = float(relaxation_factor) * (current_array / downstream_array)
        conditions = [
            adaptive_relaxation_factor < 0.5,
            (adaptive_relaxation_factor >= 0.5) & (adaptive_relaxation_factor <= 5),
            adaptive_relaxation_factor > 5,
        ]
        choices = [0.5, adaptive_relaxation_factor, 5]
        adaptive_relaxation_factor = np.select(conditions, choices)
        new_inlet_profile = current_array + adaptive_relaxation_factor * (target_array - downstream_array)
        new_inlet_profile[:, 0] = np.clip(new_inlet_profile[:, 0], 0.01, None)
        new_inlet_profile[:, 1:4] = np.clip(new_inlet_profile[:, 1:4], 1e-8, None)
        new_inlet_profile[:, 4:7] = np.clip(new_inlet_profile[:, 4:7], 0.01, None)
        return new_inlet_profile


def apply_minimal_profile_safeguards(
    wong_candidate: np.ndarray,
    max_turbulence_intensity: float,
) -> np.ndarray:
    """Apply only the user-requested physical safeguards after Wong.

    The Wong update itself is unchanged.  Afterwards, mean speed is required
    to be positive and each normal stress is limited so that
    ``0 <= sqrt(R_ii)/U <= max_turbulence_intensity``.  No per-iteration trust
    region, target-relative envelope, or lower turbulence-intensity target is
    imposed.
    """
    if not np.isfinite(max_turbulence_intensity) or not (
        0.0 < max_turbulence_intensity <= 1.0
    ):
        raise ValueError(
            "max_turbulence_intensity must be finite and in (0, 1]"
        )

    out = np.asarray(wong_candidate, float).copy()
    if out.ndim != 2 or out.shape[1] < 4:
        raise ValueError("Wong candidate must contain U, R11, R22 and R33")

    original_u = out[:, 0].copy()
    valid_u = np.isfinite(original_u) & (original_u > 0.0)
    out[:, 0] = np.where(
        valid_u,
        np.maximum(original_u, MINIMUM_MEAN_SPEED),
        MINIMUM_MEAN_SPEED,
    )

    normal_stresses = out[:, 1:4]
    original_stresses = normal_stresses.copy()
    normal_stresses = np.where(
        np.isfinite(normal_stresses), normal_stresses, 0.0
    )
    max_variance = (
        max_turbulence_intensity * out[:, 0]
    )[:, None] ** 2
    out[:, 1:4] = np.clip(normal_stresses, 0.0, max_variance)

    u_repairs = int(np.count_nonzero(~np.isclose(out[:, 0], original_u)))
    stress_repairs = int(
        np.count_nonzero(~np.isclose(out[:, 1:4], original_stresses))
    )
    if u_repairs or stress_repairs:
        print(
            "Post-Wong physical safeguards applied: "
            f"U repairs={u_repairs}, normal-stress/TI caps={stress_repairs}",
            flush=True,
        )

    return out


def update_profile_wong(
    current: pd.DataFrame,
    target: pd.DataFrame,
    downstream: pd.DataFrame,
    cfg: TiltConfig,
    cal_mask: np.ndarray,
    freq: Optional[np.ndarray] = None,
    S_current: Optional[np.ndarray] = None,
    S_target: Optional[np.ndarray] = None,
    S_down: Optional[np.ndarray] = None,
    Cuw_current: Optional[np.ndarray] = None,
    Cuw_target: Optional[np.ndarray] = None,
    Cuw_down: Optional[np.ndarray] = None,
    resolved_fmin: Optional[np.ndarray] = None,
    resolved_fmax: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Profile update using the actual windlespy Wong rule.

    This uses the same array variables expected by
    ``LES._profileCalibration.new_dfsr_profile_array``:

        U, R_11, R_22, R_33, L_u, L_v, L_w[, u'w']

    The downstream R_11/R_22/R_33 and u'w' values come from the velocity time
    series statistics already stored in ``downstream``.  They are not recomputed
    from spectral integrals.  This matches the classic downstream calibration
    recipe, while the spectral-tilt step below only changes spectral shape and
    then renormalises the auto-spectra over the resolved update band.
    """
    out = current.copy()
    if cfg.use_windlespy_wong_update:
        current_array = profile_to_wong_array_from_statistics(current)
        target_array = profile_to_wong_array_from_statistics(target)
        downstream_array = profile_to_wong_array_from_statistics(downstream)

        new_array = exact_windlespy_wong_update(
            current_array,
            target_array,
            downstream_array,
            relaxation_factor=cfg.wong_relaxation_factor,
        )

        # Keep the Wong rule unchanged. Apply only the requested post-update
        # physical safeguards: U > 0 and Iu/Iv/Iw <= the absolute TI ceiling.
        new_array = apply_minimal_profile_safeguards(
            new_array,
            cfg.max_turbulence_intensity,
        )

        keep_lengths = None if cfg.update_profile_length else current
        keep_uw = None if cfg.update_uw_stress else current
        candidate = wong_array_to_profile_df(current, new_array, keep_lengths_from=keep_lengths, keep_uw_from=keep_uw)
        out.loc[cal_mask, :] = candidate.loc[cal_mask, :]
        if cfg.freeze_above_zmax:
            out.loc[~cal_mask, :] = current.loc[~cal_mask, :]
    else:
        # Legacy multiplicative update retained only for controlled comparisons.
        eps = 1e-12
        def bounded_factor(tar, down, relax, maxlog):
            f = np.exp(relax * np.clip(np.log(np.maximum(tar, eps) / np.maximum(down, eps)), -maxlog, maxlog))
            return f
        if cfg.profile_relax_U > 0:
            fac = bounded_factor(target["U"].to_numpy(float), downstream["U"].to_numpy(float), cfg.profile_relax_U, math.log(1.25))
            out.loc[cal_mask, "U"] = current.loc[cal_mask, "U"].to_numpy(float) * fac[cal_mask]
        if cfg.update_profile_intensity and cfg.profile_relax_I > 0:
            for c in ["Iu", "Iv", "Iw"]:
                fac = bounded_factor(target[c].to_numpy(float), downstream[c].to_numpy(float), cfg.profile_relax_I, math.log(2.5))
                out.loc[cal_mask, c] = current.loc[cal_mask, c].to_numpy(float) * fac[cal_mask]
        if cfg.update_profile_length and cfg.profile_relax_L > 0:
            for c in ["Lu", "Lv", "Lw"]:
                fac = bounded_factor(target[c].to_numpy(float), downstream[c].to_numpy(float), cfg.profile_relax_L, math.log(2.0))
                out.loc[cal_mask, c] = current.loc[cal_mask, c].to_numpy(float) * fac[cal_mask]
        if cfg.update_uw_stress and cfg.profile_relax_uw > 0 and "uwStress" in current.columns:
            cur = current["uwStress"].to_numpy(float)
            tar = target["uwStress"].to_numpy(float)
            down = downstream["uwStress"].to_numpy(float)
            sign = np.sign(tar)
            sign[sign == 0] = np.sign(cur[sign == 0])
            sign[sign == 0] = -1.0
            fac = bounded_factor(np.abs(tar), np.abs(down), cfg.profile_relax_uw, cfg.max_log_uw_update)
            mag_cur = np.maximum(np.abs(cur), 0.02*np.abs(tar))
            nxt = sign * mag_cur * fac
            out.loc[cal_mask, "uwStress"] = nxt[cal_mask]
        if cfg.freeze_above_zmax:
            out.loc[~cal_mask, :] = current.loc[~cal_mask, :]

    # The original realizability limiter is retained when Reynolds shear stress
    # is an actively calibrated quantity. In the other two generator modes the
    # profile stress is a diagnostic/input only and must not be modified.
    if cfg.update_uw_stress:
        sig_u = np.maximum(out["Iu"].to_numpy(float) * out["U"].to_numpy(float), 1e-12)
        sig_w = np.maximum(out["Iw"].to_numpy(float) * out["U"].to_numpy(float), 1e-12)
        bound = cfg.rho_uw_limit * sig_u * sig_w
        out["uwStress"] = np.clip(out["uwStress"].to_numpy(float), -bound, bound)
    return out



def apply_spectral_tilt(
    S_current: np.ndarray,
    S_target: np.ndarray,
    S_down: np.ndarray,
    current_profile: pd.DataFrame,
    target_profile: pd.DataFrame,
    downstream_profile: pd.DataFrame,
    updated_profile: pd.DataFrame,
    freq: np.ndarray,
    cfg: TiltConfig,
    cal_mask: np.ndarray,
    resolved_fmin: Optional[np.ndarray] = None,
    resolved_fmax: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Apply the spectral-shape correction without changing variance.

    Supported modes relevant to the three requested calibration variants:

    ``length_tilt`` / ``length``
        Use log(L_down/L_target) as the tilt control signal.  No band-energy
        correction is applied.  The resolved-band area is normalized once
        after this function returns.

    ``length_tilt_bands`` / ``length_bands``
        Same length-error-directed tilt, plus an optional band-energy correction
        inside each height's resolved frequency range.

    ``guarded_centroid`` / ``centroid_guarded``
        Use centroid difference mu_target - mu_downstream only if its implied
        tilt direction does not contradict log(L_down/L_target).  If the two
        signs conflict, use log(L_down/L_target).  No band-energy correction is
        applied unless the mode name also contains ``bands``.

    Legacy modes are preserved:
        ``moment`` uses centroid-only tilt, ``bands`` uses band correction only,
        and ``hybrid`` uses centroid tilt plus band correction.
    """
    S_new = np.maximum(S_current.copy(), FLOOR)
    f_update_min = max(cfg.f_min, float(freq[0]))
    f_update_max = min(cfg.f_max_update, float(freq[-1]))
    edges = make_log_bands(freq, f_update_min, f_update_max, cfg.n_bands)

    metrics: Dict[str, np.ndarray] = {}
    metrics["band_edges"] = edges
    E_t = band_energies(S_target, freq, edges)
    E_d = band_energies(S_down, freq, edges)
    E_c = band_energies(S_current, freq, edges)
    metrics["E_target"] = E_t
    metrics["E_downstream"] = E_d
    metrics["E_current"] = E_c

    if resolved_fmin is None:
        centroid_fmin = np.full(S_current.shape[1], f_update_min, dtype=float)
    else:
        centroid_fmin = np.maximum(np.asarray(resolved_fmin, float), f_update_min)
    if resolved_fmax is None:
        centroid_fmax = np.full(S_current.shape[1], f_update_max, dtype=float)
    else:
        centroid_fmax = np.minimum(np.asarray(resolved_fmax, float), f_update_max)

    mu_t = spectral_centroid_logf_heightwise(S_target, freq, centroid_fmin, centroid_fmax)
    mu_d = spectral_centroid_logf_heightwise(S_down, freq, centroid_fmin, centroid_fmax)
    mu_c = spectral_centroid_logf_heightwise(S_current, freq, centroid_fmin, centroid_fmax)
    metrics["mu_logf_target"] = mu_t
    metrics["mu_logf_downstream"] = mu_d
    metrics["mu_logf_current"] = mu_c
    metrics["mu_delta_target_minus_downstream"] = mu_t - mu_d

    ncomp, nH, _ = S_current.shape
    nan2 = np.full((ncomp, nH), np.nan, dtype=float)
    metrics["L_target"] = nan2.copy()
    metrics["L_downstream"] = nan2.copy()
    metrics["L_current"] = nan2.copy()
    metrics["log_L_down_over_target"] = nan2.copy()
    metrics["tilt_control_used"] = nan2.copy()
    # 0=no tilt, 1=centroid, 2=length-ratio, 3=twoband/legacy, 4=band-only
    metrics["tilt_control_source"] = np.zeros((ncomp, nH), dtype=float)
    metrics["centroid_length_contradiction"] = np.zeros((ncomp, nH), dtype=float)

    lnf = np.log(np.maximum(freq, 1e-12))
    mode_raw = cfg.mode.lower().replace("-", "_").strip()
    mode_tokens = set(mode_raw.split("_"))

    length_modes = {"length", "length_tilt", "length_only", "l_tilt"}
    length_band_modes = {"length_bands", "length_tilt_bands", "length_hybrid", "l_tilt_bands"}
    guarded_modes = {"guarded", "guarded_centroid", "centroid_guarded", "centroid_guard"}
    guarded_band_modes = {"guarded_bands", "guarded_centroid_bands", "centroid_guarded_bands"}
    centroid_modes = {"moment", "centroid", "centroid_tilt"}
    centroid_band_modes = {"hybrid", "moment_bands", "centroid_bands", "centroid_hybrid"}
    band_only_modes = {"bands", "band", "band_energy"}

    uses_length_tilt = mode_raw in length_modes or mode_raw in length_band_modes
    uses_guarded_centroid = mode_raw in guarded_modes or mode_raw in guarded_band_modes
    uses_centroid_tilt = mode_raw in centroid_modes or mode_raw in centroid_band_modes
    uses_band_correction = mode_raw in band_only_modes or mode_raw in length_band_modes or mode_raw in guarded_band_modes or mode_raw in centroid_band_modes

    # A friendly alias: a mode containing both "length" and "bands" is treated
    # as the requested variant 2 even if the exact string is not listed above.
    if "length" in mode_tokens and "bands" in mode_tokens:
        uses_length_tilt = True
        uses_band_correction = True
    elif "length" in mode_tokens:
        uses_length_tilt = True
    if "guarded" in mode_tokens and "bands" in mode_tokens:
        uses_guarded_centroid = True
        uses_band_correction = True
    elif "guarded" in mode_tokens:
        uses_guarded_centroid = True

    for comp in cfg.components:
        if comp not in COMPONENTS:
            continue
        ci = COMP_INDEX[comp]
        Lcol = L_COL[comp]
        moment_relax_i = component_parameter_value(
            cfg.moment_relax, comp, "moment_relax"
        )
        max_log_tilt_i = component_parameter_value(
            cfg.max_log_tilt, comp, "max_log_tilt"
        )
        metrics["L_target"][ci, :] = target_profile[Lcol].to_numpy(float)
        metrics["L_downstream"][ci, :] = downstream_profile[Lcol].to_numpy(float)
        metrics["L_current"][ci, :] = current_profile[Lcol].to_numpy(float)
        for j in range(S_new.shape[1]):
            if not cal_mask[j]:
                continue
            Sj = np.maximum(S_new[ci, j, :].copy(), FLOOR)

            # Active MannHybrid frequency range (the Euston workflow uses the
            # common maximumFrequency ceiling at every height).
            fmin_j = f_update_min if resolved_fmin is None else max(f_update_min, float(resolved_fmin[j]))
            fmax_j = f_update_max if resolved_fmax is None else min(f_update_max, float(resolved_fmax[j]))
            if fmax_j <= fmin_j or np.count_nonzero((freq >= fmin_j) & (freq <= fmax_j)) < 2:
                continue
            f_mask_j = (freq >= fmin_j) & (freq <= fmax_j)

            # Variance is deliberately not changed here.  This function
            # applies spectral shape only; run_calibration performs the
            # single resolved-band normalization to the Wong-updated
            # variance after all component tilts have been applied.

            # Directional controls.
            dmu = float(mu_t[ci, j] - mu_d[ci, j])
            Lt = float(target_profile[Lcol].iloc[j])
            Ld = float(downstream_profile[Lcol].iloc[j])
            lerr = _safe_log_ratio(Ld, Lt, default=0.0)
            metrics["log_L_down_over_target"][ci, j] = lerr

            control = 0.0
            source = 0
            contradiction = False

            if uses_length_tilt:
                control = lerr
                source = 2
            elif uses_guarded_centroid:
                s_mu = _direction(dmu)
                s_L = _direction(lerr)
                if s_mu != 0 and s_L != 0 and s_mu != s_L:
                    # Centroid and integral-length signals imply opposite tilt
                    # directions.  Use the physically direct L-error signal.
                    control = lerr
                    source = 2
                    contradiction = True
                elif s_mu == 0 and s_L != 0:
                    # Centroid has no clear direction, but L does.
                    control = lerr
                    source = 2
                elif s_mu != 0:
                    control = dmu
                    source = 1
                else:
                    control = 0.0
                    source = 0
            elif uses_centroid_tilt:
                control = dmu
                source = 1
            elif mode_raw == "twoband":
                source = 3
            elif mode_raw in band_only_modes:
                source = 4

            metrics["tilt_control_used"][ci, j] = control
            metrics["tilt_control_source"][ci, j] = float(source)
            metrics["centroid_length_contradiction"][ci, j] = 1.0 if contradiction else 0.0

            # Shape update only in the resolved/recoverable band.
            log_mult = np.zeros_like(freq)
            if source in {1, 2} and abs(control) > NUMERICAL_ZERO:
                # Positive control means: downstream eddies are too long for the
                # L-error controller, or downstream centroid is too low for the
                # centroid controller.  In both cases, boost high f and suppress
                # low f.  Negative control does the opposite.
                var_lnf = max(float(np.var(lnf[f_mask_j])), 1e-12)
                pivot = float(mu_c[ci, j]) if np.isfinite(mu_c[ci, j]) else float(np.mean(lnf[f_mask_j]))
                log_mult += moment_relax_i * control * (lnf - pivot) / var_lnf

            if uses_band_correction:
                if cfg.band_correction_uses_resolved_range:
                    centers, band_log = _local_band_log_update(
                        S_target[ci, j, :], S_down[ci, j, :], freq,
                        fmin_j, fmax_j, cfg.n_bands,
                        cfg.shape_relax, cfg.max_log_band_update,
                    )
                else:
                    band_log = cfg.shape_relax * np.clip(
                        np.log(np.maximum(E_t[ci, j, :], FLOOR) / np.maximum(E_d[ci, j, :], FLOOR)),
                        -cfg.max_log_band_update,
                        cfg.max_log_band_update,
                    )
                    centers = np.sqrt(edges[:-1] * edges[1:])
                log_mult += np.interp(lnf, np.log(centers), band_log, left=band_log[0], right=band_log[-1])

            if mode_raw == "twoband":
                # Legacy two-band mode retained for controlled comparisons.
                Uj = max(float(target_profile["U"].iloc[j]), 1e-12)
                Lj = max(float(target_profile[Lcol].iloc[j]), 1e-12)
                fsplit = np.clip(Uj / Lj, f_update_min*1.2, f_update_max/1.2)
                low = (freq >= fmin_j) & (freq < fsplit)
                high = (freq >= fsplit) & (freq <= fmax_j)
                if np.count_nonzero(low) >= 2 and np.count_nonzero(high) >= 2:
                    Et_low = float(trapz(S_target[ci, j, low], x=freq[low])); Ed_low = float(trapz(S_down[ci, j, low], x=freq[low]))
                    Et_high = float(trapz(S_target[ci, j, high], x=freq[high])); Ed_high = float(trapz(S_down[ci, j, high], x=freq[high]))
                    r_low = math.log(max(Et_low,FLOOR)/max(Ed_low,FLOOR))
                    r_high = math.log(max(Et_high,FLOOR)/max(Ed_high,FLOOR))
                    log_mult[low] += cfg.shape_relax * np.clip(r_low, -cfg.max_log_band_update, cfg.max_log_band_update)
                    log_mult[high] += cfg.shape_relax * np.clip(r_high, -cfg.max_log_band_update, cfg.max_log_band_update)

            log_mult = np.clip(log_mult, -max_log_tilt_i, max_log_tilt_i)

            # Apply only inside local resolved update band; leave unresolved tail unchanged.
            Sj2 = Sj.copy()
            Sj2[f_mask_j] = np.maximum(Sj[f_mask_j] * np.exp(log_mult[f_mask_j]), FLOOR)

            # Do not normalize variance inside the tilt operation.  A single
            # common-band normalization is applied by run_calibration below.
            S_new[ci, j, :] = np.maximum(Sj2, FLOOR)

    return S_new, metrics


def make_melaku_8panel_plot(target: pd.DataFrame, downstream: pd.DataFrame, current: pd.DataFrame, updated: pd.DataFrame, H: float, path: Path, title: str) -> None:
    cols = ["U", "Iu", "Iv", "Iw", "uwStress", "Lu", "Lv", "Lw"]
    labels = [r"$U/U_H$", r"$I_u$", r"$I_v$", r"$I_w$", r"$u'w'/U_H^2$", r"$L_u/H$", r"$L_v/H$", r"$L_w/H$"]
    zH = target["z"].to_numpy(float)/H
    UH = np.interp(H, target["z"], target["U"])
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), sharey=True)
    axes = axes.ravel()
    for k, col in enumerate(cols):
        ax = axes[k]
        def norm(df):
            vals = df[col].to_numpy(float)
            if col == "U":
                return vals / max(UH, 1e-12)
            if col == "uwStress":
                return vals / max(UH**2, 1e-12)
            if col.startswith("L"):
                return vals / H
            return vals
        ax.plot(norm(target), zH, "k--", lw=1.5, label="Target")
        ax.plot(norm(downstream), downstream["z"].to_numpy(float)/H, "r-", lw=1.5, label="LES downstream")
        ax.plot(norm(current), current["z"].to_numpy(float)/H, color="0.45", ls=":", lw=1.4, label="Current input")
        ax.plot(norm(updated), updated["z"].to_numpy(float)/H, color="C0", ls="-.", lw=1.4, label="Updated input")
        ax.set_xlabel(labels[k])
        ax.set_title(f"({chr(97+k)})")
        ax.grid(True, alpha=0.3)
        if k in [0,4]:
            ax.set_ylabel(r"$z/H$")
        if k == 0:
            ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=[0,0,1,0.96])
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_band_heatmap(metrics: Dict[str, np.ndarray], target: pd.DataFrame, cfg: TiltConfig, path: Path) -> None:
    edges = metrics["band_edges"]
    E_t = metrics["E_target"]
    E_d = metrics["E_downstream"]
    zH = target["z"].to_numpy(float)/cfg.building_height
    comps = list(cfg.components)
    fig, axes = plt.subplots(len(comps), 1, figsize=(10, 3.0*len(comps)), squeeze=False)
    for r, comp in enumerate(comps):
        ci = COMP_INDEX[comp]
        ratio = np.log10(np.maximum(E_d[ci,:,:], FLOOR) / np.maximum(E_t[ci,:,:], FLOOR))
        ax = axes[r,0]
        im = ax.imshow(ratio, aspect="auto", origin="lower", extent=[0, len(edges)-1, zH[0], zH[-1]], vmin=-1, vmax=1)
        ax.set_title(f"log10(downstream/target) band energy: {comp}")
        ax.set_ylabel("z/H")
        ax.set_xlabel("log-frequency band")
        ax.set_xticks(np.arange(len(edges)-1)+0.5)
        ax.set_xticklabels([f"{edges[i]:.2g}-{edges[i+1]:.2g}" for i in range(len(edges)-1)], rotation=35, ha="right")
        fig.colorbar(im, ax=ax, label="log10 ratio")
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def make_spectra_comparison_plot(freq: np.ndarray, S_target: np.ndarray, S_down: np.ndarray, S_current: np.ndarray, S_updated: np.ndarray, target: pd.DataFrame, cfg: TiltConfig, path: Path) -> None:
    heights = [0.5*cfg.building_height, 1.0*cfg.building_height, 2.0*cfg.building_height, 3.0*cfg.building_height]
    comps = list(cfg.components)
    fig, axes = plt.subplots(len(comps), len(heights), figsize=(4.2*len(heights), 3.4*len(comps)), squeeze=False)
    z = target["z"].to_numpy(float)
    for r, comp in enumerate(comps):
        ci = COMP_INDEX[comp]
        for c, zh in enumerate(heights):
            j = int(np.argmin(np.abs(z - zh)))
            ax = axes[r,c]
            ax.loglog(freq, S_target[ci,j,:], "k--", lw=1.4, label="target")
            ax.loglog(freq, S_down[ci,j,:], "r-", lw=1.1, label="down")
            ax.loglog(freq, S_current[ci,j,:], color="0.45", ls=":", lw=1.1, label="current")
            ax.loglog(freq, S_updated[ci,j,:], color="C0", ls="-.", lw=1.2, label="updated")
            ax.grid(True, which="both", alpha=0.3)
            ax.set_title(f"{comp}, z/H={z[j]/cfg.building_height:.2f}")
            if r == len(comps)-1:
                ax.set_xlabel("f [Hz]")
            if c == 0:
                ax.set_ylabel(f"S{comp}{comp}")
            if r == 0 and c == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=170)
    plt.close(fig)



def save_metrics_csv(metrics: Dict[str, np.ndarray], target: pd.DataFrame, cfg: TiltConfig, path: Path) -> None:
    edges = metrics["band_edges"]
    rows = []
    z = target["z"].to_numpy(float)
    E_t, E_d, E_c = metrics["E_target"], metrics["E_downstream"], metrics["E_current"]
    mu_t, mu_d, mu_c = metrics["mu_logf_target"], metrics["mu_logf_downstream"], metrics["mu_logf_current"]

    def mval(name: str, ci: int, j: int, default: float = math.nan) -> float:
        arr = metrics.get(name)
        if arr is None:
            return default
        try:
            return float(arr[ci, j])
        except Exception:
            return default

    for comp in COMPONENTS:
        ci = COMP_INDEX[comp]
        for j, zj in enumerate(z):
            base = {
                "component": comp,
                "z": zj,
                "z_over_H": zj/cfg.building_height,
                "mu_logf_target": mu_t[ci,j],
                "mu_logf_downstream": mu_d[ci,j],
                "mu_logf_current": mu_c[ci,j],
                "mu_delta_target_minus_downstream": mval("mu_delta_target_minus_downstream", ci, j),
                "L_target": mval("L_target", ci, j),
                "L_downstream": mval("L_downstream", ci, j),
                "L_current": mval("L_current", ci, j),
                "log_L_down_over_target": mval("log_L_down_over_target", ci, j),
                "tilt_control_used": mval("tilt_control_used", ci, j),
                "tilt_control_source": mval("tilt_control_source", ci, j),
                "centroid_length_contradiction": mval("centroid_length_contradiction", ci, j),
            }
            for b in range(len(edges)-1):
                row = dict(base)
                row.update({
                    "band": b,
                    "f_low": edges[b],
                    "f_high": edges[b+1],
                    "E_target": E_t[ci,j,b],
                    "E_downstream": E_d[ci,j,b],
                    "E_current": E_c[ci,j,b],
                    "log10_down_over_target": math.log10(max(E_d[ci,j,b], FLOOR)/max(E_t[ci,j,b], FLOOR)),
                })
                rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def update_uw_cospectrum_area(C_current: np.ndarray, C_target: np.ndarray, C_down: np.ndarray, S_updated: np.ndarray, freq: np.ndarray, current: pd.DataFrame, target: pd.DataFrame, downstream: pd.DataFrame, updated: pd.DataFrame, cfg: TiltConfig, cal_mask: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
    """Wong-style signed update of <u'w'> and area-normalization of Cuw.

    The profile update is already performed in update_profile_wong.  This function
    keeps the co-spectrum shape stable but rescales its signed area to the updated
    profile's uwStress, with pointwise realizability clipping against the updated
    Suu/Sww spectra.
    """
    C_base = np.asarray(C_current, float).copy()
    # For uncalibrated/support region, keep the current co-spectrum unless it is absent/zero.
    if C_base.shape != C_target.shape:
        C_base = C_target.copy()
    # Use target shape where current has essentially no signed area.
    for j in range(C_base.shape[0]):
        if abs(float(trapz(C_base[j, :], x=freq))) < 1e-14 and abs(float(trapz(C_target[j, :], x=freq))) > 1e-14:
            C_base[j, :] = C_target[j, :]
    C_updated = C_base.copy()
    uw_goal = current["uwStress"].to_numpy(float).copy()
    uw_goal[cal_mask] = updated["uwStress"].to_numpy(float)[cal_mask]
    C_updated = normalize_cospectrum_to_stress(C_updated, freq, uw_goal, S_updated[0], S_updated[2], cfg.rho_uw_limit)
    rows = []
    for j, zval in enumerate(updated["z"].to_numpy(float)):
        area_cur = float(trapz(C_current[j, :], x=freq)) if C_current is not None else math.nan
        area_tar = float(trapz(C_target[j, :], x=freq)) if C_target is not None else math.nan
        area_down = float(trapz(C_down[j, :], x=freq)) if C_down is not None else math.nan
        area_new = float(trapz(C_updated[j, :], x=freq))
        rows.append({
            "z": zval,
            "z_over_H": zval/cfg.building_height,
            "uw_current_profile": float(current["uwStress"].iloc[j]),
            "uw_target_profile": float(target["uwStress"].iloc[j]),
            "uw_downstream_profile": float(downstream["uwStress"].iloc[j]),
            "uw_updated_profile": float(updated["uwStress"].iloc[j]),
            "uw_current_cospec_integral": area_cur,
            "uw_target_cospec_integral": area_tar,
            "uw_downstream_cospec_integral": area_down,
            "uw_updated_cospec_integral": area_new,
            "used_in_calibration": bool(cal_mask[j]),
        })
    return C_updated, pd.DataFrame(rows)


def plot_uw_cospectra(freq: np.ndarray, C_target: np.ndarray, C_down: np.ndarray, C_current: np.ndarray, C_updated: np.ndarray, target: pd.DataFrame, cfg: TiltConfig, path: Path) -> None:
    heights = [0.5*cfg.building_height, 1.0*cfg.building_height, 2.0*cfg.building_height, 3.0*cfg.building_height]
    z = target["z"].to_numpy(float)
    fig, axes = plt.subplots(1, len(heights), figsize=(4.2*len(heights), 3.4), squeeze=False)
    for c, zh in enumerate(heights):
        j = int(np.argmin(np.abs(z - zh)))
        ax = axes[0, c]
        ax.semilogx(freq, C_target[j, :], "k--", lw=1.4, label="target")
        ax.semilogx(freq, C_down[j, :], "r-", lw=1.1, label="down")
        ax.semilogx(freq, C_current[j, :], color="0.45", ls=":", lw=1.1, label="current")
        ax.semilogx(freq, C_updated[j, :], color="C0", ls="-.", lw=1.2, label="updated")
        ax.axhline(0.0, color="0.3", lw=0.8)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(f"Cuw, z/H={z[j]/cfg.building_height:.2f}")
        ax.set_xlabel("f [Hz]")
        if c == 0:
            ax.set_ylabel("Cuw")
            ax.legend(fontsize=8)
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_calibration(cfg: TiltConfig) -> int:
    cfg = enforce_mannhybrid_inflow_mode(cfg)
    case_dir = cfg.case_dir
    profile_dir = case_dir / cfg.profile_dir_rel
    iteration, itdir = next_iteration_dir(case_dir, cfg.log_dir_name)
    print(f"MannHybrid spectral tilt calibration — iteration {iteration}", flush=True)
    print(json.dumps({k: (str(v) if isinstance(v, Path) else v) for k,v in asdict(cfg).items()}, indent=2), flush=True)

    target = read_profile_file(profile_dir / cfg.target_profile_file)
    current = read_profile_file(profile_dir / cfg.active_profile_file)
    # Use target grid as master.
    z = target["z"].to_numpy(float)
    current = pd.DataFrame({"z": z, **{col: np.interp(z, current["z"], current[col]) for col in PROFILE_COLS_UW[1:]}})
    cal_mask = (z >= cfg.z_cal_min) & (z <= cfg.z_cal_max)

    # Active and target spectra.  If missing, initialize from von-Karman-like spectra.
    active_spectra_path = profile_dir / cfg.active_spectra_file
    target_spectra_path = profile_dir / cfg.target_spectra_file
    if active_spectra_path.exists():
        sp_current = read_auto_spectra(active_spectra_path)
        nF = sp_current.S.shape[2]
    elif target_spectra_path.exists():
        sp_current = read_auto_spectra(target_spectra_path)
        nF = sp_current.S.shape[2]
    else:
        nF = int(parse_set_up(case_dir).get("nFreq", 4096))
        sp_current = SpectraProfile(z=z, S=make_vonkarman_like_spectra(current, freq_array_from_fmax(infer_fmax(case_dir, nF), nF)), uw_stress=current["uwStress"].to_numpy(float))
    fmax = infer_fmax(case_dir, nF)
    freq = freq_array_from_fmax(fmax, nF)
    # Map spectra to target z if needed.
    def map_spectra(sp: SpectraProfile, fallback_profile: pd.DataFrame) -> np.ndarray:
        S = sp.S
        if len(sp.z) == len(z) and np.allclose(sp.z, z, rtol=0, atol=1e-9):
            return S
        out = np.zeros((3, len(z), S.shape[2]), float)
        for ci in range(3):
            for k in range(S.shape[2]):
                out[ci, :, k] = np.interp(z, sp.z, S[ci, :, k])
        return np.maximum(out, FLOOR)
    S_current = map_spectra(sp_current, current)
    if target_spectra_path.exists():
        S_target = map_spectra(read_auto_spectra(target_spectra_path), target)
    else:
        S_target = make_vonkarman_like_spectra(target, freq)

    # Active and target u-w co-spectra.  These are optional, but if present
    # they will be kept consistent with the updated uwStress profile.
    active_uw_path = profile_dir / cfg.active_uw_cospectrum_file
    target_uw_path = profile_dir / cfg.target_uw_cospectrum_file
    if target_uw_path.exists():
        Cuw_target = map_cospectra_to_z(read_uw_cospectra(target_uw_path), z)
    else:
        Cuw_target = make_default_uw_cospectrum(target, freq, S_target, cfg.rho_uw_limit)
    if active_uw_path.exists():
        Cuw_current = map_cospectra_to_z(read_uw_cospectra(active_uw_path), z)
    else:
        # Prefer the target shape on the first iteration; it is then rescaled
        # to the active/current uwStress profile.
        Cuw_current = normalize_cospectrum_to_stress(Cuw_target, freq, current["uwStress"].to_numpy(float), S_current[0], S_current[2], cfg.rho_uw_limit)

    # Read LES probe data and estimate spectra/statistics.
    time_full, vel_full, positions, probe_files = read_all_probe_segments(case_dir, cfg.probes_name)
    time, vel, tinfo = clean_time_history(time_full, vel_full, cfg.burn_in_time)
    record_duration = float(time[-1] - time[0]) if len(time) > 1 else 0.0
    if int(tinfo.get("n_samples", 0)) < cfg.min_samples or record_duration < cfg.min_record_duration:
        raise RuntimeError(
            "Insufficient post-burn-in data for spectral-tilt calibration: "
            f"n_samples={tinfo.get('n_samples')}, duration={record_duration:.6g}s, "
            f"burn_in_time={cfg.burn_in_time}. "
            "Set MST_BURN_IN_TIME to the actual statistics start time, e.g. "
            "the burn_in_time from log/downstreamCalibration/sim_init.json, "
            "or reduce MST_MIN_SAMPLES/MST_MIN_RECORD_DURATION only for debugging."
        )
    downstream, S_down, smeta = compute_downstream_profiles_and_spectra(time, vel, positions, z, cfg.nperseg, freq, cfg.l_method)
    Cuw_down, cuw_meta = compute_downstream_uw_cospectrum(time, vel, positions, z, cfg.nperseg, freq)

    if cfg.use_windlespy_resolved_band:
        resolved_fmin, resolved_fmax, configured_maximum_frequencies = compute_windlespy_resolved_limits(case_dir, target, time, freq)
        pd.DataFrame({
            "z": z,
            "z_over_H": z/float(cfg.building_height),
            "resolved_fmin": resolved_fmin,
            "resolved_fmax": resolved_fmax,
            "configured_maximum_frequency": configured_maximum_frequencies,
        }).to_csv(itdir / "data" / "windlespy_resolved_frequency_limits.csv", index=False)
    else:
        resolved_fmin = np.full(len(z), max(cfg.f_min, float(freq[0])), dtype=float)
        resolved_fmax = np.full(len(z), min(cfg.f_max_update, float(freq[-1])), dtype=float)

    updated_profile = update_profile_wong(
        current, target, downstream, cfg, cal_mask,
        freq=freq,
        S_current=S_current,
        S_target=S_target,
        S_down=S_down,
        Cuw_current=Cuw_current,
        Cuw_target=Cuw_target,
        Cuw_down=Cuw_down,
        resolved_fmin=resolved_fmin,
        resolved_fmax=resolved_fmax,
    )
    S_updated, metrics = apply_spectral_tilt(S_current, S_target, S_down, current, target, downstream, updated_profile, freq, cfg, cal_mask, resolved_fmin, resolved_fmax)
    # Align spectra area with updated profile intensity columns over the resolved band.
    for comp in COMPONENTS:
        ci = COMP_INDEX[comp]
        area = (updated_profile[I_COL[comp]].to_numpy(float) * updated_profile["U"].to_numpy(float))**2
        S_updated[ci, :, :] = rescale_heightwise_resolved_area(S_updated[ci, :, :], freq, area, resolved_fmin, resolved_fmax)
    # Set updated profile I exactly from final resolved-band spectral area.
    for comp in COMPONENTS:
        ci = COMP_INDEX[comp]
        area = integrate_heightwise_between(freq, S_updated[ci, :, :], resolved_fmin, resolved_fmax)
        updated_profile[I_COL[comp]] = np.sqrt(np.maximum(area, 0.0)) / np.maximum(updated_profile["U"], 1e-12)

    # The spectrum/profile synchronisation above should preserve the requested
    # absolute TI ceiling exactly. Fail rather than silently write an
    # inconsistent profile if numerical integration ever violates it.
    active_intensities = updated_profile.loc[
        cal_mask, ["Iu", "Iv", "Iw"]
    ].to_numpy(float)
    if np.any(~np.isfinite(active_intensities)) or np.any(
        active_intensities > cfg.max_turbulence_intensity * (1.0 + 1.0e-9)
    ):
        raise RuntimeError(
            "Final active profile violates max_turbulence_intensity after "
            "spectral-area synchronisation"
        )

    # Preserve the original final realizability limiter only when Reynolds
    # shear stress is being calibrated.  Otherwise leave the profile column
    # unchanged so it remains a faithful target-versus-measured diagnostic.
    if cfg.update_uw_stress:
        sig_u_final = np.maximum(updated_profile["Iu"].to_numpy(float) * updated_profile["U"].to_numpy(float), 1e-12)
        sig_w_final = np.maximum(updated_profile["Iw"].to_numpy(float) * updated_profile["U"].to_numpy(float), 1e-12)
        bound_final = cfg.rho_uw_limit * sig_u_final * sig_w_final
        updated_profile["uwStress"] = np.clip(updated_profile["uwStress"].to_numpy(float), -bound_final, bound_final)

    if cfg.update_uw_cospectrum:
        if cfg.uw_cospectrum_resolved_band_only:
            # Preserve the current/target Kaimal-like Cuw shape; only rescale its resolved-band area.
            Cbase = Cuw_current.copy()
            # If active shape is zero/unusable at a height, fall back to target shape.
            for jj in range(Cbase.shape[0]):
                if abs(integrate_1d_between(freq, Cbase[jj, :], resolved_fmin[jj], resolved_fmax[jj])) < 1e-14:
                    Cbase[jj, :] = Cuw_target[jj, :]
            Cuw_updated = normalize_cospectrum_resolved_band(
                Cbase, freq, updated_profile["uwStress"].to_numpy(float),
                S_updated[0], S_updated[2], cfg.rho_uw_limit,
                resolved_fmin, resolved_fmax,
            )
            rows = []
            for jj, zval in enumerate(z):
                rows.append({
                    "z": float(zval), "z_over_H": float(zval/cfg.building_height),
                    "uw_current_profile": float(current["uwStress"].iloc[jj]),
                    "uw_target_profile": float(target["uwStress"].iloc[jj]),
                    "uw_downstream_profile": float(downstream["uwStress"].iloc[jj]),
                    "uw_updated_profile": float(updated_profile["uwStress"].iloc[jj]),
                    "resolved_fmin": float(resolved_fmin[jj]),
                    "resolved_fmax": float(resolved_fmax[jj]),
                    "uw_current_cospec_integral_resolved": integrate_1d_between(freq, Cuw_current[jj,:], resolved_fmin[jj], resolved_fmax[jj]),
                    "uw_target_cospec_integral_resolved": integrate_1d_between(freq, Cuw_target[jj,:], resolved_fmin[jj], resolved_fmax[jj]),
                    "uw_downstream_cospec_integral_resolved": integrate_1d_between(freq, Cuw_down[jj,:], resolved_fmin[jj], resolved_fmax[jj]),
                    "uw_updated_cospec_integral_resolved": integrate_1d_between(freq, Cuw_updated[jj,:], resolved_fmin[jj], resolved_fmax[jj]),
                    "used_in_calibration": bool(cal_mask[jj]),
                })
            uw_metrics_df = pd.DataFrame(rows)
        else:
            Cuw_updated, uw_metrics_df = update_uw_cospectrum_area(Cuw_current, Cuw_target, Cuw_down, S_updated, freq, current, target, downstream, updated_profile, cfg, cal_mask)
    else:
        Cuw_updated = Cuw_current.copy()
        uw_metrics_df = pd.DataFrame()

    # Write snapshots before overwriting active files.
    target.to_csv(itdir / "data" / "target_profile.csv", index=False)
    current.to_csv(itdir / "data" / "current_profile.csv", index=False)
    downstream.to_csv(itdir / "data" / "downstream_profile.csv", index=False)
    updated_profile.to_csv(itdir / "data" / "updated_profile.csv", index=False)
    save_metrics_csv(metrics, target, cfg, itdir / "data" / "band_energy_metrics.csv")
    uw_metrics_df.to_csv(itdir / "data" / "uw_stress_and_cospectrum_metrics.csv", index=False)
    with (itdir / "data" / "probe_time_info.json").open("w") as f:
        json.dump({"time_info": tinfo, "spectra_meta": smeta, "uw_cospectra_meta": cuw_meta, "probe_files": [str(p) for p in probe_files]}, f, indent=2)
    for fname in [cfg.active_profile_file, cfg.target_profile_file, cfg.active_spectra_file, cfg.target_spectra_file, cfg.active_uw_cospectrum_file, cfg.target_uw_cospectrum_file]:
        p = profile_dir / fname
        if p.exists():
            shutil.copy2(p, itdir / "inputs" / f"{fname}_before_update")

    # Write active inputs.
    write_profile_file(profile_dir / cfg.active_profile_file, updated_profile)
    write_auto_spectra(profile_dir / cfg.active_spectra_file, z, S_updated, uw_stress=updated_profile["uwStress"].to_numpy(float))
    if cfg.update_uw_cospectrum:
        write_uw_cospectra(profile_dir / cfg.active_uw_cospectrum_file, z, Cuw_updated, uw_stress=updated_profile["uwStress"].to_numpy(float))
    write_profile_file(itdir / "inputs" / "profile_updated", updated_profile)
    write_auto_spectra(itdir / "inputs" / "spectraProfile_updated", z, S_updated, uw_stress=updated_profile["uwStress"].to_numpy(float))
    if cfg.update_uw_cospectrum:
        write_uw_cospectra(itdir / "inputs" / "uwCoSpectrumProfile_updated", z, Cuw_updated, uw_stress=updated_profile["uwStress"].to_numpy(float))

    if cfg.write_plots:
        make_melaku_8panel_plot(target, downstream, current, updated_profile, cfg.building_height, itdir / "plots" / f"iteration{iteration:02d}_profiles_8panel.png", f"MannHybrid spectral-tilt calibration — iteration {iteration}")
        make_band_heatmap(metrics, target, cfg, itdir / "plots" / f"iteration{iteration:02d}_band_energy_heatmap.png")
        make_spectra_comparison_plot(freq, S_target, S_down, S_current, S_updated, target, cfg, itdir / "plots" / f"iteration{iteration:02d}_spectra_comparison.png")
        if cfg.update_uw_cospectrum:
            plot_uw_cospectra(freq, Cuw_target, Cuw_down, Cuw_current, Cuw_updated, target, cfg, itdir / "plots" / f"iteration{iteration:02d}_uw_cospectrum_comparison.png")

    # Return non-zero so SLURM loop continues with MannHybridTurb + LES.
    written = "profile and spectraProfile"
    if cfg.update_uw_cospectrum:
        written += " and uwCoSpectrumProfile"
    print(f"Wrote updated {written} to {profile_dir}", flush=True)
    print(f"Iteration snapshot: {itdir}", flush=True)
    return 1 if cfg.exit_always_continue else 0


if __name__ == "__main__":
    try:
        raise SystemExit(run_calibration(default_config()))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
