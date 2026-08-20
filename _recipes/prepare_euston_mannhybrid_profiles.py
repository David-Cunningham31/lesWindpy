#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare Euston Tower wind-profile inputs for MannHybridTurb.

Compatible with the current public mannHybridInflow main branch at
commit 9068d270c2448005d894d47c09fa57ac6c638b55
(VERSION: v1.2.1-time-loop-guard; README: 1.2.0).

The script:

1. reads the existing ESDU profile

       z U Iu Iv Iw Lu Lv Lw [uwStress]

2. obtains the distinct inlet boundary-face-centre elevations used by
   MannHybridTurb (these are boundary face centres, not owner-cell centres);
3. linearly maps the ESDU quantities to those elevations;
4. optionally scales the complete mean-speed profile to a prescribed target
   mean speed at a prescribed reference height above ground, while preserving
   turbulence intensities and integral length scales;
5. follows ``targetSpectraSource`` and ``uwCoSpectrumSource`` directly from
   MannHybridTurbDict, without editing that dictionary;
6. when a spectra table is required, samples the von Karman formula implemented
   by current MannHybridTurb and, by default, rescales each component to retain
   its requested variance over the finite active frequency band;
7. constructs a signed, bounded Kaimal-shaped u-w co-spectrum only when
   ``uwCoSpectrumSource=tabulated``; and
8. writes only the files that current MannHybridTurb reads for that mode:

       profile                                      always
       spectraProfile                               when required
       uwCoSpectrumProfile                          tabulated u-w only

   By default, matching target* copies are retained for the user's existing
   downstream calibration workflow. Files inactive in the selected dictionary
   mode are neither created nor overwritten.

Important: in the audited public MannHybrid version,
``uwCoSpectrumSource=none`` imposes a zero target one-point u-w co-spectrum. It
does not retain the native Mann one-point co-spectrum. Preserving native Mann
coherency while recolouring the auto-spectra would require a C++ utility mode;
it cannot be selected by changing Python output files alone.

The original ESDU profile is preserved once as profile.esduSource. On reruns,
that preserved source is used automatically, so already-interpolated output is
not recursively re-interpolated.

Spyder use:

1. edit the clearly marked ``USER SETTINGS - SPYDER`` block below;
2. set ``IDE_CASE_PATH`` to the OpenFOAM case;
3. press Run. No command-line arguments are required.

The remaining settings are also command-line defaults, so explicit terminal
arguments still override them. In a terminal, omitting the case path retains
the conventional behavior of using the current working directory. Typical
terminal use from anywhere:

    python prepare_euston_mannhybrid_profiles.py /path/to/case

For a binary mesh, install PyVista/VTK in the Python environment. For an ASCII
polyMesh, the dependency-light streaming backend is available:

    python prepare_euston_mannhybrid_profiles.py /path/to/case --backend ascii

For testing or pre-extracted face-centre elevations:

    python prepare_euston_mannhybrid_profiles.py /path/to/case \
        --heights-file inletFaceCentreZ.txt

Only NumPy is required for the numerical work. PyVista is optional and is used
only for mesh extraction.

By default spectral normalisation uses the same discrete sum(S*df) convention
as current MannHybridTurb. Use --integration-rule trapezoid only when exact
compatibility with the predecessor Python calibration convention is required.

The resolved-variance option acts on the global
[minimumFrequency, maximumFrequency] band from MannHybridTurbDict. It does not
reproduce the predecessor script's private windlespy, per-height mesh-cutoff
correction; that separate downstream calibration step needs local mesh/filter
inputs which are not present in an ESDU profile.
"""

from __future__ import annotations

import argparse
import gzip
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

cwd = os.path.dirname(os.path.abspath(__file__))
windlespy_path = r"C:\Users\david\OneDrive\Documents\PhD\Year 1"
sys.path.append(windlespy_path)
import windlespy as LES
sys.path.remove(windlespy_path)

# =============================================================================
# USER SETTINGS - SPYDER
# =============================================================================
# Edit IDE_CASE_PATH, then press Run in Spyder. The other values are used
# whenever the corresponding command-line argument is omitted. Relative paths
# are resolved from the selected case, not from the location of this file.
#
# Example on MeluXina:
# IDE_CASE_PATH = Path(
#     r"/mnt/tier2/project/p201464/euston_tower/empty_domain/170-190"
# )
#
# Example on Windows:
# IDE_CASE_PATH = Path(
#     r"C:\Users\david\OneDrive\Documents\PhD\Year 1\NHERI LES Case"
#     r"\OpenFOAM Cases\Empty Domain\170-190"
# )

IDE_CASE_PATH = Path(r"C:\Users\david\OneDrive\Documents\PhD\Year 1\Euston Tower LES\of_cases\empty_domain\330,340")

# Input/output paths. Leave optional paths as None to derive them from the
# OpenFOAM case and MannHybridTurbDict.
IDE_DICTIONARY = Path("constant/MannHybridTurbDict")
IDE_PROFILE_DIRECTORY: Optional[Path] = None
IDE_SOURCE_PROFILE: Optional[Path] = None
IDE_SOURCE_BACKUP_NAME = "profile.esduSource"
IDE_REFRESH_SOURCE_BACKUP = False

# Inlet mesh extraction. IDE_PATCH=None reads patchName from the dictionary.
# "auto" uses PyVista for binary/decomposed meshes when available and falls
# back to the built-in streaming reader for an ASCII constant/polyMesh.
IDE_PATCH: Optional[str] = None
IDE_BACKEND = "auto"                 # "auto", "pyvista", or "ascii"
IDE_FOAM_MARKER = "180.foam"
IDE_HEIGHTS_FILE: Optional[Path] = None
IDE_HEIGHT_COLUMN: Optional[int] = None
IDE_COORDINATE_TOLERANCE: Optional[float] = None
IDE_GROUND_ELEVATION = 0.0           # metres; Euston flat terrain uses 0

# Optional mean-speed scaling. Leave the target as None to retain the absolute
# speeds in profile.esduSource. Otherwise every U value is multiplied by one
# factor so U(IDE_REFERENCE_HEIGHT_AGL) equals the requested target. Iu/Iv/Iw
# and Lu/Lv/Lw remain unchanged; an input ninth-column u'w' scales by factor^2.
# For the supplied RWDI 180-degree run, equivalent choices are approximately
# (4.22 m/s, 120.0 m AGL) or (4.25975 m/s, 125.47 m AGL).

"""
DESIGN WIND SPEEDS @ BUILDING HEIGHT:
FIRST COLUMN IS WIND DIRECTION, 2ND COLUMN IS DESIGN WIND SPEED
C_dir APPLIED AND CONVERTED FROM 10-MINUTE TO 1-HOUR MEAN VALUES
SEE RWDI "WT_Test-ReferenceLoadingData_Base_Loads" EXCEL FOR REFERENCE
    
10	19.92
20	19.19
30	18.76
40	18.76
50	18.62
60	18.62
70	18.70
80	18.27
90	18.35
100	18.20
110	17.53
120	18.10
130	18.68
140	19.42
150	20.00
160	20.42
170	20.75
180	21.17
190	21.83
200	22.49
210	23.16
220	24.79
230	25.39
240	25.80
250	25.51
260	25.43
270	25.25
280	24.57
290	23.89
300	23.21
310	22.44
320	21.68
330	21.57
340	21.22
350	20.71
360	20.36
"""

IDE_TARGET_REFERENCE_MEAN_SPEED: Optional[float] = 21.395  # m/s
IDE_REFERENCE_HEIGHT_AGL: Optional[float] = 125.47       # m

# Profile and spectral construction. Cross-spectrum output is selected
# automatically from MannHybridTurbDict: "tabulated" generates the table,
# "kaimal" lets the utility construct it, and "none" omits it (target Cuw=0
# in the audited public utility).
# There is intentionally no duplicate IDE switch that could contradict the
# OpenFOAM dictionary.
IDE_EXTRAPOLATION = "hold"           # "hold", "error", or "linear"
IDE_SPECTRA_NORMALISATION = "resolved-variance"  # or "native"
IDE_INTEGRATION_RULE = "mannhybrid-sum"          # or "trapezoid"
IDE_STRESS_SOURCE = "dict"           # normally follow uwStressSource
IDE_RHO_UW: Optional[float] = None
IDE_U_STAR: Optional[float] = None
IDE_UNIFORM_UW_STRESS: Optional[float] = None
IDE_MAX_ABS_RHO_UW: Optional[float] = None

# Output and safety controls.
IDE_MAX_TABLE_VALUES = 100_000_000
IDE_PRECISION = 12
IDE_WRITE_TARGET_COPIES = True
IDE_DRY_RUN = False                  # True validates without writing files
# =============================================================================


SCRIPT_VERSION = "1.3.0"
MANNHYBRID_COMMIT = "9068d270c2448005d894d47c09fa57ac6c638b55"
PROFILE_COLUMNS = ("z", "U", "Iu", "Iv", "Iw", "Lu", "Lv", "Lw")
FLOAT_PATTERN = (
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?"
)


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def trapezoid(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    """NumPy 1.x/2.x compatible trapezoidal integration."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x, axis=axis)
    return np.trapz(y, x=x, axis=axis)


def spectral_integral(
    y: np.ndarray,
    frequencies: np.ndarray,
    rule: str,
    axis: int = -1,
) -> np.ndarray:
    """Integrate a uniformly spaced one-sided spectral table."""
    if rule == "trapezoid":
        return trapezoid(y, frequencies, axis=axis)
    if rule != "mannhybrid-sum":
        raise ValueError(f"Unknown spectral integration rule {rule!r}.")
    if frequencies.size < 1:
        raise ValueError("Cannot integrate an empty frequency array.")
    if frequencies.size == 1:
        df = float(frequencies[0])
    else:
        differences = np.diff(frequencies)
        df = float(differences[0])
        if not np.allclose(
            differences,
            df,
            rtol=1.0e-10,
            atol=1.0e-14 * max(abs(df), 1.0),
        ):
            raise ValueError(
                "mannhybrid-sum integration requires uniform frequencies."
            )
    return np.sum(y, axis=axis) * df


def strip_foam_comments(text: str) -> str:
    """Remove C/C++ comments while keeping preprocessor lines untouched."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    return text


def strip_data_comment(line: str) -> str:
    """Strip the simple comments accepted in numeric profile helper files."""
    line = line.split("#", 1)[0]
    line = line.split("//", 1)[0]
    return line.strip()


def foam_lookup(text: str, key: str) -> Optional[str]:
    """Return a top-level semicolon-terminated value assigned to a key."""
    clean = strip_foam_comments(text)
    matches = re.finditer(
        rf"(?m)^\s*{re.escape(key)}\s+([^;]+?)\s*;",
        clean,
    )
    for match in matches:
        # Ignore FoamFile and nested dictionaries (notably mann.fit, which
        # contains its own minimumFrequency/maximumFrequency entries).
        depth = 0
        in_quote = False
        escaped = False
        for character in clean[: match.start()]:
            if escaped:
                escaped = False
                continue
            if character == "\\" and in_quote:
                escaped = True
                continue
            if character == '"':
                in_quote = not in_quote
            elif not in_quote:
                if character == "{":
                    depth += 1
                elif character == "}":
                    depth -= 1
        if depth == 0:
            return match.group(1).strip()
    return None


def parse_scalar(value: str, key: str) -> float:
    match = re.fullmatch(rf"\s*({FLOAT_PATTERN})\s*", value)
    if not match:
        raise ValueError(
            f"Cannot parse scalar entry {key!r} from OpenFOAM value {value!r}. "
            "Resolve macros/#calc first or supply the corresponding CLI override."
        )
    result = float(match.group(1))
    if not math.isfinite(result):
        raise ValueError(f"OpenFOAM entry {key!r} is not finite.")
    return result


def parse_vector(value: str, key: str) -> np.ndarray:
    match = re.fullmatch(r"\s*\((.*?)\)\s*", value, flags=re.DOTALL)
    if not match:
        raise ValueError(f"OpenFOAM entry {key!r} is not a three-component vector.")
    items = match.group(1).split()
    if len(items) != 3:
        raise ValueError(f"OpenFOAM entry {key!r} must contain exactly three values.")
    result = np.asarray([float(item) for item in items], dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"OpenFOAM entry {key!r} contains a non-finite value.")
    return result


def parse_word(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1]
    return value


@dataclass(frozen=True)
class MannSettings:
    dict_path: Path
    patch_name: str
    wind_profile_directory: str
    spectra_profile_file: str
    uw_profile_file: str
    target_spectra_source: str
    uw_cospectrum_source: str
    uw_stress_source: str
    rho_uw: float
    u_star: float
    uniform_uw_stress: float
    max_abs_rho_uw: float
    scale_i: np.ndarray
    scale_l: np.ndarray
    coordinate_tolerance: float
    time_step: float
    start_time: float
    end_time: float
    nt: int
    n_freq: int
    df: float
    f_max: float
    source_f_max: float
    minimum_frequency: float
    maximum_frequency: float

    @property
    def source_df(self) -> float:
        return self.source_f_max / float(self.n_freq)

    @property
    def source_frequencies(self) -> np.ndarray:
        return (
            np.arange(1, self.n_freq + 1, dtype=float)
            * self.source_df
        )

    @property
    def needs_spectra_table(self) -> bool:
        """Whether current MannHybrid reads spectraProfile in this mode."""
        return (
            self.target_spectra_source == "tabulated"
            or self.uw_cospectrum_source == "tabulated"
        )

    @property
    def needs_uw_table(self) -> bool:
        """Whether current MannHybrid reads uwCoSpectrumProfile."""
        return self.uw_cospectrum_source == "tabulated"


def read_mann_settings(
    dict_path: Path,
    patch_override: Optional[str] = None,
    tolerance_override: Optional[float] = None,
    rho_override: Optional[float] = None,
    u_star_override: Optional[float] = None,
    uniform_stress_override: Optional[float] = None,
    max_rho_override: Optional[float] = None,
) -> MannSettings:
    if not dict_path.is_file():
        raise FileNotFoundError(f"Cannot find MannHybrid dictionary: {dict_path}")

    text = dict_path.read_text(encoding="utf-8", errors="replace")

    def required_scalar(key: str) -> float:
        raw = foam_lookup(text, key)
        if raw is None:
            raise ValueError(f"Required entry {key!r} is absent from {dict_path}.")
        return parse_scalar(raw, key)

    def optional_scalar(key: str, default: float) -> float:
        raw = foam_lookup(text, key)
        return default if raw is None else parse_scalar(raw, key)

    def optional_word(key: str, default: str) -> str:
        raw = foam_lookup(text, key)
        return default if raw is None else parse_word(raw)

    def optional_vector(key: str, default: Sequence[float]) -> np.ndarray:
        raw = foam_lookup(text, key)
        if raw is None:
            return np.asarray(default, dtype=float)
        return parse_vector(raw, key)

    dt = required_scalar("timeStep")
    start = optional_scalar("startTime", 0.0)
    end = required_scalar("endTime")
    if dt <= 0.0 or end < start:
        raise ValueError("Require timeStep > 0 and endTime >= startTime.")

    # Match Foam::round for this non-negative interval.
    nt = int(math.floor((end - start) / dt + 0.5)) + 1
    if nt < 4 or nt % 2:
        raise ValueError(
            "MannHybrid requires an even number of generated samples: "
            f"round((endTime-startTime)/timeStep)+1 = {nt}."
        )

    n_freq = nt // 2
    f_max = 1.0 / (2.0 * dt)
    df = 1.0 / (nt * dt)
    declared_f_max = optional_scalar("fMax", f_max)
    if abs(declared_f_max - f_max) > 1.0e-8 * max(f_max, 1.0):
        raise ValueError(
            f"Dictionary fMax={declared_f_max:g} does not equal "
            f"1/(2*timeStep)={f_max:g}, which current MannHybrid requires."
        )

    source_f_max = optional_scalar("spectraSourceFMax", f_max)
    minimum_frequency = optional_scalar("minimumFrequency", df)
    maximum_frequency = optional_scalar("maximumFrequency", f_max)
    if (
        minimum_frequency < 0.0
        or maximum_frequency <= minimum_frequency
        or maximum_frequency > f_max + np.finfo(float).eps * max(f_max, 1.0)
    ):
        raise ValueError(
            "Require 0 <= minimumFrequency < maximumFrequency <= fMax."
        )
    if source_f_max <= 0.0:
        raise ValueError("spectraSourceFMax must be positive.")

    patch_value = foam_lookup(text, "patchName")
    if patch_value is None:
        raise ValueError(
            f"Required entry 'patchName' is absent from {dict_path}."
        )
    dictionary_patch = parse_word(patch_value)
    if patch_override and patch_override != dictionary_patch:
        raise ValueError(
            f"--patch={patch_override!r} differs from dictionary "
            f"patchName={dictionary_patch!r}. Update MannHybridTurbDict so "
            "the generator and utility use the same inlet patch."
        )
    patch = dictionary_patch
    tolerance = (
        tolerance_override
        if tolerance_override is not None
        else optional_scalar("coordinateTolerance", 1.0e-8)
    )
    rho_uw = (
        rho_override
        if rho_override is not None
        else optional_scalar("rhoUW", -0.30)
    )
    u_star = (
        u_star_override
        if u_star_override is not None
        else optional_scalar("uStar", 0.0)
    )
    uniform_stress = (
        uniform_stress_override
        if uniform_stress_override is not None
        else optional_scalar("uniformUWStress", 0.0)
    )
    max_abs_rho = (
        max_rho_override
        if max_rho_override is not None
        else optional_scalar("maxAbsRhoUW", 0.999)
    )

    if tolerance <= 0.0:
        raise ValueError("coordinateTolerance must be positive.")
    if not -1.0 < rho_uw < 1.0:
        raise ValueError("rhoUW must lie strictly between -1 and 1.")
    if u_star < 0.0:
        raise ValueError("uStar cannot be negative.")
    if not 0.0 < max_abs_rho < 1.0:
        raise ValueError("maxAbsRhoUW must satisfy 0 < value < 1.")

    target_spectra_source = optional_word(
        "targetSpectraSource", "tabulated"
    )
    allowed_spectra_sources = {"tabulated", "vonKarman"}
    if target_spectra_source not in allowed_spectra_sources:
        raise ValueError(
            f"Unknown targetSpectraSource={target_spectra_source!r}. Current "
            "MannHybrid supports tabulated and vonKarman."
        )

    uw_cospectrum_source = optional_word(
        "uwCoSpectrumSource", "tabulated"
    )
    allowed_uw_sources = {"tabulated", "kaimal", "none"}
    if uw_cospectrum_source not in allowed_uw_sources:
        raise ValueError(
            f"Unknown uwCoSpectrumSource={uw_cospectrum_source!r}. Current "
            "MannHybrid supports tabulated, kaimal and none."
        )

    uw_stress_source = optional_word("uwStressSource", "profile")
    allowed_stress_sources = {
        "profile",
        "spectraProfile",
        "uwCoSpectrumProfile",
        "rho",
        "uStar",
        "uniform",
    }
    if uw_stress_source not in allowed_stress_sources:
        raise ValueError(
            f"Unknown uwStressSource={uw_stress_source!r}. Current "
            "MannHybrid supports profile, spectraProfile, "
            "uwCoSpectrumProfile, rho, uStar and uniform."
        )

    table_is_loaded = (
        target_spectra_source == "tabulated"
        or uw_cospectrum_source == "tabulated"
    )
    if (
        uw_stress_source in {"spectraProfile", "uwCoSpectrumProfile"}
        and not table_is_loaded
    ):
        raise ValueError(
            f"uwStressSource={uw_stress_source} requires a loaded tabulated "
            "spectral input, but targetSpectraSource="
            f"{target_spectra_source} and uwCoSpectrumSource="
            f"{uw_cospectrum_source} load no table in current MannHybrid."
        )

    return MannSettings(
        dict_path=dict_path,
        patch_name=patch,
        wind_profile_directory=optional_word(
            "windProfileDirectory", "boundaryData/windProfile"
        ),
        spectra_profile_file=optional_word(
            "spectraProfileFile", "spectraProfile"
        ),
        uw_profile_file=optional_word(
            "uwCoSpectrumProfileFile", "uwCoSpectrumProfile"
        ),
        target_spectra_source=target_spectra_source,
        uw_cospectrum_source=uw_cospectrum_source,
        uw_stress_source=uw_stress_source,
        rho_uw=float(rho_uw),
        u_star=float(u_star),
        uniform_uw_stress=float(uniform_stress),
        max_abs_rho_uw=float(max_abs_rho),
        scale_i=optional_vector("scaleI", (1.0, 1.0, 1.0)),
        scale_l=optional_vector("scaleL", (1.0, 1.0, 1.0)),
        coordinate_tolerance=float(tolerance),
        time_step=float(dt),
        start_time=float(start),
        end_time=float(end),
        nt=nt,
        n_freq=n_freq,
        df=float(df),
        f_max=float(f_max),
        source_f_max=float(source_f_max),
        minimum_frequency=float(minimum_frequency),
        maximum_frequency=float(maximum_frequency),
    )


@dataclass(frozen=True)
class SourceProfile:
    path: Path
    values: np.ndarray
    has_stress: bool

    @property
    def z(self) -> np.ndarray:
        return self.values[:, 0]


@dataclass(frozen=True)
class MeanSpeedScaling:
    enabled: bool
    reference_height_agl: Optional[float]
    source_reference_speed: Optional[float]
    target_reference_speed: Optional[float]
    factor: float


def read_source_profile(path: Path) -> SourceProfile:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find source profile: {path}")

    rows: List[List[float]] = []
    n_columns: Optional[int] = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            clean = strip_data_comment(line)
            if not clean:
                continue
            fields = clean.replace(",", " ").split()
            if len(fields) not in (8, 9):
                raise ValueError(
                    f"{path}:{line_number}: expected 8 columns, or 9 with "
                    f"uwStress; found {len(fields)}."
                )
            if n_columns is None:
                n_columns = len(fields)
            elif len(fields) != n_columns:
                raise ValueError(
                    f"{path}:{line_number}: mixed 8/9-column rows are unsafe "
                    "with the MannHybrid parser."
                )
            try:
                rows.append([float(value) for value in fields])
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: non-numeric profile value."
                ) from exc

    if len(rows) < 2:
        raise ValueError(f"{path} must contain at least two numeric profile rows.")
    values = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path} contains NaN or infinite values.")

    order = np.argsort(values[:, 0], kind="mergesort")
    if not np.array_equal(order, np.arange(values.shape[0])):
        warn(f"Sorting source profile elevations from {path}.")
        values = values[order]

    dz = np.diff(values[:, 0])
    if np.any(dz <= 0.0):
        duplicate = values[np.where(dz <= 0.0)[0][0], 0]
        raise ValueError(
            f"{path} elevations must be strictly increasing and unique; "
            f"problem near z={duplicate:g}."
        )

    if np.any(values[:, 1] <= 0.0):
        raise ValueError("All mean wind speeds U must be positive.")
    if np.any(values[:, 2:5] < 0.0):
        raise ValueError("Turbulence intensities Iu, Iv and Iw cannot be negative.")
    if np.any(values[:, 5:8] <= 0.0):
        raise ValueError("Integral length scales Lu, Lv and Lw must be positive.")

    return SourceProfile(
        path=path,
        values=values,
        has_stress=values.shape[1] == 9,
    )


def interpolate_columns(
    source_z: np.ndarray,
    source_values: np.ndarray,
    query_z: np.ndarray,
    policy: str,
) -> Tuple[np.ndarray, int, int]:
    """Linear interpolation with explicit endpoint behavior."""
    below = query_z < source_z[0]
    above = query_z > source_z[-1]
    n_below = int(np.count_nonzero(below))
    n_above = int(np.count_nonzero(above))

    if policy == "error" and (n_below or n_above):
        raise ValueError(
            f"{n_below} inlet height(s) lie below the ESDU profile and "
            f"{n_above} lie above it. Source range is "
            f"[{source_z[0]:g}, {source_z[-1]:g}] m; target range is "
            f"[{query_z.min():g}, {query_z.max():g}] m. "
            "Use --extrapolation hold to match OpenFOAM endpoint behavior, "
            "or --extrapolation linear deliberately."
        )

    output = np.empty((query_z.size, source_values.shape[1]), dtype=float)
    for column in range(source_values.shape[1]):
        y = source_values[:, column]
        output[:, column] = np.interp(query_z, source_z, y)
        if policy == "linear":
            if n_below:
                slope = (y[1] - y[0]) / (source_z[1] - source_z[0])
                output[below, column] = y[0] + slope * (
                    query_z[below] - source_z[0]
                )
            if n_above:
                slope = (y[-1] - y[-2]) / (source_z[-1] - source_z[-2])
                output[above, column] = y[-1] + slope * (
                    query_z[above] - source_z[-1]
                )

    return output, n_below, n_above


def apply_reference_mean_speed_scaling(
    source: SourceProfile,
    mapped: np.ndarray,
    target_reference_speed: Optional[float],
    reference_height_agl: Optional[float],
) -> Tuple[np.ndarray, MeanSpeedScaling]:
    """Apply one velocity factor using a source-profile AGL reference."""
    if target_reference_speed is None:
        return mapped, MeanSpeedScaling(
            enabled=False,
            reference_height_agl=reference_height_agl,
            source_reference_speed=None,
            target_reference_speed=None,
            factor=1.0,
        )

    if reference_height_agl is None:
        raise ValueError(
            "A target reference mean speed requires --reference-height-agl."
        )
    if not math.isfinite(target_reference_speed) or target_reference_speed <= 0.0:
        raise ValueError("Target reference mean speed must be finite and positive.")
    if not math.isfinite(reference_height_agl) or reference_height_agl <= 0.0:
        raise ValueError("Reference height AGL must be finite and positive.")
    if reference_height_agl < source.z[0] or reference_height_agl > source.z[-1]:
        raise ValueError(
            f"Reference height {reference_height_agl:g} m AGL lies outside "
            f"the source-profile range [{source.z[0]:g}, {source.z[-1]:g}] m. "
            "Choose a reference height inside the measured/ESDU profile; "
            "reference-speed extrapolation is deliberately not allowed."
        )

    source_reference_speed = float(
        np.interp(
            reference_height_agl,
            source.z,
            source.values[:, 1],
        )
    )
    if not math.isfinite(source_reference_speed) or source_reference_speed <= 0.0:
        raise ValueError("Interpolated source reference mean speed is not positive.")

    factor = float(target_reference_speed / source_reference_speed)
    scaled = mapped.copy()
    scaled[:, 0] *= factor
    if source.has_stress:
        # Preserve the source dimensionless u-w correlation when all velocity
        # amplitudes are scaled by the same factor.
        scaled[:, 7] *= factor**2
    if not np.all(np.isfinite(scaled)) or np.any(scaled[:, 0] <= 0.0):
        raise ValueError("Mean-speed scaling produced invalid profile values.")

    return scaled, MeanSpeedScaling(
        enabled=True,
        reference_height_agl=float(reference_height_agl),
        source_reference_speed=source_reference_speed,
        target_reference_speed=float(target_reference_speed),
        factor=factor,
    )


def unique_coordinates_like_mannhybrid(
    values: np.ndarray, tolerance: float
) -> np.ndarray:
    """Mirror RegularPatchGrid::uniqueCoordinates exactly."""
    sorted_values = np.sort(np.asarray(values, dtype=float).reshape(-1))
    if sorted_values.size == 0:
        raise ValueError("No inlet face-centre elevations were found.")
    if not np.all(np.isfinite(sorted_values)):
        raise ValueError("Inlet face-centre elevations contain NaN or infinity.")

    unique: List[float] = []
    for value in sorted_values:
        x = float(value)
        if not unique or abs(x - unique[-1]) > tolerance:
            unique.append(x)
        else:
            unique[-1] = 0.5 * (unique[-1] + x)
    return np.asarray(unique, dtype=float)


def _recursive_blocks(dataset, path: Tuple[str, ...] = ()) -> Iterator[Tuple[Tuple[str, ...], object]]:
    if hasattr(dataset, "n_blocks") and hasattr(dataset, "keys"):
        keys = list(dataset.keys())
        for index in range(int(dataset.n_blocks)):
            block = dataset[index]
            if block is None:
                continue
            key = keys[index] if index < len(keys) and keys[index] else str(index)
            yield from _recursive_blocks(block, path + (str(key),))
    else:
        yield path, dataset


def extract_face_z_pyvista(
    case_path: Path,
    patch_name: str,
    foam_marker_name: str,
) -> np.ndarray:
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError(
            "PyVista is not installed. Install pyvista (which supplies VTK), "
            "use --backend ascii for an ASCII polyMesh, or provide "
            "--heights-file."
        ) from exc

    marker = case_path / foam_marker_name
    created_marker = False
    if not marker.exists():
        marker.touch()
        created_marker = True

    try:
        has_processors = any(
            child.is_dir() and re.fullmatch(r"processor\d+", child.name)
            for child in case_path.iterdir()
        )
        reader_class = (
            getattr(pv, "POpenFOAMReader", pv.OpenFOAMReader)
            if has_processors
            else pv.OpenFOAMReader
        )
        reader = reader_class(str(marker))

        if hasattr(reader, "skip_zero_time"):
            reader.skip_zero_time = False
        if hasattr(reader, "cell_to_point_creation"):
            reader.cell_to_point_creation = False
        if hasattr(reader, "decompose_polyhedra"):
            reader.decompose_polyhedra = False
        for method_name in (
            "disable_all_cell_arrays",
            "disable_all_point_arrays",
            "disable_all_patch_arrays",
        ):
            method = getattr(reader, method_name, None)
            if method is not None:
                method()

        patch_arrays = list(getattr(reader, "patch_array_names", []))
        matches = [
            name
            for name in patch_arrays
            if str(name) == patch_name
            or str(name).rstrip("/").split("/")[-1] == patch_name
        ]
        if not matches:
            available = ", ".join(map(str, patch_arrays)) or "(none reported)"
            raise ValueError(
                f"Patch {patch_name!r} was not found by the OpenFOAM reader. "
                f"Available patch arrays: {available}"
            )
        for match in matches:
            reader.enable_patch_array(match)

        time_values = list(getattr(reader, "time_values", []))
        if time_values and hasattr(reader, "set_active_time_value"):
            reader.set_active_time_value(time_values[0])

        output = reader.read()
        candidates: List[object] = []
        candidate_paths: List[str] = []
        for block_path, block in _recursive_blocks(output):
            if not hasattr(block, "n_cells") or int(block.n_cells) == 0:
                continue
            final = block_path[-1] if block_path else ""
            joined = "/".join(block_path)
            if (
                final == patch_name
                or final.rstrip("/").split("/")[-1] == patch_name
                or joined.rstrip("/").split("/")[-1] == patch_name
            ):
                candidates.append(block)
                candidate_paths.append(joined)

        if not candidates:
            all_paths = [
                "/".join(path)
                for path, block in _recursive_blocks(output)
                if hasattr(block, "n_cells") and int(block.n_cells) > 0
            ]
            raise RuntimeError(
                f"PyVista enabled patch {patch_name!r}, but its output block "
                f"could not be identified. Non-empty blocks: {all_paths}"
            )

        centre_arrays = []
        for block in candidates:
            try:
                centres = block.cell_centers(vertex=False).points
            except TypeError:
                # Older PyVista releases did not expose the vertex keyword.
                centres = block.cell_centers().points
            if centres.ndim != 2 or centres.shape[1] != 3:
                raise RuntimeError("Unexpected PyVista cell-centre array shape.")
            centre_arrays.append(np.asarray(centres[:, 2], dtype=float))
        return np.concatenate(centre_arrays)
    finally:
        if created_marker:
            marker.unlink(missing_ok=True)


def _existing_or_gz(path: Path) -> Path:
    if path.is_file():
        return path
    gz_path = Path(str(path) + ".gz")
    if gz_path.is_file():
        return gz_path
    raise FileNotFoundError(f"Cannot find {path} or {gz_path}.")


def _open_text_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _ensure_ascii_foam_file(path: Path) -> None:
    with (
        gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")
    ) as handle:
        header = handle.read(8192).decode("ascii", errors="ignore")
    format_match = re.search(r"\bformat\s+(\w+)\s*;", header)
    if format_match and format_match.group(1).lower() != "ascii":
        raise RuntimeError(
            f"{path} is {format_match.group(1)} OpenFOAM data. The streaming "
            "backend intentionally supports ASCII polyMesh files only. Use "
            "--backend pyvista or provide --heights-file."
        )
    class_match = re.search(r"\bclass\s+(\w+)\s*;", header)
    if class_match and path.name.startswith("faces"):
        if class_match.group(1) not in ("faceList",):
            raise RuntimeError(
                f"{path} has class {class_match.group(1)!r}; the ASCII backend "
                "supports faceList only. Use --backend pyvista."
            )


def _foam_list_records(path: Path) -> Iterator[str]:
    """Yield one ordinary ASCII OpenFOAM list record at a time."""
    _ensure_ascii_foam_file(path)
    declared_count: Optional[int] = None
    opened = False
    yielded = 0
    buffer = ""
    paren_balance = 0

    with _open_text_maybe_gzip(path) as handle:
        for raw_line in handle:
            line = strip_data_comment(raw_line)
            if not line:
                continue
            if declared_count is None:
                if re.fullmatch(r"\d+", line):
                    declared_count = int(line)
                continue
            if not opened:
                if line.startswith("("):
                    opened = True
                    line = line[1:].strip()
                    if not line:
                        continue
                else:
                    continue

            if not buffer and line == ")":
                break

            if buffer:
                buffer += " " + line
            else:
                buffer = line
            paren_balance += line.count("(") - line.count(")")

            # A point or face record contains a balanced inner pair. The outer
            # list delimiter is on its own line in standard OpenFOAM output.
            if paren_balance == 0 and buffer:
                if buffer != ")":
                    yielded += 1
                    yield buffer.strip()
                buffer = ""

    if declared_count is None or not opened:
        raise RuntimeError(f"Could not locate the data list in {path}.")
    if buffer:
        raise RuntimeError(f"Unbalanced list record near the end of {path}.")
    if yielded != declared_count:
        raise RuntimeError(
            f"{path} declared {declared_count} records but {yielded} were read."
        )


def _read_patch_range(boundary_path: Path, patch_name: str) -> Tuple[int, int, List[str]]:
    with _open_text_maybe_gzip(boundary_path) as handle:
        text = strip_foam_comments(handle.read())

    block_pattern = re.compile(
        r'(?m)^\s*("?[^"\s{}]+"?)\s*\{(.*?)^\s*\}',
        flags=re.DOTALL | re.MULTILINE,
    )
    patches: List[str] = []
    selected: Optional[str] = None
    for match in block_pattern.finditer(text):
        name = match.group(1).strip('"')
        body = match.group(2)
        if re.search(r"\bnFaces\s+\d+\s*;", body):
            patches.append(name)
            if name == patch_name:
                selected = body

    if selected is None:
        raise ValueError(
            f"Patch {patch_name!r} is absent from {boundary_path}. "
            f"Available patches: {', '.join(patches)}"
        )
    n_match = re.search(r"\bnFaces\s+(\d+)\s*;", selected)
    start_match = re.search(r"\bstartFace\s+(\d+)\s*;", selected)
    if not n_match or not start_match:
        raise RuntimeError(
            f"Patch {patch_name!r} lacks nFaces/startFace in {boundary_path}."
        )
    return int(start_match.group(1)), int(n_match.group(1)), patches


def _parse_face_record(record: str) -> List[int]:
    match = re.fullmatch(r"\s*(\d+)\s*\((.*?)\)\s*", record)
    if not match:
        raise RuntimeError(f"Cannot parse ASCII face record: {record[:120]!r}")
    expected = int(match.group(1))
    indices = [int(value) for value in match.group(2).split()]
    if len(indices) != expected:
        raise RuntimeError(
            f"Face declared {expected} points but contains {len(indices)}."
        )
    if expected < 3:
        raise RuntimeError("An inlet boundary face has fewer than three points.")
    return indices


def _parse_point_record(record: str) -> np.ndarray:
    match = re.fullmatch(r"\s*\((.*?)\)\s*", record)
    if not match:
        raise RuntimeError(f"Cannot parse ASCII point record: {record[:120]!r}")
    values = match.group(1).split()
    if len(values) != 3:
        raise RuntimeError("An OpenFOAM point does not have three coordinates.")
    return np.asarray([float(value) for value in values], dtype=float)


def openfoam_face_centre(vertices: np.ndarray) -> np.ndarray:
    """Mirror the v2012 face::centre implementation."""
    if vertices.shape[0] == 3:
        return np.mean(vertices, axis=0)
    centre_point = np.mean(vertices, axis=0)
    weighted_sum = np.zeros(3, dtype=float)
    sum_area_twice = 0.0
    for index in range(vertices.shape[0]):
        current = vertices[index]
        following = vertices[(index + 1) % vertices.shape[0]]
        area_twice = float(
            np.linalg.norm(
                np.cross(current - centre_point, following - centre_point)
            )
        )
        weighted_sum += area_twice * (current + following + centre_point)
        sum_area_twice += area_twice
    if sum_area_twice <= np.finfo(float).tiny:
        return centre_point
    return weighted_sum / (3.0 * sum_area_twice)


def extract_face_z_ascii(case_path: Path, patch_name: str) -> np.ndarray:
    mesh_dir = case_path / "constant" / "polyMesh"
    boundary_path = _existing_or_gz(mesh_dir / "boundary")
    faces_path = _existing_or_gz(mesh_dir / "faces")
    points_path = _existing_or_gz(mesh_dir / "points")
    _ensure_ascii_foam_file(faces_path)
    _ensure_ascii_foam_file(points_path)

    start_face, n_faces, _ = _read_patch_range(boundary_path, patch_name)
    end_face = start_face + n_faces
    patch_faces: List[List[int]] = []
    needed_points: set[int] = set()

    for face_index, record in enumerate(_foam_list_records(faces_path)):
        if face_index < start_face:
            continue
        if face_index >= end_face:
            break
        indices = _parse_face_record(record)
        patch_faces.append(indices)
        needed_points.update(indices)
    if len(patch_faces) != n_faces:
        raise RuntimeError(
            f"Expected {n_faces} faces for patch {patch_name!r}, "
            f"but read {len(patch_faces)}."
        )

    point_coordinates: dict[int, np.ndarray] = {}
    for point_index, record in enumerate(_foam_list_records(points_path)):
        if point_index in needed_points:
            point_coordinates[point_index] = _parse_point_record(record)
            if len(point_coordinates) == len(needed_points):
                break
    missing = needed_points.difference(point_coordinates)
    if missing:
        raise RuntimeError(
            f"Could not recover {len(missing)} point(s) used by inlet faces."
        )

    elevations = np.empty(n_faces, dtype=float)
    for face_index, indices in enumerate(patch_faces):
        vertices = np.vstack([point_coordinates[index] for index in indices])
        elevations[face_index] = openfoam_face_centre(vertices)[2]
    return elevations


def read_heights_file(path: Path, column: Optional[int]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find heights file: {path}")
    rows: List[List[float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            clean = strip_data_comment(line)
            if not clean:
                continue
            fields = clean.replace(",", " ").split()
            try:
                rows.append([float(value) for value in fields])
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: non-numeric value."
                ) from exc
    if not rows:
        raise ValueError(f"{path} contains no numeric rows.")
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError(f"{path} contains inconsistent column counts.")
    width = widths.pop()
    data = np.asarray(rows, dtype=float)
    if column is None:
        if width == 1:
            selected = data[:, 0]
        elif width >= 3:
            selected = data[:, 2]
        else:
            raise ValueError(
                f"{path} has two columns; specify --height-column explicitly."
            )
    else:
        if column < 0 or column >= width:
            raise ValueError(
                f"--height-column {column} is outside the {width}-column file."
            )
        selected = data[:, column]
    if not np.all(np.isfinite(selected)):
        raise ValueError(f"{path} contains non-finite elevations.")
    return selected


def obtain_face_elevations(
    case_path: Path,
    settings: MannSettings,
    backend: str,
    foam_marker_name: str,
    heights_file: Optional[Path],
    height_column: Optional[int],
) -> Tuple[np.ndarray, str]:
    if heights_file is not None:
        return read_heights_file(heights_file, height_column), "heights file"

    if backend == "pyvista":
        return (
            extract_face_z_pyvista(
                case_path, settings.patch_name, foam_marker_name
            ),
            "PyVista/VTK",
        )
    if backend == "ascii":
        return (
            extract_face_z_ascii(case_path, settings.patch_name),
            "ASCII polyMesh",
        )

    # Auto: PyVista is the first choice because the production mesh may be
    # binary or decomposed. Fall back only when an ordinary ASCII root mesh is
    # available.
    try:
        return (
            extract_face_z_pyvista(
                case_path, settings.patch_name, foam_marker_name
            ),
            "PyVista/VTK",
        )
    except Exception as pyvista_error:
        try:
            values = extract_face_z_ascii(case_path, settings.patch_name)
            warn(
                "PyVista extraction was unavailable; used the ASCII polyMesh "
                f"backend instead. PyVista reason: {pyvista_error}"
            )
            return values, "ASCII polyMesh"
        except Exception as ascii_error:
            raise RuntimeError(
                "Automatic inlet-height extraction failed with both backends.\n"
                f"PyVista: {pyvista_error}\n"
                f"ASCII polyMesh: {ascii_error}\n"
                "Install PyVista/VTK or provide --heights-file."
            ) from ascii_error


def von_karman_spectra(
    mapped_profile: np.ndarray,
    frequencies: np.ndarray,
    scale_i: np.ndarray,
    scale_l: np.ndarray,
    active_mask: np.ndarray,
    normalisation: str,
    integration_rule: str,
) -> np.ndarray:
    """Sample the exact current MannHybridTurb von Karman formulas."""
    n_heights = mapped_profile.shape[0]
    n_freq = frequencies.size
    result = np.empty((n_heights, 3, n_freq), dtype=float)

    for height_index in range(n_heights):
        mean_u = float(mapped_profile[height_index, 0])
        intensities = mapped_profile[height_index, 1:4] * scale_i
        length_scales = mapped_profile[height_index, 4:7] * scale_l
        sigma = np.abs(intensities) * abs(mean_u)

        for component in range(3):
            length = max(abs(float(length_scales[component])), 1.0e-12)
            if component == 0:
                x = frequencies * length / abs(mean_u)
                spectrum = (
                    4.0
                    * sigma[component] ** 2
                    * (length / abs(mean_u))
                    / (1.0 + 70.8 * x**2) ** (5.0 / 6.0)
                )
            else:
                x = 2.0 * frequencies * length / abs(mean_u)
                spectrum = (
                    4.0
                    * sigma[component] ** 2
                    * (length / abs(mean_u))
                    * (1.0 + 188.4 * x**2)
                    / (1.0 + 70.8 * x**2) ** (11.0 / 6.0)
                )

            if normalisation == "resolved-variance":
                if np.count_nonzero(active_mask) < 2:
                    raise ValueError(
                        "At least two source-frequency bins must lie in the "
                        "active band for resolved-variance normalisation."
                    )
                active_area = float(
                    spectral_integral(
                        spectrum[active_mask],
                        frequencies[active_mask],
                        integration_rule,
                    )
                )
                target_variance = float(sigma[component] ** 2)
                if active_area <= 0.0:
                    if target_variance > 0.0:
                        raise ValueError(
                            "A von Karman spectrum has zero active-band area."
                        )
                else:
                    spectrum = spectrum * (target_variance / active_area)

            result[height_index, component, :] = np.maximum(
                spectrum, 1.0e-300
            )
    return result


def resolve_stress(
    requested_mode: str,
    dictionary_mode: str,
    source_has_stress: bool,
    mapped_profile: np.ndarray,
    scale_i: np.ndarray,
    rho_uw: float,
    u_star: float,
    uniform_stress: float,
) -> Tuple[np.ndarray, str]:
    mode = dictionary_mode if requested_mode == "dict" else requested_mode
    mean_u = mapped_profile[:, 0]
    # Match TargetModel exactly: the rho fallback uses the signed scaled I
    # components multiplied by |U| (the auto-spectra themselves use |I|).
    sigma_u = mapped_profile[:, 1] * scale_i[0] * np.abs(mean_u)
    sigma_w = mapped_profile[:, 3] * scale_i[2] * np.abs(mean_u)

    if mode == "profile":
        if source_has_stress:
            return mapped_profile[:, 7].copy(), "profile ninth column"
        warn(
            "The ESDU profile has no ninth u'w' column. Matching current "
            "MannHybrid behavior with rhoUW*sigmaU*sigmaW."
        )
        return rho_uw * sigma_u * sigma_w, "rhoUW fallback"
    if mode == "rho":
        return rho_uw * sigma_u * sigma_w, "rhoUW"
    if mode == "uStar":
        return np.full(mean_u.size, -(u_star**2)), "uStar"
    if mode == "uniform":
        return np.full(mean_u.size, uniform_stress), "uniformUWStress"
    if mode in ("spectraProfile", "uwCoSpectrumProfile"):
        if source_has_stress:
            return mapped_profile[:, 7].copy(), "source profile ninth column"
        warn(
            f"uwStressSource={mode} needs a stress column in a generated "
            "spectral table, but the ESDU profile has none. Using the explicit "
            "dictionary rhoUW fallback to construct that column."
        )
        return rho_uw * sigma_u * sigma_w, "rhoUW for generated table"
    raise ValueError(
        f"Unsupported stress mode {mode!r}. Expected profile, rho, uStar, "
        "uniform, spectraProfile or uwCoSpectrumProfile."
    )


def kaimal_shape(
    frequencies: np.ndarray,
    z_absolute: float,
    mean_u: float,
) -> np.ndarray:
    if z_absolute <= 0.0 or abs(mean_u) <= np.finfo(float).eps:
        return np.zeros_like(frequencies)
    nondimensional = frequencies * z_absolute / abs(mean_u)
    return (
        14.0
        * (z_absolute / abs(mean_u))
        / (1.0 + 9.6 * nondimensional) ** 2.4
    )


def _bounded_shape_to_stress(
    frequencies: np.ndarray,
    shape: np.ndarray,
    target_stress: float,
    rho_limit: np.ndarray,
    active_mask: np.ndarray,
    integration_rule: str,
    relative_tolerance: float = 1.0e-10,
    max_iterations: int = 100,
) -> Tuple[np.ndarray, float, float, bool, float]:
    output = np.zeros_like(frequencies)
    if abs(target_stress) <= 1.0e-30:
        return output, 0.0, 0.0, False, 0.0
    if np.count_nonzero(active_mask) < 2:
        raise ValueError(
            "The active frequency band contains fewer than two table bins."
        )

    f = frequencies[active_mask]
    positive_shape = np.maximum(shape[active_mask], 0.0)
    limit = np.maximum(rho_limit[active_mask], 0.0)
    if not np.any(positive_shape > 0.0):
        raise ValueError(
            "Kaimal co-spectrum shape is zero. Check that inlet elevations "
            "and mean wind speeds are positive."
        )

    target_abs = abs(float(target_stress))
    sign = 1.0 if target_stress > 0.0 else -1.0
    maximum_area = float(
        spectral_integral(limit, f, integration_rule)
    )
    if target_abs >= maximum_area * (1.0 - relative_tolerance):
        output[active_mask] = sign * limit
        realised = sign * float(
            spectral_integral(limit, f, integration_rule)
        )
        return output, realised, maximum_area, True, 1.0

    def area(scale: float) -> float:
        return float(
            spectral_integral(
                np.minimum(scale * positive_shape, limit),
                f,
                integration_rule,
            )
        )

    unbounded_area = float(
        spectral_integral(positive_shape, f, integration_rule)
    )
    lower = 0.0
    upper = max(target_abs / max(unbounded_area, 1.0e-300), 1.0e-300)
    while area(upper) < target_abs:
        upper *= 2.0
        if not math.isfinite(upper):
            raise RuntimeError("Failed to bracket bounded Kaimal normalisation.")

    for _ in range(max_iterations):
        middle = 0.5 * (lower + upper)
        if area(middle) < target_abs:
            lower = middle
        else:
            upper = middle

    values = np.minimum(upper * positive_shape, limit)
    output[active_mask] = sign * values
    realised = sign * float(
        spectral_integral(values, f, integration_rule)
    )
    clipped_fraction = float(np.mean(values >= 0.999 * limit))
    return output, realised, maximum_area, False, clipped_fraction


@dataclass(frozen=True)
class CoSpectrumResult:
    values: np.ndarray
    requested_stress: np.ndarray
    realised_stress: np.ndarray
    maximum_feasible_abs_stress: np.ndarray
    infeasible: np.ndarray
    clipped_fraction: np.ndarray
    max_abs_rho: np.ndarray


def make_bounded_kaimal_cospectra(
    z_absolute: np.ndarray,
    mapped_profile: np.ndarray,
    frequencies: np.ndarray,
    spectra: np.ndarray,
    requested_stress: np.ndarray,
    active_mask: np.ndarray,
    max_abs_rho: float,
    integration_rule: str,
) -> CoSpectrumResult:
    n_heights = z_absolute.size
    values = np.zeros((n_heights, frequencies.size), dtype=float)
    realised = np.zeros(n_heights, dtype=float)
    maximum = np.zeros(n_heights, dtype=float)
    infeasible = np.zeros(n_heights, dtype=bool)
    clipped_fraction = np.zeros(n_heights, dtype=float)
    rho_observed = np.zeros(n_heights, dtype=float)

    for index in range(n_heights):
        shape = kaimal_shape(
            frequencies,
            float(z_absolute[index]),
            float(mapped_profile[index, 0]),
        )
        denom = np.sqrt(
            np.maximum(
                spectra[index, 0, :] * spectra[index, 2, :],
                1.0e-300,
            )
        )
        limit = max_abs_rho * denom
        (
            values[index, :],
            realised[index],
            maximum[index],
            infeasible[index],
            clipped_fraction[index],
        ) = _bounded_shape_to_stress(
            frequencies,
            shape,
            float(requested_stress[index]),
            limit,
            active_mask,
            integration_rule,
        )
        rho = np.abs(values[index, :]) / denom
        rho_observed[index] = float(np.max(rho[active_mask]))

    return CoSpectrumResult(
        values=values,
        requested_stress=requested_stress.copy(),
        realised_stress=realised,
        maximum_feasible_abs_stress=maximum,
        infeasible=infeasible,
        clipped_fraction=clipped_fraction,
        max_abs_rho=rho_observed,
    )


def validate_arrays(
    z: np.ndarray,
    profile: np.ndarray,
    stress: np.ndarray,
    spectra: Optional[np.ndarray],
    cospectra: Optional[CoSpectrumResult],
    frequencies: Optional[np.ndarray],
    active_mask: Optional[np.ndarray],
    max_abs_rho: float,
    normalisation: str,
    scale_i: np.ndarray,
    integration_rule: str,
) -> dict:
    if np.any(np.diff(z) <= 0.0):
        raise RuntimeError("Generated elevations are not strictly increasing.")
    if not np.all(np.isfinite(profile)) or not np.all(np.isfinite(stress)):
        raise RuntimeError("Generated arrays contain NaN or infinity.")

    max_rho_seen: Optional[float] = None
    stress_error: Optional[float] = None
    variance_error: Optional[float] = None

    if spectra is not None:
        if frequencies is None or active_mask is None:
            raise RuntimeError("Spectral validation is missing its frequency grid.")
        expected_shape = (z.size, 3, frequencies.size)
        if spectra.shape != expected_shape:
            raise RuntimeError(
                f"Generated auto-spectra have shape {spectra.shape}; expected "
                f"{expected_shape}."
            )
        if not np.all(np.isfinite(spectra)):
            raise RuntimeError("Generated auto-spectra contain NaN or infinity.")
        if np.any(spectra <= 0.0):
            raise RuntimeError(
                "Generated auto-spectra must be strictly positive."
            )

    if spectra is not None and normalisation == "resolved-variance":
        target_variance = (
            profile[:, 0, None] * profile[:, 1:4] * scale_i[None, :]
        ) ** 2
        integrated_variance = spectral_integral(
            spectra[:, :, active_mask],
            frequencies[active_mask],
            integration_rule,
            axis=2,
        )
        variance_error = float(
            np.max(np.abs(integrated_variance - target_variance))
        )
        variance_scale = max(1.0, float(np.max(target_variance)))
        if variance_error > 1.0e-8 * variance_scale:
            raise RuntimeError(
                "Resolved-variance spectra failed their variance integral "
                f"check: maximum absolute error {variance_error:g}."
            )

    if cospectra is not None:
        if spectra is None or frequencies is None or active_mask is None:
            raise RuntimeError(
                "Co-spectrum validation requires auto-spectra and frequencies."
            )
        expected_shape = (z.size, frequencies.size)
        if cospectra.values.shape != expected_shape:
            raise RuntimeError(
                f"Generated co-spectra have shape {cospectra.values.shape}; "
                f"expected {expected_shape}."
            )
        if not np.all(np.isfinite(cospectra.values)):
            raise RuntimeError("Generated co-spectra contain NaN or infinity.")

        su = spectra[:, 0, :]
        sw = spectra[:, 2, :]
        rho = cospectra.values / np.sqrt(np.maximum(su * sw, 1.0e-300))
        max_rho_seen = float(np.max(np.abs(rho[:, active_mask])))
        if max_rho_seen > max_abs_rho * (1.0 + 1.0e-8):
            raise RuntimeError(
                f"Generated Cuw violates maxAbsRhoUW: {max_rho_seen:g}."
            )

        integrated_stress = spectral_integral(
            cospectra.values[:, active_mask],
            frequencies[active_mask],
            integration_rule,
            axis=1,
        )
        stress_error = float(
            np.max(np.abs(integrated_stress - cospectra.realised_stress))
        )
        stress_scale = max(
            1.0, float(np.max(np.abs(cospectra.realised_stress)))
        )
        if stress_error > 1.0e-8 * stress_scale:
            raise RuntimeError(
                "Generated Cuw integral does not match the written stress: "
                f"maximum absolute error {stress_error:g}."
            )

    return {
        "max_abs_rho_uw": max_rho_seen,
        "max_stress_integral_error": stress_error,
        "max_variance_integral_error": variance_error,
    }


def _temporary_output(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(descriptor)
    return Path(name)


def write_profile_file(
    path: Path,
    z: np.ndarray,
    profile: np.ndarray,
    stress: np.ndarray,
    precision: int,
) -> None:
    format_string = f"%.{precision}e"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(z.size):
            row = np.concatenate(
                (
                    np.asarray([z[index]], dtype=float),
                    profile[index, :7],
                    np.asarray([stress[index]], dtype=float),
                )
            )
            handle.write("\t".join(format_string % value for value in row))
            handle.write("\n")


def write_spectra_file(
    path: Path,
    z: np.ndarray,
    stress: np.ndarray,
    spectra: np.ndarray,
    precision: int,
) -> None:
    format_string = f"%.{precision}e"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{z.size} {spectra.shape[2]}\n")
        for index in range(z.size):
            prefix = (
                format_string % z[index],
                format_string % stress[index],
            )
            flattened = spectra[index, :, :].reshape(-1)
            handle.write("\t".join(prefix))
            handle.write("\t")
            handle.write(
                "\t".join(format_string % value for value in flattened)
            )
            handle.write("\n")


def write_uw_file(
    path: Path,
    z: np.ndarray,
    stress: np.ndarray,
    cospectrum: np.ndarray,
    precision: int,
) -> None:
    format_string = f"%.{precision}e"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{z.size} {cospectrum.shape[1]}\n")
        for index in range(z.size):
            prefix = (
                format_string % z[index],
                format_string % stress[index],
            )
            handle.write("\t".join(prefix))
            handle.write("\t")
            handle.write(
                "\t".join(
                    format_string % value
                    for value in cospectrum[index, :]
                )
            )
            handle.write("\n")


def validate_written_numeric_file(
    path: Path,
    kind: str,
    n_heights: int,
    n_freq: int,
) -> None:
    with path.open("r", encoding="utf-8") as handle:
        if kind == "profile":
            expected_columns = 9
            row_count = 0
        else:
            header = handle.readline().split()
            if header != [str(n_heights), str(n_freq)]:
                raise RuntimeError(
                    f"{path} header is {header}; expected "
                    f"{n_heights} {n_freq}."
                )
            expected_columns = (
                2 + 3 * n_freq if kind == "spectra" else 2 + n_freq
            )
            row_count = 0

        for line_number, line in enumerate(handle, start=2 if kind != "profile" else 1):
            fields = line.split()
            if len(fields) != expected_columns:
                raise RuntimeError(
                    f"{path}:{line_number} has {len(fields)} values; expected "
                    f"{expected_columns}."
                )
            row_count += 1
        if row_count != n_heights:
            raise RuntimeError(
                f"{path} has {row_count} rows; expected {n_heights}."
            )


def atomic_copy(source: Path, destination: Path) -> None:
    temp = _temporary_output(destination)
    try:
        shutil.copyfile(source, temp)
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


def required_file_note(settings: MannSettings) -> str:
    required = ["profile"]
    if settings.needs_spectra_table:
        required.append(settings.spectra_profile_file)
    if settings.needs_uw_table:
        required.append(settings.uw_profile_file)
    return ", ".join(required)


def prepare(args: argparse.Namespace) -> int:
    case_path = args.case.expanduser().resolve()
    if not case_path.is_dir():
        raise NotADirectoryError(f"OpenFOAM case directory not found: {case_path}")

    dict_path = (
        args.dictionary
        if args.dictionary.is_absolute()
        else case_path / args.dictionary
    ).resolve()
    settings = read_mann_settings(
        dict_path,
        patch_override=args.patch,
        tolerance_override=args.coordinate_tolerance,
        rho_override=args.rho_uw,
        u_star_override=args.u_star,
        uniform_stress_override=args.uniform_uw_stress,
        max_rho_override=args.max_abs_rho_uw,
    )
    needs_spectra_table = settings.needs_spectra_table
    needs_uw_table = settings.needs_uw_table

    if args.profile_directory is None:
        wind_dir = (
            case_path
            / "constant"
            / settings.wind_profile_directory
        ).resolve()
    else:
        wind_dir = (
            args.profile_directory
            if args.profile_directory.is_absolute()
            else case_path / args.profile_directory
        ).resolve()
    active_profile_path = wind_dir / "profile"
    active_spectra_path = wind_dir / settings.spectra_profile_file
    active_uw_path = wind_dir / settings.uw_profile_file
    active_output_specs = [("profile", active_profile_path)]
    if needs_spectra_table:
        active_output_specs.append(("spectra", active_spectra_path))
    if needs_uw_table:
        active_output_specs.append(("uw", active_uw_path))
    active_destinations = {
        path.resolve() for _, path in active_output_specs
    }
    if len(active_destinations) != len(active_output_specs):
        raise ValueError(
            "The MannHybrid input files active in the selected source modes "
            "must resolve to different output files."
        )
    backup_name = Path(args.source_backup_name)
    if (
        backup_name.is_absolute()
        or len(backup_name.parts) != 1
        or backup_name.name in {"", ".", ".."}
    ):
        raise ValueError(
            "--source-backup-name must be one non-empty filename, not a "
            "directory, absolute path, or path containing '..'."
        )
    backup_path = wind_dir / backup_name
    if backup_path.exists() and not backup_path.is_file():
        raise ValueError(
            f"The ESDU source-backup destination exists but is not a regular "
            f"file: {backup_path}"
        )
    if backup_path.resolve() in active_destinations:
        raise ValueError(
            "--source-backup-name must not collide with an active "
            "MannHybrid input filename."
        )
    if not args.no_target_copies:
        target_copy_specs = [
            (active_profile_path, wind_dir / "targetProfile")
        ]
        if needs_spectra_table:
            target_copy_specs.append(
                (active_spectra_path, wind_dir / "targetSpectraProfile")
            )
        if needs_uw_table:
            target_copy_specs.append(
                (active_uw_path, wind_dir / "targetUWCoSpectrumProfile")
            )
        target_destinations = tuple(
            target for _, target in target_copy_specs
        )
        if backup_path.resolve() in {
            path.resolve() for path in target_destinations
        }:
            raise ValueError(
                "--source-backup-name must not collide with a target* "
                "compatibility-copy filename."
            )
        for source_file, target_file in target_copy_specs:
            if (
                target_file.resolve() in active_destinations
                and target_file.resolve() != source_file.resolve()
            ):
                raise ValueError(
                    f"Compatibility copy {target_file.name!r} would overwrite "
                    "a different active MannHybrid input. Use "
                    "--no-target-copies or change the dictionary filenames."
                )
    if args.source_profile is not None:
        source_path = (
            args.source_profile
            if args.source_profile.is_absolute()
            else case_path / args.source_profile
        ).resolve()
    elif backup_path.is_file() and not args.refresh_source_backup:
        source_path = backup_path
    else:
        source_path = active_profile_path
    source = read_source_profile(source_path)
    if (
        args.source_profile is not None
        and backup_path.is_file()
        and source_path.resolve() != backup_path.resolve()
        and not args.refresh_source_backup
    ):
        warn(
            f"Using explicit source {source_path}, but preserving the existing "
            f"{backup_path}. A later no-argument rerun will use that existing "
            "backup. Add --refresh-source-backup to install the explicit "
            "source as the new persistent ESDU baseline."
        )

    heights_path = None
    if args.heights_file is not None:
        heights_path = (
            args.heights_file
            if args.heights_file.is_absolute()
            else case_path / args.heights_file
        ).resolve()

    raw_face_z, mesh_backend = obtain_face_elevations(
        case_path,
        settings,
        args.backend,
        args.foam_marker,
        heights_path,
        args.height_column,
    )
    target_z_absolute = unique_coordinates_like_mannhybrid(
        raw_face_z,
        settings.coordinate_tolerance,
    )
    target_z_agl = target_z_absolute - args.ground_elevation
    if np.any(target_z_agl <= 0.0):
        raise ValueError(
            "All inlet face-centre heights above ground must be positive. "
            f"Minimum absolute z={target_z_absolute.min():g}, "
            f"ground elevation={args.ground_elevation:g}."
        )

    mapped, n_below, n_above = interpolate_columns(
        source.z,
        source.values[:, 1:],
        target_z_agl,
        args.extrapolation,
    )
    if args.extrapolation == "hold" and (n_below or n_above):
        warn(
            f"Endpoint-held ESDU data at {n_below} height(s) below and "
            f"{n_above} height(s) above the source range. This matches "
            "OpenFOAM interpolateXY behavior."
        )

    mapped, speed_scaling = apply_reference_mean_speed_scaling(
        source,
        mapped,
        args.target_reference_mean_speed,
        args.reference_height_agl,
    )

    # First seven mapped columns are U, Iu, Iv, Iw, Lu, Lv, Lw. An optional
    # eighth is source uwStress.
    mapped_profile = mapped[:, :7]
    if np.any(mapped_profile[:, 0] <= 0.0):
        raise ValueError("Mapped mean wind speed is non-positive.")
    if np.any(mapped_profile[:, 1:4] < 0.0):
        raise ValueError("Mapped turbulence intensity is negative.")
    if np.any(mapped_profile[:, 4:7] <= 0.0):
        raise ValueError("Mapped integral length scale is non-positive.")

    requested_stress, stress_description = resolve_stress(
        args.stress_source,
        settings.uw_stress_source,
        source.has_stress,
        mapped,
        settings.scale_i,
        settings.rho_uw,
        settings.u_star,
        settings.uniform_uw_stress,
    )
    effective_stress_mode = (
        settings.uw_stress_source
        if args.stress_source == "dict"
        else args.stress_source
    )
    if (
        speed_scaling.enabled
        and settings.uw_cospectrum_source != "none"
        and effective_stress_mode in {"uStar", "uniform"}
    ):
        warn(
            f"Mean speeds were scaled by {speed_scaling.factor:.8g}, but "
            f"uwStressSource={effective_stress_mode} is a dimensional "
            "dictionary input and was not rescaled automatically. Update "
            "uStar by the same factor, or uniformUWStress by factor^2, if "
            "dynamic similarity is intended."
        )
    if (
        settings.uw_cospectrum_source != "none"
        and args.stress_source != "dict"
        and args.stress_source != settings.uw_stress_source
    ):
        warn(
            f"--stress-source={args.stress_source} overrides generation only, "
            f"while MannHybridTurbDict still says "
            f"uwStressSource={settings.uw_stress_source}. Update the "
            "dictionary to the same source before running MannHybrid."
        )
    frequencies: Optional[np.ndarray] = None
    active_mask: Optional[np.ndarray] = None
    spectra: Optional[np.ndarray] = None
    cospectra: Optional[CoSpectrumResult] = None
    effective_spectra_normalisation = args.spectra_normalisation

    # When analytic von Karman auto-spectra are paired with a tabulated Cuw,
    # construct and bound Cuw against the same native analytical autos that
    # MannHybrid will use. A resolved-variance support table would otherwise
    # give a different rho clipping limit even though its auto columns are not
    # used as targets by the utility.
    if (
        settings.target_spectra_source == "vonKarman"
        and needs_uw_table
        and effective_spectra_normalisation != "native"
    ):
        warn(
            "targetSpectraSource=vonKarman with a tabulated u-w co-spectrum "
            "requires native auto-spectra for consistent rho clipping. "
            "Using native normalisation for the support spectra table."
        )
        effective_spectra_normalisation = "native"

    if needs_spectra_table:
        frequencies = settings.source_frequencies
        if abs(settings.source_f_max - settings.f_max) > 1.0e-10 * max(
            settings.f_max, 1.0
        ):
            raise ValueError(
                "This generator requires spectraSourceFMax=fMax when a "
                "spectral table is active, so its bins coincide exactly with "
                "MannHybrid's synthesis bins. Found spectraSourceFMax="
                f"{settings.source_f_max:g} and fMax={settings.f_max:g}."
            )
        active_mask = (
            (frequencies >= settings.minimum_frequency)
            & (frequencies <= settings.maximum_frequency)
        )
        if np.count_nonzero(active_mask) < 2:
            raise ValueError(
                "The tabulated source grid contains fewer than two "
                "frequencies inside [minimumFrequency, maximumFrequency]."
            )
        if frequencies[0] > settings.minimum_frequency * (1.0 + 1.0e-10):
            warn(
                "The first tabulated frequency exceeds minimumFrequency; "
                "MannHybrid will endpoint-hold the first PSD bin."
            )
        if frequencies[-1] < settings.maximum_frequency * (1.0 - 1.0e-10):
            warn(
                "spectraSourceFMax is below maximumFrequency; MannHybrid will "
                "endpoint-hold the final PSD bin."
            )

        values_per_bin = 3 + int(needs_uw_table)
        table_values = (
            target_z_absolute.size * settings.n_freq * values_per_bin
        )
        if table_values > args.max_table_values:
            estimated_gib = table_values * 8.0 / 1024.0**3
            raise MemoryError(
                f"Requested active tables contain about {table_values:,} "
                f"floating values ({estimated_gib:.2f} GiB in memory before "
                "text formatting), exceeding --max-table-values="
                f"{args.max_table_values:,}. Check coordinateTolerance and "
                "the inlet-height extraction."
            )

        spectra = von_karman_spectra(
            mapped_profile,
            frequencies,
            settings.scale_i,
            settings.scale_l,
            active_mask,
            effective_spectra_normalisation,
            args.integration_rule,
        )

    realised_stress = requested_stress.copy()
    if needs_uw_table:
        if spectra is None or frequencies is None or active_mask is None:
            raise RuntimeError(
                "Internal error: tabulated Cuw requires a spectra grid."
            )
        cospectra = make_bounded_kaimal_cospectra(
            target_z_absolute,
            mapped_profile,
            frequencies,
            spectra,
            requested_stress,
            active_mask,
            settings.max_abs_rho_uw,
            args.integration_rule,
        )

        realised_stress = cospectra.realised_stress
        infeasible_count = int(np.count_nonzero(cospectra.infeasible))
        max_stress_change = float(
            np.max(np.abs(realised_stress - requested_stress))
        )
        if infeasible_count:
            if settings.uw_stress_source in ("rho", "uStar", "uniform"):
                raise ValueError(
                    f"{infeasible_count} height(s) cannot realise the "
                    "dictionary stress from uwStressSource="
                    f"{settings.uw_stress_source} under maxAbsRhoUW. That "
                    "source ignores the reduced stress written to profile/"
                    "table files, so the case would be internally "
                    "inconsistent. Reduce the requested stress, use a "
                    "profile/table stress source, or revise the spectra."
                )
            warn(
                f"{infeasible_count} height(s) could not realise the "
                "requested u'w' under maxAbsRhoUW. Their profile/table stress "
                f"was reduced to the feasible co-spectral integral; max "
                f"change {max_stress_change:.6g} m2/s2."
            )

    validation = validate_arrays(
        target_z_absolute,
        mapped_profile,
        realised_stress,
        spectra,
        cospectra,
        frequencies,
        active_mask,
        settings.max_abs_rho_uw,
        effective_spectra_normalisation,
        settings.scale_i,
        args.integration_rule,
    )

    print("Euston MannHybrid profile preparation")
    print(f"  script version              : {SCRIPT_VERSION}")
    print(f"  MannHybrid source commit    : {MANNHYBRID_COMMIT}")
    print(f"  case                        : {case_path}")
    print(f"  source profile              : {source.path}")
    print(
        "  source height range         : "
        f"{source.z[0]:.6g} to {source.z[-1]:.6g} m AGL "
        f"({source.values.shape[0]} rows)"
    )
    if speed_scaling.enabled:
        print(
            "  source reference mean U     : "
            f"{speed_scaling.source_reference_speed:.8g} m/s at "
            f"{speed_scaling.reference_height_agl:.8g} m AGL"
        )
        print(
            "  target reference mean U     : "
            f"{speed_scaling.target_reference_speed:.8g} m/s at "
            f"{speed_scaling.reference_height_agl:.8g} m AGL"
        )
        print(
            "  mean-speed scale factor     : "
            f"{speed_scaling.factor:.10g}"
        )
        if source.has_stress:
            print(
                "  input ninth-column factor   : "
                f"{speed_scaling.factor**2:.10g}"
            )
    else:
        print("  mean-speed scaling          : disabled; source U retained")
    print(f"  inlet patch                 : {settings.patch_name}")
    print(f"  height extraction           : {mesh_backend}")
    print(
        "  inlet faces -> z levels     : "
        f"{raw_face_z.size:,} -> {target_z_absolute.size:,}"
    )
    print(
        "  target z range              : "
        f"{target_z_absolute[0]:.6g} to "
        f"{target_z_absolute[-1]:.6g} m absolute"
    )
    print(f"  ground elevation            : {args.ground_elevation:.6g} m")
    print(
        "  dictionary spectral sources : "
        f"{settings.target_spectra_source}/"
        f"{settings.uw_cospectrum_source}"
    )
    if needs_spectra_table:
        print(
            "  tabulated frequency grid    : "
            f"{settings.n_freq} bins, df_source={settings.source_df:.8g} Hz, "
            f"fMax_source={settings.source_f_max:.8g} Hz"
        )
        print(
            "  active frequency band       : "
            f"{settings.minimum_frequency:.8g} to "
            f"{settings.maximum_frequency:.8g} Hz"
        )
        if settings.target_spectra_source == "tabulated":
            print(
                "  auto-spectrum mode          : tabulated, "
                f"{effective_spectra_normalisation}"
            )
        else:
            print(
                "  auto-spectrum mode          : MannHybrid analytic "
                "vonKarman; written spectra table is the Cuw grid carrier"
            )
            print(
                "  support-spectrum mode       : "
                f"{effective_spectra_normalisation}"
            )
        print(f"  spectral integration        : {args.integration_rule}")
    else:
        print(
            "  auto-spectrum mode          : MannHybrid analytic "
            "vonKarman; no spectra table written"
        )

    if settings.uw_cospectrum_source == "tabulated":
        print(
            "  u-w co-spectrum mode        : tabulated bounded Kaimal "
            "table generated by this script"
        )
        print(f"  shear-stress construction   : {stress_description}")
        print(
            "  max |rho_uw(f)|             : "
            f"{validation['max_abs_rho_uw']:.8g}"
        )
        print(
            "  max Cuw integral error      : "
            f"{validation['max_stress_integral_error']:.3e}"
        )
    elif settings.uw_cospectrum_source == "kaimal":
        print(
            "  u-w co-spectrum mode        : MannHybrid analytic Kaimal; "
            "no uwCoSpectrumProfile written"
        )
        print(f"  shear-stress construction   : {stress_description}")
        print(
            "  Cuw normalisation/clipping  : performed internally by "
            "MannHybrid"
        )
    else:
        print(
            "  u-w co-spectrum mode        : none; target Cuw=0 "
            "(not native Mann Cuw)"
        )
        print(
            "  shear-stress construction   : unused by the target Cuw; "
            f"profile column uses {stress_description}"
        )
        print(
            "  none-mode note              : current public MannHybrid "
            "recolouring removes the native one-point Mann Cuw"
        )
    print(f"  files currently required    : {required_file_note(settings)}")

    if args.dry_run:
        print("Dry run: validation passed; no files were written.")
        return 0

    wind_dir.mkdir(parents=True, exist_ok=True)

    if not backup_path.exists() or args.refresh_source_backup:
        atomic_copy(source_path, backup_path)
        print(f"  preserved ESDU source       : {backup_path}")

    output_specs = active_output_specs

    temporary: dict[str, Path] = {}
    try:
        for kind, destination in output_specs:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary[kind] = _temporary_output(destination)
        write_profile_file(
            temporary["profile"],
            target_z_absolute,
            mapped_profile,
            realised_stress,
            args.precision,
        )
        validate_written_numeric_file(
            temporary["profile"],
            "profile",
            target_z_absolute.size,
            settings.n_freq,
        )
        if needs_spectra_table:
            if spectra is None:
                raise RuntimeError(
                    "Internal error: active spectra output has no data."
                )
            write_spectra_file(
                temporary["spectra"],
                target_z_absolute,
                realised_stress,
                spectra,
                args.precision,
            )
            validate_written_numeric_file(
                temporary["spectra"],
                "spectra",
                target_z_absolute.size,
                settings.n_freq,
            )
        if needs_uw_table:
            if cospectra is None:
                raise RuntimeError(
                    "Internal error: active Cuw output has no data."
                )
            write_uw_file(
                temporary["uw"],
                target_z_absolute,
                realised_stress,
                cospectra.values,
                args.precision,
            )
            validate_written_numeric_file(
                temporary["uw"],
                "uw",
                target_z_absolute.size,
                settings.n_freq,
            )
        for kind, destination in output_specs:
            os.replace(temporary[kind], destination)
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)

    written = [path for _, path in output_specs]
    if not args.no_target_copies:
        target_copies = [(active_profile_path, wind_dir / "targetProfile")]
        if needs_spectra_table:
            target_copies.append(
                (active_spectra_path, wind_dir / "targetSpectraProfile")
            )
        if needs_uw_table:
            target_copies.append(
                (active_uw_path, wind_dir / "targetUWCoSpectrumProfile")
            )
        for source_file, target_file in target_copies:
            if source_file.resolve() != target_file.resolve():
                atomic_copy(source_file, target_file)
                written.append(target_file)

    print("Written and validated:")
    for path in written:
        print(f"  {path}")
    if not args.no_target_copies:
        print(
            "The target* files are compatibility copies for downstream "
            "calibration; current MannHybridTurb itself reads only the active "
            "files named above."
        )
    inactive_candidates = []
    if not needs_spectra_table:
        inactive_candidates.append(active_spectra_path)
        if not args.no_target_copies:
            inactive_candidates.append(wind_dir / "targetSpectraProfile")
    if not needs_uw_table:
        inactive_candidates.append(active_uw_path)
        if not args.no_target_copies:
            inactive_candidates.append(wind_dir / "targetUWCoSpectrumProfile")
    inactive_files = []
    inactive_resolved = set()
    for candidate in inactive_candidates:
        resolved = candidate.resolve()
        if resolved not in active_destinations and resolved not in inactive_resolved:
            inactive_files.append(candidate)
            inactive_resolved.add(resolved)
    if inactive_files:
        print(
            "Inactive/ignored dictionary-mode files were not written or "
            "removed:"
        )
        for path in inactive_files:
            print(f"  {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Map an ESDU profile to inlet boundary-face-centre heights and "
            "write only the profile/spectral inputs required by the current "
            "MannHybridTurbDict source modes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "case",
        nargs="?",
        type=Path,
        default=Path("."),
        help=(
            "OpenFOAM case directory; terminal default is the current "
            "directory and Spyder uses IDE_CASE_PATH"
        ),
    )
    parser.add_argument(
        "--dictionary",
        type=Path,
        default=IDE_DICTIONARY,
        help="MannHybridTurbDict path, relative to the case unless absolute",
    )
    parser.add_argument(
        "--profile-directory",
        type=Path,
        default=IDE_PROFILE_DIRECTORY,
        help=(
            "Output windProfile directory; by default use "
            "constant/windProfileDirectory from the dictionary"
        ),
    )
    parser.add_argument(
        "--source-profile",
        type=Path,
        default=IDE_SOURCE_PROFILE,
        help=(
            "Explicit ESDU source profile; default is profile.esduSource when "
            "present, otherwise the active profile; relative paths use the "
            "case directory"
        ),
    )
    parser.add_argument(
        "--source-backup-name",
        default=IDE_SOURCE_BACKUP_NAME,
        help="One-time preserved source filename inside windProfile",
    )
    refresh_group = parser.add_mutually_exclusive_group()
    refresh_group.add_argument(
        "--refresh-source-backup",
        dest="refresh_source_backup",
        action="store_true",
        default=IDE_REFRESH_SOURCE_BACKUP,
        help=(
            "Replace the preserved source with the current/explicit source; "
            "use only when intentionally installing a new ESDU profile"
        ),
    )
    refresh_group.add_argument(
        "--no-refresh-source-backup",
        dest="refresh_source_backup",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Keep the existing preserved ESDU source",
    )
    parser.add_argument(
        "--patch",
        default=IDE_PATCH,
        help=(
            "Optional expected inlet patch name; it must match patchName in "
            "MannHybridTurbDict"
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "pyvista", "ascii"),
        default=IDE_BACKEND,
        help="Inlet face-centre extraction backend",
    )
    parser.add_argument(
        "--foam-marker",
        default=IDE_FOAM_MARKER,
        help="OpenFOAM marker filename used by PyVista",
    )
    parser.add_argument(
        "--heights-file",
        type=Path,
        default=IDE_HEIGHTS_FILE,
        help=(
            "Optional pre-extracted inlet face-centre elevations or xyz "
            "coordinates, relative to the case unless absolute; bypasses "
            "mesh reading"
        ),
    )
    parser.add_argument(
        "--height-column",
        type=int,
        default=IDE_HEIGHT_COLUMN,
        help=(
            "Zero-based elevation column in --heights-file; auto-select one "
            "column or z from a 3+ column file"
        ),
    )
    parser.add_argument(
        "--coordinate-tolerance",
        type=float,
        default=IDE_COORDINATE_TOLERANCE,
        help=(
            "Override height de-duplication tolerance; default is the exact "
            "coordinateTolerance used by MannHybridTurb"
        ),
    )
    parser.add_argument(
        "--ground-elevation",
        type=float,
        default=IDE_GROUND_ELEVATION,
        help=(
            "Absolute mesh z corresponding to ESDU z=0; Euston flat terrain "
            "uses zero"
        ),
    )
    reference_speed_group = parser.add_mutually_exclusive_group()
    reference_speed_group.add_argument(
        "--target-reference-mean-speed",
        type=float,
        default=IDE_TARGET_REFERENCE_MEAN_SPEED,
        help=(
            "Optional target mean wind speed in m/s; when supplied, scale "
            "the complete U profile so it has this value at "
            "--reference-height-agl"
        ),
    )
    reference_speed_group.add_argument(
        "--retain-source-mean-speed",
        dest="target_reference_mean_speed",
        action="store_const",
        const=None,
        default=argparse.SUPPRESS,
        help=(
            "Disable reference-speed scaling and retain source-profile mean "
            "speeds; overrides a non-None Spyder default"
        ),
    )
    parser.add_argument(
        "--reference-height-agl",
        type=float,
        default=IDE_REFERENCE_HEIGHT_AGL,
        help=(
            "Reference height above ground in metres used by target mean-"
            "speed scaling"
        ),
    )
    parser.add_argument(
        "--extrapolation",
        choices=("hold", "error", "linear"),
        default=IDE_EXTRAPOLATION,
        help=(
            "Behavior outside the ESDU height range; hold matches current "
            "OpenFOAM interpolation"
        ),
    )
    parser.add_argument(
        "--spectra-normalisation",
        choices=("native", "resolved-variance"),
        default=IDE_SPECTRA_NORMALISATION,
        help=(
            "native exactly samples MannHybrid's built-in von Karman model; "
            "resolved-variance rescales each component so its active-band "
            "integral equals (U*I)^2"
        ),
    )
    parser.add_argument(
        "--integration-rule",
        choices=("mannhybrid-sum", "trapezoid"),
        default=IDE_INTEGRATION_RULE,
        help=(
            "Spectral integral used for variance/stress normalisation; "
            "mannhybrid-sum mirrors the current utility's discrete sum*df, "
            "while trapezoid reproduces the predecessor Python workflow"
        ),
    )
    parser.add_argument(
        "--stress-source",
        choices=(
            "dict",
            "profile",
            "rho",
            "uStar",
            "uniform",
            "spectraProfile",
            "uwCoSpectrumProfile",
        ),
        default=IDE_STRESS_SOURCE,
        help="Shear-stress construction mode; dict follows uwStressSource",
    )
    parser.add_argument(
        "--rho-uw",
        type=float,
        default=IDE_RHO_UW,
        help="Override rhoUW from MannHybridTurbDict",
    )
    parser.add_argument(
        "--u-star",
        type=float,
        default=IDE_U_STAR,
        help="Override uStar from MannHybridTurbDict",
    )
    parser.add_argument(
        "--uniform-uw-stress",
        type=float,
        default=IDE_UNIFORM_UW_STRESS,
        help="Override uniformUWStress from MannHybridTurbDict",
    )
    parser.add_argument(
        "--max-abs-rho-uw",
        type=float,
        default=IDE_MAX_ABS_RHO_UW,
        help="Override maxAbsRhoUW from MannHybridTurbDict",
    )
    parser.add_argument(
        "--max-table-values",
        type=int,
        default=IDE_MAX_TABLE_VALUES,
        help=(
            "Memory guard for active spectral arrays: three auto-spectra "
            "plus one co-spectrum only when its tabulated mode is selected"
        ),
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=IDE_PRECISION,
        help="Digits after the decimal in scientific notation",
    )
    target_copy_group = parser.add_mutually_exclusive_group()
    target_copy_group.add_argument(
        "--no-target-copies",
        dest="no_target_copies",
        action="store_true",
        default=not IDE_WRITE_TARGET_COPIES,
        help=(
            "Do not write target* compatibility copies for the files active "
            "in the selected dictionary modes"
        ),
    )
    target_copy_group.add_argument(
        "--target-copies",
        dest="no_target_copies",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Write the target* downstream-calibration compatibility copies",
    )
    run_mode_group = parser.add_mutually_exclusive_group()
    run_mode_group.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=IDE_DRY_RUN,
        help="Read, generate and validate in memory without writing files",
    )
    run_mode_group.add_argument(
        "--write",
        dest="dry_run",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Write validated output files (overrides IDE_DRY_RUN=True)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {SCRIPT_VERSION}",
    )
    return parser


def validate_parsed_options(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    """Validate editable defaults as strictly as explicit CLI choices."""
    for value, label in (
        (IDE_REFRESH_SOURCE_BACKUP, "IDE_REFRESH_SOURCE_BACKUP"),
        (IDE_WRITE_TARGET_COPIES, "IDE_WRITE_TARGET_COPIES"),
        (IDE_DRY_RUN, "IDE_DRY_RUN"),
    ):
        if not isinstance(value, bool):
            parser.error(f"{label} must be either True or False.")

    choice_rules = (
        ("backend", {"auto", "pyvista", "ascii"}, "IDE_BACKEND/--backend"),
        (
            "extrapolation",
            {"hold", "error", "linear"},
            "IDE_EXTRAPOLATION/--extrapolation",
        ),
        (
            "spectra_normalisation",
            {"native", "resolved-variance"},
            "IDE_SPECTRA_NORMALISATION/--spectra-normalisation",
        ),
        (
            "integration_rule",
            {"mannhybrid-sum", "trapezoid"},
            "IDE_INTEGRATION_RULE/--integration-rule",
        ),
        (
            "stress_source",
            {
                "dict",
                "profile",
                "rho",
                "uStar",
                "uniform",
                "spectraProfile",
                "uwCoSpectrumProfile",
            },
            "IDE_STRESS_SOURCE/--stress-source",
        ),
    )
    for attribute, allowed, label in choice_rules:
        value = getattr(args, attribute)
        if value not in allowed:
            parser.error(
                f"{label} must be one of {', '.join(sorted(allowed))}; "
                f"got {value!r}."
            )

    for attribute, label in (
        ("refresh_source_backup", "IDE_REFRESH_SOURCE_BACKUP"),
        ("no_target_copies", "IDE_WRITE_TARGET_COPIES"),
        ("dry_run", "IDE_DRY_RUN"),
    ):
        if not isinstance(getattr(args, attribute), bool):
            parser.error(f"{label} must be either True or False.")

    target_speed = args.target_reference_mean_speed
    reference_height = args.reference_height_agl
    if target_speed is not None and (
        not math.isfinite(target_speed) or target_speed <= 0.0
    ):
        parser.error(
            "IDE_TARGET_REFERENCE_MEAN_SPEED/"
            "--target-reference-mean-speed must be finite and positive."
        )
    if reference_height is not None and (
        not math.isfinite(reference_height) or reference_height <= 0.0
    ):
        parser.error(
            "IDE_REFERENCE_HEIGHT_AGL/--reference-height-agl must be finite "
            "and positive."
        )
    if target_speed is not None and reference_height is None:
        parser.error(
            "A target reference mean speed requires "
            "IDE_REFERENCE_HEIGHT_AGL/--reference-height-agl."
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    if argv is None and "spyder_kernels" in sys.modules:
        # Spyder can retain unrelated kernel arguments. Its Run button is
        # intentionally driven only by the USER SETTINGS block above.
        args = parser.parse_args([str(IDE_CASE_PATH)])
    else:
        args = parser.parse_args(argv)
    validate_parsed_options(parser, args)
    if args.precision < 6 or args.precision > 17:
        parser.error("--precision must be between 6 and 17.")
    if args.max_table_values <= 0:
        parser.error("--max-table-values must be positive.")
    try:
        return prepare(args)
    except (OSError, ValueError, RuntimeError, MemoryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

#%%

target_profile_df = LES._profileCalibration.get_dfsr_target_profile_df(IDE_CASE_PATH)

LES._caseFiles.write_probes_from_target_profile(-450, 0, IDE_CASE_PATH, target_profile_df, "probes2")