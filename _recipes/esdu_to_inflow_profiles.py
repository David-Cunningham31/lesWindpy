#!/usr/bin/env python3
"""Convert an ESDU E0108 workbook to DFSR/MannHybrid ``profile`` files.

The generated files contain numeric rows only, with exactly eight columns:

    z  U  Iu  Iv  Iw  Lu  Lv  Lw

where ``Lu``, ``Lv`` and ``Lw`` are read from the ESDU along-wind
``xLu``, ``xLv`` and ``xLw`` sections.  This headerless eight-column layout is
accepted by both DFSR and MannHybridTurb.  MannHybridTurb can additionally use
an optional ninth u'w' column, but this converter deliberately does not invent
one because an ordinary ESDU E0108 profile workbook does not provide it.

The parser discovers the selected height rows and active direction columns
dynamically.  It does not assume 33 heights, 16 profiles, or fixed Excel row
numbers.

Dependencies
------------
Python 3.9+ and pandas are required.  For an XLSB workbook, install pyxlsb::

    python -m pip install pandas pyxlsb

For XLSX/XLSM input, install openpyxl instead::

    python -m pip install pandas openpyxl

On Windows, the script can also read XLSB through an installed copy of Excel
when pywin32 is available.  The workbook is opened read-only, with macros,
events and external-link updates disabled.

Spyder usage
------------
Edit the USER SETTINGS immediately below, then run this file.  Alternatively,
leave INPUT_WORKBOOK blank and put this script beside exactly one XLSB/XLSX
workbook.

Command-line usage
------------------
    python esdu_to_inflow_profiles.py input.xlsb --output-root ESDU_profiles
    python esdu_to_inflow_profiles.py input.xlsx --dry-run
    python esdu_to_inflow_profiles.py input.xlsb --overwrite

The default output layout is::

    ESDU_profiles/
      10,350,360/profile
      20-40/profile
      ...
      profiles_manifest.csv

No existing profile is overwritten unless OVERWRITE_EXISTING is set below or
``--overwrite`` is supplied.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - dependency failure path
    raise SystemExit(
        "pandas is required. Install it with: python -m pip install pandas"
    ) from exc


# =============================================================================
# USER SETTINGS (used when no corresponding command-line option is supplied)
# =============================================================================

# Example:
# INPUT_WORKBOOK = r"C:\Users\david\...\E0108v2023A1_profiles.xlsb"
# Leave blank to auto-detect exactly one XLSB/XLSX/XLSM workbook beside the
# script or in the current working directory.
INPUT_WORKBOOK = r"C:\Users\david\OneDrive\Documents\PhD\Year 1\Euston Tower LES\E0108v2023A1_Euston_Tower_ESDU_Match_RWDI_Heights.xlsb"

# Leave blank to create <workbook stem>_OpenFOAM_profiles beside the workbook.
OUTPUT_ROOT = r"C:\Users\david\OneDrive\Documents\PhD\Year 1\Euston Tower LES\of_cases\empty_domain"

# Informational only: the emitted eight-column profile is shared by both.
# Accepted values: "both", "dfsr", "mannhybrid".
UTILITY = "both"

# Existing profile/manifest files are protected by default.
OVERWRITE_EXISTING = True

# Number of significant digits written for every scalar.
SIGNIFICANT_DIGITS = 6

# Folder names can be the ESDU direction label itself, or include the case.
# Accepted values: "direction", "case-direction".
FOLDER_STYLE = "direction"


REQUIRED_ROLES = ("wind", "u", "v", "w")
OUTPUT_COLUMNS = ("z", "U", "Iu", "Iv", "Iw", "Lu", "Lv", "Lw")

SHEET_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "wind": ("Wind speed",),
    "u": ("u-component turbulence", "u component turbulence"),
    "v": ("v-component turbulence", "v component turbulence"),
    "w": ("w-component turbulence", "w component turbulence"),
}

METRIC_DEFINITIONS: Mapping[str, Tuple[str, Tuple[str, ...]]] = {
    "U": ("wind", ("vz", "vzms", "uav", "uavms")),
    "Iu": ("u", ("iu",)),
    "Iv": ("v", ("iv",)),
    "Iw": ("w", ("iw",)),
    "Lu": ("u", ("xlu", "xlum")),
    "Lv": ("v", ("xlv", "xlvm")),
    "Lw": ("w", ("xlw", "xlwm")),
}


class ConverterError(RuntimeError):
    """A user-correctable workbook or conversion error."""


@dataclass(frozen=True)
class DirectionAxis:
    row: int
    labels: Tuple[str, ...]
    keys: Tuple[str, ...]
    columns_by_key: Mapping[str, int]
    cases_by_key: Mapping[str, int]


@dataclass(frozen=True)
class MetricBlock:
    name: str
    sheet_name: str
    heights: Tuple[float, ...]
    values: Tuple[Tuple[float, ...], ...]


@dataclass(frozen=True)
class WorkbookData:
    source: Path
    backend: str
    sheet_names: Mapping[str, str]
    direction_labels: Tuple[str, ...]
    cases: Tuple[int, ...]
    heights: Tuple[float, ...]
    metrics: Mapping[str, MetricBlock]


def normalise_text(value: object) -> str:
    """Return a case/punctuation-insensitive key for labels and sheet names."""
    text = cell_text(value).lower()
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    return re.sub(r"[^a-z0-9]+", "", text)


def cell_text(value: object) -> str:
    """Convert an Excel scalar to clean display text; blanks become ''."""
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        if value.is_integer():
            return str(int(value))
        return format(value, ".15g")
    return str(value).strip()


def numeric_value(value: object, context: str) -> float:
    """Return a finite float or raise a detailed workbook-data error."""
    if value is None:
        raise ConverterError(f"Missing numeric value: {context}")
    if isinstance(value, bool):
        raise ConverterError(f"Boolean found where a number is required: {context}")
    try:
        if bool(pd.isna(value)):
            raise ConverterError(f"Blank numeric value: {context}")
    except (TypeError, ValueError):
        pass
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConverterError(
            f"Non-numeric value {value!r} found at {context}. "
            "Open Excel, recalculate the ESDU workbook, save it, and retry."
        ) from exc
    if not math.isfinite(number):
        raise ConverterError(f"Non-finite value {number!r} found at {context}")
    return number


def optional_numeric(value: object) -> Optional[float]:
    if cell_text(value) == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def excel_column_name(index_zero_based: int) -> str:
    """Convert a zero-based column index to an Excel-style column name."""
    number = index_zero_based + 1
    letters = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def dataframe_from_excel_used_range(sheet: object) -> "pd.DataFrame":
    """Convert a win32com worksheet UsedRange.Value2 result to a DataFrame."""
    used = sheet.UsedRange
    start_row = int(used.Row)
    start_col = int(used.Column)
    row_count = int(used.Rows.Count)
    col_count = int(used.Columns.Count)
    raw = used.Value2

    if row_count == 1 and col_count == 1:
        matrix: List[List[object]] = [[raw]]
    elif row_count == 1:
        matrix = [list(raw)]
    elif col_count == 1:
        matrix = [[item[0] if isinstance(item, tuple) else item] for item in raw]
    else:
        matrix = [list(row) for row in raw]

    full_rows = start_row - 1 + row_count
    full_cols = start_col - 1 + col_count
    padded: List[List[object]] = [
        [None] * full_cols for _ in range(full_rows)
    ]
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            padded[start_row - 1 + i][start_col - 1 + j] = value
    return pd.DataFrame(padded, dtype=object)


def read_with_excel_com(
    path: Path,
    requested_sheets: Mapping[str, Optional[str]],
) -> Tuple[Dict[str, "pd.DataFrame"], Dict[str, str], str]:
    """Read required sheets from Excel on Windows without modifying the file."""
    if os.name != "nt":
        raise ConverterError("Excel COM fallback is available only on Windows")
    try:
        import win32com.client  # type: ignore
    except ImportError as exc:
        raise ConverterError(
            "pywin32 is not installed, so the Excel fallback is unavailable"
        ) from exc

    excel = None
    workbook = None
    old_security = None
    try:
        excel = win32com.client.DispatchEx("Excel.Application")
        excel.Visible = False
        excel.DisplayAlerts = False
        excel.EnableEvents = False
        excel.AskToUpdateLinks = False
        try:
            old_security = excel.AutomationSecurity
            excel.AutomationSecurity = 3  # msoAutomationSecurityForceDisable
        except Exception:
            old_security = None

        workbook = excel.Workbooks.Open(
            str(path.resolve()),
            UpdateLinks=0,
            ReadOnly=True,
            IgnoreReadOnlyRecommended=True,
            AddToMru=False,
        )
        available = [str(ws.Name) for ws in workbook.Worksheets]
        resolved = resolve_required_sheet_names(available, requested_sheets)
        frames = {
            role: dataframe_from_excel_used_range(
                workbook.Worksheets(resolved[role])
            )
            for role in REQUIRED_ROLES
        }
        return frames, resolved, "Excel COM (read-only)"
    except ConverterError:
        raise
    except Exception as exc:
        raise ConverterError(f"Excel could not read {path}: {exc}") from exc
    finally:
        if workbook is not None:
            try:
                workbook.Close(SaveChanges=False)
            except Exception:
                pass
        if excel is not None:
            if old_security is not None:
                try:
                    excel.AutomationSecurity = old_security
                except Exception:
                    pass
            try:
                excel.Quit()
            except Exception:
                pass


def candidate_pandas_engines(path: Path, requested: str) -> Tuple[str, ...]:
    if requested != "auto":
        return (requested,)
    suffix = path.suffix.lower()
    if suffix == ".xlsb":
        return ("pyxlsb", "calamine")
    if suffix in {".xlsx", ".xlsm"}:
        return ("openpyxl", "calamine")
    if suffix == ".xls":
        return ("xlrd", "calamine")
    raise ConverterError(
        f"Unsupported workbook extension {path.suffix!r}; use XLSB, XLSX, XLSM or XLS"
    )


def resolve_required_sheet_names(
    available: Sequence[str],
    requested: Mapping[str, Optional[str]],
) -> Dict[str, str]:
    norm_to_names: Dict[str, List[str]] = {}
    for name in available:
        norm_to_names.setdefault(normalise_text(name), []).append(name)

    resolved: Dict[str, str] = {}
    for role in REQUIRED_ROLES:
        explicit = requested.get(role)
        aliases: Iterable[str] = (explicit,) if explicit else SHEET_ALIASES[role]
        matches: List[str] = []
        for alias in aliases:
            if not alias:
                continue
            matches.extend(norm_to_names.get(normalise_text(alias), []))
        matches = list(dict.fromkeys(matches))
        if len(matches) != 1:
            target = explicit or " / ".join(SHEET_ALIASES[role])
            if not matches:
                raise ConverterError(
                    f"Could not find required worksheet {target!r}. "
                    f"Available worksheets: {', '.join(available)}"
                )
            raise ConverterError(
                f"Worksheet name {target!r} is ambiguous: {', '.join(matches)}"
            )
        resolved[role] = matches[0]
    return resolved


def read_required_sheets(
    path: Path,
    requested_sheets: Mapping[str, Optional[str]],
    engine: str,
) -> Tuple[Dict[str, "pd.DataFrame"], Dict[str, str], str]:
    """Read only the four output sheets needed for profile generation."""
    if engine == "excel-com":
        return read_with_excel_com(path, requested_sheets)

    errors: List[str] = []
    for candidate in candidate_pandas_engines(path, engine):
        try:
            excel_file = pd.ExcelFile(path, engine=candidate)
            resolved = resolve_required_sheet_names(
                list(excel_file.sheet_names), requested_sheets
            )
            frames = {
                role: pd.read_excel(
                    excel_file,
                    sheet_name=resolved[role],
                    header=None,
                    dtype=object,
                )
                for role in REQUIRED_ROLES
            }
            return frames, resolved, f"pandas/{candidate}"
        except ConverterError:
            raise
        except Exception as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")

    if path.suffix.lower() == ".xlsb" and os.name == "nt" and engine == "auto":
        try:
            return read_with_excel_com(path, requested_sheets)
        except ConverterError as exc:
            errors.append(f"Excel COM: {exc}")

    install_hint = ""
    if path.suffix.lower() == ".xlsb":
        install_hint = (
            "\nInstall XLSB support with: python -m pip install pyxlsb\n"
            "Alternatively, save a recalculated copy as XLSX and install openpyxl."
        )
    raise ConverterError(
        f"Could not read workbook {path}. Tried:\n  "
        + "\n  ".join(errors)
        + install_hint
    )


def direction_key(label: str) -> str:
    key = normalise_text(label)
    if not key:
        raise ConverterError(f"Empty/invalid direction label {label!r}")
    return key


def find_case_values(
    frame: "pd.DataFrame",
    direction_row: int,
    labels: Sequence[str],
    columns: Sequence[int],
) -> Dict[str, int]:
    """Find the nearby Case row; fall back to one-based profile order."""
    search_start = max(0, direction_row - 4)
    search_stop = min(len(frame.index), direction_row + 2)
    case_row: Optional[int] = None
    for row in range(search_start, search_stop):
        for value in frame.iloc[row, : min(columns)].tolist():
            if normalise_text(value).startswith("case"):
                case_row = row
                break
        if case_row is not None:
            break

    result: Dict[str, int] = {}
    for ordinal, (label, column) in enumerate(zip(labels, columns), start=1):
        case = ordinal
        if case_row is not None:
            raw = optional_numeric(frame.iat[case_row, column])
            if raw is not None and raw.is_integer():
                case = int(raw)
        result[direction_key(label)] = case
    return result


def find_direction_axis(frame: "pd.DataFrame", sheet_name: str) -> DirectionAxis:
    """Find the active ESDU columns from the populated wind-direction row."""
    candidates: List[Tuple[int, int, List[str], List[int]]] = []
    rows, cols = frame.shape
    for row in range(rows):
        for col in range(cols):
            key = normalise_text(frame.iat[row, col])
            if "winddirection" not in key:
                continue
            labels: List[str] = []
            columns: List[int] = []
            for profile_col in range(col + 1, cols):
                label = cell_text(frame.iat[row, profile_col])
                if not label:
                    if labels:
                        break
                    continue
                labels.append(label)
                columns.append(profile_col)
            if labels:
                candidates.append((row, col, labels, columns))

    if not candidates:
        raise ConverterError(
            f"No populated 'Wind direction' row found in worksheet {sheet_name!r}"
        )
    # Prefer the candidate with the most populated, contiguous profile labels.
    direction_row, _, labels, columns = max(candidates, key=lambda x: len(x[2]))
    keys = [direction_key(label) for label in labels]
    if len(set(keys)) != len(keys):
        raise ConverterError(
            f"Duplicate direction labels found in worksheet {sheet_name!r}: {labels}"
        )
    cases = find_case_values(frame, direction_row, labels, columns)
    return DirectionAxis(
        row=direction_row,
        labels=tuple(labels),
        keys=tuple(keys),
        columns_by_key=dict(zip(keys, columns)),
        cases_by_key=cases,
    )


def metric_label_matches(value: object, aliases: Sequence[str]) -> bool:
    key = normalise_text(value)
    if not key:
        return False
    alias_keys = {normalise_text(alias) for alias in aliases}
    if key in alias_keys:
        return True
    # Unit suffixes sometimes vary between ESDU releases.
    return any(key.startswith(alias) for alias in alias_keys if len(alias) >= 3)


def find_metric_and_height_header(
    frame: "pd.DataFrame",
    sheet_name: str,
    metric_name: str,
    aliases: Sequence[str],
) -> Tuple[int, int]:
    """Find the metric label and its associated Height-above-ground header."""
    rows, cols = frame.shape
    candidates: List[Tuple[int, int, int, int]] = []
    for metric_row in range(rows):
        for metric_col in range(cols):
            if not metric_label_matches(frame.iat[metric_row, metric_col], aliases):
                continue
            # In E0108, the height header is on the metric row or just below it.
            for height_row in range(metric_row, min(rows, metric_row + 4)):
                for height_col in range(cols):
                    height_key = normalise_text(frame.iat[height_row, height_col])
                    if height_key.startswith("heightaboveground"):
                        distance = abs(height_row - metric_row) + abs(
                            height_col - metric_col
                        )
                        candidates.append(
                            (distance, height_row, height_col, metric_row)
                        )

    if not candidates:
        raise ConverterError(
            f"Could not locate metric {metric_name!r} and its height header "
            f"in worksheet {sheet_name!r}"
        )
    _, height_row, height_col, _ = min(candidates)
    return height_row, height_col


def extract_metric_block(
    frame: "pd.DataFrame",
    sheet_name: str,
    metric_name: str,
    aliases: Sequence[str],
    ordered_profile_columns: Sequence[int],
) -> MetricBlock:
    """Extract one contiguous height/value block beneath a discovered header."""
    header_row, height_col = find_metric_and_height_header(
        frame, sheet_name, metric_name, aliases
    )
    heights: List[float] = []
    values: List[Tuple[float, ...]] = []
    started = False

    for row in range(header_row + 1, len(frame.index)):
        height = optional_numeric(frame.iat[row, height_col])
        if height is None:
            if started:
                break
            continue

        row_values: List[float] = []
        for column in ordered_profile_columns:
            coordinate = f"{sheet_name}!{excel_column_name(column)}{row + 1}"
            row_values.append(numeric_value(frame.iat[row, column], coordinate))
        heights.append(float(height))
        values.append(tuple(row_values))
        started = True

    if not heights:
        raise ConverterError(
            f"Metric {metric_name!r} in worksheet {sheet_name!r} has no numeric rows"
        )
    return MetricBlock(
        name=metric_name,
        sheet_name=sheet_name,
        heights=tuple(heights),
        values=tuple(values),
    )


def heights_match(left: Sequence[float], right: Sequence[float]) -> bool:
    return len(left) == len(right) and all(
        math.isclose(a, b, rel_tol=1.0e-10, abs_tol=1.0e-9)
        for a, b in zip(left, right)
    )


def validate_workbook_data(data: WorkbookData) -> List[str]:
    """Return warnings after raising on all unsafe/invalid conditions."""
    warnings: List[str] = []
    heights = data.heights
    if not heights:
        raise ConverterError("The extracted height vector is empty")
    if any(not math.isfinite(z) or z < 0.0 for z in heights):
        raise ConverterError("Heights must be finite and non-negative")
    if any(b <= a for a, b in zip(heights, heights[1:])):
        raise ConverterError("Heights must be unique and strictly increasing")

    for name, block in data.metrics.items():
        if not heights_match(heights, block.heights):
            raise ConverterError(
                f"Height vector mismatch for {name} in {block.sheet_name!r}.\n"
                f"Mean-wind heights: {list(heights)}\n"
                f"{name} heights: {list(block.heights)}\n"
                "Use the same selected heights for every ESDU output block; "
                "the converter will not interpolate silently."
            )
        if any(len(row) != len(data.direction_labels) for row in block.values):
            raise ConverterError(f"Incomplete profile columns in metric {name}")

    for profile_index, label in enumerate(data.direction_labels):
        for height_index, z in enumerate(heights):
            row = {
                name: data.metrics[name].values[height_index][profile_index]
                for name in ("U", "Iu", "Iv", "Iw", "Lu", "Lv", "Lw")
            }
            if row["U"] <= 0.0:
                raise ConverterError(
                    f"Mean velocity must be positive: profile {label!r}, z={z:g} m"
                )
            for name in ("Iu", "Iv", "Iw"):
                if row[name] < 0.0:
                    raise ConverterError(
                        f"{name} must be non-negative: profile {label!r}, z={z:g} m"
                    )
                if row[name] > 1.0:
                    warnings.append(
                        f"Suspicious {name}={row[name]:g} (>1) in profile "
                        f"{label!r} at z={z:g} m; intensities must be fractions, "
                        "not percentages."
                    )
            for name in ("Lu", "Lv", "Lw"):
                if row[name] <= 0.0:
                    raise ConverterError(
                        f"{name} must be positive: profile {label!r}, z={z:g} m"
                    )
    if heights[0] > 0.0:
        warnings.append(
            f"The first supplied height is {heights[0]:g} m rather than 0 m. "
            "This is preserved exactly; verify that the utility's extrapolation "
            "below the first point is appropriate for your inlet mesh."
        )
    return list(dict.fromkeys(warnings))


def load_workbook_data(
    path: Path,
    engine: str,
    requested_sheets: Mapping[str, Optional[str]],
) -> WorkbookData:
    frames, sheet_names, backend = read_required_sheets(
        path, requested_sheets, engine
    )
    axes = {
        role: find_direction_axis(frames[role], sheet_names[role])
        for role in REQUIRED_ROLES
    }
    base_axis = axes["wind"]
    base_keys = base_axis.keys
    base_key_set = set(base_keys)

    for role in REQUIRED_ROLES[1:]:
        found = set(axes[role].keys)
        if found != base_key_set:
            missing = [
                base_axis.labels[base_keys.index(key)]
                for key in base_keys
                if key not in found
            ]
            extra = [
                axes[role].labels[axes[role].keys.index(key)]
                for key in axes[role].keys
                if key not in base_key_set
            ]
            raise ConverterError(
                f"Direction labels in worksheet {sheet_names[role]!r} do not "
                f"match {sheet_names['wind']!r}. Missing={missing}; extra={extra}"
            )

    metrics: Dict[str, MetricBlock] = {}
    for metric_name, (role, aliases) in METRIC_DEFINITIONS.items():
        columns = [axes[role].columns_by_key[key] for key in base_keys]
        metrics[metric_name] = extract_metric_block(
            frames[role],
            sheet_names[role],
            metric_name,
            aliases,
            columns,
        )

    cases = tuple(base_axis.cases_by_key[key] for key in base_keys)
    result = WorkbookData(
        source=path,
        backend=backend,
        sheet_names=sheet_names,
        direction_labels=base_axis.labels,
        cases=cases,
        heights=metrics["U"].heights,
        metrics=metrics,
    )
    validate_workbook_data(result)
    return result


WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def safe_folder_component(label: str) -> str:
    """Preserve readable ESDU labels while removing cross-platform hazards."""
    clean = label.replace("–", "-").replace("—", "-").replace("−", "-")
    clean = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", clean)
    clean = re.sub(r"\s+", " ", clean).strip().rstrip(". ")
    if not clean:
        clean = "unnamed_profile"
    if clean.upper() in WINDOWS_RESERVED_NAMES:
        clean = f"_{clean}"
    return clean


def build_folder_names(
    labels: Sequence[str],
    cases: Sequence[int],
    style: str,
) -> Tuple[str, ...]:
    result: List[str] = []
    used: set[str] = set()
    for label, case in zip(labels, cases):
        direction = safe_folder_component(label)
        if style == "case-direction":
            folder = f"case_{case:02d}__{direction.replace(',', '_')}"
        else:
            folder = direction
        candidate = folder
        suffix = 1
        while candidate.casefold() in used:
            suffix += 1
            candidate = f"{folder}__case_{case:02d}_{suffix}"
        used.add(candidate.casefold())
        result.append(candidate)
    return tuple(result)


def format_scalar(value: float, significant_digits: int) -> str:
    text = format(float(value), f".{significant_digits}g")
    # Avoid a textual negative zero if a source value is extremely small.
    try:
        if float(text) == 0.0:
            return "0"
    except ValueError:  # pragma: no cover
        pass
    return text


def build_profile_text(
    data: WorkbookData,
    profile_index: int,
    significant_digits: int,
) -> str:
    lines: List[str] = []
    for height_index, z in enumerate(data.heights):
        row = [
            z,
            data.metrics["U"].values[height_index][profile_index],
            data.metrics["Iu"].values[height_index][profile_index],
            data.metrics["Iv"].values[height_index][profile_index],
            data.metrics["Iw"].values[height_index][profile_index],
            data.metrics["Lu"].values[height_index][profile_index],
            data.metrics["Lv"].values[height_index][profile_index],
            data.metrics["Lw"].values[height_index][profile_index],
        ]
        lines.append(
            "\t".join(format_scalar(value, significant_digits) for value in row)
        )
    return "\n".join(lines) + "\n"


def build_manifest_text(
    data: WorkbookData,
    folder_names: Sequence[str],
    utility: str,
) -> str:
    buffer = io.StringIO(newline="")
    fieldnames = [
        "case",
        "esdu_direction_label",
        "folder",
        "profile_file",
        "utility",
        "columns",
        "n_heights",
        "z_min_m",
        "z_max_m",
        "source_workbook",
        "wind_sheet",
        "u_sheet",
        "v_sheet",
        "w_sheet",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for case, label, folder in zip(
        data.cases, data.direction_labels, folder_names
    ):
        writer.writerow(
            {
                "case": case,
                "esdu_direction_label": label,
                "folder": folder,
                "profile_file": f"{folder}/constant/boundaryData/windProfile/profile",
                "utility": utility,
                "columns": " ".join(OUTPUT_COLUMNS),
                "n_heights": len(data.heights),
                "z_min_m": format(data.heights[0], ".15g"),
                "z_max_m": format(data.heights[-1], ".15g"),
                "source_workbook": str(data.source),
                "wind_sheet": data.sheet_names["wind"],
                "u_sheet": data.sheet_names["u"],
                "v_sheet": data.sheet_names["v"],
                "w_sheet": data.sheet_names["w"],
            }
        )
    return buffer.getvalue()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_outputs(
    data: WorkbookData,
    output_root: Path,
    utility: str,
    significant_digits: int,
    folder_style: str,
    overwrite: bool,
    dry_run: bool,
) -> Tuple[Tuple[str, ...], List[str]]:
    warnings = validate_workbook_data(data)
    folder_names = build_folder_names(
        data.direction_labels, data.cases, folder_style
    )
    profile_paths = [output_root / folder / "constant" / "boundaryData" / "windProfile" / "profile" for folder in folder_names]
    manifest_path = output_root / "profiles_manifest.csv"
    conflicts = [path for path in [*profile_paths, manifest_path] if path.exists()]
    if conflicts and not overwrite:
        preview = "\n  ".join(str(path) for path in conflicts[:10])
        more = "" if len(conflicts) <= 10 else f"\n  ...and {len(conflicts)-10} more"
        raise ConverterError(
            "Output files already exist and overwrite protection is enabled:\n  "
            + preview
            + more
            + "\nUse --overwrite (or set OVERWRITE_EXISTING=True) to replace only "
            "these profile/manifest files. Other folder contents are retained."
        )

    if dry_run:
        return folder_names, warnings

    # Build all text before the first filesystem write so data errors cannot
    # leave a partially generated profile set.
    profile_texts = [
        build_profile_text(data, index, significant_digits)
        for index in range(len(data.direction_labels))
    ]
    manifest_text = build_manifest_text(data, folder_names, utility)
    for path, text in zip(profile_paths, profile_texts):
        atomic_write_text(path, text)
    atomic_write_text(manifest_path, manifest_text)
    return folder_names, warnings


def discover_input_workbook(cli_value: Optional[str]) -> Path:
    explicit = cli_value or INPUT_WORKBOOK.strip()
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise ConverterError(f"Input workbook does not exist: {path}")
        return path.resolve()

    search_directories = [Path.cwd(), Path(__file__).resolve().parent]
    candidates: List[Path] = []
    seen: set[Path] = set()
    for directory in search_directories:
        for suffix in ("*.xlsb", "*.xlsx", "*.xlsm", "*.xls"):
            for path in directory.glob(suffix):
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    candidates.append(resolved)
    if len(candidates) != 1:
        listing = "\n  ".join(str(path) for path in candidates) or "(none found)"
        raise ConverterError(
            "No input workbook was specified and auto-detection did not find "
            f"exactly one candidate. Candidates:\n  {listing}\n"
            "Set INPUT_WORKBOOK at the top of the script or pass the path as "
            "the first command-line argument."
        )
    return candidates[0]


def default_output_root(workbook: Path, cli_value: Optional[str]) -> Path:
    explicit = cli_value or OUTPUT_ROOT.strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    return workbook.with_name(f"{workbook.stem}_OpenFOAM_profiles")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert populated ESDU E0108 wind-profile outputs into one "
            "headerless DFSR/MannHybrid profile file per direction group."
        )
    )
    parser.add_argument("workbook", nargs="?", help="Input XLSB/XLSX/XLSM/XLS")
    parser.add_argument("--output-root", help="Root directory for profile folders")
    parser.add_argument(
        "--utility",
        choices=("both", "dfsr", "mannhybrid"),
        default=UTILITY.lower(),
        help="Target utility label; all choices emit the shared eight columns",
    )
    parser.add_argument(
        "--engine",
        choices=("auto", "pyxlsb", "calamine", "openpyxl", "xlrd", "excel-com"),
        default="auto",
        help="Spreadsheet reader backend (default: auto)",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=SIGNIFICANT_DIGITS,
        help="Significant digits per scalar (default: %(default)s)",
    )
    parser.add_argument(
        "--folder-style",
        choices=("direction", "case-direction"),
        default=FOLDER_STYLE,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=OVERWRITE_EXISTING,
        help="Replace existing profile/manifest files only",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report without writing files",
    )
    parser.add_argument("--wind-sheet", help="Override wind-speed worksheet name")
    parser.add_argument("--u-sheet", help="Override u-turbulence worksheet name")
    parser.add_argument("--v-sheet", help="Override v-turbulence worksheet name")
    parser.add_argument("--w-sheet", help="Override w-turbulence worksheet name")
    return parser


def run(args: argparse.Namespace) -> int:
    if not (6 <= args.precision <= 17):
        raise ConverterError("--precision must be between 6 and 17 significant digits")
    workbook = discover_input_workbook(args.workbook)
    output_root = default_output_root(workbook, args.output_root)
    requested_sheets = {
        "wind": args.wind_sheet,
        "u": args.u_sheet,
        "v": args.v_sheet,
        "w": args.w_sheet,
    }

    data = load_workbook_data(workbook, args.engine, requested_sheets)
    folder_names, warnings = write_outputs(
        data=data,
        output_root=output_root,
        utility=args.utility,
        significant_digits=args.precision,
        folder_style=args.folder_style,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    action = "Validated" if args.dry_run else "Wrote"
    print(f"{action} {len(data.direction_labels)} profile(s).")
    print(
        f"Heights: {len(data.heights)} rows, "
        f"{data.heights[0]:g} to {data.heights[-1]:g} m"
    )
    print(f"Columns: {' '.join(OUTPUT_COLUMNS)}")
    print(f"Reader: {data.backend}")
    print(f"Output root: {output_root}")
    for case, label, folder in zip(
        data.cases, data.direction_labels, folder_names
    ):
        print(f"  case {case:02d}: {label} -> {folder}/profile")
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if args.utility in {"both", "mannhybrid"}:
        print(
            "MannHybrid note: this file intentionally has eight columns. "
            "Configure uwStressSource rho (or provide a separate defensible "
            "nine-column stress profile if uwStressSource profile is required)."
        )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = make_parser()
    try:
        return run(parser.parse_args(argv))
    except ConverterError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
