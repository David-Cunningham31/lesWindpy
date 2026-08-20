# -*- coding: utf-8 -*-
"""
Convert the NHERI/FSU pressure-tap coordinates to LES/OpenFOAM coordinates,
rotate them with the building for an arbitrary wind-angle case, and write
order-safe patchProbes dictionaries plus explicit probe-index manifests.

Rotation convention
-------------------
CASE_ANGLE_DEG is the angle through which the BUILDING has been rotated in
plan about the vertical axis through LES_BUILDING_BOTTOM_CENTRE.

    positive angle = anticlockwise when viewed from +z (plan view)
    negative angle = clockwise when viewed from +z

For a local model coordinate (X, Y), the plan transformation is

    x = x0 + X*cos(theta) - Y*sin(theta)
    y = y0 + X*sin(theta) + Y*cos(theta)
    z = z0 + Z

The spreadsheet origin is the bottom-centre of the model, so this is exactly a
rotation about the LES point (2.5, 0, 0) when the default settings are used.

Tap-order warning
-----------------
The spreadsheet row order is not the same as the DesignSafe .mat-column order.
In particular, roof taps 10909 and 10910 have DesignSafe IDs 509 and 510 even
though they occur among the early roof rows.  Therefore:

* DesignSafe ID is the canonical order used for experimental Cp matching.
* A combined file, probesAllTaps, is written in exact DesignSafe order 1..510.
* Per-surface files are also written, but each has a manifest identifying the
  exact DesignSafe ID corresponding to every OpenFOAM Probe index.
* Do not concatenate the five per-surface pressure arrays blindly.  Surface 1
  contains IDs 1..88 and 509..510; use the generated reassembly manifest.

The script creates QA plots and CSV files before optionally copying the probe
files to <case>/system.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#%% --------------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------------

CASE_PATH = r"C:\Users\david\OneDrive\Documents\PhD\Year 1\NHERI LES Case\OpenFOAM Cases\Building Case\060\mannHybrid"

TAP_LAYOUT_XLSX = r"C:\Users\david\OneDrive\Documents\PhD\Year 1\Wind Tunnel Test Data\NHERI BLWT Tall Building\Pressure Taps\Tap Layout - FSU - Mid-Rise-Model.xlsx"

# Positive = anticlockwise building rotation in plan, viewed from +z.
# Example: use 10.0 for the 10-degree case described in the request.
CASE_ANGLE_DEG = 60.0

# Vertical rotation axis passes through this bottom-centre point.
LES_BUILDING_BOTTOM_CENTRE = (2.5, 0.0, 0.0)

# NHERI model dimensions after conversion to metres.
BUILDING_B = 0.30  # local X extent
BUILDING_W = 0.60  # local Y extent
BUILDING_H = 0.50

INPUT_UNITS = "mm"  # "mm" or "m"

# Optional coordinate-system changes.  For the supplied workbook and the
# stated LES geometry these should remain False.
SWAP_XY = False
FLIP_X = False
FLIP_Y = False

# OpenFOAM patchProbes settings.
PATCH_NAME = "building"
FIELD_NAME = "p"
PROBE_FILE_PREFIX = "probesSurface"
COMBINED_PROBE_FILE_NAME = "probesAllTaps"
PROBE_WRITE_CONTROL = "adjustableRunTime"
PROBE_WRITE_INTERVAL = 0.0016  # None -> every time step (writeInterval 1)

# Canonical probe ordering.  Keep DesignSafe ordering for comparison with the
# experimental .mat pressure columns.
PROBE_ORDER = "designsafe_id"  # "designsafe_id", "spreadsheet", or "tap"
EXPECTED_VALID_TAP_COUNT = 510
REQUIRE_CONTIGUOUS_DESIGNSAFE_IDS = True
KEEP_ONLY_DESIGNSAFE_VALID_TAPS = True

# Write both an exact-global-order file and the legacy per-surface files.
WRITE_COMBINED_PROBE_FILE = True
WRITE_PER_SURFACE_PROBE_FILES = True
ADD_ID_COMMENTS_TO_PROBE_FILES = True

# The deterministic local writer is the default because it preserves the
# DataFrame row order exactly and embeds the tap IDs as comments.  The optional
# windlespy writer is retained only for compatibility.
USE_WINDLESPY_WRITER_IF_AVAILABLE = False

WRITE_PROBE_FILES = True
WRITE_TO_CASE_SYSTEM = False
BACKUP_EXISTING_SYSTEM_PROBES = True

# If None, an angle-specific folder below the case postProcessing directory is
# used.  This avoids overwriting QA results from another angle.
OUTPUT_DIR_OVERRIDE = None

# Optional validation against a completed OpenFOAM mesh.
VALIDATE_AGAINST_MESH = False
FOAM_FILE = None  # e.g. os.path.join(CASE_PATH, "case.foam")
MAX_NEAREST_PATCH_DISTANCE = 0.005  # distance to nearest patch cell centre [m]

# Numerical tolerances.
PLANE_TOLERANCE_M = 2.0e-8
ROTATION_TOLERANCE_M = 2.0e-10


#%% --------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | os.PathLike) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def angle_tag(angle_deg: float) -> str:
    """Filesystem-safe angle tag, e.g. +10 -> angle_p010p00deg."""
    value = f"{float(angle_deg):+07.2f}"
    value = value.replace("+", "p").replace("-", "m").replace(".", "p")
    return f"angle_{value}deg"


def resolved_output_dir() -> str:
    if OUTPUT_DIR_OVERRIDE:
        return str(OUTPUT_DIR_OVERRIDE)
    return os.path.join(
        CASE_PATH,
        "postProcessing",
        "pressureTapCoordinateQA",
        angle_tag(CASE_ANGLE_DEG),
    )


def _numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _units_scale(units: str) -> float:
    text = str(units).lower()
    if text in {"mm", "millimetre", "millimetres", "millimeter", "millimeters"}:
        return 1.0e-3
    if text in {"m", "metre", "metres", "meter", "meters"}:
        return 1.0
    raise ValueError("INPUT_UNITS must be 'mm' or 'm'.")


def _format_openfoam_scalar(value) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.12g}"
    return str(value)


def _normalise_probe_write_settings(
    write_control: Optional[str] = None,
    write_interval=None,
) -> Tuple[str, object]:
    if write_control is None:
        write_control = PROBE_WRITE_CONTROL
    if write_interval is None:
        write_interval = PROBE_WRITE_INTERVAL
    if write_interval is None:
        write_interval = 1
    if isinstance(write_interval, (int, float)) and write_interval <= 0:
        raise ValueError("write_interval must be positive or None.")
    return str(write_control), write_interval


#%% --------------------------------------------------------------------------
# Workbook reading and order validation
# ---------------------------------------------------------------------------

def read_tap_layout(
    xlsx_path: str | os.PathLike,
    keep_only_designsafe_valid: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read, clean, and validate the NHERI tap layout.

    Returns
    -------
    taps
        Valid taps, sorted in canonical DesignSafe order.
    corners
        Model corner coordinates.
    excluded_rows
        Spreadsheet rows excluded from the active probe set.
    """
    taps_all = pd.read_excel(xlsx_path, sheet_name="Taps")
    corners = pd.read_excel(xlsx_path, sheet_name="Model Corners")

    required = ["Tap", "Surface", "X", "Y", "Z"]
    missing = [c for c in required if c not in taps_all.columns]
    if missing:
        raise ValueError(f"Tap-layout sheet is missing required columns: {missing}")

    taps_all = taps_all.copy()
    taps_all.insert(0, "excel_row", np.arange(2, len(taps_all) + 2, dtype=int))
    taps_all["spreadsheet_data_index_zero_based"] = np.arange(len(taps_all), dtype=int)

    for col in ["Surface", "X", "Y", "Z"]:
        taps_all[col] = _numeric_series(taps_all[col])

    if "ID (.mat file) - DesignSafe" in taps_all.columns:
        taps_all["designsafe_id"] = _numeric_series(
            taps_all["ID (.mat file) - DesignSafe"]
        )
    else:
        warnings.warn(
            "No DesignSafe ID column was found; sequential IDs are being generated. "
            "This is not recommended for experimental Cp matching."
        )
        taps_all["designsafe_id"] = np.arange(1, len(taps_all) + 1, dtype=int)

    has_geometry = taps_all[["Surface", "X", "Y", "Z"]].notna().all(axis=1)
    has_designsafe = np.isfinite(taps_all["designsafe_id"].to_numpy(dtype=float))
    active_mask = has_geometry & (has_designsafe if keep_only_designsafe_valid else True)

    excluded_rows = taps_all.loc[~active_mask].copy()
    taps = taps_all.loc[active_mask].copy()

    taps["Surface"] = taps["Surface"].astype(int)
    taps["designsafe_id"] = taps["designsafe_id"].astype(int)
    taps["spreadsheet_valid_order_zero_based"] = np.arange(len(taps), dtype=int)

    # Validate unique identifiers before sorting.
    duplicated_ids = taps.loc[taps["designsafe_id"].duplicated(False), "designsafe_id"]
    if not duplicated_ids.empty:
        raise ValueError(
            "Duplicate DesignSafe IDs found: "
            + ", ".join(map(str, sorted(duplicated_ids.unique())))
        )

    numeric_taps = pd.to_numeric(taps["Tap"], errors="coerce")
    if numeric_taps.isna().any():
        bad_rows = taps.loc[numeric_taps.isna(), ["excel_row", "Tap"]]
        raise ValueError(f"Active rows contain non-numeric Tap IDs:\n{bad_rows}")
    taps["Tap"] = numeric_taps.astype(int)
    duplicated_taps = taps.loc[taps["Tap"].duplicated(False), "Tap"]
    if not duplicated_taps.empty:
        raise ValueError(
            "Duplicate physical Tap IDs found: "
            + ", ".join(map(str, sorted(duplicated_taps.unique())))
        )

    if EXPECTED_VALID_TAP_COUNT is not None and len(taps) != EXPECTED_VALID_TAP_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_VALID_TAP_COUNT} valid taps but found {len(taps)}."
        )

    sorted_ids = np.sort(taps["designsafe_id"].to_numpy(dtype=int))
    if REQUIRE_CONTIGUOUS_DESIGNSAFE_IDS:
        expected_ids = np.arange(1, len(taps) + 1, dtype=int)
        if not np.array_equal(sorted_ids, expected_ids):
            missing_ids = sorted(set(expected_ids) - set(sorted_ids))
            extra_ids = sorted(set(sorted_ids) - set(expected_ids))
            raise ValueError(
                "DesignSafe IDs are not the expected contiguous range 1..N. "
                f"Missing={missing_ids}; extra={extra_ids}."
            )

    # Canonical order used by the experimental .mat pressure columns.
    taps = taps.sort_values("designsafe_id", kind="mergesort").reset_index(drop=True)
    taps["designsafe_array_index_zero_based"] = taps["designsafe_id"] - 1
    taps["canonical_order_one_based"] = np.arange(1, len(taps) + 1, dtype=int)

    corners = corners.copy()
    for col in ["X", "Y", "Z"]:
        corners[col] = _numeric_series(corners[col])
    corners = corners.dropna(subset=["X", "Y", "Z"]).reset_index(drop=True)
    if len(corners) != 8:
        raise ValueError(f"Expected 8 model corners but found {len(corners)}.")

    return taps, corners, excluded_rows


def workbook_order_qa(taps: pd.DataFrame) -> pd.DataFrame:
    """Create a table exposing differences among workbook, Tap, and .mat order."""
    qa = taps[
        [
            "excel_row",
            "spreadsheet_data_index_zero_based",
            "spreadsheet_valid_order_zero_based",
            "designsafe_id",
            "designsafe_array_index_zero_based",
            "Tap",
            "Surface",
            "Module",
            "Channel",
            "X",
            "Y",
            "Z",
        ]
    ].copy()
    qa["excel_position_if_sorted_by_designsafe_one_based"] = np.arange(1, len(qa) + 1)
    qa["is_excel_row_order_equal_to_designsafe_order"] = (
        qa["spreadsheet_valid_order_zero_based"]
        == qa["designsafe_array_index_zero_based"]
    )
    return qa


def print_ordering_summary(taps: pd.DataFrame, excluded_rows: pd.DataFrame) -> None:
    print(f"Read {len(taps)} valid DesignSafe taps.")
    print(f"Excluded {len(excluded_rows)} spreadsheet row(s) from active probes.")
    if len(excluded_rows):
        cols = [
            c
            for c in [
                "excel_row",
                "Module",
                "Channel",
                "Tap",
                "Surface",
                "X",
                "Y",
                "Z",
                "ID (.mat file) - DesignSafe",
            ]
            if c in excluded_rows.columns
        ]
        print(excluded_rows[cols].to_string(index=False))

    print("\nActive tap counts by surface:")
    print(taps.groupby("Surface")["designsafe_id"].count().rename("n_taps"))

    print("\nDesignSafe IDs by surface:")
    for sid, group in taps.groupby("Surface", sort=True):
        ids = group["designsafe_id"].to_numpy(dtype=int)
        contiguous = np.all(np.diff(ids) == 1) if len(ids) > 1 else True
        print(
            f"  Surface {sid}: n={len(ids)}, min={ids.min()}, max={ids.max()}, "
            f"contiguous_within_file={contiguous}"
        )
        if not contiguous:
            print(f"    exact non-contiguous tail/head: {ids[:10].tolist()} ... {ids[-10:].tolist()}")

    roof_ids = taps.loc[taps["Surface"] == 1, "designsafe_id"].to_numpy(dtype=int)
    if 509 in roof_ids and 510 in roof_ids:
        print(
            "\nIMPORTANT ORDERING NOTE: Surface 1 contains DesignSafe IDs "
            "1..88 and 509..510.  Do not reconstruct global experimental order "
            "by simply concatenating Surface 1, 2, 3, 4, 5 arrays."
        )


#%% --------------------------------------------------------------------------
# Coordinate transformation
# ---------------------------------------------------------------------------

def planar_pretransform_matrix(
    swap_xy: bool = False,
    flip_x: bool = False,
    flip_y: bool = False,
) -> np.ndarray:
    """Matrix applied before the case-angle rotation."""
    matrix = np.eye(2, dtype=float)
    if swap_xy:
        matrix = np.array([[0.0, 1.0], [1.0, 0.0]]) @ matrix
    if flip_x:
        matrix = np.array([[-1.0, 0.0], [0.0, 1.0]]) @ matrix
    if flip_y:
        matrix = np.array([[1.0, 0.0], [0.0, -1.0]]) @ matrix
    return matrix


def planar_rotation_matrix(rotation_deg: float) -> np.ndarray:
    theta = np.deg2rad(float(rotation_deg))
    return np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=float,
    )


def total_planar_transform_matrix(
    rotation_deg: float,
    swap_xy: bool = False,
    flip_x: bool = False,
    flip_y: bool = False,
) -> np.ndarray:
    return planar_rotation_matrix(rotation_deg) @ planar_pretransform_matrix(
        swap_xy=swap_xy,
        flip_x=flip_x,
        flip_y=flip_y,
    )


def raw_surface_definition(surface_id: int) -> Tuple[str, np.ndarray, float]:
    """Return face label, raw-local outward unit normal, and plane offset d.

    The plane is n dot r = d in raw spreadsheet-local coordinates.
    """
    definitions = {
        1: ("roof", np.array([0.0, 0.0, 1.0]), BUILDING_H),
        2: ("local-x-minus", np.array([-1.0, 0.0, 0.0]), BUILDING_B / 2.0),
        3: ("local-x-plus", np.array([1.0, 0.0, 0.0]), BUILDING_B / 2.0),
        4: ("local-y-plus", np.array([0.0, 1.0, 0.0]), BUILDING_W / 2.0),
        5: ("local-y-minus", np.array([0.0, -1.0, 0.0]), BUILDING_W / 2.0),
    }
    if int(surface_id) not in definitions:
        raise ValueError(f"Unknown surface ID: {surface_id}")
    return definitions[int(surface_id)]


def transformed_surface_normal(surface_id: int, planar_matrix: np.ndarray) -> np.ndarray:
    _, normal_raw, _ = raw_surface_definition(surface_id)
    if int(surface_id) == 1:
        return normal_raw.copy()
    normal_xy = planar_matrix @ normal_raw[:2]
    normal = np.array([normal_xy[0], normal_xy[1], 0.0], dtype=float)
    normal /= np.linalg.norm(normal)
    return normal


def convert_units_xyz(df: pd.DataFrame, units: str = "mm") -> pd.DataFrame:
    out = df.copy()
    scale = _units_scale(units)
    out[["X_m", "Y_m", "Z_m"]] = out[["X", "Y", "Z"]].to_numpy(dtype=float) * scale
    return out


def transform_points_to_les(
    points: pd.DataFrame,
    origin_les: Sequence[float] = (2.5, 0.0, 0.0),
    input_units: str = "mm",
    swap_xy: bool = False,
    flip_x: bool = False,
    flip_y: bool = False,
    rotation_deg: float = 0.0,
) -> pd.DataFrame:
    """Transform raw model coordinates to LES coordinates."""
    df = convert_units_xyz(points, input_units)

    raw_xy = df[["X_m", "Y_m"]].to_numpy(dtype=float).T
    pre_matrix = planar_pretransform_matrix(swap_xy, flip_x, flip_y)
    rotation_matrix = planar_rotation_matrix(rotation_deg)
    total_matrix = rotation_matrix @ pre_matrix

    building_local_xy = pre_matrix @ raw_xy
    rotated_xy = total_matrix @ raw_xy

    x0, y0, z0 = map(float, origin_les)

    # Explicit coordinate columns make QA and later post-processing unambiguous.
    df["x_model_raw"] = raw_xy[0]
    df["y_model_raw"] = raw_xy[1]
    df["z_model_raw"] = df["Z_m"].to_numpy(dtype=float)
    df["x_building_local"] = building_local_xy[0]
    df["y_building_local"] = building_local_xy[1]
    df["x_rotated_relative"] = rotated_xy[0]
    df["y_rotated_relative"] = rotated_xy[1]
    df["x"] = x0 + rotated_xy[0]
    df["y"] = y0 + rotated_xy[1]
    df["z"] = z0 + df["Z_m"].to_numpy(dtype=float)
    df["case_angle_deg"] = float(rotation_deg)

    if "Surface" in df.columns:
        normals = np.vstack(
            [
                transformed_surface_normal(int(sid), total_matrix)
                for sid in df["Surface"].to_numpy(dtype=int)
            ]
        )
        df["normal_x"] = normals[:, 0]
        df["normal_y"] = normals[:, 1]
        df["normal_z"] = normals[:, 2]

    # Keep input coordinates with explicit names for traceability.
    df = df.rename(columns={"X": "X_input", "Y": "Y_input", "Z": "Z_input"})
    return df


def transform_taps_to_les(taps: pd.DataFrame) -> pd.DataFrame:
    return transform_points_to_les(
        taps,
        origin_les=LES_BUILDING_BOTTOM_CENTRE,
        input_units=INPUT_UNITS,
        swap_xy=SWAP_XY,
        flip_x=FLIP_X,
        flip_y=FLIP_Y,
        rotation_deg=CASE_ANGLE_DEG,
    )


def transform_model_corners_to_les(corners: pd.DataFrame) -> pd.DataFrame:
    corner_df = corners.copy()
    corner_df["corner_id"] = np.arange(1, len(corner_df) + 1, dtype=int)
    return transform_points_to_les(
        corner_df,
        origin_les=LES_BUILDING_BOTTOM_CENTRE,
        input_units=INPUT_UNITS,
        swap_xy=SWAP_XY,
        flip_x=FLIP_X,
        flip_y=FLIP_Y,
        rotation_deg=CASE_ANGLE_DEG,
    )


#%% --------------------------------------------------------------------------
# Coordinate and surface validation
# ---------------------------------------------------------------------------

def expected_surface_description() -> pd.DataFrame:
    total_matrix = total_planar_transform_matrix(
        CASE_ANGLE_DEG, SWAP_XY, FLIP_X, FLIP_Y
    )
    rows = []
    for sid in [1, 2, 3, 4, 5]:
        face, _, offset = raw_surface_definition(sid)
        normal = transformed_surface_normal(sid, total_matrix)
        rows.append(
            {
                "Surface": sid,
                "expected_face": face,
                "normal_x": normal[0],
                "normal_y": normal[1],
                "normal_z": normal[2],
                "plane_offset_from_rotation_axis_m": offset,
                "plane_equation": (
                    f"n dot ([x,y,z]-[{LES_BUILDING_BOTTOM_CENTRE[0]},"
                    f"{LES_BUILDING_BOTTOM_CENTRE[1]},"
                    f"{LES_BUILDING_BOTTOM_CENTRE[2]}]) = {offset}"
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_coordinate_ranges(
    taps_les: pd.DataFrame,
    corners_les: pd.DataFrame,
    tol: float = 1.0e-9,
) -> pd.DataFrame:
    required = ["x", "y", "z", "x_model_raw", "y_model_raw", "z_model_raw"]
    values = taps_les[required].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = taps_les.loc[~np.isfinite(values).all(axis=1)]
        raise ValueError(f"Non-finite transformed tap coordinates found:\n{bad}")

    max_abs_global = float(np.max(np.abs(taps_les[["x", "y", "z"]].to_numpy(dtype=float))))
    if max_abs_global > 100.0:
        raise ValueError(
            f"Coordinates look too large for model-scale metres: max abs={max_abs_global:.6g}."
        )

    z0 = float(LES_BUILDING_BOTTOM_CENTRE[2])
    zmin = float(taps_les["z"].min())
    zmax = float(taps_les["z"].max())
    if zmin < z0 - tol or zmax > z0 + BUILDING_H + tol:
        raise ValueError(
            f"Tap z range [{zmin:.9g}, {zmax:.9g}] is inconsistent with "
            f"[{z0:.9g}, {z0 + BUILDING_H:.9g}]."
        )

    # Rotation must preserve the plan centre and every corner radius.
    x0, y0, _ = map(float, LES_BUILDING_BOTTOM_CENTRE)
    plan_centre = corners_les[["x", "y"]].mean().to_numpy(dtype=float)
    if not np.allclose(plan_centre, [x0, y0], atol=ROTATION_TOLERANCE_M, rtol=0.0):
        raise ValueError(
            f"Rotated corner plan-centre {plan_centre} does not match [{x0}, {y0}]."
        )

    r_raw = np.hypot(
        corners_les["x_model_raw"].to_numpy(dtype=float),
        corners_les["y_model_raw"].to_numpy(dtype=float),
    )
    r_global = np.hypot(
        corners_les["x"].to_numpy(dtype=float) - x0,
        corners_les["y"].to_numpy(dtype=float) - y0,
    )
    if not np.allclose(r_raw, r_global, atol=ROTATION_TOLERANCE_M, rtol=0.0):
        raise ValueError("Plan rotation did not preserve corner distances from the centroid.")

    duplicate_coords = taps_les.duplicated(subset=["x", "y", "z"], keep=False)
    if duplicate_coords.any():
        dup = taps_les.loc[
            duplicate_coords,
            ["designsafe_id", "Tap", "Surface", "x", "y", "z"],
        ]
        raise ValueError(f"Duplicate transformed pressure-tap coordinates found:\n{dup}")

    return taps_les.groupby("Surface")[["x", "y", "z"]].agg(["min", "max", "count"])


def validate_surface_planes(
    taps_les: pd.DataFrame,
    tolerance_m: float = PLANE_TOLERANCE_M,
) -> pd.DataFrame:
    """Angle-aware check using the transformed plane equations."""
    x0, y0, z0 = map(float, LES_BUILDING_BOTTOM_CENTRE)
    centre = np.array([x0, y0, z0], dtype=float)
    position_rel = taps_les[["x", "y", "z"]].to_numpy(dtype=float) - centre

    rows = []
    for sid in [1, 2, 3, 4, 5]:
        face, _, expected_offset = raw_surface_definition(sid)
        g = taps_les[taps_les["Surface"] == sid]
        if g.empty:
            rows.append(
                {
                    "Surface": sid,
                    "expected_face": face,
                    "n_taps": 0,
                    "max_abs_plane_error_m": np.nan,
                    "mean_plane_error_m": np.nan,
                    "status": "CHECK",
                }
            )
            continue

        idx = g.index.to_numpy(dtype=int)
        normals = g[["normal_x", "normal_y", "normal_z"]].to_numpy(dtype=float)
        # All taps on one surface have the same normal; use row-wise dot product
        # so the CSV remains auditable even if this changes later.
        signed_coordinate = np.einsum("ij,ij->i", position_rel[idx], normals)
        error = signed_coordinate - float(expected_offset)
        rows.append(
            {
                "Surface": sid,
                "expected_face": face,
                "n_taps": len(g),
                "normal_x": float(normals[0, 0]),
                "normal_y": float(normals[0, 1]),
                "normal_z": float(normals[0, 2]),
                "expected_plane_offset_m": float(expected_offset),
                "max_abs_plane_error_m": float(np.max(np.abs(error))),
                "mean_plane_error_m": float(np.mean(error)),
                "status": "OK" if np.max(np.abs(error)) <= tolerance_m else "CHECK",
            }
        )

    result = pd.DataFrame(rows)
    bad = result[result["status"] != "OK"]
    if not bad.empty:
        raise ValueError(
            "One or more pressure-tap surfaces are not on the rotated building planes:\n"
            + bad.to_string(index=False)
        )
    return result


#%% --------------------------------------------------------------------------
# Probe ordering, manifests, and OpenFOAM dictionaries
# ---------------------------------------------------------------------------

def order_probe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if PROBE_ORDER == "designsafe_id":
        keys = ["designsafe_id"]
    elif PROBE_ORDER == "spreadsheet":
        keys = ["excel_row"]
    elif PROBE_ORDER == "tap":
        keys = ["Tap"]
    else:
        raise ValueError(
            "PROBE_ORDER must be 'designsafe_id', 'spreadsheet', or 'tap'."
        )
    return df.sort_values(keys, kind="mergesort").reset_index(drop=True)


def build_probe_file_specs(taps_les: pd.DataFrame) -> List[Tuple[str, pd.DataFrame, str]]:
    """Return (file_name, ordered_rows, kind) definitions."""
    specs: List[Tuple[str, pd.DataFrame, str]] = []
    if WRITE_COMBINED_PROBE_FILE:
        specs.append(
            (
                COMBINED_PROBE_FILE_NAME,
                order_probe_dataframe(taps_les.copy()),
                "combined",
            )
        )
    if WRITE_PER_SURFACE_PROBE_FILES:
        for sid in sorted(taps_les["Surface"].unique()):
            group = taps_les[taps_les["Surface"] == sid].copy()
            specs.append(
                (f"{PROBE_FILE_PREFIX}{int(sid)}", order_probe_dataframe(group), "surface")
            )
    return specs


def make_probe_manifest(file_name: str, ordered_df: pd.DataFrame, file_kind: str) -> pd.DataFrame:
    manifest = ordered_df.copy().reset_index(drop=True)
    manifest.insert(0, "probe_file", file_name)
    manifest.insert(1, "probe_file_kind", file_kind)
    manifest.insert(2, "openfoam_probe_index_zero_based", np.arange(len(manifest), dtype=int))
    manifest.insert(3, "pressure_value_index_zero_based", np.arange(len(manifest), dtype=int))
    manifest.insert(4, "text_column_one_based_including_time", np.arange(2, len(manifest) + 2, dtype=int))
    manifest["canonical_designsafe_sort_position_one_based"] = manifest["designsafe_id"]
    cols = [
        "probe_file",
        "probe_file_kind",
        "openfoam_probe_index_zero_based",
        "pressure_value_index_zero_based",
        "text_column_one_based_including_time",
        "designsafe_id",
        "designsafe_array_index_zero_based",
        "canonical_designsafe_sort_position_one_based",
        "Tap",
        "Surface",
        "Module",
        "Channel",
        "excel_row",
        "spreadsheet_valid_order_zero_based",
        "X_input",
        "Y_input",
        "Z_input",
        "x_model_raw",
        "y_model_raw",
        "z_model_raw",
        "x",
        "y",
        "z",
        "normal_x",
        "normal_y",
        "normal_z",
        "case_angle_deg",
    ]
    return manifest[[c for c in cols if c in manifest.columns]]


def write_openfoam_probe_file_local(
    output_path: str | os.PathLike,
    coords_df: pd.DataFrame,
    patch_name: str = "building",
    field_name: str = "p",
    write_control: Optional[str] = None,
    write_interval=None,
) -> None:
    """Write a deterministic patchProbes dictionary in the given row order."""
    ordered = coords_df.reset_index(drop=True).copy()
    write_control, write_interval = _normalise_probe_write_settings(
        write_control, write_interval
    )
    write_interval_text = _format_openfoam_scalar(write_interval)

    lines: List[str] = []
    lines.append("/*--------------------------------*- C++ -*----------------------------------*\\")
    lines.append("| NHERI pressure taps generated in deterministic DesignSafe order            |")
    lines.append("\\*---------------------------------------------------------------------------*/")
    lines.append("")
    lines.append(f"// Building case angle: {CASE_ANGLE_DEG:.12g} deg")
    lines.append("// Positive angle is anticlockwise in plan, viewed from +z.")
    lines.append(f"// Probe ordering key: {PROBE_ORDER}")
    lines.append("// OpenFOAM Probe index is zero-based in the output header.")
    lines.append("")
    lines.append('#includeEtc "caseDicts/postProcessing/probes/probes.cfg"')
    lines.append("")
    lines.append("type            patchProbes;")
    lines.append('libs            ("libsampling.so");')
    lines.append(f"writeControl    {write_control};")
    lines.append(f"writeInterval   {write_interval_text};")
    lines.append(f"patch           {patch_name};")
    lines.append("")
    lines.append("fields")
    lines.append("(")
    lines.append(f"    {field_name}")
    lines.append(");")
    lines.append("")
    lines.append("probeLocations")
    lines.append("(")

    for probe_index, row in ordered.iterrows():
        vector = f"({row['x']:.12g} {row['y']:.12g} {row['z']:.12g})"
        if ADD_ID_COMMENTS_TO_PROBE_FILES:
            comment = (
                f" // Probe {probe_index}; DesignSafeID {int(row['designsafe_id'])}; "
                f"Tap {int(row['Tap'])}; Surface {int(row['Surface'])}"
            )
        else:
            comment = ""
        lines.append(f"    {vector}{comment}")

    lines.append(");")
    lines.append("")
    lines.append("// ************************************************************************* //")
    lines.append("")
    Path(output_path).write_text("\n".join(lines), encoding="utf-8", newline="\n")


def parse_written_probe_locations(path: str | os.PathLike) -> np.ndarray:
    """Parse the vectors written inside probeLocations for round-trip QA."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    block_match = re.search(r"probeLocations\s*\((.*?)\);", text, flags=re.S)
    if not block_match:
        raise ValueError(f"Could not find probeLocations block in {path}")
    vector_pattern = re.compile(
        r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)"
    )
    vectors = vector_pattern.findall(block_match.group(1))
    return np.asarray([[float(a), float(b), float(c)] for a, b, c in vectors], dtype=float)


def _try_write_with_windlespy(
    output_path: str,
    coords_df: pd.DataFrame,
    output_dir: str,
    file_name: str,
    write_control: str,
    write_interval,
) -> bool:
    """Compatibility path; returns True only when the writer is safely usable."""
    if not USE_WINDLESPY_WRITER_IF_AVAILABLE:
        return False
    try:
        import inspect
        import windlespy as LES

        writer = getattr(getattr(LES, "_caseFiles", object()), "write_surf_presure_probes", None)
        if writer is None:
            return False
        supported = set(inspect.signature(writer).parameters)
        if not {"write_control", "write_interval"}.issubset(supported):
            return False

        tmp_case = os.path.join(output_dir, "_tmp_probe_write_case")
        tmp_system = ensure_dir(os.path.join(tmp_case, "system"))
        try:
            writer(
                FIELD_NAME,
                coords_df,
                PATCH_NAME,
                tmp_case,
                file_name,
                write_control=write_control,
                write_interval=write_interval,
            )
            shutil.copyfile(os.path.join(tmp_system, file_name), output_path)
        finally:
            shutil.rmtree(tmp_case, ignore_errors=True)
        return True
    except Exception as exc:
        print(f"windlespy probe writer unavailable; using local writer. Reason: {exc}")
        return False


def write_probe_files_and_manifests(
    taps_les: pd.DataFrame,
    output_dir: str,
    case_path: Optional[str] = None,
    write_to_case_system: bool = False,
) -> Tuple[List[str], pd.DataFrame]:
    probe_out_dir = ensure_dir(os.path.join(output_dir, "probe_files_for_system"))
    manifest_dir = ensure_dir(os.path.join(output_dir, "probe_manifests"))
    write_control, write_interval = _normalise_probe_write_settings()

    all_manifests: List[pd.DataFrame] = []
    written: List[str] = []

    for file_name, ordered_df, file_kind in build_probe_file_specs(taps_les):
        output_path = os.path.join(probe_out_dir, file_name)
        used_windlespy = _try_write_with_windlespy(
            output_path,
            ordered_df,
            output_dir,
            file_name,
            write_control,
            write_interval,
        )
        if not used_windlespy:
            write_openfoam_probe_file_local(
                output_path,
                ordered_df,
                patch_name=PATCH_NAME,
                field_name=FIELD_NAME,
                write_control=write_control,
                write_interval=write_interval,
            )

        # Round-trip check: the dictionary must contain exactly the requested
        # coordinates in exactly the requested order.
        written_xyz = parse_written_probe_locations(output_path)
        expected_xyz = ordered_df[["x", "y", "z"]].to_numpy(dtype=float)
        if written_xyz.shape != expected_xyz.shape or not np.allclose(
            written_xyz, expected_xyz, atol=2.0e-11, rtol=0.0
        ):
            raise RuntimeError(
                f"Probe dictionary order/coordinates failed round-trip QA: {output_path}"
            )

        manifest = make_probe_manifest(file_name, ordered_df, file_kind)
        manifest.to_csv(
            os.path.join(manifest_dir, f"{file_name}_manifest.csv"),
            index=False,
            float_format="%.12e",
        )
        all_manifests.append(manifest)
        written.append(output_path)

        if write_to_case_system:
            if case_path is None:
                raise ValueError("case_path is required when write_to_case_system=True")
            system_dir = ensure_dir(os.path.join(case_path, "system"))
            destination = os.path.join(system_dir, file_name)
            if BACKUP_EXISTING_SYSTEM_PROBES and os.path.exists(destination):
                backup = destination + f".bak_before_{angle_tag(CASE_ANGLE_DEG)}"
                if not os.path.exists(backup):
                    shutil.copyfile(destination, backup)
            shutil.copyfile(output_path, destination)

    master_manifest = pd.concat(all_manifests, ignore_index=True)
    master_manifest.to_csv(
        os.path.join(manifest_dir, "all_probe_files_master_manifest.csv"),
        index=False,
        float_format="%.12e",
    )

    # Mapping needed if the legacy per-surface files are used.  Sorting this
    # table by designsafe_id reconstructs the experimental .mat column order.
    per_surface = master_manifest[master_manifest["probe_file_kind"] == "surface"].copy()
    if not per_surface.empty:
        reassembly = per_surface.sort_values("designsafe_id", kind="mergesort").reset_index(drop=True)
        reassembly.insert(0, "reassembled_designsafe_index_zero_based", np.arange(len(reassembly), dtype=int))
        reassembly.to_csv(
            os.path.join(manifest_dir, "per_surface_reassembly_to_designsafe_order.csv"),
            index=False,
            float_format="%.12e",
        )

    combined = master_manifest[master_manifest["probe_file_kind"] == "combined"].copy()
    if not combined.empty and PROBE_ORDER == "designsafe_id":
        expected = np.arange(1, len(combined) + 1, dtype=int)
        actual = combined["designsafe_id"].to_numpy(dtype=int)
        if not np.array_equal(expected, actual):
            raise RuntimeError("Combined probe file is not in exact DesignSafe order 1..N.")

    return written, master_manifest


#%% --------------------------------------------------------------------------
# QA plots
# ---------------------------------------------------------------------------

def _plot_building_edges(ax, corners_les: pd.DataFrame, color="k", linewidth=1.0) -> None:
    pts = corners_les[["x", "y", "z"]].to_numpy(dtype=float)
    raw = corners_les[["x_model_raw", "y_model_raw", "z_model_raw"]].to_numpy(dtype=float)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            diff = np.abs(raw[i] - raw[j])
            if np.count_nonzero(diff > 1.0e-12) == 1:
                ax.plot(
                    [pts[i, 0], pts[j, 0]],
                    [pts[i, 1], pts[j, 1]],
                    [pts[i, 2], pts[j, 2]],
                    color=color,
                    linewidth=linewidth,
                )


def make_qa_plots(
    taps_les: pd.DataFrame,
    corners_les: pd.DataFrame,
    plane_check: pd.DataFrame,
    master_manifest: Optional[pd.DataFrame],
    output_dir: str,
) -> None:
    fig_dir = ensure_dir(os.path.join(output_dir, "figures"))
    x0, y0, z0 = map(float, LES_BUILDING_BOTTOM_CENTRE)

    # 1. Rotated 3D geometry.
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for sid, group in taps_les.groupby("Surface", sort=True):
        ax.scatter(group["x"], group["y"], group["z"], s=13, label=f"Surface {int(sid)}")
    _plot_building_edges(ax, corners_les)
    ax.scatter([x0], [y0], [z0], s=80, marker="x", label="rotation axis at base")
    ax.set_xlabel("global x [m]")
    ax.set_ylabel("global y [m]")
    ax.set_zlabel("global z [m]")
    ax.set_title(f"NHERI pressure taps: building rotated {CASE_ANGLE_DEG:g}° CCW")
    ax.legend(loc="best")
    fig.savefig(os.path.join(fig_dir, "01_taps_3d_rotated_global.png"), dpi=250, bbox_inches="tight")
    plt.close(fig)

    # 2. Global top view with local building axes.
    fig, ax = plt.subplots(figsize=(8, 8))
    for sid, group in taps_les.groupby("Surface", sort=True):
        ax.scatter(group["x"], group["y"], s=14, label=f"Surface {int(sid)}")
    # Draw top perimeter from top corners.
    top = corners_les[np.isclose(corners_les["z_model_raw"], BUILDING_H)].copy()
    if len(top) == 4:
        centre = np.array([x0, y0])
        angles = np.arctan2(top["y"].to_numpy() - y0, top["x"].to_numpy() - x0)
        top = top.iloc[np.argsort(angles)]
        closed = np.vstack([top[["x", "y"]].to_numpy(), top[["x", "y"]].to_numpy()[0]])
        ax.plot(closed[:, 0], closed[:, 1], "k-", lw=1.5)
    transform = total_planar_transform_matrix(CASE_ANGLE_DEG, SWAP_XY, FLIP_X, FLIP_Y)
    ex = transform @ np.array([1.0, 0.0])
    ey = transform @ np.array([0.0, 1.0])
    axis_len = 0.18
    ax.arrow(x0, y0, axis_len * ex[0], axis_len * ex[1], width=0.0015, length_includes_head=True)
    ax.text(x0 + axis_len * ex[0], y0 + axis_len * ex[1], " local +X")
    ax.arrow(x0, y0, axis_len * ey[0], axis_len * ey[1], width=0.0015, length_includes_head=True)
    ax.text(x0 + axis_len * ey[0], y0 + axis_len * ey[1], " local +Y")
    ax.scatter([x0], [y0], marker="x", s=80, label="rotation centre")
    ax.set_xlabel("global x [m]")
    ax.set_ylabel("global y [m]")
    ax.set_title(f"Plan rotation QA: {CASE_ANGLE_DEG:g}° anticlockwise")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(os.path.join(fig_dir, "02_top_view_rotated_global.png"), dpi=250, bbox_inches="tight")
    plt.close(fig)

    # 3. Model-local per-surface layout.  This remains directly comparable with
    # the spreadsheet irrespective of case angle.
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.2), constrained_layout=True)
    scatter = None
    for ax, sid in zip(axes, [1, 2, 3, 4, 5]):
        group = taps_les[taps_les["Surface"] == sid]
        if sid == 1:
            xx, yy = group["x_model_raw"], group["y_model_raw"]
            xlabel, ylabel = "model X [m]", "model Y [m]"
        elif sid in (2, 3):
            xx, yy = group["y_model_raw"], group["z_model_raw"]
            xlabel, ylabel = "model Y [m]", "model Z [m]"
        else:
            xx, yy = group["x_model_raw"], group["z_model_raw"]
            xlabel, ylabel = "model X [m]", "model Z [m]"
        scatter = ax.scatter(xx, yy, c=group["designsafe_id"], s=18)
        ax.set_title(f"Surface {sid}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes, shrink=0.8)
        cbar.set_label("DesignSafe ID")
    fig.savefig(os.path.join(fig_dir, "03_per_surface_model_local_layout.png"), dpi=250, bbox_inches="tight")
    plt.close(fig)

    # 4. Rotated plane-alignment residuals.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(plane_check["Surface"].astype(str), plane_check["max_abs_plane_error_m"])
    ax.axhline(PLANE_TOLERANCE_M, ls="--", lw=1.0, label="tolerance")
    ax.set_xlabel("Surface")
    ax.set_ylabel("max rotated-plane error [m]")
    ax.set_title("Angle-aware pressure-tap surface alignment")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.savefig(os.path.join(fig_dir, "04_rotated_surface_plane_error.png"), dpi=250, bbox_inches="tight")
    plt.close(fig)

    # 5. Probe index versus DesignSafe ID, exposing the surface-1 discontinuity.
    if master_manifest is not None and not master_manifest.empty:
        files = list(master_manifest["probe_file"].drop_duplicates())
        ncols = 2
        nrows = int(np.ceil(len(files) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows), squeeze=False)
        for ax, file_name in zip(axes.ravel(), files):
            group = master_manifest[master_manifest["probe_file"] == file_name]
            ax.plot(
                group["openfoam_probe_index_zero_based"],
                group["designsafe_id"],
                marker=".",
                ms=3,
                lw=0.8,
            )
            ax.set_title(file_name)
            ax.set_xlabel("OpenFOAM Probe index [zero-based]")
            ax.set_ylabel("DesignSafe ID")
            ax.grid(True, alpha=0.3)
        for ax in axes.ravel()[len(files):]:
            ax.axis("off")
        fig.suptitle("Exact probe-file ordering used for pressure post-processing")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "05_probe_index_to_designsafe_id.png"), dpi=250, bbox_inches="tight")
        plt.close(fig)


#%% --------------------------------------------------------------------------
# Optional mesh validation
# ---------------------------------------------------------------------------

def validate_against_mesh_if_requested(
    taps_les: pd.DataFrame,
    output_dir: str,
) -> Optional[pd.DataFrame]:
    if not VALIDATE_AGAINST_MESH:
        return None
    try:
        import pyvista as pv
        from scipy.spatial import cKDTree
    except Exception as exc:
        warnings.warn(f"Mesh validation requested but dependencies are unavailable: {exc}")
        return None

    if FOAM_FILE is None:
        warnings.warn("VALIDATE_AGAINST_MESH=True but FOAM_FILE=None; skipping.")
        return None

    mesh = pv.read(FOAM_FILE)
    try:
        patch = mesh["boundary"][PATCH_NAME]
    except Exception as exc:
        warnings.warn(f"Could not access patch '{PATCH_NAME}' in mesh: {exc}")
        return None

    centres = patch.cell_centers().points
    tree = cKDTree(centres)
    xyz = taps_les[["x", "y", "z"]].to_numpy(dtype=float)
    distance, cell_index = tree.query(xyz)

    result = taps_les.copy()
    result["nearest_patch_cell_centre_distance_m"] = distance
    result["nearest_patch_cell_id"] = cell_index
    result.to_csv(
        os.path.join(output_dir, "mesh_nearest_patch_validation.csv"),
        index=False,
        float_format="%.12e",
    )
    print("\nNearest building-patch cell-centre distance summary:")
    print(result["nearest_patch_cell_centre_distance_m"].describe())
    if np.max(distance) > MAX_NEAREST_PATCH_DISTANCE:
        warnings.warn(
            f"Some taps are farther than {MAX_NEAREST_PATCH_DISTANCE:g} m from the nearest "
            f"'{PATCH_NAME}' patch cell centre.  Check the angle and mesh."
        )
    return result


#%% --------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    output_dir = ensure_dir(resolved_output_dir())

    print("\nReading NHERI pressure-tap layout...")
    taps_raw, corners_raw, excluded_rows = read_tap_layout(
        TAP_LAYOUT_XLSX,
        keep_only_designsafe_valid=KEEP_ONLY_DESIGNSAFE_VALID_TAPS,
    )
    print_ordering_summary(taps_raw, excluded_rows)

    order_qa = workbook_order_qa(taps_raw)
    order_qa.to_csv(
        os.path.join(output_dir, "spreadsheet_order_vs_designsafe_order.csv"),
        index=False,
    )
    excluded_rows.to_csv(
        os.path.join(output_dir, "excluded_spreadsheet_rows.csv"),
        index=False,
    )

    print(f"\nRotating taps {CASE_ANGLE_DEG:g}° anticlockwise about {LES_BUILDING_BOTTOM_CENTRE}...")
    taps_les = transform_taps_to_les(taps_raw)
    corners_les = transform_model_corners_to_les(corners_raw)

    coordinate_summary = validate_coordinate_ranges(taps_les, corners_les)
    plane_check = validate_surface_planes(taps_les)

    print("\nTransformed coordinate summary by surface:")
    print(coordinate_summary)
    print("\nAngle-aware surface-plane check:")
    print(plane_check.to_string(index=False))

    coordinate_csv = os.path.join(
        output_dir,
        f"nheri_pressure_taps_les_coordinates_{angle_tag(CASE_ANGLE_DEG)}.csv",
    )
    taps_les.to_csv(coordinate_csv, index=False, float_format="%.12e")
    corners_les.to_csv(
        os.path.join(output_dir, f"rotated_model_corners_{angle_tag(CASE_ANGLE_DEG)}.csv"),
        index=False,
        float_format="%.12e",
    )
    coordinate_summary.to_csv(
        os.path.join(output_dir, "coordinate_summary_by_surface.csv")
    )
    plane_check.to_csv(
        os.path.join(output_dir, "rotated_surface_plane_check.csv"),
        index=False,
        float_format="%.12e",
    )
    expected_surface_description().to_csv(
        os.path.join(output_dir, "rotated_surface_plane_definitions.csv"),
        index=False,
        float_format="%.12e",
    )

    master_manifest = None
    if WRITE_PROBE_FILES:
        print("\nWriting order-safe OpenFOAM patchProbes files...")
        written, master_manifest = write_probe_files_and_manifests(
            taps_les,
            output_dir,
            case_path=CASE_PATH,
            write_to_case_system=WRITE_TO_CASE_SYSTEM,
        )
        for path in written:
            print(f"  {path}")
        if WRITE_TO_CASE_SYSTEM:
            print("Probe files were copied to the case system directory.")
        else:
            print(
                "Probe files were written to the QA folder only.  Review the QA plots/manifests "
                "before setting WRITE_TO_CASE_SYSTEM=True."
            )

    print("\nCreating QA plots...")
    make_qa_plots(taps_les, corners_les, plane_check, master_manifest, output_dir)
    validate_against_mesh_if_requested(taps_les, output_dir)

    print("\nDone.")
    print(f"Case angle: {CASE_ANGLE_DEG:g}° anticlockwise")
    print(f"QA output folder:\n  {output_dir}")
    print(f"Canonical coordinate table:\n  {coordinate_csv}")
    if WRITE_COMBINED_PROBE_FILE:
        print(
            f"Recommended exact-global-order probe dictionary:\n  "
            f"{os.path.join(output_dir, 'probe_files_for_system', COMBINED_PROBE_FILE_NAME)}"
        )
    print(
        "For legacy per-surface outputs, use probe_manifests/"
        "per_surface_reassembly_to_designsafe_order.csv when rebuilding the 510-column array."
    )


if __name__ == "__main__":
    main()
