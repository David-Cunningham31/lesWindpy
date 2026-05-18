# -*- coding: utf-8 -*-
"""
Convert NHERI pressure-tap coordinates to LES-domain coordinates and write/QA probe files.

Purpose
-------
The NHERI tap-layout spreadsheet is in model-scale millimetres and uses the bottom
centre of the building plan as the coordinate reference point:

    model corners: X = +/-150 mm, Y = +/-300 mm, Z = 0..500 mm

This script converts those taps to your LES/OpenFOAM coordinate system using your
building bottom-centre reference point:

    LES building bottom centre = (2.5, 0.0, 0.0) m

Default transform:
    x_LES = 2.5 + X_mm / 1000
    y_LES = 0.0 + Y_mm / 1000
    z_LES = 0.0 + Z_mm / 1000

It also creates QA plots so you can verify that:
    - Surface 1 lies on the roof, z = H.
    - Surface 2 lies on x = x0 - B/2.
    - Surface 3 lies on x = x0 + B/2.
    - Surface 4 lies on y = y0 + W/2.
    - Surface 5 lies on y = y0 - W/2.
    - All coordinates are in metres and sit on the expected building box.

By default, probe files are written to:
    <case>/postProcessing/pressureTapCoordinateQA/probe_files_for_system

Set WRITE_TO_CASE_SYSTEM = True only after checking the QA plots.

Author: David Cunningham workflow, generated helper script
"""

import os
import sys
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%% --------------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------------

CASE_PATH = r"C:\Users\david\OneDrive\Documents\PhD\Year 1\NHERI LES Case\OpenFOAM Cases\Building Case"

TAP_LAYOUT_XLSX = r"C:\Users\david\OneDrive\Documents\PhD\Year 1\Wind Tunnel Test Data\NHERI BLWT Tall Building\Pressure Taps\Tap Layout - FSU - Mid-Rise-Model.xlsx"

# LES-domain reference point: bottom centre of building plan.
LES_BUILDING_BOTTOM_CENTRE = (2.5, 0.0, 0.0)

# NHERI model dimensions, in metres after conversion.
BUILDING_B = 0.30  # x extent if no swap_xy
BUILDING_W = 0.60  # y extent if no swap_xy
BUILDING_H = 0.50

# Spreadsheet coordinate units.
INPUT_UNITS = "mm"  # "mm" or "m"

# Optional coordinate-system changes.
# Leave all False/0 for the provided spreadsheet if your LES building uses:
# x extent = 0.30 m, y extent = 0.60 m, z extent = 0.50 m.
SWAP_XY = False
FLIP_X = False
FLIP_Y = False
ROTATION_DEG = 0.0  # rotation about z after swap/flip, before translation

# Surface-probe file settings.
PATCH_NAME = "building"
FIELD_NAME = "p"
PROBE_FILE_PREFIX = "probesSurface"

# Probe write-frequency settings.
# Leave PROBE_WRITE_INTERVAL = None to write every time step, i.e. writeInterval 1.
# Set PROBE_WRITE_INTERVAL to an integer N to write every N time steps when
# PROBE_WRITE_CONTROL = "timeStep".
# You can also change PROBE_WRITE_CONTROL to "adjustableRunTime" and set
# PROBE_WRITE_INTERVAL to a physical time interval, if that is what you need.
PROBE_WRITE_CONTROL = "timeStep"
PROBE_WRITE_INTERVAL = None

# There is one tap with coordinates but DesignSafe ID="-".
# For Cp comparison against the .mat files, drop it so the layout contains 510 taps.
KEEP_ONLY_DESIGNSAFE_VALID_TAPS = True

# Output control.
OUTPUT_DIR = os.path.join(CASE_PATH, "postProcessing", "pressureTapCoordinateQA")
WRITE_PROBE_FILES = True

# Safety: write converted probes to OUTPUT_DIR by default.
# Set this True only after reviewing the QA plots.
WRITE_TO_CASE_SYSTEM = False
BACKUP_EXISTING_SYSTEM_PROBES = True

# Use windLespy's updated writer first. If the local installed copy is old/unavailable, use local fallback.
USE_WINDLESPY_WRITER_IF_AVAILABLE = True

# Optional mesh validation: requires pyvista and a readable .foam file.
# This is intentionally off by default because some Windows setups do not have pyvista.
VALIDATE_AGAINST_MESH = False
FOAM_FILE = None  # e.g. os.path.join(CASE_PATH, "case.foam")
MAX_NEAREST_PATCH_DISTANCE = 0.005  # m; choose based on wall-cell size


#%% --------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _numeric_series(s):
    return pd.to_numeric(s, errors="coerce")


def read_tap_layout(xlsx_path, keep_only_designsafe_valid=True):
    """Read and clean the NHERI tap layout."""
    taps = pd.read_excel(xlsx_path, sheet_name="Taps")
    corners = pd.read_excel(xlsx_path, sheet_name="Model Corners")

    required = ["Tap", "Surface", "X", "Y", "Z"]
    missing = [c for c in required if c not in taps.columns]
    if missing:
        raise ValueError(f"Tap-layout sheet is missing required columns: {missing}")

    taps = taps.copy()
    taps["Surface"] = _numeric_series(taps["Surface"])
    taps["X"] = _numeric_series(taps["X"])
    taps["Y"] = _numeric_series(taps["Y"])
    taps["Z"] = _numeric_series(taps["Z"])

    taps = taps.dropna(subset=["Surface", "X", "Y", "Z"]).copy()
    taps["Surface"] = taps["Surface"].astype(int)

    if "ID (.mat file) - DesignSafe" in taps.columns:
        taps["designsafe_id"] = _numeric_series(taps["ID (.mat file) - DesignSafe"])
    else:
        taps["designsafe_id"] = np.arange(1, len(taps) + 1, dtype=int)

    if keep_only_designsafe_valid:
        before = len(taps)
        taps = taps[np.isfinite(taps["designsafe_id"])].copy()
        after = len(taps)
        if after != before:
            print(f"Dropped {before - after} tap(s) with invalid DesignSafe ID.")

    taps["designsafe_id"] = taps["designsafe_id"].astype(int)

    # Stable ordering for comparison to .mat Cp columns.
    taps = taps.sort_values("designsafe_id").reset_index(drop=True)

    corners = corners.copy()
    for c in ["X", "Y", "Z"]:
        corners[c] = _numeric_series(corners[c])

    return taps, corners


def convert_units_xyz(df, units="mm"):
    out = df.copy()
    if units.lower() in ("mm", "millimetre", "millimetres", "millimeter", "millimeters"):
        scale = 1.0 / 1000.0
    elif units.lower() in ("m", "metre", "metres", "meter", "meters"):
        scale = 1.0
    else:
        raise ValueError("INPUT_UNITS must be 'mm' or 'm'.")
    out[["X_m", "Y_m", "Z_m"]] = out[["X", "Y", "Z"]].to_numpy(dtype=float) * scale
    return out


def transform_taps_to_les(
    taps,
    origin_les=(2.5, 0.0, 0.0),
    input_units="mm",
    swap_xy=False,
    flip_x=False,
    flip_y=False,
    rotation_deg=0.0,
):
    """Transform tap coordinates from NHERI model coordinates to LES coordinates."""
    df = convert_units_xyz(taps, input_units)

    x = df["X_m"].to_numpy(dtype=float)
    y = df["Y_m"].to_numpy(dtype=float)
    z = df["Z_m"].to_numpy(dtype=float)

    if swap_xy:
        x, y = y.copy(), x.copy()

    if flip_x:
        x = -x
    if flip_y:
        y = -y

    theta = np.deg2rad(float(rotation_deg))
    xr = x * np.cos(theta) - y * np.sin(theta)
    yr = x * np.sin(theta) + y * np.cos(theta)

    x0, y0, z0 = [float(v) for v in origin_les]

    df["x"] = x0 + xr
    df["y"] = y0 + yr
    df["z"] = z0 + z

    # Keep original coordinates with explicit names for traceability.
    df = df.rename(columns={"X": "X_input", "Y": "Y_input", "Z": "Z_input"})

    return df


def transform_model_corners_to_les(
    corners,
    origin_les=(2.5, 0.0, 0.0),
    input_units="mm",
    swap_xy=False,
    flip_x=False,
    flip_y=False,
    rotation_deg=0.0,
):
    c = corners.copy()
    c["Tap"] = np.arange(1, len(c) + 1)
    c["Surface"] = 0
    c["designsafe_id"] = np.arange(1, len(c) + 1)
    c = transform_taps_to_les(
        c,
        origin_les=origin_les,
        input_units=input_units,
        swap_xy=swap_xy,
        flip_x=flip_x,
        flip_y=flip_y,
        rotation_deg=rotation_deg,
    )
    return c


def validate_coordinate_ranges(taps_les, corners_les, H=0.5, tol=1e-6):
    """Sanity checks that catch mm/m and origin mistakes."""
    xyz = taps_les[["x", "y", "z"]].to_numpy(dtype=float)

    max_abs_global = float(np.nanmax(np.abs(xyz)))
    if max_abs_global > 100.0:
        raise ValueError(
            f"Coordinates look far too large for model-scale metres: max abs = {max_abs_global:.3g}. "
            "They may still be in mm."
        )

    zmin, zmax = float(np.nanmin(taps_les["z"])), float(np.nanmax(taps_les["z"]))
    z0 = float(LES_BUILDING_BOTTOM_CENTRE[2])
    if zmin < z0 - tol or zmax > z0 + H + tol:
        raise ValueError(
            f"Tap z range [{zmin:.6g}, {zmax:.6g}] is inconsistent with "
            f"LES base/H [{z0:.6g}, {z0 + H:.6g}]."
        )

    summary = taps_les.groupby("Surface")[["x", "y", "z"]].agg(["min", "max", "count"])
    return summary


def expected_surface_description():
    x0, y0, z0 = [float(v) for v in LES_BUILDING_BOTTOM_CENTRE]
    return pd.DataFrame(
        [
            {"Surface": 1, "expected_face": "roof", "expected_constant": "z", "expected_value_m": z0 + BUILDING_H},
            {"Surface": 2, "expected_face": "x-minus side", "expected_constant": "x", "expected_value_m": x0 - BUILDING_B / 2},
            {"Surface": 3, "expected_face": "x-plus side", "expected_constant": "x", "expected_value_m": x0 + BUILDING_B / 2},
            {"Surface": 4, "expected_face": "y-plus side", "expected_constant": "y", "expected_value_m": y0 + BUILDING_W / 2},
            {"Surface": 5, "expected_face": "y-minus side", "expected_constant": "y", "expected_value_m": y0 - BUILDING_W / 2},
        ]
    )


def validate_surface_constants(taps_les, tol=2e-5):
    """Check that each surface lies on the expected plane."""
    desc = expected_surface_description()
    rows = []
    for _, r in desc.iterrows():
        sid = int(r["Surface"])
        g = taps_les[taps_les["Surface"] == sid]
        coord = r["expected_constant"]
        expected = float(r["expected_value_m"])
        err = g[coord].to_numpy(dtype=float) - expected
        rows.append(
            {
                "Surface": sid,
                "expected_face": r["expected_face"],
                "expected_constant": coord,
                "expected_value_m": expected,
                "n_taps": len(g),
                "max_abs_plane_error_m": float(np.nanmax(np.abs(err))) if len(g) else np.nan,
                "mean_plane_error_m": float(np.nanmean(err)) if len(g) else np.nan,
                "status": "OK" if len(g) and np.nanmax(np.abs(err)) <= tol else "CHECK",
            }
        )
    return pd.DataFrame(rows)


def _normalise_probe_write_settings(write_control=None, write_interval=None):
    """Return OpenFOAM writeControl/writeInterval values for probe dictionaries."""
    if write_control is None:
        write_control = PROBE_WRITE_CONTROL

    if write_interval is None:
        write_interval = PROBE_WRITE_INTERVAL

    if write_interval is None:
        # Default requested behaviour: if no specific frequency is chosen, write every time step.
        write_interval = 1

    if isinstance(write_interval, (int, float)) and write_interval <= 0:
        raise ValueError("write_interval must be positive or None.")

    return str(write_control), write_interval


def _format_openfoam_scalar(value):
    """Format numeric values cleanly for OpenFOAM dictionaries."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.12g}"
    return str(value)


def write_openfoam_probe_file_local(
    output_path,
    coords_df,
    patch_name="building",
    field_name="p",
    write_control=None,
    write_interval=None,
):
    """Fallback local writer for one OpenFOAM patchProbes dictionary."""
    coords_df = coords_df.copy()
    coords = coords_df[["x", "y", "z"]].to_numpy(dtype=float)
    write_control, write_interval = _normalise_probe_write_settings(write_control, write_interval)
    write_interval = _format_openfoam_scalar(write_interval)

    lines = []
    lines.append("/*--------------------------------*- C++ -*----------------------------------*\\")
    lines.append("| Generated by convert_nheri_pressure_taps_to_les.py                         |")
    lines.append("\\*---------------------------------------------------------------------------*/")
    lines.append("")
    lines.append('#includeEtc "caseDicts/postProcessing/probes/probes.cfg"')
    lines.append("")
    lines.append("type            patchProbes;")
    lines.append('libs            ("libsampling.so");')
    lines.append(f"writeControl    {write_control};")
    lines.append(f"writeInterval   {write_interval};")
    lines.append(f"patch           {patch_name};")
    lines.append("")
    lines.append("fields")
    lines.append("(")
    lines.append(f"    {field_name}")
    lines.append(");")
    lines.append("")
    lines.append("probeLocations")
    lines.append("(")
    for x, y, z in coords:
        lines.append(f"    ({x:.10g} {y:.10g} {z:.10g})")
    lines.append(");")
    lines.append("")
    lines.append("// ************************************************************************* //")
    lines.append("")
    Path(output_path).write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_probe_files(
    taps_les,
    output_dir,
    case_path=None,
    write_to_case_system=False,
    write_control=None,
    write_interval=None,
):
    """Write one probe file per surface, using windLespy writer if available."""
    write_control, write_interval = _normalise_probe_write_settings(write_control, write_interval)
    probe_out_dir = ensure_dir(os.path.join(output_dir, "probe_files_for_system"))

    windlespy_writer_available = False
    LES = None
    if USE_WINDLESPY_WRITER_IF_AVAILABLE:
        try:
            cwd = os.path.dirname(os.path.abspath(__file__))
            candidate = os.path.abspath(os.path.join(cwd, "..", ".."))
            if os.path.isdir(os.path.join(candidate, "windlespy")) and candidate not in sys.path:
                sys.path.append(candidate)
            import inspect
            import windlespy as LES  # noqa: F401
            windlespy_writer_available = hasattr(LES, "_caseFiles") and hasattr(LES._caseFiles, "write_surf_presure_probes")
            if windlespy_writer_available:
                sig = inspect.signature(LES._caseFiles.write_surf_presure_probes)
                supported = set(sig.parameters)
                if not ({"write_control", "write_interval"} <= supported):
                    print(
                        "windLespy writer found, but it does not expose write_control/write_interval; "
                        "using local writer so the requested probe write frequency is honoured."
                    )
                    windlespy_writer_available = False
        except Exception as exc:
            print(f"windLespy writer not available; using local writer. Reason: {exc}")
            windlespy_writer_available = False

    written = []
    for sid in sorted(taps_les["Surface"].unique()):
        g = taps_les[taps_les["Surface"] == sid].copy()
        fname = f"{PROBE_FILE_PREFIX}{int(sid)}"
        local_path = os.path.join(probe_out_dir, fname)

        if windlespy_writer_available:
            # Use a temporary case whose system directory is probe_out_dir.
            # windLespy's writer expects a case path and writes into case/system/filename.
            tmp_case = os.path.join(output_dir, "_tmp_probe_write_case")
            tmp_system = ensure_dir(os.path.join(tmp_case, "system"))
            try:
                LES._caseFiles.write_surf_presure_probes(
                    FIELD_NAME,
                    g,
                    PATCH_NAME,
                    tmp_case,
                    fname,
                    write_control=write_control,
                    write_interval=write_interval,
                )
                shutil.copyfile(os.path.join(tmp_system, fname), local_path)
            finally:
                shutil.rmtree(tmp_case, ignore_errors=True)
        else:
            write_openfoam_probe_file_local(
                local_path,
                g,
                patch_name=PATCH_NAME,
                field_name=FIELD_NAME,
                write_control=write_control,
                write_interval=write_interval,
            )

        written.append(local_path)

        if write_to_case_system:
            if case_path is None:
                raise ValueError("case_path must be supplied when write_to_case_system=True.")
            system_dir = ensure_dir(os.path.join(case_path, "system"))
            dest = os.path.join(system_dir, fname)

            if BACKUP_EXISTING_SYSTEM_PROBES and os.path.exists(dest):
                backup = dest + ".bak_before_nheri_tap_unit_fix"
                if not os.path.exists(backup):
                    shutil.copyfile(dest, backup)

            shutil.copyfile(local_path, dest)

    return written


def _plot_building_edges(ax, corners_les, color="k", linewidth=1.0):
    """Draw rectangular prism edges from eight corners."""
    pts = corners_les[["x", "y", "z"]].to_numpy(dtype=float)

    # Connect pairs that differ in exactly one coordinate.
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            diff = np.abs(pts[i] - pts[j])
            nonzero = np.count_nonzero(diff > 1e-9)
            if nonzero == 1:
                ax.plot(
                    [pts[i, 0], pts[j, 0]],
                    [pts[i, 1], pts[j, 1]],
                    [pts[i, 2], pts[j, 2]],
                    color=color,
                    linewidth=linewidth,
                )


def make_qa_plots(taps_les, corners_les, output_dir):
    fig_dir = ensure_dir(os.path.join(output_dir, "figures"))

    # 3D plot.
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for sid, g in taps_les.groupby("Surface"):
        ax.scatter(g["x"], g["y"], g["z"], s=16, label=f"Surface {int(sid)}")
    _plot_building_edges(ax, corners_les)
    ax.scatter(
        [LES_BUILDING_BOTTOM_CENTRE[0]],
        [LES_BUILDING_BOTTOM_CENTRE[1]],
        [LES_BUILDING_BOTTOM_CENTRE[2]],
        s=80,
        marker="x",
        label="LES bottom centre",
    )
    ax.set_xlabel("x LES [m]")
    ax.set_ylabel("y LES [m]")
    ax.set_zlabel("z LES [m]")
    ax.set_title("NHERI pressure taps transformed to LES coordinates")
    ax.legend(loc="best")
    fig.savefig(os.path.join(fig_dir, "01_taps_3d_les_coordinates.png"), dpi=250, bbox_inches="tight")
    plt.close(fig)

    # Orthographic projections.
    projections = [
        ("top_xy", "x", "y", "Top view: x-y"),
        ("front_xz", "x", "z", "x-z view"),
        ("side_yz", "y", "z", "y-z view"),
    ]
    for name, xc, yc, title in projections:
        fig, ax = plt.subplots(figsize=(8, 7))
        for sid, g in taps_les.groupby("Surface"):
            ax.scatter(g[xc], g[yc], s=18, label=f"Surface {int(sid)}")
        ax.scatter(
            [LES_BUILDING_BOTTOM_CENTRE[0] if xc == "x" else LES_BUILDING_BOTTOM_CENTRE[1] if xc == "y" else LES_BUILDING_BOTTOM_CENTRE[2]],
            [LES_BUILDING_BOTTOM_CENTRE[0] if yc == "x" else LES_BUILDING_BOTTOM_CENTRE[1] if yc == "y" else LES_BUILDING_BOTTOM_CENTRE[2]],
            s=80,
            marker="x",
            label="LES bottom centre",
        )
        ax.set_xlabel(f"{xc} LES [m]")
        ax.set_ylabel(f"{yc} LES [m]")
        ax.set_title(title)
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.savefig(os.path.join(fig_dir, f"02_{name}.png"), dpi=250, bbox_inches="tight")
        plt.close(fig)

    # Per-surface local coordinates, useful for checking tap ordering and density.
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), constrained_layout=True)
    for ax, sid in zip(axes, [1, 2, 3, 4, 5]):
        g = taps_les[taps_les["Surface"] == sid]
        if sid == 1:
            xx, yy = g["x"], g["y"]
            xlabel, ylabel = "x [m]", "y [m]"
        elif sid in (2, 3):
            xx, yy = g["y"], g["z"]
            xlabel, ylabel = "y [m]", "z [m]"
        else:
            xx, yy = g["x"], g["z"]
            xlabel, ylabel = "x [m]", "z [m]"
        sc = ax.scatter(xx, yy, c=g["designsafe_id"], s=18)
        ax.set_title(f"Surface {sid}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=axes, shrink=0.75)
    cbar.set_label("DesignSafe tap ID")
    fig.savefig(os.path.join(fig_dir, "03_per_surface_local_layout.png"), dpi=250, bbox_inches="tight")
    plt.close(fig)

    # Expected face plane check plot.
    plane_df = validate_surface_constants(taps_les)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(plane_df["Surface"].astype(str), plane_df["max_abs_plane_error_m"])
    ax.set_xlabel("Surface")
    ax.set_ylabel("max |plane error| [m]")
    ax.set_title("Surface plane-alignment check")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(os.path.join(fig_dir, "04_surface_plane_alignment_error.png"), dpi=250, bbox_inches="tight")
    plt.close(fig)


def validate_against_mesh_if_requested(taps_les, output_dir):
    if not VALIDATE_AGAINST_MESH:
        return None
    try:
        import pyvista as pv
        from scipy.spatial import cKDTree
    except Exception as exc:
        warnings.warn(f"Mesh validation requested but dependencies are unavailable: {exc}")
        return None

    if FOAM_FILE is None:
        warnings.warn("VALIDATE_AGAINST_MESH=True but FOAM_FILE=None. Skipping mesh validation.")
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
    dist, idx = tree.query(xyz)

    out = taps_les.copy()
    out["nearest_patch_distance_m"] = dist
    out["nearest_patch_cell_id"] = idx
    out.to_csv(os.path.join(output_dir, "mesh_nearest_patch_validation.csv"), index=False)

    print("\nNearest patch distance summary:")
    print(out["nearest_patch_distance_m"].describe())

    if np.nanmax(dist) > MAX_NEAREST_PATCH_DISTANCE:
        warnings.warn(
            f"Some taps are farther than {MAX_NEAREST_PATCH_DISTANCE} m from patch '{PATCH_NAME}'. "
            "Check transform/orientation."
        )

    return out


#%% --------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ensure_dir(OUTPUT_DIR)

    print("\nReading NHERI pressure-tap layout...")
    taps_raw, corners_raw = read_tap_layout(
        TAP_LAYOUT_XLSX,
        keep_only_designsafe_valid=KEEP_ONLY_DESIGNSAFE_VALID_TAPS,
    )
    print(f"Read {len(taps_raw)} valid taps.")
    print("Raw tap coordinate range from spreadsheet:")
    print(taps_raw[["X", "Y", "Z"]].agg(["min", "max"]))

    print("\nTransforming taps to LES coordinates...")
    taps_les = transform_taps_to_les(
        taps_raw,
        origin_les=LES_BUILDING_BOTTOM_CENTRE,
        input_units=INPUT_UNITS,
        swap_xy=SWAP_XY,
        flip_x=FLIP_X,
        flip_y=FLIP_Y,
        rotation_deg=ROTATION_DEG,
    )
    corners_les = transform_model_corners_to_les(
        corners_raw,
        origin_les=LES_BUILDING_BOTTOM_CENTRE,
        input_units=INPUT_UNITS,
        swap_xy=SWAP_XY,
        flip_x=FLIP_X,
        flip_y=FLIP_Y,
        rotation_deg=ROTATION_DEG,
    )

    summary = validate_coordinate_ranges(taps_les, corners_les, H=BUILDING_H)
    plane_check = validate_surface_constants(taps_les)

    print("\nTransformed coordinate summary by surface:")
    print(summary)
    print("\nExpected surface-plane check:")
    print(plane_check.to_string(index=False))

    out_csv = os.path.join(OUTPUT_DIR, "nheri_pressure_taps_les_coordinates.csv")
    taps_les.to_csv(out_csv, index=False)

    summary.to_csv(os.path.join(OUTPUT_DIR, "nheri_pressure_taps_les_coordinate_summary_by_surface.csv"))
    plane_check.to_csv(os.path.join(OUTPUT_DIR, "nheri_pressure_taps_les_surface_plane_check.csv"), index=False)
    expected_surface_description().to_csv(os.path.join(OUTPUT_DIR, "expected_surface_mapping.csv"), index=False)

    print(f"\nWrote transformed tap CSV:\n  {out_csv}")

    print("\nCreating QA plots...")
    make_qa_plots(taps_les, corners_les, OUTPUT_DIR)

    validate_against_mesh_if_requested(taps_les, OUTPUT_DIR)

    if WRITE_PROBE_FILES:
        print("\nWriting OpenFOAM probe files...")
        _wc, _wi = _normalise_probe_write_settings(PROBE_WRITE_CONTROL, PROBE_WRITE_INTERVAL)
        print(f"Probe write settings: writeControl {_wc}; writeInterval {_format_openfoam_scalar(_wi)};")
        written = write_probe_files(
            taps_les,
            OUTPUT_DIR,
            case_path=CASE_PATH,
            write_to_case_system=WRITE_TO_CASE_SYSTEM,
            write_control=PROBE_WRITE_CONTROL,
            write_interval=PROBE_WRITE_INTERVAL,
        )
        for p in written:
            print(f"  {p}")

        if WRITE_TO_CASE_SYSTEM:
            print("\nProbe files were copied into the case system directory.")
        else:
            print(
                "\nProbe files were written to the QA output folder only. "
                "Review the plots before setting WRITE_TO_CASE_SYSTEM=True."
            )

    print("\nDone.")
    print(f"QA output folder:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
