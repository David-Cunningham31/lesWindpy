# -*- coding: utf-8 -*-
"""
Plot downstream DFSR/Wong calibration iteration profiles against the NHERI
wind-tunnel approach-flow data, using the Melaku-style figure layout.

Important notes
---------------
1. EXP points are the NHERI wind-tunnel measurements computed directly from
   the approach-flow .mat file. They are NOT the digitised Melaku points and
   they are NOT smoothed/extended.
2. Iteration files are read from:
       case_path/log/downstreamCalibration/iteration*/
   using newInletProfile as the DFSR/inlet profile and postCorrectionProfile
   as the downstream LES profile by default.
3. The expected calibration profile format is:
       z, U, Iu, Iv, Iw, Lu, Lv, Lw
   where Iu/Iv/Iw are turbulence intensities already, not variances.
4. The output uses the same axis limits/ticks for every iteration to make
   side-by-side review meaningful.
"""

import os
import re
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------------

case_path = r"C:\Users\david\OneDrive\Documents\PhD\Year 1\Spectral Calibration Method\empty_domain_test_case\regularDFSR"
approach_flow_data = r"C:\Users\david\OneDrive\Documents\PhD\Year 1\Wind Tunnel Test Data\NHERI BLWT Tall Building\Approach Flow\Approach Flow - EH160 - Marine Spires - 1200 RPM - 091721_1028.mat"

# windLespy import. This script is intended to live in windlespy/_recipes.
cwd = os.path.dirname(os.path.abspath(__file__))
windlespy_path = os.path.abspath(os.path.join(cwd, "..", ".."))
if windlespy_path not in sys.path:
    sys.path.append(windlespy_path)
import windlespy as LES
try:
    sys.path.remove(windlespy_path)
except ValueError:
    pass

# Iteration file names. These are the usual downstreamCalibration outputs.
DFSR_PROFILE_FILE = "newInletProfile"          # still read only for optional QA; not plotted
LES_PROFILE_FILE = "postCorrectionProfile"    # solid red curve

# Output folder under log/downstreamCalibration.
OUTPUT_SUBDIR = "profile_review_nheri_melaku_style_les_only_melaku_aspect"

# Use raw NHERI measured points only. Do not extend or smooth.
USE_RAW_NHERI_ONLY = True

# Optional CSV output for checking exactly what was plotted.
WRITE_QA_CSV = True

# Figure styling.
FIG_DPI = 300
SAVE_PDF = False
DARK_STYLE = False

# Animation output from the per-iteration profile PNGs.
CREATE_ANIMATION = True
ANIMATION_FPS = 2
ANIMATION_FILENAME = "profile_iteration_progression.gif"

# Marker/line styling similar to the paper.
EXP_MARKER_SIZE = 26
EXP_MARKER_LINEWIDTH = 1.1
DFSR_LINEWIDTH = 1.6
LES_LINEWIDTH = 1.9

# The paper figure has the Reynolds shear stress panel too, but you asked to
# ignore that and plot the 7 profiles only.
PLOT_KEYS = [
    "U_over_UH",
    "Iu",
    "Iv",
    "Iw",
    "Lu_over_H",
    "Lv_over_H",
    "Lw_over_H",
]

# Axis limits/ticks matching the Melaku figure style for these 7 panels.
AXIS_CONFIG = {
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
    "Lu_over_H": {
        "xlim": (0.0, 4.0),
        "xticks": [0.0, 1.0, 2.0, 3.0, 4.0],
        "xlabel": r"$L_u/H$",
        "panel": "(e)",
    },
    "Lv_over_H": {
        "xlim": (0.0, 2.0),
        "xticks": [0.0, 0.5, 1.0, 1.5, 2.0],
        "xlabel": r"$L_v/H$",
        "panel": "(f)",
    },
    "Lw_over_H": {
        "xlim": (0.0, 2.0),
        "xticks": [0.0, 0.5, 1.0, 1.5, 2.0],
        "xlabel": r"$L_w/H$",
        "panel": "(g)",
    },
}

Y_LIM = (0.0, 3.0)
Y_TICKS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def natural_iteration_key(path):
    m = re.search(r"iteration(\d+)$", os.path.basename(str(path)))
    return int(m.group(1)) if m else 10**9


def get_iteration_dirs(calib_dir):
    dirs = []
    for name in os.listdir(calib_dir):
        p = os.path.join(calib_dir, name)
        if os.path.isdir(p) and re.fullmatch(r"iteration\d+", name):
            dirs.append(p)
    return sorted(dirs, key=natural_iteration_key)


def read_numeric_profile_file(path):
    """Read an OpenFOAM/windLespy-style profile file as a numeric array.

    This is deliberately tolerant: it ignores comments, parentheses, and
    non-numeric tokens. It expects one data row per line once cleaned.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("#"):
                continue
            line = line.replace("(", " ").replace(")", " ").replace(";", " ")
            parts = line.split()
            vals = []
            for part in parts:
                try:
                    vals.append(float(part))
                except ValueError:
                    pass
            if len(vals) >= 2:
                rows.append(vals)

    if not rows:
        raise ValueError(f"No numeric data found in {path}")

    # Keep only rows with the modal numeric width. This avoids accidentally
    # parsing OpenFOAM dictionary metadata as data.
    widths = pd.Series([len(r) for r in rows])
    modal_width = int(widths.mode().iloc[0])
    data_rows = [r for r in rows if len(r) == modal_width]
    arr = np.asarray(data_rows, dtype=float)

    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError(f"Could not parse a valid 2D profile from {path}; got shape {arr.shape}")

    return arr


def column_lookup(df, candidates):
    """Return the first matching column from a set of possible names."""
    norm = {str(c).strip().lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower().replace(" ", "").replace("_", "")
        if key in norm:
            return norm[key]
    return None


def safe_interp(x, y, x0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(m) < 2:
        return np.nan
    order = np.argsort(x[m])
    return float(np.interp(x0, x[m][order], y[m][order]))


def profile_array_to_quantities(arr, U_H_ref, H, label="profile"):
    """Convert calibration profile array into plotted quantities.

    Expected windLespy calibration-profile format:
        z, U, Iu, Iv, Iw, Lu, Lv, Lw

    Important: the iteration profile files written by the Wong/downstream
    calibration recipes store turbulence intensities directly. They do not
    store Reynolds-stress variances. Therefore these columns must be plotted
    directly. Do not apply sqrt(.)/U here.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{label}: profile array must be 2D, got {arr.shape}")

    # Some old files may have an index column before z. The user confirmed z is
    # always first for the current case, so we keep that as the default and only
    # warn if the first column does not look height-like.
    if arr.shape[1] < 8:
        raise ValueError(
            f"{label}: expected at least 8 columns: z, U, Iu, Iv, Iw, Lu, Lv, Lw. "
            f"Got shape {arr.shape}."
        )

    z = arr[:, 0]
    U = arr[:, 1]
    Iu = arr[:, 2]
    Iv = arr[:, 3]
    Iw = arr[:, 4]
    Lu = arr[:, 5]
    Lv = arr[:, 6]
    Lw = arr[:, 7]

    m = np.isfinite(z) & np.isfinite(U)
    z = z[m]
    U = U[m]
    Iu = Iu[m]
    Iv = Iv[m]
    Iw = Iw[m]
    Lu = Lu[m]
    Lv = Lv[m]
    Lw = Lw[m]

    order = np.argsort(z)
    z = z[order]
    U = U[order]
    Iu = Iu[order]
    Iv = Iv[order]
    Iw = Iw[order]
    Lu = Lu[order]
    Lv = Lv[order]
    Lw = Lw[order]

    U_safe = np.maximum(np.abs(U), 1e-12)
    U_H_safe = max(abs(float(U_H_ref)), 1e-12)

    return {
        "z": z,
        "z_over_H": z / H,
        "U_over_UH": U / U_H_safe,
        "Iu": Iu,
        "Iv": Iv,
        "Iw": Iw,
        "Lu_over_H": Lu / H,
        "Lv_over_H": Lv / H,
        "Lw_over_H": Lw / H,
    }


def build_nheri_experimental_quantities(approach_flow_mat, H):
    """Build raw NHERI experimental profile quantities from the .mat data.

    This avoids Melaku digitised data and avoids smoothed/extended profiles.
    It uses windLespy's NHERI helpers where possible.
    """
    # Raw time series from the wind-tunnel approach flow.
    vel_array_3d = LES._windTunnel.get_nheri_vel_time_series(approach_flow_mat)
    mean_vel_array_2d = LES._profileAnalysis.mean_vel(vel_array_3d)

    # Use windLespy's profile dataframe for the measured z locations and mean
    # profile columns. Do NOT extend/smooth it.
    target_profile_df = LES._windTunnel.get_nheri_profile_df(approach_flow_mat).copy()

    # Add integral length scales calculated directly from the NHERI time series.
    try:
        int_length_scales = LES._windTunnel.calc_nheri_int_length_scales(vel_array_3d)
        target_profile_df = LES._windTunnel.add_nheri_int_length_scales(target_profile_df, int_length_scales)
    except Exception as exc:
        warnings.warn(f"Could not calculate/add NHERI integral length scales through windLespy: {exc}")

    # Find required columns robustly. Different windLespy versions may use
    # slightly different column names.
    z_col = column_lookup(target_profile_df, ["z", "Z", "height", "Height"])
    U_col = column_lookup(target_profile_df, ["U", "Uav", "U_av", "meanU", "Ux", "Umean"])
    Iu_col = column_lookup(target_profile_df, ["Iu", "I_u", "uIntensity", "Iu_exp"])
    Iv_col = column_lookup(target_profile_df, ["Iv", "I_v", "vIntensity", "Iv_exp"])
    Iw_col = column_lookup(target_profile_df, ["Iw", "I_w", "wIntensity", "Iw_exp"])
    Lu_col = column_lookup(target_profile_df, ["Lu", "L_u", "xLu", "x_Lu", "LengthScaleU"])
    Lv_col = column_lookup(target_profile_df, ["Lv", "L_v", "xLv", "x_Lv", "LengthScaleV"])
    Lw_col = column_lookup(target_profile_df, ["Lw", "L_w", "xLw", "x_Lw", "LengthScaleW"])

    if z_col is None:
        raise ValueError(f"Could not identify z column in NHERI profile dataframe: {list(target_profile_df.columns)}")

    z = target_profile_df[z_col].to_numpy(dtype=float)

    if U_col is not None:
        U = target_profile_df[U_col].to_numpy(dtype=float)
    else:
        # Fallback from time series. mean_vel_array_2d usually has shape
        # (3, n_heights), but handle the transposed case too.
        mv = np.asarray(mean_vel_array_2d, dtype=float)
        if mv.shape[0] == 3:
            U = mv[0, :]
        else:
            U = mv[:, 0]
        U = U[: len(z)]

    # If intensity columns are absent, compute directly from velocity time series.
    if Iu_col is not None and Iv_col is not None and Iw_col is not None:
        Iu = target_profile_df[Iu_col].to_numpy(dtype=float)
        Iv = target_profile_df[Iv_col].to_numpy(dtype=float)
        Iw = target_profile_df[Iw_col].to_numpy(dtype=float)
    else:
        U_safe = np.maximum(np.abs(U), 1e-12)
        v = np.asarray(vel_array_3d, dtype=float)
        # Expected shape from windLespy: (3, n_time, n_heights).
        if v.shape[0] == 3:
            fluc = v - np.nanmean(v, axis=1, keepdims=True)
            sig = np.nanstd(fluc, axis=1, ddof=0)
            Iu = sig[0, : len(z)] / U_safe
            Iv = sig[1, : len(z)] / U_safe
            Iw = sig[2, : len(z)] / U_safe
        else:
            raise ValueError(
                "Cannot infer NHERI turbulence intensities from velocity array shape "
                f"{v.shape}; expected (3, n_time, n_heights)."
            )

    def maybe_col(col, name):
        if col is None:
            warnings.warn(f"NHERI {name} column not found. Filling with NaN.")
            return np.full_like(z, np.nan, dtype=float)
        return target_profile_df[col].to_numpy(dtype=float)

    Lu = maybe_col(Lu_col, "Lu")
    Lv = maybe_col(Lv_col, "Lv")
    Lw = maybe_col(Lw_col, "Lw")

    # Ensure common length.
    n = min(len(z), len(U), len(Iu), len(Iv), len(Iw), len(Lu), len(Lv), len(Lw))
    z = z[:n]
    U = U[:n]
    Iu = Iu[:n]
    Iv = Iv[:n]
    Iw = Iw[:n]
    Lu = Lu[:n]
    Lv = Lv[:n]
    Lw = Lw[:n]

    m = np.isfinite(z) & np.isfinite(U)
    order = np.argsort(z[m])
    z = z[m][order]
    U = U[m][order]
    Iu = Iu[m][order]
    Iv = Iv[m][order]
    Iw = Iw[m][order]
    Lu = Lu[m][order]
    Lv = Lv[m][order]
    Lw = Lw[m][order]

    U_H_ref = safe_interp(z, U, H)
    if not np.isfinite(U_H_ref):
        raise ValueError("Could not interpolate NHERI U at z=H.")

    q = {
        "z": z,
        "z_over_H": z / H,
        "U_over_UH": U / max(abs(U_H_ref), 1e-12),
        "Iu": Iu,
        "Iv": Iv,
        "Iw": Iw,
        "Lu_over_H": Lu / H,
        "Lv_over_H": Lv / H,
        "Lw_over_H": Lw / H,
    }

    return target_profile_df, q, float(U_H_ref)


def apply_plot_style():
    """Return colours and set Matplotlib defaults.

    DARK_STYLE=False gives a sharp white background for both the figure and
    axes.  This is deliberately explicit so no dark rcParams survive from a
    previous run in Spyder.
    """
    if not DARK_STYLE:
        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "text.color": "black",
            "font.family": "serif",
            "mathtext.fontset": "cm",
        })
        return {
            "bg": "white",
            "fg": "black",
            "grid": "0.70",
            "exp_edge": "black",
            "exp_face": "none",
            "dfsr": "tab:purple",
            "les": "tab:red",
        }

    plt.rcParams.update({
        "figure.facecolor": "#222b38",
        "axes.facecolor": "#222b38",
        "savefig.facecolor": "#222b38",
        "axes.edgecolor": "#d5dce8",
        "axes.labelcolor": "#d5dce8",
        "xtick.color": "#d5dce8",
        "ytick.color": "#d5dce8",
        "text.color": "#d5dce8",
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })
    return {
        "bg": "#222b38",
        "fg": "#d5dce8",
        "grid": "#7d8795",
        "exp_edge": "#d5dce8",
        "exp_face": "none",
        "dfsr": "#bb6bd9",
        "les": "#ff3b30",
    }


def create_iteration_animation(frame_paths, output_dir, filename=ANIMATION_FILENAME, fps=ANIMATION_FPS):
    """Create a GIF animation from the per-iteration profile PNG files.

    Frames are padded to a common size before writing. This prevents GIF
    writing failures if bbox_inches='tight' creates PNGs that differ by a few
    pixels between iterations.
    """
    frame_paths = [Path(p) for p in frame_paths if Path(p).exists()]
    if not frame_paths:
        warnings.warn("No frame PNGs available for animation.")
        return None

    out_path = Path(output_dir) / filename
    duration_ms = int(round(1000.0 / max(float(fps), 1.0e-9)))

    try:
        from PIL import Image, ImageOps
    except Exception as exc:
        warnings.warn(f"Could not import PIL/Pillow; skipping animation: {exc}")
        return None

    frames = []
    max_w = 0
    max_h = 0

    for path in frame_paths:
        im = Image.open(path).convert("RGB")
        frames.append(im)
        max_w = max(max_w, im.width)
        max_h = max(max_h, im.height)

    padded_frames = []
    bg = (255, 255, 255) if not DARK_STYLE else (34, 43, 56)

    for im in frames:
        canvas = Image.new("RGB", (max_w, max_h), bg)
        x0 = (max_w - im.width) // 2
        y0 = (max_h - im.height) // 2
        canvas.paste(im, (x0, y0))
        padded_frames.append(canvas)

    padded_frames[0].save(
        out_path,
        save_all=True,
        append_images=padded_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )

    return str(out_path)

def plot_iteration_figure(iteration_number, exp_q, dfsr_q, les_q, output_dir):
    """Plot one iteration with a Melaku-like panel aspect and spacing.

    Layout choice
    -------------
    The Melaku figure panels are relatively tall and narrow.  To keep that
    aspect ratio without making the 7-panel figure illegible, this version uses
    explicit axes positions measured in inches:

      * panel (a) is the same size as the other panels and is centred vertically
        in the left column;
      * panels (b)-(d) are the top row to the right;
      * panels (e)-(g) are the bottom row to the right;
      * DFSR is not plotted;
      * LES is plotted over the NHERI experimental points.

    This avoids the previous overlapping-axis problem while keeping the visual
    proportions close to the Melaku/Bitsuamlak plots.
    """
    colours = apply_plot_style()

    # ------------------------------------------------------------------
    # Physical layout in inches.  Increase/decrease these if you want the
    # whole figure larger/smaller while preserving panel aspect ratio.
    # ------------------------------------------------------------------
    panel_w = 2.15          # individual panel width
    panel_h = 3.05          # individual panel height; tall/narrow like Melaku
    gap_x = 0.58            # horizontal gap between panels
    gap_y = 0.58            # vertical gap between top and bottom rows
    left_margin = 0.62
    right_margin = 0.25
    bottom_margin = 0.70
    top_margin = 0.62
    title_extra = 0.18

    fig_w = left_margin + 4 * panel_w + 3 * gap_x + right_margin
    fig_h = bottom_margin + 2 * panel_h + gap_y + top_margin + title_extra

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
    fig.patch.set_facecolor(colours["bg"])

    # Convert inch positions to figure fractions.
    def rect_in(x, y, w=panel_w, h=panel_h):
        return [x / fig_w, y / fig_h, w / fig_w, h / fig_h]

    y_bottom = bottom_margin
    y_top = bottom_margin + panel_h + gap_y
    y_centered = bottom_margin + 0.5 * (panel_h + gap_y)

    x0 = left_margin
    x1 = left_margin + panel_w + gap_x
    x2 = x1 + panel_w + gap_x
    x3 = x2 + panel_w + gap_x

    axes = {
        # Mean profile: same size as the others, centred vertically on the left.
        "U_over_UH": fig.add_axes(rect_in(x0, y_centered)),
        "Iu":        fig.add_axes(rect_in(x1, y_top)),
        "Iv":        fig.add_axes(rect_in(x2, y_top)),
        "Iw":        fig.add_axes(rect_in(x3, y_top)),
        "Lu_over_H": fig.add_axes(rect_in(x1, y_bottom)),
        "Lv_over_H": fig.add_axes(rect_in(x2, y_bottom)),
        "Lw_over_H": fig.add_axes(rect_in(x3, y_bottom)),
    }

    axis_layout = [
        ("U_over_UH", axes["U_over_UH"]),
        ("Iu",        axes["Iu"]),
        ("Iv",        axes["Iv"]),
        ("Iw",        axes["Iw"]),
        ("Lu_over_H", axes["Lu_over_H"]),
        ("Lv_over_H", axes["Lv_over_H"]),
        ("Lw_over_H", axes["Lw_over_H"]),
    ]

    for key, ax in axis_layout:
        cfg = AXIS_CONFIG[key]

        # EXP raw NHERI wind-tunnel data: white open circles.
        ax.scatter(
            exp_q[key], exp_q["z_over_H"],
            s=EXP_MARKER_SIZE,
            facecolors=colours["exp_face"],
            edgecolors=colours["exp_edge"],
            linewidths=EXP_MARKER_LINEWIDTH,
            label="EXP",
            zorder=4,
        )

        # LES downstream/postCorrectionProfile: solid red, drawn on top.
        ax.plot(
            les_q[key], les_q["z_over_H"],
            color=colours["les"],
            linestyle="-",
            linewidth=LES_LINEWIDTH,
            label="LES",
            zorder=7,
        )

        ax.set_xlim(*cfg["xlim"])
        ax.set_xticks(cfg["xticks"])
        ax.set_ylim(*Y_LIM)
        ax.set_yticks(Y_TICKS)
        ax.set_xlabel(cfg["xlabel"], fontsize=10)

        # Smaller, less dominant panel labels than the previous revision.
        ax.set_title(cfg["panel"], fontsize=11, pad=2)

        # Keep y labels readable but not repetitive: one for the centred mean
        # panel, one for the TI row, and one for the length-scale row.
        if key in ("U_over_UH", "Iu", "Lu_over_H"):
            ax.set_ylabel(r"$z/H$", fontsize=10)
        else:
            ax.set_ylabel("")

        ax.tick_params(
            axis="both",
            which="both",
            direction="in",
            length=4.0,
            width=0.8,
            labelbottom=True,
            labelleft=True,
            labelsize=8.5,
            colors=colours["fg"],
        )

        ax.grid(True, which="major", linestyle="--", linewidth=0.50, alpha=0.35, color=colours["grid"])
        for spine in ax.spines.values():
            spine.set_linewidth(0.85)
            spine.set_color(colours["fg"])

    # Legend only in the mean-profile panel.
    axes["U_over_UH"].legend(
        loc="upper left",
        frameon=True,
        facecolor=colours["bg"],
        edgecolor=colours["fg"],
        fontsize=9,
        handlelength=1.8,
        borderpad=0.45,
        labelspacing=0.35,
    )

    fig.suptitle(
        f"Downstream LES profile comparison — iteration {iteration_number}",
        fontsize=13,
        y=0.982,
        color=colours["fg"],
    )

    ensure_dir(output_dir)
    png_path = os.path.join(output_dir, f"iteration{iteration_number:02d}_profiles_nheri_melaku_style_les_only.png")
    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")

    if SAVE_PDF:
        pdf_path = os.path.join(output_dir, f"iteration{iteration_number:02d}_profiles_nheri_melaku_style_les_only.pdf")
        fig.savefig(pdf_path, bbox_inches="tight")

    plt.close(fig)
    return png_path

def quantities_to_dataframe(q, source, iteration=None):
    n = len(q["z_over_H"])
    out = pd.DataFrame({
        "source": source,
        "iteration": iteration,
        "z": q.get("z", np.full(n, np.nan)),
        "z_over_H": q["z_over_H"],
        "U_over_UH": q["U_over_UH"],
        "Iu": q["Iu"],
        "Iv": q["Iv"],
        "Iw": q["Iw"],
        "Lu_over_H": q["Lu_over_H"],
        "Lv_over_H": q["Lv_over_H"],
        "Lw_over_H": q["Lw_over_H"],
    })
    return out


def main():
    variable_dict = LES._caseFiles.parse_setup_file(case_path)
    H = float(variable_dict["buildingHeight"])

    calib_dir = os.path.join(case_path, "log", "downstreamCalibration")
    output_dir = ensure_dir(os.path.join(calib_dir, OUTPUT_SUBDIR))

    print("Building experimental target from raw NHERI wind-tunnel data...")
    exp_df, exp_q, U_H_ref = build_nheri_experimental_quantities(approach_flow_data, H)
    print(f"  building height H = {H:.6g} m")
    print(f"  NHERI U_H = {U_H_ref:.6g}")
    print(f"  NHERI measured profile points = {len(exp_q['z_over_H'])}")
    if WRITE_QA_CSV:
        exp_df.to_csv(os.path.join(output_dir, "nheri_raw_profile_dataframe_used.csv"), index=False)
        quantities_to_dataframe(exp_q, "EXP_NHERI", None).to_csv(
            os.path.join(output_dir, "nheri_raw_profile_quantities_used.csv"), index=False
        )

    iteration_dirs = get_iteration_dirs(calib_dir)
    if not iteration_dirs:
        raise RuntimeError(f"No iteration directories found in {calib_dir}")
    print(f"Found {len(iteration_dirs)} iterations.")
    print(f"Writing figures to: {output_dir}")

    qa_rows = []
    written = []

    for iter_dir in iteration_dirs:
        iteration_number = natural_iteration_key(iter_dir)
        dfsr_path = os.path.join(iter_dir, DFSR_PROFILE_FILE)
        les_path = os.path.join(iter_dir, LES_PROFILE_FILE)

        if not os.path.exists(dfsr_path):
            warnings.warn(f"Skipping iteration {iteration_number}: missing {dfsr_path}")
            continue
        if not os.path.exists(les_path):
            warnings.warn(f"Skipping iteration {iteration_number}: missing {les_path}")
            continue

        dfsr_arr = read_numeric_profile_file(dfsr_path)
        les_arr = read_numeric_profile_file(les_path)

        dfsr_q = profile_array_to_quantities(dfsr_arr, U_H_ref=U_H_ref, H=H, label=f"iteration {iteration_number} DFSR")
        les_q = profile_array_to_quantities(les_arr, U_H_ref=U_H_ref, H=H, label=f"iteration {iteration_number} LES")

        fig_path = plot_iteration_figure(iteration_number, exp_q, dfsr_q, les_q, output_dir)
        written.append(fig_path)
        print(f"  wrote {fig_path}")

        if WRITE_QA_CSV:
            qa_rows.append(quantities_to_dataframe(dfsr_q, "DFSR_newInletProfile", iteration_number))
            qa_rows.append(quantities_to_dataframe(les_q, "LES_postCorrectionProfile", iteration_number))

    if WRITE_QA_CSV and qa_rows:
        all_q = pd.concat([quantities_to_dataframe(exp_q, "EXP_NHERI", None)] + qa_rows, ignore_index=True)
        all_q.to_csv(os.path.join(output_dir, "all_plotted_profile_quantities.csv"), index=False)

    animation_path = None
    if CREATE_ANIMATION:
        animation_path = create_iteration_animation(written, output_dir)
        if animation_path is not None:
            print(f"  wrote animation {animation_path}")

    print("\nDone.")
    print(f"Figures written: {len(written)}")
    if animation_path is not None:
        print(f"Animation written: {animation_path}")
    return written


if __name__ == "__main__":
    main()
