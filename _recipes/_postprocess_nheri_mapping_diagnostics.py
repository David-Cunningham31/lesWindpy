#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NHERI LES/WT tap-order diagnostics.

This script is designed to sit next to _postprocess_nheri_pressure_statistics.py
and reuse its readers/mapping utilities. It adds stronger diagnostics than
scatter-only comparisons:

1) Pairing maps
   - WT taps and LES probes plotted on each face
   - arrows from LES probe -> mapped WT tap
   - optional comparison to a reference hypothesis (default: excel_as_is)

2) Spatial fields at matched locations
   - WT mean / LES mean / error
   - WT rms  / LES rms  / error

3) Landmark tap checks
   - top corner / bottom corner / centre / edge-centre taps per face
   - CSV + annotated plots showing which LES probe maps to each landmark tap

4) Ordered row/column sequence comparisons
   - Surfaces 2 and 3 by default
   - mean and rms sequence plots along every tap row and tap column
   - useful for spotting reversals / transpositions / scrambled ordering

5) Neighbour-consistency metrics
   - compares local gradients between neighbouring taps in ordered rows/cols
   - more sensitive than global scatter when taps are permuted within a face

6) MAT-ID checks
   - compares Excel "ID (.mat file)" to assigned cp_col+1
   - inventories likely ID arrays in the MAT file when present

Outputs go to:
    <CASE_DIR>/ppd_mapdiag
"""

from __future__ import annotations
import importlib.util
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
BASE_SCRIPT = Path(__file__).with_name("_postprocess_nheri_pressure_statistics.py")
OUT_NAME = "ppd_mapdiag"

HYPOTHESES = [
    "excel_as_is",
    "pdf_face_axes",
    "pdf_face_axes_flip_horizontal",
    "pdf_face_axes_flip_vertical",
    "pdf_face_axes_flip_both",
]
REFERENCE_HYPOTHESIS = "excel_as_is"

COMPARE_LABELS = [
    "WT_full_all_REP1-REP5",
    "WT_matchedDuration_all_REP1-REP5",
    "WT_full_REP1",
    "WT_matchedDuration_REP3",
    "WT_matchedDuration_REP4",
]

SURFACES_FOR_SEQUENCE_DIAG = {"2", "3"}
STATS_TO_CHECK = ["mean", "rms"]

PAIRING_MAX_ANNOTATIONS_PER_FACE = 30
FIG_DPI = 180


# ---------------------------------------------------------------------
# Load base script as a module
# ---------------------------------------------------------------------
def load_base_module(path: Path):
    spec = importlib.util.spec_from_file_location("nheri_base", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


base = load_base_module(BASE_SCRIPT)
OUT_DIR = base.CASE_DIR / OUT_NAME


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def san(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def surface_key(x) -> str:
    return base.canonical_surface(x)


def face_xy_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Choose plot axes within the face. Prefer physical z vertically if it varies,
    otherwise use the two most-varying coordinates.
    """
    if all(c in df.columns for c in ["x_les", "y_les", "z_les"]):
        cols = ["x_les", "y_les", "z_les"]
    else:
        cols = ["x_wt", "y_wt", "z_wt"]

    ranges = {}
    for c in cols:
        a = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        ranges[c] = np.nanmax(a) - np.nanmin(a) if np.any(np.isfinite(a)) else 0.0

    if "z_les" in cols and ranges.get("z_les", 0.0) > 1e-12:
        other = max([c for c in cols if c != "z_les"], key=lambda c: ranges.get(c, 0.0))
        return other, "z_les"
    if "z_wt" in cols and ranges.get("z_wt", 0.0) > 1e-12:
        other = max([c for c in cols if c != "z_wt"], key=lambda c: ranges.get(c, 0.0))
        return other, "z_wt"

    chosen = sorted(cols, key=lambda c: ranges.get(c, 0.0), reverse=True)[:2]
    return chosen[0], chosen[1]


def hypothesis_alias(h: str) -> str:
    return getattr(base, "HYP_ALIAS", {}).get(h, san(h)[:12])


def comparison_alias(c: str) -> str:
    return getattr(base, "COMP_ALIAS", {}).get(c, san(c)[:16])


def unique_preserve(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ---------------------------------------------------------------------
# MAT-ID / array checks
# ---------------------------------------------------------------------
def inspect_mat_id_alignment(layout: pd.DataFrame, first_wt, out_dir: Path) -> None:
    rows = []
    if "mat_id" in layout.columns:
        valid = layout[layout["mat_id"].notna()].copy()
        if not valid.empty:
            valid = valid.reset_index(drop=True)
            valid["assigned_cp_col_1based"] = np.arange(1, len(valid) + 1)
            valid["mat_id_matches_assigned"] = valid["mat_id"].astype(float) == valid["assigned_cp_col_1based"].astype(float)
            valid.to_csv(out_dir / "excel_mat_id_vs_assigned_cpcol.csv", index=False)

    raw = base.loadmat(first_wt.path, squeeze_me=True, struct_as_record=False)
    scan = base.get_scanivalve_struct(raw, first_wt.path)
    arrays = base.find_numeric_arrays_recursive({"MAT_ROOT": raw, "ScanivalveOnly": scan})
    ncp = first_wt.cp.shape[1]
    recs = []
    for name, arr in arrays:
        arr = np.asarray(arr)
        if arr.ndim == 1 and arr.shape[0] == ncp:
            vals = arr.astype(float, copy=False)
            is_1_to_n = np.array_equal(vals, np.arange(1, ncp + 1))
            is_0_to_n1 = np.array_equal(vals, np.arange(0, ncp))
            recs.append({
                "name": name,
                "shape": tuple(arr.shape),
                "min": float(np.nanmin(vals)),
                "max": float(np.nanmax(vals)),
                "is_exact_1_to_n": bool(is_1_to_n),
                "is_exact_0_to_n1": bool(is_0_to_n1),
            })
    pd.DataFrame(recs).sort_values(["is_exact_1_to_n", "is_exact_0_to_n1", "name"], ascending=[False, False, True]).to_csv(
        out_dir / "mat_id_like_arrays.csv", index=False
    )


# ---------------------------------------------------------------------
# Compute one hypothesis and one comparison table
# ---------------------------------------------------------------------
def prepare_hypothesis_context(hypothesis: str, les_stats: pd.DataFrame, first_wt, out_dir: Path):
    hyp_dir = ensure_dir(out_dir / hypothesis_alias(hypothesis))
    diag_dir = ensure_dir(hyp_dir / "diagnostics")

    layout = base.read_tap_layout()
    layout_h = base.apply_layout_hypothesis(layout, hypothesis, diag_dir)
    layout_v = base.prepare_layout_for_cp_columns(layout_h, first_wt.cp.shape[1], diag_dir)
    mapping = base.build_les_to_wt_mapping(les_stats, layout_v, diag_dir)
    mapping = mapping.dropna(subset=["cp_col"]).copy()
    mapping["cp_col"] = mapping["cp_col"].astype(int)

    # enrich from layout
    keep_cols = [c for c in ["tap", "surface", "tap_row", "tap_col", "mat_id", "cp_col", "x_wt", "y_wt", "z_wt"] if c in layout_v.columns]
    layout_lookup = layout_v[keep_cols].copy()
    mapping = mapping.merge(layout_lookup.drop(columns=[c for c in ["tap", "surface", "x_wt", "y_wt", "z_wt"] if c in layout_lookup.columns]),
                            on="cp_col", how="left")
    mapping.to_csv(diag_dir / "mapping_enriched.csv", index=False)

    return hyp_dir, mapping, layout_v


def comparison_point_table(label: str, first_wt, files, mapping, les_stats, ref_info):
    needed = mapping["cp_col"].to_numpy(int)
    les_duration = ref_info["les_duration"]

    def get_record(f):
        return first_wt if f == first_wt.path else base.read_scanivalve_cp_file(f)

    if label == "WT_full_all_REP1-REP5":
        acc = base.init_stats_acc(len(needed))
        used_first = False
        for f in files:
            wt = first_wt if (not used_first and f == first_wt.path) else base.read_scanivalve_cp_file(f)
            if f == first_wt.path:
                used_first = True
            base.update_cp_stats_acc_for_columns(acc, wt.cp, needed)
        wt_stats = base.finalize_stats_acc(acc)
        return base.assemble_pointwise_comparison(les_stats, mapping, wt_stats, label)

    if label == "WT_matchedDuration_all_REP1-REP5":
        acc = base.init_stats_acc(len(needed))
        used_first = False
        for f in files:
            wt = first_wt if (not used_first and f == first_wt.path) else base.read_scanivalve_cp_file(f)
            if f == first_wt.path:
                used_first = True
            matched = base.select_time_record(wt.cp, wt.fs, base.WT_MATCH_START_TIME, les_duration)
            base.update_cp_stats_acc_for_columns(acc, matched, needed)
        wt_stats = base.finalize_stats_acc(acc)
        return base.assemble_pointwise_comparison(les_stats, mapping, wt_stats, label)

    # REP-specific
    m = re.search(r"REP(\d+)", label)
    if not m:
        raise ValueError(label)
    rep = f"REP{m.group(1)}"
    target = None
    for f in files:
        if rep in f.stem.upper():
            target = f
            break
    if target is None:
        raise FileNotFoundError(rep)
    wt = first_wt if target == first_wt.path else base.read_scanivalve_cp_file(target)
    cp_record = wt.cp if "WT_full_" in label else base.select_time_record(wt.cp, wt.fs, base.WT_MATCH_START_TIME, les_duration)
    wt_stats = base.cp_stats_for_columns(cp_record, needed)
    return base.assemble_pointwise_comparison(les_stats, mapping, wt_stats, label)


# ---------------------------------------------------------------------
# Stronger diagnostics
# ---------------------------------------------------------------------
def plot_pairing_map(point: pd.DataFrame, mapping: pd.DataFrame, hypothesis: str, ref_map: pd.DataFrame | None, out_dir: Path) -> None:
    fig_dir = ensure_dir(out_dir / "pairing_maps")
    surfaces = sorted(unique_preserve(point["surface"].map(surface_key)))
    for surf in surfaces:
        face = point.loc[point["surface"].map(surface_key) == surf].copy()
        if face.empty:
            continue
        xw = face["x_wt"].to_numpy(float)
        yw = face["z_wt"].to_numpy(float) if np.nanmax(face["z_wt"]) - np.nanmin(face["z_wt"]) > 1e-12 else face["y_wt"].to_numpy(float)
        xl = face["x_les"].to_numpy(float)
        yl = face["z_les"].to_numpy(float) if np.nanmax(face["z_les"]) - np.nanmin(face["z_les"]) > 1e-12 else face["y_les"].to_numpy(float)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(xw, yw, s=22, c="tab:blue", label="WT taps")
        ax.scatter(xl, yl, s=22, c="tab:orange", label="LES probes")

        # arrows
        for _, r in face.iterrows():
            ax.plot([r["x_les"], r["x_wt"]], [r["z_les"] if "z_les" in face else r["y_les"], r["z_wt"] if "z_wt" in face else r["y_wt"]],
                    color="0.65", linewidth=0.6, alpha=0.8)

        # annotate subset of tap ids
        ann = face.sort_values("tap").head(PAIRING_MAX_ANNOTATIONS_PER_FACE)
        for _, r in ann.iterrows():
            ax.text(r["x_wt"], r["z_wt"] if "z_wt" in face else r["y_wt"], str(int(r["tap"])) if pd.notna(r["tap"]) else "",
                    fontsize=6, color="tab:blue")

        # changed from reference
        if ref_map is not None:
            ref = ref_map[["les_index", "cp_col"]].rename(columns={"cp_col": "cp_col_ref"})
            cmp = face.merge(ref, on="les_index", how="left")
            changed = cmp["cp_col_ref"].notna() & (cmp["cp_col_ref"].astype(int) != cmp["cp_col"].astype(int))
            if changed.any():
                ax.scatter(cmp.loc[changed, "x_les"], cmp.loc[changed, "z_les"], s=60, facecolors="none",
                           edgecolors="red", linewidths=1.2, label="changed vs excel")

        ax.set_title(f"Surface {surf} pairing map [{hypothesis}]")
        ax.set_xlabel("horizontal coord")
        ax.set_ylabel("vertical coord")
        ax.set_aspect("equal", adjustable="box")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"pair_s{surf}_{hypothesis_alias(hypothesis)}.png", dpi=FIG_DPI)
        plt.close(fig)


def plot_spatial_error_fields(point: pd.DataFrame, label: str, hypothesis: str, out_dir: Path):
    fig_dir = ensure_dir(out_dir / "spatial_fields")
    surfaces = sorted(unique_preserve(point["surface"].map(surface_key)))
    for stat in STATS_TO_CHECK:
        for surf in surfaces:
            face = point.loc[point["surface"].map(surface_key) == surf].copy()
            if face.empty:
                continue
            xcol, ycol = face_xy_columns(face)
            x = face[xcol].to_numpy(float)
            y = face[ycol].to_numpy(float)
            wt = face[f"Cp_{stat}_WT"].to_numpy(float)
            les = face[f"Cp_{stat}_LES"].to_numpy(float)
            err = les - wt

            fig, axs = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
            for ax, vals, ttl, cmap in [
                (axs[0], wt, f"WT {stat}", "RdBu_r" if stat == "mean" else "viridis"),
                (axs[1], les, f"LES {stat}", "RdBu_r" if stat == "mean" else "viridis"),
                (axs[2], err, f"LES-WT {stat}", "RdBu_r"),
            ]:
                sc = ax.scatter(x, y, c=vals, s=42, cmap=cmap)
                for _, r in face.iterrows():
                    ax.text(r[xcol], r[ycol], str(int(r["tap"])) if pd.notna(r["tap"]) else "", fontsize=5, alpha=0.7)
                ax.set_title(ttl)
                ax.set_aspect("equal", adjustable="box")
                plt.colorbar(sc, ax=ax, shrink=0.85)
            fig.suptitle(f"{label} | surface {surf} | {hypothesis}")
            fig.savefig(fig_dir / f"spatial_s{surf}_{stat}_{comparison_alias(label)}_{hypothesis_alias(hypothesis)}.png", dpi=FIG_DPI)
            plt.close(fig)


def choose_landmarks(face: pd.DataFrame, xcol: str, ycol: str):
    pts = face[[xcol, ycol]].to_numpy(float)
    x = pts[:, 0]
    y = pts[:, 1]
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    targets = {
        "top_left": np.array([xmin, ymax]),
        "top_right": np.array([xmax, ymax]),
        "bottom_left": np.array([xmin, ymin]),
        "bottom_right": np.array([xmax, ymin]),
        "center": np.array([(xmin+xmax)/2, (ymin+ymax)/2]),
        "left_mid": np.array([xmin, (ymin+ymax)/2]),
        "right_mid": np.array([xmax, (ymin+ymax)/2]),
    }
    rows = []
    for name, t in targets.items():
        d = np.linalg.norm(pts - t[None, :], axis=1)
        j = int(np.nanargmin(d))
        rows.append(face.iloc[j].copy())
        rows[-1]["landmark"] = name
    lm = pd.DataFrame(rows).drop_duplicates(subset=["tap", "landmark"])
    return lm


def export_landmarks(point: pd.DataFrame, hypothesis: str, out_dir: Path):
    csv_dir = ensure_dir(out_dir / "landmarks")
    plot_dir = ensure_dir(csv_dir / "figures")
    surfaces = sorted(unique_preserve(point["surface"].map(surface_key)))
    all_rows = []
    for surf in surfaces:
        face = point.loc[point["surface"].map(surface_key) == surf].copy()
        if face.empty:
            continue
        xcol, ycol = face_xy_columns(face)
        lm = choose_landmarks(face, xcol, ycol)
        lm["hypothesis"] = hypothesis
        all_rows.append(lm)

        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.scatter(face[xcol], face[ycol], s=22, c="0.7")
        ax.scatter(lm[xcol], lm[ycol], s=60, c="red")
        for _, r in lm.iterrows():
            ax.text(r[xcol], r[ycol], f"{r['landmark']} | tap {int(r['tap'])}", fontsize=7)
        ax.set_title(f"Surface {surf} landmarks [{hypothesis}]")
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        fig.savefig(plot_dir / f"landmarks_s{surf}_{hypothesis_alias(hypothesis)}.png", dpi=FIG_DPI)
        plt.close(fig)

    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(csv_dir / f"landmarks_{hypothesis_alias(hypothesis)}.csv", index=False)


def ordered_sequences(point: pd.DataFrame, hypothesis: str, label: str, out_dir: Path):
    seq_dir = ensure_dir(out_dir / "ordered_sequences")
    rows_out = []
    for surf in sorted(SURFACES_FOR_SEQUENCE_DIAG):
        face = point.loc[point["surface"].map(surface_key) == surf].copy()
        if face.empty or "tap_row" not in face.columns or "tap_col" not in face.columns:
            continue
        for stat in STATS_TO_CHECK:
            # row sequences
            for row_id, grp in face.groupby("tap_row"):
                g = grp.sort_values("tap_col")
                rows_out.append({
                    "surface": surf, "direction": "row", "group_id": row_id, "stat": stat,
                    "hypothesis": hypothesis, "corr_sequence": float(np.corrcoef(g[f"Cp_{stat}_WT"], g[f"Cp_{stat}_LES"])[0,1]) if len(g) > 1 else np.nan,
                    "mae_sequence": float(np.mean(np.abs(g[f"Cp_{stat}_LES"] - g[f"Cp_{stat}_WT"]))),
                })
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(g["tap_col"], g[f"Cp_{stat}_WT"], "-o", label="WT")
                ax.plot(g["tap_col"], g[f"Cp_{stat}_LES"], "-o", label="LES")
                for _, r in g.iterrows():
                    ax.text(r["tap_col"], r[f"Cp_{stat}_WT"], str(int(r["tap"])), fontsize=6, alpha=0.7)
                ax.set_title(f"S{surf} row {row_id} {stat} | {label} | {hypothesis}")
                ax.set_xlabel("tap_col")
                ax.legend()
                fig.tight_layout()
                fig.savefig(seq_dir / f"seq_s{surf}_row{int(row_id)}_{stat}_{comparison_alias(label)}_{hypothesis_alias(hypothesis)}.png", dpi=FIG_DPI)
                plt.close(fig)

            # column sequences
            for col_id, grp in face.groupby("tap_col"):
                g = grp.sort_values("tap_row")
                rows_out.append({
                    "surface": surf, "direction": "col", "group_id": col_id, "stat": stat,
                    "hypothesis": hypothesis, "corr_sequence": float(np.corrcoef(g[f"Cp_{stat}_WT"], g[f"Cp_{stat}_LES"])[0,1]) if len(g) > 1 else np.nan,
                    "mae_sequence": float(np.mean(np.abs(g[f"Cp_{stat}_LES"] - g[f"Cp_{stat}_WT"]))),
                })
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(g["tap_row"], g[f"Cp_{stat}_WT"], "-o", label="WT")
                ax.plot(g["tap_row"], g[f"Cp_{stat}_LES"], "-o", label="LES")
                for _, r in g.iterrows():
                    ax.text(r["tap_row"], r[f"Cp_{stat}_WT"], str(int(r["tap"])), fontsize=6, alpha=0.7)
                ax.set_title(f"S{surf} col {col_id} {stat} | {label} | {hypothesis}")
                ax.set_xlabel("tap_row")
                ax.legend()
                fig.tight_layout()
                fig.savefig(seq_dir / f"seq_s{surf}_col{int(col_id)}_{stat}_{comparison_alias(label)}_{hypothesis_alias(hypothesis)}.png", dpi=FIG_DPI)
                plt.close(fig)

    if rows_out:
        pd.DataFrame(rows_out).to_csv(seq_dir / f"sequence_metrics_{comparison_alias(label)}_{hypothesis_alias(hypothesis)}.csv", index=False)


def neighbour_metric(point: pd.DataFrame, hypothesis: str, label: str):
    rows = []
    for surf in sorted(SURFACES_FOR_SEQUENCE_DIAG):
        face = point.loc[point["surface"].map(surface_key) == surf].copy()
        if face.empty or "tap_row" not in face.columns or "tap_col" not in face.columns:
            continue
        for stat in STATS_TO_CHECK:
            # row neighbours
            diffs = []
            for _, grp in face.groupby("tap_row"):
                g = grp.sort_values("tap_col")
                if len(g) < 2:
                    continue
                wt_d = np.diff(g[f"Cp_{stat}_WT"].to_numpy(float))
                les_d = np.diff(g[f"Cp_{stat}_LES"].to_numpy(float))
                diffs.extend(np.abs(les_d - wt_d))
            row_pen = float(np.mean(diffs)) if diffs else np.nan

            diffs = []
            for _, grp in face.groupby("tap_col"):
                g = grp.sort_values("tap_row")
                if len(g) < 2:
                    continue
                wt_d = np.diff(g[f"Cp_{stat}_WT"].to_numpy(float))
                les_d = np.diff(g[f"Cp_{stat}_LES"].to_numpy(float))
                diffs.extend(np.abs(les_d - wt_d))
            col_pen = float(np.mean(diffs)) if diffs else np.nan

            rows.append({
                "comparison": label,
                "hypothesis": hypothesis,
                "surface": surf,
                "stat": stat,
                "row_neighbour_penalty": row_pen,
                "col_neighbour_penalty": col_pen,
                "combined_penalty": np.nanmean([row_pen, col_pen]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------
def main():
    ensure_dir(OUT_DIR)
    les_stats, les_cp_df, ref_info, les_coords = base.read_les_pressure_and_reference()
    files = base.list_cp_files()
    first = base.read_scanivalve_cp_file(files[0])

    inspect_mat_id_alignment(base.read_tap_layout(), first, ensure_dir(OUT_DIR / "mat_id_checks"))

    # precompute mapping for reference
    ref_hdir, ref_mapping, _ = prepare_hypothesis_context(REFERENCE_HYPOTHESIS, les_stats, first, OUT_DIR)

    all_rank = []
    for hyp in HYPOTHESES:
        print(f"Running map diagnostics for hypothesis: {hyp}")
        hyp_dir, mapping, layout_v = prepare_hypothesis_context(hyp, les_stats, first, OUT_DIR)

        # enrich mapping with tap_row/col/mat_id from layout_v
        extra = [c for c in ["cp_col", "tap_row", "tap_col", "mat_id"] if c in layout_v.columns]
        mapping = mapping.merge(layout_v[extra].drop_duplicates("cp_col"), on="cp_col", how="left")

        for label in COMPARE_LABELS:
            point = comparison_point_table(label, first, files, mapping, les_stats, ref_info)
            point = point.merge(mapping[["les_index", "tap_row", "tap_col", "mat_id"]], on="les_index", how="left")
            point.to_csv(ensure_dir(hyp_dir / "csv") / f"point_{comparison_alias(label)}_{hypothesis_alias(hyp)}.csv", index=False)

            plot_pairing_map(point, mapping, hyp, ref_mapping if hyp != REFERENCE_HYPOTHESIS else None, hyp_dir)
            plot_spatial_error_fields(point, label, hyp, hyp_dir)
            export_landmarks(point, hyp, hyp_dir)
            ordered_sequences(point, hyp, label, hyp_dir)

            nm = neighbour_metric(point, hyp, label)
            if not nm.empty:
                nm.to_csv(ensure_dir(hyp_dir / "sequence_metrics") / f"neigh_{comparison_alias(label)}_{hypothesis_alias(hyp)}.csv", index=False)
                all_rank.append(nm)

    if all_rank:
        rank = pd.concat(all_rank, ignore_index=True)
        summary = rank.groupby(["hypothesis", "stat"], as_index=False)["combined_penalty"].mean()
        summary.to_csv(OUT_DIR / "neighbour_penalty_summary.csv", index=False)

    print(f"Done. Outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
