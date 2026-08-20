# -*- coding: utf-8 -*-
"""Initialise an empty-domain LES calibration using explicit time windows.

The historical recipe inferred ``u_star`` and ``z0`` from a two-parameter
log-law fit and then transplanted the Kim et al. channel-flow values of
20 and 40 turnover times. Those assumptions are not appropriate for the
full-scale Euston ABL cases. This replacement keeps the legacy filename so
existing Slurm drivers continue to work, but obtains the two durations from
explicit user inputs.

Input precedence
----------------
1. Command-line options ``--burn-in-time`` and ``--averaging-time``.
2. Environment variables ``CALIBRATION_BURN_IN_TIME`` and
   ``CALIBRATION_AVERAGING_TIME``.
3. Optional ``setUp`` entries ``calibrationBurnInTime`` and
   ``calibrationAveragingTime``.
4. Editable standalone defaults ``DEFAULT_BURN_IN_TIME`` and
   ``DEFAULT_AVERAGING_TIME`` (400 s and 3600 s as supplied).

The MeluXina driver exports the environment variables, providing one obvious,
auditable source of truth for batch jobs. The output retains the established
JSON keys consumed by downstream windlespy recipes and adds explicit metadata.
"""

from __future__ import print_function

import argparse
import json
import math
import os
import sys
from pathlib import Path


def _import_windlespy():
    """Import an installed checkout, with the historical repo fallback."""
    try:
        import windlespy as module

        return module
    except ImportError:
        recipe_dir = Path(__file__).resolve().parent
        repository_parent = recipe_dir.parent.parent
        sys.path.insert(0, str(repository_parent))
        try:
            import windlespy as module

            return module
        finally:
            try:
                sys.path.remove(str(repository_parent))
            except ValueError:
                pass


# Editable standalone/IDE defaults.  The Slurm driver exports values with the
# same names, so batch jobs remain controlled from the job script.
IDE_CASE_PATH = None  # e.g. r"C:\path\to\OpenFOAM\case"; otherwise use the IDE working directory
DEFAULT_BURN_IN_TIME = 400.0
DEFAULT_AVERAGING_TIME = 3600.0


def _finite_positive(name, value):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be numeric; got {!r}".format(name, value)) from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError("{} must be finite and greater than zero; got {!r}".format(name, value))
    return number


def _optional_positive(name, value):
    if value is None or str(value).strip() == "":
        return None
    return _finite_positive(name, value)


def _first_present(mapping, names):
    for name in names:
        if name in mapping and mapping[name] is not None and str(mapping[name]).strip() != "":
            return mapping[name], "setUp:{}".format(name)
    return None, None


def _resolve_value(cli_value, env_name, setup_values, setup_names, default, label):
    if cli_value is not None:
        return _finite_positive(label, cli_value), "command-line"

    env_value = os.environ.get(env_name)
    if env_value is not None and env_value.strip() != "":
        return _finite_positive(label, env_value), "environment:{}".format(env_name)

    setup_value, setup_source = _first_present(setup_values, setup_names)
    if setup_value is not None:
        return _finite_positive(label, setup_value), setup_source

    return _finite_positive(label, default), "script default"


def _is_grid_aligned(value, delta_t):
    intervals = value / delta_t
    return math.isclose(intervals, round(intervals), rel_tol=1.0e-10, abs_tol=1.0e-8)


def build_parser():
    default_case = os.environ.get("CASE_DIR") or IDE_CASE_PATH or os.getcwd()
    parser = argparse.ArgumentParser(
        description=(
            "Write log/downstreamCalibration/sim_init.json using explicit "
            "empty-domain burn-in and retained-averaging durations."
        )
    )
    parser.add_argument(
        "case",
        nargs="?",
        default=default_case,
        help=(
            "OpenFOAM case directory (default precedence: CASE_DIR, "
            "IDE_CASE_PATH, current working directory)."
        ),
    )
    parser.add_argument(
        "--burn-in-time",
        type=float,
        default=None,
        help=(
            "Discarded burn-in duration in simulation seconds "
            "(standalone default: {}).".format(DEFAULT_BURN_IN_TIME)
        ),
    )
    parser.add_argument(
        "--averaging-time",
        type=float,
        default=None,
        help=(
            "Retained statistical averaging duration in simulation seconds "
            "(standalone default: {}).".format(DEFAULT_AVERAGING_TIME)
        ),
    )
    parser.add_argument(
        "--delta-t",
        type=float,
        default=None,
        help=(
            "Optional solver time step. If supplied, both timing boundaries "
            "must lie on its time grid and deltaT is recorded in the JSON."
        ),
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the generated metadata after writing it.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.case:
        raise SystemExit("ERROR: provide a case path or set CASE_DIR")

    case_path = os.path.abspath(os.path.expanduser(args.case))
    if not os.path.isdir(case_path):
        raise SystemExit("ERROR: case directory does not exist: {}".format(case_path))

    LES = _import_windlespy()
    variable_dict = LES._caseFiles.parse_setup_file(case_path)

    try:
        burn_in_time, burn_source = _resolve_value(
            args.burn_in_time,
            "CALIBRATION_BURN_IN_TIME",
            variable_dict,
            ("calibrationBurnInTime", "burnInTime"),
            DEFAULT_BURN_IN_TIME,
            "burn-in duration",
        )
        averaging_duration, averaging_source = _resolve_value(
            args.averaging_time,
            "CALIBRATION_AVERAGING_TIME",
            variable_dict,
            ("calibrationAveragingTime", "averagingTime", "minAvgTime"),
            DEFAULT_AVERAGING_TIME,
            "averaging duration",
        )

        delta_t_value = args.delta_t
        delta_t_source = "command-line" if args.delta_t is not None else None
        if delta_t_value is None:
            env_delta_t = os.environ.get("CALIBRATION_DELTA_T")
            if env_delta_t is not None and env_delta_t.strip() != "":
                delta_t_value = env_delta_t
                delta_t_source = "environment:CALIBRATION_DELTA_T"
        # Do not copy a possibly stale setUp deltaT into JSON. The Slurm driver
        # deliberately resolves controlDict before setUp when no explicit
        # --delta-t/CALIBRATION_DELTA_T was supplied.
        delta_t = _optional_positive("deltaT", delta_t_value)
    except ValueError as exc:
        raise SystemExit("ERROR: {}".format(exc))

    statistics_start_time = burn_in_time
    statistics_end_time = burn_in_time + averaging_duration

    if delta_t is not None:
        for name, value in (
            ("burn-in duration", burn_in_time),
            ("averaging duration", averaging_duration),
            ("total duration", statistics_end_time),
        ):
            if not _is_grid_aligned(value, delta_t):
                raise SystemExit(
                    "ERROR: {}={} is not aligned with deltaT={}".format(name, value, delta_t)
                )

    # Established duration fields are retained for downstream compatibility.
    metadata = {
        "schema_version": 2,
        "timing_mode": "manual",
        "burn_in_time": burn_in_time,
        "retained_averaging_time": averaging_duration,
        "averaging_duration": averaging_duration,
        "min_avg_time": averaging_duration,
        "statistics_start_time": statistics_start_time,
        "statistics_end_time": statistics_end_time,
        "initial_sim_duration": statistics_end_time,
        "timing_sources": {
            "burn_in_time": burn_source,
            "retained_averaging_time": averaging_source,
        },
    }
    if delta_t is not None:
        metadata["deltaT"] = delta_t
        metadata["timing_sources"]["deltaT"] = delta_t_source

    LES._caseFiles.write_dfsr_les_init_json(case_path, metadata)

    json_path = os.path.join(case_path, "log", "downstreamCalibration", "sim_init.json")
    print("Calibration timing mode : manual")
    print("Burn-in duration        : {:.12g} s ({})".format(burn_in_time, burn_source))
    print(
        "Retained averaging      : {:.12g} s ({})".format(
            averaging_duration,
            averaging_source,
        )
    )
    print(
        "Statistics window       : [{:.12g}, {:.12g}] s".format(
            statistics_start_time,
            statistics_end_time,
        )
    )
    print("Total solver duration   : {:.12g} s".format(statistics_end_time))
    if delta_t is not None:
        print("Solver deltaT           : {:.12g} s ({})".format(delta_t, delta_t_source))
    print("Wrote                   : {}".format(json_path))

    if args.print_json:
        print(json.dumps(metadata, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
