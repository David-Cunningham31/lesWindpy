#!/usr/bin/env python3
"""Simplified Euston Tower MannHybrid spectral-tilt calibration recipe.

This is the small, case-specific configuration layer. The numerical engine is
implemented in
:mod:`euston_tower_mannhybrid_spectral_tilt_core_simplified`; keep both files
together in the OpenFOAM case directory or in ``windlespy/_recipes``.

Exit-code contract used by the MeluXina driver
-----------------------------------------------
0
    Calibration engine deliberately reports completion.
1
    A valid update was written and another MannHybrid/LES pass is required.
2
    Configuration, input, or numerical failure. This must never be treated as
    a request for another calibration pass.
"""

import sys
import traceback
from pathlib import Path


# MannHybridTurb generator behaviour:
#   rawMann                 : Reynolds shear stress is plotted, not calibrated.
#   sameComponentCoherence  : auto-spectra use the established calibration;
#                             Reynolds shear stress is plotted, not calibrated.
#   reynoldsShearStress     : the established auto-spectrum calibration is used
#                             and the profile Reynolds shear stress is calibrated.
#
# No standalone u-w cospectrum file is tilted in any revised mode. The utility
# constructs the Mann-shaped u-w cospectrum internally from the profile stress.
INFLOW_MODE = "sameComponentCoherence"


# Optional Spyder/IDE setting. Leave as None when CASE_DIR is exported or the
# IDE working directory is the OpenFOAM case directory.
IDE_CASE_PATH = None  # e.g. Path(r"C:\path\to\euston_empty_domain_case")


def main() -> int:
    """Build the Euston-specific configuration and run one calibration pass."""
    from euston_tower_mannhybrid_spectral_tilt_core_simplified import (
        default_config,
        run_calibration,
    )

    # Deliberately simplified controller:
    #   - Profile U, R11, R22, R33 and, when enabled by INFLOW_MODE, u'w' use
    #     the unchanged windlespy Wong update. The only post-update safeguards
    #     are U > 0 and Iu/Iv/Iw <= 0.50.
    #   - Spectral tilt direction is controlled by log(L_down/L_target).
    #   - One common moment_relax gain and one common max_log_tilt apply to
    #     u, v and w. There are no component-specific gains, engineering
    #     length-error deadband, control-error cap, or height smoothing.
    #   - No band-energy correction is applied.
    #   - Final auto-spectrum area is renormalised over the resolved frequency
    #     band to the Wong-updated variance.
    case_override = {}
    if IDE_CASE_PATH is not None:
        case_override["case_dir"] = Path(IDE_CASE_PATH)

    cfg = default_config(
        mode="length_tilt",
        inflow_mode=INFLOW_MODE,
        components=("u", "v", "w"),
        # Welch uses 4096-sample Hann segments with 50% overlap. Integral
        # lengths use the original NHERI e-fold estimator. The active upper
        # frequency remains the required setUp:maximumFrequency ceiling.
        f_min=0.0,
        f_max_update=float("inf"),
        nperseg=4096,
        l_method="efold",
        log_dir_name="eustonTower_spectralTiltCalibration_simplified_welch4096_efold",
        variance_relax=0.0,
        # r_i in the controller equation is this single scalar value for every
        # component. max_log_tilt is a separate hard per-bin safety limit.
        moment_relax=0.50,
        max_log_tilt=0.70,
        shape_relax=0.0,
        use_windlespy_wong_update=True,
        wong_relaxation_factor=0.9,
        max_turbulence_intensity=0.50,
        update_profile_intensity=True,
        update_profile_length=False,
        # These are enforced from inflow_mode by the common engine. They are
        # shown explicitly so the intended behaviour is immediately visible.
        update_uw_stress=(INFLOW_MODE == "reynoldsShearStress"),
        update_uw_cospectrum=False,
        use_windlespy_resolved_band=True,
        uw_cospectrum_resolved_band_only=True,
        **case_override,
    )
    return int(run_calibration(cfg))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except SystemExit:
        raise
    except Exception as exc:
        print(
            "ERROR: Euston Tower spectral-tilt calibration failed: {}".format(exc),
            file=sys.stderr,
        )
        traceback.print_exc()
        raise SystemExit(2)
