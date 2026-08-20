#!/usr/bin/env python3
"""Euston Tower MannHybrid downstream spectral-tilt calibration recipe.

This is the small, case-specific configuration layer. The numerical engine is
implemented in :mod:`euston_tower_mannhybrid_spectral_tilt_core`; keep both
files together in the OpenFOAM case directory or in ``windlespy/_recipes``.

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


def main() -> int:
    """Build the Euston-specific configuration and run one calibration pass."""
    from euston_tower_mannhybrid_spectral_tilt_core import (
        default_config,
        run_calibration,
    )

    # Variety 1 (unchanged algorithm and tuned values):
    #   - Profile U, R11, R22, R33 and, when enabled by INFLOW_MODE, u'w' use
    #     the actual windlespy Wong update without additional relaxation.  The
    #     core applies only a final U >= 0.01 m/s positivity safeguard.
    #   - Spectral tilt direction is controlled by log(L_down/L_target).
    #   - No band-energy correction is applied.
    #   - Final auto-spectrum area is renormalised over the resolved frequency
    #     band to the Wong-updated variance.
    cfg = default_config(
        mode="length_tilt",
        inflow_mode=INFLOW_MODE,
        components=("u", "v", "w"),
        # Euston/full-scale frequency and length-scale controls.
        # The core derives f_min from 1/T_record and f_max from the required
        # setUp scalar maximumFrequency.  nperseg=0 selects a full-record Hann
        # estimate on (or interpolated to) the spectraProfile frequency grid.
        f_min=0.0,
        f_max_update=float("inf"),
        nperseg=0,
        l_method="first_zero",
        log_dir_name="eustonTower_spectralTiltCalibration_lengthTilt_noBands",
        variance_relax=0.0,
        moment_relax=0.70,
        shape_relax=0.0,
        use_windlespy_wong_update=True,
        wong_relaxation_factor=0.9,
        max_log_band_update=1.50,
        max_log_tilt=1.40,
        max_log_uw_update=1.10,
        update_profile_intensity=True,
        update_profile_length=False,
        # These are enforced from inflow_mode by the common engine. They are
        # shown explicitly so the intended behaviour is immediately visible.
        update_uw_stress=(INFLOW_MODE == "reynoldsShearStress"),
        update_uw_cospectrum=False,
        use_windlespy_resolved_band=True,
        uw_cospectrum_resolved_band_only=True,
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
