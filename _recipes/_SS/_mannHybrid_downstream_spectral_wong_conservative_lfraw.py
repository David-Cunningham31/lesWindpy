# -*- coding: utf-8 -*-
"""wong_conservative update with low-frequency mode=raw.

Thin wrapper for cluster test matrix. It sets MANN_CAL_LOW_FREQ_MODE
and then imports the base recipe.
"""
import os
os.environ.setdefault("MANN_CAL_LOW_FREQ_MODE", "raw")
os.environ.setdefault("MANN_CAL_LOW_PLATEAU_HZ", "0.20")
from _mannHybrid_downstream_spectral_wong_conservative import *  # noqa: F401,F403
