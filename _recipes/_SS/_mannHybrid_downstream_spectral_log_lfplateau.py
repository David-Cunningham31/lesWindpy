# -*- coding: utf-8 -*-
"""log update with low-frequency mode=plateau.

Thin wrapper for cluster test matrix. It sets MANN_CAL_LOW_FREQ_MODE
and then imports the base recipe.
"""
import os
os.environ.setdefault("MANN_CAL_LOW_FREQ_MODE", "plateau")
os.environ.setdefault("MANN_CAL_LOW_PLATEAU_HZ", "0.20")
from _mannHybrid_downstream_spectral_log import *  # noqa: F401,F403
