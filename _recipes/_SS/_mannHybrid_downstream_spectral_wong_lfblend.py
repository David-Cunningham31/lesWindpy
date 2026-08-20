# -*- coding: utf-8 -*-
"""wong update with low-frequency mode=blendRaw.

Thin wrapper for cluster test matrix. It sets MANN_CAL_LOW_FREQ_MODE
and then imports the base recipe.
"""
import os
os.environ.setdefault("MANN_CAL_LOW_FREQ_MODE", "blendRaw")
os.environ.setdefault("MANN_CAL_LOW_PLATEAU_HZ", "0.20")
from _mannHybrid_downstream_spectral_wong import *  # noqa: F401,F403
