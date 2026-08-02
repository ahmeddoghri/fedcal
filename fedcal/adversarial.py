"""Seeds used to characterize the worst-client calibration finding, and a
disjoint holdout evaluated exactly once.
"""
from __future__ import annotations

# Seeds used while characterizing the finding and tuning the calibrate_v2
# margin/calibration-set fix.
TUNING_SEEDS = list(range(60))

# A disjoint set of seeds, evaluated exactly once, after the fix's margin
# (0.03) was frozen against TUNING_SEEDS above.
HOLDOUT_SEEDS = list(range(1000, 1020))
