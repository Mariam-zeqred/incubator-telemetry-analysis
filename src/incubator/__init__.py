"""Analysis of real queen-bee incubator telemetry.

Modules
-------
load        -- read the raw files in ``data/`` into tidy, UTC-aware DataFrames
features    -- heater duty cycle, heat-up / cool-down rates, overshoot, time-to-target
anomaly     -- rolling z-score and IsolationForest detectors, and a comparison of the two
thermal_fit -- least-squares fit of a first-order thermal model (feeds the control-sim repo)
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("incubator-telemetry-analysis")
except PackageNotFoundError:  # running from a source checkout without pip install
    __version__ = "0.1.0"

__all__ = ["__version__"]
