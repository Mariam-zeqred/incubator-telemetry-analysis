"""Two unsupervised anomaly detectors on the per-reading feature table, and a comparison.

There is no ground truth for "anomalous" in this data set, so both detectors are reported
side by side and the comparison is descriptive: which readings each one flags, how much
they agree, and what those readings look like. The sensible use of the output is as a
list of moments worth looking at, not as a verdict.

Detector 1 -- rolling z-score of the temperature slope, per heater state.
    Inside a heater-on run the chamber should warm at roughly the same rate as it did in
    recent heater-on readings; inside a heater-off run it should cool (or drift) like recent
    heater-off readings. The z-score is computed against a rolling mean/std of the slope
    over the previous ``window`` readings *of the same heater state*, so switching the
    heater does not itself trigger it.

Detector 2 -- IsolationForest on the whole feature table.
    Multivariate: it can flag a reading whose slope is normal but whose combination of
    (chamber temperature, ambient, humidity, setpoint error, time since switch) is unusual.
    ``contamination`` sets the fraction of readings it will label.
    :class:`ReferenceIsolationForest` is the same model fitted on a known-good day and
    applied to new data.

Detector 3 -- :func:`model_residual`, added in session 2: drift of the measured
    temperature away from what the fitted heater-lag model predicts from the logged relay
    states. The only one of the three that knows what the heater *should* be doing.

Scoring against injected faults lives in :mod:`incubator.faults`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from incubator.features import FEATURE_COLUMNS
from incubator.thermal_fit import SecondOrderFit


def rolling_zscore(
    features: pd.DataFrame,
    column: str = "temp_rate",
    window: int = 40,
    min_periods: int = 10,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Flag readings whose ``column`` is more than ``threshold`` rolling sigmas from the
    rolling mean of the same heater state. Returns ``z`` and ``flag`` columns, aligned to
    ``features``. The first ``min_periods`` readings of each state get NaN z (not flagged).
    """
    z = pd.Series(np.nan, index=features.index, name="z")
    for _, part in features.groupby("heater", sort=False):
        x = part[column]
        prev = x.shift(1)  # exclude the current reading from its own baseline
        mean = prev.rolling(window, min_periods=min_periods).mean()
        std = prev.rolling(window, min_periods=min_periods).std().clip(lower=1e-3)
        z.loc[part.index] = (x - mean) / std
    out = pd.DataFrame({"z": z})
    out["flag"] = out["z"].abs() > threshold
    return out


def isolation_forest(
    features: pd.DataFrame,
    contamination: float = 0.03,
    random_state: int = 0,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """IsolationForest on standardised feature columns. Returns ``score`` (higher = more
    anomalous, the negated sklearn decision function) and ``flag``.
    """
    cols = columns or FEATURE_COLUMNS
    x = features[cols].to_numpy(dtype=float)
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)
    model = IsolationForest(
        n_estimators=300, contamination=contamination, random_state=random_state
    )
    model.fit(x)
    score = -model.decision_function(x)
    flag = model.predict(x) == -1
    return pd.DataFrame({"score": score, "flag": flag}, index=features.index)


@dataclass(frozen=True)
class Comparison:
    n_a: int
    n_b: int
    n_both: int
    jaccard: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n_a": self.n_a,
            "n_b": self.n_b,
            "n_both": self.n_both,
            "jaccard": round(self.jaccard, 3),
        }


def compare(flag_a: pd.Series, flag_b: pd.Series) -> Comparison:
    """How much two boolean flag series (same index) agree."""
    a = flag_a.astype(bool)
    b = flag_b.astype(bool).reindex(a.index, fill_value=False)
    both = int((a & b).sum())
    union = int((a | b).sum())
    return Comparison(int(a.sum()), int(b.sum()), both, both / union if union else 1.0)


def flagged_episodes(flag: pd.Series, max_gap: str = "2min") -> pd.DataFrame:
    """Group consecutive flagged readings into episodes (start, end, readings).

    Two flagged readings belong to the same episode when they are at most ``max_gap``
    apart. This turns hundreds of per-reading flags into a short list a human can review.
    """
    times = flag.index[flag.astype(bool).to_numpy()]
    if len(times) == 0:
        return pd.DataFrame(columns=["start", "end", "readings", "minutes"])
    gap = pd.Timedelta(max_gap)
    new_episode = np.r_[True, np.diff(times.to_numpy()) > gap.to_timedelta64()]
    ids = np.cumsum(new_episode)
    rows = []
    for k in np.unique(ids):
        ep = times[ids == k]
        rows.append(
            {
                "start": ep[0],
                "end": ep[-1],
                "readings": int(len(ep)),
                "minutes": float((ep[-1] - ep[0]).total_seconds() / 60.0),
            }
        )
    return pd.DataFrame(rows)


class ReferenceIsolationForest:
    """IsolationForest fitted once on a known-good trace and applied to new data.

    The plain :func:`isolation_forest` is refitted on whatever it is given and always
    labels its ``contamination`` fraction, so on a trace with a real fault it still spends
    most of its budget on the most unusual *normal* period (here: the cold start). The
    deployment-shaped alternative is to fit on a reference day and flag new readings whose
    score exceeds the ``quantile`` of the reference scores. Same model, different question:
    "is this unlike the day we know was fine?"
    """

    def __init__(
        self,
        quantile: float = 0.97,
        n_estimators: int = 300,
        random_state: int = 0,
        columns: list[str] | None = None,
    ) -> None:
        self.quantile = quantile
        self.columns = columns or FEATURE_COLUMNS
        self.model = IsolationForest(n_estimators=n_estimators, random_state=random_state)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.threshold_: float = float("nan")

    def _standardise(self, features: pd.DataFrame) -> np.ndarray:
        x = features[self.columns].to_numpy(dtype=float)
        assert self.mean_ is not None and self.std_ is not None
        return (x - self.mean_) / self.std_

    def fit(self, features: pd.DataFrame) -> ReferenceIsolationForest:
        x = features[self.columns].to_numpy(dtype=float)
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0) + 1e-9
        self.model.fit(self._standardise(features))
        ref = -self.model.score_samples(self._standardise(features))
        self.threshold_ = float(np.quantile(ref, self.quantile))
        return self

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        """``score`` (higher = more anomalous) and ``flag`` (score above the reference
        quantile) for new readings."""
        s = -self.model.score_samples(self._standardise(features))
        return pd.DataFrame({"score": s, "flag": s > self.threshold_}, index=features.index)


def model_residual(
    df: pd.DataFrame,
    fit: SecondOrderFit | tuple[float, float, float],
    window: str = "5min",
    threshold_c: float = 0.3,
) -> pd.DataFrame:
    """Detector 3 -- drift of the measured temperature away from the fitted thermal model.

    The heater-lag model's delivered-heat state ``E`` depends only on the logged relay
    history, so it can be reconstructed exactly from the telemetry. From it and the
    measured temperature the model predicts the next 15-second temperature step; the
    residual (measured minus predicted step) is summed over a rolling ``window`` to give the
    *drift*: how far the chamber has moved from what the physics says it should have done
    over the last few minutes. ``|drift| > threshold_c`` is flagged.

    This is the detector that knows what the heater is supposed to do, so a relay that is on
    while logged off, or an open door, shows up as drift even when the slope itself looks
    ordinary. It needs a fitted model (from a known-good day) and it inherits the model's
    structural errors. Returns ``drift`` (°C) and ``flag``.
    """
    if isinstance(fit, SecondOrderFit):
        a, tau_h, b = fit.a_c_per_min, fit.tau_h_min, fit.b_per_min
    else:
        a, tau_h, b = fit
    minutes = (df.index - df.index[0]).total_seconds().to_numpy() / 60.0
    dt = np.diff(minutes)
    u = df["heater"].to_numpy(dtype=float)
    temp = df["temp"].to_numpy(dtype=float)
    t_out = df["temp_out"].to_numpy(dtype=float)
    flow = np.zeros(len(df))
    for k in range(1, len(df)):
        target = a * u[k - 1]
        flow[k] = target + (flow[k - 1] - target) * np.exp(-dt[k - 1] / tau_h)
    predicted_step = dt * (0.5 * (flow[:-1] + flow[1:]) - b * (temp[:-1] - t_out[:-1]))
    residual = np.r_[0.0, np.diff(temp) - predicted_step]
    drift = pd.Series(residual, index=df.index).rolling(window).sum()
    out = pd.DataFrame({"drift": drift})
    out["flag"] = out["drift"].abs() > threshold_c
    return out
