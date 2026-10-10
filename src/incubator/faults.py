"""Inject synthetic faults into the real trace and score the anomaly detectors on them.

The day in ``data/`` contains no known fault, so the detectors in :mod:`incubator.anomaly`
cannot be scored on it directly. This module adds faults of known kind, time and length to
the real readings and measures what each detector finds. Six kinds are supported:

``stuck_sensor``     the temperature (and humidity) sensor repeats its last value.
``spike``            one reading is off by ``magnitude`` °C (a glitch).
``heater_stuck_on``  the relay stays closed although the firmware logs it as open.
``heater_stuck_off`` the relay stays open although the firmware logs it as closed.
``door_open``        the loss to ambient is multiplied by ``magnitude`` (a step in ``b``).
``dropout``          no readings arrive for the fault window.

The three physical faults change the chamber temperature in a way the real trace cannot
show, so their effect is *synthesised* with the fitted second-order model and added to the
measured temperature: the model is linear, so the deviation caused by a change of input
(or of loss coefficient) can be simulated on its own and does not depend on the unknown
initial heater state. Measured readings therefore keep their real noise and the fault adds
exactly what the model predicts. This is the honest limit of what one real day allows; the
README says so.

Labels: ``fault`` marks readings inside the fault window; ``affected`` adds a settle margin
after it (the chamber takes minutes to recover, and a stuck sensor's jump back to reality
is a symptom of the fault, not a false alarm). Reading-level precision/recall use ``fault``
as truth and ignore flags in the margin; a fault counts as *detected* when any reading in
``affected`` is flagged. For ``dropout`` nothing can be flagged inside the gap, so only the
first reading after it carries the label.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from incubator import anomaly as an
from incubator import features as fe
from incubator.thermal_fit import SecondOrderFit

FAULT_KINDS = (
    "stuck_sensor",
    "spike",
    "heater_stuck_on",
    "heater_stuck_off",
    "door_open",
    "dropout",
)

#: (duration, magnitude) used by :func:`make_fault` when nothing else is given.
DEFAULTS: dict[str, tuple[pd.Timedelta, float]] = {
    "stuck_sensor": (pd.Timedelta(minutes=15), 0.0),
    "spike": (pd.Timedelta(seconds=0), 1.5),
    "heater_stuck_on": (pd.Timedelta(minutes=10), 0.0),
    "heater_stuck_off": (pd.Timedelta(minutes=20), 0.0),
    "door_open": (pd.Timedelta(minutes=15), 4.0),
    "dropout": (pd.Timedelta(minutes=10), 0.0),
}

SETTLE_MARGIN = pd.Timedelta(minutes=5)
#: Centred-difference features leak two readings backwards, so flags just before the fault
#: window are a symptom too, not a false alarm.
PRE_MARGIN = pd.Timedelta(minutes=1)

Detector = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class Fault:
    kind: str
    start: pd.Timestamp
    duration: pd.Timedelta
    magnitude: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in FAULT_KINDS:
            raise ValueError(f"unknown fault kind {self.kind!r}; choose from {FAULT_KINDS}")

    @property
    def end(self) -> pd.Timestamp:
        return self.start + self.duration


def make_fault(kind: str, start: pd.Timestamp, magnitude: float | None = None) -> Fault:
    """A fault of ``kind`` with the default duration (and magnitude unless given)."""
    duration, default_mag = DEFAULTS[kind]
    return Fault(kind, start, duration, default_mag if magnitude is None else magnitude)


def _deviation(
    df: pd.DataFrame, fault: Fault, model: SecondOrderFit | tuple[float, float, float]
) -> np.ndarray:
    """Temperature deviation (°C per reading) that the model predicts for a physical fault.

    Simulates only the *difference* from the measured trace: with ``du`` the change in relay
    state and ``m`` the loss multiplier inside the window,

        dDE/dt = (a*du - DE) / tau_h
        dD/dt  = DE - b_eff*D - (m - 1)*b*(T_meas - T_out)*[inside window]

    where ``b_eff = m*b`` inside the window and ``b`` outside.
    """
    if isinstance(model, SecondOrderFit):
        a, tau_h, b = model.a_c_per_min, model.tau_h_min, model.b_per_min
    else:
        a, tau_h, b = model
    minutes = (df.index - df.index[0]).total_seconds().to_numpy() / 60.0
    dt = np.diff(minutes)
    inside = np.asarray((df.index >= fault.start) & (df.index <= fault.end))
    u = df["heater"].to_numpy(dtype=float)
    temp = df["temp"].to_numpy(dtype=float)
    t_out = df["temp_out"].to_numpy(dtype=float)
    du = np.zeros(len(df))
    mult = np.ones(len(df))
    if fault.kind == "heater_stuck_on":
        du[inside] = 1.0 - u[inside]
    elif fault.kind == "heater_stuck_off":
        du[inside] = -u[inside]
    elif fault.kind == "door_open":
        mult[inside] = fault.magnitude
    dev = np.zeros(len(df))
    d, de = 0.0, 0.0
    for k in range(1, len(df)):
        step = dt[k - 1]
        forcing = (mult[k - 1] - 1.0) * b * (temp[k - 1] - t_out[k - 1])
        target = a * du[k - 1]
        de_next = target + (de - target) * np.exp(-step / tau_h)
        d = d + step * (0.5 * (de + de_next) - mult[k - 1] * b * d - forcing)
        de = de_next
        dev[k] = d
    return dev


def inject(
    df: pd.DataFrame,
    fault: Fault,
    model: SecondOrderFit | tuple[float, float, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(faulted history frame, labels)``.

    The faulted frame has the same schema as the input (``dt_s`` recomputed). ``labels`` is
    aligned to the faulted frame with boolean columns ``fault`` and ``affected``.
    ``model`` is required for the three physical faults.
    """
    out = df.copy()
    inside = (out.index >= fault.start) & (out.index <= fault.end)
    if fault.kind in ("heater_stuck_on", "heater_stuck_off", "door_open"):
        if model is None:
            raise ValueError(f"{fault.kind} needs a fitted thermal model")
        out["temp"] = np.round(out["temp"].to_numpy() + _deviation(df, fault, model), 2)
    elif fault.kind == "stuck_sensor":
        if not inside.any():
            raise ValueError("fault window contains no readings")
        first = out.index[inside][0]
        out.loc[inside, "temp"] = out.at[first, "temp"]
        out.loc[inside, "hum"] = out.at[first, "hum"]
    elif fault.kind == "spike":
        pos = out.index.searchsorted(fault.start)
        pos = min(pos, len(out) - 1)
        out.iloc[pos, out.columns.get_loc("temp")] += fault.magnitude
        inside = np.zeros(len(out), dtype=bool)
        inside[pos] = True
        inside = pd.Series(inside, index=out.index)
    elif fault.kind == "dropout":
        out = out.loc[~inside]
        after = out.index.searchsorted(fault.end)
        inside = pd.Series(np.zeros(len(out), dtype=bool), index=out.index)
        if after < len(out):
            inside.iloc[after] = True
    out["dt_s"] = out.index.to_series().diff().dt.total_seconds()
    inside = pd.Series(np.asarray(inside, dtype=bool), index=out.index)
    affected = inside | (
        (out.index >= fault.start - PRE_MARGIN) & (out.index <= fault.end + SETTLE_MARGIN)
    )
    labels = pd.DataFrame({"fault": inside, "affected": affected}, index=out.index)
    return out, labels


# ----------------------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------------------


def zscore_detector(df: pd.DataFrame) -> pd.Series:
    """The session-1 rolling z-score detector as a ``history frame -> flags`` callable."""
    ft = fe.feature_table(df)
    return an.rolling_zscore(ft)["flag"].reindex(df.index, fill_value=False)


def isoforest_detector(df: pd.DataFrame) -> pd.Series:
    """The session-1 IsolationForest detector (3 % contamination) as a callable."""
    ft = fe.feature_table(df)
    return an.isolation_forest(ft)["flag"].reindex(df.index, fill_value=False)


def reference_isoforest_detector(reference: pd.DataFrame) -> Detector:
    """IsolationForest fitted on the clean ``reference`` day, applied to the faulted trace."""
    model = an.ReferenceIsolationForest().fit(fe.feature_table(reference))

    def detect(df: pd.DataFrame) -> pd.Series:
        return model.score(fe.feature_table(df))["flag"].reindex(df.index, fill_value=False)

    return detect


def model_residual_detector(
    reference: pd.DataFrame,
    model: SecondOrderFit | tuple[float, float, float],
    quantile: float = 0.997,
) -> Detector:
    """Model-drift detector with its threshold calibrated on the clean ``reference`` day:
    the ``quantile`` of |drift| there (0.997 flags about 4 of 1 367 clean readings)."""
    ref = an.model_residual(reference, model, threshold_c=np.inf)["drift"].abs().dropna()
    threshold = float(np.quantile(ref, quantile))

    def detect(df: pd.DataFrame) -> pd.Series:
        return an.model_residual(df, model, threshold_c=threshold)["flag"].reindex(
            df.index, fill_value=False
        )

    detect.threshold_c = threshold  # type: ignore[attr-defined]
    return detect


def default_detectors(
    reference: pd.DataFrame, model: SecondOrderFit | tuple[float, float, float]
) -> dict[str, Detector]:
    """The four detectors compared in the README, keyed by their short names."""
    return {
        "zscore": zscore_detector,
        "isoforest": isoforest_detector,
        "isoforest_ref": reference_isoforest_detector(reference),
        "model_drift": model_residual_detector(reference, model),
    }


def score(flag: pd.Series, labels: pd.DataFrame) -> dict[str, float | int | bool]:
    """Reading-level counts for one injected fault plus the event-level verdict."""
    f = flag.reindex(labels.index, fill_value=False).astype(bool)
    truth = labels["fault"].astype(bool)
    affected = labels["affected"].astype(bool)
    tp = int((f & truth).sum())
    fn = int((~f & truth).sum())
    fp = int((f & ~affected).sum())
    fp_episodes = len(an.flagged_episodes(f & ~affected))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "fp_episodes": int(fp_episodes),
        "detected": bool((f & affected).any()),
        "flags_total": int(f.sum()),
    }


def random_starts(
    df: pd.DataFrame, duration: pd.Timedelta, n: int, seed: int = 0
) -> list[pd.Timestamp]:
    """``n`` start times, uniform over readings, such that the whole window and the settle
    margin fit inside the trace and the first 5 min (detector warm-up) are avoided."""
    rng = np.random.default_rng(seed)
    lo = df.index[0] + pd.Timedelta(minutes=5)
    hi = df.index[-1] - duration - SETTLE_MARGIN
    candidates = df.index[(df.index >= lo) & (df.index <= hi)]
    if len(candidates) == 0:
        raise ValueError("trace too short for this fault")
    picks = rng.choice(len(candidates), size=n, replace=len(candidates) < n)
    return [candidates[i] for i in sorted(picks)]


def run_trials(
    df: pd.DataFrame,
    model: SecondOrderFit | tuple[float, float, float],
    kinds: tuple[str, ...] = FAULT_KINDS,
    detectors: dict[str, Detector] | None = None,
    n_trials: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    """Inject each fault kind ``n_trials`` times at random positions and score each detector.

    One row per (kind, trial, detector). ``detected_null`` is the same detector's verdict
    for the same window on the *clean* trace: the chance that a window of that length is
    "detected" by the detector's ordinary false alarms. A detection rate is only meaningful
    against that floor. The plain IsolationForest is refitted on each faulted trace, as it
    would be in practice (it is unsupervised and sees the fault too).
    """
    detectors = detectors or default_detectors(df, model)
    clean_flags = {name: det(df) for name, det in detectors.items()}
    rows = []
    for kind in kinds:
        duration, _ = DEFAULTS[kind]
        for trial, start in enumerate(random_starts(df, duration, n_trials, seed)):
            fault = make_fault(kind, start)
            faulted, labels = inject(df, fault, model)
            for name, det in detectors.items():
                null = clean_flags[name].reindex(labels.index, fill_value=False)
                rows.append(
                    {
                        "kind": kind,
                        "trial": trial,
                        "start_utc": start,
                        "detector": name,
                        **score(det(faulted), labels),
                        "detected_null": bool((null & labels["affected"]).any()),
                    }
                )
    return pd.DataFrame(rows)


def summarise(trials: pd.DataFrame) -> pd.DataFrame:
    """Per (kind, detector): detection rate, pooled precision/recall, false alarms per trial.

    Precision pools tp and fp over the trials (so a kind with few labelled readings is not
    dominated by one trial); ``fp_episodes_per_trial`` is the mean number of false-alarm
    episodes outside the fault, which is the number a user would actually suffer.
    """
    g = trials.groupby(["kind", "detector"], sort=False)
    out = pd.DataFrame(
        {
            "trials": g.size(),
            "detection_rate": g["detected"].mean(),
            "null_detection_rate": g["detected_null"].mean(),
            "precision": g["tp"].sum() / (g["tp"].sum() + g["fp"].sum()).replace(0, np.nan),
            "recall": g["tp"].sum() / (g["tp"].sum() + g["fn"].sum()).replace(0, np.nan),
            "fp_episodes_per_trial": g["fp_episodes"].mean(),
        }
    )
    return out.reset_index()
