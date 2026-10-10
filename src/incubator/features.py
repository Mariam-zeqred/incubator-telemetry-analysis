"""Feature extraction from the tidy history frame produced by :mod:`incubator.load`.

Everything here works on the index (tz-aware UTC timestamps) plus the columns ``temp``,
``temp_out``, ``target``, ``heater`` and ``dt_s``. Rates are in °C per minute.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEGMENT_COLUMNS = [
    "heater",
    "start",
    "end",
    "n",
    "duration_min",
    "temp_start",
    "temp_end",
    "temp_out_mean",
    "target",
    "rate_c_per_min",
    "post_switch_rise",
]


def heater_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Split the trace into maximal runs of constant heater state.

    One row per run with its duration, start/end temperatures, the mean outside
    temperature, the setpoint at the start, a least-squares temperature slope over the run
    (°C/min, NaN for runs shorter than 3 readings) and ``post_switch_rise``: how much the
    temperature kept rising *after* the run ended (max over the next 2 min minus the
    temperature at the switch). For heater-on runs this is the thermal lag of the heater
    and sensor, which is what makes a plain on/off controller overshoot.
    """
    seg_id = (df["heater"] != df["heater"].shift()).cumsum()
    rows: list[dict[str, object]] = []
    times = df.index
    for _, part in df.groupby(seg_id, sort=True):
        start, end = part.index[0], part.index[-1]
        minutes = (part.index - start).total_seconds() / 60.0
        slope = np.nan
        if len(part) >= 3:
            slope = float(np.polyfit(minutes, part["temp"].to_numpy(), 1)[0])
        after = df.loc[(times > end) & (times <= end + pd.Timedelta(minutes=2)), "temp"]
        rise = float(after.max() - part["temp"].iloc[-1]) if len(after) else np.nan
        rows.append(
            {
                "heater": bool(part["heater"].iloc[0]),
                "start": start,
                "end": end,
                "n": int(len(part)),
                "duration_min": float(minutes[-1]),
                "temp_start": float(part["temp"].iloc[0]),
                "temp_end": float(part["temp"].iloc[-1]),
                "temp_out_mean": float(part["temp_out"].mean()),
                "target": float(part["target"].iloc[0]),
                "rate_c_per_min": slope,
                "post_switch_rise": rise,
            }
        )
    out = pd.DataFrame(rows, columns=SEGMENT_COLUMNS)
    out.index.name = "segment"
    return out


def duty_cycle(df: pd.DataFrame, window: str = "10min") -> pd.Series:
    """Fraction of each wall-clock window during which the heater was on.

    Each reading is weighted by the time until the next reading, so a 74 s gap counts for
    74 s, not 15 s. Windows with no readings are dropped.
    """
    hold_s = df["dt_s"].shift(-1).fillna(df["dt_s"].median())
    on_s = hold_s.where(df["heater"], 0.0)
    grouped_on = on_s.resample(window).sum()
    grouped_all = hold_s.resample(window).sum()
    dc = (grouped_on / grouped_all).dropna()
    dc.name = "duty_cycle"
    return dc


def heating_rates(segments: pd.DataFrame, min_duration_min: float = 2.0) -> pd.DataFrame:
    """Heat-up (heater on) and cool-down (heater off) slopes for runs long enough to trust.

    Short runs (the 1-minute heater pulses in steady state) are excluded because a slope
    over four noisy readings means nothing. Returns the filtered segment rows.
    """
    ok = segments["duration_min"] >= min_duration_min
    return segments.loc[ok & segments["rate_c_per_min"].notna()].copy()


def setpoint_changes(df: pd.DataFrame, include_initial: bool = True) -> pd.DataFrame:
    """Every time ``target`` changed: the time, old and new setpoint, and the temperature then.

    With ``include_initial`` the first reading counts as a change with ``old_target`` NaN,
    so the initial approach to the setpoint is analysed like any other step.
    """
    changed = df["target"].ne(df["target"].shift())
    if not include_initial:
        changed &= df["target"].shift().notna()
    rows = df.loc[changed]
    return pd.DataFrame(
        {
            "time": rows.index,
            "old_target": df["target"].shift().loc[changed].to_numpy(),
            "new_target": rows["target"].to_numpy(),
            "temp_at_change": rows["temp"].to_numpy(),
        }
    ).reset_index(drop=True)


def time_to_target(df: pd.DataFrame, tolerance_c: float = 0.2) -> pd.DataFrame:
    """For each setpoint change, minutes until the temperature first came within tolerance.

    The search is limited to the period before the next setpoint change; if the target was
    never reached in that period ``minutes`` is NaN and ``reached`` is False. Also reports
    the maximum temperature in that period relative to the new target (``overshoot_c``,
    positive means the chamber went above the setpoint; only meaningful for upward steps).
    """
    changes = setpoint_changes(df)
    bounds = list(changes["time"]) + [df.index[-1] + pd.Timedelta(seconds=1)]
    rows: list[dict[str, object]] = []
    for i, change in changes.iterrows():
        t0, t1 = bounds[i], bounds[i + 1]
        period = df.loc[(df.index >= t0) & (df.index < t1)]
        target = float(change["new_target"])
        upward = target > float(change["temp_at_change"])
        if upward:
            hit = period["temp"] >= target - tolerance_c
        else:
            hit = period["temp"] <= target + tolerance_c
        reached = bool(hit.any())
        minutes = float((hit.idxmax() - t0).total_seconds() / 60.0) if reached else np.nan
        rows.append(
            {
                "time": t0,
                "old_target": float(change["old_target"]),
                "new_target": target,
                "upward": upward,
                "reached": reached,
                "minutes": minutes,
                "overshoot_c": float(period["temp"].max() - target) if upward else np.nan,
                "period_min": float((period.index[-1] - t0).total_seconds() / 60.0),
            }
        )
    return pd.DataFrame(rows)


def inside_outside_delta(df: pd.DataFrame) -> pd.Series:
    """Chamber minus ambient temperature, °C."""
    delta = df["temp"] - df["temp_out"]
    delta.name = "delta_in_out"
    return delta


def temp_rate(df: pd.DataFrame, span: int = 2) -> pd.Series:
    """Local temperature slope in °C/min from a centred difference over ``span`` readings.

    The sensor reports to 0.01 °C at 15 s, so a one-step difference is dominated by
    quantisation (one LSB step is 0.04 °C/min). Using ±2 readings (60 s) tames that.
    """
    minutes = (df.index - df.index[0]).total_seconds().to_numpy() / 60.0
    temp = df["temp"].to_numpy()
    rate = np.full(len(df), np.nan)
    rate[span:-span] = (temp[2 * span :] - temp[: -2 * span]) / (
        minutes[2 * span :] - minutes[: -2 * span]
    )
    return pd.Series(rate, index=df.index, name="temp_rate")


def minutes_since_switch(df: pd.DataFrame) -> pd.Series:
    """Minutes since the heater last changed state (0 at each switch)."""
    switched = df["heater"].ne(df["heater"].shift())
    last_switch = df.index.to_series().where(switched).ffill()
    out = (df.index.to_series() - last_switch).dt.total_seconds() / 60.0
    out.name = "min_since_switch"
    return out


FEATURE_COLUMNS = [
    "temp",
    "hum",
    "temp_out",
    "delta_in_out",
    "target_error",
    "temp_rate",
    "hum_rate",
    "heater",
    "min_since_switch",
]


def feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-reading features for the anomaly detectors (one row per reading, no NaNs).

    The first and last readings, where the centred slope is undefined, are dropped.
    """
    feats = pd.DataFrame(index=df.index)
    feats["temp"] = df["temp"]
    feats["hum"] = df["hum"]
    feats["temp_out"] = df["temp_out"]
    feats["delta_in_out"] = inside_outside_delta(df)
    feats["target_error"] = df["temp"] - df["target"]
    feats["temp_rate"] = temp_rate(df)
    hum_df = df.assign(temp=df["hum"])
    feats["hum_rate"] = temp_rate(hum_df).to_numpy()
    feats["heater"] = df["heater"].astype(int)
    feats["min_since_switch"] = minutes_since_switch(df)
    return feats[FEATURE_COLUMNS].dropna()
