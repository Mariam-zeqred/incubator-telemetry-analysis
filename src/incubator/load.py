"""Load the two raw telemetry files into tidy, UTC-aware DataFrames.

Both files were exported from one real queen-bee incubator (device id ``hive-1C5A00``).

History file (``data/hive-1C5A00_history_*.json``)
    A JSON list of readings, one every ~15 s, as stored by the incubator's cloud history.
    Raw fields: ``ts`` (epoch seconds, UTC), ``iso`` (device local time, UTC+3, no offset
    in the string), ``temp`` / ``hum`` (inside, °C / %RH), ``tempOut`` / ``humOut``
    (outside the chamber), ``targetTemp`` (setpoint, °C), ``heater`` (relay on/off),
    ``door`` ("shut"/"open"), ``cycleDay``, ``cycleCode``, ``elapsedSec`` (seconds since
    the incubation cycle started).

Stress log (``data/heater_stress_log_*.csv``)
    A CSV written by a bench script that polled the device every ~2 min during a heater
    test, with the device's phase (HEAT / PEAK / COOL), run state, watchdog / crash
    counters and the action the script took on each poll. ``time_local`` is device local
    time (UTC+3). Note: the file in this repo covers about two hours, not the full test.

Tidy schema returned by :func:`load_history`
    index ``time`` (tz-aware UTC, sorted, unique), columns:
    ``temp``, ``hum``, ``temp_out``, ``hum_out``, ``target`` (float, °C / %RH),
    ``heater`` (bool), ``door_open`` (bool), ``cycle_day`` (int), ``cycle_code`` (str),
    ``elapsed_s`` (int), ``dt_s`` (float, seconds since the previous reading; NaN first).

Tidy schema returned by :func:`load_stress_log`
    index ``time`` (tz-aware UTC), columns ``phase``, ``temp``, ``hum``, ``temp_out``,
    ``target``, ``heater`` (bool), ``fan`` (bool), ``alarm``, ``run_state``,
    ``crash_count`` (int), ``reset`` (str), ``heater_on_ms`` (int), ``uptime_ms`` (int),
    ``action`` (str).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
HISTORY_FILE = DATA_DIR / "hive-1C5A00_history_2026-10-03.json"
STRESS_LOG_FILE = DATA_DIR / "heater_stress_log_2026-10-10.csv"

#: The device reports ``iso`` in Egypt local time without an offset. In the history file the
#: difference ``iso - ts`` is exactly +3 h for every row, so we use a fixed offset here.
DEVICE_UTC_OFFSET = pd.Timedelta(hours=3)

_HISTORY_RENAME = {
    "tempOut": "temp_out",
    "humOut": "hum_out",
    "targetTemp": "target",
    "cycleDay": "cycle_day",
    "cycleCode": "cycle_code",
    "elapsedSec": "elapsed_s",
}

_STRESS_RENAME = {
    "tempOut": "temp_out",
    "targetTemp": "target",
    "runState": "run_state",
    "crashCount": "crash_count",
    "heaterOnMs": "heater_on_ms",
    "uptimeMs": "uptime_ms",
}

HISTORY_COLUMNS = [
    "temp",
    "hum",
    "temp_out",
    "hum_out",
    "target",
    "heater",
    "door_open",
    "cycle_day",
    "cycle_code",
    "elapsed_s",
    "dt_s",
]


def load_history(path: str | Path = HISTORY_FILE) -> pd.DataFrame:
    """Read the JSON history export into the tidy schema documented in this module.

    Readings are sorted by time and exact-duplicate timestamps are dropped (none exist in
    the shipped file, but the cloud export can repeat a row when the app re-syncs).
    """
    with Path(path).open(encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: expected a non-empty JSON list of readings")

    df = pd.DataFrame(raw).rename(columns=_HISTORY_RENAME)
    missing = {"ts", "temp", "heater", "target"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing fields {sorted(missing)}")

    df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df["door_open"] = df["door"].astype(str).str.lower().ne("shut")
    df["heater"] = df["heater"].astype(bool)
    for col in ("temp", "hum", "temp_out", "hum_out", "target"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["cycle_day"] = df["cycle_day"].astype(int)
    df["elapsed_s"] = df["elapsed_s"].astype(int)
    df["cycle_code"] = df["cycle_code"].astype(str)

    df = (
        df.sort_values("time")
        .drop_duplicates(subset="time", keep="first")
        .set_index("time")
    )
    df["dt_s"] = df.index.to_series().diff().dt.total_seconds()
    return df[HISTORY_COLUMNS]


def load_stress_log(path: str | Path = STRESS_LOG_FILE) -> pd.DataFrame:
    """Read the heater stress-test CSV into a tidy, UTC-indexed DataFrame."""
    df = pd.read_csv(path).rename(columns=_STRESS_RENAME)
    local = pd.to_datetime(df["time_local"])
    df["time"] = (local - DEVICE_UTC_OFFSET).dt.tz_localize("UTC")
    for col in ("heater", "fan"):
        df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False})
        if df[col].isna().any():
            raise ValueError(f"{path}: column {col!r} has values other than True/False")
        df[col] = df[col].astype(bool)
    for col in ("temp", "hum", "temp_out", "target"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["reset"] = df["reset"].fillna("none").astype(str)
    df = df.sort_values("time").set_index("time")
    return df[
        [
            "phase",
            "temp",
            "hum",
            "temp_out",
            "target",
            "heater",
            "fan",
            "alarm",
            "run_state",
            "crash_count",
            "reset",
            "heater_on_ms",
            "uptime_ms",
            "action",
        ]
    ]


def summary(df: pd.DataFrame) -> dict[str, object]:
    """A few facts about a history frame, used by the README/figure script and tests."""
    return {
        "readings": int(len(df)),
        "start_utc": df.index[0].isoformat(),
        "end_utc": df.index[-1].isoformat(),
        "duration_h": round((df.index[-1] - df.index[0]).total_seconds() / 3600, 2),
        "median_interval_s": float(df["dt_s"].median()),
        "max_gap_s": float(df["dt_s"].max()),
        "heater_on_fraction": float(df["heater"].mean()),
        "setpoints": sorted(df["target"].unique().tolist()),
    }
