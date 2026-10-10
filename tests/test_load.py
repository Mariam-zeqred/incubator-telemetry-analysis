import pandas as pd

from incubator.load import HISTORY_COLUMNS, summary


def test_history_schema(history):
    assert list(history.columns) == HISTORY_COLUMNS
    assert history.index.name == "time"
    assert str(history.index.tz) == "UTC"
    assert history.index.is_monotonic_increasing and history.index.is_unique
    assert history["heater"].dtype == bool and history["door_open"].dtype == bool
    assert history["temp"].between(20, 45).all()


def test_history_facts(history):
    facts = summary(history)
    assert facts["readings"] == 1367
    assert facts["median_interval_s"] == 15.0
    assert facts["setpoints"] == [30.0, 32.0, 34.5, 38.8]
    assert history.index[0] == pd.Timestamp("2026-10-03 10:01:51", tz="UTC")


def test_stress_log_schema(stress):
    assert len(stress) == 33
    assert str(stress.index.tz) == "UTC"
    assert stress["heater"].dtype == bool and stress["fan"].dtype == bool
    assert set(stress["phase"]) == {"HEAT", "PEAK", "COOL"}
    # 14:07:38 local (UTC+3) -> 11:07:38 UTC
    assert stress.index[0] == pd.Timestamp("2026-10-10 11:07:38", tz="UTC")
    assert stress["reset"].iloc[0] == "none"
