import numpy as np
import pandas as pd
import pytest

from incubator.load import load_history, load_stress_log


@pytest.fixture(scope="session")
def history() -> pd.DataFrame:
    return load_history()


@pytest.fixture(scope="session")
def stress() -> pd.DataFrame:
    return load_stress_log()


def synthetic_trace(
    a: float = 0.4,
    b: float = 0.008,
    dead_time: int = 3,
    n: int = 1200,
    dt_s: float = 15.0,
    noise_c: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """A history-shaped frame generated from the first-order model with a bang-bang controller.

    Used by the thermal-fit test: the fit must recover ``a``, ``b`` and ``dead_time``.
    """
    rng = np.random.default_rng(seed)
    times = pd.date_range("2026-01-01", periods=n, freq=f"{int(dt_s)}s", tz="UTC")
    temp_out = 27.0 + 0.3 * np.sin(np.arange(n) / 300.0)
    target = np.where(np.arange(n) < n // 2, 32.0, 36.0)
    temp = np.empty(n)
    heater = np.zeros(n, dtype=bool)
    temp[0] = 26.0
    for k in range(1, n):
        heater[k - 1] = temp[k - 1] < target[k - 1] - 0.1
        u = heater[max(k - 1 - dead_time, 0)]
        temp[k] = temp[k - 1] + (dt_s / 60.0) * (a * u - b * (temp[k - 1] - temp_out[k - 1]))
    heater[-1] = heater[-2]
    measured = np.round(temp + rng.normal(0.0, noise_c, n), 2)
    df = pd.DataFrame(
        {
            "temp": measured,
            "hum": 60.0,
            "temp_out": temp_out,
            "hum_out": 50.0,
            "target": target,
            "heater": heater,
            "door_open": False,
            "cycle_day": 1,
            "cycle_code": "SYNTH",
            "elapsed_s": np.arange(n) * int(dt_s),
        },
        index=pd.Index(times, name="time"),
    )
    df["dt_s"] = df.index.to_series().diff().dt.total_seconds()
    return df
