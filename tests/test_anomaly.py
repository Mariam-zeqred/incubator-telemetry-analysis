import numpy as np
import pandas as pd

from incubator import anomaly as a
from incubator import features as f


def test_rolling_zscore_flags_an_injected_spike(history):
    ft = f.feature_table(history)
    base = a.rolling_zscore(ft)
    spiked = ft.copy()
    k = 900  # inside the long steady-state stretch
    spiked.iloc[k, spiked.columns.get_loc("temp_rate")] = 5.0
    out = a.rolling_zscore(spiked)
    assert bool(out["flag"].iloc[k]) and not bool(base["flag"].iloc[k])
    assert out["flag"].sum() <= base["flag"].sum() + 2


def test_isolation_forest_contamination(history):
    ft = f.feature_table(history)
    out = a.isolation_forest(ft, contamination=0.03)
    assert abs(out["flag"].mean() - 0.03) < 0.01
    assert out["score"].loc[out["flag"]].min() >= out["score"].loc[~out["flag"]].max() - 1e-9


def test_compare_and_episodes():
    idx = pd.date_range("2026-01-01", periods=10, freq="15s", tz="UTC")
    x = pd.Series([1, 1, 0, 0, 0, 0, 0, 0, 1, 0], index=idx, dtype=bool)
    y = pd.Series([1, 0, 0, 0, 0, 0, 0, 0, 1, 1], index=idx, dtype=bool)
    c = a.compare(x, y)
    assert (c.n_a, c.n_b, c.n_both) == (3, 3, 2)
    assert np.isclose(c.jaccard, 0.5)
    ep = a.flagged_episodes(x, max_gap="1min")
    assert ep["readings"].tolist() == [2, 1]
    assert a.flagged_episodes(x & False).empty
