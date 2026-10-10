import numpy as np
import pandas as pd

from incubator import features as f
from incubator.features import FEATURE_COLUMNS, SEGMENT_COLUMNS


def test_segments_cover_trace(history):
    seg = f.heater_segments(history)
    assert list(seg.columns) == SEGMENT_COLUMNS
    assert seg["n"].sum() == len(history)
    # consecutive runs alternate heater state
    assert (seg["heater"].iloc[1:].to_numpy() != seg["heater"].iloc[:-1].to_numpy()).all()
    assert seg["start"].is_monotonic_increasing


def test_rates_have_the_right_sign(history):
    rates = f.heating_rates(f.heater_segments(history))
    on = rates.loc[rates["heater"], "rate_c_per_min"]
    assert len(on) >= 3 and (on > 0.1).all()
    long_off = rates.loc[~rates["heater"] & (rates["duration_min"] > 60), "rate_c_per_min"]
    assert len(long_off) == 1 and float(long_off.iloc[0]) < 0


def test_duty_cycle_bounds(history):
    dc = f.duty_cycle(history, "10min")
    assert dc.between(0, 1).all()
    assert 30 <= len(dc) <= 36
    assert abs(dc.mean() - history["heater"].mean()) < 0.05


def test_setpoint_changes_and_time_to_target(history):
    ch = f.setpoint_changes(history)
    assert ch["new_target"].tolist() == [32.0, 34.5, 38.8, 30.0]
    assert np.isnan(ch["old_target"].iloc[0])
    ttt = f.time_to_target(history)
    assert len(ttt) == 4
    assert ttt["reached"].tolist() == [True, True, True, False]
    assert (ttt.loc[ttt["reached"], "minutes"] > 0).all()


def test_feature_table(history):
    ft = f.feature_table(history)
    assert list(ft.columns) == FEATURE_COLUMNS
    assert len(ft) == len(history) - 4
    assert not ft.isna().any().any()
    assert ft["min_since_switch"].min() == 0.0


def test_absolute_humidity_known_values():
    # 20 °C saturated air holds about 17.3 g/m³; 30 °C at 50 % about 15.2 g/m³
    assert abs(f.absolute_humidity(np.array([20.0]), np.array([100.0]))[0] - 17.3) < 0.1
    assert abs(f.absolute_humidity(np.array([30.0]), np.array([50.0]))[0] - 15.2) < 0.1
    s = f.absolute_humidity(pd.Series([25.0, 25.0]), pd.Series([40.0, 80.0]))
    assert isinstance(s, pd.Series) and np.isclose(s.iloc[1], 2 * s.iloc[0])


def test_humidity_table(history):
    h = f.humidity_table(history)
    assert len(h) == len(history) and not h.isna().any().any()
    assert (h["ah_in"] > 0).all() and (h["ah_in"] < 40).all()
    # the real-data fact: RH inside stays in a narrow band while the chamber warms 12 °C,
    # so the vapour density inside *rises* with temperature instead of staying constant
    assert h["rh_in"].between(65, 85).all()
    assert h["ah_in"].corr(history["temp"]) > 0.9
    assert h["ah_in"].iloc[-100:].mean() > h["ah_in"].iloc[:100].mean() + 5
    assert h["ah_out"].std() < 1.0


def test_lag_horizon_argument(history):
    short = f.heater_segments(history, lag_horizon="2min")
    long = f.heater_segments(history, lag_horizon="20min")
    on = short["heater"] & short["post_switch_rise"].notna()
    assert (long.loc[on, "post_switch_rise"] >= short.loc[on, "post_switch_rise"] - 1e-9).all()
    assert long.loc[on, "post_switch_rise"].max() > 3.0
