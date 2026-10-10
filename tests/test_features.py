import numpy as np

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
