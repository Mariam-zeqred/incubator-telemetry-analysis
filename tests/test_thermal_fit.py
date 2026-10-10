import numpy as np

from incubator import thermal_fit as t
from tests.conftest import synthetic_trace


def test_fit_recovers_parameters_on_synthetic_trace():
    df = synthetic_trace(a=0.4, b=0.008, dead_time=3, noise_c=0.01)
    fit = t.fit_first_order(df, max_dead_time=6)
    assert fit.dead_time_readings == 3
    assert abs(fit.a_c_per_min - 0.4) < 0.02
    assert abs(fit.b_per_min - 0.008) < 0.0008
    assert fit.rmse_c < 0.05
    assert fit.r2 > 0.99


def test_fit_on_windows_and_physical():
    df = synthetic_trace(a=0.3, b=0.01, dead_time=0, n=800)
    windows = [("2026-01-01 00:00:00", "2026-01-01 01:30:00")]
    fit = t.fit_first_order(df, windows=windows, max_dead_time=2)
    assert fit.n < len(df)
    assert abs(fit.a_c_per_min - 0.3) < 0.02
    phys = fit.physical(heater_power_w=50.0)
    # C = P / a  with a in °C/s ; R = 1 / (b C)
    assert np.isclose(phys["C_j_per_c"], 50.0 / (fit.a_c_per_min / 60.0))
    assert np.isclose(phys["R_c_per_w"] * phys["C_j_per_c"], 60.0 / fit.b_per_min)
    assert np.isclose(fit.full_power_rise_c, fit.a_c_per_min / fit.b_per_min)


def test_simulate_matches_generator():
    df = synthetic_trace(a=0.4, b=0.008, dead_time=2, n=300)
    heater, t_out = df["heater"].to_numpy(), df["temp_out"].to_numpy()
    sim = t.simulate((0.4, 0.008, 2), heater, t_out, 26.0, 0.25)
    assert np.abs(sim - df["temp"].to_numpy()).max() < 0.01


def test_real_data_fit_is_sane(history):
    fit = t.fit_first_order(history, max_dead_time=16)
    assert 0.2 < fit.a_c_per_min < 1.0
    assert 30 < fit.tau_min < 300
    assert fit.r2 > 0.95
