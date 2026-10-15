import numpy as np
import pytest

from incubator import thermal_fit as t
from tests.conftest import synthetic_trace, synthetic_trace_second_order


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


def test_second_order_recovers_parameters_on_synthetic_trace():
    df = synthetic_trace_second_order(a=0.5, tau_h=5.0, b=0.01, noise_c=0.01)
    fit = t.fit_second_order(df)
    assert fit.dead_time_readings == 0 and fit.physical_branch
    assert abs(fit.a_c_per_min - 0.5) < 0.02
    assert abs(fit.tau_h_min - 5.0) < 0.3
    assert abs(fit.b_per_min - 0.01) < 0.001
    assert fit.rmse_c < 0.05
    assert fit.time_constants_min[0] < fit.time_constants_min[1]
    assert np.isclose(fit.full_power_rise_c, fit.a_c_per_min / fit.b_per_min)


def test_second_order_simulate_matches_generator():
    df = synthetic_trace_second_order(a=0.5, tau_h=5.0, b=0.01, n=400)
    sim = t.simulate_second_order(
        (0.5, 5.0, 0.01), df["heater"].to_numpy(), df["temp_out"].to_numpy(), 26.0, 0.25
    )
    assert np.abs(sim - df["temp"].to_numpy()).max() < 0.01


def test_second_order_beats_first_order_on_real_data(history):
    first = t.fit_first_order(history, max_dead_time=16)
    second = t.fit_second_order(history, max_dead_time=2)
    assert second.dead_time_readings == 0  # the lag model needs no dead time
    assert second.rmse_c < 0.7 * first.rmse_c
    assert 2 < second.tau_h_min < 10
    assert 60 < second.tau_min < 150
    cmp = t.compare_models(first, second)
    assert cmp["delta_aic"] < 0
    # the two branches share the heater response; only the ambient path differs
    other = t.fit_second_order(history, physical=False)
    assert not other.physical_branch
    assert abs(other.full_power_rise_c - second.full_power_rise_c) < 3.0


def test_compare_models_requires_same_readings():
    df = synthetic_trace(n=400)
    first = t.fit_first_order(df, max_dead_time=1)
    second = t.fit_second_order(df.iloc[:300])
    with pytest.raises(ValueError):
        t.compare_models(first, second)


def test_two_mass_with_phi_zero_is_the_heater_lag_model():
    df = synthetic_trace_second_order(a=0.5, tau_h=5.0, b=0.01, n=300)
    heater, t_out = df["heater"].to_numpy(dtype=float), df["temp_out"].to_numpy()
    dt = np.full(len(df) - 1, 0.25)
    sim3 = t.simulate_second_order((0.5, 5.0, 0.01), heater, t_out, 26.0, 0.25)
    sim4 = t._simulate_two_mass(0.5, 5.0, 0.01, 0.0, heater, t_out, 26.0, 0.0, dt)
    assert np.array_equal(sim3, sim4)


def test_two_mass_recovers_phi_zero_on_a_three_parameter_trace():
    df = synthetic_trace_second_order(a=0.5, tau_h=5.0, b=0.01, noise_c=0.01)
    fit = t.fit_two_mass(df)
    assert not fit.b_fixed
    assert abs(fit.phi) < 2 * fit.phi_se and fit.phi_se < 0.05
    assert not fit.phi_is_distinguishable_from_zero
    assert abs(fit.a_c_per_min - 0.5) < 0.02 and abs(fit.b_per_min - 0.01) < 0.001
    pinned = t.fit_two_mass(df, fix_b=0.01)
    assert pinned.b_fixed and pinned.b_per_min == 0.01 and pinned.b_se == 0.0
    assert abs(pinned.phi) < 0.05


def test_two_mass_is_not_determined_on_the_real_day(history):
    second = t.fit_second_order(history)
    free = t.fit_two_mass(history)
    # the free fit leaves the physical branch: b at its bound, large phi, small RMSE gain
    assert free.b_per_min > 0.09 and free.phi > 0.5
    cmp = t.compare_two_mass(second, free)
    assert 0.8 < cmp["rmse_ratio"] < 1.0
    profile = t.two_mass_profile(history, phis=(-0.2, 0.0, 0.4))
    rmses = [r["rmse_c"] for r in profile]
    assert max(rmses) - min(rmses) < 0.03 * min(rmses)  # a ridge, not a minimum
    assert profile[0]["tau_min"] > 1.5 * profile[-1]["tau_min"]  # phi trades against b
    with pytest.raises(ValueError):
        t.compare_two_mass(second, t.fit_two_mass(history.iloc[:400]))


def test_cooldown_loss_rate_is_model_free_and_sane(history):
    cd = t.cooldown_loss_rate(history, "2026-10-03 10:16:36", "2026-10-03 11:48:56")
    assert cd["n"] > 300 and cd["slope_c_per_min"] < 0
    assert 0.007 < cd["b_per_min"] < 0.013
    with pytest.raises(ValueError):
        t.cooldown_loss_rate(history, "2026-10-03 10:02:00", "2026-10-03 10:05:00")
