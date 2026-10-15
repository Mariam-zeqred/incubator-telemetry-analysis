import numpy as np
import pandas as pd
import pytest

from incubator import anomaly as an
from incubator import faults as fl
from incubator import thermal_fit as t

MODEL = (0.59, 5.2, 0.011)  # a, tau_h, b close to the real-data fit


@pytest.fixture(scope="module")
def start(history) -> pd.Timestamp:
    return pd.Timestamp("2026-10-03 14:00:00", tz="UTC")


def test_every_kind_injects_with_labels(history, start):
    for kind in fl.FAULT_KINDS:
        out, labels = fl.inject(history, fl.make_fault(kind, start), MODEL)
        assert list(out.columns) == list(history.columns)
        assert labels.index.equals(out.index)
        assert labels["fault"].any() and (labels["affected"] | ~labels["fault"]).all()
        assert out["dt_s"].iloc[1:].notna().all()


def test_stuck_sensor_and_spike_shapes(history, start):
    out, labels = fl.inject(history, fl.make_fault("stuck_sensor", start))
    assert out.loc[labels["fault"], "temp"].nunique() == 1
    assert labels["fault"].sum() == 60  # 15 min at 15 s
    out, labels = fl.inject(history, fl.make_fault("spike", start, magnitude=2.0))
    assert labels["fault"].sum() == 1
    k = int(np.flatnonzero(labels["fault"].to_numpy())[0])
    assert np.isclose(out["temp"].iloc[k] - history["temp"].iloc[k], 2.0)
    assert (out["temp"].drop(out.index[k]) == history["temp"].drop(history.index[k])).all()


def test_physical_faults_push_the_right_way(history, start):
    on, lab = fl.inject(history, fl.make_fault("heater_stuck_on", start), MODEL)
    dev_on = (on["temp"] - history["temp"]).loc[lab["affected"]]
    assert dev_on.max() > 1.0 and (dev_on >= -1e-9).all()
    door, lab = fl.inject(history, fl.make_fault("door_open", start), MODEL)
    dev_door = (door["temp"] - history["temp"]).loc[lab["affected"]]
    assert dev_door.min() < -1.0 and (dev_door <= 1e-9).all()
    with pytest.raises(ValueError):
        fl.inject(history, fl.make_fault("door_open", start))


def test_dropout_removes_readings(history, start):
    out, labels = fl.inject(history, fl.make_fault("dropout", start))
    assert 30 <= len(history) - len(out) <= 45
    assert labels["fault"].sum() == 1
    assert out["dt_s"].max() >= 600


def test_score_counts():
    idx = pd.date_range("2026-01-01", periods=8, freq="15s", tz="UTC")
    labels = pd.DataFrame(
        {
            "fault": [False, False, True, True, False, False, False, False],
            "affected": [False, True, True, True, True, False, False, False],
        },
        index=idx,
    )
    flag = pd.Series([True, False, True, False, True, False, False, True], index=idx)
    s = fl.score(flag, labels)
    assert (s["tp"], s["fn"], s["fp"], s["fp_episodes"], s["detected"]) == (
        1,
        1,
        2,
        1,
        True,
    )  # 0 and 7 are 105 s apart: one episode
    assert fl.score(flag & False, labels)["detected"] is False


def test_model_residual_is_quiet_on_the_clean_day_and_loud_on_a_fault(history, start):
    fit = t.fit_second_order(history)
    det = fl.model_residual_detector(history, fit)
    assert det(history).sum() <= 8
    faulted, labels = fl.inject(history, fl.make_fault("heater_stuck_on", start), fit)
    assert fl.score(det(faulted), labels)["detected"]
    drift = an.model_residual(history, fit)["drift"]
    assert drift.abs().max() < 0.5


def test_run_trials_and_summarise(history):
    detectors = {
        "zscore": fl.zscore_detector,
        "model_drift": fl.model_residual_detector(history, MODEL),
    }
    trials = fl.run_trials(
        history, MODEL, kinds=("spike", "door_open"), detectors=detectors, n_trials=2, seed=1
    )
    assert len(trials) == 2 * 2 * 2
    summary = fl.summarise(trials)
    assert set(summary["kind"]) == {"spike", "door_open"}
    assert summary["detection_rate"].between(0, 1).all()
    assert summary.loc[summary["detector"] == "model_drift", "detection_rate"].min() > 0.5


def test_interval_detector_catches_dropout_and_nothing_else(history, start):
    det = fl.default_detectors(history, MODEL)
    assert "interval" in det and det["interval"](history).sum() == 1
    for kind in fl.FAULT_KINDS:
        faulted, labels = fl.inject(history, fl.make_fault(kind, start), MODEL)
        s = fl.score(det["interval"](faulted), labels)
        if kind == "dropout":
            assert s["detected"] and s["tp"] == 1 and s["fn"] == 0
        else:
            assert s["flags_total"] == 1  # only the clean day's own 74 s gap
