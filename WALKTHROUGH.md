# Walkthrough (for Mariam)

## Why it is built this way

The repo answers one question with real data from your own incubator: *how does the chamber
actually heat, cool and lag, and what does that mean for the controller and for spotting
faults?* The code is a small package instead of one notebook so that the README numbers, the
figures and the tests all call the same functions; `figures/results.json` is written by the
figure script and the README quotes only from it, so a professor who re-runs `make check`
gets exactly the numbers in the README. CI runs the same command on every push.

Two decisions are worth saying out loud. Both thermal fits use **simulation error, not
slope regression**: the sensor reports 0.01 °C steps every 15 s, so per-reading slopes are
mostly quantisation noise, while integrating over the 92-minute cool-down pins the time
constant down. And the detectors are scored on **injected faults**, because the real day has
none; the physical ones are synthesised with the fitted model and the README says so.

## File map

- `src/incubator/load.py` – raw JSON/CSV to tidy, UTC-indexed frames (schema in the docstring).
- `src/incubator/features.py` – `heater_segments`, `duty_cycle`, `time_to_target` (also
  overshoot), `feature_table` for the detectors, `absolute_humidity` / `humidity_table`.
- `src/incubator/thermal_fit.py` – `fit_first_order` (a, b, dead time) and
  `fit_second_order` (a, τh, b: the heater-lag model), `compare_models` (RMSE, AIC).
- `src/incubator/anomaly.py` – `rolling_zscore`, `isolation_forest`,
  `ReferenceIsolationForest` (fit on a good day, apply to a new one), `model_residual`
  (drift from the physics), `flagged_episodes`.
- `src/incubator/faults.py` – `inject` (six fault kinds, labels), `run_trials`, `summarise`.
- `scripts/make_figures.py` – nine PNGs, `results.json`, `model_params.json`.
- `tests/` – the two synthetic generators in `conftest.py` prove each fit recovers known
  parameters; the rest pin real-data facts and the fault-scoring mechanics.

## Three questions a professor will probably ask

**"Why a second-order model, and why those three parameters?"**
Because the first-order fit needed a 3.5-minute dead time and still missed the peaks after
switch-off by 0.4 °C. Letting the delivered heat lag the relay with a time constant τh ≈ 5 min
removes the dead time and halves the error. I kept three parameters because that is what one
temperature sensor can identify: the full four-parameter heater-body model was tried and its
extra parameter came out indistinguishable from zero. There is also a mirror solution with
the time constants swapped; I rejected it because the 92-minute cool-down measures the
chamber loss directly.

**"Your best detector uses the model you fitted on the same day. Isn't that circular?"**
Partly, and the README says the threshold was calibrated on this day. The defence is that
the model is three physical numbers, not a fit to the faults: the faults were injected
afterwards and the detector was not tuned on them. The proper test is a second day of data,
which is the first item in `TODO.md`.

**"What do the precision/recall numbers actually mean here?"**
They are per reading, pooled over 20 random positions, with a chance floor next to them: a
15-minute window on the clean day is "detected" by the z-score 40 % of the time anyway. The honest summary is that the
drift detector separates relay and door faults from normal operation and the others do not;
the sensor-stuck case is weak for everyone and needs a different signal.

## One thing you could extend

Turn `a` and `b` into real R, C and P: measure the heater's electrical power once (a plug
meter will do), call `FirstOrderFit.physical(heater_power_w)`, and you have the chamber's
heat capacity and insulation in physical units, which is a number nobody else has for this
device and a direct input to the control-simulation repo.
