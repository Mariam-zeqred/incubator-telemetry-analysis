# Walkthrough (for Mariam)

## Why it is built this way

The repo answers one question with real data from your own incubator: *how does the chamber
heat, cool and lag, and what does that mean for the controller and for spotting faults?* It
is a small package, not one notebook, so the README numbers, the figures and the tests all
call the same functions; the README quotes only `figures/results.json`, which the figure
script writes, so anyone who re-runs `make check` gets exactly the README's numbers. CI runs
the same command on every push.

Two decisions matter. The thermal fits use **simulation error, not slope regression**: the
sensor reports 0.01 °C steps every 15 s, so per-reading slopes are quantisation noise, while
integrating over the 92-minute cool-down pins the time constant down. And the detectors are
scored on **injected faults**, because the real day has none; the physical ones are
synthesised with the fitted model and the README says so.

## File map

- `src/incubator/load.py` – raw JSON/CSV to tidy UTC frames (schema in the docstring).
- `src/incubator/features.py` – `heater_segments`, `duty_cycle`, `time_to_target` (also
  overshoot), `feature_table` for the detectors, `absolute_humidity` / `humidity_table`.
- `src/incubator/thermal_fit.py` – `fit_first_order` (a, b, dead time),
  `fit_second_order` (a, τh, b: the heater-lag model), `compare_models` (RMSE, AIC),
  `fit_two_mass` + `two_mass_profile` (the four-parameter trial), `cooldown_loss_rate`.
- `src/incubator/anomaly.py` – `rolling_zscore`, `isolation_forest`,
  `ReferenceIsolationForest` (fit on a good day, apply to a new one), `model_residual`
  (drift from the physics), `interval_watchdog` (gaps), `flagged_episodes`.
- `src/incubator/faults.py` – `inject` (six fault kinds, labels), `run_trials`, `summarise`.
- `scripts/make_figures.py` – ten PNGs, `results.json`, `model_params.json`.
- `tests/` – synthetic generators in `conftest.py` prove each fit recovers known parameters;
  the rest pin real-data facts and the fault-scoring mechanics.

## Three questions a professor will probably ask

**"Why a second-order model, and why those three parameters?"**
Because the first-order fit needed a 3.5-minute dead time and still missed the post-switch
peaks by 0.4 °C; letting the delivered heat lag the relay with τh ≈ 5 min removes the dead
time and halves the error. I kept three parameters because the fourth
cannot be identified from this data, not because it is zero: `fit_two_mass` adds the heater
body's share of the heat capacity, and figure 10 shows it only trades against the chamber
loss `b` (the error changes under 2 % from φ = −0.2 to 0.4 while the chamber time constant
slides from 109 to 52 min); left free, it runs into the mirror solution with the time
constants swapped, which I reject because the 92-minute cool-down measures the chamber loss
directly.

**"Your best detector uses the model you fitted on the same day. Isn't that circular?"**
Partly, and the README says the threshold was calibrated on this day. The defence: the model
is three physical numbers, not a fit to the faults, which were injected afterwards without
tuning. The proper test is a second day of data, item 1 in `TODO.md`.

**"What do the precision/recall numbers actually mean here?"**
Per reading, pooled over 20 random positions, with a chance floor next to them: the z-score
"detects" a 15-minute window on the clean day 40 % of the time. Summary: the drift
detector separates relay and door faults from normal operation and the others do not; a
stuck sensor is weak for everyone; dropouts need no model, the interval watchdog catches all.

## One thing you could extend

Turn `a` and `b` into real R, C and P: measure the heater's electrical power once (a plug
meter will do) and call `FirstOrderFit.physical(heater_power_w)`. That gives the chamber's
heat capacity and insulation in physical units, which nobody else has for this device.
