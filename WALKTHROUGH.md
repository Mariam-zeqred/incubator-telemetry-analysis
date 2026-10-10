# Walkthrough (for Mariam)

## Why it is built this way

The repo answers one question with real data from your own incubator: *how does the chamber
actually heat, cool and lag, and what does that mean for the controller?* Everything else is
in service of that. The code is a small package instead of one notebook so that the README
numbers, the figures and the tests all call the same functions; `figures/results.json` is
written by the figure script and the README quotes only from it, so a professor who re-runs
`make check` gets exactly the numbers in the README.

The thermal fit uses **simulation error, not slope regression**. The sensor reports
0.01 °C steps every 15 s, so a per-reading slope is mostly quantisation noise; integrating the
model over the 92-minute cool-down and comparing temperatures is what pins down the time
constant. That decision is explained in the `thermal_fit.py` docstring and is a good thing to
be able to say out loud.

## File map

- `src/incubator/load.py` – raw JSON/CSV to tidy, UTC-indexed frames. The schema is in the
  module docstring. Note the UTC+3 handling.
- `src/incubator/features.py` – `heater_segments` (runs of constant heater state),
  `duty_cycle`, `heating_rates`, `setpoint_changes`, `time_to_target` (also overshoot),
  `feature_table` (one row per reading for the detectors).
- `src/incubator/anomaly.py` – `rolling_zscore` (per heater state), `isolation_forest`,
  `compare`, `flagged_episodes`.
- `src/incubator/thermal_fit.py` – `fit_first_order` (grid over dead time, least squares
  on `a`, `b`), `simulate`, `FirstOrderFit.physical` for R and C when the heater power is known.
- `scripts/make_figures.py` – the six PNGs and `results.json`; `CLEAN_WINDOWS` there is the
  one hand-chosen input in the repo (the first heat-up/cool-down and the 38.8 °C step).
- `scripts/build_notebooks.py` – generates the two notebooks so they never hold stale outputs.
- `tests/` – the synthetic trace in `conftest.py` is the proof that the fit recovers known
  parameters; the other tests pin the real-data facts (1 367 readings, four setpoints).

## Three questions a professor will probably ask

**"Your model has three parameters R, C, P. How did you get all three from temperature?"**
I did not. Temperature alone identifies `a = P/C` and `b = 1/(RC)`; the time constant
`RC = 1/b ≈ 100 min` and the full-power rise `PR = a/b ≈ 55 °C` are what the data gives.
Separating them needs the heater's electrical power, which `physical()` takes as an input.
Saying this plainly is better than quoting three numbers.

**"Why is the dead time 3–4 minutes? Air heats instantly."**
The heater element and the enclosure store heat; after a 10-minute run the chamber kept
rising for a quarter of an hour. A first-order model has no state for that, so the fit
absorbs it as a delay. The honest fix is a second-order model (element + chamber), which is
the first item in `TODO.md` and the reason the control-simulation repo exists.

**"Which anomaly detector is better?"**
Neither, because the day contains no known fault to score against. The z-score reacts to
sudden slope changes (heater switches during big steps); the IsolationForest flags the
unusual *state* (a cold chamber with a large setpoint error). They overlap on 3 readings.
The next step is injected faults (stuck sensor, heater stuck on, door open) to get
precision/recall; without that, any claim of "better" would be made up.

## One thing you could extend

Add humidity. `hum` drops from 81 % to 45 % as the chamber warms, which is the expected
relative-humidity effect of heating at constant moisture. Fitting absolute humidity (from
`temp` and `hum`) would show whether the chamber is actually drying out, which matters for
queen cells and would be a genuinely new result from your own data.
