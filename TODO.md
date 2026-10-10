# TODO (next session on this repo)

Status after session 1 (2026-10-10): package, tests, figures, notebooks, README and
WALKTHROUGH are in place and `make check` passes. Spec items 1–7 of PLAN.md #1 are done at
first-order level. What remains, in priority order:

1. **Second-order thermal model** in `thermal_fit.py`: heater element temperature `H` with
   its own `C_h`, `R_h` feeding the chamber. Fit it the same output-error way, compare RMSE
   and AIC with the first-order fit, and check whether it removes the 3–4 min dead time and
   reproduces the 1.4–3.4 °C post-switch rise. Export both parameter sets to
   `figures/model_params.json` for the control-simulation repo (PLAN.md #13).
2. **Scored anomaly comparison**: inject synthetic faults into the real trace (sensor stuck,
   heater relay stuck on/off, door open = step increase in loss, dropout) with a generator
   in `src/incubator/faults.py`, and report precision/recall per detector in
   `02_anomalies.ipynb` and the README. Keep the "no real faults in this day" statement.
3. **Humidity**: absolute humidity from `temp`/`hum`, inside vs outside, during heat-up;
   one figure and a paragraph (see WALKTHROUGH extension).
4. **Data**: the `heater_stress_log` in the repo is 2 h, not the 72 h endurance run the plan
   describes. If Mariam exports the full log or more history days, add them under `data/`
   (keep under 10 MB each), extend `load.py` to load several history files, and re-run.
5. **CI**: `.github/workflows/check.yml` running `make check` on Python 3.11/3.12.
6. Small: `heater_segments` could take a `lag_horizon` argument instead of the fixed 2 min;
   the `CLEAN_WINDOWS` constant in `scripts/make_figures.py` could be derived from the
   segment table instead of hand-typed.
