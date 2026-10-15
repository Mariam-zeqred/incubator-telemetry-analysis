# TODO (next session on this repo)

Status after session 3 (2026-10-15): every item of the PLAN.md #1 spec is done, and the two
cloud-doable items left by sessions 1–2 are closed. The four-parameter two-mass trial is now
code (`thermal_fit.fit_two_mass`, `two_mass_profile`, `cooldown_loss_rate`, figure 10,
`results.json` → `two_mass_trial`), and running it changed the claim: the extra parameter is
**not identifiable** on this day (it trades against `b` along a flat ridge and the free fit
runs into the mirror branch), rather than "indistinguishable from zero" as the old docstring
and WALKTHROUGH said. The sampling-interval watchdog (`anomaly.interval_watchdog`) is the
fifth scored detector. `make check` passes with 35 tests and ten figures. What is left needs
data from Mariam or belongs to other repos:

1. **A second day of data.** Everything in the README rests on 3 Oct 2026. When Mariam
   exports another history day (or the full 72-hour heater log), put it under `data/`
   (under 10 MB), extend `load_history` to accept several files, re-fit the models on one
   day and score the drift detector on the other. That answers the "circular" question in
   WALKTHROUGH.md properly. A day with a larger ambient swing (or a logged door opening)
   would also be the only way to pin the two-mass model's fourth parameter.
2. **Heater power.** One measurement of the heater's electrical power turns `a`, `b` into
   R, C, P (`FirstOrderFit.physical`). Record it in `data/README` or `model_params.json`.
3. Optional: the IsolationForest table rows could be revisited with the cold start removed
   from the reference day; expected to help little, and it is documented as a finding now.
4. `CLEAN_WINDOWS` and `COOLDOWN_WINDOW` in `scripts/make_figures.py` stay hand-typed on
   purpose (the comments in the script say why); fine to leave.

## Open
- `data/heater_stress_log_2026-10-10.csv` is 33 rows (2 h), not the 72 h log PLAN.md
  describes; README and item 1 already say so. Needs the real export from Mariam.
