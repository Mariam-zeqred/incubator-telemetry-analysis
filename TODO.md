# TODO (next session on this repo)

Status after session 2 (2026-10-10): every item of the PLAN.md #1 spec is done. Second-order
model, scored detectors (with a model-drift detector added), humidity, CI and the small
refactors from session 1's list are all in; `make check` passes with 29 tests and nine
figures. What is left is data-bound or belongs to other repos:

1. **A second day of data.** Everything in the README rests on 3 Oct 2026. When Mariam
   exports another history day (or the full 72-hour heater log), put it under `data/`
   (under 10 MB), extend `load_history` to accept several files, re-fit the models on one
   day and score the drift detector on the other. That answers the "circular" question in
   WALKTHROUGH.md properly. Until then nothing more can honestly be claimed.
2. **Heater power.** One measurement of the heater's electrical power turns `a`, `b` into
   R, C, P (`FirstOrderFit.physical`). Record it in `data/README` or `model_params.json`.
3. **Sampling-interval watchdog**: a trivial rule (`dt_s > 60`) in `anomaly.py` so that
   dropouts are caught on purpose rather than as a side effect of the drift detector; add it
   to `faults.default_detectors` and the table.
4. Optional: the IsolationForest table rows could be revisited with the cold start removed
   from the reference day; expected to help little, and it is documented as a finding now.
5. `CLEAN_WINDOWS` in `scripts/make_figures.py` stays hand-typed on purpose (documented in
   the WALKTHROUGH as the one hand-chosen input); fine to leave.
