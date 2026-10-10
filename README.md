# incubator-telemetry-analysis

Analysis of real telemetry from one queen-bee incubator (the device whose firmware and apps
are in [queen-bee-incubator](https://github.com/Mariam-zeqred/queen-bee-incubator)). This
repo turns a day of 15-second readings into heater statistics, a side-by-side comparison of
two anomaly detectors, and a fitted first-order thermal model whose parameters are meant to
drive a control-simulation repo. Every number below is produced by `scripts/make_figures.py`
and stored in `figures/results.json`.

```
data/*.json, *.csv ──► load.py ──► tidy UTC frame ──► features.py ──► segments, duty cycle,
                                        │                               rates, overshoot
                                        ├──► anomaly.py ──► z-score vs IsolationForest flags
                                        └──► thermal_fit.py ──► a, b, dead time (τ = 1/b)
                                                        │
                       scripts/make_figures.py ◄────────┘ ──► figures/*.png + results.json
```

## Data

| File | What it is |
|---|---|
| `data/hive-1C5A00_history_2026-10-03.json` | 1 367 readings at 15 s, 3 Oct 2026 10:02–15:45 UTC: chamber and ambient temperature/humidity, setpoint, heater relay, door. Includes a hands-on session where the setpoint was stepped 32 → 34.5 → 38.8 → 30 °C. |
| `data/heater_stress_log_2026-10-10.csv` | A bench script's 2-minute polls during a heater test on 10 Oct 2026. **Only 33 rows (2 h) are in the repo**, not the full endurance run; it is used for one cross-check and the device-state columns (`runState`, `crashCount`, `reset`, `action`). |

Timestamps: `ts` is epoch UTC; the device's `iso` and the CSV's `time_local` are Egypt local
time (UTC+3), which the loader converts.

## Run it

```bash
git clone https://github.com/Mariam-zeqred/incubator-telemetry-analysis && cd incubator-telemetry-analysis
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
make check          # ruff + pytest (15 tests) + regenerate figures/ and results.json
make notebooks      # rebuild the two exploration notebooks (outputs cleared)
jupyter lab notebooks/01_explore.ipynb
```

## Results

### Timeline and duty cycle
![timeline](figures/01_timeline.png)
![duty cycle](figures/02_duty_cycle.png)

The heater was on 19 % of the time overall. In the steady stretch at 38.8 °C (13:35–15:39 UTC)
it fired 19 short pulses (about one per 6.5 min, duty cycle 19 %) and held the chamber at
38.65 °C mean, 38.57–38.77 °C range, 10.4 °C above ambient.

### Heat-up, cool-down and thermal lag
![rates](figures/03_rates.png)

| Quantity | Value (from the data) |
|---|---|
| Heat-up slope, heater runs ≥ 2 min (5 runs) | 0.15 – 0.41 °C/min; 0.35 °C/min from ambient |
| Heat-up slope in the 10 Oct test log (31.5 → 38.2 °C) | 0.35 °C/min, consistent with 3 Oct |
| Longest cool-down (92 min, heater off, ~4 °C above ambient) | −0.035 °C/min, 31.92 → 30.22 °C |
| Rise *after* switch-off following the three long runs (7.5–10 min) | +1.4, +1.9, +3.4 °C, peaking 9–16 min after OFF |
| Time to target / overshoot: 32 °C (from cold) | 14.8 min / **+1.42 °C** |
| Time to target / overshoot: 34.5 °C | 9.0 min / 0.00 °C |
| Time to target / overshoot: 38.8 °C | 24.8 min / −0.03 °C |

The lag is the main story of this data set: once the heater has run for several minutes the
chamber keeps warming for a quarter of an hour. The firmware's long first run from cold
produced the 1.4 °C overshoot; the later steps, driven by short pulses, did not overshoot.

### Anomaly detectors, side by side
![anomalies](figures/04_anomalies.png)

There is no labelled fault in this day, so the comparison is descriptive. The rolling
z-score of the temperature slope (per heater state, window 40, |z| > 3) flags 17 readings in
6 short episodes, all at heater switches during the big setpoint steps. The IsolationForest
(3 % contamination) flags 41 readings, 38 of them the initial warm-up from a cold chamber,
the most unusual *state* in the day. They agree on 3 readings (Jaccard 0.05). Lesson: the two
detectors answer different questions (a sudden change vs an unusual condition), and
neither finds a "fault" because the data contains none that we know of.

### First-order thermal model
![model fit](figures/05_model_fit.png)

Model: `dT/dt = a·u(t − d) − b·(T − T_out)`, with `a = P/C`, `b = 1/(RC)`, fitted by
simulating the model and minimising the temperature error (not by regressing on noisy
15-second differences).

| Fit | a (°C/min) | b (1/min) | τ = RC (min) | dead time | RMSE open-loop |
|---|---|---|---|---|---|
| Whole trace (1 367 readings) | 0.554 ± 0.002 | 0.0101 ± 0.0001 | 99 ± 1 | 3.5 min | 0.29 °C over 5.7 h |
| Two clean windows (582 readings) | 0.534 ± 0.002 | 0.0085 ± 0.0001 | 118 ± 2 | 4.0 min | 0.26 °C in-window |

The ± values are least-squares standard errors and are optimistic: residuals are
autocorrelated and the model is wrong in a structured way right after each switch. Only
`a` and `b` are identifiable from temperatures; `R`, `C` and `P` separately need the heater's
electrical power (`FirstOrderFit.physical(heater_power_w)` does that conversion). The
full-power steady-state rise `a/b` is about 55 °C above ambient.

### Heater test log (10 Oct)
![stress log](figures/06_stress_log.png)

The device was `PAUSED` on firmware 1.1.5 when the script started, was auto-resumed, then
ran one heat-up to 38.75 °C and a scripted cool-down with the setpoint lowered 0.1–0.2 °C
every 2 min. `crashCount` went 6 → 7 with reset reason "other watchdog" and alarm `W01`
raised throughout the run. Two hours is too short to say anything about endurance.

## Findings the data supports
- Full-power heating from ambient is about 0.35 °C/min on both days.
- Cooling near 4 °C above ambient is about 0.035 °C/min; the fitted time constant is 1.5–2 h.
- The dominant control problem is lag: 3–4 min of dead time and up to 3.4 °C of post-switch
  rise after a long heater run, which produced a 1.4 °C overshoot on the first approach.
- Once near the setpoint the firmware's short pulses held a 0.2 °C band, slightly below target.

## Limitations
One device, one day of history plus two hours of test log; one ambient (27–28 °C); no door
openings (the door was shut throughout); no labelled faults, so the anomaly comparison has
no precision/recall. The first-order model ignores the heater's own thermal mass, which is
what the dead time and the post-switch rise are standing in for.

## Relation to the published work
The incubator and its control approach are described in "Design and Development of a Smart
Incubator for Bee Queens Based on IoT Technology", *Journal of Computer Science and
Engineering Research* 3(1), June 2026, DOI
[10.64820/AEPJCSER.31.13.22.62026](https://doi.org/10.64820/AEPJCSER.31.13.22.62026). This
repo adds a quantitative look at the production device's heating behaviour that the paper
did not include. The fitted `a`, `b` and dead time are the input for a companion
control-simulation repo (planned) that compares on/off, PID and predictive control on this
model.

## Next steps
See `TODO.md`: a second-order (heater element + chamber) model to explain the lag, injected
faults to score the detectors, a humidity analysis, and more days of data when available.

## Layout
`src/incubator/` package · `tests/` pytest · `scripts/` figure and notebook builders ·
`notebooks/` exploration · `figures/` outputs · `data/` raw exports · MIT licence.
