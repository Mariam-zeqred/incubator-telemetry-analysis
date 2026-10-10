"""Write the two exploration notebooks (outputs cleared) from plain Python cell lists.

Keeping the notebook source here means the notebooks are reproducible, diff-able and never
carry stale outputs. Run ``python scripts/build_notebooks.py`` after editing the cells.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

SETUP = """import sys
sys.path.insert(0, "../src")
import pandas as pd
import matplotlib.pyplot as plt
from incubator.load import load_history, load_stress_log, summary
from incubator import features as fe, anomaly as an, thermal_fit as tf
pd.set_option("display.width", 160)"""

EXPLORE = [
    (
        "md",
        "# 01 · Explore the incubator history\n\nOne device, one day (3 Oct 2026), "
        "1 367 readings at 15 s. The same functions are used by `scripts/make_figures.py`, "
        "so anything you see here is what the README reports.",
    ),
    ("code", SETUP),
    ("code", "df = load_history()\nsummary(df)"),
    ("code", "df.head()"),
    (
        "md",
        "## Timeline\nShaded spans would be heater-on; here we just plot the three temperatures.",
    ),
    (
        "code",
        "ax = df[['temp', 'temp_out', 'target']].plot(figsize=(11, 4), lw=1.2)\n"
        "ax.set_ylabel('°C');",
    ),
    ("md", "## Heater runs\nEach row is a maximal run with constant heater state."),
    ("code", "seg = fe.heater_segments(df)\nseg[seg['duration_min'] >= 2]"),
    ("code", "fe.heating_rates(seg).groupby('heater')['rate_c_per_min'].describe()"),
    ("md", "## Duty cycle, setpoint steps, time-to-target and overshoot"),
    (
        "code",
        "dc = fe.duty_cycle(df, '10min')\ndc.plot.bar(figsize=(11, 3), width=0.9)\n"
        "plt.xticks([]);\ndc.describe()",
    ),
    ("code", "fe.time_to_target(df)"),
    (
        "md",
        "## Thermal lag after switch-off\n`post_switch_rise` is the extra warming in the "
        "2 min after the heater goes off; look at the long runs.",
    ),
    (
        "code",
        "seg[seg['heater']].plot.scatter(x='duration_min', y='post_switch_rise', figsize=(6, 4));",
    ),
    (
        "md",
        "## The heater test log (10 Oct)\nOnly ~2 h of 2-minute polls, but its heat-up "
        "slope can be compared with the 3 Oct history.",
    ),
    (
        "code",
        "stress = load_stress_log()\n"
        "stress[['phase', 'temp', 'target', 'heater', 'run_state', 'action']]",
    ),
]

ANOMALIES = [
    (
        "md",
        "# 02 · Anomaly detectors, side by side\n\nThere is no labelled ground truth, so this "
        "notebook only shows *what* each detector flags and how much they agree.",
    ),
    ("code", SETUP),
    ("code", "df = load_history()\nft = fe.feature_table(df)\nft.describe().T"),
    ("md", "## Detector 1 · rolling z-score of the temperature slope, per heater state"),
    ("code", "z = an.rolling_zscore(ft, window=40, threshold=3.0)\nan.flagged_episodes(z['flag'])"),
    ("md", "## Detector 2 · IsolationForest on the feature table"),
    ("code", "iso = an.isolation_forest(ft, contamination=0.03)\nan.flagged_episodes(iso['flag'])"),
    ("md", "## Agreement"),
    ("code", "an.compare(z['flag'], iso['flag']).as_dict()"),
    (
        "code",
        "fig, ax = plt.subplots(figsize=(11, 4))\n"
        "ax.plot(df.index, df['temp'], lw=1)\n"
        "zf = ft.index[z['flag']]; isf = ft.index[iso['flag']]\n"
        "ax.scatter(zf, df.loc[zf, 'temp'], facecolors='none', edgecolors='C3', label='z-score')\n"
        "ax.scatter(isf, df.loc[isf, 'temp'] - 0.3, marker='v', color='C1', "
        "label='IsolationForest')\n"
        "ax.legend();",
    ),
    (
        "md",
        "## What the flagged readings look like\nTry changing `window`, `threshold` and "
        "`contamination` above and re-running; the sets change a lot, which is the point: "
        "unsupervised 'anomalies' are a choice of definition, not a fact about the data.",
    ),
    ("code", "ft.loc[z['flag']].head(20)"),
    ("code", "ft.loc[iso['flag']].head(20)"),
    ("md", "## Thermal model fit (used by the control-simulation repo)"),
    ("code", "fit = tf.fit_first_order(df)\nfit.as_dict()"),
]


def build(name: str, cells: list[tuple[str, str]]) -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nb["cells"] = [
        nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        for kind, src in cells
    ]
    nbf.write(nb, NB_DIR / name)


if __name__ == "__main__":
    NB_DIR.mkdir(exist_ok=True)
    build("01_explore.ipynb", EXPLORE)
    build("02_anomalies.ipynb", ANOMALIES)
    print("wrote", sorted(p.name for p in NB_DIR.glob("*.ipynb")))
