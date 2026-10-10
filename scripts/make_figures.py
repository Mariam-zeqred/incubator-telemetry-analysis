"""Regenerate every figure in ``figures/`` and ``figures/results.json`` from the raw data.

Run from the repo root: ``python scripts/make_figures.py``. The README quotes numbers only
from ``results.json``, so the README and the figures can never disagree with the code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from incubator import anomaly as an  # noqa: E402
from incubator import features as fe  # noqa: E402
from incubator import thermal_fit as tf  # noqa: E402
from incubator.load import load_history, load_stress_log, summary  # noqa: E402

FIG_DIR = ROOT / "figures"

# Colours: the validated categorical palette from the dataviz reference instance.
BLUE, ORANGE, AQUA, YELLOW, RED, GRAY = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e34948",
    "#52514e",
)
HEATER_SHADE = "#f3d9cf"

#: Windows where the first-order model is plausible: the first heat-up and the long
#: cool-down (before the unexplained warming at ~11:57 UTC), and the second big heat-up.
CLEAN_WINDOWS = [
    ("2026-10-03 10:01:51", "2026-10-03 11:56:41"),
    ("2026-10-03 13:09:31", "2026-10-03 13:40:00"),
]

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 130,
        "axes.grid": True,
        "grid.color": "#e6e5e1",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": GRAY,
        "xtick.color": GRAY,
        "ytick.color": GRAY,
        "font.size": 9,
        "legend.frameon": False,
    }
)


def _shade_heater(ax: plt.Axes, df: pd.DataFrame) -> None:
    seg = fe.heater_segments(df)
    for _, row in seg[seg["heater"]].iterrows():
        ax.axvspan(row["start"], row["end"] + pd.Timedelta(seconds=15), color=HEATER_SHADE, lw=0)


def _time_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz="UTC"))
    ax.set_xlabel("time (UTC, 3 Oct 2026)")


def fig_timeline(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    _shade_heater(ax, df)
    ax.plot(df.index, df["target"], color=AQUA, lw=1.5, drawstyle="steps-post", label="setpoint")
    ax.plot(df.index, df["temp"], color=BLUE, lw=1.5, label="chamber")
    ax.plot(df.index, df["temp_out"], color=ORANGE, lw=1.5, label="ambient")
    for _, ch in fe.setpoint_changes(df, include_initial=False).iterrows():
        ax.annotate(
            f"→ {ch['new_target']:.1f} °C",
            (ch["time"], ch["new_target"]),
            xytext=(4, 6),
            textcoords="offset points",
            fontsize=8,
            color=GRAY,
        )
    ax.set_ylabel("°C")
    ax.set_title("Chamber, ambient and setpoint; shaded = heater on")
    ax.legend(loc="lower right", ncol=3)
    _time_axis(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_timeline.png")
    plt.close(fig)


def fig_duty_cycle(df: pd.DataFrame) -> pd.Series:
    dc = fe.duty_cycle(df, "10min")
    fig, ax = plt.subplots(figsize=(10, 3))
    _shade_heater(ax, df)
    ax.bar(dc.index, dc.to_numpy(), width=pd.Timedelta(minutes=9), align="edge", color=BLUE, lw=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of window heater on")
    ax.set_title("Heater duty cycle per 10-minute window")
    _time_axis(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_duty_cycle.png")
    plt.close(fig)
    return dc


def fig_rates(seg: pd.DataFrame) -> pd.DataFrame:
    rates = fe.heating_rates(seg, min_duration_min=2.0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4))
    for heater, color, label in ((True, ORANGE, "heater on"), (False, BLUE, "heater off")):
        part = rates[rates["heater"] == heater]
        x = part["temp_start"] - part["temp_out_mean"]
        ax1.scatter(x, part["rate_c_per_min"], s=36, color=color, label=label, zorder=3)
    ax1.axhline(0, color=GRAY, lw=0.8)
    ax1.set_xlabel("chamber − ambient at run start (°C)")
    ax1.set_ylabel("temperature slope (°C/min)")
    ax1.set_title("Slope of each heater run ≥ 2 min")
    ax1.legend()
    on = seg[seg["heater"] & seg["post_switch_rise"].notna()]
    ax2.scatter(on["duration_min"], on["post_switch_rise"], s=36, color=ORANGE, zorder=3)
    ax2.set_xlabel("heater-on run length (min)")
    ax2.set_ylabel("rise after switch-off (°C, next 2 min)")
    ax2.set_title("Thermal lag: how much it keeps warming after OFF")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_rates.png")
    plt.close(fig)
    return rates


def fig_anomalies(df: pd.DataFrame) -> dict[str, object]:
    ft = fe.feature_table(df)
    z = an.rolling_zscore(ft)
    iso = an.isolation_forest(ft)
    comp = an.compare(z["flag"], iso["flag"])
    fig, ax = plt.subplots(figsize=(10, 4))
    _shade_heater(ax, df)
    ax.plot(df.index, df["temp"], color=BLUE, lw=1.2, label="chamber")
    zf = ft.index[z["flag"].to_numpy()]
    isf = ft.index[iso["flag"].to_numpy()]
    ax.scatter(
        zf,
        df.loc[zf, "temp"],
        s=42,
        marker="o",
        facecolors="none",
        edgecolors=RED,
        lw=1.4,
        label=f"rolling z-score ({comp.n_a})",
        zorder=4,
    )
    ax.scatter(
        isf,
        df.loc[isf, "temp"] - 0.35,
        s=30,
        marker="v",
        color=YELLOW,
        label=f"IsolationForest ({comp.n_b})",
        zorder=4,
    )
    ax.set_ylabel("°C")
    ax.set_title(
        f"Readings flagged by each detector (both: {comp.n_both}, Jaccard {comp.jaccard:.2f})"
    )
    ax.legend(loc="lower right")
    _time_axis(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_anomalies.png")
    plt.close(fig)
    return {
        "comparison": comp.as_dict(),
        "zscore_episodes": _episodes(an.flagged_episodes(z["flag"])),
        "isoforest_episodes": _episodes(an.flagged_episodes(iso["flag"])),
    }


def _episodes(ep: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {
            "start_utc": r["start"].strftime("%H:%M:%S"),
            "end_utc": r["end"].strftime("%H:%M:%S"),
            "readings": int(r["readings"]),
        }
        for _, r in ep.iterrows()
    ]


def fig_model(df: pd.DataFrame) -> dict[str, object]:
    full = tf.fit_first_order(df)
    clean = tf.fit_first_order(df, windows=CLEAN_WINDOWS)
    dt = np.diff((df.index - df.index[0]).total_seconds().to_numpy() / 60.0)
    sim_full = tf.simulate(
        full, df["heater"].to_numpy(), df["temp_out"].to_numpy(), df["temp"].iloc[0], dt
    )
    sim_clean = tf.simulate(
        clean, df["heater"].to_numpy(), df["temp_out"].to_numpy(), df["temp"].iloc[0], dt
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    _shade_heater(ax, df)
    ax.plot(df.index, df["temp"], color=BLUE, lw=1.5, label="measured")
    ax.plot(df.index, sim_full, color=ORANGE, lw=1.5, ls="--", label="model, fitted on whole trace")
    ax.plot(df.index, sim_clean, color=AQUA, lw=1.5, ls=":", label="model, fitted on clean windows")
    for start, end in CLEAN_WINDOWS:
        ax.axvspan(
            pd.Timestamp(start, tz="UTC"),
            pd.Timestamp(end, tz="UTC"),
            ymin=0,
            ymax=0.03,
            color=AQUA,
            lw=0,
        )
    ax.set_ylabel("°C")
    ax.set_title("First-order model simulated open-loop from the first reading vs measured")
    ax.legend(loc="lower right")
    _time_axis(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_model_fit.png")
    plt.close(fig)
    return {
        "full_trace": _round(full.as_dict()),
        "clean_windows": _round(clean.as_dict()),
        "clean_windows_utc": CLEAN_WINDOWS,
        "open_loop_rmse_c_full": round(
            float(np.sqrt(np.mean((sim_full - df["temp"].to_numpy()) ** 2))), 3
        ),
        "open_loop_rmse_c_clean": round(
            float(np.sqrt(np.mean((sim_clean - df["temp"].to_numpy()) ** 2))), 3
        ),
    }


def fig_second_order(df: pd.DataFrame, seg: pd.DataFrame) -> dict[str, object]:
    """Figure 07: first- vs second-order model, whole trace and a zoom on the first run.

    Also writes ``figures/model_params.json``, the hand-over file for the control-simulation
    repo, and returns the comparison facts for ``results.json``.
    """
    first = tf.fit_first_order(df)
    second = tf.fit_second_order(df, max_dead_time=4)  # max_dead_time>0 only to show d=0 wins
    other_branch = tf.fit_second_order(df, physical=False)
    heater, t_out = df["heater"].to_numpy(), df["temp_out"].to_numpy()
    dt = np.diff((df.index - df.index[0]).total_seconds().to_numpy() / 60.0)
    sim1 = tf.simulate(first, heater, t_out, df["temp"].iloc[0], dt)
    sim2 = tf.simulate_second_order(second, heater, t_out, df["temp"].iloc[0], dt)
    measured = df["temp"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[1.4, 1])
    for ax in (ax1, ax2):
        _shade_heater(ax, df)
        ax.plot(df.index, measured, color=BLUE, lw=1.6, label="measured")
        ax.plot(
            df.index,
            sim1,
            color=ORANGE,
            lw=1.3,
            ls="--",
            label=f"1st order + {first.dead_time_s / 60:.1f} min dead time "
            f"(RMSE {first.rmse_c:.2f} °C)",
        )
        ax.plot(
            df.index,
            sim2,
            color=AQUA,
            lw=1.3,
            label=f"2nd order, heater lag τh = {second.tau_h_min:.1f} min "
            f"(RMSE {second.rmse_c:.2f} °C)",
        )
        ax.set_ylabel("°C")
    ax1.set_title("Both models simulated open-loop from the first reading")
    ax1.legend(loc="lower right")
    _time_axis(ax1)
    zoom_lo, zoom_hi = (
        pd.Timestamp("2026-10-03 10:00", tz="UTC"),
        pd.Timestamp("2026-10-03 10:50", tz="UTC"),
    )
    ax2.set_xlim(zoom_lo, zoom_hi)
    m = (df.index >= zoom_lo) & (df.index <= zoom_hi)
    ax2.set_ylim(measured[m].min() - 0.5, measured[m].max() + 0.7)
    ax2.set_title(
        "Zoom: first heater run from cold; the rise after switch-off is the lag "
        "the 2nd-order model explains"
    )
    _time_axis(ax2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_second_order.png")
    plt.close(fig)

    # does the lag model reproduce the post-switch peaks after the three long runs?
    peaks = []
    for _, row in seg[seg["heater"] & (seg["duration_min"] >= 5)].iterrows():
        i_off = df.index.get_loc(row["end"])
        after = slice(i_off, i_off + 80)  # 20 min
        peaks.append(
            {
                "run_start_utc": row["start"].strftime("%H:%M"),
                "measured_rise_c": round(float(measured[after].max() - measured[i_off]), 2),
                "first_order_rise_c": round(float(sim1[after].max() - sim1[i_off]), 2),
                "second_order_rise_c": round(float(sim2[after].max() - sim2[i_off]), 2),
                "measured_minutes_to_peak": round(float(np.argmax(measured[after]) * 0.25), 1),
                "second_order_minutes_to_peak": round(float(np.argmax(sim2[after]) * 0.25), 1),
            }
        )

    params = {
        "source": "incubator-telemetry-analysis/scripts/make_figures.py, fitted on "
        "data/hive-1C5A00_history_2026-10-03.json (1 367 readings, one device, one day)",
        "units": {"temperature": "°C", "time": "min", "heater": "0/1 relay state"},
        "first_order": {
            "equation": "dT/dt = a*u(t - d) - b*(T - T_out)",
            **_round(first.as_dict()),
        },
        "second_order": {
            "equation": "dE/dt = (a*u - E)/tau_h ; dT/dt = E - b*(T - T_out)",
            **_round(second.as_dict()),
        },
        "second_order_other_branch": {
            "note": "mirror solution (slow heater, fast chamber); same heater response, "
            "rejected because the 92-min cool-down measures b ~ 0.01, see thermal_fit.py",
            **_round(other_branch.as_dict()),
        },
        "notes": [
            "a, b and tau_h are identifiable from temperature alone; R, C and P separately "
            "need the heater's electrical power (FirstOrderFit.physical).",
            "Standard errors are OLS values with autocorrelated residuals: optimistic.",
            "Ambient on the day was 27.2-28.3 °C; the door stayed shut.",
        ],
    }
    (FIG_DIR / "model_params.json").write_text(json.dumps(params, indent=2) + "\n")
    return {
        "second_order": _round(second.as_dict()),
        "second_order_other_branch_rmse_c": round(other_branch.rmse_c, 4),
        "comparison": _round(tf.compare_models(first, second)),
        "post_switch_peaks": peaks,
    }


def fig_stress(stress: pd.DataFrame) -> dict[str, object]:
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for _, row in stress[stress["heater"]].iterrows():
        ax.axvspan(row.name, row.name + pd.Timedelta(minutes=2), color=HEATER_SHADE, lw=0)
    ax.plot(
        stress.index, stress["target"], color=AQUA, lw=1.5, drawstyle="steps-post", label="setpoint"
    )
    ax.plot(stress.index, stress["temp"], color=BLUE, lw=1.5, marker="o", ms=3, label="chamber")
    ax.plot(stress.index, stress["temp_out"], color=ORANGE, lw=1.5, label="ambient")
    for phase, part in stress.groupby("phase"):
        ax.annotate(phase, (part.index[0], 41.2), fontsize=8, color=GRAY)
    ax.set_ylim(26, 42)
    ax.set_ylabel("°C")
    ax.set_title("Heater test log, 10 Oct 2026 (2-minute polls; shaded = heater on)")
    ax.legend(loc="lower right", ncol=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz="UTC"))
    ax.set_xlabel("time (UTC, 10 Oct 2026)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_stress_log.png")
    plt.close(fig)
    # heat-up slope over the continuous heater-on stretch
    on = stress[stress["heater"]]
    minutes = (on.index - on.index[0]).total_seconds().to_numpy() / 60.0
    slope = (
        float(np.polyfit(minutes, on["temp"].to_numpy(), 1)[0]) if len(on) >= 3 else float("nan")
    )
    return {
        "rows": int(len(stress)),
        "span_min": round(float((stress.index[-1] - stress.index[0]).total_seconds() / 60.0), 1),
        "heater_on_rows": int(len(on)),
        "heatup_slope_c_per_min": round(slope, 3),
        "heatup_from_to_c": [float(on["temp"].iloc[0]), float(on["temp"].iloc[-1])],
        "peak_c": float(stress["temp"].max()),
        "crash_count_range": [int(stress["crash_count"].min()), int(stress["crash_count"].max())],
        "run_states": sorted(stress["run_state"].unique().tolist()),
    }


def _long_run_lag(df: pd.DataFrame, seg: pd.DataFrame) -> list[dict[str, object]]:
    """For heater runs >= 5 min: peak chamber temperature in the 20 min after switch-off."""
    out = []
    for _, row in seg[seg["heater"] & (seg["duration_min"] >= 5)].iterrows():
        after = df.loc[
            (df.index > row["end"]) & (df.index <= row["end"] + pd.Timedelta(minutes=20))
        ]
        peak_time = after["temp"].idxmax()
        out.append(
            {
                "run_start_utc": row["start"].strftime("%H:%M"),
                "run_min": round(float(row["duration_min"]), 1),
                "temp_at_off": float(row["temp_end"]),
                "peak_c": float(after["temp"].max()),
                "rise_c": round(float(after["temp"].max() - row["temp_end"]), 2),
                "minutes_to_peak": round(float((peak_time - row["end"]).total_seconds() / 60.0), 1),
            }
        )
    return out


def _round(d: dict[str, object]) -> dict[str, object]:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    df = load_history()
    stress = load_stress_log()
    seg = fe.heater_segments(df)

    fig_timeline(df)
    dc = fig_duty_cycle(df)
    rates = fig_rates(seg)
    anomalies = fig_anomalies(df)
    model = fig_model(df)
    second = fig_second_order(df, seg)
    stress_facts = fig_stress(stress)

    on = rates[rates["heater"]]
    long_off = rates[~rates["heater"] & (rates["duration_min"] >= 60)]
    steady = df.loc["2026-10-03 13:35":"2026-10-03 15:39"]
    ttt = fe.time_to_target(df)
    results = {
        "history": summary(df),
        "heater_runs": int(len(seg)),
        "heatup_runs_ge_2min": {
            "count": int(len(on)),
            "slope_c_per_min": [round(float(v), 3) for v in on["rate_c_per_min"]],
            "min": round(float(on["rate_c_per_min"].min()), 3),
            "max": round(float(on["rate_c_per_min"].max()), 3),
        },
        "longest_cooldown": {
            "start_utc": long_off["start"].iloc[0].strftime("%H:%M"),
            "duration_min": round(float(long_off["duration_min"].iloc[0]), 1),
            "slope_c_per_min": round(float(long_off["rate_c_per_min"].iloc[0]), 4),
            "from_to_c": [
                float(long_off["temp_start"].iloc[0]),
                float(long_off["temp_end"].iloc[0]),
            ],
        },
        "post_switch_rise_c": {
            "median_2min": round(float(seg.loc[seg["heater"], "post_switch_rise"].median()), 2),
            "max_2min": round(float(seg.loc[seg["heater"], "post_switch_rise"].max()), 2),
            "long_runs_20min": _long_run_lag(df, seg),
        },
        "time_to_target": [
            {
                "new_target": float(r["new_target"]),
                "reached": bool(r["reached"]),
                "minutes": None if pd.isna(r["minutes"]) else round(float(r["minutes"]), 1),
                "overshoot_c": None
                if pd.isna(r["overshoot_c"])
                else round(float(r["overshoot_c"]), 2),
            }
            for _, r in ttt.iterrows()
        ],
        "steady_state_38_8": {
            "window_utc": ["13:35", "15:39"],
            "temp_mean": round(float(steady["temp"].mean()), 2),
            "temp_min": float(steady["temp"].min()),
            "temp_max": float(steady["temp"].max()),
            "duty_cycle": round(float(steady["heater"].mean()), 3),
            "pulses": int((steady["heater"] & ~steady["heater"].shift(fill_value=False)).sum()),
            "delta_in_out_mean": round(float((steady["temp"] - steady["temp_out"]).mean()), 2),
        },
        "duty_cycle_10min": {"mean": round(float(dc.mean()), 3), "max": round(float(dc.max()), 3)},
        "anomalies": anomalies,
        "thermal_model": model,
        "second_order_model": second,
        "stress_log": stress_facts,
    }
    (FIG_DIR / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
