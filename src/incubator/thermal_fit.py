"""Least-squares fit of a first-order thermal model to the incubator trace.

Model (lumped chamber, one thermal mass):

    C * dT/dt = P * u(t - d) - (T - T_out) / R

    T      chamber temperature (°C)        T_out  ambient temperature (°C)
    u      heater relay state (0/1)        d      dead time (readings) before the heater
    P      heater power reaching the air (W)      acts on the sensor
    R      thermal resistance to ambient (°C/W)
    C      thermal capacitance (J/°C)

Dividing by C gives the form that is actually identifiable from temperatures alone:

    dT/dt = a * u(t - d) - b * (T - T_out),     a = P/C,  b = 1/(R C)

``a`` is the full-power heating rate (°C/min here) and ``b`` the inverse time constant
(1/min). From temperature data alone the three physical parameters cannot be separated:
only ``a``, ``b`` (hence ``tau = R C = 1/b`` and the full-power steady-state rise
``P R = a/b``) are determined. If the heater's electrical power is known, pass it as
``heater_power_w`` and ``C = P/a`` and ``R = 1/(b C)`` follow. That is how the fit is meant
to be used in the companion control-simulation repo.

Uncertainties are 1-sigma standard errors from the ordinary-least-squares covariance
(``sigma^2 (X^T X)^-1``). They describe fit noise, not model error: the model ignores the
heater's own thermal mass, so the residuals right after a switch are structured, and the
reported errors are optimistic. The dead time ``d`` is chosen by grid search on SSE.

Second-order model (:func:`fit_second_order`): the heat actually reaching the chamber,
``E`` (in °C/min of chamber warming), lags the relay with its own time constant instead of
following it instantly:

    dE/dt = (a * u - E) / tau_h
    dT/dt = E - b * (T - T_out)

    a      full-power heating rate once the heater is warmed through (°C/min), as above
    tau_h  time constant of the heater body (min): how long the delivered heat takes to
           build up after switch-on and to die away after switch-off
    b      1 / (R C), chamber loss to ambient (1/min), as above

Because ``E`` keeps flowing after the relay opens, the sensed temperature keeps rising after
switch-off and starts late after switch-on: that is what the first-order fit had to fake with
a dead time. It is the limit of a two-mass model (heater body + chamber) in which the
body's heat capacity is negligible next to the chamber's. The general four-parameter
two-mass model is :func:`fit_two_mass`, and on this data its extra parameter ``phi`` (the
body's share of the heat capacity) is *not determined*: holding ``phi`` anywhere between
-0.2 and 0.4 and refitting the rest changes the RMSE by under 2 % while ``b`` moves to
compensate, and left free the fit runs into the mirror branch described below. The
three-parameter form is kept because the fourth parameter cannot be identified from one
temperature sensor and 1 °C of ambient variation, not because it was measured to be zero
(numbers in ``figures/results.json`` under ``two_mass_trial``, figure 10).

Identifiability and the two branches: from one temperature the relay-to-temperature response
fixes the gain ``a / tau_h``, the sum ``1/tau_h + b`` and the product ``b / tau_h``, which
leaves two mirror solutions: (fast heater, slow chamber) or (slow heater, fast chamber).
Only the ambient path tells them apart, and on the day in the repo the outside sensor rose
by 1 °C *together with* the chamber (correlation 0.96), so the "fast chamber" branch fits
slightly better for a reason that may well be the box warming its own outside sensor. The
92-minute heater-off cool-down settles it independently: the chamber lost 0.035 °C/min at
3.5–4 °C above ambient, i.e. ``b`` is about 0.01 1/min, not 0.2. ``fit_second_order``
therefore bounds the search to the physical branch (``tau_h <= 30 min``, ``b <= 0.1``) by
default; pass ``physical=False`` to see the other one.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

Windows = list[tuple[str, str]]


@dataclass(frozen=True)
class FirstOrderFit:
    a_c_per_min: float  # P / C
    a_se: float
    b_per_min: float  # 1 / (R C)
    b_se: float
    dead_time_readings: int
    dead_time_s: float
    n: int
    rmse_c: float  # RMS of simulated minus measured chamber temperature (°C)
    r2: float

    @property
    def tau_min(self) -> float:
        """Time constant R*C in minutes."""
        return 1.0 / self.b_per_min

    @property
    def tau_se_min(self) -> float:
        return self.b_se / self.b_per_min**2

    @property
    def full_power_rise_c(self) -> float:
        """Steady-state rise above ambient with the heater on all the time: P*R = a/b."""
        return self.a_c_per_min / self.b_per_min

    def physical(self, heater_power_w: float) -> dict[str, float]:
        """R (°C/W) and C (J/°C) for a known heater power P (W), with propagated 1-sigma errors."""
        a_s = self.a_c_per_min / 60.0
        a_s_se = self.a_se / 60.0
        b_s = self.b_per_min / 60.0
        b_s_se = self.b_se / 60.0
        c = heater_power_w / a_s
        c_se = c * a_s_se / a_s
        r = 1.0 / (b_s * c)
        r_se = r * np.hypot(b_s_se / b_s, c_se / c)
        return {
            "P_w": heater_power_w,
            "C_j_per_c": c,
            "C_se": c_se,
            "R_c_per_w": r,
            "R_se": r_se,
            "tau_min": self.tau_min,
        }

    def as_dict(self) -> dict[str, float | int]:
        d = asdict(self)
        d["tau_min"] = self.tau_min
        d["tau_se_min"] = self.tau_se_min
        d["full_power_rise_c"] = self.full_power_rise_c
        return d


def _delayed_heater(df: pd.DataFrame, dead_time: int) -> np.ndarray:
    u = df["heater"].to_numpy(dtype=float)
    return np.r_[np.full(dead_time, u[0]), u[:-dead_time]] if dead_time else u


def _window_slices(df: pd.DataFrame, windows: Windows | None) -> list[slice]:
    if not windows:
        return [slice(0, len(df))]
    out = []
    for start, end in windows:
        lo = df.index.searchsorted(pd.Timestamp(start, tz="UTC"), side="left")
        hi = df.index.searchsorted(pd.Timestamp(end, tz="UTC"), side="right")
        if hi - lo >= 10:
            out.append(slice(lo, hi))
    if not out:
        raise ValueError("no window contains at least 10 readings")
    return out


def _simulate_window(
    a: float, b: float, u: np.ndarray, t_out: np.ndarray, temp0: float, dt_min: np.ndarray
) -> np.ndarray:
    temp = np.empty(len(u))
    temp[0] = temp0
    for k in range(1, len(u)):
        temp[k] = temp[k - 1] + dt_min[k - 1] * (a * u[k - 1] - b * (temp[k - 1] - t_out[k - 1]))
    return temp


def fit_first_order(
    df: pd.DataFrame,
    windows: Windows | None = None,
    max_dead_time: int = 16,
) -> FirstOrderFit:
    """Output-error fit of ``a``, ``b`` and the dead time on a history frame.

    For each candidate dead time the model is simulated from the measured temperature at
    the start of each window and ``a``, ``b`` are chosen by non-linear least squares on the
    simulated-minus-measured temperature. Simulating (rather than regressing on
    differences) matters here: the sensor reports 0.01 °C steps at 15 s, so per-reading
    slopes are mostly quantisation noise, while the integrated error over a 90-minute
    cool-down pins ``b`` down well.

    ``windows`` is an optional list of ``(start, end)`` UTC timestamps; each window is
    simulated separately from its own first reading. Without it the whole trace is one
    window. The dead time is searched over ``0..max_dead_time`` readings.
    """
    minutes = (df.index - df.index[0]).total_seconds().to_numpy() / 60.0
    temp = df["temp"].to_numpy(dtype=float)
    t_out = df["temp_out"].to_numpy(dtype=float)
    slices = _window_slices(df, windows)
    median_dt = float(df["dt_s"].median())

    def residuals(params: np.ndarray, u_all: np.ndarray) -> np.ndarray:
        a, b = params
        parts = []
        for sl in slices:
            dt = np.diff(minutes[sl])
            sim = _simulate_window(a, b, u_all[sl], t_out[sl], temp[sl][0], dt)
            parts.append(sim - temp[sl])
        return np.concatenate(parts)

    best: FirstOrderFit | None = None
    for d in range(max_dead_time + 1):
        u_all = _delayed_heater(df, d)
        sol = least_squares(
            residuals, x0=np.array([0.3, 0.01]), bounds=([0.0, 0.0], [5.0, 1.0]), args=(u_all,)
        )
        resid = sol.fun
        n = len(resid)
        dof = max(n - 2, 1)
        sigma2 = float(resid @ resid) / dof
        jac = sol.jac
        cov = sigma2 * np.linalg.pinv(jac.T @ jac)
        se = np.sqrt(np.diag(cov))
        ss_tot = float(((temp[np.r_[tuple(np.r_[sl] for sl in slices)]] - temp.mean()) ** 2).sum())
        fit = FirstOrderFit(
            a_c_per_min=float(sol.x[0]),
            a_se=float(se[0]),
            b_per_min=float(sol.x[1]),
            b_se=float(se[1]),
            dead_time_readings=d,
            dead_time_s=d * median_dt,
            n=int(n),
            rmse_c=float(np.sqrt(resid @ resid / n)),
            r2=1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan"),
        )
        if best is None or fit.rmse_c < best.rmse_c:
            best = fit
    assert best is not None
    return best


def simulate(
    fit: FirstOrderFit | tuple[float, float, int],
    heater: np.ndarray,
    temp_out: np.ndarray,
    temp0: float,
    dt_min: float | np.ndarray,
) -> np.ndarray:
    """Integrate the fitted model forward (explicit Euler) over a heater/ambient sequence.

    ``fit`` can be a :class:`FirstOrderFit` or a plain ``(a, b, dead_time)`` tuple.
    ``dt_min`` is the step in minutes, a scalar or one value per step (``len(heater) - 1``).
    Returns the simulated chamber temperature, same length as ``heater``.
    """
    if isinstance(fit, FirstOrderFit):
        a, b, d = fit.a_c_per_min, fit.b_per_min, fit.dead_time_readings
    else:
        a, b, d = fit
    u = np.asarray(heater, dtype=float)
    u = np.r_[np.full(d, u[0]), u[:-d]] if d else u
    t_out = np.asarray(temp_out, dtype=float)
    dt = np.broadcast_to(np.asarray(dt_min, dtype=float), (max(len(u) - 1, 0),))
    return _simulate_window(a, b, u, t_out, temp0, dt)


# ----------------------------------------------------------------------------------------
# Second order: heater body lag + chamber
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SecondOrderFit:
    a_c_per_min: float  # full-power heating rate once warmed through
    a_se: float
    tau_h_min: float  # heater body time constant
    tau_h_se: float
    b_per_min: float  # 1 / (R C)
    b_se: float
    dead_time_readings: int
    dead_time_s: float
    n: int
    rmse_c: float
    r2: float
    physical_branch: bool

    @property
    def tau_min(self) -> float:
        """Chamber time constant R*C in minutes."""
        return 1.0 / self.b_per_min

    @property
    def tau_se_min(self) -> float:
        return self.b_se / self.b_per_min**2

    @property
    def full_power_rise_c(self) -> float:
        """Steady-state rise above ambient with the heater always on: a / b (same as 1st order)."""
        return self.a_c_per_min / self.b_per_min

    @property
    def time_constants_min(self) -> tuple[float, float]:
        """(fast, slow) of the two time constants ``tau_h`` and ``1/b``."""
        pair = sorted((self.tau_h_min, self.tau_min))
        return float(pair[0]), float(pair[1])

    def as_dict(self) -> dict[str, float | int | bool]:
        d = asdict(self)
        d["tau_min"] = self.tau_min
        d["tau_se_min"] = self.tau_se_min
        d["full_power_rise_c"] = self.full_power_rise_c
        return d


def _simulate_second_order(
    a: float,
    tau_h: float,
    b: float,
    u: np.ndarray,
    t_out: np.ndarray,
    temp0: float,
    e0: float,
    dt_min: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Explicit Euler over one window; plain Python floats because this is the inner loop.

    Returns (sensed temperature, delivered heating rate E).
    """
    n = len(u)
    temp = np.empty(n)
    flow = np.empty(n)
    t, e = float(temp0), float(e0)
    temp[0], flow[0] = t, e
    ul, tl, dl = u.tolist(), t_out.tolist(), dt_min.tolist()
    for k in range(1, n):
        step = dl[k - 1]
        target = a * ul[k - 1]
        # E has an exact solution for a constant relay state over the step; using it keeps
        # the integration stable across the occasional long gap between readings.
        e_next = target + (e - target) * math.exp(-step / tau_h)
        t = t + step * (0.5 * (e + e_next) - b * (t - tl[k - 1]))
        e = e_next
        temp[k], flow[k] = t, e
    return temp, flow


def fit_second_order(
    df: pd.DataFrame,
    windows: Windows | None = None,
    max_dead_time: int = 0,
    physical: bool = True,
) -> SecondOrderFit:
    """Output-error fit of ``a``, ``tau_h`` and ``b`` (and optionally a dead time).

    Same recipe as :func:`fit_first_order`: simulate from the measured temperature at the
    start of each window and minimise simulated-minus-measured. The delivered heat ``E``
    starts at zero at each window start, so windows should begin after the heater has been
    off for a while (10 min or more on this device).

    ``max_dead_time`` defaults to 0 because this model does not need one; set it larger to
    check that claim (the figure script does). ``physical=True`` restricts the search to
    ``tau_h <= 30 min`` and ``b <= 0.1 /min`` (see the module docstring for why).
    """
    minutes = (df.index - df.index[0]).total_seconds().to_numpy() / 60.0
    temp = df["temp"].to_numpy(dtype=float)
    t_out = df["temp_out"].to_numpy(dtype=float)
    slices = _window_slices(df, windows)
    median_dt = float(df["dt_s"].median())
    if physical:
        lower, upper = [0.0, 0.1, 1e-4], [5.0, 30.0, 0.1]
        starts = ((0.5, 4.0, 0.01), (0.3, 10.0, 0.02))
    else:
        lower, upper = [0.0, 0.1, 1e-4], [100.0, 600.0, 2.0]
        starts = ((0.5, 4.0, 0.01), (10.0, 100.0, 0.2))

    def residuals(params: np.ndarray, u_all: np.ndarray) -> np.ndarray:
        a, tau_h, b = params
        parts = []
        for sl in slices:
            dt = np.diff(minutes[sl])
            sim, _ = _simulate_second_order(a, tau_h, b, u_all[sl], t_out[sl], temp[sl][0], 0.0, dt)
            parts.append(sim - temp[sl])
        return np.concatenate(parts)

    idx = np.r_[tuple(np.r_[sl] for sl in slices)]
    ss_tot = float(((temp[idx] - temp[idx].mean()) ** 2).sum())
    best: SecondOrderFit | None = None
    for d in range(max_dead_time + 1):
        u_all = _delayed_heater(df, d)
        sols = [
            least_squares(residuals, x0=np.array(x0), bounds=(lower, upper), args=(u_all,))
            for x0 in starts
        ]
        sol = min(sols, key=lambda s: float(s.fun @ s.fun))
        resid = sol.fun
        n = len(resid)
        sse = float(resid @ resid)
        sigma2 = sse / max(n - 3, 1)
        cov = sigma2 * np.linalg.pinv(sol.jac.T @ sol.jac)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        fit = SecondOrderFit(
            a_c_per_min=float(sol.x[0]),
            a_se=float(se[0]),
            tau_h_min=float(sol.x[1]),
            tau_h_se=float(se[1]),
            b_per_min=float(sol.x[2]),
            b_se=float(se[2]),
            dead_time_readings=d,
            dead_time_s=d * median_dt,
            n=int(n),
            rmse_c=float(np.sqrt(sse / n)),
            r2=1.0 - sse / ss_tot if ss_tot > 0 else float("nan"),
            physical_branch=physical,
        )
        if best is None or fit.rmse_c < best.rmse_c:
            best = fit
    assert best is not None
    return best


def simulate_second_order(
    fit: SecondOrderFit | tuple[float, float, float],
    heater: np.ndarray,
    temp_out: np.ndarray,
    temp0: float,
    dt_min: float | np.ndarray,
    flow0: float = 0.0,
    dead_time: int = 0,
) -> np.ndarray:
    """Integrate the second-order model forward; returns the sensed temperature.

    ``fit`` is a :class:`SecondOrderFit` or a plain ``(a, tau_h, b)`` tuple. ``flow0`` is
    the heat already being delivered at the start (0 after a long heater-off period).
    """
    if isinstance(fit, SecondOrderFit):
        a, tau_h, b = fit.a_c_per_min, fit.tau_h_min, fit.b_per_min
        dead_time = fit.dead_time_readings
    else:
        a, tau_h, b = fit
    u = np.asarray(heater, dtype=float)
    u = np.r_[np.full(dead_time, u[0]), u[:-dead_time]] if dead_time else u
    t_out = np.asarray(temp_out, dtype=float)
    dt = np.broadcast_to(np.asarray(dt_min, dtype=float), (max(len(u) - 1, 0),))
    temp, _ = _simulate_second_order(a, tau_h, b, u, t_out, temp0, flow0, dt)
    return temp


def aic(n: int, rmse_c: float, n_params: int) -> float:
    """Akaike information criterion for a Gaussian least-squares fit: n ln(SSE/n) + 2k.

    Only differences between models fitted on the *same* readings mean anything, and with
    autocorrelated residuals the absolute size of a difference is overstated.
    """
    return n * float(np.log(rmse_c**2)) + 2 * n_params


def compare_models(first: FirstOrderFit, second: SecondOrderFit) -> dict[str, float | int]:
    """RMSE and AIC side by side. Parameter counts: a, b, d = 3 for the first-order model;
    a, tau_h, b (+ d if non-zero) = 3 or 4 for the second-order one."""
    if first.n != second.n:
        raise ValueError("fits must be on the same readings to compare AIC")
    k1 = 2 + (1 if first.dead_time_readings else 0)
    k2 = 3 + (1 if second.dead_time_readings else 0)
    aic1 = aic(first.n, first.rmse_c, k1)
    aic2 = aic(second.n, second.rmse_c, k2)
    return {
        "n": first.n,
        "first_order_rmse_c": first.rmse_c,
        "second_order_rmse_c": second.rmse_c,
        "rmse_ratio": second.rmse_c / first.rmse_c,
        "first_order_aic": aic1,
        "second_order_aic": aic2,
        "delta_aic": aic2 - aic1,
    }


# ----------------------------------------------------------------------------------------
# Four parameters: the general two-mass model (heater body + chamber), to test whether the
# three-parameter heater-lag form leaves anything on the table
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TwoMassFit:
    """Fit of the general two-mass model; ``phi`` is the heater body's share of the total
    heat capacity, ``C_h / (C + C_h)``. ``phi = 0`` is exactly the heater-lag model."""

    a_c_per_min: float
    a_se: float
    tau_h_min: float
    tau_h_se: float
    b_per_min: float
    b_se: float
    phi: float
    phi_se: float
    b_fixed: bool
    n: int
    rmse_c: float
    r2: float

    @property
    def tau_min(self) -> float:
        return 1.0 / self.b_per_min

    @property
    def phi_is_distinguishable_from_zero(self) -> bool:
        """True when ``phi`` is more than two standard errors away from zero."""
        return abs(self.phi) > 2.0 * self.phi_se

    def as_dict(self) -> dict[str, float | int | bool]:
        d = asdict(self)
        d["tau_min"] = self.tau_min
        d["phi_is_distinguishable_from_zero"] = self.phi_is_distinguishable_from_zero
        return d


def _simulate_two_mass(
    a: float,
    tau_h: float,
    b: float,
    phi: float,
    u: np.ndarray,
    t_out: np.ndarray,
    temp0: float,
    e0: float,
    dt_min: np.ndarray,
) -> np.ndarray:
    """Same integrator as :func:`_simulate_second_order` with the chamber feeding back on
    the delivered heat: ``dE/dt = (a*u + phi*b*(T - T_out) - E) / tau_h``. With
    ``phi = 0`` the two functions compute identical numbers."""
    n = len(u)
    temp = np.empty(n)
    t, e = float(temp0), float(e0)
    temp[0] = t
    ul, tl, dl = u.tolist(), t_out.tolist(), dt_min.tolist()
    k_fb = phi * b
    for k in range(1, n):
        step = dl[k - 1]
        target = a * ul[k - 1] + k_fb * (t - tl[k - 1])
        e_next = target + (e - target) * math.exp(-step / tau_h)
        t = t + step * (0.5 * (e + e_next) - b * (t - tl[k - 1]))
        e = e_next
        temp[k] = t
    return temp


def fit_two_mass(
    df: pd.DataFrame, windows: Windows | None = None, fix_b: float | None = None
) -> TwoMassFit:
    """Output-error fit of the general two-mass (heater body + chamber) model.

    Two lumped masses, the heater body at an unobserved temperature ``T_h`` and the chamber
    at ``T``::

        C_h dT_h/dt = P u - (T_h - T) / R_h
        C   dT/dt   = (T_h - T) / R_h - (T - T_out) / R

    With ``E = (T_h - T) / (R_h C)`` (the chamber warming rate delivered by the body) this is

        dE/dt = (a u - E) / tau_h + (phi b / tau_h) (T - T_out)
        dT/dt = E - b (T - T_out)

    where ``phi = C_h / (C + C_h)`` is the body's share of the heat capacity, ``b = 1/(RC)``
    and ``tau_h = R_h C C_h / (C + C_h)``. The heater-lag model of :func:`fit_second_order`
    is the special case ``phi = 0``: a body whose heat capacity is negligible next to the
    chamber's. The only new thing the fourth parameter can do is let the chamber temperature
    feed back on the delivered heat, so a fit with ``phi`` indistinguishable from zero means
    this data does not support (or need) the extra parameter.

    ``phi`` is searched over ``[-1, 1]`` rather than the physical ``[0, 1)`` on purpose: a
    bound at zero would hide whether the data pushes it below zero, and the question here is
    whether it is *determined*, not whether it is physical. Same window handling, start
    values and physical-branch bounds as :func:`fit_second_order`; no dead time. ``fix_b``
    holds the chamber loss at a given value (e.g. the one the long cool-down measures) and
    fits only ``a``, ``tau_h`` and ``phi``; its standard errors are then conditional on ``b``.
    """
    minutes = (df.index - df.index[0]).total_seconds().to_numpy() / 60.0
    temp = df["temp"].to_numpy(dtype=float)
    t_out = df["temp_out"].to_numpy(dtype=float)
    u = df["heater"].to_numpy(dtype=float)
    slices = _window_slices(df, windows)
    lower, upper = [0.0, 0.1, 1e-4, -1.0], [5.0, 30.0, 0.1, 1.0]
    starts: tuple[tuple[float, ...], ...] = ((0.5, 4.0, 0.01, 0.0), (0.3, 10.0, 0.02, 0.3))
    if fix_b is not None:
        lower, upper = [lower[0], lower[1], lower[3]], [upper[0], upper[1], upper[3]]
        starts = ((0.5, 4.0, 0.0), (0.3, 10.0, 0.3))

    def residuals(params: np.ndarray) -> np.ndarray:
        if fix_b is None:
            a, tau_h, b, phi = params
        else:
            (a, tau_h, phi), b = params, fix_b
        parts = []
        for sl in slices:
            dt = np.diff(minutes[sl])
            sim = _simulate_two_mass(a, tau_h, b, phi, u[sl], t_out[sl], temp[sl][0], 0.0, dt)
            parts.append(sim - temp[sl])
        return np.concatenate(parts)

    sols = [least_squares(residuals, x0=np.array(x0), bounds=(lower, upper)) for x0 in starts]
    sol = min(sols, key=lambda s: float(s.fun @ s.fun))
    resid = sol.fun
    n = len(resid)
    sse = float(resid @ resid)
    sigma2 = sse / max(n - len(sol.x), 1)
    cov = sigma2 * np.linalg.pinv(sol.jac.T @ sol.jac)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    idx = np.r_[tuple(np.r_[sl] for sl in slices)]
    ss_tot = float(((temp[idx] - temp[idx].mean()) ** 2).sum())
    if fix_b is None:
        x = [float(v) for v in sol.x]
        e = [float(v) for v in se]
    else:
        x = [float(sol.x[0]), float(sol.x[1]), float(fix_b), float(sol.x[2])]
        e = [float(se[0]), float(se[1]), 0.0, float(se[2])]
    return TwoMassFit(
        a_c_per_min=x[0],
        a_se=e[0],
        tau_h_min=x[1],
        tau_h_se=e[1],
        b_per_min=x[2],
        b_se=e[2],
        phi=x[3],
        phi_se=e[3],
        b_fixed=fix_b is not None,
        n=int(n),
        rmse_c=float(np.sqrt(sse / n)),
        r2=1.0 - sse / ss_tot if ss_tot > 0 else float("nan"),
    )


def compare_two_mass(second: SecondOrderFit, two_mass: TwoMassFit) -> dict[str, float | int]:
    """RMSE and AIC of the three-parameter heater-lag fit against the four-parameter one,
    on the same readings. A negative ``delta_aic`` would favour the fourth parameter."""
    if second.n != two_mass.n:
        raise ValueError("fits must be on the same readings to compare AIC")
    k2 = 3 + (1 if second.dead_time_readings else 0)
    aic2 = aic(second.n, second.rmse_c, k2)
    aic4 = aic(two_mass.n, two_mass.rmse_c, 4)
    return {
        "n": second.n,
        "second_order_rmse_c": second.rmse_c,
        "two_mass_rmse_c": two_mass.rmse_c,
        "rmse_ratio": two_mass.rmse_c / second.rmse_c,
        "second_order_aic": aic2,
        "two_mass_aic": aic4,
        "delta_aic": aic4 - aic2,
    }


def two_mass_profile(
    df: pd.DataFrame, phis: tuple[float, ...] = (-0.2, 0.0, 0.2, 0.4, 0.6, 0.8)
) -> list[dict[str, float]]:
    """RMSE of the two-mass model with ``phi`` held at each value and ``a``, ``tau_h``, ``b``
    refitted (physical-branch bounds). Shows whether the fourth parameter is determined by
    the data or only trades against ``b`` along a flat ridge."""
    minutes = (df.index - df.index[0]).total_seconds().to_numpy() / 60.0
    temp = df["temp"].to_numpy(dtype=float)
    t_out = df["temp_out"].to_numpy(dtype=float)
    u = df["heater"].to_numpy(dtype=float)
    dt = np.diff(minutes)
    rows = []
    for phi in phis:

        def residuals(params: np.ndarray, phi: float = phi) -> np.ndarray:
            a, tau_h, b = params
            return _simulate_two_mass(a, tau_h, b, phi, u, t_out, temp[0], 0.0, dt) - temp

        sols = [
            least_squares(residuals, x0=np.array(x0), bounds=([0.0, 0.1, 1e-4], [5.0, 30.0, 0.1]))
            for x0 in ((0.5, 4.0, 0.01), (0.3, 10.0, 0.02))
        ]
        sol = min(sols, key=lambda s: float(s.fun @ s.fun))
        rows.append(
            {
                "phi": float(phi),
                "a_c_per_min": float(sol.x[0]),
                "tau_h_min": float(sol.x[1]),
                "b_per_min": float(sol.x[2]),
                "tau_min": 1.0 / float(sol.x[2]),
                "rmse_c": float(np.sqrt(sol.fun @ sol.fun / len(sol.fun))),
            }
        )
    return rows


def cooldown_loss_rate(
    df: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp, skip_min: float = 10.0
) -> dict[str, float]:
    """``b`` measured directly on a heater-off stretch, with no thermal model involved.

    Fits a straight line to the chamber temperature between ``start`` and ``end`` (UTC),
    skipping the first ``skip_min`` minutes so that the heat still leaving the heater body
    (about two ``tau_h``) does not bias the slope, and divides the slope by the mean excess
    over ambient: ``b = -slope / mean(T - T_out)``. Exact only if the excess is roughly
    constant over the stretch; here it falls from 4.4 to 3.6 °C over 80 min, so the result is
    a plain average of a slowly changing quantity. Used to pin ``b`` independently of the
    least-squares fits.
    """
    w = df.loc[pd.Timestamp(start, tz="UTC") : pd.Timestamp(end, tz="UTC")]
    w = w[w.index >= w.index[0] + pd.Timedelta(minutes=skip_min)]
    if len(w) < 10 or bool(w["heater"].any()):
        raise ValueError("need at least 10 heater-off readings after the skipped minutes")
    x = (w.index - w.index[0]).total_seconds().to_numpy() / 60.0
    slope = float(np.polyfit(x, w["temp"].to_numpy(dtype=float), 1)[0])
    excess = float((w["temp"] - w["temp_out"]).mean())
    return {
        "slope_c_per_min": slope,
        "mean_excess_c": excess,
        "b_per_min": -slope / excess,
        "tau_min": -excess / slope,
        "n": int(len(w)),
    }
