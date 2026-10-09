"""calibrate_sigma.py: the size of the random kick, measured on the dev drifters (#89).

sigma sets how far the cloud spreads: sigma * sqrt(t) metres per axis. D026 releases every
particle from one point, so at sigma = 0 there is no cloud and no map. This measures sigma
against real buoys, on dev tracks only (D025), in four stages:

  stage1   One particle per window at sigma = 0, with alpha = 0.02 (the engine) and
           alpha = 0. The gap to the buoy, hour by hour, east/north and downwind/crosswind.
           It gives Table A (drogued against undrogued), the gate (the model must beat
           "the buoy stays put" and "the buoy keeps its first velocity" at 24 h), the growth
           exponent beta, sigma by horizon, and sigma_0, the start of the ladder.
  twin     Fake buoys with a KNOWN sigma, driven by the real ocean, put through the same
           method. Shows whether the method gives back what was planted.
  ladder   Real ensembles at sigma_0 x {0.5, 0.7, 1, 1.4, 2}. sigma* is where the buoy is
           inside the 90 % region in 90 % of windows at 24 h (P1, Aditya, 1 Oct 2026).
           Coverage, not closeness: the kick has zero mean, so it can only widen the cloud.
  calibrate  sigma*, then the person's crosswind slide that a round buoy cannot show
           (Allen 2005, 0.51 % of the wind, side unknown), added because the random side
           makes its covariance with everything else zero.

Each stage runs on a compute node as one task of a Slurm array over the start days:

    python -m sar.validate.drift_windows --data /home/26p67/data --out <dir>/windows
    python -m sar.validate.calibrate_sigma stage1 --data /home/26p67/data \
        --windows <dir>/windows --task K --tasks M --out <dir>/stage1
    python -m sar.validate.calibrate_sigma fit --windows <dir>/windows --stage1 <dir>/stage1 \
        --out <dir>/fit.json
    python -m sar.validate.calibrate_sigma ladder ... --sigmas 25 35 50 70 100
    python -m sar.validate.calibrate_sigma twin ... ;  ... calibrate ...

THE HORIZON. sigma is matched at one hour T, because the real gap grows faster than
sigma * sqrt(t) and one sigma is right at one time only. P1 chose 24 h on 1 Oct; D028's
amendment of 4 Oct moved the engine to 4 h, the end of the longest search. `--hour` on fit,
twin and calibrate picks T (default 24), so both calibrations re-run from the same code.
fit records the hour it used, and calibrate refuses a fit made at another hour, because
the crosswind slide in it is matched at that hour.

THE RANDOM VELOCITY (vault D030, 4-6 Oct). The ladder can also run the random term as a
velocity error with memory (`--sigmas-u`, `--memory-h`): the same engine, sigma = 0, and
sigma_u laddered with T_L fixed. Its rows say so in a `model` column, and calibrate adds the
person's crosswind slide as a velocity, sigma_c / sqrt(T): over the hours of a search both
grow in a straight line, so their velocities add in quadrature as the random walk's kicks
did.

SIGMA_U BY THE CURRENT AT THE START (vault D033, 9 Oct; docs/ADR007.md). One pooled sigma_u
covers 90 % of buoys on average, but not by water: the quiet-water cloud is too wide and the
jet's far too narrow (L23, L35). So sigma_u is sized by HYCOM's current speed at the window's
start, stage 1's `current0`, the one thing a forecaster knows when the call comes in. Never
by the buoy's own later speed, which is the answer:

    ladder --stage1 DIR --min-start-speed 0.5 --sigmas-u ...   the fast windows only, higher up
    rule --ladder DIR [DIR ...] --stage1 DIR --fit FIT --hour 4 --out rule.json
        per speed bin, sigma_u* where 90 % are inside the 90 % region; a weighted straight
        line sigma_u = a + b x speed through them, resampled by group; capped at the fastest
        bin with enough groups; the crosswind slide added in quadrature
    ladder --stage1 DIR --sigma-u-rule rule.json ...   every window at its own rule value
    rule ... --confirm DIR    adds the confirmation: coverage by speed bin and lead

docs/sigma-calibration.md sets out the maths, the decisions and the results.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time as clock
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from sar.model.drift import LEEWAY_COEFFICIENT
from sar.model.interpolate import OutOfCoverageError
from sar.model.position import MEMORY_TIME_S
from sar.pipeline.gridded import GriddedForcing
from sar.pipeline.track import DriftPipeline
from sar.utils.geo import M_PER_DEG_LAT, metres_per_degree_lon
from sar.validate.drift_windows import Windows
from sar.validate.region import LEVELS, local_metres, score_cloud

TIMESTEP_S = 60.0                        # D009
CALIBRATION_HOUR = 24                    # P1: decided by Aditya, 1 Oct 2026
HORIZONS_H = (6, 12, 24, 48)
LEVEL = 0.9                              # R2c's 90 % region
# A round Gaussian cloud with sigma*sqrt(t) per axis holds 90 % within this many of them.
RADIUS_FACTOR = float(np.sqrt(-2.0 * np.log(1.0 - LEVEL)))      # 2.146
LADDER = (0.5, 0.7, 1.0, 1.4, 2.0)       # steps of sqrt(2) about sigma_0
ALPHAS = (LEEWAY_COEFFICIENT, 0.0)       # the engine, and D018's alpha-off control
# Allen 2005 via D018: a person in the water also slides 0.51 % of the wind speed across
# it, to a side nobody can know in advance.
CROSSWIND_SLIDE = 0.0051
STRATA_MS = (0.3, 1.0)                   # current speed at the start: < 0.3, 0.3-1, > 1 m/s
# D033: the bins sigma_u is measured in, by HYCOM's current speed at the start (m/s), and the
# fewest shared-water groups a bin needs to count. The edges put the 0.3 and 1.0 of STRATA_MS
# on bin boundaries and split the slow water, where nearly all the windows are, finer.
SPEED_BINS_MS = (0.0, 0.15, 0.3, 0.5, 0.75, 1.0, np.inf)
MIN_GROUPS = 10
BETA_HOURS = (6, 48)
N_BOOT = 1000
FORCING_ERRORS = (OutOfCoverageError, FileNotFoundError)   # ForcingGapError is one of the first
RANDOM_WALK, RANDOM_VELOCITY = "random walk", "random velocity"   # the ladder's two models


# ---------------------------------------------------------------------------- forcing

def gridded(root):
    """Open the real forcing for each window: GriddedForcing.from_dir(root, start, end)."""
    def open_(start, end):
        return GriddedForcing.from_dir(root, start, end)
    return open_


def fixed(forcing):
    """The same forcing for every window, already open (tests use ConstantForcing)."""
    def open_(start, end):
        return contextlib.nullcontext(forcing)
    return open_


class SlidingForcing:
    """A person's crosswind slide added to the current. The twin experiment only.

    Each particle slides `slide` x the wind speed at right angles to the wind, to its
    own fixed side (+1 right of downwind, -1 left), drawn once per particle. The engine
    never uses this: D002 keeps the model at three terms and carries the slide in eta.
    """

    def __init__(self, base, slide: float, sides):
        self.base, self.slide, self.sides = base, float(slide), np.asarray(sides, float)

    def sample(self, lats, lons, time):
        current, wind = self.base.sample(lats, lons, time)
        right = np.column_stack([wind[:, 1], -wind[:, 0]])          # wind turned 90 deg clockwise
        return current + self.slide * self.sides[:, None] * right, wind

    def describe(self) -> dict:
        return {"backend": "sliding", "slide": self.slide, "base": self.base.describe()}


# ---------------------------------------------------------------------------- running

def run_batch(forcing, t0, lats, lons, hours: int, leeway: float, sigma: float, seed,
              record_hours, sigma_u=0.0, memory_time_s: float = MEMORY_TIME_S) -> dict:
    """One run of the engine from t0 for every particle given.

    Returns, at each recorded hour: positions (H, N, 2); the wind run, the integral of the
    10 m wind along the particle's path so far, in metres (H, N, 2), whose direction is the
    downwind frame and whose length is what the leeway term multiplies; and whether the
    particle has had finite forcing at every step so far (H, N).
    """
    pipeline = DriftPipeline(forcing, TIMESTEP_S, leeway=leeway, sigma=sigma, seed=seed,
                             sigma_u=sigma_u, memory_time_s=memory_time_s)
    per_hour = round(3600.0 / TIMESTEP_S)
    want = {int(h) * per_hour: i for i, h in enumerate(record_hours)}
    n = np.size(lats)
    pos = np.full((len(want), n, 2), np.nan)
    run = np.zeros((len(want), n, 2))
    alive = np.zeros((len(want), n), bool)
    cum, ok, current0 = np.zeros((n, 2)), np.ones(n, bool), None
    for state in pipeline.track(t0, lats, lons, hours * 3600.0):
        if state.step in want:
            i = want[state.step]
            pos[i], run[i], alive[i] = state.positions, cum, ok
        if state.drift is not None:
            if current0 is None:
                current0 = state.current.copy()
            ok = ok & np.isfinite(state.drift).all(axis=1)
            cum = cum + np.where(np.isfinite(state.wind), state.wind, 0.0) * TIMESTEP_S
    return {"pos": pos, "wind_run": run, "alive": alive, "current0": current0}


def task_batches(windows: Windows, task: int = 0, tasks: int = 1, days=None, seed=None):
    """This task's share of the start days, interleaved so every task sees every season.

    `days` keeps a random subset of that many start days (the twin), drawn with `seed`.
    Returns (global batch number, row indices) pairs; the number seeds each batch's kicks.
    """
    batches = list(enumerate(windows.batches()))
    if days is not None and days < len(batches):
        keep = np.sort(np.random.default_rng(seed).choice(len(batches), days, replace=False))
        batches = [batches[k] for k in keep]
    return batches[task::tasks]


def gap_metres(lat_b, lon_b, lat_m, lon_m) -> np.ndarray:
    """Buoy minus model, [east, north] metres, cos(lat) at the midpoint."""
    dlon = (np.asarray(lon_b) - lon_m + 180.0) % 360.0 - 180.0
    mid = 0.5 * (np.asarray(lat_b) + lat_m)
    return np.stack([dlon * metres_per_degree_lon(mid),
                     (np.asarray(lat_b) - lat_m) * M_PER_DEG_LAT], axis=-1)


def residual_rows(rows_idx, hours: int, truth_lat, truth_lon, truth_ok, start_lat, start_lon,
                  ve0, vn0, model: dict, alpha: float) -> pd.DataFrame:
    """The gap at every hour of every window in one batch, with the two naive forecasts.

    truth_* are (n, hours + 1); model["pos"] and model["wind_run"] are (hours + 1, n, 2).
    gap = buoy - model. In the wind frame, downwind is along the wind run so far and
    crosswind is to its right; a model running ahead downwind gives a negative gap_dw.
    """
    lat_m = model["pos"][..., 0].T
    lon_m = model["pos"][..., 1].T
    gap = gap_metres(truth_lat, truth_lon, lat_m, lon_m)
    run = np.transpose(model["wind_run"], (1, 0, 2))
    run_len = np.hypot(run[..., 0], run[..., 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        d = run / run_len[..., None]
    d[run_len < 1.0] = np.nan                    # no wind yet: no frame
    gap_dw = np.sum(gap * d, axis=-1)
    gap_cw = gap[..., 0] * d[..., 1] - gap[..., 1] * d[..., 0]   # positive to the right

    t = np.arange(hours + 1) * 3600.0
    still = gap_metres(truth_lat, truth_lon, start_lat[:, None], start_lon[:, None])
    p_lat = start_lat[:, None] + vn0[:, None] * t / M_PER_DEG_LAT
    p_lon = start_lon[:, None] + ve0[:, None] * t / metres_per_degree_lon(start_lat)[:, None]
    persist = gap_metres(truth_lat, truth_lon, p_lat, p_lon)

    n = len(rows_idx)
    valid = truth_ok & model["alive"].T
    out = pd.DataFrame({
        "w": np.repeat(rows_idx, hours + 1).astype(np.int32),
        "alpha": np.float32(alpha),
        "hour": np.tile(np.arange(hours + 1), n).astype(np.int16),
        "valid": valid.ravel(),
        "gap_e": gap[..., 0].ravel(), "gap_n": gap[..., 1].ravel(),
        "gap_dw": gap_dw.ravel(), "gap_cw": gap_cw.ravel(),
        "run_e": run[..., 0].ravel(), "run_n": run[..., 1].ravel(),
        "sep": np.hypot(gap[..., 0], gap[..., 1]).ravel(),
        "sep_still": np.hypot(still[..., 0], still[..., 1]).ravel(),
        "sep_persist": np.hypot(persist[..., 0], persist[..., 1]).ravel(),
    })
    floats = out.columns[out.dtypes == np.float64]
    out[floats] = out[floats].astype(np.float32)
    return out


def stage1(windows: Windows, open_forcing, batches, alphas=ALPHAS, log=print):
    """Stage 1 over the given batches: residual rows, and one status row per window."""
    hours = windows.hours
    tab = windows.table
    parts, status = [], []
    for b, rows in batches:
        t0 = tab["t0"].iloc[rows[0]].to_datetime64()
        end = t0 + np.timedelta64(hours, "h")
        lat0, lon0 = tab["lat0"].to_numpy()[rows], tab["lon0"].to_numpy()[rows]
        try:
            with open_forcing(t0, end) as forcing:
                for alpha in alphas:
                    m = run_batch(forcing, t0, lat0, lon0, hours, alpha, 0.0, None,
                                  range(hours + 1))
                    parts.append(residual_rows(
                        rows, hours, windows.lat[rows], windows.lon[rows], windows.ok[rows],
                        lat0, lon0, tab["ve0"].to_numpy()[rows], tab["vn0"].to_numpy()[rows],
                        m, alpha))
                    if alpha == alphas[0]:
                        c0 = m["current0"]
                        for k, r in enumerate(rows):
                            status.append({"w": int(r), "batch": b, "status": "ok",
                                           "current0_u": c0[k, 0], "current0_v": c0[k, 1]})
        except FORCING_ERRORS as error:
            status += [{"w": int(r), "batch": b, "status": type(error).__name__,
                        "reason": str(error)} for r in rows]
            log(f"batch {b} at {t0}: skipped, {type(error).__name__}: {error}")
    rows_out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return rows_out, pd.DataFrame(status)


def score_rows(windows_rows, sigma, lead, pos, alive, truth_lat, truth_lon, truth_ok,
               bandwidths=(1.0,), subsample=()) -> list[dict]:
    """Score each window's cloud at one lead. pos is (n, N, 2), alive (n, N)."""
    out = []
    for k, w in enumerate(windows_rows):
        if not truth_ok[k]:
            continue
        pts = local_metres(pos[k, :, 0], pos[k, :, 1], truth_lat[k], truth_lon[k])
        base = {"w": int(w), "sigma": float(sigma), "lead": int(lead),
                "beached": float(1.0 - alive[k].mean())}
        for n_use in (pts.shape[0], *subsample):
            for bw in (bandwidths if n_use == pts.shape[0] else (1.0,)):
                s = score_cloud(pts[:n_use], [0.0, 0.0], levels=(0.5, LEVEL), bandwidth=bw)
                out.append({**base, "n": int(n_use), "bandwidth": float(bw), "rank": s.rank,
                            "area90_km2": s.area_m2[LEVEL] / 1e6, "energy_m": s.energy,
                            "centroid_m": s.centroid_error_m, "spread_m": s.spread_m})
    return out


def ladder(windows: Windows, open_forcing, batches, sigmas, particles: int, seed: int,
           leads=HORIZONS_H, truth=None, bandwidths=(1.0,), subsample=(), tag="real",
           log=print, model: str = RANDOM_WALK,
           memory_time_s: float = MEMORY_TIME_S, sigma_u_per_window=None) -> pd.DataFrame:
    """Stage 2: real ensembles at each sigma, scored against the buoy at each lead.

    The engine exactly as it will be used: alpha = 0.02, datum spread 0 (D026), 60 s
    steps, beached particles frozen and kept (D016). `truth` replaces the real buoys
    with (lat, lon, ok) arrays shaped like windows.lat (the twin). With model = random
    velocity, `sigmas` are sigma_u in m/s and T_L is memory_time_s (D030); the rows keep
    the ladder value in `sigma` and say which model in `model`.

    `sigma_u_per_window` (one value per window of `windows`) runs a single rung instead, every
    window at its own sigma_u (D033's rule); each row's `sigma` is that window's value and
    its `rung` is "rule".
    """
    if model not in (RANDOM_WALK, RANDOM_VELOCITY):
        raise ValueError(f"model must be {RANDOM_WALK!r} or {RANDOM_VELOCITY!r}, got {model!r}")
    per_window = None
    if sigma_u_per_window is not None:
        per_window = np.asarray(sigma_u_per_window, dtype=float)
        if model != RANDOM_VELOCITY or per_window.shape != (len(windows.table),):
            raise ValueError("a sigma_u per window needs the random velocity and one value "
                             "for every window")
        sigmas = [float("nan")]
    tab = windows.table
    t_lat, t_lon, t_ok = truth if truth is not None else (windows.lat, windows.lon, windows.ok)
    out = []
    for b, rows in batches:
        t0 = tab["t0"].iloc[rows[0]].to_datetime64()
        end = t0 + np.timedelta64(max(leads), "h")
        lat0 = np.repeat(tab["lat0"].to_numpy()[rows], particles)
        lon0 = np.repeat(tab["lon0"].to_numpy()[rows], particles)
        try:
            with open_forcing(t0, end) as forcing:
                for j, sigma in enumerate(sigmas):
                    seq = np.random.SeedSequence([seed, b, j])
                    walk = sigma if model == RANDOM_WALK else 0.0
                    flight = sigma if model == RANDOM_VELOCITY else 0.0
                    if per_window is not None:
                        flight = np.repeat(per_window[rows], particles)
                    m = run_batch(forcing, t0, lat0, lon0, max(leads), LEEWAY_COEFFICIENT,
                                  walk, seq, leads, sigma_u=flight, memory_time_s=memory_time_s)
                    pos = m["pos"].reshape(len(leads), len(rows), particles, 2)
                    alive = m["alive"].reshape(len(leads), len(rows), particles)
                    for i, lead in enumerate(leads):
                        for r in score_rows(rows, sigma, lead, pos[i], alive[i],
                                            t_lat[rows, lead], t_lon[rows, lead],
                                            t_ok[rows, lead], bandwidths, subsample):
                            extra = {}
                            if per_window is not None:
                                extra = {"sigma": float(per_window[r["w"]]), "rung": "rule"}
                            out.append({**r, **extra, "tag": tag, "model": model,
                                        "memory_h": memory_time_s / 3600.0})
        except FORCING_ERRORS as error:
            log(f"batch {b} at {t0}: skipped, {type(error).__name__}: {error}")
    return pd.DataFrame(out)


def fake_buoys(windows: Windows, open_forcing, batches, sigma_true: float, seed: int,
               slide: float = 0.0):
    """The twin's buoys: one engine particle per window with a KNOWN sigma.

    Returns (lat, lon, ok) shaped like windows.lat. With `slide`, each fake buoy also
    slides that fraction of the wind across it, to a random fixed side: a fake person.
    """
    hours = windows.hours
    tab = windows.table
    lat = np.full_like(windows.lat, np.nan)
    lon = np.full_like(windows.lon, np.nan)
    ok = np.zeros_like(windows.ok)
    for b, rows in batches:
        t0 = tab["t0"].iloc[rows[0]].to_datetime64()
        lat0, lon0 = tab["lat0"].to_numpy()[rows], tab["lon0"].to_numpy()[rows]
        rng = np.random.default_rng(np.random.SeedSequence([seed, b, 7]))
        try:
            with open_forcing(t0, t0 + np.timedelta64(hours, "h")) as forcing:
                if slide:
                    forcing = SlidingForcing(forcing, slide, rng.choice([-1.0, 1.0], len(rows)))
                m = run_batch(forcing, t0, lat0, lon0, hours, LEEWAY_COEFFICIENT, sigma_true,
                              rng, range(hours + 1))
        except FORCING_ERRORS:
            continue
        lat[rows], lon[rows] = m["pos"][..., 0].T, m["pos"][..., 1].T
        ok[rows] = m["alive"].T
    return lat, lon, ok


# ---------------------------------------------------------------------------- statistics

def boot_weights(labels, n_boot: int = N_BOOT, seed: int = 0) -> np.ndarray:
    """(n_boot, n) weights from resampling whole clusters (groups or months).

    Windows of one buoy, or of buoys that shared water, are not independent samples, so
    the confidence interval resamples the cluster, never the single window.
    """
    codes, uniques = pd.factorize(pd.Series(labels))
    g = len(uniques)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, g, (n_boot, g))
    counts = np.stack([np.bincount(d, minlength=g) for d in draws])
    return counts[:, codes].astype(float)


def _wmean(x, w):
    return (w @ x) / w.sum(axis=-1)


def _wvar(x, w):
    m = _wmean(x, w)
    return _wmean(x * x, w) - m * m


def _wquantile(x, w, q):
    order = np.argsort(x)
    xs, ws = x[order], w[..., order]
    cum = np.cumsum(ws, axis=-1)
    at = (cum >= q * cum[..., -1:]).argmax(axis=-1)
    return xs[at]


def horizon_stats(p: pd.DataFrame, hour: int, w) -> dict:
    """sigma by every estimator from the gaps at one horizon, per weight row."""
    if hour <= 0:
        raise ValueError(f"hour must be positive, got {hour}")
    if len(p) == 0:
        raise ValueError("no residuals to fit: the table is empty")
    t = hour * 3600.0
    persist = p["sep_persist"].to_numpy(float)
    has_v = np.isfinite(persist)
    e, n = p["gap_e"].to_numpy(float), p["gap_n"].to_numpy(float)
    dw, cw = p["gap_dw"].to_numpy(float), p["gap_cw"].to_numpy(float)
    sep = p["sep"].to_numpy(float)
    run_len2 = (p["run_e"].to_numpy(float) ** 2 + p["run_n"].to_numpy(float) ** 2)
    return {
        "sigma_raw": np.sqrt(_wmean(e * e + n * n, w) / (2 * t)),
        "sigma_debiased": np.sqrt((_wvar(dw, w) + _wvar(cw, w)) / (2 * t)),
        "sigma_quantile": _wquantile(sep, w, LEVEL) / (RADIUS_FACTOR * np.sqrt(t)),
        "sigma_downwind": np.sqrt(_wvar(dw, w) / t),
        "sigma_crosswind": np.sqrt(_wvar(cw, w) / t),
        "median_sep_km": _wquantile(sep, w, 0.5) / 1e3,
        "p90_sep_km": _wquantile(sep, w, LEVEL) / 1e3,
        "model_lead_downwind_km": -_wmean(dw, w) / 1e3,
        "mean_crosswind_km": _wmean(cw, w) / 1e3,
        "wind_run_km": np.sqrt(_wmean(run_len2, w)) / 1e3,
        # gap_dw ~ (alpha_buoy - alpha_model) |wind run| + current error, so through the
        # origin this is how much more (or less) of the wind the buoy feels than the model.
        "windage_minus_model_pct": 100 * (w @ (dw * np.sqrt(run_len2))) / (w @ run_len2),
        "median_still_km": _wquantile(p["sep_still"].to_numpy(float), w, 0.5) / 1e3,
        # Only where the buoy's first velocity is known; GDP serves NaN for a few fixes.
        "median_persist_km": _wquantile(persist[has_v], w[..., has_v], 0.5) / 1e3,
    }


def _ci(point, boots) -> dict:
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"value": float(point), "ci95": [float(lo), float(hi)]}


def with_ci(stat, p: pd.DataFrame, labels: dict, n_boot=N_BOOT, seed=0) -> dict:
    """stat(p, w) at unit weight, with a CI from each clustering, and the wider of them."""
    ones = np.ones(len(p))
    point = stat(p, ones)
    boots = {name: stat(p, boot_weights(lab, n_boot, seed)) for name, lab in labels.items()}
    out = {}
    for key in point:
        cis = {name: _ci(point[key], b[key])["ci95"] for name, b in boots.items()}
        out[key] = {"value": float(point[key]), **{f"ci95_by_{k}": v for k, v in cis.items()},
                    "ci95": [min(c[0] for c in cis.values()), max(c[1] for c in cis.values())]}
    return out


def beta_fit(rows: pd.DataFrame, w_labels: dict, hours=BETA_HOURS, n_boot=N_BOOT) -> dict:
    """The growth exponent: the slope of log(mean squared gap) against log(t).

    Uses windows valid at every hour in the range, so the same windows make every point.
    1 is a random walk, 2 is a velocity error that persists; sigma * sqrt(t) is right at
    one horizon only unless this is near 1.
    """
    lo, hi = hours
    span = rows[(rows["hour"] >= lo) & (rows["hour"] <= hi)]
    complete = span.groupby("w")["valid"].all()
    keep = complete.index[complete.to_numpy()]
    span = span[span["w"].isin(keep) & span["valid"]]
    sq = (span["gap_e"].astype(float) ** 2 + span["gap_n"].astype(float) ** 2)
    x = span.assign(sq=sq.to_numpy()).pivot(index="w", columns="hour", values="sq")
    hours_used = x.columns.to_numpy(int)
    x = x.to_numpy()
    logt = np.log(hours_used.astype(float))

    def slope(w):
        msd = np.log((w @ x) / w.sum(axis=-1)[..., None])
        lt = logt - logt.mean()
        return (msd - msd.mean(axis=-1, keepdims=True)) @ lt / (lt @ lt)

    labels = {k: v.loc[keep].to_numpy() for k, v in w_labels.items()}
    point = float(slope(np.ones(len(keep))))
    cis = {k: list(np.percentile(slope(boot_weights(lab, n_boot)), [2.5, 97.5]))
           for k, lab in labels.items()}
    ci = [float(min(v[0] for v in cis.values())), float(max(v[1] for v in cis.values()))]
    return {"value": point, "windows": int(len(keep)),
            **{f"ci95_by_{k}": [float(a) for a in v] for k, v in cis.items()},
            "ci95": ci,
            # A random walk grows its mean squared gap as t^1. If 1 is outside the CI, a
            # single sigma is right at the calibration horizon only.
            "random_walk_consistent": bool(ci[0] <= 1.0 <= ci[1]),
            "msd_km2_by_hour": dict(zip(hours_used.tolist(),
                                        (np.mean(x, axis=0) / 1e6).round(3).tolist()))}


def gate(p: pd.DataFrame, group_col="group") -> dict:
    """Does the model beat both naive forecasts? Paired, with a sign test over groups.

    Windows are not independent, so the test counts GROUPS whose median paired difference
    is below zero, not windows: D025's paired one-sided sign test, made honest about
    clustering.
    """
    out = {}
    for name, col in (("stationary", "sep_still"), ("persistence", "sep_persist")):
        q = p[np.isfinite(p[col])]
        diff = (q["sep"] - q[col]).astype(float)
        per_group = diff.groupby(q[group_col]).median()
        k, g = int((per_group < 0).sum()), int(per_group.size)
        test = stats.binomtest(k, g, 0.5, alternative="greater")
        out[name] = {"windows": int(len(q)), "model_better_windows": float((diff < 0).mean()),
                     "groups": g, "model_better_groups": k, "p_value": float(test.pvalue),
                     "median_model_km": float(q["sep"].median() / 1e3),
                     "median_baseline_km": float(q[col].median() / 1e3)}
    out["passes"] = all(out[k]["p_value"] < 0.05 for k in ("stationary", "persistence"))
    return out


def stratum(speed) -> np.ndarray:
    lo, hi = STRATA_MS
    return np.where(speed < lo, f"<{lo}", np.where(speed < hi, f"{lo}-{hi}", f">{hi}"))


def load_stage1(windows: Windows, stage1_dir) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = Path(stage1_dir)
    rows = pd.concat([pd.read_parquet(f) for f in sorted(d.glob("rows-*.parquet"))],
                     ignore_index=True)
    status = pd.concat([pd.read_parquet(f) for f in sorted(d.glob("status-*.parquet"))],
                       ignore_index=True)
    tab = windows.table.reset_index().rename(columns={"index": "w"})
    st = status.merge(tab[["w", "tier", "group", "month"]], on="w", how="left")
    speed = np.hypot(st["current0_u"].astype(float), st["current0_v"].astype(float))
    st["stratum"] = stratum(speed.to_numpy())
    rows = rows.merge(st[["w", "tier", "group", "month", "stratum"]], on="w", how="left")
    return rows, st


def fit(windows: Windows, stage1_dir, n_boot=N_BOOT, hour: int = CALIBRATION_HOUR,
        horizons=HORIZONS_H, beta_hours=BETA_HOURS) -> dict:
    """Everything stage 1 says: Table A, the gate, sigma by horizon, beta, sigma_0.

    The gate, the strata, sigma_0 and the crosswind slide are all at `hour`; Table A covers
    `horizons` and `hour` together.
    """
    rows, status = load_stage1(windows, stage1_dir)
    valid = rows[rows["valid"]]
    horizons = tuple(sorted(set(int(h) for h in horizons) | {int(hour)}))
    out = {"hour": int(hour), "windows": {"total": int(len(windows)),
                       "run": int((status["status"] == "ok").sum()),
                       "skipped": status.loc[status["status"] != "ok", "status"]
                       .value_counts().to_dict()},
           "table_a": {}, "sigma": {}, "beta": {}, "gate": {}, "strata": {}}
    for tier in ("drogued", "undrogued"):
        for alpha in ALPHAS:
            key = f"{tier}, alpha={alpha:g}"
            sub = valid[(valid["tier"] == tier) & np.isclose(valid["alpha"], alpha)]
            out["table_a"][key] = {}
            for h in horizons:
                p = sub[sub["hour"] == h]
                labels = {"group": p["group"].to_numpy(), "month": p["month"].to_numpy()}
                s = with_ci(lambda q, w: horizon_stats(q, h, w), p, labels, n_boot)
                s["windows"], s["groups"] = int(len(p)), int(p["group"].nunique())
                s["K_m2_s"] = {"value": s["sigma_debiased"]["value"] ** 2 / 2}
                out["table_a"][key][h] = s
            every = sub
            labels = {"group": every.groupby("w")["group"].first(),
                      "month": every.groupby("w")["month"].first()}
            out["beta"][key] = beta_fit(rows[(rows["tier"] == tier)
                                             & np.isclose(rows["alpha"], alpha)], labels,
                                        hours=tuple(beta_hours), n_boot=n_boot)
            at_t = sub[sub["hour"] == hour]
            out["gate"][key] = gate(at_t)
            out["strata"][key] = {}
            for name, p in at_t.groupby("stratum"):
                labels = {"group": p["group"].to_numpy(), "month": p["month"].to_numpy()}
                s = with_ci(lambda q, w: horizon_stats(q, hour, w), p, labels,
                            n_boot)
                out["strata"][key][name] = {
                    "windows": int(len(p)),
                    **{k: s[k] for k in ("sigma_quantile", "sigma_debiased", "median_sep_km")}}
    head = out["table_a"][f"undrogued, alpha={LEEWAY_COEFFICIENT:g}"][hour]
    out["sigma_0"] = head["sigma_quantile"]["value"]
    out["sigma_c"] = crosswind_sigma(valid, n_boot, hour)
    return out


def crosswind_sigma(valid: pd.DataFrame, n_boot=N_BOOT, hour=CALIBRATION_HOUR) -> dict:
    """Stage 3: the person's crosswind slide as a sigma, matched on the crosswind axis at T.

    The slide is +/- CROSSWIND_SLIDE x the wind run, at right angles to it, so at T its
    variance across the wind is (a_c |wind run|)^2. A random walk has sigma^2 T on that
    axis, so sigma_c = a_c sqrt(E|wind run|^2) / sqrt(T). The isotropic match (half that
    variance per axis) is reported beside it; the twin says which the coverage follows.
    """
    p = valid[(valid["tier"] == "undrogued") & np.isclose(valid["alpha"], LEEWAY_COEFFICIENT)
              & (valid["hour"] == hour)]
    run2 = (p["run_e"].astype(float) ** 2 + p["run_n"].astype(float) ** 2).to_numpy()
    t = hour * 3600.0

    def stat(_, w):
        return {"crosswind_axis": CROSSWIND_SLIDE * np.sqrt(_wmean(run2, w) / t),
                "isotropic": CROSSWIND_SLIDE * np.sqrt(_wmean(run2, w) / (2 * t)),
                "rms_wind_ms": np.sqrt(_wmean(run2, w)) / t}
    return with_ci(stat, p, {"group": p["group"].to_numpy()}, n_boot)


def coverage_analytic(sep, sigma: float, hour: int, level: float = LEVEL) -> float:
    """The fraction of gaps inside a round Gaussian cloud's level region, sigma sqrt(t)
    per axis: the check on stage 1's formula before any ensemble is run."""
    radius = sigma * np.sqrt(hour * 3600.0) * np.sqrt(-2.0 * np.log(1.0 - level))
    return float(np.mean(np.asarray(sep, float) <= radius))


# ---------------------------------------------------------------------------- calibration

def crossing(sigmas, values, target) -> float:
    """Where values crosses target, by linear interpolation in log sigma; NaN if it does not."""
    s, v = np.log(np.asarray(sigmas, float)), np.asarray(values, float)
    for k in range(len(s) - 1):
        if (v[k] - target) * (v[k + 1] - target) <= 0 and v[k] != v[k + 1]:
            return float(np.exp(s[k] + (target - v[k]) * (s[k + 1] - s[k]) / (v[k + 1] - v[k])))
    return float("nan")


def parabola_minimum(sigmas, values) -> float:
    """The minimum of a parabola in log sigma through the lowest point and its neighbours."""
    s, v = np.log(np.asarray(sigmas, float)), np.asarray(values, float)
    k = int(np.clip(np.argmin(v), 1, len(s) - 2))
    a, b, _ = np.polyfit(s[k - 1:k + 2], v[k - 1:k + 2], 2)
    return float(np.exp(-b / (2 * a))) if a > 0 else float(np.exp(s[np.argmin(v)]))


def coverage_table(ladder_rows: pd.DataFrame, windows: Windows, lead=CALIBRATION_HOUR,
                   level=LEVEL, n=None, bandwidth=1.0) -> tuple[np.ndarray, pd.DataFrame]:
    """Windows x sigma: inside the level region (bool), for windows scored at every sigma."""
    q = ladder_rows[(ladder_rows["lead"] == lead) & np.isclose(ladder_rows["bandwidth"], bandwidth)]
    if n is not None:
        q = q[q["n"] == n]
    inside = (q.assign(inside=q["rank"] <= level)
              .pivot_table(index="w", columns="sigma", values="inside", aggfunc="first"))
    inside = inside.dropna()
    tab = windows.table.iloc[inside.index.to_numpy()]
    return inside.columns.to_numpy(float), inside.astype(float).assign(
        group=tab["group"].to_numpy(), month=tab["month"].to_numpy(), tier=tab["tier"].to_numpy())


def calibrate_sigma_star(ladder_rows, windows, tier="undrogued", n_boot=N_BOOT,
                         hour: int = CALIBRATION_HOUR) -> dict:
    """sigma* where coverage of the 90 % region is 90 % at `hour`, with clustered CIs."""
    sigmas, tab = coverage_table(ladder_rows, windows, lead=hour)
    tab = tab[tab["tier"] == tier]
    x = tab[list(sigmas)].to_numpy()
    cover = x.mean(axis=0)
    point = crossing(sigmas, cover, LEVEL)
    es = (ladder_rows[(ladder_rows["lead"] == hour)
                      & np.isclose(ladder_rows["bandwidth"], 1.0)]
          .groupby("sigma")["energy_m"].mean())
    out = {"hour": int(hour), "sigmas": sigmas.tolist(), "coverage": cover.round(4).tolist(),
           "windows": int(len(tab)), "sigma_star": {"value": point},
           "energy_score_m": es.round(1).to_dict(),
           "sigma_energy_optimum": parabola_minimum(es.index.to_numpy(), es.to_numpy())}
    for name in ("group", "month"):
        w = boot_weights(tab[name].to_numpy(), n_boot)
        draws = np.array([crossing(sigmas, c, LEVEL) for c in (w @ x) / w.sum(axis=1)[:, None]])
        out["sigma_star"][f"ci95_by_{name}"] = np.nanpercentile(draws, [2.5, 97.5]).tolist()
    out["sigma_star"]["ci95"] = [min(out["sigma_star"]["ci95_by_group"][0],
                                     out["sigma_star"]["ci95_by_month"][0]),
                                 max(out["sigma_star"]["ci95_by_group"][1],
                                     out["sigma_star"]["ci95_by_month"][1])]
    return out


# ---------------------------------------------------------------- sigma_u by the current (D033)

def start_speeds(stage1_dir, n_windows: int) -> np.ndarray:
    """HYCOM's current speed at each window's start, m/s (stage 1's current0); NaN if skipped."""
    status = _read_all(stage1_dir, "status-*.parquet")
    speed = np.full(int(n_windows), np.nan)
    ok = status[status["status"] == "ok"].drop_duplicates("w")
    speed[ok["w"].to_numpy(int)] = np.hypot(ok["current0_u"].astype(float),
                                            ok["current0_v"].astype(float))
    return speed


def speed_bin(speed, edges=SPEED_BINS_MS) -> np.ndarray:
    """Each speed's bin number, 0 to len(edges) - 2; -1 for NaN."""
    speed = np.asarray(speed, float)
    k = np.searchsorted(np.asarray(edges, float), speed, side="right") - 1
    return np.where(np.isfinite(speed), np.clip(k, 0, len(edges) - 2), -1)


def inside_table(ladder_rows: pd.DataFrame, windows: Windows, hour: int, tier="undrogued",
                 level=LEVEL) -> pd.DataFrame:
    """Windows x sigma_u: inside the level region at `hour` (1/0), NaN where not run.

    Unlike `coverage_table` it keeps a window that was not run at every rung: the fast
    windows were laddered higher than the rest (D033), so each bin uses the rungs its own
    windows have.
    """
    q = ladder_rows[(ladder_rows["lead"] == hour) & np.isclose(ladder_rows["bandwidth"], 1.0)]
    q = q[q["n"] == q["n"].max()]
    tab = windows.table
    q = q[tab["tier"].to_numpy()[q["w"].to_numpy(int)] == tier]
    return (q.assign(inside=(q["rank"] <= level).astype(float))
            .pivot_table(index="w", columns="sigma", values="inside", aggfunc="first"))


def _bin_block(inside: pd.DataFrame, rows) -> pd.DataFrame:
    """One bin's windows on the rungs (nearly) all of them were run at, complete rows only."""
    block = inside.loc[rows]
    cols = [c for c in block.columns if block[c].notna().mean() >= 0.95]
    return block[cols].dropna()


def _line(x, y, weight) -> tuple[float, float]:
    """Weighted least squares y = a + b x."""
    w = np.asarray(weight, float)
    X = np.column_stack([np.ones_like(x), x]) * np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(X, np.asarray(y, float) * np.sqrt(w), rcond=None)
    return float(coef[0]), float(coef[1])


def _rung_sets(block: pd.DataFrame):
    """Windows grouped by which rungs they were run at: (rows, log rungs, inside values)."""
    mask = block.notna().to_numpy()
    values = block.to_numpy()
    rungs = np.log(block.columns.to_numpy(float))
    sets = {}
    for i, m in enumerate(map(tuple, mask)):
        sets.setdefault(m, []).append(i)
    return [(np.array(rows), rungs[np.array(m)], values[np.array(rows)][:, np.array(m)])
            for m, rows in sets.items() if any(m)]


def _inside_at(sets, n: int, log_sigma) -> np.ndarray:
    """Each window's chance of being inside at its own sigma: its ladder, interpolated in log
    sigma between the two rungs either side (the end rung's value beyond the ladder)."""
    out = np.empty(n)
    for rows, lc, vals in sets:
        if len(lc) == 1:
            out[rows] = vals[:, 0]
            continue
        x = np.clip(log_sigma[rows], lc[0], lc[-1])
        j = np.clip(np.searchsorted(lc, x, side="right") - 1, 0, len(lc) - 2)
        f = (x - lc[j]) / (lc[j + 1] - lc[j])
        r = np.arange(len(rows))
        out[rows] = vals[r, j] * (1.0 - f) + vals[r, j + 1] * f
    return out


SHAPES = ("quadrature", "linear")


def shape_core(shape: str, a, b, speed):
    """The rule's core before the slide: a + b s, or sqrt(a^2 + (b s)^2).

    quadrature: a floor of model error that does not care about the current, and an error
    proportional to the current, independent of each other, so their variances add (as the
    crosswind slide's already does). Flat in slow water, then rising with slope b.
    """
    if shape == "linear":
        return a + b * speed
    if shape == "quadrature":
        return np.hypot(a, b * speed)
    raise ValueError(f"shape must be one of {SHAPES}, got {shape!r}")


def fit_speed_rule(ladder_rows: pd.DataFrame, windows: Windows, speeds, hour: int = 4,
                   edges=SPEED_BINS_MS, min_groups: int = MIN_GROUPS, slide: dict | None = None,
                   n_boot: int = N_BOOT, seed: int = 0, shapes=SHAPES) -> dict:
    """sigma_u as a function of the start speed, chosen so every speed bin is covered 90 % (D033).

    Two steps, both resampled by whole shared-water groups for their intervals:

      1. PER BIN, what one sigma_u would do: sigma_u*, where the share of buoys inside the
         90 % region crosses 90 % (`crossing`). It shows the shape, and seeds the fit. A bin
         counts only with at least `min_groups` groups.
      2. EACH SHAPE, fitted directly (`shape_core`, two numbers a and b). Every window gets
         its own sigma_u = core(a, b, min(speed, cap)), its chance of being inside read off
         its own ladder between the rungs either side, and (a, b) minimise sum over bins of
         groups x (coverage - 90 %)^2: the confirmation's own test made the objective. The
         shape with the smaller loss is `chosen`; both are reported. On windows planted with
         a known line (9 Oct) the fit gives it back inside its 95 % interval.

    The cap is the 95th percentile of the start speeds in the fastest bin used: the rule is
    never extrapolated into faster water than the buoys measured, and faster windows are
    fitted at the cap too, as the engine will run them.
    """
    from scipy.optimize import minimize

    inside = inside_table(ladder_rows, windows, hour)
    tab = windows.table
    w_all = inside.index.to_numpy(int)
    groups = tab["group"].to_numpy()[w_all]
    sp = np.asarray(speeds, float)[w_all]
    bins = speed_bin(sp, edges)
    boot_w = boot_weights(groups, n_boot, seed)                 # (n_boot, windows)
    out_bins, xs, stars, star_draws, used = [], [], [], [], []
    for k in range(len(edges) - 1):
        rows = np.flatnonzero(bins == k)
        entry = {"from_ms": float(edges[k]), "to_ms": float(edges[k + 1]),
                 "windows": int(len(rows)), "groups": int(len(set(groups[rows]))), "used": False}
        if len(rows):
            entry["median_speed_ms"] = float(np.median(sp[rows]))
        if entry["groups"] >= min_groups:
            block = _bin_block(inside, w_all[rows])
            sig = block.columns.to_numpy(float)
            idx = np.searchsorted(w_all, block.index.to_numpy(int))
            cover = block.to_numpy().mean(axis=0)
            star = crossing(sig, cover, LEVEL)
            wb = boot_w[:, idx]
            boot = np.array([crossing(sig, c, LEVEL)
                             for c in (wb @ block.to_numpy()) / wb.sum(axis=1)[:, None]])
            entry.update({"windows_scored": int(len(block)), "rungs": sig.tolist(),
                          "coverage": cover.round(4).tolist(), "sigma_u_star": star,
                          "ci95": np.nanpercentile(boot, [2.5, 97.5]).tolist()
                          if np.isfinite(boot).any() else [None, None]})
            if np.isfinite(star) and np.isfinite(boot).mean() > 0.95:
                entry["used"] = True
                used.append(k)
                xs.append(entry["median_speed_ms"])
                stars.append(star)
                star_draws.append(boot)
        out_bins.append(entry)
    if len(used) < 2:
        raise ValueError("fewer than two speed bins have a sigma_u*: no line to fit")
    xs, stars = np.array(xs), np.array(stars)
    var = np.nanvar(np.array(star_draws), axis=1)
    a0, b0 = _line(xs, stars, 1.0 / np.maximum(var, 1e-12))

    in_fit = np.isin(bins, used)
    cap = float(np.percentile(sp[bins == used[-1]], 95))
    block = inside.iloc[np.flatnonzero(in_fit)]
    sets = _rung_sets(block)
    n_fit = len(block)
    fit_bins = bins[in_fit]
    s_fit = np.minimum(sp[in_fit], cap)
    g_bin = np.array([out_bins[k]["groups"] for k in used], float)
    members = [fit_bins == k for k in used]
    boot_fit = boot_w[:, np.flatnonzero(in_fit)]

    def coverage(shape, params, weight):
        sigma = shape_core(shape, params[0], params[1], s_fit)
        if np.any(sigma <= 0.0) or params[1] < 0.0:
            return None
        v = _inside_at(sets, n_fit, np.log(sigma))
        return np.array([np.sum(weight[m] * v[m]) / np.sum(weight[m]) for m in members])

    def loss(params, shape, weight):
        c = coverage(shape, params, weight)
        return 1e6 if c is None else float(np.sum(g_bin * (c - LEVEL) ** 2))

    def solve(shape, weight, start):
        r = minimize(loss, start, args=(shape, weight), method="Nelder-Mead",
                     options={"xatol": 1e-5, "fatol": 1e-10, "maxiter": 4000})
        return r.x, float(r.fun)

    ones = np.ones(n_fit)
    fits = {}
    for shape in shapes:
        start = [a0, b0] if shape == "linear" else [float(stars[0]), max(b0, 0.1) * 1.5]
        (a, b), point_loss = solve(shape, ones, start)
        cov = coverage(shape, (a, b), ones)
        draws = np.array([solve(shape, wt, [a, b])[0] for wt in boot_fit
                          if all(wt[m].sum() > 0 for m in members)])
        fits[shape] = {"a": float(a), "b": float(b),
                       "a_ci95": np.percentile(draws[:, 0], [2.5, 97.5]).tolist(),
                       "b_ci95": np.percentile(draws[:, 1], [2.5, 97.5]).tolist(),
                       "loss": point_loss,
                       "coverage_under_rule": {f"{out_bins[k]['from_ms']:g}-"
                                               f"{out_bins[k]['to_ms']:g} m/s": float(c)
                                               for k, c in zip(used, cov)},
                       "sigma_u_at_bins": {f"{x:.3f} m/s": float(shape_core(shape, a, b,
                                                                            min(x, cap)))
                                           for x in xs}}
    chosen = min(fits, key=lambda k: fits[k]["loss"])
    out = {"hour": int(hour), "level": LEVEL, "model": RANDOM_VELOCITY,
           "speed": "HYCOM surface current at the window's start (stage 1 current0), m/s",
           "method": "each window at core(a, b, min(speed, cap)); (a, b) minimise sum over "
                     "bins of groups x (coverage - 0.9)^2; coverage read off each window's "
                     "ladder; the shape with the smaller loss is chosen",
           "bins": out_bins,
           "fits": fits,
           "chosen": chosen,
           "line": {"shape": chosen, "a": fits[chosen]["a"], "b": fits[chosen]["b"],
                    "a_ci95": fits[chosen]["a_ci95"], "b_ci95": fits[chosen]["b_ci95"],
                    "cap_speed_ms": cap,
                    "through_the_bins": {"shape": "linear", "a": a0, "b": b0}},
           "windows_in_fit": int(n_fit), "boot": int(n_boot)}
    if slide is not None:
        out["slide"] = slide
        out["formula"] = ("sigma_u = sqrt(core(a, b, min(speed, cap))^2 + slide^2); "
                          "core = sqrt(a^2 + (b s)^2) or a + b s")
        out["examples"] = {f"{v:g} m/s": float(rule_sigma_u(out, v))
                           for v in (0.0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0)}
    return out


def rule_sigma_u(rule: dict, speed) -> np.ndarray:
    """The rule written by `fit_speed_rule` at these speeds: what the engine will use."""
    line = rule["line"]
    speed = np.clip(np.asarray(speed, float), 0.0, line["cap_speed_ms"])
    core = shape_core(line.get("shape", "linear"), line["a"], line["b"], speed)
    slide = rule.get("slide", {}).get("value", 0.0)
    return np.hypot(core, slide)


def rule_confirmation(confirm_rows: pd.DataFrame, windows: Windows, speeds,
                      edges=SPEED_BINS_MS, tier="undrogued") -> dict:
    """Coverage of the 90 % region by speed bin and lead, every window at its rule sigma_u."""
    q = confirm_rows[np.isclose(confirm_rows["bandwidth"], 1.0)]
    q = q[q["n"] == q["n"].max()]
    tab = windows.table
    q = q[tab["tier"].to_numpy()[q["w"].to_numpy(int)] == tier]
    q = q.assign(bin=speed_bin(np.asarray(speeds, float)[q["w"].to_numpy(int)], edges),
                 group=tab["group"].to_numpy()[q["w"].to_numpy(int)])
    out = {"all": {}, "by_bin": {}}
    for lead, g in q.groupby("lead"):
        out["all"][int(lead)] = {"windows": int(len(g)),
                                 "coverage90": float(np.mean(g["rank"] <= LEVEL)),
                                 "median_area90_km2": float(g["area90_km2"].median())}
    for k, gb in q.groupby("bin"):
        if k < 0:
            continue
        name = f"{edges[k]:g}-{edges[k + 1]:g} m/s"
        out["by_bin"][name] = {
            "groups": int(gb["group"].nunique()),
            **{int(lead): {"windows": int(len(g)), "coverage90": float(np.mean(g["rank"] <= LEVEL)),
                           "sigma_u": float(g["sigma"].median()),
                           "median_area90_km2": float(g["area90_km2"].median())}
               for lead, g in gb.groupby("lead")}}
    return out


def reliability(rows: pd.DataFrame, windows: Windows, levels=LEVELS) -> dict:
    """Coverage at every level, lead and tier, from the rank (inside L iff rank <= L)."""
    tab = windows.table
    q = rows[np.isclose(rows["bandwidth"], 1.0) & (rows["n"] == rows["n"].max())]
    q = q.assign(tier=tab["tier"].to_numpy()[q["w"].to_numpy()])
    out = {}
    for (tier, lead), g in q.groupby(["tier", "lead"]):
        r = g["rank"].to_numpy()
        out.setdefault(tier, {})[int(lead)] = {
            "windows": int(len(g)),
            "coverage": {str(level): float(np.mean(r <= level)) for level in levels},
            "median_area90_km2": float(g["area90_km2"].median()),
            "median_centroid_km": float(g["centroid_m"].median() / 1e3),
            "median_spread_km": float(g["spread_m"].median() / 1e3),
            "mean_beached": float(g["beached"].mean())}
    return out


# ---------------------------------------------------------------------------- command line

def _load_windows(args) -> Windows:
    return Windows.load(args.windows)


def _cmd_stage1(args) -> None:
    w = _load_windows(args)
    t = clock.perf_counter()
    batches = task_batches(w, args.task, args.tasks)
    rows, status = stage1(w, gridded(args.data), batches)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out / f"rows-{args.task:03d}.parquet", index=False)
    status.to_parquet(out / f"status-{args.task:03d}.parquet", index=False)
    print(json.dumps({"task": args.task, "batches": len(batches), "rows": len(rows),
                      "seconds": round(clock.perf_counter() - t, 1)}))


def _cmd_fit(args) -> None:
    result = fit(_load_windows(args), args.stage1, args.boot, args.hour, tuple(args.horizons),
                 tuple(args.beta_hours))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, default=float))
    head = result["table_a"][f"undrogued, alpha={LEEWAY_COEFFICIENT:g}"][args.hour]
    print(json.dumps({"hour": args.hour, "sigma_0": result["sigma_0"],
                      "sigma_c": result["sigma_c"], "beta": result["beta"],
                      "gate": result["gate"], f"undrogued_{args.hour}h": head},
                     indent=1, default=float))


def _cmd_ladder(args) -> None:
    w = _load_windows(args)
    tiers = set(args.tiers)
    chosen = w.table["tier"].isin(tiers).to_numpy().copy()
    speeds = None
    if args.min_start_speed is not None or args.sigma_u_rule:
        if not args.stage1:
            raise SystemExit("--min-start-speed and --sigma-u-rule need --stage1 (the start "
                             "current of every window)")
        speeds = start_speeds(args.stage1, len(w.table))
        if args.min_start_speed is not None:
            chosen &= speeds >= args.min_start_speed
        if args.sigma_u_rule:
            chosen &= np.isfinite(speeds)
    keep = np.flatnonzero(chosen)
    w = w.subset(keep)
    t = clock.perf_counter()
    batches = task_batches(w, args.task, args.tasks, args.days, args.seed)
    model = RANDOM_WALK if args.sigmas else RANDOM_VELOCITY
    per_window = None
    if args.sigma_u_rule:
        per_window = rule_sigma_u(json.loads(Path(args.sigma_u_rule).read_text()), speeds[keep])
    rows = ladder(w, gridded(args.data), batches, args.sigmas or args.sigmas_u or [],
                  args.particles, args.seed, tuple(args.leads),
                  bandwidths=tuple(args.bandwidths), subsample=tuple(args.subsample),
                  model=model, memory_time_s=args.memory_h * 3600.0,
                  sigma_u_per_window=per_window)
    if len(rows):
        rows["w"] = keep[rows["w"].to_numpy()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out / f"ladder-{args.task:03d}.parquet", index=False)
    print(json.dumps({"task": args.task, "batches": len(batches), "rows": len(rows),
                      "seconds": round(clock.perf_counter() - t, 1)}))


def _cmd_twin(args) -> None:
    w = _load_windows(args)
    keep = np.flatnonzero((w.table["tier"] == "undrogued").to_numpy())
    w = w.subset(keep)
    t = clock.perf_counter()
    batches = task_batches(w, args.task, args.tasks, args.days, args.seed)
    cases = [(f"sigma{s:g}", s, 0.0) for s in args.sigmas_true]
    cases += [(f"sigma{s:g}+slide", s, CROSSWIND_SLIDE) for s in args.slide_cases]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    det = {}
    for b, rows in batches:                       # the sigma = 0 model, once per batch
        tab = w.table
        t0 = tab["t0"].iloc[rows[0]].to_datetime64()
        try:
            with gridded(args.data)(t0, t0 + np.timedelta64(w.hours, "h")) as f:
                det[b] = run_batch(f, t0, tab["lat0"].to_numpy()[rows],
                                   tab["lon0"].to_numpy()[rows], w.hours,
                                   LEEWAY_COEFFICIENT, 0.0, None, range(w.hours + 1))
        except FORCING_ERRORS:
            pass
    for name, sigma_true, slide in cases:
        truth = fake_buoys(w, gridded(args.data), batches, sigma_true, args.seed, slide)
        res = []
        for b, rows in batches:
            if b not in det:
                continue
            tab = w.table
            res.append(residual_rows(rows, w.hours, truth[0][rows], truth[1][rows],
                                     truth[2][rows], tab["lat0"].to_numpy()[rows],
                                     tab["lon0"].to_numpy()[rows], tab["ve0"].to_numpy()[rows],
                                     tab["vn0"].to_numpy()[rows], det[b], LEEWAY_COEFFICIENT))
        res = pd.concat(res, ignore_index=True)
        res["w"] = keep[res["w"].to_numpy()]
        res.assign(case=name).to_parquet(out / f"twin-stage1-{name}-{args.task:03d}.parquet",
                                         index=False)
        sigmas = [sigma_true * k for k in LADDER]
        rows_l = ladder(w, gridded(args.data), batches, sigmas, args.particles, args.seed + 1,
                        (args.hour,), truth=truth, tag=name)
        rows_l["w"] = keep[rows_l["w"].to_numpy()]
        rows_l.to_parquet(out / f"twin-ladder-{name}-{args.task:03d}.parquet", index=False)
    print(json.dumps({"task": args.task, "batches": len(batches), "cases": len(cases),
                      "seconds": round(clock.perf_counter() - t, 1)}))


def _read_all(directory, pattern) -> pd.DataFrame:
    files = sorted(Path(directory).glob(pattern))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else None


def twin_summary(windows: Windows, twin_dir, n_boot=N_BOOT,
                 hour: int = CALIBRATION_HOUR) -> dict:
    """What the twin gave back for each planted sigma: stage 1's formula and stage 2's."""
    out = {}
    s1 = _read_all(twin_dir, "twin-stage1-*.parquet")
    lad = _read_all(twin_dir, "twin-ladder-*.parquet")
    tab = windows.table
    for case, rows in s1.groupby("case"):
        p = rows[rows["valid"] & (rows["hour"] == hour)]
        p = p.assign(group=tab["group"].to_numpy()[p["w"].to_numpy()])
        s = with_ci(lambda q, w: horizon_stats(q, hour, w), p,
                    {"group": p["group"].to_numpy()}, n_boot)
        r = rows.assign(group=tab["group"].to_numpy()[rows["w"].to_numpy()],
                        month=tab["month"].to_numpy()[rows["w"].to_numpy()])
        labels = {"group": r.groupby("w")["group"].first()}
        lc = lad[lad["tag"] == case]
        star = calibrate_sigma_star(lc, windows, n_boot=n_boot, hour=hour)
        out[case] = {"windows": int(len(p)),
                     "stage1_sigma_quantile": s["sigma_quantile"],
                     "stage1_sigma_debiased": s["sigma_debiased"],
                     "stage1_sigma_downwind": s["sigma_downwind"],
                     "stage1_sigma_crosswind": s["sigma_crosswind"],
                     "beta": beta_fit(r, labels, n_boot=n_boot),
                     "stage2": star}
    return out


def ladder_model(rows: pd.DataFrame) -> str:
    """Which random term a ladder ran; rows from before D030 have no column and were walks."""
    if "model" not in rows:
        return RANDOM_WALK
    models = set(rows["model"].unique())
    if len(models) != 1:
        raise ValueError(f"a ladder must run one model, these rows hold {sorted(models)}")
    return models.pop()


def slide_for(sigma_c: dict, model: str, hour: int) -> dict:
    """The crosswind slide in the ladder's units: m/s^0.5 for a walk, m/s for a velocity.

    The fit matches the slide as a random walk, sigma_c sqrt(T) = a_c |wind run| at T.
    A slide is a steady velocity a_c |wind|, so as a velocity it is sigma_c / sqrt(T).
    """
    if model == RANDOM_WALK:
        return sigma_c
    root_t = np.sqrt(hour * 3600.0)
    return {"value": float(sigma_c["value"] / root_t),
            "ci95": [float(v / root_t) for v in sigma_c["ci95"]]}


def _cmd_rule(args) -> None:
    w = _load_windows(args)
    lad = pd.concat([_read_all(d, "ladder-*.parquet") for d in args.ladder], ignore_index=True)
    if ladder_model(lad) != RANDOM_VELOCITY:
        raise SystemExit("the rule is fitted on random-velocity ladders (--sigmas-u)")
    speeds = start_speeds(args.stage1, len(w.table))
    slide = None
    if args.fit:
        fit_result = json.loads(Path(args.fit).read_text())
        if int(fit_result.get("hour", 24)) != args.hour:
            raise SystemExit(f"{args.fit} was not fitted at {args.hour} h")
        slide = slide_for(fit_result["sigma_c"]["crosswind_axis"], RANDOM_VELOCITY, args.hour)
    out = fit_speed_rule(lad, w, speeds, args.hour, slide=slide, n_boot=args.boot)
    out["ladders"] = [str(d) for d in args.ladder]
    if args.confirm:
        conf = _read_all(args.confirm, "ladder-*.parquet")
        out["confirmation"] = rule_confirmation(conf, w, speeds)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: out[k] for k in ("line", "slide", "examples") if k in out}, indent=1,
                     default=float))
    for b in out["bins"]:
        print(f"{b['from_ms']:g}-{b['to_ms']:g} m/s: {b['windows']} windows, {b['groups']} "
              f"groups, sigma_u* {b.get('sigma_u_star', float('nan')):.3f} "
              f"{b.get('ci95')} {'used' if b['used'] else ''}")


def _cmd_calibrate(args) -> None:
    w = _load_windows(args)
    fit_result = json.loads(Path(args.fit).read_text())
    # A fit written before --hour existed was made at 24 h.
    fit_hour = int(fit_result.get("hour", 24))
    if fit_hour != args.hour:
        raise SystemExit(f"{args.fit} was fitted at {fit_hour} h, but calibrate is at "
                         f"{args.hour} h: its crosswind slide is matched at {fit_hour} h. "
                         f"Re-run fit with --hour {args.hour}.")
    out = {"hour": args.hour, "fit_sigma_0": fit_result["sigma_0"],
           "sigma_c": fit_result["sigma_c"]}
    if args.ladder:
        lad = _read_all(args.ladder, "ladder-*.parquet")
        out["ladder"] = calibrate_sigma_star(lad, w, n_boot=args.boot, hour=args.hour)
        star = out["ladder"]["sigma_star"]
        model = ladder_model(lad)
        sc = slide_for(fit_result["sigma_c"]["crosswind_axis"], model, args.hour)
        out["model"] = model
        out["sigma_final"] = {
            "value": float(np.hypot(star["value"], sc["value"])),
            "ci95": [float(np.hypot(star["ci95"][0], sc["ci95"][0])),
                     float(np.hypot(star["ci95"][1], sc["ci95"][1]))],
            "slide": sc,
            "rule": ("sqrt(sigma*^2 + sigma_c^2), sigma_c matched on the crosswind axis"
                     if model == RANDOM_WALK else
                     "sqrt(sigma_u*^2 + (sigma_c / sqrt(T))^2): the slide as a velocity")}
    if args.twin:
        out["twin"] = twin_summary(w, args.twin, args.boot, args.hour)
    if args.confirm:
        conf = _read_all(args.confirm, "ladder-*.parquet")
        out["confirm"] = {"sigma": float(conf["sigma"].iloc[0]), "model": ladder_model(conf),
                          "reliability": reliability(conf, w)}
    if args.check:
        chk = _read_all(args.check, "ladder-*.parquet")
        q = chk[chk["lead"] == args.hour]
        out["check"] = {f"n={int(n)}, bandwidth={bw:g}": {
            "windows": int(len(g)), "coverage90": float(np.mean(g["rank"] <= LEVEL)),
            "coverage50": float(np.mean(g["rank"] <= 0.5))}
            for (n, bw), g in q.groupby(["n", "bandwidth"])}
    Path(args.out).write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "twin"}, indent=1, default=float))
    if "twin" in out:
        for case, v in out["twin"].items():
            print(case, "stage1 quantile", round(v["stage1_sigma_quantile"]["value"], 1),
                  "stage2 sigma*", v["stage2"]["sigma_star"])


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, run=True):
        sp.add_argument("--windows", required=True,
                        help="stem written by sar.validate.drift_windows")
        if run:
            sp.add_argument("--data", required=True, help="archive root holding raw/")
            sp.add_argument("--task", type=int, default=0)
            sp.add_argument("--tasks", type=int, default=1)
        sp.add_argument("--out", required=True)

    def horizon(sp):
        sp.add_argument("--hour", type=int, default=CALIBRATION_HOUR,
                        help="the horizon sigma is matched at, hours (P1: 24; D028 amended: 4)")

    s1 = sub.add_parser("stage1", help="sigma = 0 runs: the gap per hour, alpha 0.02 and 0")
    common(s1)
    s1.set_defaults(func=_cmd_stage1)

    f = sub.add_parser("fit", help="Table A, the gate, sigma by horizon, beta, sigma_0")
    common(f, run=False)
    f.add_argument("--stage1", required=True)
    f.add_argument("--boot", type=int, default=N_BOOT)
    horizon(f)
    f.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS_H),
                   help="Table A's hours; --hour is always added")
    f.add_argument("--beta-hours", type=int, nargs=2, default=list(BETA_HOURS),
                   metavar=("FROM", "TO"), help="the hours beta is fitted over")
    f.set_defaults(func=_cmd_fit)

    lad = sub.add_parser("ladder", help="ensembles at each sigma, scored on the 90 % region")
    common(lad)
    rungs = lad.add_mutually_exclusive_group(required=True)
    rungs.add_argument("--sigmas", type=float, nargs="+",
                       help="random-walk sigmas, m/s^0.5 (D028)")
    rungs.add_argument("--sigmas-u", type=float, nargs="+",
                       help="random-velocity sigma_u values, m/s per axis (D030)")
    rungs.add_argument("--sigma-u-rule",
                       help="rule.json: every window at its own sigma_u from the rule (D033)")
    lad.add_argument("--stage1", help="stage 1's folder: the current at each window's start")
    lad.add_argument("--min-start-speed", type=float,
                     help="only windows whose start current is at least this, m/s (D033)")
    lad.add_argument("--memory-h", type=float, default=MEMORY_TIME_S / 3600.0,
                     help=f"T_L for --sigmas-u, hours, default {MEMORY_TIME_S / 3600.0:g}")
    lad.add_argument("--particles", type=int, default=1000)
    lad.add_argument("--seed", type=int, default=20261002)
    lad.add_argument("--tiers", nargs="+", default=["undrogued"])
    lad.add_argument("--leads", type=int, nargs="+", default=list(HORIZONS_H))
    lad.add_argument("--days", type=int, help="only this many random start days")
    lad.add_argument("--bandwidths", type=float, nargs="+", default=[1.0])
    lad.add_argument("--subsample", type=int, nargs="*", default=[])
    lad.set_defaults(func=_cmd_ladder)

    tw = sub.add_parser("twin", help="fake buoys with a known sigma through stages 1 and 2")
    common(tw)
    tw.add_argument("--sigmas-true", type=float, nargs="+", default=[20.0, 50.0, 100.0])
    tw.add_argument("--slide-cases", type=float, nargs="*", default=[50.0])
    tw.add_argument("--particles", type=int, default=1000)
    tw.add_argument("--days", type=int, default=150)
    tw.add_argument("--seed", type=int, default=20261003)
    horizon(tw)
    tw.set_defaults(func=_cmd_twin)

    r = sub.add_parser("rule", help="D033: sigma_u = a + b x start current, from ladders")
    common(r, run=False)
    r.add_argument("--ladder", nargs="+", required=True, help="random-velocity ladder folders")
    r.add_argument("--stage1", required=True)
    r.add_argument("--fit", help="fit.json at --hour, for the crosswind slide")
    r.add_argument("--confirm", help="a ladder run with --sigma-u-rule, to confirm the rule")
    r.add_argument("--boot", type=int, default=N_BOOT)
    horizon(r)
    r.set_defaults(func=_cmd_rule)

    c = sub.add_parser("calibrate", help="sigma*, sigma_final, the twin and the checks")
    common(c, run=False)
    c.add_argument("--fit", required=True)
    c.add_argument("--ladder")
    c.add_argument("--twin")
    c.add_argument("--confirm")
    c.add_argument("--check")
    c.add_argument("--boot", type=int, default=N_BOOT)
    horizon(c)
    c.set_defaults(func=_cmd_calibrate)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
