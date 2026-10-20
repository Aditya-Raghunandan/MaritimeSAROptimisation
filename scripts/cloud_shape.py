"""cloud_shape.py: how the real buoys sit against their clouds, along and across the drift (L41).

The clouds are round (D026's one start point, D030's random velocity the same size east and
north) while the real buoys' errors are stretched along the drift and lag behind the centre
(L41, 13 Oct, 181 held-out buoys). This measures that on any set of truth windows
(`scripts/truth_windows.py`: a real buoy's rvc cloud, saved every 5 min, and its hourly fixes)
and fits what would correct it, so the fit is made on the TRAINING buoys and checked on the
held-out ones.

`measure`, per window and per hour h with a fix (1, 2, 3 h): the cloud's weighted centre c; the
drift direction u, the unit vector from the last known position to c; the buoy's offset from c
along u and across it (left of the drift positive); the cloud's own spread along and across;
and, sampled at the centre every saved frame and integrated from the call to h, the current's
and the 2 % leeway's displacement along and across u. Also whether the buoy is inside the
cloud's 90 % ellipse (Gaussian, chi-square 2 dof: a stand-in for the 90 % region, for before
and after on the same footing).

`fit`, from measure's CSV: the buoy's along offset = b_c x the current's along displacement +
b_w x the leeway's, no intercept, so the model's current is scaled by 1 + b_c and the leeway is
2 % x (1 + b_w); 95 % intervals resampling groups. The correction used is one scale for the whole
drift (the split is not identifiable, see the 18 Oct notebook). Then, per axis, the factor on the
random velocity that puts 90 % of the training buoys inside the new cloud's central 90 %.

    python scripts/cloud_shape.py measure --windows DIR/train --rows DIR/rows_train.csv \\
        --forcing-dir DATA --out shape_train.csv [--part K --parts P]
    python scripts/cloud_shape.py fit --csv shape_train.csv [more.csv ...] --out shape_fit.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sar.model.drift import LEEWAY_COEFFICIENT
from sar.rl.windows import load
from sar.search.sweep import relative_m
from sar.utils.geo import offset_position

CHI2_90_2DOF = 4.605170185988092      # -2 ln(0.1): the 90 % ellipse of a 2-D Gaussian
Z90 = 1.6448536269514722              # the central 90 % of one Gaussian axis
MIN_DRIFT_M = 50.0                    # below this the drift has no direction worth projecting on
SAMPLE_S = 300.0                      # how often the forcing is sampled along the centre's path
BOOTSTRAP_SEED, RESAMPLES = 20261018, 2000


def _row_key(start, lat, lon) -> tuple:
    return (str(np.datetime64(str(start), "m")), round(float(lat), 3), round(float(lon) % 360.0, 3))


def measure_one(w, forcing, group) -> list[dict]:
    """One window file's rows: one per hour with a fix inside the saved cloud."""
    rows = []
    lkp_lat, lkp_lon = w.lkp
    weight = np.asarray(w.weight, dtype=float) / float(np.sum(w.weight))
    east, north = relative_m(w.lat, w.lon, lkp_lat, lkp_lon)          # (F, N) metres from the LKP
    centres = np.column_stack([east @ weight, north @ weight])        # (F, 2)
    # The current and the 2 % leeway at the centre every SAMPLE_S from the call. A windows file
    # keeps frames only from the first arrival on, so before it the centre is taken straight
    # from the last known position at t = 0 to the first frame kept.
    t_all = np.concatenate([[0.0], np.asarray(w.t_s, dtype=float)])
    c_all = np.vstack([[0.0, 0.0], centres])
    grid = np.arange(0.0, t_all[-1] + 1e-6, SAMPLE_S)
    path = np.column_stack([np.interp(grid, t_all, c_all[:, 0]), np.interp(grid, t_all, c_all[:, 1])])
    c_lat, c_lon = offset_position(lkp_lat, lkp_lon, path[:, 0], path[:, 1])
    times = w.start + (grid * 1e6).astype("timedelta64[us]")
    cur, lee = np.empty((len(grid), 2)), np.empty((len(grid), 2))
    for j, t in enumerate(times):
        current, wind = forcing.sample([float(c_lat[j])], [float(c_lon[j]) % 360.0], t)
        cur[j], lee[j] = current[0], LEEWAY_COEFFICIENT * wind[0]
    for h_s, b_lat, b_lon in zip(w.truth_t_s, w.truth_lat, w.truth_lon):
        if h_s <= 0 or h_s > w.t_s[-1] + 1e-6:
            continue
        k = int(np.argmin(np.abs(w.t_s - h_s)))
        if abs(w.t_s[k] - h_s) > 1.0:
            continue
        D = centres[k]
        dist = float(np.hypot(*D))
        upto = grid <= h_s + 1e-6
        if dist < MIN_DRIFT_M or not (np.isfinite(cur[upto]).all() and np.isfinite(lee[upto]).all()):
            continue
        u = D / dist
        v = np.array([-u[1], u[0]])                                   # left of the drift
        off = np.array(relative_m(b_lat, b_lon, lkp_lat, lkp_lon)) - D
        pe, pn = east[k] - D[0], north[k] - D[1]
        pa, pc = pe * u[0] + pn * u[1], pe * v[0] + pn * v[1]
        cov = np.cov(np.vstack([pe, pn]), aweights=weight)
        m2 = float(off @ np.linalg.solve(cov, off))
        C = np.trapezoid(cur[upto], grid[upto], axis=0)
        L = np.trapezoid(lee[upto], grid[upto], axis=0)
        rows.append({
            "name": w.name, "group": group, "hour": round(h_s / 3600.0),
            "drift_m": dist, "along_m": float(off @ u), "across_m": float(off @ v),
            "sd_along_m": float(np.sqrt(weight @ pa ** 2)),
            "sd_across_m": float(np.sqrt(weight @ pc ** 2)), "mahalanobis2": m2,
            "inside90": bool(m2 <= CHI2_90_2DOF),
            "current_along_m": float(C @ u), "current_across_m": float(C @ v),
            "leeway_along_m": float(L @ u), "leeway_across_m": float(L @ v),
        })
    return rows


def measure(windows_dir, rows_csv, forcing_dir, out, part: int, parts: int) -> Path:
    from sar.pipeline.gridded import ForcingGapError, GriddedForcing

    table = pd.read_csv(rows_csv)
    groups = {_row_key(r.start, r.lat, r.lon): int(r.group) for r in table.itertuples()}
    files = sorted(Path(windows_dir).glob("windows_*.npz"))[part::parts]
    rows = []
    for path in files:
        w = load(path)
        if w.truth_t_s is None:
            continue
        group = groups.get(_row_key(w.start, w.lkp[0], w.lkp[1]), -1)
        end = w.start + np.timedelta64(int(w.t_s[-1]), "s")
        try:
            with GriddedForcing.from_dir(forcing_dir, w.start, end) as forcing:
                rows += measure_one(w, forcing, group)
        except ForcingGapError as err:
            print(f"{path.name}: skipped, {err}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{len(files)} files -> {len(rows)} rows: {out}")
    return out


def start_currents(rows_csv, forcing_dir, out, part: int, parts: int) -> Path:
    """Each truth row's model current at its start (D033's s): what a forecaster has at the call."""
    from sar.pipeline.ensemble import start_current_ms
    from sar.rl.windows import WINDOW_S, windows_name
    from sar.search.platform import STEP_S
    from sar.pipeline.gridded import ForcingGapError, GriddedForcing

    rows = pd.read_csv(rows_csv).iloc[part::parts]
    arrivals, particles = [60.0 * m for m in (30, 60, 120, 180)], 5000     # as the truth windows were built
    found = []
    for r in rows.itertuples():
        start = np.datetime64(str(r.start), "us")
        try:
            with GriddedForcing.from_dir(forcing_dir, start, start) as forcing:
                s = start_current_ms(forcing, float(r.lat), float(r.lon), start)
        except (ForcingGapError, ValueError, FileNotFoundError) as err:
            print(f"{r.scenario}: skipped, {err}", flush=True)
            continue
        # The windows file's own name (scripts/truth_windows.py build), so this joins measure's rows.
        run = {"lat": float(r.lat), "lon": float(r.lon), "start": str(r.start), "particles": particles,
               "timestep": STEP_S, "duration": max(arrivals) + WINDOW_S, "seed": int(r.seed)}
        found.append({"name": windows_name(run, arrivals), "scenario": r.scenario,
                      "group": int(r.group), "start_current_ms": float(s)})
    out = Path(out)
    if parts > 1:                                   # one file per part: OUT/part.csv -> OUT/part7.csv
        out = out.with_name(f"{out.stem}{part}{out.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(found).to_csv(out, index=False)
    print(f"{len(found)} rows -> {out}", flush=True)
    return out


def _fit_lag(f: pd.DataFrame) -> np.ndarray:
    X = f[["current_along_m", "leeway_along_m"]].to_numpy()
    return np.linalg.lstsq(X, f["along_m"].to_numpy(), rcond=None)[0]


def _fit_one(f: pd.DataFrame) -> float:
    """One scale for the whole drift: along = b x (current + leeway) along."""
    x = (f["current_along_m"] + f["leeway_along_m"]).to_numpy()
    return float(x @ f["along_m"].to_numpy() / (x @ x))


def _shape(f: pd.DataFrame, b) -> dict:
    resid = f["along_m"] - (b[0] * f["current_along_m"] + b[1] * f["leeway_along_m"])
    out = {}
    for h, g in f.assign(resid=resid).groupby("hour"):
        out[int(h)] = {"rms_along_units": float(np.sqrt(np.mean((g["resid"] / g["sd_along_m"]) ** 2))),
                       "rms_across_units": float(np.sqrt(np.mean((g["across_m"] / g["sd_across_m"]) ** 2))),
                       "rms_along_units_before": float(np.sqrt(np.mean((g["along_m"] / g["sd_along_m"]) ** 2))),
                       "mean_along_m_before": float(g["along_m"].mean()),
                       "mean_along_m_after": float(g["resid"].mean()),
                       "inside90_before": float(g["inside90"].mean()), "rows": int(len(g))}
    return out


def fit(csvs, out) -> dict:
    f = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    f = f[f["group"] >= 0]
    b = _fit_lag(f)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    groups = f["group"].unique()
    by = {g: part for g, part in f.groupby("group")}
    one = _fit_one(f)
    draws, ones = [], []
    for _ in range(RESAMPLES):
        part = pd.concat([by[g] for g in rng.choice(groups, size=groups.size, replace=True)])
        draws.append(_fit_lag(part))
        ones.append(_fit_one(part))
    lo, hi = np.percentile(np.array(draws), [2.5, 97.5], axis=0)
    one_lo, one_hi = np.percentile(ones, [2.5, 97.5])
    # The correction taken forward is the one scale: the two-way split cannot be told apart
    # from a noisy forecast measured in its own direction (18 Oct), so it is reported, not used.
    shape = _shape(f, (one, one))
    # The factors: per axis, the one that puts 90 % of the training buoys within the new cloud's
    # central 90 % (+-1.645 sd), the criterion D030 and D033 calibrated the size on. The first
    # try matched the rms instead (1.21 along, 0.94 across), and a few large errors at 1 h pulled
    # it so far that the held-out cloud covered 96-97 % (18 Oct): rms is not robust here.
    a = (f["along_m"] - one * (f["current_along_m"] + f["leeway_along_m"])) / f["sd_along_m"]
    c = f["across_m"] / f["sd_across_m"]
    along = float(np.quantile(np.abs(a), 0.9) / Z90)
    across = float(np.quantile(np.abs(c), 0.9) / Z90)
    result = {
        "rows": int(len(f)), "groups": int(groups.size), "windows": int(f["name"].nunique()),
        "b_current": float(b[0]), "b_current_ci": [float(lo[0]), float(hi[0])],
        "b_leeway": float(b[1]), "b_leeway_ci": [float(lo[1]), float(hi[1])],
        "two_way_current_scale": 1.0 + float(b[0]),
        "two_way_leeway": LEEWAY_COEFFICIENT * (1.0 + float(b[1])),
        "b_one": one, "b_one_ci": [float(one_lo), float(one_hi)],
        "current_scale": 1.0 + one, "wind_scale": 1.0 + one,
        "along": along, "across": across,
        "rms_along": float(np.mean([s["rms_along_units"] for s in shape.values()])),
        "rms_across": float(np.mean([s["rms_across_units"] for s in shape.values()])),
        "by_hour": shape,
    }
    Path(out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "by_hour"}, indent=1))
    return result


def _needed(f: pd.DataFrame, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Per row, the offset left after the drift scale, along and across, in the round cloud's sd."""
    a = (f["along_m"] - (scale - 1.0) * (f["current_along_m"] + f["leeway_along_m"])) / f["sd_along_m"]
    return np.abs(a.to_numpy()), np.abs((f["across_m"] / f["sd_across_m"]).to_numpy())


SPEED_EDGES = (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.55, 0.8, 99.0)   # finer where the shape is
MIN_BIN_ROWS = 60


def _rule_from_bins(f: pd.DataFrame, scale: float) -> dict:
    """Per speed bin and axis, the sigma that puts 90 % of the bin inside; then one floor a shared by
    both axes and a current part per axis, sigma_axis = sqrt(a^2 + (b_axis s)^2): least squares on
    sigma^2, relative, weighted by rows. The floor is shared because the error that does not grow with the
    current has no reason to know the current's direction, and slow water is round (20 Oct)."""
    from sar.model.position import SIGMA_U_RULE, sigma_u_for_current

    cap = SIGMA_U_RULE["cap_speed_ms"]
    k = np.clip(np.searchsorted(SPEED_EDGES, f["start_current_ms"], side="right") - 1, 0,
                len(SPEED_EDGES) - 2)
    pts = []
    for _, g in f.assign(bin=k).groupby("bin"):
        if len(g) < MIN_BIN_ROWS:
            continue
        al, ac = _needed(g, scale)
        s = float(np.median(np.minimum(g["start_current_ms"], cap)))
        base = sigma_u_for_current(s)
        pts.append((s, len(g), base * np.quantile(al, 0.9) / Z90, base * np.quantile(ac, 0.9) / Z90))
    pts = np.array(pts)
    n, s2 = pts[:, 1], pts[:, 0] ** 2
    need = np.concatenate([pts[:, 2] ** 2, pts[:, 3] ** 2])
    # Relative error, weighted by rows: on sigma^2 itself the fast bins' large values pulled the
    # floor down and slow water was covered 85 % (20 Oct).
    w = np.sqrt(np.concatenate([n, n])) / need
    X = np.vstack([np.column_stack([np.ones_like(s2), s2, 0 * s2]),
                   np.column_stack([np.ones_like(s2), 0 * s2, s2])]) * w[:, None]
    y = need * w
    a2, ba2, bc2 = np.linalg.lstsq(X, y, rcond=None)[0]
    a = float(np.sqrt(max(a2, 0.0)))
    return {"a_along": a, "b_along": float(np.sqrt(max(ba2, 0.0))), "a_across": a,
            "b_across": float(np.sqrt(max(bc2, 0.0))),
            "bins": [{"s": float(s), "rows": int(m), "sigma_along": float(sa), "sigma_across": float(sc)}
                     for s, m, sa, sc in pts]}


def _isotonic(y, w) -> np.ndarray:
    """Weighted pool-adjacent-violators: the closest non-decreasing sequence."""
    blocks = [[float(v), float(x), 1] for v, x in zip(y, w)]          # value, weight, count
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            v0, w0, c0 = blocks[i]
            v1, w1, c1 = blocks.pop(i + 1)
            blocks[i] = [(v0 * w0 + v1 * w1) / (w0 + w1), w0 + w1, c0 + c1]
            i = max(i - 1, 0)
        else:
            i += 1
    return np.concatenate([[v] * c for v, _, c in blocks])


def _table(bins: list[dict]) -> dict:
    """The table rule: each bin's 90 % sigma per axis, made non-decreasing in speed."""
    s = np.array([b["s"] for b in bins]); n = np.array([b["rows"] for b in bins], dtype=float)
    return {"table_s": s.tolist(),
            "table_along": _isotonic([b["sigma_along"] for b in bins], n).tolist(),
            "table_across": _isotonic([b["sigma_across"] for b in bins], n).tolist()}


def fit_rule(csvs, s0_csvs, out) -> dict:
    """D035: the shape by the current. Joins measure's rows to each window's start current."""
    from sar.model.position import shape_for_current

    f = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    s0 = pd.concat([pd.read_csv(c) for c in s0_csvs], ignore_index=True)[["name", "start_current_ms"]]
    f = f[f["group"] >= 0].merge(s0.drop_duplicates("name"), on="name")
    scale = 1.0 + _fit_one(f)
    rule = _rule_from_bins(f, scale)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    groups = f["group"].unique()
    by = {g: part for g, part in f.groupby("group")}
    draws = []
    for _ in range(500):
        part = pd.concat([by[g] for g in rng.choice(groups, size=groups.size, replace=True)])
        r = _rule_from_bins(part, 1.0 + _fit_one(part))
        draws.append([r["a_along"], r["b_along"], r["a_across"], r["b_across"], r["b_along"] - r["b_across"]])
    lo, hi = np.percentile(np.array(draws), [2.5, 97.5], axis=0)
    keys = ("a_along", "b_along", "a_across", "b_across", "b_along_minus_b_across")
    curve = {k: rule[k] for k in keys[:4]}
    # The rule taken forward is the table: one curve cannot cover slow and fast water at once
    # (on sigma^2 slow water got 85 %, on relative error fast water 78 %, 20 Oct).
    rule_only = _table(rule["bins"])
    # The check on the same rows: per axis and per bin, the share inside the new cloud's central 90 %.
    a, c = _needed(f, scale)
    fa, fc = np.array([shape_for_current(s, rule_only) for s in f["start_current_ms"]]).T
    band = pd.cut(f["start_current_ms"], [0, 0.3, 1.0, 99])
    check = (pd.DataFrame({"band": band, "along": a <= Z90 * fa, "across": c <= Z90 * fc,
                           "ellipse": (a / fa) ** 2 + (c / fc) ** 2 <= CHI2_90_2DOF})
             .groupby("band", observed=True).mean().round(3))
    result = {"rows": int(len(f)), "windows": int(f["name"].nunique()), "groups": int(groups.size),
              "current_scale": scale, "wind_scale": scale, "rule": rule_only, "curve": curve,
              "rule_ci": {k: [float(l), float(h)] for k, l, h in zip(keys, lo, hi)},
              "fraction_of_draws_stretched": float(np.mean(np.array(draws)[:, 4] > 0)),
              "bins": rule["bins"], "check_by_band": check.reset_index().astype({"band": str}).to_dict("records")}
    Path(out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "bins"}, indent=1))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--windows", required=True)
    m.add_argument("--rows", required=True)
    m.add_argument("--forcing-dir", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--part", type=int, default=0)
    m.add_argument("--parts", type=int, default=1)
    s = sub.add_parser("start-current")
    s.add_argument("--rows", required=True)
    s.add_argument("--forcing-dir", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--part", type=int, default=0)
    s.add_argument("--parts", type=int, default=1)
    r = sub.add_parser("fit-rule", help="D035: per-axis sigma by the start current")
    r.add_argument("--csv", nargs="+", required=True)
    r.add_argument("--s0", nargs="+", required=True)
    r.add_argument("--out", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--csv", nargs="+", required=True)
    f.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "measure":
        measure(args.windows, args.rows, args.forcing_dir, args.out, args.part, args.parts)
        # On the cluster the interpreter finished its work and then never exited, its forcing
        # readers' threads still alive (job 59970, 18 Oct), so a shell waiting on it hung. The
        # CSV is written and closed by now.
        sys.stdout.flush()
        os._exit(0)
    elif args.command == "fit-rule":
        fit_rule(args.csv, args.s0, args.out)
    elif args.command == "start-current":
        start_currents(args.rows, args.forcing_dir, args.out, args.part, args.parts)
        sys.stdout.flush()
        os._exit(0)
    else:
        fit(args.csv, args.out)


if __name__ == "__main__":
    main()
