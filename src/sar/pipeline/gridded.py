"""gridded.py: the real HYCOM current and ERA5 wind at every particle (issue #88).

The drift engine asks one question every step, `sample(lats, lons, time)`: the current and
the wind at each particle, as two (N, 2) arrays in m/s. `ConstantForcing` answers it with
one invented value. This answers it from the stored files, the same way
`sar.model.interpolate.sample_field` does for one point, for every particle at once.

THE MATHS. Bilinear in space: a particle's value is a weighted average of the four grid
points around it, each weighted by closeness, which is straight-line interpolation along
the cell's bottom and top edges and then between them. Linear in time between the two
stored slices either side of `time`. Components (u, v) are interpolated separately, never
speed and direction. docs/gridded-forcing.md works one example through.

WHY NOT LOOP OVER sample_field. It costs 2.58 ms a point (laptop, 29 Sep 2026): 41 hours
of sampling for one 48 h run at 10^4 particles. And it raises on a land corner, so the
first particle near the coast would stop the whole ensemble.

WHERE THE WORK GOES. The two slices around `time` change only when time crosses a stored
step: every 180 steps for HYCOM, 60 for ERA5. So the grid-sized work is done then, once:
a table holding [u now, v now, u next, v next] for every grid point, land set to zero, and
a mask of which points are wet. Each step then costs four lookups per particle, one per
corner, and a blend. Blending the whole grid every step would cost 113,000 points a call
however few particles there are, which at N = 1 (#89 calls it ~38 million times) is hours.

LAND (D016, the rule Aditya chose on 29 Sep 2026). A particle whose NEAREST grid point is
land has entered a land cell: its current is NaN, and `DriftPipeline` freezes it with its
mass. Otherwise only the wet corners are used and their weights renormalised. The nearest
corner always carries at least a quarter of the weight, so that never divides by zero.
One refinement: HYCOM is a C-grid, where water crosses cell EDGES, never corners, so the
diagonal corner counts only if an edge neighbour is wet. Without it, water on the far side
of a one-point-thick barrier of cays leaks into the average (0.198 m/s instead of 1 in the
test that pins it).

GAPS (D011). HYCOM's 6 h gaps are blended across; its 12 h gap on 2020-10-25 is refused
with `ForcingGapError`. `require(start, end)` checks a whole run's window before it starts,
because D011 excludes whole scenarios; the per-step check is a backstop.

THE BOX. Outside the grids both fields are NaN, and `outside()` says which particles those
were, so out of domain can be told from beached. The box is where BOTH grids are: HYCOM's
last longitude is 296.96, not 297 (measured 29 Sep), while ERA5 reaches 297.

CLI. Samples random points in the box at one time and prints what it found, as JSON;
`--check K` compares K all-sea points with `sample_field` on the real files:

    python -m sar.pipeline.gridded --forcing-dir C:/maritime-data \\
        --time 2021-01-05T07:30 --particles 100000 [--seed 1] [--check 1000]
"""

from __future__ import annotations

import argparse
import json
import re
import time as clock
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sar.fetch.current import box_tag
from sar.model.interpolate import (
    CURRENT_VARS,
    EDGE_EPS,
    WIND_VARS,
    MissingCornerError,
    OutOfCoverageError,
    sample_field,
)
from sar.pipeline.forcing import as_particle_axes
from sar.utils.data_io import open_forcing
from sar.utils.geo import regular_axis_step, to_display_longitude

# D011: blend across HYCOM's 6 h gaps, refuse longer. ERA5 had no gaps (cluster, 29 Sep).
MAX_GAP_S = {"current": 6 * 3600, "wind": 3600}

PRODUCTS = {"current": ("hycom", CURRENT_VARS), "wind": ("era5", WIND_VARS)}

_NS = 1_000_000_000
_FILE_DATES = re.compile(r"_(\d{8})-(\d{8})\.nc$")
_SLICES_KEPT = 3


class ForcingGapError(OutOfCoverageError):
    """The stored steps either side of a time are further apart than D011 allows."""


def as_ns(time) -> int:
    """A time as integer nanoseconds since 1970, so time fractions are exact."""
    t = np.datetime64(time, "ns")
    if np.isnat(t):
        raise ValueError(f"time must be a real instant, got {time!r}")
    return int(t.astype(np.int64))


@dataclass(frozen=True)
class RegularAxis:
    """One ascending, regularly spaced grid axis: origin, step and number of points."""

    name: str
    origin: float
    step: float
    size: int

    @classmethod
    def from_values(cls, values, name: str) -> RegularAxis:
        values = np.asarray(values, dtype=float)
        step = regular_axis_step(values, name)
        if step <= 0.0:
            raise ValueError(f"{name} must ascend (D020), got a step of {step}")
        return cls(name, float(values[0]), float(step), int(values.size))

    @property
    def last(self) -> float:
        return self.origin + self.step * (self.size - 1)

    def inside(self, x) -> np.ndarray:
        """True where x is on the axis, to within EDGE_EPS; NaN is never inside."""
        x = np.asarray(x, dtype=float)
        return (x >= self.origin - EDGE_EPS) & (x <= self.last + EDGE_EPS)

    def locate(self, x) -> tuple[np.ndarray, np.ndarray]:
        """The lower grid line of each x's cell, and how far across the cell x is, 0 to 1.

        On the last grid line the cell below is used, with the fraction 1, so every x on
        the axis has a cell. Callers pass only positions that are `inside`.
        """
        pos = (np.asarray(x, dtype=float) - self.origin) / self.step
        i0 = np.clip(np.floor(pos), 0, self.size - 2).astype(np.intp)
        return i0, np.clip(pos - i0, 0.0, 1.0)


class TimeAxis:
    """The stored times of several files as one axis, in integer nanoseconds.

    Each entry remembers which file and which index in it holds that time, so the files
    stay open lazily and are never concatenated: concatenating lazily opened files loads
    them all into memory (measured 29 Sep, ~6.6 GB for the HYCOM archive).
    """

    def __init__(self, per_file, names):
        spans = [np.asarray(t, dtype="datetime64[ns]").astype(np.int64) for t in per_file]
        for name, t in zip(names, spans):
            if t.size == 0 or np.any(np.diff(t) <= 0):
                raise ValueError(f"{name}: times must be present and strictly increasing")
        for (a, ta), (b, tb) in zip(zip(names, spans), zip(names[1:], spans[1:])):
            if tb[0] <= ta[-1]:
                raise ValueError(f"{a} and {b} overlap in time; give one or the other")
        self.ns = np.concatenate(spans)
        self.file = np.concatenate([np.full(t.size, k) for k, t in enumerate(spans)])
        self.local = np.concatenate([np.arange(t.size) for t in spans])

    def bracket(self, t_ns: int) -> tuple[int, float]:
        """The stored step at or before t, and how far t is toward the next, 0 to 1.

        The same contract as `interpolate.bracket_time`: at the final time the fraction
        is 0, and a time outside the files raises OutOfCoverageError.
        """
        if t_ns < self.ns[0] or t_ns > self.ns[-1]:
            first, last = (np.datetime64(int(self.ns[k]), "ns") for k in (0, -1))
            raise OutOfCoverageError(f"time {np.datetime64(t_ns, 'ns')} is outside the files, "
                                     f"which cover {first} to {last}")
        if t_ns == self.ns[-1]:
            return self.ns.size - 1, 0.0
        k = int(np.searchsorted(self.ns, t_ns, side="right") - 1)
        return k, (t_ns - int(self.ns[k])) / (int(self.ns[k + 1]) - int(self.ns[k]))

    def gaps(self, start_ns: int, end_ns: int, longer_than_s: float) -> list[tuple]:
        """Every pair of consecutive steps further apart than longer_than_s that a run
        from start to end would have to blend across, as (before, after, hours)."""
        lo = max(int(np.searchsorted(self.ns, start_ns, side="right")) - 1, 0)
        hi = min(int(np.searchsorted(self.ns, end_ns, side="left")), self.ns.size - 1)
        out = []
        for k in range(lo, hi):
            span = int(self.ns[k + 1]) - int(self.ns[k])
            if span > longer_than_s * _NS:
                out.append((str(np.datetime64(int(self.ns[k]), "ns")),
                            str(np.datetime64(int(self.ns[k + 1]), "ns")), span / 3600 / _NS))
        return out


def cell_weights(fy, fx) -> np.ndarray:
    """The four bilinear weights, (4, N), in corner order lower-left, lower-right,
    upper-left, upper-right. They are the products of straight-line weights on each axis,
    so they sum to 1."""
    return np.stack([(1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx])


def usable_corners(wet, fy, fx) -> tuple[np.ndarray, np.ndarray]:
    """Which corners may be averaged, and which particles are beached.

    wet is (4, N) in the corner order of `cell_weights`. A particle is beached when the
    corner NEAREST it is dry (D016). The diagonal corner counts only through a wet edge
    neighbour, because water in HYCOM's C-grid crosses cell edges, never corners.
    """
    corner = np.arange(4)[:, None]
    near = 2 * (fy >= 0.5) + (fx >= 0.5)            # 0 LL, 1 LR, 2 UL, 3 UR
    diagonal = corner == (near ^ 3)
    # The two corners that are neither the nearest nor its diagonal are its edge neighbours.
    through_an_edge = (wet & (corner != near) & ~diagonal).any(axis=0)
    usable = wet & (~diagonal | through_an_edge)
    return usable, ~(wet & (corner == near)).any(axis=0)


class Field:
    """One product, current or wind: its grid, its files, and the slices it has loaded."""

    def __init__(self, name: str, datasets, var_names, max_gap_s: float):
        if not datasets:
            raise ValueError(f"no {name} files given")
        datasets = sorted(datasets, key=lambda ds: ds["time"].values[0])
        self.name, self.datasets, self.var_names = name, datasets, tuple(var_names)
        self.max_gap_s = float(max_gap_s)
        self.lat = RegularAxis.from_values(datasets[0]["lat"].values, f"{name} lat")
        self.lon = RegularAxis.from_values(datasets[0]["lon"].values, f"{name} lon")
        for ds in datasets[1:]:
            if (not np.allclose(ds["lat"].values, datasets[0]["lat"].values, atol=1e-9)
                    or not np.allclose(ds["lon"].values, datasets[0]["lon"].values, atol=1e-9)):
                raise ValueError(f"{name} files are on different grids: "
                                 f"{self.path(datasets[0])} and {self.path(ds)}")
        self.times = TimeAxis([ds["time"].values for ds in datasets],
                              [self.path(ds) for ds in datasets])
        self.loads = 0
        self._slices: dict[int, np.ndarray] = {}
        self._bracket = (-1, 0, 0)                       # k, its time, the next time (ns)
        self._table = None

    @staticmethod
    def path(ds) -> str:
        return Path(ds.attrs.get("sar_source_path", "?")).name

    def inside(self, lats, lons) -> np.ndarray:
        return self.lat.inside(lats) & self.lon.inside(lons)

    def _slice(self, k: int) -> np.ndarray:
        """Stored step k as a (lat * lon, 2) float32 array of (u, v), read once."""
        if k not in self._slices:
            if len(self._slices) >= _SLICES_KEPT:
                # Drop the slice furthest from k: the oldest when time runs forward, and
                # still the right one when a caller jumps back to an earlier window (#89).
                del self._slices[max(self._slices, key=lambda j: abs(j - k))]
            ds = self.datasets[int(self.times.file[k])]
            local = int(self.times.local[k])
            u, v = (ds[name].isel(time=local).values for name in self.var_names)
            self._slices[k] = np.stack([u, v], axis=-1).reshape(-1, 2).astype(np.float32)
            self.loads += 1
        return self._slices[k]

    def _table_for(self, k: int):
        """The per-bracket table: (values, wet now, wet now and next)."""
        if self._table is None or self._table[0] != k:
            now = self._slice(k)
            nxt = self._slice(k + 1) if k + 1 < self.times.ns.size else now
            wet_now = np.isfinite(now).all(axis=1)
            wet_both = wet_now & np.isfinite(nxt).all(axis=1)
            values = np.nan_to_num(np.concatenate([now, nxt], axis=1), nan=0.0)
            self._table = (k, values, wet_now, wet_both)
        return self._table[1:]

    def bracket(self, t_ns: int) -> tuple[int, float]:
        """`TimeAxis.bracket`, remembered between calls, refusing an over-long gap."""
        k, t_k, t_next = self._bracket
        if not t_k <= t_ns < t_next:
            k, _ = self.times.bracket(t_ns)
            t_k = int(self.times.ns[k])
            t_next = int(self.times.ns[k + 1]) if k + 1 < self.times.ns.size else t_k + 1
            self._bracket = (k, t_k, t_next)
        frac = (t_ns - t_k) / (t_next - t_k) if t_ns != t_k else 0.0
        if frac > 0.0 and t_next - t_k > self.max_gap_s * _NS:
            raise ForcingGapError(
                f"{self.name} has no data between {np.datetime64(t_k, 'ns')} and "
                f"{np.datetime64(t_next, 'ns')} ({(t_next - t_k) / 3600 / _NS:g} h), longer "
                f"than the {self.max_gap_s / 3600:g} h D011 allows to blend across")
        return k, frac

    def sample(self, lats, lons, t_ns: int, inside) -> np.ndarray:
        """(u, v) at each particle, NaN where it is beached or not `inside`. A new array."""
        k, frac = self.bracket(t_ns)
        values, wet_now, wet_both = self._table_for(k)
        wet = wet_now if frac == 0.0 else wet_both

        i0, fy = self.lat.locate(np.where(inside, lats, self.lat.origin))
        j0, fx = self.lon.locate(np.where(inside, lons, self.lon.origin))
        width = self.lon.size
        base = i0 * width + j0
        corners = np.stack([base, base + 1, base + width, base + width + 1])

        usable, beached = usable_corners(wet[corners], fy, fx)
        weights = cell_weights(fy, fx) * usable
        # Weight first, blend in time once: both are linear, so the order does not change
        # the answer, and one blend is a quarter of the work of four. Land is 0 in the
        # table, and its weight is 0 here, so no NaN ever enters a product.
        both = np.zeros((lats.size, 4))
        for w, c in zip(weights, corners):
            both += w[:, None] * np.take(values, c, axis=0)
        out = both[:, :2]
        if frac:
            out = out + frac * (both[:, 2:] - out)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = out / weights.sum(axis=0)[:, None]
        out[beached | ~inside] = np.nan
        return out

    def describe(self) -> dict:
        return {"files": [self.path(ds) for ds in self.datasets],
                "coverage": [str(np.datetime64(int(self.times.ns[0]), "ns")),
                             str(np.datetime64(int(self.times.ns[-1]), "ns"))],
                "grid": {"lat": [self.lat.origin, self.lat.step, self.lat.size],
                         "lon": [self.lon.origin, self.lon.step, self.lon.size]},
                "max_gap_h": self.max_gap_s / 3600}


class GriddedForcing:
    """The real current and wind, behind the same `sample` as `ConstantForcing`."""

    def __init__(self, current_paths, wind_paths):
        opened = {"current": [], "wind": []}
        try:
            for product, paths in (("current", current_paths), ("wind", wind_paths)):
                for path in paths:
                    opened[product].append(open_forcing(path))
            self.current = Field("current", opened["current"], CURRENT_VARS,
                                 MAX_GAP_S["current"])
            self.wind = Field("wind", opened["wind"], WIND_VARS, MAX_GAP_S["wind"])
        except Exception:
            for ds in opened["current"] + opened["wind"]:
                ds.close()
            raise
        self.window = None
        self.gaps_blended: list = []

    @classmethod
    def from_dir(cls, root, start, end) -> GriddedForcing:
        """The files under <root>/raw that a run from start to end needs, checked.

        Chosen by the dates in their names, with a day's slack either side because one
        older wind file's name gives an inclusive end; after that only the time axes
        inside the files are trusted. Refuses the window if it is not covered or crosses
        a gap D011 does not allow.
        """
        raw = Path(root) / "raw"
        lo, hi = as_ns(start) - 86_400 * _NS, as_ns(end) + 86_400 * _NS
        picked = {}
        for product, (prefix, _) in PRODUCTS.items():
            picked[product] = []
            for path in sorted(raw.glob(f"{prefix}_{box_tag()}_*.nc")):
                m = _FILE_DATES.search(path.name)
                if m and as_ns(_day(m[1])) <= hi and as_ns(_day(m[2])) >= lo:
                    picked[product].append(path)
            if not picked[product]:
                raise FileNotFoundError(f"no {prefix}_{box_tag()}_*.nc in {raw} covers "
                                        f"{start} to {end}")
        forcing = cls(picked["current"], picked["wind"])
        try:
            forcing.require(start, end)
        except Exception:
            forcing.close()
            raise
        return forcing

    def require(self, start, end) -> list:
        """Refuse a run from start to end the files cannot support; list the gaps it blends.

        Raises OutOfCoverageError if either end is outside the files and ForcingGapError
        if the window crosses a gap longer than D011 allows (the whole scenario is
        excluded, as D011 says). Returns the shorter gaps it will blend across, which
        `describe` then records.
        """
        s, e = as_ns(start), as_ns(end)
        if e < s:
            raise ValueError(f"end {end} is before start {start}")
        found = []
        for field in (self.current, self.wind):
            field.times.bracket(s)
            field.times.bracket(e)
            usual = float(np.median(np.diff(field.times.ns))) / _NS
            for gap in field.times.gaps(s, e, usual):
                if gap[2] * 3600 > field.max_gap_s:
                    raise ForcingGapError(f"{field.name} has a {gap[2]:g} h gap from {gap[0]} "
                                          f"to {gap[1]}, inside {start} to {end}; D011 "
                                          "excludes a scenario that crosses it")
                found.append({"field": field.name, "from": gap[0], "to": gap[1],
                              "hours": gap[2]})
        self.window = [str(np.datetime64(s, "ns")), str(np.datetime64(e, "ns"))]
        self.gaps_blended = found
        return found

    def outside(self, lats, lons) -> np.ndarray:
        """True for particles outside the box both grids cover, NaN positions included."""
        lats, lons = as_particle_axes(lats, lons)
        return ~(self.current.inside(lats, lons) & self.wind.inside(lats, lons))

    def sample(self, lats, lons, time) -> tuple[np.ndarray, np.ndarray]:
        """The current and the wind at every particle, each (N, 2) in m/s.

        The current is NaN at a beached particle, and both are NaN outside the box.
        """
        lats, lons = as_particle_axes(lats, lons)
        inside = self.current.inside(lats, lons) & self.wind.inside(lats, lons)
        t = as_ns(time)
        return (self.current.sample(lats, lons, t, inside),
                self.wind.sample(lats, lons, t, inside))

    def describe(self) -> dict:
        """What this backend is, for the header of an output file. JSON-safe."""
        return {"backend": "gridded",
                "land": "beached where the nearest HYCOM point is land; otherwise wet "
                        "corners renormalised, the diagonal only through a wet edge",
                "window": self.window,
                "gaps_blended": self.gaps_blended,
                "current": self.current.describe(),
                "wind": self.wind.describe()}

    def close(self) -> None:
        for ds in self.current.datasets + self.wind.datasets:
            ds.close()

    def __enter__(self) -> GriddedForcing:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _day(yyyymmdd: str) -> str:
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}"


def check_against_sample_field(forcing: GriddedForcing, time, k: int, rng) -> dict:
    """The largest difference from `sample_field` over k random all-sea points.

    `sample_field` measures offsets from the STORED grid lines, which HYCOM keeps in
    float32, up to 3.4e-5 degrees off the regular grid used here. So the two differ by up
    to (that departure / the step) x the corner spread; the bound is reported beside the
    difference.
    """
    t = as_ns(time)
    out = {}
    for field in (forcing.current, forcing.wind):
        k_t, _ = field.times.bracket(t)
        nxt = min(k_t + 1, field.times.ns.size - 1)
        if field.times.file[k_t] != field.times.file[nxt]:
            raise ValueError(f"{time} falls between two {field.name} files; check another time")
        ds = field.datasets[int(field.times.file[k_t])]
        lat_dep = np.abs(ds["lat"].values - (field.lat.origin
                                              + field.lat.step * np.arange(field.lat.size))).max()
        lon_dep = np.abs(ds["lon"].values - (field.lon.origin
                                              + field.lon.step * np.arange(field.lon.size))).max()
        shift = lat_dep / field.lat.step + lon_dep / field.lon.step
        worst, worst_bound, n = 0.0, 0.0, 0
        while n < k:
            lat = rng.uniform(field.lat.origin, field.lat.last)
            lon = rng.uniform(field.lon.origin, field.lon.last)
            try:
                ref = sample_field(ds, field.var_names, lat, lon, np.datetime64(t, "ns"))
            except MissingCornerError:
                continue
            ours = field.sample(np.array([lat]), np.array([lon]), t, np.array([True]))[0]
            theirs = ref.weights @ ref.corners
            spread = ref.corners.max(axis=0) - ref.corners.min(axis=0)
            worst = max(worst, float(np.abs(ours - theirs).max()))
            worst_bound = max(worst_bound, float((shift * spread).max()) + 1e-9)
            n += 1
        out[field.name] = {"points": n, "max_difference_ms": worst,
                           "bound_ms": worst_bound, "within_bound": worst <= worst_bound}
    return out


def _cli(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--forcing-dir", required=True,
                        help="data root holding raw/hycom_* and raw/era5_*, e.g. C:/maritime-data")
    parser.add_argument("--time", required=True, help="the instant to sample, ISO 8601")
    parser.add_argument("--particles", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--check", type=int, default=0,
                        help="also compare this many all-sea points with sample_field")
    args = parser.parse_args(argv)
    if args.particles < 1 or args.check < 0:
        parser.error("--particles must be at least 1 and --check at least 0")

    rng = np.random.default_rng(args.seed)
    with GriddedForcing.from_dir(args.forcing_dir, args.time, args.time) as forcing:
        box = forcing.current
        lats = rng.uniform(box.lat.origin, box.lat.last, args.particles)
        lons = rng.uniform(box.lon.origin, box.lon.last, args.particles)
        current, wind = forcing.sample(lats, lons, args.time)
        times = []
        for _ in range(5):
            t0 = clock.perf_counter()
            forcing.sample(lats, lons, args.time)
            times.append(clock.perf_counter() - t0)
        outside = forcing.outside(lats, lons)
        beached = ~np.isfinite(current).all(axis=1) & ~outside
        sea = ~beached & ~outside
        result = {
            "time": args.time, "particles": args.particles,
            "sea": int(sea.sum()), "beached": int(beached.sum()), "outside": int(outside.sum()),
            "mean_current_speed_ms": float(np.hypot(*current[sea].T).mean()) if sea.any() else None,
            "mean_wind_speed_ms": float(np.hypot(*wind[~outside].T).mean()),
            "ms_per_call": 1000.0 * float(np.median(times)),
            "box": {"lat": [box.lat.origin, box.lat.last],
                    "lon": [float(to_display_longitude(box.lon.origin)),
                            float(to_display_longitude(box.lon.last))]},
            "forcing": forcing.describe(),
        }
        if args.check:
            result["check_against_sample_field"] = check_against_sample_field(
                forcing, args.time, args.check, rng)
    return result


if __name__ == "__main__":
    print(json.dumps(_cli(), indent=2))
