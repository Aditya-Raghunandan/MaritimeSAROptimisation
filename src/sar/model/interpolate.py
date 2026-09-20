"""interpolate.py: bilinear current and wind at a given position and time (issue #10).

Given a latitude, longitude and time, this module returns the resultant current vector and
the resultant wind vector as JSON, each with an uncertainty estimate. The drift engine
needs forcing at positions that rarely coincide with a grid point, so the four grid points
around the position are interpolated: bilinearly in space, linearly in time.

Vectors are interpolated as components (u eastward, v northward), never as speed and
direction, and the bearing is derived from the summed components afterwards. See
docs/resultant-vector.md for the physics, the weights and the error formulae.

Finding the four points uses the nearest grid point and one direction sign per axis, so it
is index arithmetic with no distance search.

CLI. Both paths are real files the fetchers wrote under data/raw/, not placeholders, so
pull them first with `python -m sar.fetch.current --help` and `python -m sar.fetch.wind
--help`. The position must fall inside the box those files cover, and the time inside
their window:

    python -m sar.model.interpolate \
        --current data/raw/hycom_17-36N_82-63W_2021-01-01-2021-01-09.nc \
        --wind data/raw/era5_17-36N_82-63W_2021-01-01-2021-01-09.nc \
        --lat 26.53 --lon 281.4 --time 2021-01-05T07:30 [--with-wind | --no-wind] [--plot out.png]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from sar.utils.geo import regular_axis_step, to_store_longitude

INCLUDE_WIND = True

CURRENT_VARS = ("water_u", "water_v")
WIND_VARS = ("u10", "v10")

# Below this coherence the corner directions nearly cancel, so a direction spread is
# meaningless and is reported as None.
COHERENCE_FLOOR = 0.1

# How far outside the first and last grid line a position may sit and still count as
# inside. Without it, asking for exactly the southern edge of the box can be rejected,
# because 17.0 read back from a float32 axis is 16.999999... and a bare `<` comparison
# then calls it outside. 1e-9 degrees is about 0.1 mm, so it admits only that rounding
# error and nothing a caller could have meant.
EDGE_EPS = 1e-9


class OutOfCoverageError(ValueError):
    """The requested position or time lies outside the loaded grid."""


class MissingCornerError(ValueError):
    """One or more of the four surrounding grid points has no data (land)."""


@dataclass(frozen=True)
class Sample:
    """Everything found for one field at one position and time.

    corners holds the four (u, v) pairs already blended to the requested time, in the
    order nearest, x-neighbour, y-neighbour, diagonal. weights follows the same order.
    """

    lat: float              # the position asked for, degrees north
    lon: float              # the position asked for, degrees east in the stored 0 to 360
    i: int                  # row index of the nearest grid point, along latitude
    j: int                  # column index of the nearest grid point, along longitude
    sy: int                 # +1 if the position is north of that point, -1 if south
    sx: int                 # +1 if the position is east of that point, -1 if west
    dy: float               # latitude offset to that point, signed degrees, within dlat/2
    dx: float               # longitude offset to that point, signed degrees, within dlon/2
    dlat: float             # the grid's latitude step in degrees (0.04 for HYCOM)
    dlon: float             # the grid's longitude step in degrees (0.08 for HYCOM)
    fy: float               # abs(dy) as a fraction of one latitude step, so 0 to 0.5
    fx: float               # abs(dx) as a fraction of one longitude step, so 0 to 0.5
    corners: np.ndarray     # the four (u, v) pairs in m/s, shape (4, 2), blended in time
    weights: np.ndarray     # the four bilinear weights, same corner order, summing to 1

    def corner_offsets(self) -> np.ndarray:
        """Corner positions relative to the nearest grid point, as (lon, lat) degrees."""
        return np.array(
            [
                [0.0, 0.0],
                [self.sx * self.dlon, 0.0],
                [0.0, self.sy * self.dlat],
                [self.sx * self.dlon, self.sy * self.dlat],
            ]
        )


def grid_step(axis, name: str = "axis") -> float:
    """The spacing of one grid axis, checked to be regular.

    Every weight below assumes a regular grid, so an irregular axis must fail loudly
    rather than produce a quietly wrong answer. The judgement itself lives in
    `sar.utils.geo.regular_axis_step`, because the browser's manifest and the table
    pivot have to make exactly the same one.
    """
    return regular_axis_step(axis, name)


def nearest_index(axis, value: float, name: str = "axis") -> tuple[int, float]:
    """The nearest grid line on one ascending axis, and the signed offset to it.

    Returns (i, d) with d = value - axis[i], so d lies within half a step either side.
    Raises OutOfCoverageError if value is outside the axis range.
    """
    axis = np.asarray(axis, dtype=float)
    step = grid_step(axis)
    if value < axis[0] - EDGE_EPS or value > axis[-1] + EDGE_EPS:
        raise OutOfCoverageError(
            f"{name} {value} is outside the grid, which covers {axis[0]} to {axis[-1]}"
        )
    i = int(np.clip(round((value - axis[0]) / step), 0, axis.size - 1))
    return i, float(value - axis[i])


def neighbour_direction(dy: float, dx: float) -> tuple[int, int]:
    """Which side of the nearest grid point the position is on, one sign per axis.

    Returns (sy, sx), each +1 if the offset is zero or positive (toward higher index) and
    -1 otherwise. Zero counts as +1; the far corner then has weight zero.
    """
    return (1 if dy >= 0 else -1), (1 if dx >= 0 else -1)


def keep_inside(i: int, sign: int, n: int) -> int:
    """Flip a direction sign inward if it would point off the end of an axis of n points.

    This only happens for a position on the first or last grid line, where the offset is
    zero, so the far-side weight is zero and flipping the sign changes nothing.
    """
    if not 0 <= i + sign < n:
        return -sign
    return sign


def cell_corners(i: int, j: int, sy: int, sx: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    """The four (row, column) indices around the position.

    Order: the nearest point, its neighbour in x, its neighbour in y, then the diagonal.
    Raises OutOfCoverageError if a neighbour falls off a grid of the given (rows, columns).
    """
    n_rows, n_cols = shape
    offsets = [(0, 0), (0, sx), (sy, 0), (sy, sx)]
    corners = [(i + di, j + dj) for di, dj in offsets]
    for row, col in corners:
        if not (0 <= row < n_rows and 0 <= col < n_cols):
            raise OutOfCoverageError(
                f"the cell around grid point ({i}, {j}) leaves a grid of shape {shape}"
            )
    return corners


def bilinear_weights(fy: float, fx: float) -> np.ndarray:
    """The four corner weights from the fractional offsets fy and fx.

    fy and fx are the distances from the nearest point as fractions of one step. Order
    matches cell_corners: nearest, x-neighbour, y-neighbour, diagonal. Sums to 1.
    """
    if not (0.0 <= fy <= 1.0 and 0.0 <= fx <= 1.0):
        raise ValueError(f"fractional offsets must be within 0 to 1, got fy={fy}, fx={fx}")
    return np.array(
        [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy]
    )


def bracket_time(times, t) -> tuple[int, float]:
    """The time step at or before t, and how far t is toward the next one.

    Returns (k, frac) with 0 <= frac < 1, except at the final time step where frac is 0.
    Raises OutOfCoverageError if t is outside the time axis.
    """
    times = np.asarray(times, dtype="datetime64[ns]")
    t = np.datetime64(t, "ns")
    if t < times[0] or t > times[-1]:
        raise OutOfCoverageError(f"time {t} is outside the file, which covers {times[0]} to {times[-1]}")
    if times.size == 1 or t == times[-1]:
        return int(times.size - 1) if t == times[-1] else 0, 0.0
    k = int(np.searchsorted(times, t, side="right") - 1)
    frac = float((t - times[k]) / (times[k + 1] - times[k]))
    return k, frac


def time_blend(corners_t1, corners_t2, frac: float) -> np.ndarray:
    """Linear blend of the corner values between two time steps.

    The single-instant form of interpolate_series in sar.utils.interpolation: piece m of
    n from that function equals this with frac = m / n.
    """
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"frac must be within 0 to 1, got {frac}")
    c1 = np.asarray(corners_t1, dtype=float)
    c2 = np.asarray(corners_t2, dtype=float)
    return c1 + frac * (c2 - c1)


def resultant_vector(corners, weights) -> tuple[float, float]:
    """The weighted sum of the corner (u, v) pairs, component by component."""
    u, v = np.asarray(weights, dtype=float) @ np.asarray(corners, dtype=float)
    return float(u), float(v)


def speed_direction(u: float, v: float) -> tuple[float, float | None]:
    """Speed in m/s and the compass bearing the vector points toward, in degrees.

    The bearing is clockwise from north (0 north, 90 east) and is None for a zero vector,
    where it is undefined. For wind, add 180 degrees to get the meteorological direction.
    """
    speed = float(np.hypot(u, v))
    if speed == 0.0:
        return 0.0, None
    return speed, float(np.degrees(np.arctan2(u, v)) % 360.0)


def bilinear_error_bound(dlon: float, dlat: float, d2x: float, d2y: float) -> float:
    """Classical bound on the bilinear error for a field with known curvature.

    d2x and d2y are the largest absolute second derivatives along longitude and latitude.
    Four corners cannot estimate curvature, so this is for testing on analytic fields, not
    for use at run time.
    """
    return dlon**2 / 8.0 * d2x + dlat**2 / 8.0 * d2y


def interpolation_uncertainty(corners, weights, product_sigma: float = 0.0) -> dict:
    """Indicators of how far the interpolated vector can be trusted.

    Computed from the four corners alone, so they measure variability within the cell, not
    a guaranteed bound. product_sigma is the caller's estimate of the source model's own
    error in m/s; it defaults to 0 because it has not been measured.

    Returns sigma_spatial_ms (weighted corner scatter), coherence (0 to 1),
    speed_loss_ms, direction_spread_deg (None when coherence is below COHERENCE_FLOOR),
    sigma_total_ms and n_corners.
    """
    if product_sigma < 0:
        raise ValueError(f"product_sigma must not be negative, got {product_sigma}")
    corners = np.asarray(corners, dtype=float)
    weights = np.asarray(weights, dtype=float)

    vp = weights @ corners
    sigma_spatial = float(np.sqrt(weights @ np.sum((corners - vp) ** 2, axis=1)))

    mean_speed = float(weights @ np.hypot(corners[:, 0], corners[:, 1]))
    speed_p = float(np.hypot(*vp))
    coherence = 1.0 if mean_speed == 0.0 else min(speed_p / mean_speed, 1.0)

    spread = None
    if coherence >= COHERENCE_FLOOR:
        spread = float(np.degrees(np.sqrt(-2.0 * np.log(coherence))))

    return {
        "sigma_spatial_ms": sigma_spatial,
        "coherence": coherence,
        "speed_loss_ms": mean_speed - speed_p,
        "direction_spread_deg": spread,
        "sigma_total_ms": float(np.hypot(sigma_spatial, product_sigma)),
        "n_corners": int(corners.shape[0]),
    }


def _read_corners(ds: xr.Dataset, var_names, k: int, rows_cols, when: str) -> np.ndarray:
    """The four (u, v) pairs at time index k, reading only the 2 by 2 window needed.

    Raises MissingCornerError if any is NaN.
    """
    rows = [r for r, _ in rows_cols]
    cols = [c for _, c in rows_cols]
    r0, c0 = min(rows), min(cols)
    window = []
    for name in var_names:
        block = ds[name].isel(
            time=k, lat=slice(r0, max(rows) + 1), lon=slice(c0, max(cols) + 1)
        ).values
        window.append([block[r - r0, c - c0] for r, c in rows_cols])
    corners = np.array(window, dtype=float).T          # shape (4, 2)

    missing = np.isnan(corners).any(axis=1)
    if missing.any():
        n = int(missing.sum())
        where = [rows_cols[m] for m in np.flatnonzero(missing)]
        how_many = "all 4" if n == 4 else f"{n} of 4"
        raise MissingCornerError(
            f"{how_many} surrounding grid points have no data at {when}, at (row, column) "
            f"{where}; the position is within one cell of land. Repairing this is not "
            "implemented; see docs/resultant-vector.md."
        )
    return corners


def sample_field(ds: xr.Dataset, var_names, lat: float, lon: float, time) -> Sample:
    """Find the four grid points around (lat, lon) and blend them to the given time.

    ds must be in the D020 convention (sar.utils.data_io.open_forcing does this). var_names
    is the (u, v) variable pair, for example CURRENT_VARS or WIND_VARS.
    """
    lat_axis = np.asarray(ds["lat"].values, dtype=float)
    lon_axis = np.asarray(ds["lon"].values, dtype=float)
    lon = float(to_store_longitude(lon))

    dlat, dlon = grid_step(lat_axis), grid_step(lon_axis)
    i, dy = nearest_index(lat_axis, lat, "lat")
    j, dx = nearest_index(lon_axis, lon, "lon")
    sy, sx = neighbour_direction(dy, dx)
    sy, sx = keep_inside(i, sy, lat_axis.size), keep_inside(j, sx, lon_axis.size)
    rows_cols = cell_corners(i, j, sy, sx, (lat_axis.size, lon_axis.size))

    fy, fx = abs(dy) / dlat, abs(dx) / dlon
    weights = bilinear_weights(fy, fx)

    k, frac = bracket_time(ds["time"].values, time)
    c1 = _read_corners(ds, var_names, k, rows_cols, f"time step {k}")
    if frac == 0.0:
        corners = c1
    else:
        c2 = _read_corners(ds, var_names, k + 1, rows_cols, f"time step {k + 1}")
        corners = time_blend(c1, c2, frac)

    return Sample(lat, lon, i, j, sy, sx, dy, dx, dlat, dlon, fy, fx, corners, weights)


def _describe(sample: Sample, product_sigma: float) -> dict:
    u, v = resultant_vector(sample.corners, sample.weights)
    speed, bearing = speed_direction(u, v)
    return {
        "u": u,
        "v": v,
        "speed": speed,
        "direction_to_deg": bearing,
        "uncertainty": interpolation_uncertainty(sample.corners, sample.weights, product_sigma),
    }


def resultant_at(
    lat: float,
    lon: float,
    time,
    current_ds: xr.Dataset,
    wind_ds: xr.Dataset | None = None,
    include_wind: bool = INCLUDE_WIND,
    current_sigma: float = 0.0,
    wind_sigma: float = 0.0,
) -> dict:
    """The resultant current and, if include_wind, the resultant wind at (lat, lon, time).

    The wind key is absent, not null, when wind is excluded. Raises ValueError if wind is
    requested but no wind dataset is given.
    """
    if include_wind and wind_ds is None:
        raise ValueError("include_wind is True but no wind dataset was given")

    out = {
        "lat": float(lat),
        "lon": float(to_store_longitude(lon)),
        "time": str(np.datetime64(time, "s")),
        "current": _describe(sample_field(current_ds, CURRENT_VARS, lat, lon, time), current_sigma),
    }
    if include_wind:
        out["wind"] = _describe(sample_field(wind_ds, WIND_VARS, lat, lon, time), wind_sigma)
    return out


# ---------------------------------------------------------------------------
# The diagram. Everything below draws what the functions above compute, and is
# the only part that needs matplotlib.
# ---------------------------------------------------------------------------


def _pyplot():
    """matplotlib.pyplot, configured headless, or an error that says how to get it.

    matplotlib is a declared dependency in pyproject.toml, but a virtual environment made
    before it was added will not have it, and the bare ModuleNotFoundError that follows
    names neither the package nor the cure. Only the drawing needs it: the JSON path runs
    without matplotlib installed at all.
    """
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "drawing needs matplotlib, which this environment does not have, although "
            "pyproject.toml declares it. Install the project's dependencies with "
            "`pip install -e .` from the repository root, or just `pip install "
            "matplotlib`. Everything except --plot and --diagram works without it."
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def synthetic_sample() -> Sample:
    """A made-up HYCOM-shaped cell whose current turns across it, for the diagram."""
    dlat, dlon = 0.04, 0.08
    dy, dx = 0.012, -0.024
    sy, sx = neighbour_direction(dy, dx)
    fy, fx = abs(dy) / dlat, abs(dx) / dlon
    corners = np.array([[0.60, 0.10], [0.40, 0.55], [0.70, -0.20], [0.10, 0.65]])
    return Sample(26.5, 281.0, 0, 0, sy, sx, dy, dx, dlat, dlon, fy, fx, corners,
                  bilinear_weights(fy, fx))


def wrong_average(corners, weights) -> tuple[float, float]:
    """The resultant got by averaging speed and bearing separately (the wrong way).

    Returned as (u, v) so it can be drawn beside the correct one.
    """
    corners = np.asarray(corners, dtype=float)
    weights = np.asarray(weights, dtype=float)
    speeds = np.hypot(corners[:, 0], corners[:, 1])
    bearings = np.degrees(np.arctan2(corners[:, 0], corners[:, 1]))
    s, b = weights @ speeds, np.radians(weights @ bearings)
    return float(s * np.sin(b)), float(s * np.cos(b))


def plot_bearings(sample: Sample, ax) -> None:
    """Draw the bearings at the datum as arcs swept clockwise from north.

    The other two panels show the vectors; this one shows the ANGLES, which are what a
    reader is asked to trust when the JSON quotes `direction_to_deg`. Lengths are dropped,
    and each of the four corners gets its own arrow, its own arc swept from north, and its
    own degrees written on that arc, with the resultant's outside them all.

    Each corner is drawn at its OWN radius, widely spaced, rather than all at radius 1.
    On real data the four corners of a cell usually point within a few degrees of one
    another, measured 4.2 degrees apart at 27.173 N, 70.369 W, so arrows and arcs at a
    common radius collapse into a single stroke and the panel appears to show nothing at
    all. Separating them radially is what makes four distinct angles visible exactly when
    the answer is most confident, and each arc's label sits at that arc's own midpoint,
    which separates the labels for the same reason.

    Bearings are clockwise from north, while matplotlib measures anticlockwise from east,
    so every angle here is converted as 90 - bearing. An arc from north to a bearing is
    therefore drawn from (90 - bearing) anticlockwise to 90, which is the clockwise sweep
    a compass reader expects. Getting that backwards produces a picture that looks
    plausible and mirrors every angle, so it is worth stating.
    """
    from matplotlib.patches import Arc

    def bearing_of(u, v):
        return float(np.degrees(np.arctan2(u, v)) % 360.0)

    def point_at(bearing, radius):
        a = np.radians(90.0 - bearing)
        return radius * np.cos(a), radius * np.sin(a)

    u_p, v_p = resultant_vector(sample.corners, sample.weights)
    resultant_bearing = bearing_of(u_p, v_p)
    spread = interpolation_uncertainty(sample.corners, sample.weights)["direction_spread_deg"]

    # Radii: one per corner, then the resultant outside them.
    corner_radii = [0.45, 0.70, 0.95, 1.20]
    resultant_radius = 1.50

    # North reference, and the compass letters, so the frame is unambiguous.
    ax.plot([0, 0], [0, resultant_radius + 0.12], color="0.6", lw=1, ls="--", zorder=1)
    for letter, bearing in (("N", 0), ("E", 90), ("S", 180), ("W", 270)):
        x, y = point_at(bearing, resultant_radius + 0.32)
        ax.text(x, y, letter, ha="center", va="center", fontsize=10, color="0.45")

    # The spread the uncertainty reports, as a wedge about the resultant. It is the
    # circular standard deviation of the corners, not a bound, so it is drawn faintly.
    if spread is not None and spread > 0:
        edges = np.radians(90.0 - np.linspace(resultant_bearing - spread,
                                              resultant_bearing + spread, 40))
        r = resultant_radius
        ax.fill(np.concatenate([[0], r * np.cos(edges)]),
                np.concatenate([[0], r * np.sin(edges)]),
                color="tab:red", alpha=0.10, zorder=1)

    # Four shades rather than one, so an arrow can be matched to its arc and its label
    # even where the four angles nearly coincide.
    shades = ["#9ecae1", "#6baed6", "#3182bd", "#08519c"]
    labels = ["nearest", "x neighbour", "y neighbour", "diagonal"]
    rows = []

    for k, ((u, v), name) in enumerate(zip(sample.corners, labels)):
        b = bearing_of(u, v)
        radius = corner_radii[k]
        colour = shades[k]

        # Arrow out to this corner's own radius, so it ends on its own arc.
        x, y = point_at(b, radius)
        ax.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 1.4},
                    zorder=3)
        # The arc: the swept angle itself, from north round to that arrow.
        ax.add_patch(Arc((0, 0), 2 * radius, 2 * radius, theta1=90.0 - b, theta2=90.0,
                         color=colour, lw=1.6, zorder=3))
        # The degrees, written on the arc itself. Each label sits at a different fraction
        # along its own arc rather than all at the midpoint, which separates them by angle
        # as well as by radius and keeps the outermost corner clear of the resultant's
        # label, which sits at its own midpoint.
        mx, my = point_at(b * (0.35 + 0.10 * k), radius)
        ax.text(mx, my, f"{b:.1f}", ha="center", va="center", fontsize=7.5, color=colour,
                zorder=5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0})
        rows.append((name, b, colour))

    # The resultant: outside the four, thicker, and the only red thing here.
    x, y = point_at(resultant_bearing, resultant_radius)
    ax.annotate("", xy=(x, y), xytext=(0, 0),
                arrowprops={"arrowstyle": "-|>", "color": "tab:red", "lw": 2.4}, zorder=4)
    ax.add_patch(Arc((0, 0), 2 * resultant_radius, 2 * resultant_radius,
                     theta1=90.0 - resultant_bearing, theta2=90.0,
                     color="tab:red", lw=2.2, zorder=4))
    mid_x, mid_y = point_at(resultant_bearing / 2.0, resultant_radius)
    ax.text(mid_x, mid_y, f"{resultant_bearing:.1f} deg", ha="center", va="center",
            fontsize=9.5, color="tab:red", zorder=6,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5})

    # A key, each line in its own arc's colour, so the names from the first panel can be
    # matched to the angles here without crowding the circle. Top left: the arcs sweep
    # clockwise from north, so that corner is the one they cannot reach until a bearing
    # passes 270 degrees, and it is empty for the great majority of cells.
    ax.text(0.0, 1.0, "theta at each corner (deg)", transform=ax.transAxes,
            fontsize=7, color="0.35", family="monospace", ha="left", va="top")
    for k, (name, b, colour) in enumerate(rows):
        ax.text(0.0, 0.955 - 0.042 * k, f"{name:<12s}{b:6.1f}", transform=ax.transAxes,
                fontsize=7, color=colour, family="monospace", ha="left", va="top")

    ax.plot(0, 0, "*", color="tab:red", ms=13, zorder=4)
    caption = f"resultant {resultant_bearing:.1f} deg toward"
    if spread is not None:
        caption += f", corner spread {spread:.1f} deg"
    ax.set_title("Bearings at the datum, clockwise from north", fontsize=10)
    ax.set_xlabel(caption, fontsize=8)
    ax.set_xlim(-1.95, 1.95)
    ax.set_ylim(-1.95, 1.95)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def plot_resultant_cell(sample: Sample, path: str) -> None:
    """Write the three-panel diagram for one sample to path.

    The left panel is the grid cell drawn to scale, so a HYCOM cell (0.08 by 0.04 deg) is
    visibly not square: the particle, the four corner vectors labelled with their weights,
    and the resultant. The middle panel is in velocity space and sets the correct resultant
    beside the one from averaging speed and bearing, so the lost speed is visible. The
    right panel drops the lengths and shows the bearings alone, swept from north, which is
    where the degrees quoted in the JSON can be read off.

    matplotlib is imported here rather than at the top of the module, as in sar.viz.fields,
    so that importing this module for its maths costs nothing. That also keeps the JSON
    path working in an environment that has no matplotlib: only the drawing needs it.
    """
    plt = _pyplot()

    fig, (ax, av, ab) = plt.subplots(1, 3, figsize=(16, 5))

    # Left panel: the cell in degrees, with the nearest grid point at the origin.
    pos = sample.corner_offsets()
    labels = ["nearest", "x neighbour", "y neighbour", "diagonal"]
    top = float(np.max(np.hypot(sample.corners[:, 0], sample.corners[:, 1])))
    scale = 0.45 * sample.dlat / top                     # degrees of arrow per m/s

    xs = [0.0, pos[1, 0], pos[3, 0], pos[2, 0], 0.0]
    ys = [0.0, pos[1, 1], pos[3, 1], pos[2, 1], 0.0]
    ax.plot(xs, ys, color="0.5", lw=1)
    for (x, y), (u, v), w, name in zip(pos, sample.corners, sample.weights, labels):
        ax.plot(x, y, "o", color="0.3")
        ax.quiver(x, y, u * scale, v * scale, angles="xy", scale_units="xy", scale=1,
                  color="tab:blue", width=0.006)
        ax.annotate(f"{name}\nw = {w:.2f}", (x, y), textcoords="offset points",
                    xytext=(6, -22 if y == 0 else 8), fontsize=8)
    u_p, v_p = resultant_vector(sample.corners, sample.weights)
    ax.plot(sample.dx, sample.dy, "*", color="tab:red", ms=14, label="particle")
    ax.quiver(sample.dx, sample.dy, u_p * scale, v_p * scale, angles="xy",
              scale_units="xy", scale=1, color="tab:red", width=0.008, label="resultant")
    ax.set_aspect("equal")
    ax.set_xlabel("longitude offset from nearest grid point (deg)")
    ax.set_ylabel("latitude offset (deg)")
    # Rounded: a step read back off a float32 axis prints as 0.07999999999998408, which
    # is true and unreadable in a title.
    ax.set_title(f"Cell to scale ({sample.dlon:.4g} by {sample.dlat:.4g} deg)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8)
    reach = 0.6 * sample.dlat                            # room for arrows and labels
    ax.set_xlim(min(pos[:, 0]) - reach, max(pos[:, 0]) + reach)
    ax.set_ylim(min(pos[:, 1]) - reach, max(pos[:, 1]) + reach)

    # Right panel: velocity space.
    for u, v in sample.corners:
        av.quiver(0, 0, u, v, angles="xy", scale_units="xy", scale=1,
                  color="tab:blue", alpha=0.5, width=0.005)
    wu, wv = wrong_average(sample.corners, sample.weights)
    av.quiver(0, 0, wu, wv, angles="xy", scale_units="xy", scale=1, color="tab:orange",
              width=0.008, label=f"average speed and bearing: {np.hypot(wu, wv):.2f} m/s")
    speed_p, _ = speed_direction(u_p, v_p)
    av.quiver(0, 0, u_p, v_p, angles="xy", scale_units="xy", scale=1, color="tab:red",
              width=0.008, label=f"average components: {speed_p:.2f} m/s")
    # Limits from what is actually drawn, in both signs. Sizing them from the largest
    # speed alone assumes the flow goes north east, which the Gulf Stream mostly does and
    # a real cell need not: a westward current drew entirely outside the axes.
    drawn = np.vstack([sample.corners, [[u_p, v_p], [wu, wv], [0.0, 0.0]]])
    pad = 0.15 * max(np.ptp(drawn[:, 0]), np.ptp(drawn[:, 1]), 1e-6)
    av.set_xlim(drawn[:, 0].min() - pad, drawn[:, 0].max() + pad)
    av.set_ylim(drawn[:, 1].min() - pad, drawn[:, 1].max() + pad)
    av.set_aspect("equal")
    av.axhline(0, color="0.8", lw=0.5)
    av.axvline(0, color="0.8", lw=0.5)
    av.set_xlabel("u, eastward (m/s)")
    av.set_ylabel("v, northward (m/s)")
    av.set_title("The right and the wrong average")
    av.legend(loc="lower right", fontsize=8)

    # Right panel: the angles on their own.
    plot_bearings(sample, ab)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Resultant current and wind at a position and time, as JSON"
    )
    p.add_argument("--current", help="path to a stored HYCOM current file")
    p.add_argument("--wind", help="path to a stored ERA5 wind file")
    p.add_argument("--lat", type=float)
    p.add_argument("--lon", type=float, help="-180 to 180 or 0 to 360")
    p.add_argument("--time", help="ISO 8601, for example 2021-01-05T07:30")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--with-wind", action="store_true", help="include wind")
    group.add_argument("--no-wind", action="store_true", help="leave wind out")
    p.add_argument("--plot", help="write a diagram of the queried grid cell to this path")
    p.add_argument(
        "--diagram",
        nargs="?",
        const="figures/report/resultant_vector_diagram.png",
        help="draw the explanatory diagram from synthetic values and exit; needs no data",
    )
    args = p.parse_args()

    # The diagram stands alone: it is what to run when there is no archive yet.
    if args.diagram:
        plot_resultant_cell(synthetic_sample(), args.diagram)
        print(f"wrote {args.diagram}")
        return

    missing = [n for n in ("current", "lat", "lon", "time") if getattr(args, n) is None]
    if missing:
        p.error("--" + ", --".join(missing) + " are required unless --diagram is given")

    # A missing file is the commonest way to arrive here, usually by copying the example
    # paths out of a docstring. open_forcing would raise FileNotFoundError naming the
    # path, which is accurate and tells nobody how to get one, so say it here instead.
    for flag, path in (("--current", args.current), ("--wind", args.wind)):
        if path and not Path(path).exists():
            p.error(
                f"{flag} {path} does not exist. This module reads files the fetchers "
                "wrote; it does not fetch. Either pull one with `python -m sar.fetch."
                f"{'current' if flag == '--current' else 'wind'} --help`, or write a "
                "small invented pair with `python scripts/make_demo_forcing.py --out "
                "data`. To see the method drawn without any data at all, run "
                "`python -m sar.model.interpolate --diagram`."
            )

    from sar.utils.data_io import open_forcing

    include_wind = INCLUDE_WIND
    if args.with_wind:
        include_wind = True
    if args.no_wind:
        include_wind = False

    current_ds = open_forcing(args.current)
    wind_ds = open_forcing(args.wind) if (include_wind and args.wind) else None

    # Land and out-of-coverage are designed answers, not crashes: a traceback would
    # suggest the module is broken when it is doing exactly what it was asked to do.
    try:
        result = resultant_at(args.lat, args.lon, args.time, current_ds, wind_ds, include_wind)
    except (OutOfCoverageError, MissingCornerError) as exc:
        raise SystemExit(f"{type(exc).__name__}: {exc}") from None

    print(json.dumps(result, indent=2))

    if args.plot:
        sample = sample_field(current_ds, CURRENT_VARS, args.lat, args.lon, args.time)
        plot_resultant_cell(sample, args.plot)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
