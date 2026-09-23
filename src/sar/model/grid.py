"""grid.py: the probability map's grid, and position-to-cell translation (issue #7).

WHAT THIS IS. The Monte Carlo ensemble is a cloud of particle positions. The probability
map is that cloud counted into cells. This module owns the cells: where they are, which
one a position falls in, and how counts on them are normalised and coarsened. It owns no
physics and reads no files.

WHY IT IS NOT WHAT ISSUE #7 ASKED FOR. #7 specifies one scalar
`resolution=GRID_RESOLUTION_DEG` defaulting to 1/12 degree square, "matching the HYCOM
native grid". Three things are wrong with that, and ADR002 is the decision:

  1. The HYCOM grid used here is 0.04 deg lat x 0.08 deg lon -- neither 1/12 nor square,
     and LATITUDE IS THE FINER AXIS. One scalar cannot express two steps.
  2. All four of #7's acceptance criteria pass against a square grid that does not exist.
     The defect would surface later as a north-south bias in the binned output, and would
     be indistinguishable from a physical result.
  3. The probability map should not be on the forcing grid at all. An 8 km cell is about
     four times the whole width of a tight ensemble at arrival, so the map would be a
     handful of cells across at the moment it carries the most information.

WHAT SETS THE CELL SIZE, since it is the question this module makes people ask. NOT the
sweep width. W is how wide a strip the searcher clears; a cell here is how finely the
target's POSITION DISTRIBUTION is described. Those are different quantities, and they are
decoupled by ADR002 section 4, which puts coverage on the particles rather than on a grid
mask -- so the reward is exact at any cell size. What actually bounds it, over a 100 km box:

    below ~200 m   sampling noise. 1/sqrt(k) per cell, so noise scales as 1/cell:
                   15.5 % at 250 m and 39 % at 100 m, both at N = 1e6
    memory         scales as 1/cell^2. Counts are uint16, lossless here (peak cell count
                   2,487 at 250 m for a tight ensemble, against a 65,535 ceiling) and half
                   the bytes of float32: 45 MB per scenario at 250 m, 11 MB at 500 m
    the browser    a published chunk is 48 frames. Measured at 1.11 MB and 45 ms for the
                   forcing store; 14.6 MB at 250 m and 3.7 MB at 500 m. THIS is why the
                   archive is 500 m and the episode map, which is binned on demand and
                   never written to disk, can be finer
    above ~400 m   a cell comparable to the whole distribution gives a blob, not a shape.
                   The bound rests on an ensemble spread that has NOT been measured yet

So `cell_m` is a required argument: the defensible statement is a window of roughly
200-500 m, not a single number, and the experiments pick inside it.


So the three acceptance criteria that are about ARITHMETIC are met exactly, and the
fourth -- the default -- is deliberately refused: `cell_m` is required and has no default.
A default resolution is a resolution nobody chose.

THE THREE GRIDS. `grid` has been denoting three different objects in this project, which
is why #7 was unanswerable:

    forcing       ERA5 0.25 deg, HYCOM 0.08 x 0.04 deg    given by the data products
    probability   250-500 m                               THIS MODULE
    observation   32x32 to 64x64                          `coarsen`, for the RL env

CELL CENTRES, NOT CORNERS. (lat0, lon0) is the CENTRE of cell (0, 0), which is what #7
means by "each data point is the centre of a cell" and what makes `position_to_cell` and
`cell_to_position` exact inverses at a centre. It is also how `sar.viz.export.grid_spec`
already describes the forcing grids, so `to_spec()` needs no translation.

LONGITUDE. Stored 0-360 per D020, converted to -180..180 only in `to_spec()`, which is a
presentation boundary. Indexing unwraps around `lon0`, so a caller may pass either
convention and get the same cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sar.model.interpolate import OutOfCoverageError
from sar.utils.geo import (
    M_PER_DEG_LAT,
    metres_per_degree_lon,
    to_display_longitude,
    to_store_longitude,
)


@dataclass(frozen=True)
class ProbabilityGrid:
    """A regular lat/lon grid with SEPARATE steps, sized so cells are square in metres.

    Six fields, and they are deliberately the same six that `sar.viz.export.grid_spec`
    produces and `frontend/src/layers.js` consumes, so a map written here is renderable
    without a second description of the same grid.

    `dlat` and `dlon` differ by 1/cos(latitude) -- 1.117 at 26.5 N. That ratio is the
    whole reason a single `resolution` argument cannot work. Keeping two steps costs one
    extra field and buys square cells without introducing a map projection: over a 100 km
    box the residual distortion is 0.78 %, about a metre on a 250 m cell.
    """

    lat0: float
    dlat: float
    nlat: int
    lon0: float         # STORE convention, 0-360 (D020)
    dlon: float
    nlon: int

    def __post_init__(self) -> None:
        if self.dlat <= 0 or self.dlon <= 0:
            raise ValueError(
                f"steps must be positive and ascending, got dlat={self.dlat}, "
                f"dlon={self.dlon}. A descending axis is D020 trap #2 and selects empty."
            )
        if self.nlat < 1 or self.nlon < 1:
            raise ValueError(f"a grid needs at least one cell, got {self.nlat}x{self.nlon}")

    # -- description ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        """(rows, columns) = (nlat, nlon), matching every array this module returns."""
        return (self.nlat, self.nlon)

    @property
    def lats(self) -> np.ndarray:
        """Cell-centre latitudes, ascending."""
        return self.lat0 + self.dlat * np.arange(self.nlat)

    @property
    def lons(self) -> np.ndarray:
        """Cell-centre longitudes in STORE convention (0-360), ascending."""
        return self.lon0 + self.dlon * np.arange(self.nlon)

    @property
    def centre_lat(self) -> float:
        """The latitude the cell size is square at."""
        return float(self.lat0 + self.dlat * (self.nlat - 1) / 2)

    @property
    def cell_size_m(self) -> tuple[float, float]:
        """(north-south, east-west) cell size in metres at the box centre.

        The two agree to within the box's own distortion when the grid came from
        `from_envelope`; they are reported separately rather than averaged so that a grid
        built by hand cannot quietly claim to be square when it is not.
        """
        return (
            self.dlat * M_PER_DEG_LAT,
            self.dlon * float(metres_per_degree_lon(self.centre_lat)),
        )

    @property
    def extent_km(self) -> tuple[float, float]:
        """(north-south, east-west) size of the whole box in km, centre to centre."""
        ns, ew = self.cell_size_m
        return ((self.nlat - 1) * ns / 1000.0, (self.nlon - 1) * ew / 1000.0)

    # -- construction -----------------------------------------------------------------

    @classmethod
    def centred_on(
        cls,
        lat: float,
        lon: float,
        box_km: float,
        cell_m: float,
        cells_multiple_of: int | None = None,
    ) -> "ProbabilityGrid":
        """A box of a GIVEN size, anchored on a point. Square in metres, so square in cells.

        THIS IS THE ONE THE EPISODE USES. ADR002 fixes the box size across scenarios, and
        `from_envelope` cannot do that because it fits the size to whatever cloud it is
        handed. A constant box size is what makes one agent cell mean the same distance in
        every episode -- otherwise a policy learns "the mass is three cells away" and three
        cells is a different distance each time.

        Anchor it on the cloud's centroid AT THE SEARCHER'S ARRIVAL TIME. During the
        45-minute on-scene window the cloud drifts 4.9 km, so one box per episode holds;
        it is between episodes that it has to be re-fitted.

        Mass outside the box is `lost_mass` (D016), and the boundary is a wall for the
        agent. Use `from_envelope` first, on a representative scenario, to find out what
        `box_km` needs to be.
        """
        if box_km <= 0:
            raise ValueError(f"box_km must be positive, got {box_km}")
        if cell_m <= 0:
            raise ValueError(f"cell_m must be positive metres, got {cell_m}")

        dlat = cell_m / M_PER_DEG_LAT
        dlon = cell_m / float(metres_per_degree_lon(lat))
        n = cls._cell_count(box_km * 1000.0 / M_PER_DEG_LAT, dlat, cells_multiple_of)
        return cls(
            lat0=lat - dlat * (n - 1) / 2, dlat=dlat, nlat=n,
            lon0=float(to_store_longitude(lon - dlon * (n - 1) / 2)), dlon=dlon, nlon=n,
        )

    @classmethod
    def from_envelope(
        cls,
        lats,
        lons,
        cell_m: float,
        margin_km: float = 0.0,
        cells_multiple_of: int | None = None,
        max_cells: int = 20_000_000,
        max_box_km: float = 2100.0,
    ) -> "ProbabilityGrid":
        """Fit a box to a particle cloud, with cells square in metres at its centre.

        `cell_m` is REQUIRED. The defensible statement is a WINDOW of roughly 200-500 m,
        not a single number -- see the module docstring for what sets each end -- so it is
        a parameter the experiments pick inside, and never a constant.

        FIT TO THE SPREAD, NEVER TO THE TRAVEL. Pass the cloud AT THE SEARCHER'S ARRIVAL
        TIME, not the whole run. At 1.8 m/s a target travels 467 km in 72 h, so a box
        containing an entire run is ~495 km at 48 h and costs 361 MB per scenario against
        23 MB for the 100 km on-scene box. During the 45-minute on-scene window the cloud
        drifts only 4.9 km, which is why one fixed box per episode is enough.

        `margin_km` widens the box beyond the cloud -- the searcher's reach, so the agent
        can fly to where the mass is going. `cells_multiple_of` rounds the cell counts up,
        for callers that will later `coarsen` by that factor and need it to divide evenly.

        This fits the box to the cloud it is given, so the size varies between scenarios.
        Use it to find out what box size a scenario needs; use `centred_on` to build the
        one the episode runs in, which ADR002 requires to be the same size every time.
        """
        if cell_m <= 0:
            raise ValueError(f"cell_m must be positive metres, got {cell_m}")
        if margin_km < 0:
            raise ValueError(f"margin_km cannot be negative, got {margin_km}")

        lats = np.atleast_1d(np.asarray(lats, dtype=float))
        lons = np.atleast_1d(to_store_longitude(np.asarray(lons, dtype=float)))
        if lats.size == 0 or lats.size != lons.size:
            raise ValueError(
                f"need matching non-empty lats and lons, got {lats.size} and {lons.size}"
            )
        if not (np.isfinite(lats).all() and np.isfinite(lons).all()):
            raise ValueError("cloud contains NaN or infinite positions; beached and lost "
                             "particles keep real positions (D016), so this is a bug upstream")

        # Unwrap longitude onto a continuous axis before taking a span, or a cloud either
        # side of the 0/360 seam reads as 360 degrees wide. D020 stores 0-360, so the seam
        # is at Greenwich and this project's box never meets it -- but a silent 4000 km box
        # is not a failure worth saving three lines on.
        lons = lons[0] + ((lons - lons[0] + 180.0) % 360.0) - 180.0
        if np.ptp(lons) > 180.0:
            raise ValueError(
                f"the cloud spans {np.ptp(lons):.1f} degrees of longitude, which is not a "
                "search box; check the longitude convention of the input"
            )

        centre_lat = float((lats.min() + lats.max()) / 2)
        dlat = cell_m / M_PER_DEG_LAT
        dlon = cell_m / float(metres_per_degree_lon(centre_lat))

        margin_m = margin_km * 1000.0
        span_lat = float(np.ptp(lats)) + 2 * margin_m / M_PER_DEG_LAT
        span_lon = float(np.ptp(lons)) + 2 * margin_m / float(metres_per_degree_lon(centre_lat))

        # Two guards, because either alone has a hole. A box 176 degrees wide and a
        # tenth of a degree tall is only 3 million cells and is still not a search box;
        # a compact box at a 1 m cell size is a sane extent and 10^10 cells.
        box_km = (span_lat * M_PER_DEG_LAT / 1000.0,
                  span_lon * float(metres_per_degree_lon(centre_lat)) / 1000.0)
        if max(box_km) > max_box_km:
            raise ValueError(
                f"this cloud spans {box_km[0]:.0f} x {box_km[1]:.0f} km, over the "
                f"{max_box_km:.0f} km limit -- which is the width of the whole study "
                "domain. The map is not the domain. Fit to the cloud at ARRIVAL, not to "
                "the whole run: travel is 467 km in 72 h against 4.9 km of drift during "
                "the on-scene window."
            )

        nlat = cls._cell_count(span_lat, dlat, cells_multiple_of)
        nlon = cls._cell_count(span_lon, dlon, cells_multiple_of)
        if nlat * nlon > max_cells:
            raise ValueError(
                f"fitting this cloud needs a {nlat}x{nlon} grid, which is "
                f"{nlat * nlon / 1e6:.1f} million cells and over the {max_cells / 1e6:.0f} "
                "million limit. The map is not the domain: the whole 17-36 N box at 500 m "
                "is 17.6 million cells and 9.9 GB per scenario, which ADR002 rejects. Fit "
                "to the cloud at ARRIVAL, not to the whole run -- a run's travel is 467 km "
                "in 72 h against a 4.9 km drift during the on-scene window."
            )

        # Centre the box on the cloud so that rounding the cell count up adds the same
        # margin on both sides, rather than all of it to the north-east.
        centre_lon = float((lons.min() + lons.max()) / 2)
        return cls(
            lat0=centre_lat - dlat * (nlat - 1) / 2, dlat=dlat, nlat=nlat,
            lon0=float(to_store_longitude(centre_lon - dlon * (nlon - 1) / 2)),
            dlon=dlon, nlon=nlon,
        )

    @staticmethod
    def _cell_count(span_deg: float, step_deg: float, multiple_of: int | None) -> int:
        """Cells needed to cover a span, rounded up, optionally to a multiple."""
        n = int(np.ceil(span_deg / step_deg)) + 1
        if multiple_of:
            if multiple_of < 1:
                raise ValueError(f"cells_multiple_of must be at least 1, got {multiple_of}")
            n = int(np.ceil(n / multiple_of) * multiple_of)
        return max(n, multiple_of or 1)

    # -- the translation issue #7 asked for -------------------------------------------

    def position_to_cell(self, lat, lon):
        """(lat, lon) -> (row, column). Scalars in, scalars out; arrays in, arrays out.

        Raises OutOfCoverageError if any position is outside the box -- the same exception
        `sar.model.interpolate` raises for the same situation, so a caller catches one
        type. Use `bin` instead when positions outside are expected and should be counted
        as lost mass rather than raised on (D016).
        """
        rows, cols, inside = self._cells(lat, lon)
        if not inside.all():
            bad = int((~inside).sum())
            first = int(np.argmin(inside))
            raise OutOfCoverageError(
                f"{bad} of {inside.size} position(s) fall outside the grid, which covers "
                f"{self.lats[0]:.4f}..{self.lats[-1]:.4f} N and "
                f"{to_display_longitude(self.lons[0]):.4f}.."
                f"{to_display_longitude(self.lons[-1]):.4f} E; "
                f"the first is row {int(rows.flat[first])}, column {int(cols.flat[first])}"
            )
        if np.isscalar(lat) and np.isscalar(lon):
            return int(rows.item()), int(cols.item())
        return rows, cols

    def cell_to_position(self, row, col):
        """(row, column) -> the (lat, lon) of that cell's CENTRE, in display longitude.

        The exact inverse of `position_to_cell` at a cell centre, which is #7's second
        acceptance criterion. Longitude comes back as -180..180 because every consumer of
        a position -- figures, KML (R8c), the frontend -- is a presentation boundary.
        """
        row = np.asarray(row)
        col = np.asarray(col)
        if ((row < 0) | (row >= self.nlat) | (col < 0) | (col >= self.nlon)).any():
            raise OutOfCoverageError(
                f"cell ({row}, {col}) is outside a grid of shape {self.shape}"
            )
        lat = self.lat0 + self.dlat * row
        lon = to_display_longitude(self.lon0 + self.dlon * col)
        if row.ndim == 0 and col.ndim == 0:
            return float(lat), float(lon)
        return lat, lon

    def _cells(self, lat, lon):
        """Row and column indices plus an inside mask. The ONE implementation.

        `position_to_cell` and `bin` both go through here. Two implementations of the
        same index arithmetic is how the browser and the engine came to disagree by 28
        degrees on the same sample, and it is not a mistake worth making twice.

        Ties -- a position exactly on a cell boundary -- go to the even index, because
        `np.rint` is half-to-even, as is `interpolate.nearest_index`. It is a measure-zero
        case on float positions; what matters is that both modules resolve it the same way.
        """
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        if lat.shape != lon.shape:
            raise ValueError(f"lat and lon must match in shape, got {lat.shape} and {lon.shape}")

        # Unwrap around lon0 so either longitude convention lands in the same cell.
        lon = self.lon0 + ((lon - self.lon0 + 180.0) % 360.0) - 180.0

        rows = np.rint((lat - self.lat0) / self.dlat).astype(np.int64)
        cols = np.rint((lon - self.lon0) / self.dlon).astype(np.int64)
        inside = (rows >= 0) & (rows < self.nlat) & (cols >= 0) & (cols < self.nlon)
        return np.atleast_1d(rows), np.atleast_1d(cols), np.atleast_1d(inside)

    # -- counting -----------------------------------------------------------------------

    def bin(self, lats, lons) -> tuple[np.ndarray, int]:
        """Count a particle cloud into cells. Returns (counts, lost).

        `lost` is the number of particles outside the box, and it is RETURNED RATHER THAN
        RAISED because D016 requires it to be logged per timestep: mass that leaves the
        grid has left the problem, and a map that silently renormalises over what remains
        hides it. Beached particles are frozen in place and keep their mass, so they are
        inside `counts` and not in `lost`.

        Plain binning, not kernel density estimation. ADR002 section 2: a bandwidth h
        inflates the estimated spread from sigma to sqrt(sigma^2 + h^2), so the 90 %
        contour R2c validates against would be too big and MORE true drifter positions
        would fall inside it -- the validation would pass more easily because of an
        estimator choice. +11.8 % at sigma = 2 km, which is exactly when the ensemble is
        tight and the map matters most. The speckle that smoothing would hide is answered
        with particles instead: N = 1e6 costs 504 s per scenario, ~8 minutes of wall clock
        as a Slurm array over 35 scenarios.
        """
        rows, cols, inside = self._cells(lats, lons)
        flat = rows[inside] * self.nlon + cols[inside]
        counts = np.bincount(flat, minlength=self.nlat * self.nlon)
        return counts.reshape(self.shape), int((~inside).sum())

    def coarsen(self, counts, factor: int) -> tuple["ProbabilityGrid", np.ndarray]:
        """Block-sum counts by an integer factor. Returns (coarser grid, coarser counts).

        MASS-PRESERVING BY CONSTRUCTION -- it sums, it does not resample. R5a says the
        agent's observation is "coarsened", and an image resize would interpolate, which
        both invents mass where there was none and loses it at the edges. The reward is
        newly swept probability mass, so an observation that does not conserve mass is an
        observation that misreports the reward.

        The grid comes back with it, because a coarsened array whose grid was not
        coarsened alongside it is a map 'painted' in the wrong place -- this project's
        most repeated defect.
        """
        counts = np.asarray(counts)
        if counts.shape != self.shape:
            raise ValueError(f"counts has shape {counts.shape}, this grid is {self.shape}")
        if factor < 1:
            raise ValueError(f"factor must be at least 1, got {factor}")
        if self.nlat % factor or self.nlon % factor:
            raise ValueError(
                f"factor {factor} does not divide a {self.nlat}x{self.nlon} grid. Build the "
                f"grid with from_envelope(..., cells_multiple_of={factor}) so that every "
                "block is full; trimming or padding here would move mass."
            )

        n_rows, n_cols = self.nlat // factor, self.nlon // factor
        coarse = counts.reshape(n_rows, factor, n_cols, factor).sum(axis=(1, 3))
        grid = ProbabilityGrid(
            # A block of `factor` cells has its centre half a block in from the first.
            lat0=self.lat0 + self.dlat * (factor - 1) / 2, dlat=self.dlat * factor, nlat=n_rows,
            lon0=self.lon0 + self.dlon * (factor - 1) / 2, dlon=self.dlon * factor, nlon=n_cols,
        )
        return grid, coarse

    # -- handing it on ------------------------------------------------------------------

    def to_spec(self) -> dict:
        """The six-field grid description the manifest and the browser already speak.

        Identical in form and convention to `sar.viz.export.grid_spec`, so a probability
        map publishes through the existing `raster` layer with no new client code:
        `RasterLayer.gridAt(frame)` already reads a per-frame grid, because D021 forbids a
        renderer from assuming one. Longitude is converted to display here and nowhere
        else.
        """
        return {
            "lat0": float(self.lat0), "dlat": float(self.dlat), "nlat": int(self.nlat),
            "lon0": float(to_display_longitude(self.lon0)),
            "dlon": float(self.dlon), "nlon": int(self.nlon),
        }


def normalise(counts, beached_mass: float = 0.0) -> np.ndarray:
    """Counts -> a distribution summing to 1 over the grid.

    D016 and D007: the map is a proper distribution per timestep, and BEACHED PARTICLES
    KEEP THEIR MASS -- they are frozen in place, not removed, so they are already inside
    `counts` and `beached_mass` is only asserted against, never added. Passing it makes
    the caller state what it believes, and a disagreement means beached particles were
    dropped somewhere upstream.

    Mass that left the grid is NOT restored: the result is the distribution CONDITIONAL on
    the target still being in the box, and `bin`'s `lost` count is what says how much that
    excludes. The reward is accumulated on raw weight BEFORE this is applied, or rewards
    are not comparable between timesteps.
    """
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        raise ValueError(
            "cannot normalise an empty map: every particle is outside the grid, which "
            "means the box was fitted to the wrong time or the wrong cloud"
        )
    if beached_mass and beached_mass > total:
        raise ValueError(
            f"beached mass {beached_mass} exceeds the {total} particles on the grid; "
            "beached particles are frozen in place and stay on the map (D016)"
        )
    return counts / total
