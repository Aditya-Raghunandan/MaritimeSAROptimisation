"""Tests for the probability map grid (issue #7, ADR002). Pure arithmetic, no network.

Three of issue #7's four acceptance criteria are checked here exactly as written. The
fourth -- "resolution defaults to 1/12 degree" -- is REFUSED by ADR002 and there is a test
asserting that it stays refused, because a default that comes back is a default nobody
chose and it would reintroduce the bias silently.
"""

import numpy as np
import pytest

from sar.model.grid import ProbabilityGrid, normalise
from sar.model.interpolate import OutOfCoverageError
from sar.utils.geo import M_PER_DEG_LAT, metres_per_degree_lon, to_display_longitude
from sar.viz.export import grid_spec

CELL_M = 250.0
CENTRE_LAT, CENTRE_LON = 26.5, -76.0


def cloud(n=2000, centre_lat=CENTRE_LAT, centre_lon=CENTRE_LON, spread_km=20.0, seed=7):
    """A synthetic particle cloud, isotropic IN METRES about a centre.

    Built in metres and converted per latitude, so the same physical cloud can be placed
    at any latitude and must produce the same binned picture. That is the equal-area test.
    """
    rng = np.random.default_rng(seed)
    east_m, north_m = rng.normal(0, spread_km * 1000, (2, n))
    lats = centre_lat + north_m / M_PER_DEG_LAT
    lons = centre_lon + east_m / float(metres_per_degree_lon(centre_lat))
    return lats, lons


def square_grid(n=41):
    """A small hand-built grid centred on the box centre, for the arithmetic tests."""
    dlat = CELL_M / M_PER_DEG_LAT
    dlon = CELL_M / float(metres_per_degree_lon(CENTRE_LAT))
    return ProbabilityGrid(
        lat0=CENTRE_LAT, dlat=dlat, nlat=n,
        lon0=(CENTRE_LON % 360.0), dlon=dlon, nlon=n,
    )


# --- issue #7's acceptance criteria -------------------------------------------------


def test_origin_maps_to_cell_zero_zero():
    """#7 criterion 1: position_to_cell(origin, origin) returns (0, 0)."""
    g = square_grid()
    assert g.position_to_cell(g.lat0, to_display_longitude(g.lon0)) == (0, 0)


def test_round_trip_at_a_cell_centre_is_exact():
    """#7 criterion 2: cell_to_position inverts position_to_cell at a cell centre."""
    g = square_grid()
    for row, col in [(0, 0), (1, 0), (0, 1), (17, 23), (40, 40)]:
        lat, lon = g.cell_to_position(row, col)
        assert g.position_to_cell(lat, lon) == (row, col)


def test_a_position_nearer_a_neighbour_lands_in_that_neighbour():
    """#7 criterion 3: past the halfway line, the neighbouring cell owns the position."""
    g = square_grid()
    lat, lon = g.cell_to_position(5, 5)
    just_inside = g.position_to_cell(lat + 0.49 * g.dlat, lon + 0.49 * g.dlon)
    just_past = g.position_to_cell(lat + 0.51 * g.dlat, lon + 0.51 * g.dlon)
    assert just_inside == (5, 5)
    assert just_past == (6, 6)


def test_resolution_has_no_default_and_that_is_deliberate():
    """#7 criterion 4 is REFUSED. ADR002: cell size is a parameter nobody gets by accident.

    1/12 degree square would be ~9.3 x 8.3 km here, against a sweep width of 185 m, and it
    is not even the HYCOM grid it claims to match (0.04 lat x 0.08 lon, latitude finer).
    """
    lats, lons = cloud()
    with pytest.raises(TypeError):
        ProbabilityGrid.from_envelope(lats, lons)       # no cell_m
    with pytest.raises(ValueError, match="positive metres"):
        ProbabilityGrid.from_envelope(lats, lons, cell_m=0)


# --- the anisotropy that a single `resolution` cannot express -----------------------


def test_longitude_step_is_larger_than_latitude_step_by_one_over_cos_lat():
    g = ProbabilityGrid.from_envelope(*cloud(), cell_m=CELL_M)
    expected = 1.0 / np.cos(np.radians(g.centre_lat))
    assert g.dlon / g.dlat == pytest.approx(expected, rel=1e-9)
    assert expected == pytest.approx(1.117, abs=1e-3)   # the number ADR002 quotes


def test_cells_are_square_in_metres_at_every_latitude():
    """The bias test in its simplest form: 250 m cells are 250 m cells at 17 N and 36 N."""
    for lat in (17.0, 26.5, 36.0):
        g = ProbabilityGrid.from_envelope(*cloud(centre_lat=lat), cell_m=CELL_M)
        ns, ew = g.cell_size_m
        assert ns == pytest.approx(CELL_M, rel=1e-9)
        assert ew == pytest.approx(CELL_M, rel=1e-3)


def column_spread(counts):
    """The east-west spread of a binned cloud, in cells. The bias shows up here."""
    weight = counts.sum(axis=0).astype(float)
    index = np.arange(weight.size)
    mean = (weight * index).sum() / weight.sum()
    return float(np.sqrt((weight * (index - mean) ** 2).sum() / weight.sum()))


def test_the_same_cloud_bins_identically_at_17_N_and_36_N():
    """EQUAL-AREA. The defect this catches is a north-south bias that looks like physics.

    The identical physical cloud, placed 19 degrees further north, must produce the same
    picture. It does, to the only precision the question has: 0.018 % of cells differ by
    exactly one particle, which is particles sitting within a float ulp of a cell boundary
    landing on either side of it after two different metre-to-degree conversions. What
    must NOT differ is the spread, and the control below shows what it looks like when it
    does.
    """
    south = ProbabilityGrid.from_envelope(*cloud(centre_lat=17.0), cell_m=CELL_M)
    north = ProbabilityGrid.from_envelope(*cloud(centre_lat=36.0), cell_m=CELL_M)

    counts_s, lost_s = south.bin(*cloud(centre_lat=17.0))
    counts_n, lost_n = north.bin(*cloud(centre_lat=36.0))

    assert (lost_s, lost_n) == (0, 0)
    assert counts_s.shape == counts_n.shape
    assert counts_s.sum() == counts_n.sum()
    assert column_spread(counts_n) == pytest.approx(column_spread(counts_s), rel=1e-3)
    assert np.abs(counts_s - counts_n).max() <= 1
    assert (counts_s != counts_n).mean() < 1e-3


def test_a_fixed_degree_grid_would_have_shown_the_bias():
    """The control: without the cos(lat) step the same cloud reads 18 % wider at 36 N.

    Included so the equal-area test above cannot pass vacuously. It also fixes the
    direction, which is the opposite of the intuition: a degree of longitude SHRINKS
    going north, so a fixed-degree grid slices the same physical cloud into MORE columns
    and the map reads broader and thinner -- a search area that looks bigger and less
    certain than it is, purely because of where it is. cos(17)/cos(36) = 1.182.
    """
    dlat = CELL_M / M_PER_DEG_LAT
    spreads = {}
    for lat in (17.0, 36.0):
        lats, lons = cloud(centre_lat=lat)
        naive = ProbabilityGrid(                     # one step for both axes: the bug
            lat0=lat - dlat * 600, dlat=dlat, nlat=1201,
            lon0=(lons.mean() % 360.0) - dlat * 600, dlon=dlat, nlon=1201,
        )
        counts, lost = naive.bin(lats, lons)
        assert lost == 0                             # not clipping, genuinely wider
        spreads[lat] = column_spread(counts)

    ratio = spreads[36.0] / spreads[17.0]
    assert ratio == pytest.approx(np.cos(np.radians(17.0)) / np.cos(np.radians(36.0)),
                                  rel=0.02)
    assert ratio > 1.15                              # and it is a big effect, not a rounding one


# --- fitting the box ------------------------------------------------------------------


def test_the_box_contains_the_cloud_it_was_fitted_to():
    lats, lons = cloud()
    g = ProbabilityGrid.from_envelope(lats, lons, cell_m=CELL_M)
    _, lost = g.bin(lats, lons)
    assert lost == 0


def test_margin_widens_the_box_by_the_searchers_reach():
    lats, lons = cloud()
    tight = ProbabilityGrid.from_envelope(lats, lons, cell_m=CELL_M)
    wide = ProbabilityGrid.from_envelope(lats, lons, cell_m=CELL_M, margin_km=10.0)
    assert wide.extent_km[0] - tight.extent_km[0] == pytest.approx(20.0, abs=CELL_M / 1000)
    assert wide.centre_lat == pytest.approx(tight.centre_lat, abs=1e-9)


def test_the_box_is_centred_on_the_cloud_not_hung_off_one_corner():
    lats, lons = cloud()
    g = ProbabilityGrid.from_envelope(lats, lons, cell_m=CELL_M)
    assert g.centre_lat == pytest.approx((lats.min() + lats.max()) / 2, abs=g.dlat)


def test_cell_counts_can_be_forced_to_a_multiple_for_coarsening():
    g = ProbabilityGrid.from_envelope(*cloud(), cell_m=CELL_M, cells_multiple_of=32)
    assert g.nlat % 32 == 0 and g.nlon % 32 == 0


def test_a_cloud_too_wide_to_be_a_search_box_is_refused():
    """The map is not the domain. A 250 m grid across the Atlantic is 1e9 cells."""
    with pytest.raises(ValueError, match="the map is not the domain|The map is not the domain"):
        ProbabilityGrid.from_envelope([26.0, 26.1], [-76.0, 100.0], cell_m=CELL_M)


def test_a_cloud_straddling_the_0_360_seam_is_not_read_as_a_whole_globe():
    """D020 stores 0-360, so the seam is at Greenwich. This box is 22 km wide, not 4000."""
    g = ProbabilityGrid.from_envelope([0.0, 0.0], [359.9, 0.1], cell_m=CELL_M)
    assert g.extent_km[1] == pytest.approx(22.3, abs=0.5)


# --- the box the episode actually runs in -----------------------------------------


def test_centred_on_gives_the_box_size_it_was_asked_for():
    g = ProbabilityGrid.centred_on(CENTRE_LAT, CENTRE_LON, box_km=100.0, cell_m=500.0)
    ns, ew = g.extent_km
    assert ns == pytest.approx(100.0, abs=1.0)
    assert ew == pytest.approx(100.0, abs=1.0)
    assert g.shape[0] == g.shape[1]                  # square in metres means square in cells


def test_centred_on_is_the_same_size_at_every_latitude():
    """ADR002's reason for it: one agent cell must mean the same distance every episode."""
    sizes = [ProbabilityGrid.centred_on(lat, CENTRE_LON, 100.0, 500.0).extent_km
             for lat in (17.0, 26.5, 36.0)]
    assert len({s[0] for s in sizes}) == 1
    for _, ew in sizes:
        assert ew == pytest.approx(sizes[0][1], rel=1e-6)


def test_centred_on_puts_the_anchor_at_the_middle_of_the_box():
    g = ProbabilityGrid.centred_on(CENTRE_LAT, CENTRE_LON, box_km=100.0, cell_m=500.0)
    mid = (g.nlat - 1) // 2
    lat, lon = g.cell_to_position(mid, mid)
    assert (lat, lon) == pytest.approx((CENTRE_LAT, CENTRE_LON), abs=g.dlon)


# --- conventions ----------------------------------------------------------------------


def test_either_longitude_convention_lands_in_the_same_cell():
    """D020 stores 0-360 and the frontend speaks -180..180. Both must index the same."""
    g = square_grid()
    lat, lon = g.cell_to_position(10, 10)
    assert g.position_to_cell(lat, lon) == g.position_to_cell(lat, lon % 360.0)


def test_to_spec_matches_what_grid_spec_would_say_about_the_same_axes():
    """The manifest contract: the browser must describe this grid the way it does HYCOM's."""
    g = ProbabilityGrid.from_envelope(*cloud(), cell_m=CELL_M)
    spec, theirs = g.to_spec(), grid_spec(g.lats, g.lons)
    assert spec.keys() == theirs.keys()
    for key in spec:
        assert spec[key] == pytest.approx(theirs[key], rel=1e-9), key


def test_to_spec_reports_display_longitude():
    g = square_grid()
    assert g.to_spec()["lon0"] == pytest.approx(CENTRE_LON, abs=1e-9)


# --- binning, mass and coarsening -----------------------------------------------------


def test_bin_agrees_with_position_to_cell_particle_by_particle():
    """One implementation or two? If these ever disagree, there are two."""
    g = ProbabilityGrid.from_envelope(*cloud(n=200), cell_m=CELL_M)
    lats, lons = cloud(n=200)
    counts, _ = g.bin(lats, lons)
    rows, cols = g.position_to_cell(lats, lons)
    by_hand = np.zeros(g.shape, dtype=int)
    np.add.at(by_hand, (rows, cols), 1)
    np.testing.assert_array_equal(counts, by_hand)


def test_particles_outside_the_box_are_counted_as_lost_not_raised():
    """D016 wants lost mass logged per timestep, so binning cannot raise on it."""
    g = square_grid(n=11)
    inside_lat, inside_lon = g.cell_to_position(5, 5)
    lats = np.array([inside_lat, inside_lat + 90 * g.dlat])
    lons = np.array([inside_lon, inside_lon])
    counts, lost = g.bin(lats, lons)
    assert (counts.sum(), lost) == (1, 1)


def test_position_to_cell_still_raises_for_a_position_off_the_grid():
    g = square_grid(n=11)
    with pytest.raises(OutOfCoverageError, match="outside the grid"):
        g.position_to_cell(g.lat0 + 90 * g.dlat, to_display_longitude(g.lon0))


def test_normalise_sums_to_one_with_beached_mass_still_on_the_map():
    """D016: beached particles are frozen in place and KEEP their mass."""
    g = square_grid(n=11)
    counts, _ = g.bin(*cloud(n=500, spread_km=0.4))
    beached = counts[0].sum()                       # pretend the bottom row is ashore
    p = normalise(counts, beached_mass=beached)
    assert p.sum() == pytest.approx(1.0)
    assert p[0].sum() == pytest.approx(beached / counts.sum())


def test_normalise_refuses_an_empty_map_rather_than_dividing_by_zero():
    with pytest.raises(ValueError, match="every particle is outside"):
        normalise(np.zeros((4, 4)))


def test_normalise_refuses_more_beached_mass_than_there_are_particles():
    with pytest.raises(ValueError, match="exceeds"):
        normalise(np.ones((4, 4)), beached_mass=100.0)


def test_coarsen_preserves_total_mass():
    """R5a's observation is coarsened, and a resize would not conserve mass."""
    g = ProbabilityGrid.from_envelope(*cloud(), cell_m=CELL_M, cells_multiple_of=8)
    counts, _ = g.bin(*cloud())
    _, coarse = g.coarsen(counts, 8)
    assert coarse.sum() == counts.sum()
    assert coarse.shape == (g.nlat // 8, g.nlon // 8)


def test_coarsen_returns_a_grid_that_still_describes_the_same_ground():
    """A coarsened array whose grid was not coarsened with it is a map in the wrong place."""
    g = ProbabilityGrid.from_envelope(*cloud(), cell_m=CELL_M, cells_multiple_of=8)
    coarse_grid, _ = g.coarsen(np.zeros(g.shape), 8)
    for first in (True, False):
        fine = g.lat0 + g.dlat * (0 if first else g.nlat - 1)
        coarse = coarse_grid.lat0 + coarse_grid.dlat * (0 if first else coarse_grid.nlat - 1)
        assert abs(coarse - fine) == pytest.approx(3.5 * g.dlat)   # half a block of 8
    assert coarse_grid.centre_lat == pytest.approx(g.centre_lat, abs=1e-9)


def test_coarsen_refuses_a_factor_that_would_leave_a_partial_block():
    g = ProbabilityGrid.from_envelope(*cloud(), cell_m=CELL_M, cells_multiple_of=8)
    with pytest.raises(ValueError, match="does not divide"):
        g.coarsen(np.zeros(g.shape), 7)


def test_coarsen_to_an_agent_observation_of_the_expected_size():
    """The end-to-end shape the RL env asks for: a 100 km box seen as 32x32."""
    lats, lons = cloud(spread_km=8.0)
    g = ProbabilityGrid.centred_on(CENTRE_LAT, CENTRE_LON, box_km=100.0, cell_m=500.0,
                                   cells_multiple_of=32)
    counts, lost = g.bin(lats, lons)
    obs_grid, obs = g.coarsen(counts, g.nlat // 32)

    assert obs.shape == (32, 32)
    assert obs.sum() == counts.sum() == lats.size - lost
    # One agent cell is kilometres across, which is why coverage is swept on the particles
    # and not on this grid -- ADR002 section 4. At 64x64 over 100 km one pass sweeps 185 m
    # of a 1562 m cell, leaving 88 % of it unsearched but looking visited.
    assert obs_grid.cell_size_m[0] > 1000.0


# --- construction guards ---------------------------------------------------------------


def test_a_descending_axis_is_refused():
    """D020 trap #2: a descending axis selects empty instead of raising."""
    with pytest.raises(ValueError, match="ascending"):
        ProbabilityGrid(lat0=26.5, dlat=-0.002, nlat=10, lon0=284.0, dlon=0.002, nlon=10)


def test_a_cloud_with_nan_positions_is_refused():
    with pytest.raises(ValueError, match="NaN"):
        ProbabilityGrid.from_envelope([26.0, np.nan], [-76.0, -76.1], cell_m=CELL_M)


def test_mismatched_lat_and_lon_shapes_are_refused():
    g = square_grid()
    with pytest.raises(ValueError, match="match in shape"):
        g.position_to_cell(np.zeros(3), np.zeros(4))
