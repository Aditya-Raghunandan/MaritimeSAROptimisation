"""Tests for sar.pipeline.ensemble. Steady analytic fields and small runs, no network."""

import dataclasses
import json

import numpy as np
import pytest
from scipy.stats import chi2_contingency

import sar.pipeline.ensemble as ens
from sar.model.grid import ProbabilityGrid, normalise
from sar.model.position import EARTH_RADIUS_M
from sar.pipeline.ensemble import (
    Ensemble,
    describe_run,
    estimate_rows,
    format_span,
    parse_run_name,
    parse_span,
    read_csv,
    run_ensemble,
    run_name,
    save_stride,
    spawn_positions,
    synthetic_cloud,
    write_csv,
)
from sar.pipeline.forcing import ConstantForcing, as_particle_axes
from sar.pipeline.track import DriftPipeline

START = "2019-06-01T06:00"
LAT, LON = 26.5, -79.0
HOUR = 3600.0
EAST = ConstantForcing(current=(1.0, 0.0))

# Metres per degree on the engine's own sphere, so offsets convert back exactly.
M_PER_DEG = EARTH_RADIUS_M * np.pi / 180.0


def offsets_km(lat, lon, lat0=LAT, lon0=LON):
    """North and east offsets in km from a point, on the engine's sphere."""
    north = (lat - lat0) * M_PER_DEG / 1000.0
    east = (((lon - lon0 + 180.0) % 360.0) - 180.0) * M_PER_DEG * np.cos(np.radians(lat0)) / 1000.0
    return north, east


def small_run(**overrides):
    """A 10-particle, one-hour run saved every 15 minutes."""
    spec = {"forcing": EAST, "particles": 10, "start": START, "lat": LAT, "lon": LON,
            "duration": HOUR, "timestep": 60.0, "datum_sigma_km": 2.0, "sigma": 0.1,
            "seed": 7, "save_every": 900.0}
    spec.update(overrides)
    return spec


class LandNorthOf:
    """No forcing north of a latitude, as a land cell reads in the stored grids."""

    def __init__(self, lat):
        self.lat = lat

    def sample(self, lats, lons, time):
        lats, _ = as_particle_axes(lats, lons)
        current = np.tile([0.0, 1.0], (lats.size, 1))
        current[lats > self.lat] = np.nan
        return current, np.zeros_like(current)

    def describe(self):
        return {"backend": "test land"}


class TestEnsemble:
    def test_the_fields_are_exactly_those_46_names(self):
        names = [f.name for f in dataclasses.fields(Ensemble)]
        assert names == ["times", "lat", "lon", "weight", "beached"]

    def test_the_shapes_are_t_by_n(self):
        e = run_ensemble(**small_run())
        assert e.times.shape == (5,)
        assert e.lat.shape == e.lon.shape == e.beached.shape == (5, 10)
        assert e.weight.shape == (10,)

    def test_weights_that_do_not_sum_to_one_raise(self):
        with pytest.raises(ValueError, match="sum to 1"):
            Ensemble(np.array([np.datetime64(START)]), np.zeros((1, 2)), np.zeros((1, 2)),
                     np.array([0.5, 0.4]), np.zeros((1, 2), dtype=bool))

    def test_times_that_are_not_one_dimensional_raise(self):
        with pytest.raises(ValueError, match="1-D"):
            Ensemble(np.array([[np.datetime64(START)]]), np.zeros((1, 2)), np.zeros((1, 2)),
                     np.full(2, 0.5), np.zeros((1, 2), dtype=bool))

    def test_a_mismatched_shape_raises(self):
        with pytest.raises(ValueError, match="expected"):
            Ensemble(np.array([np.datetime64(START)]), np.zeros((1, 3)), np.zeros((1, 4)),
                     np.full(3, 1 / 3), np.zeros((1, 3), dtype=bool))


class TestSpawnPositions:
    def test_the_same_seed_gives_the_same_cloud(self):
        a = spawn_positions(LAT, LON, 100, 2.0, np.random.default_rng(3))
        b = spawn_positions(LAT, LON, 100, 2.0, np.random.default_rng(3))
        assert np.array_equal(a, b)

    def test_the_spread_matches_datum_sigma_at_ten_thousand(self):
        cloud = spawn_positions(LAT, LON, 10_000, 2.0, np.random.default_rng(1))
        north, east = offsets_km(cloud[:, 0], cloud[:, 1])
        # The sd of 10^4 draws has a standard error of 0.7 %, so 3 % is about four of them.
        assert north.std() == pytest.approx(2.0, rel=0.03)
        assert east.std() == pytest.approx(2.0, rel=0.03)

    def test_zero_spread_puts_every_particle_on_the_datum(self):
        cloud = spawn_positions(LAT, LON, 5, 0.0, np.random.default_rng(0))
        assert cloud == pytest.approx(np.tile([LAT, LON % 360.0], (5, 1)))

    def test_a_negative_spread_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            spawn_positions(LAT, LON, 10, -1.0, np.random.default_rng(0))

    @pytest.mark.parametrize("n", [0, -5, 2.5, True, "10"])
    def test_a_count_that_is_not_a_positive_integer_raises(self, n):
        with pytest.raises(ValueError, match="positive integer"):
            spawn_positions(LAT, LON, n, 1.0, np.random.default_rng(0))


class TestRunEnsemble:
    def test_the_same_seed_reproduces_the_run(self):
        assert np.array_equal(run_ensemble(**small_run()).lat, run_ensemble(**small_run()).lat)

    def test_the_same_seed_sequence_reproduces_the_run_although_spawn_mutates_it(self):
        seq = np.random.SeedSequence(7).spawn(1)[0]
        assert np.array_equal(run_ensemble(**small_run(seed=seq)).lat,
                              run_ensemble(**small_run(seed=seq)).lat)

    def test_two_particles_from_one_point_diverge(self):
        e = run_ensemble(**small_run(particles=2, datum_sigma_km=0.0, sigma=0.5))
        assert e.lat[0, 0] == e.lat[0, 1]
        assert e.lat[-1, 0] != e.lat[-1, 1]

    def test_one_particle_is_unchanged_by_the_others(self):
        """test_track's test_particles_do_not_interact, at ensemble scale."""
        e = run_ensemble(**small_run(particles=200, sigma=0.0))
        for k in (0, 57, 199):
            alone = list(DriftPipeline(EAST, 60.0).track(START, e.lat[0, k], e.lon[0, k], HOUR))
            assert alone[-1].positions[0] == pytest.approx([e.lat[-1, k], e.lon[-1, k]], abs=1e-12)

    def test_the_spread_grows_as_the_square_root_of_time(self):
        e = run_ensemble(**small_run(particles=20_000, datum_sigma_km=0.0, sigma=1.0,
                                     duration=4 * HOUR, save_every=HOUR, forcing=ConstantForcing()))
        north, _ = offsets_km(e.lat, e.lon)
        hours = np.arange(1, 5)
        expected = 1.0 * np.sqrt(hours * HOUR) / 1000.0
        assert north[1:].std(axis=1) == pytest.approx(expected, rel=0.03)

    def test_the_saved_states_are_every_save_every_and_the_end(self):
        e = run_ensemble(**small_run(duration=2100.0, save_every=900.0))
        offsets = (e.times - e.times[0]) / np.timedelta64(1, "s")
        assert offsets.tolist() == [0.0, 900.0, 1800.0, 2100.0]

    def test_the_weights_sum_to_one(self):
        assert run_ensemble(**small_run()).weight.sum() == pytest.approx(1.0)

    def test_the_forcing_log_holds_one_pair_per_particle_per_saved_state(self):
        log = {}
        run_ensemble(**small_run(), forcing_log=log)
        assert log["drift"].shape == (5, 10, 2)
        assert np.isnan(log["drift"][-1]).all()


class TestSpawnedStreams:
    """Spawning exists so that two runs of N/2 are worth one run of N."""

    N, REPEATS = 1000, 200

    def centroid(self, n, seed):
        e = run_ensemble(**small_run(particles=n, sigma=0.5, duration=600.0, save_every=None,
                                     seed=seed))
        return e.lat[-1], e.lon[-1]

    def scatter(self, pooled):
        north = np.array([offsets_km(*c)[0].mean() for c in pooled])
        return north.std(ddof=1)

    def test_two_spawned_halves_map_like_one_run(self):
        one = self.centroid(self.N, 11)
        halves = [self.centroid(self.N // 2, s) for s in np.random.SeedSequence(11).spawn(2)]
        pooled = np.concatenate([h[0] for h in halves]), np.concatenate([h[1] for h in halves])
        grid = ProbabilityGrid.from_envelope(np.r_[one[0], pooled[0]], np.r_[one[1], pooled[1]],
                                             cell_m=1000)
        table = np.array([grid.bin(*one)[0].ravel(), grid.bin(*pooled)[0].ravel()])
        assert chi2_contingency(table[:, table.sum(axis=0) >= 10])[1] > 0.01

    def test_spawned_halves_scatter_like_one_run_and_forked_halves_do_not(self):
        bases = np.random.SeedSequence(2026).spawn(self.REPEATS)
        single, spawned, forked = [], [], []
        for base in bases:
            single.append(self.centroid(self.N, base))
            a, b = (self.centroid(self.N // 2, s) for s in base.spawn(2))
            spawned.append((np.r_[a[0], b[0]], np.r_[a[1], b[1]]))
            # Seeding once and forking: both halves draw the same stream.
            c, d = self.centroid(self.N // 2, base), self.centroid(self.N // 2, base)
            forked.append((np.r_[c[0], d[0]], np.r_[c[1], d[1]]))
        reference = self.scatter(single)
        assert self.scatter(spawned) / reference == pytest.approx(1.0, abs=0.2)
        assert self.scatter(forked) / reference == pytest.approx(np.sqrt(2), abs=0.2)


class TestBeaching:
    def test_a_particle_with_no_forcing_is_frozen_flagged_and_keeps_its_mass(self):
        e = run_ensemble(**small_run(forcing=LandNorthOf(LAT + 0.01), particles=50, sigma=0.0,
                                     datum_sigma_km=1.0, duration=2 * HOUR))
        on_land = e.lat[0] > LAT + 0.01
        assert on_land.any() and (~on_land).any()
        assert e.beached[:, on_land].all()
        assert np.array_equal(e.lat[-1, on_land], e.lat[0, on_land])
        assert np.isfinite(e.lat).all() and np.isfinite(e.lon).all()

    def test_the_map_still_sums_to_one_with_beached_mass_included(self):
        e = run_ensemble(**small_run(forcing=LandNorthOf(LAT), particles=200, sigma=0.0))
        grid = ProbabilityGrid.from_envelope(e.lat[-1], e.lon[-1], cell_m=500)
        counts, lost = grid.bin(e.lat[-1], e.lon[-1])
        p = normalise(counts, beached_mass=int(e.beached[-1].sum()))
        assert e.beached[-1].any() and lost == 0
        assert p.sum() == pytest.approx(1.0) and e.weight.sum() == pytest.approx(1.0)


class TestSyntheticCloud:
    def test_one_instant_with_the_centre_and_spread_asked_for(self):
        e = synthetic_cloud((LAT, LON), 3.0, 10_000, np.random.default_rng(5), time=START)
        north, east = offsets_km(e.lat[0], e.lon[0])
        assert e.lat.shape == (1, 10_000)
        assert north.mean() == pytest.approx(0.0, abs=0.1)
        assert east.std() == pytest.approx(3.0, rel=0.03)


class TestSpans:
    @pytest.mark.parametrize("text, seconds", [("48h", 172800.0), ("15m", 900.0),
                                               ("90s", 90.0), ("3600", 3600.0)])
    def test_a_span_parses_to_seconds(self, text, seconds):
        assert parse_span(text) == seconds

    def test_a_span_that_is_not_one_raises(self):
        with pytest.raises(ValueError, match="not a span"):
            parse_span("two days")

    def test_format_is_the_inverse_of_parse(self):
        assert [format_span(parse_span(t)) for t in ("48h", "90m", "45s")] == ["48h", "90m", "45s"]

    def test_a_fractional_span_cannot_be_named(self):
        with pytest.raises(ValueError, match="whole number of seconds"):
            format_span(1.5)


class TestSaveStride:
    def test_none_saves_every_step(self):
        assert save_stride(None, 60.0) == 1

    def test_fifteen_minutes_is_fifteen_steps(self):
        assert save_stride(900.0, 60.0) == 15

    def test_a_ragged_interval_raises(self):
        with pytest.raises(ValueError, match="whole number"):
            save_stride(90.0, 60.0)


PARAMS = {"lat": 26.5, "lon": -79.0, "start": "2019-06-01T06:00", "particles": 10000,
          "timestep": 60.0, "duration": 172800.0, "seed": 20260923}


class TestRunName:
    def test_the_worked_example(self):
        assert run_name(PARAMS) == \
            "ensemble_2650N07900W_20190601T0600_N10000_dt60s_T48h_seed20260923"

    def test_the_name_round_trips(self):
        assert parse_run_name(run_name(PARAMS)) == PARAMS

    def test_a_run_without_a_seed_round_trips_as_none(self):
        params = {**PARAMS, "seed": None, "lat": -12.25, "lon": 45.5}
        assert run_name(params).endswith("seednone")
        assert parse_run_name(run_name(params)) == params

    def test_a_store_longitude_names_the_same_as_a_display_one(self):
        assert run_name({**PARAMS, "lon": 281.0}) == run_name(PARAMS)

    def test_the_sidecar_parameters_round_trip_through_the_name(self):
        run = describe_run(**small_run(seed=4))
        assert parse_run_name(run_name(run)) == {k: run[k] for k in ens.NAME_KEYS}

    def test_a_name_that_does_not_follow_the_convention_raises(self):
        with pytest.raises(ValueError, match="naming convention"):
            parse_run_name("era5_17-36N_82-63W_20210101-20210108.nc")


class TestEstimateRows:
    def test_forty_eight_hours_every_fifteen_minutes(self):
        assert estimate_rows(10_000, 172800.0, 60.0, 900.0) == 10_000 * 193

    def test_a_final_state_off_the_save_grid_is_counted(self):
        assert estimate_rows(10, 2100.0, 60.0, 900.0) == 40

    def test_a_run_over_the_threshold_is_refused(self):
        with pytest.raises(ValueError, match="threshold"):
            estimate_rows(10_000, 172800.0, 60.0, None)

    def test_force_overrides_the_threshold(self):
        assert estimate_rows(10_000, 172800.0, 60.0, None, force=True) == 10_000 * 2881


class TestWriteAndRead:
    def write(self, tmp_path, **overrides):
        spec = small_run(**overrides)
        return run_ensemble(**spec), write_csv(run_ensemble(**spec), describe_run(**spec), tmp_path)

    def test_positions_read_back_to_full_precision(self, tmp_path):
        e, path = self.write(tmp_path)
        back = read_csv(path)
        assert np.array_equal(back.lat, e.lat) and np.array_equal(back.lon, e.lon)
        assert np.array_equal(back.times, e.times)
        assert np.array_equal(back.beached, e.beached)

    def test_the_file_is_named_by_the_convention_under_derived(self, tmp_path):
        _, path = self.write(tmp_path)
        assert path.parent == tmp_path / "derived"
        assert parse_run_name(path)["particles"] == 10

    def test_the_sidecar_records_the_datum_the_count_and_the_seed(self, tmp_path):
        _, path = self.write(tmp_path)
        run = json.loads(path.with_suffix(".json").read_text())
        assert (run["datum_sigma_km"], run["particles"], run["seed"]) == (2.0, 10, 7)

    def test_the_columns_are_the_long_format(self, tmp_path):
        _, path = self.write(tmp_path)
        assert path.read_text().splitlines()[0] == "step,seconds,time,particle,lat,lon,beached"

    def test_the_forcing_columns_come_only_when_asked_for(self, tmp_path):
        spec, log = small_run(), {}
        path = write_csv(run_ensemble(**spec, forcing_log=log), describe_run(**spec), tmp_path,
                         forcing=log)
        assert path.read_text().splitlines()[0].endswith("drift_u,drift_v")

    def test_a_write_over_the_threshold_is_refused_and_forced(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ens, "MAX_BYTES", 100)
        with pytest.raises(ValueError, match="threshold"):
            self.write(tmp_path)
        assert not (tmp_path / "derived").exists()
        spec = small_run()
        assert write_csv(run_ensemble(**spec), describe_run(**spec), tmp_path, force=True).exists()

    def test_an_existing_file_is_not_overwritten_unless_forced(self, tmp_path):
        _, path = self.write(tmp_path)
        before = path.read_text()
        with pytest.raises(FileExistsError, match="already exists"):
            self.write(tmp_path, datum_sigma_km=5.0)
        assert path.read_text() == before
        spec = small_run(datum_sigma_km=5.0)
        write_csv(run_ensemble(**spec), describe_run(**spec), tmp_path, force=True)
        assert path.read_text() != before

    def test_a_cloud_with_no_time_cannot_be_written(self, tmp_path):
        cloud = synthetic_cloud((LAT, LON), 1.0, 5, np.random.default_rng(0))
        with pytest.raises(ValueError, match="real instant"):
            write_csv(cloud, {**PARAMS, "particles": 5}, tmp_path)

    def test_a_ragged_file_raises(self, tmp_path):
        _, path = self.write(tmp_path)
        path.write_text("\n".join(path.read_text().splitlines()[:-1]))
        with pytest.raises(ValueError, match="saved times"):
            read_csv(path)


class TestCli:
    ARGS = ("--lat", "26.5", "--lon", "-79", "--datum-sigma-km", "2", "--start", START,
            "--timestep", "60", "--duration", "1h", "--particles", "20", "--sigma", "0.1",
            "--constant-current", "1.8", "0", "--save-every", "15m")

    def test_help_runs(self, capsys):
        with pytest.raises(SystemExit) as exit_:
            ens._cli(["--help"])
        assert exit_.value.code == 0
        assert "--datum-sigma-km" in capsys.readouterr().out

    def test_a_small_run_writes_the_csv_and_sidecar_and_says_where(self, tmp_path, capsys):
        path = ens._cli([*self.ARGS, "--seed", "3", "--out", str(tmp_path)])
        assert path.exists() and path.with_suffix(".json").exists()
        assert str(path) in capsys.readouterr().out

    def test_a_run_without_a_seed_records_the_entropy_it_used(self, tmp_path):
        path = ens._cli([*self.ARGS, "--out", str(tmp_path)])
        run = json.loads(path.with_suffix(".json").read_text())
        assert run["seed"] is None and isinstance(run["seed_entropy"], int)
        again = run_ensemble(**small_run(particles=20, forcing=ConstantForcing((1.8, 0.0)),
                                         seed=np.random.SeedSequence(run["seed_entropy"])))
        assert np.array_equal(again.lat, read_csv(path).lat)

    @pytest.mark.parametrize("missing", ["--lat", "--datum-sigma-km", "--timestep", "--particles"])
    def test_what_defines_a_run_is_required(self, tmp_path, missing):
        args = [*self.ARGS, "--out", str(tmp_path)]
        i = args.index(missing)
        with pytest.raises(SystemExit):
            ens._cli(args[:i] + args[i + 2:])

    def test_a_second_unseeded_run_does_not_replace_the_first(self, tmp_path, capsys):
        path = ens._cli([*self.ARGS, "--out", str(tmp_path)])
        before = path.read_text()
        with pytest.raises(SystemExit) as exit_:
            ens._cli([*self.ARGS, "--out", str(tmp_path)])
        assert exit_.value.code == 2 and "already exists" in capsys.readouterr().err
        assert path.read_text() == before

    @pytest.mark.parametrize("bad, message", [(("--particles", "0"), "positive integer"),
                                              (("--save-every", "90s"), "whole number"),
                                              (("--datum-sigma-km", "-1"), "non-negative")])
    def test_bad_input_is_a_usage_error_not_a_traceback(self, tmp_path, capsys, bad, message):
        with pytest.raises(SystemExit) as exit_:
            ens._cli([*self.ARGS, *bad, "--out", str(tmp_path)])
        assert exit_.value.code == 2 and message in capsys.readouterr().err

    def test_out_is_required(self):
        with pytest.raises(SystemExit):
            ens._cli(list(self.ARGS))
