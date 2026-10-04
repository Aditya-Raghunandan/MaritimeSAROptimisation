"""Tests for sar.search.episode. Hand-computed geometry and synthetic clouds, no files."""

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

import sar.search
from sar.pipeline.ensemble import Ensemble
from sar.search.episode import (
    STEPS,
    SearchEpisode,
    Waypoints,
    _cli,
    heading_policy,
    pattern_policy,
    replay_policy,
    run,
)
from sar.search.patterns import MarkerTrack, expanding_square, parallel_track
from sar.search.platform import SEARCH_SPEED_MS, STEP_S, SWEEP_WIDTH_M
from sar.search.scenario import steady_search
from sar.search.sweep import relative_m
from sar.utils.geo import offset_position

LAT0, LON0 = 26.5, 281.0
HALF = SWEEP_WIDTH_M / 2.0            # 92.6 m
LEG = SEARCH_SPEED_MS * STEP_S        # 2,778 m, one step at 90 kt
S = SWEEP_WIDTH_M


def window_of(east, north, velocity=(0.0, 0.0), steps=STEPS, weight=None):
    """Particles at (east, north) metres from the reference, moving at `velocity` m/s."""
    east, north = np.atleast_1d(np.asarray(east, float)), np.atleast_1d(np.asarray(north, float))
    t = np.arange(steps + 1) * STEP_S
    lat, lon = offset_position(LAT0, LON0, east[None, :] + velocity[0] * t[:, None],
                               north[None, :] + velocity[1] * t[:, None])
    times = np.datetime64("2019-06-01T06:00", "us") + (t * 1e6).astype("timedelta64[us]")
    n = east.size
    w = np.full(n, 1.0 / n) if weight is None else np.asarray(weight, float)
    return Ensemble(times, lat, lon, w, np.zeros(lat.shape, dtype=bool))


def still_marker(east=0.0, north=0.0, steps=STEPS):
    lat, lon = offset_position(LAT0, LON0, east, north)
    return MarkerTrack.fixed(float(lat), float(lon), steps * STEP_S)


def episode_of(east, north, velocity=(0.0, 0.0), marker=None, **kw):
    steps = kw.pop("steps", STEPS)
    marker = marker or still_marker(steps=steps)
    return SearchEpisode(window_of(east, north, velocity, steps), marker, steps=steps, **kw)


def offset_now(ep):
    return relative_m(*ep.position, LAT0, LON0)


class TestTheEpisode:
    def test_an_episode_is_45_steps_of_60_s(self):
        ep = episode_of([5e4], [0.0])
        m = run(heading_policy(90.0), ep)
        assert STEPS == 45
        assert ep.done and ep.k == 45 and ep.t_s == 45 * 60.0
        assert m["steps"] == 45 and m["elapsed_s"] == 2700.0
        assert len(m["removed_per_step"]) == 45

    def test_a_step_after_done_raises(self):
        ep = episode_of([5e4], [0.0], steps=2)
        ep.step(0.0)
        ep.step(0.0)
        with pytest.raises(RuntimeError, match="over"):
            ep.step(0.0)

    def test_each_step_moves_speed_times_60_s_along_the_heading(self):
        ep = episode_of([5e4], [0.0])
        ep.step(90.0)
        assert offset_now(ep) == pytest.approx((LEG, 0.0), abs=1e-6)
        ep.step(0.0)
        assert offset_now(ep) == pytest.approx((LEG, LEG), abs=1e-6)

    def test_each_step_sweeps_its_segment(self):
        # On the leg, just inside the strip, just outside it, and past the end's cap.
        ep = episode_of([1000.0, 1000.0, 1000.0, LEG + HALF + 1.0], [0.0, 90.0, 95.0, 0.0])
        removed = ep.step(90.0)
        assert removed == pytest.approx(0.5)
        assert list(ep.weight == 0.0) == [True, True, False, False]

    def test_pos_is_the_sum_of_what_each_step_removed_and_never_exceeds_the_mass(self):
        setup = steady_search((LAT0, LON0), 2.0, 5000, np.random.default_rng(3))
        ep = SearchEpisode(setup.window, setup.marker)
        m = run(pattern_policy(expanding_square(S, 0.0)), ep)
        assert m["pos"] == pytest.approx(sum(m["removed_per_step"]), abs=1e-12)
        assert m["pos"] == pytest.approx(m["initial_mass"] - m["remaining"], abs=1e-12)
        assert 0.0 < m["pos"] <= m["initial_mass"] == pytest.approx(1.0)

    def test_hovering_over_a_still_cloud_removes_the_disc_once(self):
        setup = steady_search((LAT0, LON0), 2.0, 20_000, np.random.default_rng(4))
        ep = SearchEpisode(setup.window, setup.marker, speed_ms=0.0)
        m = run(heading_policy(0.0), ep)
        east, north = relative_m(setup.window.lat[0], setup.window.lon[0], *ep.position)
        disc = float(np.sum(setup.window.weight[np.hypot(east, north) <= HALF]))
        assert disc > 0.0
        assert m["removed_per_step"][0] == pytest.approx(disc, abs=1e-12)
        assert m["removed_per_step"][1:] == [0.0] * 44
        assert m["distance_m"] == 0.0

    def test_distance_is_speed_times_elapsed(self):
        ep = episode_of([5e4], [0.0])
        m = run(heading_policy(45.0), ep)
        assert m["distance_m"] == pytest.approx(SEARCH_SPEED_MS * 2700.0)
        ep = episode_of([5e4], [0.0])
        m = run(pattern_policy(expanding_square(S, 0.0)), ep)
        assert m["distance_m"] == pytest.approx(SEARCH_SPEED_MS * 2700.0)

    def test_the_cloud_it_was_given_is_left_alone(self):
        window = window_of([1000.0], [0.0])
        ep = SearchEpisode(window, still_marker())
        ep.step(90.0)
        assert window.weight[0] == 1.0 and ep.weight[0] == 0.0
        with pytest.raises(ValueError):
            ep.weight[0] = 1.0


class TestNothingImportsGymnasium:
    def test_no_module_in_sar_search_imports_an_rl_library(self):
        for path in Path(sar.search.__file__).parent.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    assert name.split(".")[0] not in {"gymnasium", "gym", "stable_baselines3"}, \
                        f"{path.name} imports {name}"


class TestClosestApproach:
    # The helicopter flies east from the origin; it passes x = 2500 m at 0.9 of the step.
    def test_it_finds_a_particle_the_start_of_step_rule_misses(self):
        # 150 m south at the start, moving north 1.8 m/s: 52.8 m south when overflown.
        ep = episode_of([2500.0], [-150.0], velocity=(0.0, 1.8))
        assert ep.step(90.0) == pytest.approx(1.0)

    def test_it_does_not_count_a_particle_that_moved_away(self):
        # 50 m south at the start, so a start-of-step check counts it; it is 147 m south
        # by the time the helicopter arrives.
        ep = episode_of([2500.0], [-50.0], velocity=(0.0, -1.8))
        assert ep.step(90.0) == 0.0


class TestPatterns:
    def test_a_4_s_leg_is_swept_along_the_waypoints_not_the_chord(self):
        # The Expanding Square's corner (-S, 2S) is flown in the first minute, 414 m from
        # the chord between where the helicopter is at 0 s and at 60 s.
        pattern = expanding_square(S, 0.0)
        assert pattern.t_s[1] == pytest.approx(4.0, abs=0.01)
        flown = episode_of([-S], [2 * S])
        assert pattern_policy(pattern)(flown).t_s.size > 5
        assert flown.fly(pattern_policy(pattern)(flown)) == pytest.approx(1.0)

        chord = episode_of([-S], [2 * S])
        e60, n60 = pattern.offset_at(60.0)
        assert chord.fly(Waypoints([60.0], [e60], [n60])) == 0.0

    def test_the_marker_frame_carries_a_pattern_with_a_uniform_current(self):
        still = steady_search((LAT0, LON0), 2.0, 4000, np.random.default_rng(5))
        moving = steady_search((LAT0, LON0), 2.0, 4000, np.random.default_rng(5), (1.8, 0.0))
        policy = pattern_policy(expanding_square(S, 30.0))
        a = run(policy, SearchEpisode(still.window, still.marker))
        b = run(policy, SearchEpisode(moving.window, moving.marker))
        assert a["pos"] > 0.3
        # Equal but for the flat earth: carried 4.86 km east, a particle 2 km off the
        # marker's latitude sits under a metre from where the still case has it, which can
        # tip a particle on the strip's edge. Two particles in 4,000 at most.
        assert b["pos"] == pytest.approx(a["pos"], abs=2 / 4000)

    def test_a_pattern_that_ends_early_is_refused(self):
        short = parallel_track(length_m=1000.0, width_m=1000.0)
        assert short.duration_s < 2700.0
        with pytest.raises(ValueError, match="ends at"):
            run(pattern_policy(short), episode_of([5e4], [0.0]))


class TestMetrics:
    def test_expected_time_to_detection_is_mass_weighted(self):
        # One particle swept in step 1 (credited at 60 s), one in step 2 (120 s), one never.
        ep = episode_of([1000.0, LEG + 1000.0, -5000.0], [0.0, 0.0, 0.0])
        m = run(heading_policy(90.0), ep)
        assert m["pos"] == pytest.approx(2 / 3)
        assert m["expected_ttd_s"] == pytest.approx((60 + 120) / 2)

    def test_time_to_detection_is_none_when_nothing_is_found(self):
        m = run(heading_policy(90.0), episode_of([0.0], [-5000.0]))
        assert m["pos"] == 0.0 and m["expected_ttd_s"] is None

    def test_the_worked_example_one_leg_through_a_2_km_cloud(self):
        # ADR004 section 2: a 2,778 m leg through the centre of a 2 km cloud clears 1.89 %.
        n = 200_000
        setup = steady_search((LAT0, LON0), 2.0, n, np.random.default_rng(6))
        lat, lon = offset_position(*setup.datum, -LEG / 2, 0.0)
        ep = SearchEpisode(setup.window, MarkerTrack.fixed(float(lat), float(lon), 2700.0))
        expected = ((norm.cdf(HALF / 2000) - norm.cdf(-HALF / 2000))
                    * (norm.cdf(LEG / 2 / 2000) - norm.cdf(-LEG / 2 / 2000)))
        assert expected == pytest.approx(0.0189, abs=1e-4)
        noise = math.sqrt(expected * (1 - expected) / n)
        # The capsule's end caps add ~0.0001 to the strip.
        assert ep.step(90.0) == pytest.approx(expected, abs=4 * noise + 2e-4)

    def test_pod_below_one_never_removes_more_than_there_was(self):
        # Back and forth over 0-600 m: every pass sweeps the first particle, never the second.
        ep = episode_of([300.0, 300.0], [0.0, 3000.0], pod=0.5, speed_ms=10.0, steps=20)
        m = run(lambda e: 90.0 if e.k % 2 == 0 else 270.0, ep)
        assert ep.weight[0] == pytest.approx(0.5 * 0.5 ** 20)
        assert m["pos"] == pytest.approx(0.5 - ep.weight[0], abs=1e-12)
        assert m["pos"] <= m["initial_mass"]


class TestTheTarget:
    def test_it_is_found_at_the_hand_computed_crossing(self):
        # A buoy 1 km east and 50 m north; the gap first closes to 92.6 m at x = 922.1 m.
        lat, lon = offset_position(LAT0, LON0, 1000.0, 50.0)
        target = MarkerTrack.fixed(float(lat), float(lon), 2700.0)
        ep = episode_of([5e4], [0.0], target=target)
        m = run(heading_policy(90.0), ep)
        x = 1000.0 - math.sqrt(HALF ** 2 - 50.0 ** 2)
        assert m["target"]["found"] is True
        assert m["target"]["found_s"] == pytest.approx(x / SEARCH_SPEED_MS, abs=1e-4)
        assert m["target"]["closest_m"] == pytest.approx(50.0, abs=1e-6)

    def test_a_miss_reports_how_close_it_came(self):
        lat, lon = offset_position(LAT0, LON0, 1000.0, 300.0)
        target = MarkerTrack.fixed(float(lat), float(lon), 2700.0)
        m = run(heading_policy(90.0), episode_of([5e4], [0.0], target=target))
        assert m["target"]["found"] is False and m["target"]["found_s"] is None
        assert m["target"]["closest_m"] == pytest.approx(300.0, abs=1e-6)
        assert m["target"]["closest_s"] == pytest.approx(1000.0 / SEARCH_SPEED_MS, abs=1e-6)

    def test_time_outside_its_track_is_a_gap(self):
        lat, lon = offset_position(LAT0, LON0, 1000.0, 0.0)
        target = MarkerTrack.fixed(float(lat), float(lon), 120.0)
        m = run(heading_policy(90.0), episode_of([5e4], [0.0], target=target))
        assert m["target"]["gaps"] == 43


class TestSearchers:
    def test_decide_every_holds_a_heading(self):
        calls = []
        m = run(lambda e: calls.append(e.k) or 90.0, episode_of([5e4], [0.0]), decide_every=5)
        assert calls == list(range(0, 45, 5))
        assert m["steps"] == 45

    def test_decide_every_does_not_apply_to_waypoints(self):
        with pytest.raises(ValueError, match="decide_every"):
            run(pattern_policy(expanding_square(S, 0.0)), episode_of([5e4], [0.0]),
                decide_every=5)

    def test_a_replay_reproduces_a_heading_run(self):
        headings = list(np.random.default_rng(7).uniform(0.0, 360.0, STEPS))
        setup = steady_search((LAT0, LON0), 2.0, 3000, np.random.default_rng(8))
        a = run(lambda e: headings[e.k], SearchEpisode(setup.window, setup.marker))
        b = run(replay_policy({"headings_deg": headings}),
                SearchEpisode(setup.window, setup.marker))
        c = run(replay_policy({"t_s": [60.0 * k for k in range(STEPS)], "heading_deg": headings}),
                SearchEpisode(setup.window, setup.marker))
        assert b == a
        assert c["pos"] == pytest.approx(a["pos"], abs=1e-12)

    def test_a_timed_replay_turns_mid_step(self):
        ep = episode_of([5e4], [0.0])
        ep.fly(replay_policy({"t_s": [0.0, 30.0], "heading_deg": [90.0, 0.0]})(ep))
        assert offset_now(ep) == pytest.approx((LEG / 2, LEG / 2), abs=1e-6)

    def test_a_replay_is_read_from_a_file(self, tmp_path):
        path = tmp_path / "flight.json"
        path.write_text(json.dumps({"headings_deg": [90.0] * STEPS}))
        assert run(replay_policy(path), episode_of([1000.0], [0.0]))["pos"] == 1.0

    def test_a_short_record_raises(self):
        with pytest.raises(ValueError, match="headings"):
            run(replay_policy({"headings_deg": [90.0]}), episode_of([5e4], [0.0]))


class TestRefusals:
    def test_a_path_faster_than_the_helicopter(self):
        ep = episode_of([5e4], [0.0])
        with pytest.raises(ValueError, match="faster"):
            ep.fly(Waypoints([60.0], [LEG * 1.01], [0.0]))

    def test_waypoints_must_end_at_the_end_of_the_step(self):
        with pytest.raises(ValueError, match="end of the step"):
            episode_of([5e4], [0.0]).fly(Waypoints([30.0], [0.0], [0.0]))

    def test_waypoint_times_must_increase(self):
        with pytest.raises(ValueError, match="increasing"):
            Waypoints([30.0, 30.0, 60.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    def test_the_window_must_have_steps_plus_one_frames(self):
        with pytest.raises(ValueError, match="46 frames"):
            SearchEpisode(window_of([0.0], [0.0], steps=44), still_marker())

    def test_the_frames_must_be_60_s_apart(self):
        w = window_of([0.0], [0.0])
        bad = Ensemble(w.times + np.arange(46).astype("timedelta64[s]"), w.lat, w.lon,
                       w.weight, w.beached)
        with pytest.raises(ValueError, match="60 s apart"):
            SearchEpisode(bad, still_marker())

    def test_the_marker_must_cover_the_episode(self):
        with pytest.raises(ValueError, match="marker"):
            SearchEpisode(window_of([0.0], [0.0]), still_marker(steps=10))

    @pytest.mark.parametrize("kw", [{"speed_ms": -1.0}, {"sweep_width_m": 0.0},
                                    {"pod": 1.5}, {"steps": 0}])
    def test_bad_settings(self, kw):
        with pytest.raises(ValueError):
            episode_of([0.0], [0.0], **kw)

    def test_a_missing_heading(self):
        with pytest.raises(ValueError, match="heading"):
            episode_of([0.0], [0.0]).step(None)


class TestCli:
    def test_a_synthetic_expanding_square(self):
        out = _cli(["--spread-km", "2", "--particles", "2000", "--seed", "1"])
        assert out["policy"] == "expanding-square"
        assert 0.3 < out["pos"] < 0.9
        assert out["distance_m"] == pytest.approx(SEARCH_SPEED_MS * 2700.0)
        json.dumps(out)

    def test_the_first_leg_follows_the_current(self):
        out = _cli(["--particles", "500", "--seed", "1", "--current", "0", "1.8"])
        assert out["first_bearing_deg"] == pytest.approx(0.0)
        out = _cli(["--particles", "500", "--seed", "1", "--current", "1.8", "0"])
        assert out["first_bearing_deg"] == pytest.approx(90.0)

    def test_a_scenario_needs_all_three_flags(self):
        with pytest.raises(SystemExit):
            _cli(["--csv", "scenarios.csv"])
