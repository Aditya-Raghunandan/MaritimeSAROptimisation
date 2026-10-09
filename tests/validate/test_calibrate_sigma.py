"""Tests for sar.validate.calibrate_sigma (#89). Steady analytic fields only.

The buoys here are made by DriftPipeline itself, with a known sigma, so the model is
exactly right and the method must give the planted sigma back. Then the cases the issue
names: a persistent current error must show as beta near 2, a crosswind slide must show
as more crosswind than downwind spread, and a buoy that feels less wind must show as the
model running ahead downwind by exactly the difference.
"""

import numpy as np
import pandas as pd
import pytest

from sar.pipeline.forcing import ConstantForcing
from sar.pipeline.track import DriftPipeline
from sar.validate import calibrate_sigma as cs
from sar.validate.drift_windows import Windows

HOURS = 12
CURRENT, WIND = (0.3, 0.1), (6.0, 0.0)


def synthetic(days=12, per_day=40, sigma=50.0, buoy_current=CURRENT, leeway=0.02, slide=0.0,
              seed=0):
    """Windows whose buoys are engine particles from a known model."""
    rng = np.random.default_rng(seed)
    rows, lat, lon = [], [], []
    for d in range(days):
        t0 = np.datetime64("2021-01-01T00:00", "ns") + np.timedelta64(2 * d, "D")
        lat0 = 25.0 + rng.uniform(-1, 1, per_day)
        lon0 = 285.0 + rng.uniform(-1, 1, per_day)
        forcing = ConstantForcing(buoy_current, WIND)
        if slide:
            forcing = cs.SlidingForcing(forcing, slide, rng.choice([-1.0, 1.0], per_day))
        pipe = DriftPipeline(forcing, 60.0, leeway=leeway, sigma=sigma, seed=rng)
        states = list(pipe.track(t0, lat0, lon0, HOURS * 3600.0))
        pos = np.stack([states[h * 60].positions for h in range(HOURS + 1)], axis=1)
        lat.append(pos[..., 0])
        lon.append(pos[..., 1])
        for k in range(per_day):
            rows.append({"window": f"{d}-{k}", "unit": f"u{d}", "ID": f"b{k}",
                         "group": (d * per_day + k) // 4, "tier": "undrogued", "t0": t0,
                         "month": "2021-01", "lat0": lat0[k], "lon0": lon0[k],
                         "ve0": buoy_current[0] + leeway * WIND[0],
                         "vn0": buoy_current[1] + leeway * WIND[1]})
    lat, lon = np.concatenate(lat), np.concatenate(lon)
    return Windows(pd.DataFrame(rows), lat, lon, np.ones_like(lat, bool), HOURS)


def stage1_rows(w, alphas=(0.02,)):
    rows, status = cs.stage1(w, cs.fixed(ConstantForcing(CURRENT, WIND)),
                             list(enumerate(w.batches())), alphas=alphas)
    tab = w.table
    rows["group"] = tab["group"].to_numpy()[rows["w"].to_numpy()]
    rows["month"] = tab["month"].to_numpy()[rows["w"].to_numpy()]
    return rows, status


def at(rows, hour, alpha=0.02):
    return rows[(rows["hour"] == hour) & np.isclose(rows["alpha"], alpha) & rows["valid"]]


@pytest.fixture(scope="module")
def planted():
    return stage1_rows(synthetic())[0]


class TestStage1:
    def test_the_gap_at_t0_is_zero(self, planted):
        assert np.all(at(planted, 0)["sep"] == 0.0)

    def test_every_window_ran(self):
        _, status = stage1_rows(synthetic(days=2, per_day=3))
        assert (status["status"] == "ok").all() and len(status) == 6

    def test_a_planted_sigma_comes_back_by_every_estimator(self, planted):
        p = at(planted, HOURS)
        s = cs.horizon_stats(p, HOURS, np.ones(len(p)))
        for key in ("sigma_quantile", "sigma_raw", "sigma_debiased"):
            assert s[key] == pytest.approx(50.0, rel=0.12), key

    def test_a_random_walk_grows_with_exponent_one(self, planted):
        labels = {"group": planted.groupby("w")["group"].first()}
        b = cs.beta_fit(planted, labels, hours=(2, HOURS), n_boot=200)
        assert b["value"] == pytest.approx(1.0, abs=0.15) and b["random_walk_consistent"]

    def test_a_persistent_current_error_grows_with_exponent_two_and_is_flagged(self):
        rows, _ = stage1_rows(synthetic(days=4, per_day=10, sigma=0.0,
                                        buoy_current=(0.4, 0.1)))
        labels = {"group": rows.groupby("w")["group"].first()}
        b = cs.beta_fit(rows, labels, hours=(2, HOURS), n_boot=200)
        assert b["value"] == pytest.approx(2.0, abs=0.05) and not b["random_walk_consistent"]

    def test_a_crosswind_slide_spreads_more_across_the_wind_than_along_it(self):
        rows, _ = stage1_rows(synthetic(days=6, per_day=40, sigma=5.0, slide=0.0051))
        p = at(rows, HOURS)
        s = cs.horizon_stats(p, HOURS, np.ones(len(p)))
        assert s["sigma_crosswind"] > 1.5 * s["sigma_downwind"]
        # Crosswind is to the right of the wind run: the wind blows east, so right is south.
        assert np.allclose(p["gap_cw"], -p["gap_n"], atol=1e-3)

    def test_a_buoy_that_feels_half_the_wind_shows_the_model_one_percent_ahead(self):
        rows, _ = stage1_rows(synthetic(days=2, per_day=5, sigma=0.0, leeway=0.01))
        p = at(rows, HOURS)
        s = cs.horizon_stats(p, HOURS, np.ones(len(p)))
        assert s["windage_minus_model_pct"] == pytest.approx(-1.0, abs=1e-3)
        lead = 0.01 * 6.0 * HOURS * 3600.0 / 1e3
        assert s["model_lead_downwind_km"] == pytest.approx(lead, rel=1e-3)

    def test_the_alpha_off_run_is_kept_apart(self):
        rows, _ = stage1_rows(synthetic(days=1, per_day=3, sigma=0.0), alphas=(0.02, 0.0))
        assert sorted(np.unique(rows["alpha"]).round(2)) == [0.0, 0.02]

    def test_an_empty_table_or_a_non_positive_hour_raises(self, planted):
        with pytest.raises(ValueError, match="empty"):
            cs.horizon_stats(planted.iloc[:0], HOURS, np.ones(0))
        with pytest.raises(ValueError, match="positive"):
            cs.horizon_stats(at(planted, HOURS), 0, np.ones(1))


class TestCoverage:
    def test_gaussian_gaps_at_the_true_sigma_cover_ninety_percent(self):
        rng = np.random.default_rng(1)
        t = 24 * 3600.0
        gaps = 40.0 * np.sqrt(t) * rng.standard_normal((20_000, 2))
        cover = cs.coverage_analytic(np.hypot(*gaps.T), 40.0, 24)
        assert cover == pytest.approx(0.9, abs=0.007)    # 3 standard errors at n = 20,000

    def test_the_crossing_is_found_in_log_sigma(self):
        assert cs.crossing([10, 100], [0.8, 1.0], 0.9) == pytest.approx(np.sqrt(1000))
        assert np.isnan(cs.crossing([10, 100], [0.95, 0.99], 0.9))

    def test_the_energy_optimum_of_a_parabola_in_log_sigma(self):
        s = np.array([25.0, 35.0, 50.0, 70.0, 100.0])
        assert cs.parabola_minimum(s, (np.log(s) - np.log(44.0)) ** 2) == pytest.approx(44.0)

    def test_the_ladder_brackets_a_planted_sigma(self):
        w = synthetic(days=3, per_day=20, sigma=50.0, seed=2)
        rows = cs.ladder(w, cs.fixed(ConstantForcing(CURRENT, WIND)),
                         list(enumerate(w.batches())), [25.0, 50.0, 100.0], 200, seed=3,
                         leads=(HOURS,))
        cover = rows.groupby("sigma")["rank"].apply(lambda r: np.mean(r <= 0.9))
        assert cover[25.0] < cover[50.0] < cover[100.0]
        assert cover[50.0] == pytest.approx(0.9, abs=0.1)


class TestClusters:
    def test_windows_of_one_group_share_a_weight(self):
        w = cs.boot_weights(["a", "a", "b", "c"], n_boot=50)
        assert w.shape == (50, 4) and np.array_equal(w[:, 0], w[:, 1])
        # each resample draws three groups, so the three groups' counts add to three
        assert np.all(w[:, 0] + w[:, 2] + w[:, 3] == 3)

    def test_the_gate_counts_groups_not_windows(self):
        p = pd.DataFrame({"sep": [1.0, 1.0, 1.0, 9.0], "sep_still": [5.0] * 4,
                          "sep_persist": [5.0] * 4, "group": [1, 1, 1, 2]})
        g = cs.gate(p)
        assert g["stationary"]["groups"] == 2 and g["stationary"]["model_better_groups"] == 1


class TestHorizon:
    """--hour (D028's amendment, 4 Oct): sigma matched at a horizon other than 24 h."""

    @pytest.fixture(scope="class")
    def fitted(self, tmp_path_factory):
        w = synthetic(days=4, per_day=20, sigma=50.0, seed=5)
        w.table.loc[w.table.index % 2 == 0, "tier"] = "drogued"
        rows, status = cs.stage1(w, cs.fixed(ConstantForcing(CURRENT, WIND)),
                                 list(enumerate(w.batches())))
        d = tmp_path_factory.mktemp("stage1")
        rows.to_parquet(d / "rows-000.parquet", index=False)
        status.to_parquet(d / "status-000.parquet", index=False)
        return w, d, cs.fit(w, d, n_boot=50, hour=4, horizons=(2, 12), beta_hours=(1, 6))

    def test_the_fit_records_its_hour_and_tables_it_with_the_others(self, fitted):
        _, _, out = fitted
        assert out["hour"] == 4
        assert sorted(out["table_a"]["undrogued, alpha=0.02"]) == [2, 4, 12]

    def test_sigma_0_and_the_gate_are_read_at_that_hour(self, fitted):
        w, d, out = fitted
        rows, _ = cs.load_stage1(w, d)
        p = rows[rows["valid"] & (rows["hour"] == 4) & (rows["tier"] == "undrogued")
                 & np.isclose(rows["alpha"], 0.02)]
        gate = out["gate"]["undrogued, alpha=0.02"]["stationary"]
        assert gate["median_model_km"] == pytest.approx(p["sep"].median() / 1e3)
        head = out["table_a"]["undrogued, alpha=0.02"][4]
        assert out["sigma_0"] == head["sigma_quantile"]["value"]

    def test_the_slide_is_matched_at_that_hour_and_grows_as_root_t(self, fitted):
        # A steady 6 m/s wind runs 6 t metres, so sigma_c = 0.0051 x 6 x sqrt(t).
        _, _, out = fitted
        expected = cs.CROSSWIND_SLIDE * WIND[0] * np.sqrt(4 * 3600.0)
        assert out["sigma_c"]["crosswind_axis"]["value"] == pytest.approx(expected, rel=1e-3)

    def test_calibrate_refuses_a_fit_made_at_another_hour(self, fitted, tmp_path):
        w, _, _ = fitted
        w.save(tmp_path / "windows")
        (tmp_path / "fit.json").write_text('{"sigma_0": 50, "sigma_c": {}}')   # no hour: 24 h
        with pytest.raises(SystemExit, match="fitted at 24 h"):
            cs.main(["calibrate", "--windows", str(tmp_path / "windows"),
                     "--fit", str(tmp_path / "fit.json"), "--hour", "4",
                     "--out", str(tmp_path / "calibration.json")])

    def test_sigma_star_reads_the_ladder_at_that_lead_only(self):
        w = synthetic(days=1, per_day=10, sigma=0.0)
        inside_4 = {10.0: [0.5] * 5 + [0.95] * 5, 40.0: [0.5] * 10, 160.0: [0.5] * 10}
        inside_24 = {10.0: [0.95] * 10, 40.0: [0.5] * 5 + [0.95] * 5,  # never reaches 90 %
                     160.0: [0.5] * 7 + [0.95] * 3}
        rows = pd.DataFrame([{"w": k, "sigma": s, "lead": lead, "bandwidth": 1.0, "n": 100,
                              "rank": r, "energy_m": 1.0}
                             for lead, table in ((4, inside_4), (24, inside_24))
                             for s, ranks in table.items() for k, r in enumerate(ranks)])
        at_4 = cs.calibrate_sigma_star(rows, w, n_boot=20, hour=4)["sigma_star"]["value"]
        assert at_4 == pytest.approx(10.0 * 4.0 ** 0.8)      # 0.9 is 4/5 of the way, in log
        assert np.isnan(cs.calibrate_sigma_star(rows, w, n_boot=20, hour=24)["sigma_star"]["value"])


class TestRandomVelocityLadder:
    """D030: the same ladder over sigma_u, with T_L fixed."""

    def test_the_ladder_brackets_the_buoys_with_sigma_u(self):
        # Buoys from a random walk of 50 m/s^0.5: at 12 h, sigma sqrt(t) = 10.4 km per axis.
        # A random velocity with T_L = 1 h spreads sqrt(2 su^2 T_L (t - T_L (1 - e^-12))),
        # 16.9 km x su, so su = 0.615 m/s matches them.
        w = synthetic(days=3, per_day=20, sigma=50.0, seed=2)
        rows = cs.ladder(w, cs.fixed(ConstantForcing(CURRENT, WIND)),
                         list(enumerate(w.batches())), [0.2, 0.6, 1.8], 200, seed=3,
                         leads=(HOURS,), model=cs.RANDOM_VELOCITY, memory_time_s=3600.0)
        assert set(rows["model"]) == {cs.RANDOM_VELOCITY}
        assert set(rows["memory_h"]) == {1.0}
        cover = rows.groupby("sigma")["rank"].apply(lambda r: np.mean(r <= 0.9))
        assert cover[0.2] < cover[0.6] < cover[1.8]
        assert cover[0.6] == pytest.approx(0.9, abs=0.1)

    def test_an_unknown_model_is_refused(self):
        w = synthetic(days=1, per_day=2, sigma=0.0)
        with pytest.raises(ValueError, match="model must be"):
            cs.ladder(w, cs.fixed(ConstantForcing()), [], [1.0], 2, seed=0, model="flight")

    def test_old_ladder_rows_are_a_random_walk(self):
        assert cs.ladder_model(pd.DataFrame({"sigma": [1.0]})) == cs.RANDOM_WALK
        with pytest.raises(ValueError, match="one model"):
            cs.ladder_model(pd.DataFrame({"model": [cs.RANDOM_WALK, cs.RANDOM_VELOCITY]}))

    def test_the_slide_becomes_a_velocity(self):
        # sigma_c sqrt(T) = a_c |wind| T, so the slide's velocity a_c |wind| is sigma_c / sqrt(T).
        sigma_c = {"value": cs.CROSSWIND_SLIDE * 6.0 * np.sqrt(4 * 3600.0), "ci95": [1.0, 2.0]}
        v = cs.slide_for(sigma_c, cs.RANDOM_VELOCITY, 4)
        assert v["value"] == pytest.approx(cs.CROSSWIND_SLIDE * 6.0)
        assert v["ci95"] == pytest.approx([1.0 / 120.0, 2.0 / 120.0])
        assert cs.slide_for(sigma_c, cs.RANDOM_WALK, 4) is sigma_c

    def test_the_cli_takes_sigmas_u(self, tmp_path, monkeypatch):
        seen = {}

        def fake_ladder(w, open_forcing, batches, sigmas, particles, seed, leads, **kw):
            seen.update(sigmas=sigmas, **kw)
            return pd.DataFrame()

        w = synthetic(days=1, per_day=4, sigma=0.0)
        w.save(tmp_path / "windows")
        monkeypatch.setattr(cs, "ladder", fake_ladder)
        cs.main(["ladder", "--windows", str(tmp_path / "windows"), "--data", str(tmp_path),
                 "--sigmas-u", "0.2", "0.3", "--memory-h", "10", "--out", str(tmp_path / "o")])
        assert seen["sigmas"] == [0.2, 0.3] and seen["model"] == cs.RANDOM_VELOCITY
        assert seen["memory_time_s"] == 36000.0


class TestSigmaUByTheCurrent:
    """D033: sigma_u sized by HYCOM's current at the start, a straight line through bins."""

    @staticmethod
    def planted(a=0.17, b=0.25, n=3000, seed=4, rungs=(0.12, 0.15, 0.18, 0.2, 0.22, 0.24,
                                                      0.27, 0.32, 0.36, 0.42, 0.5, 0.6, 0.75,
                                                      0.9)):
        """Windows whose 90 % crossing is exactly a + b x speed, as ladder rows at 4 h."""
        rng = np.random.default_rng(seed)
        speed = rng.uniform(0.0, 1.4, n)
        # A window is inside at sigma >= tau; tau <= a + b s with probability 0.9.
        tau = (a + b * speed) * np.exp(0.25 * (rng.standard_normal(n) - 1.2815516))
        tab = pd.DataFrame({"group": np.arange(n) // 3, "tier": "undrogued",
                            "month": "2021-01"})
        w = Windows(tab, np.zeros((n, 5)), np.zeros((n, 5)), np.ones((n, 5), bool), 4)
        rows = pd.DataFrame([{"w": k, "sigma": s, "lead": 4, "n": 1000, "bandwidth": 1.0,
                              "rank": 0.5 if s >= tau[k] else 0.99, "area90_km2": 1.0,
                              "model": cs.RANDOM_VELOCITY}
                             for k in range(n) for s in rungs])
        return w, rows, speed

    def test_the_line_through_the_bins_is_the_planted_one(self):
        w, rows, speed = self.planted()
        out = cs.fit_speed_rule(rows, w, speed, hour=4, n_boot=60)
        line = out["line"]
        assert line["a_ci95"][0] < 0.17 < line["a_ci95"][1]
        assert line["b_ci95"][0] < 0.25 < line["b_ci95"][1]
        assert line["through_the_bins"]["b"] == pytest.approx(0.25, abs=0.02)
        used = [b for b in out["bins"] if b["used"]]
        assert len(used) == 6 and line["cap_speed_ms"] > 1.0
        # The objective is the confirmation's test: every bin near 90 % under the rule (two
        # numbers cannot put six bins at exactly 90 %).
        assert [b["coverage_under_rule"] for b in used] == pytest.approx([0.9] * 6, abs=0.02)

    def test_a_bin_with_too_few_groups_is_left_out_and_caps_the_rule(self):
        w, rows, speed = self.planted(n=1500)
        fast = np.flatnonzero(speed > 1.0)
        keep = np.setdiff1d(np.arange(len(speed)), fast[6:])     # two groups' worth above 1 m/s
        rows = rows[rows["w"].isin(keep)]
        out = cs.fit_speed_rule(rows, w, speed, hour=4, n_boot=20)
        last = out["bins"][-1]
        assert not last["used"] and last["groups"] < cs.MIN_GROUPS
        assert out["line"]["cap_speed_ms"] < 1.0

    def test_the_rule_adds_the_slide_and_never_extrapolates(self):
        rule = {"line": {"a": 0.17, "b": 0.25, "cap_speed_ms": 1.0}, "slide": {"value": 0.035}}
        got = cs.rule_sigma_u(rule, [0.0, 0.5, 1.0, 3.0])
        assert got[0] == pytest.approx(np.hypot(0.17, 0.035))
        assert got[1] == pytest.approx(np.hypot(0.295, 0.035))
        assert got[3] == got[2]                                     # capped
        assert np.all(np.diff(got[:3]) > 0)

    def test_speeds_come_from_stage1_and_bins_from_the_edges(self, tmp_path):
        pd.DataFrame({"w": [0, 1, 2], "status": ["ok", "ok", "FileNotFoundError"],
                      "current0_u": [0.3, 0.0, np.nan], "current0_v": [0.4, 1.2, np.nan]}
                     ).to_parquet(tmp_path / "status-000.parquet")
        speed = cs.start_speeds(tmp_path, 4)
        assert speed[:2] == pytest.approx([0.5, 1.2]) and np.isnan(speed[2:]).all()
        assert cs.speed_bin([0.1, 0.3, 0.74, 1.2, np.nan]).tolist() == [0, 2, 3, 5, -1]

    def test_a_window_at_its_own_sigma_u_matches_the_scalar_rung(self):
        w = synthetic(days=1, per_day=6, sigma=20.0, seed=5)
        batches = list(enumerate(w.batches()))
        forcing = cs.fixed(ConstantForcing(CURRENT, WIND))
        scalar = cs.ladder(w, forcing, batches, [0.3], 50, seed=7, leads=(4,),
                           model=cs.RANDOM_VELOCITY)
        each = cs.ladder(w, forcing, batches, [], 50, seed=7, leads=(4,),
                         model=cs.RANDOM_VELOCITY, sigma_u_per_window=np.full(6, 0.3))
        assert set(each["rung"]) == {"rule"} and np.allclose(each["sigma"], 0.3)
        assert each["rank"].tolist() == scalar["rank"].tolist()

    def test_the_cli_ladders_only_the_fast_windows(self, tmp_path, monkeypatch):
        seen = {}

        def fake_ladder(w, open_forcing, batches, sigmas, particles, seed, leads, **kw):
            seen.update(n=len(w.table), sigmas=sigmas)
            return pd.DataFrame()

        w = synthetic(days=1, per_day=4, sigma=0.0)
        w.save(tmp_path / "windows")
        pd.DataFrame({"w": [0, 1, 2, 3], "status": "ok", "current0_u": [0.1, 0.6, 0.9, 0.2],
                      "current0_v": 0.0}).to_parquet(tmp_path / "status-000.parquet")
        monkeypatch.setattr(cs, "ladder", fake_ladder)
        cs.main(["ladder", "--windows", str(tmp_path / "windows"), "--data", str(tmp_path),
                 "--sigmas-u", "0.4", "--stage1", str(tmp_path), "--min-start-speed", "0.5",
                 "--out", str(tmp_path / "o")])
        assert seen == {"n": 2, "sigmas": [0.4]}
