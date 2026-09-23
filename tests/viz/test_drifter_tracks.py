"""Tests for the drifter track export the site reads (issue #50). Synthetic, no network.

The acceptance criterion that matters most: every fix the site draws is a fix in the
data, at the same time and position. `TestTrackRecord::test_every_fix_round_trips`.
"""

import json

import numpy as np
import pandas as pd
import pytest

from sar.viz import drifters as dv


def loaded(bid: str, start: str, hours: int, *, lost: str | None = None, seg_break: int | None = None,
           far_at: tuple[int, ...] = (), uncertain: bool = False, product: str = "hourly") -> pd.DataFrame:
    """One buoy's fixes in the shape `load_drifters` returns: 0-360 longitude, UTC times."""
    t = pd.date_range(start, periods=hours + 1, freq="h", tz="UTC")
    seg = np.zeros(len(t), dtype=int) + 10
    if seg_break is not None:
        seg[seg_break:] = 11
    gap = np.ones(len(t))
    gap[list(far_at)] = 5.0
    lost_ts = pd.Timestamp(lost, tz="UTC") if lost else None
    return pd.DataFrame({
        "ID": bid, "time": t,
        "lat": np.linspace(28.0, 28.5, len(t)), "lon": np.linspace(285.0, 285.5, len(t)),
        "segment_id": seg,
        "undrogued": (t >= lost_ts) if lost_ts is not None else False,
        "tier_uncertain": uncertain, "fix_gap_h": gap, "product": product,
    })


class TestTrackRecord:
    def test_every_fix_round_trips(self):
        """Time from t0 + h, position to 4 decimal places: the same fix as the data."""
        g = loaded("A", "2021-06-01 05:00", 30)
        rec = dv.track_record(g)
        times = pd.Timestamp(rec["t0"]) + pd.to_timedelta(rec["h"], unit="h")
        assert (times == g["time"].reset_index(drop=True)).all()
        assert np.allclose(rec["lat"], g["lat"], atol=1e-4)
        assert np.allclose(rec["lon"], g["lon"] - 360.0, atol=1e-4)

    def test_hours_are_whole_numbers_from_the_first_fix(self):
        rec = dv.track_record(loaded("A", "2021-06-01", 5))
        assert rec["h"] == [0, 1, 2, 3, 4, 5]
        assert rec["t0"] == "2021-06-01T00:00:00Z"

    def test_input_order_does_not_matter(self):
        g = loaded("A", "2021-06-01", 10)
        assert dv.track_record(g.sample(frac=1.0, random_state=1)) == dv.track_record(g)

    def test_segments_are_renumbered_from_zero(self):
        rec = dv.track_record(loaded("A", "2021-06-01", 6, seg_break=3))
        assert rec["seg"] == [0, 0, 0, 1, 1, 1, 1]

    def test_drogue_state_and_far_from_fix_are_per_fix(self):
        rec = dv.track_record(loaded("A", "2021-06-01", 4, lost="2021-06-01 02:00", far_at=(3,)))
        assert rec["und"] == [0, 0, 1, 1, 1]
        assert rec["far"] == [0, 0, 0, 1, 0]

    def test_longitude_is_the_display_convention(self):
        rec = dv.track_record(loaded("A", "2021-06-01", 2))
        assert all(-180.0 <= x < 180.0 for x in rec["lon"])


class TestBuoyIndex:
    def test_one_entry_per_buoy_sorted_by_first_fix(self):
        df = pd.concat([loaded("B", "2021-06-02", 5), loaded("A", "2021-06-01", 5)])
        assert [r["id"] for r in dv.buoy_index(df)] == ["A", "B"]

    def test_the_tier_summary(self):
        df = pd.concat([
            loaded("D", "2021-01-01", 5),
            loaded("U", "2021-01-02", 5, lost="2021-01-01"),
            loaded("M", "2021-01-03", 5, lost="2021-01-03 02:00"),
            loaded("Q", "2021-01-04", 5, uncertain=True),
        ])
        tiers = {r["id"]: r["tier"] for r in dv.buoy_index(df)}
        assert tiers == {"D": "drogued", "U": "undrogued", "M": "mixed", "Q": "uncertain"}

    def test_a_mixed_buoy_says_when_it_lost_its_drogue(self):
        entry = dv.buoy_index(loaded("M", "2021-01-03", 5, lost="2021-01-03 02:00"))[0]
        assert entry["drogue_lost"] == "2021-01-03T02:00:00Z"
        assert dv.buoy_index(loaded("U", "2021-01-01", 5, lost="2020-12-01"))[0]["drogue_lost"] is None

    def test_splits_and_the_sealed_flag_come_from_the_units(self):
        df = pd.concat([loaded("A", "2021-01-01", 5), loaded("B", "2021-01-02", 5),
                        loaded("C", "2021-01-03", 5)])
        units = pd.DataFrame({"ID": ["A", "B", "B"], "split": ["dev", "sealed", "holdout-2023"]})
        index = {r["id"]: r for r in dv.buoy_index(df, units)}
        assert index["A"]["splits"] == ["dev"] and not index["A"]["sealed"]
        assert index["B"]["splits"] == ["holdout-2023", "sealed"] and index["B"]["sealed"]
        assert index["C"]["splits"] == [] and not index["C"]["sealed"]

    def test_first_and_last_fix_and_where_it_was_first_seen(self):
        entry = dv.buoy_index(loaded("A", "2021-06-01 05:00", 10))[0]
        assert entry["start"] == "2021-06-01T05:00:00Z"
        assert entry["end"] == "2021-06-01T15:00:00Z"
        assert entry["lat0"] == pytest.approx(28.0) and entry["lon0"] == pytest.approx(-75.0)
        assert entry["file"] == "drifter_tracks/A.json"


class TestPublishTracks:
    def test_writes_the_index_and_one_file_per_buoy(self, tmp_path):
        df = pd.concat([loaded("A", "2021-01-01", 5), loaded("B", "2021-01-02", 5)])
        summary = dv.publish_tracks(df, None, tmp_path)

        index = json.loads((tmp_path / dv.INDEX_NAME).read_text(encoding="utf-8"))
        assert index["counts"] == {"buoys": 2, "fixes": 12, "sealed_buoys": 0}
        for entry in index["buoys"]:
            track = json.loads((tmp_path / entry["file"]).read_text(encoding="utf-8"))
            assert track["id"] == entry["id"] and len(track["h"]) == entry["fixes"]
        assert summary["buoys"] == 2 and summary["track_bytes_total"] > 0
