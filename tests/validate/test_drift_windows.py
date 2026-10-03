"""Tests for sar.validate.drift_windows: dev tracks cut into scoreable windows (#89)."""

import numpy as np
import pandas as pd
import pytest

from sar.validate.drift_windows import (
    SealedUnitError,
    Windows,
    first_midnight,
    make_windows,
)


def fixes(bid="A", start="2021-01-01 05:00", hours=120, gap=0.0):
    t = pd.date_range(start, periods=hours + 1, freq="h", tz="UTC")
    return pd.DataFrame({"ID": bid, "time": t, "lat": 25.0 + 0.01 * np.arange(hours + 1),
                         "lon": 285.0, "ve": 0.1, "vn": 0.2, "fix_gap_h": gap})


def unit(bid="A", start="2021-01-01 05:00", hours=120, split="dev", tier="undrogued"):
    s = pd.Timestamp(start, tz="UTC")
    return pd.DataFrame([{"unit": f"{bid}@{s:%Y-%m-%dT%H:%M}", "ID": bid, "start": s,
                          "end": s + pd.Timedelta(hours=hours), "tier": tier, "group": 3,
                          "split": split}])


class TestWindows:
    def test_windows_start_at_midnight_and_do_not_overlap(self):
        w = make_windows(unit(), fixes())
        assert w.table["t0"].tolist() == [pd.Timestamp("2021-01-02"), pd.Timestamp("2021-01-04")]
        assert len(w) == 2 and w.lat.shape == (2, 49)

    def test_a_window_carries_the_buoy_hour_by_hour(self):
        w = make_windows(unit(), fixes())
        # 2021-01-02 00:00 is 19 h after the first fix
        assert w.lat[0, 0] == pytest.approx(25.0 + 0.19)
        assert w.lat[0, 48] == pytest.approx(25.0 + 0.67)
        assert w.table.loc[0, "lat0"] == pytest.approx(25.19)
        assert w.ok.all()

    def test_a_sealed_or_holdout_unit_raises_and_there_is_no_override(self):
        for split in ("sealed", "holdout-2023"):
            with pytest.raises(SealedUnitError, match="never see"):
                make_windows(unit(split=split), fixes())

    def test_truth_more_than_3_h_from_a_fix_does_not_count(self):
        f = fixes()
        f.loc[f["time"] == pd.Timestamp("2021-01-02 10:00", tz="UTC"), "fix_gap_h"] = 5.0
        w = make_windows(unit(), f)
        assert not w.ok[0, 10] and w.ok[0, 9] and w.ok[0, 11]

    def test_an_interpolated_start_drops_the_window(self):
        f = fixes()
        f.loc[f["time"] == pd.Timestamp("2021-01-02 00:00", tz="UTC"), "fix_gap_h"] = 6.0
        w = make_windows(unit(), f)
        assert len(w) == 1 and w.dropped["start_fix_missing_or_interpolated"] == 1

    def test_a_short_unit_gives_no_window(self):
        assert len(make_windows(unit(hours=60), fixes(hours=60))) == 0

    def test_non_positive_hours_raise(self):
        with pytest.raises(ValueError, match="positive"):
            make_windows(unit(), fixes(), hours=0)

    def test_first_midnight(self):
        assert first_midnight(pd.Timestamp("2021-01-01 00:00", tz="UTC")).hour == 0
        assert first_midnight(pd.Timestamp("2021-01-01 00:01", tz="UTC")).day == 2

    def test_batches_group_windows_that_start_together(self):
        u = pd.concat([unit("A"), unit("B")], ignore_index=True)
        w = make_windows(u, pd.concat([fixes("A"), fixes("B")]))
        assert [len(b) for b in w.batches()] == [2, 2]

    def test_save_and_load_round_trip(self, tmp_path):
        w = make_windows(unit(), fixes())
        w.save(tmp_path / "w")
        back = Windows.load(tmp_path / "w")
        assert back.table["window"].tolist() == w.table["window"].tolist()
        assert np.array_equal(back.lat, w.lat) and back.hours == 48
