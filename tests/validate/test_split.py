"""Tests for the sealed drifter validation set (issue #51). Synthetic frames, no network.

The load-bearing guarantee is that nothing reported can have been debugged on:
no shared-water group straddles dev and sealed, and the checker refuses a split
where one does. Everything else here supports that.
"""

import numpy as np
import pandas as pd
import pytest

from sar.validate import split as sp


def track(bid: str, start: str, hours: int, *, lat: float = 28.0, lon: float = 285.0,
          lost: str | None = None, uncertain: bool = False, product: str = "hourly",
          segment: int = 0, step: str = "h") -> pd.DataFrame:
    """One buoy's fixes in the loaded shape: hourly by default, drogued unless `lost`."""
    t = pd.date_range(start, periods=hours + 1, freq=step, tz="UTC")
    lost_ts = pd.Timestamp(lost, tz="UTC") if lost else None
    return pd.DataFrame({
        "ID": bid, "time": t, "lat": lat, "lon": lon, "segment_id": segment,
        "undrogued": (t >= lost_ts) if lost_ts is not None else False,
        "tier_uncertain": uncertain, "product": product,
    })


class TestValidationUnits:
    def test_a_run_that_loses_its_drogue_becomes_two_units(self):
        df = track("A", "2021-01-01", 120, lost="2021-01-03 06:00")   # 53 h, then 66 h
        units = sp.validation_units(df)
        assert sorted(units["tier"]) == ["drogued", "undrogued"]

    def test_a_piece_shorter_than_48_h_is_dropped(self):
        df = track("A", "2021-01-01", 100, lost="2021-01-02")    # 24 h drogued, 76 h not
        assert sp.validation_units(df)["tier"].tolist() == ["undrogued"]

    def test_48_hourly_fixes_span_47_hours_and_do_not_count(self):
        assert sp.validation_units(track("A", "2021-01-01", 47)).empty
        assert len(sp.validation_units(track("A", "2021-01-01", 48))) == 1

    def test_a_tier_uncertain_buoy_is_its_own_tier(self):
        units = sp.validation_units(track("U", "2021-01-01", 60, uncertain=True))
        assert units["tier"].tolist() == ["uncertain"]

    def test_the_unit_key_is_the_buoy_and_its_first_fix(self):
        units = sp.validation_units(track("A", "2021-01-01 06:00", 60))
        assert units["unit"].tolist() == ["A@2021-01-01T06:00"]

    def test_the_key_does_not_depend_on_row_order(self):
        df = pd.concat([track("A", "2021-01-01", 60), track("B", "2021-03-01", 60, segment=1)])
        shuffled = df.sample(frac=1.0, random_state=3)
        assert (sorted(sp.validation_units(df)["unit"])
                == sorted(sp.validation_units(shuffled)["unit"]))


class TestSharedWaterPairs:
    def test_buoys_5_km_apart_at_the_same_time_share_water(self):
        a = track("A", "2021-01-01", 2, lat=28.0)
        b = track("B", "2021-01-01", 2, lat=28.045, segment=1)       # ~5 km north
        assert sp.shared_water_pairs(pd.concat([a, b])) == {("A", "B")}

    def test_the_same_place_at_different_times_does_not(self):
        a = track("A", "2021-01-01", 2)
        b = track("B", "2021-02-01", 2, segment=1)
        assert sp.shared_water_pairs(pd.concat([a, b])) == set()

    def test_15_km_apart_is_not_shared_at_10_km(self):
        a = track("A", "2021-01-01", 2, lat=28.0)
        b = track("B", "2021-01-01", 2, lat=28.135, segment=1)       # ~15 km
        assert sp.shared_water_pairs(pd.concat([a, b]), km=10) == set()
        assert sp.shared_water_pairs(pd.concat([a, b]), km=20) == {("A", "B")}

    def test_pairs_are_ordered_so_each_appears_once(self):
        a = track("Z", "2021-01-01", 3)
        b = track("A", "2021-01-01", 3, segment=1)
        assert sp.shared_water_pairs(pd.concat([a, b])) == {("A", "Z")}


class TestBuoyGroups:
    def test_a_chain_of_shared_water_is_one_group(self):
        groups = sp.buoy_groups(["A", "B", "C", "D"], {("A", "B"), ("B", "C")})
        assert groups["A"] == groups["B"] == groups["C"]
        assert groups["D"] != groups["A"]

    def test_numbering_follows_the_smallest_member(self):
        groups = sp.buoy_groups(["Q", "B", "A"], {("B", "Q")})
        assert groups["A"] == 0 and groups["B"] == groups["Q"] == 1


def many_units(n_groups: int = 40) -> tuple[pd.DataFrame, dict[str, int]]:
    """Units for `n_groups` single-buoy groups, alternating tiers, plus two 2023 units."""
    rows = []
    for k in range(n_groups):
        rows.append({"unit": f"B{k}@x", "ID": f"B{k:02d}", "product": "hourly",
                     "tier": "undrogued" if k % 3 else "drogued"})
    rows.append({"unit": "S1@x", "ID": "S1", "product": "6-hourly", "tier": "undrogued"})
    rows.append({"unit": "S2@x", "ID": "S2", "product": "6-hourly", "tier": "drogued"})
    units = pd.DataFrame(rows)
    groups = sp.buoy_groups(units["ID"], set())
    return units, groups


class TestAssignSplit:
    def test_every_six_hourly_unit_is_the_2023_holdout(self):
        units, groups = many_units()
        split = sp.assign_split(units, groups)
        assert (split[units["product"] == "6-hourly"] == "holdout-2023").all()
        assert not (split[units["product"] == "hourly"] == "holdout-2023").any()

    def test_a_group_never_straddles_dev_and_sealed(self):
        units, _ = many_units()
        groups = {bid: (0 if bid in ("B00", "B01", "B02") else i + 1)
                  for i, bid in enumerate(units["ID"])}
        split = sp.assign_split(units, groups, seed=1)
        assert split[units["ID"].isin(["B00", "B01", "B02"])].nunique() == 1

    def test_the_same_seed_gives_the_same_split(self):
        units, groups = many_units()
        assert sp.assign_split(units, groups, seed=7).equals(sp.assign_split(units, groups, seed=7))

    def test_a_different_seed_gives_a_different_split(self):
        units, groups = many_units()
        assert not sp.assign_split(units, groups, seed=7).equals(sp.assign_split(units, groups, seed=8))

    def test_both_tiers_are_sealed_near_the_target(self):
        units, groups = many_units(120)
        units["split"] = sp.assign_split(units, groups, sealed_fraction=0.25)
        hourly = units[units["product"] == "hourly"]
        share = hourly.groupby("tier")["split"].apply(lambda s: (s == "sealed").mean())
        assert share.between(0.2, 0.32).all()

    def test_a_fraction_outside_zero_to_one_is_refused(self):
        units, groups = many_units()
        with pytest.raises(ValueError, match="sealed_fraction"):
            sp.assign_split(units, groups, sealed_fraction=1.2)


class TestCheckSplit:
    def split_units(self) -> tuple[pd.DataFrame, dict[str, int]]:
        units, groups = many_units()
        units["split"] = sp.assign_split(units, groups)
        return units, groups

    def test_an_honest_split_passes(self):
        units, groups = self.split_units()
        sp.check_split(units, groups)

    def test_a_group_on_both_sides_is_refused(self):
        units, groups = self.split_units()
        a, b = units.index[(units["product"] == "hourly")][:2]
        groups = {**groups, units.at[b, "ID"]: groups[units.at[a, "ID"]]}
        units.loc[a, "split"], units.loc[b, "split"] = "dev", "sealed"
        with pytest.raises(ValueError, match="straddle"):
            sp.check_split(units, groups)

    def test_too_few_sealed_units_is_refused(self):
        units, groups = self.split_units()
        with pytest.raises(ValueError, match="R6d"):
            sp.check_split(units, groups, min_sealed=10_000)

    def test_a_sealed_set_missing_a_tier_is_refused(self):
        units, groups = self.split_units()
        units.loc[(units["product"] == "hourly") & (units["tier"] == "drogued"), "split"] = "dev"
        with pytest.raises(ValueError, match="no drogued unit is sealed"):
            sp.check_split(units, groups, min_sealed=1)

    def test_a_2023_unit_outside_the_holdout_is_refused(self):
        units, groups = self.split_units()
        units.loc[units["product"] == "6-hourly", "split"] = "dev"
        with pytest.raises(ValueError, match="2023 holdout"):
            sp.check_split(units, groups)


class TestBuild:
    def test_end_to_end_on_synthetic_drifters(self):
        """Twelve buoys, two of which drift together; 2023 tracks join the holdout."""
        frames = []
        for k in range(12):
            frames.append(track(f"H{k:02d}", "2021-01-01", 72, lat=20.0 + k, segment=k,
                                lost="2021-01-02" if k % 2 else None))
        frames.append(track("H99", "2021-01-01", 72, lat=20.02, segment=99))   # beside H00
        frames.append(track("S00", "2023-02-01", 12, product="6-hourly", step="6h", segment=200))
        df = pd.concat(frames, ignore_index=True)

        units, groups = sp.build(df, sealed_fraction=0.4, min_sealed=2)

        assert groups["H00"] == groups["H99"]
        assert units.loc[units["ID"].isin(["H00", "H99"]), "split"].nunique() == 1
        assert units.loc[units["ID"] == "S00", "split"].tolist() == ["holdout-2023"]
        assert set(units["split"]) <= set(sp.SPLITS)
