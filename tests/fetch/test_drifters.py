"""Tests for the GDP drifter fetch (issue #55). No network: the one function that
reads a URL is replaced by a synthetic ERDDAP-shaped frame.

What is under test is what we decide -- which columns each product is asked for,
the date convention, where files go, and that `--out` cannot be forgotten -- not
the server, which was asked directly on 2026-09-23 and is recorded in the module.
"""

import urllib.parse

import pandas as pd
import pytest

from sar.fetch import drifters


def decoded(url: str) -> str:
    return urllib.parse.unquote(url)


class TestCoverageUrl:
    def test_end_is_exclusive_and_start_inclusive(self):
        url = decoded(drifters.coverage_url("drifter_hourly_qc", "2022-11-01", "2024-01-01", ["ID"]))
        assert "time>=2022-11-01T00:00:00Z" in url
        assert "time<2024-01-01T00:00:00Z" in url
        assert "time<=" not in url

    def test_constrains_to_the_study_box(self):
        url = decoded(drifters.coverage_url("drifter_hourly_qc", "2019-01-01", "2019-01-02", ["ID"]))
        for part in ("latitude>=17.0", "latitude<=36.0", "longitude>=-82.0", "longitude<=-63.0"):
            assert part in url

    def test_distinct_is_asked_for_only_when_wanted(self):
        plain = decoded(drifters.coverage_url("d", "2019-01-01", "2019-01-02", ["ID"]))
        unique = decoded(drifters.coverage_url("d", "2019-01-01", "2019-01-02", ["ID"], distinct=True))
        assert "distinct()" not in plain
        assert unique.endswith("&distinct()")

    def test_names_the_dataset_and_columns(self):
        url = drifters.coverage_url("drifter_6hour_qc", "2022-11-01", "2024-01-01", ["ID", "time"])
        assert "/drifter_6hour_qc.csv?ID,time&" in url


class TestProducts:
    def test_hourly_asks_for_the_fix_gap(self):
        assert "gap" in drifters.PRODUCTS["hourly"]["tracks"]

    def test_six_hourly_does_not_ask_for_fields_it_does_not_serve(self):
        # Read off the dataset's info endpoint: no `gap`, no `location_type`.
        spec = drifters.PRODUCTS["6-hourly"]
        assert "gap" not in spec["tracks"]
        assert "location_type" not in spec["buoys"]

    def test_the_unusable_error_fields_are_not_pulled(self):
        for spec in drifters.PRODUCTS.values():
            assert "err_lat" not in spec["tracks"] and "err_lon" not in spec["tracks"]

    def test_each_product_records_its_own_sst_units(self):
        assert drifters.PRODUCTS["hourly"]["sst_units"] == "Kelvin"
        assert drifters.PRODUCTS["6-hourly"]["sst_units"] == "degree_C"

    def test_an_unknown_product_is_refused_by_name(self):
        with pytest.raises(ValueError, match="unknown product"):
            drifters.product_spec("daily")


class TestOutputPaths:
    def test_hourly_keeps_the_existing_file_name(self, tmp_path):
        tracks, buoys = drifters.output_paths(tmp_path, "hourly", "2019-01-01", "2024-01-01")
        assert tracks == tmp_path / "raw" / "gdp_hourly_17-36N_82-63W_20190101-20240101.csv"
        assert buoys == tmp_path / "raw" / "gdp_hourly_buoys_17-36N_82-63W_20190101-20240101.csv"

    def test_six_hourly_is_named_so_the_loader_can_tell(self, tmp_path):
        tracks, _ = drifters.output_paths(tmp_path, "6-hourly", "2022-11-01", "2024-01-01")
        assert "_6hour_" in tracks.name


class TestReadErddapCsv:
    def test_drops_the_units_row_and_reads_dates_as_utc(self, tmp_path):
        p = tmp_path / "erddap.csv"
        p.write_text("ID,time,latitude\n,UTC,degrees_north\n1,2019-06-04T05:00:00Z,30.5\n",
                     encoding="utf-8")
        df = drifters.read_erddap_csv(str(p), ["time"])
        assert len(df) == 1
        assert df["latitude"].dtype.kind == "f"
        assert str(df["time"].dt.tz) == "UTC"


class TestFetch:
    def test_writes_tracks_and_buoys_without_the_network(self, tmp_path, monkeypatch):
        tracks = pd.DataFrame({
            "ID": [1, 1], "time": pd.to_datetime(["2021-06-01T00:00Z", "2021-06-01T01:00Z"]),
            "latitude": [30.0, 30.1], "longitude": [-70.0, -69.9], "ve": 0.1, "vn": 0.2,
            "sst": 298.15, "drogue_lost_date": pd.NaT, "gap": [3600.0, 14400.0],
        })
        buoys = pd.DataFrame({"ID": [1], "location_type": ["GPS"], "DrogueCenterDepth": ["15 m"]})
        asked = []

        def fake_read(url, dates):
            asked.append(url)
            return buoys if "distinct" in url else tracks

        monkeypatch.setattr(drifters, "read_erddap_csv", fake_read)
        t, b = drifters.fetch("hourly", "2021-06-01", "2021-06-02", tmp_path)

        assert t.exists() and b.exists()
        assert len(pd.read_csv(t)) == 2 and len(pd.read_csv(b)) == 1
        assert len(asked) == 2 and "distinct" in asked[1]


class TestCli:
    def test_out_is_required(self, monkeypatch):
        # It used to default to `data` in the working directory, which is how raw
        # CSVs land inside OneDrive (CLAUDE.md rule 4).
        monkeypatch.setattr("sys.argv", ["drifters", "--start", "2021-01-01", "--end", "2021-01-02"])
        with pytest.raises(SystemExit):
            drifters.main()
