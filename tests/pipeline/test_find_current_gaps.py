"""Unit tests for scripts/find_current_gaps.py.

scripts/ is not an installed package, so it is imported by path.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "find_current_gaps.py"
spec = importlib.util.spec_from_file_location("find_current_gaps", SCRIPT_PATH)
find_current_gaps = importlib.util.module_from_spec(spec)
sys.modules["find_current_gaps"] = find_current_gaps
spec.loader.exec_module(find_current_gaps)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2019-01-01T00:00:00"] * 3 + ["2019-01-01T03:00:00"] * 3
            ),
            "lat": [17.0, 17.1, 17.2, 17.0, 17.1, 17.2],
            "lon": [280.0, 280.1, 280.2, 280.0, 280.1, 280.2],
            "water_u": [0.5, np.nan, 0.3, 0.4, np.nan, np.nan],
            "water_v": [0.1, np.nan, 0.2, 0.1, 0.05, np.nan],
        }
    )


class TestFindGaps:
    def test_selects_rows_with_any_nan(self, sample_df):
        gaps = find_current_gaps.find_gaps(sample_df)
        assert len(gaps) == 3
        assert set(gaps["lat"]) == {17.1, 17.2}

    def test_no_gaps_returns_empty(self):
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2019-01-01T00:00:00"]),
                "lat": [17.0], "lon": [280.0],
                "water_u": [0.5], "water_v": [0.1],
            }
        )
        gaps = find_current_gaps.find_gaps(df)
        assert len(gaps) == 0


class TestReadRecords:
    def test_reads_csv(self, tmp_path, sample_df):
        path = tmp_path / "current_x.csv"
        sample_df.to_csv(path, index=False)
        df = find_current_gaps.read_records(path)
        assert len(df) == len(sample_df)
        assert df["water_u"].isna().sum() == sample_df["water_u"].isna().sum()

    def test_reads_parquet(self, tmp_path, sample_df):
        path = tmp_path / "current_x.parquet"
        sample_df.to_parquet(path, index=False)
        df = find_current_gaps.read_records(path)
        assert len(df) == len(sample_df)

    def test_reads_txt_written_by_fetch_current_range(self, tmp_path, sample_df):
        # Mirrors scripts/fetch_current_range.py's txt writer: timestamps must
        # not contain a raw space, or whitespace-delimited parsing misaligns.
        out_df = sample_df.copy()
        out_df["time"] = out_df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        path = tmp_path / "current_x.txt"
        out_df.to_string(path, index=False)

        df = find_current_gaps.read_records(path)
        assert len(df) == len(sample_df)
        assert df["time"].iloc[0] == sample_df["time"].iloc[0]
        assert df["water_u"].isna().sum() == sample_df["water_u"].isna().sum()

    def test_unrecognised_extension_raises(self, tmp_path):
        path = tmp_path / "current_x.nc"
        path.write_text("not real data")
        with pytest.raises(ValueError, match="unrecognised"):
            find_current_gaps.read_records(path)


class TestWriteReport:
    def test_report_contains_summary_and_rows(self, tmp_path, sample_df):
        gaps = find_current_gaps.find_gaps(sample_df)
        out_path = tmp_path / "gaps_current_x.txt"
        find_current_gaps.write_report(sample_df, gaps, Path("current_x.csv"), out_path)

        text = out_path.read_text()
        assert "NaN records        : 3" in text
        assert "distinct locations : 2" in text
        assert "17.1" in text and "17.2" in text

    def test_report_handles_zero_gaps(self, tmp_path):
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2019-01-01T00:00:00"]),
                "lat": [17.0], "lon": [280.0],
                "water_u": [0.5], "water_v": [0.1],
            }
        )
        gaps = find_current_gaps.find_gaps(df)
        out_path = tmp_path / "gaps_current_x.txt"
        find_current_gaps.write_report(df, gaps, Path("current_x.csv"), out_path)

        text = out_path.read_text()
        assert "NaN records        : 0" in text
        assert "(none)" in text
