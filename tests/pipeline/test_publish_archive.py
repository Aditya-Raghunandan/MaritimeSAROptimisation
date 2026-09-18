"""Unit tests for scripts/publish_archive.py.

scripts/ is not an installed package, so it is imported by path, the same way
tests/pipeline/test_find_current_gaps.py does it.

`huggingface_hub` is an optional [publish] extra and is NOT installed in CI, so
nothing here may import it. The script imports it inside the functions that
actually talk to the Hub for exactly that reason, which leaves the two gates
worth testing -- what gets counted, and what gets refused -- importable and
testable on their own.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "publish_archive.py"
spec = importlib.util.spec_from_file_location("publish_archive", SCRIPT_PATH)
publish_archive = importlib.util.module_from_spec(spec)
sys.modules["publish_archive"] = publish_archive
spec.loader.exec_module(publish_archive)


def _manifest(tmp_path, product="wind", **tier_overrides):
    """A published directory shaped like one sar.viz.archive writes."""
    tier = {
        "path": f"{product}_hourly.zarr",
        "frames": 48,
        "step_seconds": 3600,
        "regular": True,
        "gaps": None,
        "start": "2021-01-01T00:00:00Z",
        "end": "2021-01-02T23:00:00Z",
        "chunks": {"time": 48, "lat": 77, "lon": 77},
        "bytes": 1_000_000,
        "compression": "zstd-19",
        "compression_ratio": 1.58,
        "files": 17,
        "verified": {"at": {}, "published": 1.0, "source": 1.0, "match": True},
    }
    tier.update(tier_overrides)
    store = tmp_path / tier["path"]
    (store / "u10" / "c" / "0").mkdir(parents=True)
    (store / "u10" / "c" / "0" / "0").write_bytes(b"x" * 1024)
    (store / "zarr.json").write_text("{}", encoding="utf-8")
    (tmp_path / f"{product}_archive.json").write_text(
        json.dumps({"product": product, "tiers": {"hourly": tier}}), encoding="utf-8")
    return tmp_path


class TestSummarise:
    def test_counts_files_and_bytes(self, tmp_path):
        info = publish_archive.summarise(_manifest(tmp_path))
        assert info["files"] == 3          # the chunk, zarr.json, the manifest
        assert info["bytes"] > 1024

    def test_names_the_stores_and_manifests(self, tmp_path):
        info = publish_archive.summarise(_manifest(tmp_path))
        assert info["stores"] == ["wind_hourly.zarr"]
        assert info["manifests"] == ["wind_archive.json"]

    def test_an_empty_directory_reports_nothing_rather_than_raising(self, tmp_path):
        info = publish_archive.summarise(tmp_path)
        assert info == {"files": 0, "bytes": 0, "stores": [], "manifests": []}


class TestCheckManifests:
    """The second gate.

    sar.viz.archive already refuses to WRITE an irregular or unverified tier.
    This refuses to PUBLISH one, because the two steps can run days apart and
    what reaches the browser is what matters.
    """

    def test_a_clean_archive_has_no_problems(self, tmp_path):
        assert publish_archive.check_manifests(_manifest(tmp_path)) == []

    def test_an_irregular_time_axis_is_refused(self, tmp_path):
        src = _manifest(tmp_path, regular=False,
                        gaps=[{"after": "2021-01-01T23:00:00Z", "gap_hours": 1225.0}])
        problems = publish_archive.check_manifests(src)
        assert len(problems) == 1
        assert "irregular time axis" in problems[0]
        assert "1 gap" in problems[0]

    def test_a_failed_round_trip_is_refused(self, tmp_path):
        src = _manifest(tmp_path,
                        verified={"at": {}, "published": 1.0, "source": 2.0, "match": False})
        problems = publish_archive.check_manifests(src)
        assert len(problems) == 1
        assert "round-trip" in problems[0]

    def test_both_faults_are_reported_together(self, tmp_path):
        src = _manifest(tmp_path, regular=False, gaps=[{"after": "x", "gap_hours": 9.0}],
                        verified={"at": {}, "match": False})
        assert len(publish_archive.check_manifests(src)) == 2

    def test_a_directory_with_no_manifest_reports_no_problems(self, tmp_path):
        # Nothing to check is not the same as something wrong; `publish` catches
        # the empty case separately, on the absence of any .zarr store.
        assert publish_archive.check_manifests(tmp_path) == []


class TestPublishGuards:
    """Everything `publish` refuses before it would need a credential."""

    def test_a_missing_source_directory_is_refused(self, tmp_path):
        with pytest.raises(SystemExit, match="not a directory"):
            publish_archive.publish(tmp_path / "nope", "user/repo")

    def test_a_directory_with_no_store_is_refused(self, tmp_path):
        with pytest.raises(SystemExit, match="nothing to publish"):
            publish_archive.publish(tmp_path, "user/repo", dry_run=True)

    def test_an_irregular_tier_is_refused_before_anything_else(self, tmp_path):
        src = _manifest(tmp_path, regular=False,
                        gaps=[{"after": "2021-01-01T23:00:00Z", "gap_hours": 1225.0}])
        with pytest.raises(SystemExit, match="refusing to publish"):
            publish_archive.publish(src, "user/repo", dry_run=True)

    def test_dry_run_stops_before_authenticating(self, tmp_path):
        """No credential, no network: a dry run must work on a bare machine."""
        assert publish_archive.publish(_manifest(tmp_path), "user/repo", dry_run=True) == ""

    def test_dry_run_writes_no_dataset_card(self, tmp_path):
        src = _manifest(tmp_path)
        publish_archive.publish(src, "user/repo", dry_run=True)
        assert not (src / "README.md").exists()


class TestCard:
    def test_the_card_carries_the_conventions_a_stranger_needs(self):
        card = publish_archive.CARD
        for needed in ("-180..180", "ascending", "zstd", "48", "strided"):
            assert needed in card, f"the dataset card never mentions {needed!r}"

    def test_the_card_credits_both_upstream_sources(self):
        card = publish_archive.CARD
        assert "Copernicus" in card and "HYCOM" in card

    def test_the_repo_placeholder_is_substituted_not_left_literal(self):
        filled = publish_archive.CARD.replace("{repo}", "AdityaRugs/MaritimeSARoperations")
        assert "{repo}" not in filled
        assert "AdityaRugs/MaritimeSARoperations" in filled
