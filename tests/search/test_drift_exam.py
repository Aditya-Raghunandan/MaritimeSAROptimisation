"""Tests for scripts/drift_exam.py, imported by path, on Gaussian clouds worked out by hand."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sar.utils.geo import offset_position

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exam = _load("drift_exam")
LKP = (27.0, -79.0)


def _cloud(centre_east, centre_north, s_east, s_north, n=20_000, seed=3):
    """A Gaussian cloud in degrees, its centre `centre_*` metres from the LKP."""
    rng = np.random.default_rng(seed)
    east = rng.normal(centre_east, s_east, n)
    north = rng.normal(centre_north, s_north, n)
    lat, lon = offset_position(*LKP, east, north)
    return np.asarray(lat), np.asarray(lon), np.full(n, 1.0 / n)


def _at(east, north):
    lat, lon = offset_position(*LKP, np.array([east]), np.array([north]))
    return float(np.asarray(lat)[0]), float(np.asarray(lon)[0])


def test_a_buoy_at_the_centre_is_inside_and_one_far_off_is_not():
    lat, lon, w = _cloud(3000, 0, 800, 400)
    rng = np.random.default_rng(0)
    near = exam.score_frame(lat, lon, w, LKP, _at(3000, 0), rng)
    far = exam.score_frame(lat, lon, w, LKP, _at(3000, 2000), rng)
    assert near["inside90"] and near["inside_along"] and near["inside_across"]
    assert not far["inside90"] and far["inside_along"] and not far["inside_across"]
    assert near["d_centre_m"] < 100 and near["d_stayed_m"] == pytest.approx(3000, rel=0.02)


def test_along_is_down_the_drift_and_across_is_to_its_left():
    lat, lon, w = _cloud(3000, 0, 800, 400)                 # drifting east
    out = exam.score_frame(lat, lon, w, LKP, _at(2000, 500), np.random.default_rng(0))
    assert out["along_m"] == pytest.approx(-1000, abs=60)     # behind the centre
    assert out["across_m"] == pytest.approx(500, abs=60)      # north is left of east
    assert out["spread_along_m"] == pytest.approx(800, rel=0.05)
    assert out["spread_across_m"] == pytest.approx(400, rel=0.05)


def test_a_calibrated_cloud_holds_about_ninety_percent():
    rng = np.random.default_rng(1)
    lat, lon, w = _cloud(3000, 0, 800, 800, n=5000)
    draws = rng.normal([3000, 0], 800, size=(300, 2))
    inside = [exam.score_frame(lat, lon, w, LKP, _at(*d), rng)["inside90"] for d in draws]
    assert np.mean(inside) == pytest.approx(0.9, abs=0.05)


def test_the_sign_test_counts_groups_not_rows():
    df = pd.DataFrame({"group": [1, 1, 1, 2, 3, 4, 5, 6, 7, 8],
                       "d_centre_m": [1, 1, 1, 1, 1, 1, 1, 1, 1, 9],
                       "d_stayed_m": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]})
    out = exam.sign_test(df)
    assert out["groups"] == 8 and out["centre_wins"] == 7
    assert out["sign_p"] == pytest.approx(9 / 256)
