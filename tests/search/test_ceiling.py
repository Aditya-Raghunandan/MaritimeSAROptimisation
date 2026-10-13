"""Tests for scripts/ceiling.py, imported by path, against Gaussian clouds worked out by hand."""

import importlib.util
import sys
from math import erf, exp, pi, sqrt
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ceiling = _load("ceiling")
A = ceiling.BUDGET_M2


def _cloud(sigma_along_m, sigma_across_m, bearing_deg=0.0, n=200_000, seed=1, shift=(0.0, 0.0)):
    """A Gaussian cloud, its long axis down `bearing_deg`, as (east, north, weight)."""
    rng = np.random.default_rng(seed)
    along, across = rng.normal(0, sigma_along_m, n), rng.normal(0, sigma_across_m, n)
    s, c = np.sin(np.radians(bearing_deg)), np.cos(np.radians(bearing_deg))
    return along * s + across * c + shift[0], along * c - across * s + shift[1], np.full(n, 1.0 / n)


def _box(a, b, sa, sc):
    return erf(a / 2 / (sqrt(2) * sa)) * erf(b / 2 / (sqrt(2) * sc))


def test_the_budget_is_one_window_of_sweep():
    assert A == pytest.approx(185.2 * 90 * 1852 / 3600 * 2700)
    assert A / 1e6 == pytest.approx(23.15, abs=0.01)


def test_a_round_cloud_any_shape_is_the_disc_and_the_square_nearly_reaches_it():
    east, north, w = _cloud(2000, 2000)
    disc = 1 - exp(-A / (2 * pi * 2000 ** 2))                       # 0.602
    assert ceiling.any_shape(east, north, w) == pytest.approx(disc, abs=0.01)
    assert ceiling.any_shape_split(east, north, w) == pytest.approx(disc, abs=0.01)
    side = sqrt(A)
    assert ceiling.square(east, north, w, 0.0) == pytest.approx(_box(side, side, 2000, 2000), abs=0.005)
    assert ceiling.best_rectangle(east, north, w)["mass"] == pytest.approx(0.594, abs=0.01)


def test_a_stretched_cloud_is_fitted_by_a_long_rectangle_along_it():
    east, north, w = _cloud(4000, 1500, bearing_deg=40.0)
    rect = ceiling.best_rectangle(east, north, w)
    assert rect["mass"] == pytest.approx(0.454, abs=0.01)
    assert rect["bearing_deg"] == pytest.approx(40.0, abs=5.0)
    assert 2.0 <= rect["aspect"] <= 4.0
    assert ceiling.square(east, north, w, 40.0) == pytest.approx(0.403, abs=0.01)
    assert rect["length_m"] * rect["width_m"] <= A


def test_the_rectangle_finds_a_cloud_off_the_marker():
    east, north, w = _cloud(1500, 1500, shift=(3000.0, -2000.0))
    rect = ceiling.best_rectangle(east, north, w)
    assert rect["centre_east_m"] == pytest.approx(3000.0, abs=200.0)
    assert rect["centre_north_m"] == pytest.approx(-2000.0, abs=200.0)
    assert ceiling.square(east, north, w, 0.0) < 0.5 * rect["mass"]


def test_a_few_particles_flatter_the_ceiling_and_the_split_does_not():
    east, north, w = _cloud(4000, 4000, n=2_000)
    truth = 1 - exp(-A / (2 * pi * 4000 ** 2))                      # 0.206
    assert ceiling.any_shape(east, north, w) > truth + 0.05
    assert ceiling.any_shape_split(east, north, w) < truth


def test_area90_matches_the_benchmarks_rule_on_one_cell():
    east, north, w = np.array([10.0, 20.0, 600.0]), np.array([10.0, 30.0, 600.0]), np.array([0.5, 0.45, 0.05])
    assert ceiling.area_km2(east, north, w) == pytest.approx(0.0625)
