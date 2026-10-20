"""The cloud-shape correction (vault L41, 18 Oct): the random velocity stretched along the drift,
and the forcing scaled."""

import numpy as np
import pytest

from sar.model.position import shape_velocity_error
from sar.pipeline.ensemble import run_ensemble
from sar.pipeline.forcing import ConstantForcing, ScaledForcing


def test_factors_of_one_change_nothing():
    rng = np.random.default_rng(1)
    u, d = rng.standard_normal((50, 2)), rng.standard_normal((50, 2))
    assert np.allclose(shape_velocity_error(u, d, 1.0, 1.0), u)


def test_along_and_across_by_hand():
    # Drifting east: along is east, across (left of the drift) is north.
    out = shape_velocity_error([[1.0, 1.0]], [[0.5, 0.0]], 2.0, 0.5)
    assert np.allclose(out, [[2.0, 0.5]])
    # Drifting north: along is north, across (left) is west.
    out = shape_velocity_error([[1.0, 1.0]], [[0.0, 0.3]], 2.0, 0.5)
    assert np.allclose(out, [[0.5, 2.0]])


def test_a_particle_with_no_drift_direction_keeps_its_error():
    out = shape_velocity_error([[1.0, -2.0], [1.0, 1.0]], [[0.0, 0.0], [np.nan, np.nan]], 3.0, 0.1)
    assert np.allclose(out, [[1.0, -2.0], [1.0, 1.0]])


def test_scaled_forcing_scales_each_field():
    f = ScaledForcing(ConstantForcing(current=(1.0, -0.5), wind=(10.0, 0.0)), 0.8, 0.5)
    current, wind = f.sample([25.0, 26.0], [280.0, 281.0], np.datetime64("2020-01-01"))
    assert np.allclose(current, [[0.8, -0.4]] * 2)
    assert np.allclose(wind, [[5.0, 0.0]] * 2)
    assert f.describe()["current_scale"] == 0.8
    with pytest.raises(ValueError):
        ScaledForcing(ConstantForcing(), -1.0)


def test_the_cloud_comes_out_stretched_along_the_drift():
    forcing = ConstantForcing(current=(0.5, 0.0))
    kw = dict(forcing=forcing, particles=4000, start="2020-01-01T00:00", lat=25.0, lon=280.0,
              duration=3600.0, timestep=60.0, datum_sigma_km=0.0, seed=7, sigma_u=0.2)
    round_ = run_ensemble(**kw)
    shaped = run_ensemble(**kw, shape=(1.5, 0.5))

    def spreads(e):                    # in metres: a degree of longitude is cos(lat) of one of latitude
        return np.std(e.lon[-1]) * np.cos(np.radians(25.0)), np.std(e.lat[-1])

    (re, rn), (se, sn) = spreads(round_), spreads(shaped)
    assert re / rn == pytest.approx(1.0, abs=0.05)
    assert se / sn == pytest.approx(3.0, rel=0.05)                 # 1.5 / 0.5, east is along
    assert np.mean(shaped.lon[-1]) == pytest.approx(np.mean(round_.lon[-1]), abs=1e-3)


def test_the_shape_by_the_current_is_round_in_still_water_and_long_in_fast():
    from sar.model.position import shape_for_current
    rule = {"a_along": 0.17, "b_along": 0.6, "a_across": 0.17, "b_across": 0.2}
    a0, c0 = shape_for_current(0.0, rule)
    assert a0 == pytest.approx(c0)
    a1, c1 = shape_for_current(1.0, rule)
    assert a1 / c1 > 2.0
    assert shape_for_current(5.0, rule) == shape_for_current(1.658, rule)   # D033's cap
