"""Tests for sar.pipeline.forcing. No files and no network."""

import numpy as np
import pytest

from sar.pipeline.forcing import ConstantForcing, as_particle_axes

TIME = np.datetime64("2021-01-05T07:30")


class TestAsParticleAxes:
    def test_scalars_become_one_particle(self):
        lats, lons = as_particle_axes(26.5, 281.4)
        assert lats.shape == (1,) and lons.shape == (1,)

    def test_sequences_stay_as_they_are(self):
        lats, _ = as_particle_axes([26.5, 27.0], [281.4, 281.0])
        assert lats.tolist() == [26.5, 27.0]

    def test_longitude_is_put_into_the_stored_range(self):
        """D020 stores 0 to 360, so -78.6 arrives as 281.4."""
        _, lons = as_particle_axes(26.5, -78.6)
        assert lons[0] == pytest.approx(281.4)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="do not pair up"):
            as_particle_axes([26.5, 27.0], [281.4])


class TestConstantForcing:
    def test_returns_the_same_vector_at_every_particle(self):
        forcing = ConstantForcing(current=(1.0, 0.0), wind=(5.0, 0.0))
        current, wind = forcing.sample([26.5, 27.0, 36.0], [281.4, 281.0, 280.0], TIME)
        assert current.shape == (3, 2) and wind.shape == (3, 2)
        assert current == pytest.approx(np.tile([1.0, 0.0], (3, 1)))
        assert wind == pytest.approx(np.tile([5.0, 0.0], (3, 1)))

    def test_the_default_field_is_still_water_and_calm_air(self):
        current, wind = ConstantForcing().sample(26.5, 281.4, TIME)
        assert current == pytest.approx(np.zeros((1, 2)))
        assert wind == pytest.approx(np.zeros((1, 2)))

    def test_the_time_does_not_change_the_answer(self):
        forcing = ConstantForcing(current=(1.0, 0.0))
        early, _ = forcing.sample(26.5, 281.4, np.datetime64("2021-01-05T00:00"))
        late, _ = forcing.sample(26.5, 281.4, np.datetime64("2021-06-05T00:00"))
        assert early == pytest.approx(late)

    def test_a_three_component_field_raises(self):
        with pytest.raises(ValueError, match=r"current must be one \[u, v\] pair"):
            ConstantForcing(current=(1.0, 0.0, 0.0))

    def test_describe_names_the_backend(self):
        assert ConstantForcing().describe()["backend"] == "constant"
