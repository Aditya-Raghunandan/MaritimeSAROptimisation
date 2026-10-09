"""Shared windows for the sar.rl tests: a real engine run under a steady current, saved every 5 min."""

import numpy as np
import pytest

from sar.model.position import CALIBRATED_SIGMA_U
from sar.pipeline.ensemble import run_ensemble
from sar.pipeline.forcing import ConstantForcing
from sar.rl.windows import build, save

LAT, LON = 25.1629, -80.0647                   # the Florida Current location in scripts/rl_locations.txt
START = "2019-01-01T17:00"
FORCING = ConstantForcing(current=(0.3, 1.5), wind=(4.0, -2.0))   # a test choice: a northward jet
ARRIVALS_S = (1800.0, 2700.0, 3600.0)          # the 30, 45 and 60 minute transits


def make_run(particles=600, seed=7, save_every=300.0, duration=6300.0, start=START):
    """The run's parameters as its JSON sidecar holds them, and the ensemble itself."""
    run = {"lat": round(LAT, 2), "lon": round(LON, 2), "start": start, "particles": particles,
           "timestep": 60.0, "duration": duration, "seed": seed, "datum_lat": LAT,
           "datum_lon": LON}
    ens = run_ensemble(FORCING, particles, np.datetime64(start, "us"), LAT, LON, duration, 60.0,
                       0.0, seed=seed, save_every=save_every, sigma_u=CALIBRATED_SIGMA_U)
    return run, ens


@pytest.fixture(scope="session")
def windows():
    run, ens = make_run()
    return build(ens, run, FORCING, ARRIVALS_S)


@pytest.fixture(scope="session")
def windows_dir(tmp_path_factory):
    """One location's January 2019: three weeks to train on, the last week held out for testing."""
    out = tmp_path_factory.mktemp("windows")
    for start in ("2019-01-01T17:00", "2019-01-08T17:00", "2019-01-15T17:00", "2019-01-29T17:00"):
        run, ens = make_run(start=start)
        save(build(ens, run, FORCING, ARRIVALS_S), out)
    return out
