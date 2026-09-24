# The doctrinal search (`sar.search`)

Closes #44 and #63, and covers the geometry of #48. Three modules:

- **The helicopter's numbers** (`platform`).
- **The two Coast Guard patterns** (`patterns`).
- **Where the search starts and where its marker drifts** (`datum`).

**Why each number and rule is what it is** is [ADR003](ADR003.md), cited to the USCG Addendum
(COMDTINST M16130.2F) by page.

## In one paragraph

The helicopter launches within 30 minutes of the call and flies out at 125 kt. It flies to the
**datum**, the last known position pushed on by the current and the target's leeway for as long as
that took. There it drops a **marker**, which drifts with the water current. Then it flies an
**Expanding Square** or a **Sector Search** at 90 kt *relative to the marker* for the 45-minute
window, seeing everything within 92.6 m of its track.

## Running it

```
python -m sar.search.platform                    # every constant, its source, one window
python -m sar.search.patterns expanding-square --first-bearing 45
python -m sar.search.patterns sector --first-bearing 45 --duration-min 18
python -m sar.search.datum --lat 26.5 --lon -79.0 --start 2019-06-01T06:00 \
    --elapsed-min 77 --constant-current 1.8 0.0 --constant-wind 5.0 5.0
```

```python
from sar.pipeline.forcing import ConstantForcing
from sar.search.datum import datum_at, marker_track
from sar.search.patterns import expanding_square, ground_track

forcing = ConstantForcing(current=(1.8, 0.0), wind=(5.0, 5.0))
lat, lon = datum_at(26.5, -79.0, "2019-06-01T06:00", elapsed_s=4620, forcing=forcing)
marker = marker_track(lat, lon, "2019-06-01T07:17", duration_s=2700, forcing=forcing)
pattern = expanding_square(first_bearing_deg=90.0)      # S = W, 90 kt, 45 min by default
t, glat, glon = ground_track(pattern, marker)           # every waypoint, plus every 10 s
```

## Conventions

| | |
|---|---|
| Pattern coordinates | metres **east and north of the marker**, waypoints with the time each is reached |
| Ground coordinates | `on_ground` = marker position at t + offset, with cos φ at the marker's latitude |
| Bearings | degrees true, clockwise from north, direction of travel (as `interpolate.speed_direction`) |
| Longitude | the store's 0 to 360 inside; `drift_track` accepts either and returns 0 to 360 |
| Time | seconds from the pattern's start (patterns) or from the marker drop (marker tracks) |
| A ragged duration | `drift_track` takes whole 60 s steps and then one short one |

## Guards

- A spacing, radius, speed, duration or `every_s` that is not a positive number raises
  `ValueError`.
- Asking a pattern or a marker track for a time outside it raises.
- A marker track with fewer than two strictly increasing times, or mismatched arrays, raises.
- `transit_time_s` refuses a datum beyond the H-60's 300 NM radius of action.
- `marker_track` without a forcing backend raises; a still marker is `MarkerTrack.fixed`.

## On the site (issue #64)

The **Search** preset flies the same doctrine against a real buoy. There are four steps:

1. Pick a buoy in the drifter list and move the clock to the moment it is "reported"; its
   position then is the last known position (LKP).
2. Place the base with a click.
3. Choose the pattern.
4. Choose the target type. That sets the leeway the datum is drifted with: *drogued drifter* is
   current only, *person in water* adds 2 % of the wind.

Then **Fly**. The helicopter waits 30 minutes, flies out at 125 kt to the datum, drops a marker,
and flies the pattern about the marker at 90 kt. The strip it sees is drawn at its true 185.2 m,
and the map allows zoom 13 while the view is open so it can be seen. The panel ends with *Found at
T+…* or the closest pass, and how far the datum was from where the buoy really was.

| Browser module | Mirrors | Held to Python by |
|---|---|---|
| `platform.js` | `sar.search.platform` | `tests/platform.test.js` |
| `patterns.js` | `sar.search.patterns` | `tests/patterns.test.js` |
| `pointDrift.js` | `sar.search.datum.drift_track` | `tests/patterns.test.js` (drift cases) |
| `searchRun.js` | the sequence above, and closest-approach detection | `tests/searchRun.test.js` (unit only) |

Two differences from the Python are deliberate, and both are stated in the code:

- **The site drifts on the nearest forcing frame**, hourly wind and 3-hourly current, rather than
  interpolating in time.
- **The datum is iterated against the transit time**, because the site plans from a base and
  Python is handed the elapsed time.

A buoy is the target, so detection is judged against where it **really** went (#64). Dropping a
person instead waits for the ensemble (#58), because the model's own drift of a person would only
be the model predicting itself.

## Not here, and deliberately

- **Sweeping particles** (#45) and **the episode runner** (#47). Both need the ensemble (#58).
  ADR003 §2–§3 says what they must do with a pattern: follow its sub-legs, and detect by
  closest approach.
- **The gridded forcing backend.** `datum` works with any object that has the `ConstantForcing`
  interface, so it gains real HYCOM and ERA5 when that backend exists.
- **Weather-corrected sweep width.** W is the calm-water figure (ADR003, Consequences).
- **Regenerating the browser fixture.** `python scripts/export_search_golden.py` rewrites
  `frontend/src/fixtures/search_golden.json` from this package.
