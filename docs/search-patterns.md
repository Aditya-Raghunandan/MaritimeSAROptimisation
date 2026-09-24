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

## On the site (issues #64, #68, #69, #71)

The **Search** preset flies the same doctrine against a real buoy. The panel, top-left, has two
tabs: **Coast Guard search** and **Fly it yourself**. It is built to fit a 1366 × 768 screen
without scrolling, and an e2e test fails if it does not.
- A finished step shrinks to one line with a small *change* or *move* link.
- Once a search flies, the four steps fold into a one-line summary with *Change the set-up*.

The first tab asks the Coast Guard's questions in order:

1. **Who is missing?** Pick a buoy in the Drifters list and move the clock to when it is reported
   missing. Its position then is the last known position (LKP). Once it is chosen, the list gets
   out of the way; *Choose another buoy* brings it back.
2. **Where does the helicopter start?** Place the base with a click.
3. **How does the Coast Guard search?** Expanding Square or Sector Search.
4. **Where will it be when they arrive?** Either *with the current only* (a drifter buoy) or
   *current + 2 % of the wind* (a person). **This sets the
   datum**, where the helicopter is sent and which way the first leg runs. It does not give the
   marker any wind: the marker always drifts with the current alone, and the real buoy goes
   where it really went.

Then **Fly the search**:
1. The helicopter waits 30 minutes, then flies out at 125 kt to the datum.
2. It drops a marker there.
3. It flies the pattern about the marker at 90 kt.

The map labels every mark: *last known position*, *datum: where drift predicts it*, *marker* and
*real buoy*. It draws three lines that tell the story:
- the **predicted drift**, white dots from the last known position to the datum, labelled with how
  long it covers;
- **where the buoy really went**, a pink line that grows as the search plays;
- the **datum error**, dotted, from the datum to where the buoy really was on arrival. The marker is dropped at the datum, not on the buoy, because the Coast Guard does
not know where the buoy is. That gap is the drift model's error, and it is what this project
measures.

The strip the helicopter sees is drawn at its true 185.2 m. The map allows zoom 13 while the view
is open so the strip can be seen.

**One clock.** While a search is loaded it owns the time bar at the bottom:
- the slider scrubs it, and Play or Space plays and pauses it;
- the label gives the time since the call, the phase and the playback rate (e.g. "45 s per
  second");
- a band under the slider marks *call to launch*, *flying out* and *on scene*;
- the site clock follows the search moment, so the header time, the wind and current fields, and
  the buoy all show the same instant.

Reset, or leaving the view, hands the bar back at the moment the search had reached. Before this
(#69), the search ran a second clock of its own, and the buoy showed in two places.

**Fly it yourself** (#68). *Spawn a helicopter on the map*, click where it should appear, and
steer with **W A S D** or the **arrow keys**; two keys fly a diagonal.
- It takes off on the first key and keeps its heading when the keys are released.
- It flies 90 kt with the same 185.2 m strip, for one 45-minute window, and the map follows it.
- If a buoy is chosen, passing within 92.6 m of it finds it, by the same closest-approach test the
  patterns get.
- A finished flight can be scrubbed and replayed. A flight still in the air cannot be rewound.

| Browser module | Mirrors | Held to Python by |
|---|---|---|
| `platform.js` | `sar.search.platform` | `tests/platform.test.js` |
| `patterns.js` | `sar.search.patterns` | `tests/patterns.test.js` |
| `pointDrift.js` | `sar.search.datum.drift_track` | `tests/patterns.test.js` (drift cases) |
| `searchRun.js` | the sequence above, closest-approach detection, and `ManualFlight` for flying by hand | `tests/searchRun.test.js` (unit only) |

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
