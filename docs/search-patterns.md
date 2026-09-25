# The doctrinal search (`sar.search`)

Closes #44, #63 and #75, and covers the geometry of #48. Three modules:

- **The helicopter's numbers** (`platform`).
- **The four Coast Guard patterns** (`patterns`): Expanding Square, Sector Search, Parallel Track
  and Trackline Return.
- **Where the search starts and where its marker drifts** (`datum`).

**Why each number and rule is what it is** is [ADR003](ADR003.md), cited to the USCG Addendum
(COMDTINST M16130.2F) by page.

## In one paragraph

The helicopter launches within 30 minutes of the call and flies out at 125 kt. It flies to the
**datum**, the last known position pushed on by the current and the target's leeway for as long as
that took. There it drops a **marker**, which drifts with the water current. Then it flies an
**Expanding Square**, a **Sector Search**, a **Parallel Track** or a **Trackline** at 90 kt
*relative to the marker* for the 45-minute window, seeing everything within 92.6 m of its track.
The last two are laid out by the drift model's prediction: along the line from the last known
position to the datum (ADR003 rows 12–14).

## Running it

```
python -m sar.search.platform                    # every constant, its source, one window
python -m sar.search.patterns expanding-square --first-bearing 45
python -m sar.search.patterns sector --first-bearing 45 --duration-min 18
python -m sar.search.patterns parallel --first-bearing 30              # the Z = W x V x T square
python -m sar.search.patterns trackline --first-bearing 30 --half-length-m 6000
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

- A spacing, radius, length, width, half-length, speed, duration or `every_s` that is not a
  positive number raises `ValueError`.
- A Parallel Track area narrower or shorter than one track spacing raises.
- Asking a pattern or a marker track for a time outside it raises.
- A marker track with fewer than two strictly increasing times, or mismatched arrays, raises.
- `transit_time_s` refuses a datum beyond the H-60's 300 NM radius of action.
- `marker_track` without a forcing backend raises; a still marker is `MarkerTrack.fixed`.

## On the site (issues #64, #68, #69, #71, #73, #75, #76, #79, #80, #81, #83)

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
3. **How does the Coast Guard search?** Square, Sector, Parallel or Trackline (#75). The last
   two are laid out by the prediction:
   - the **Trackline** runs along the predicted drift, from the last known position, through the
     datum, and as far again beyond it;
   - the **Parallel Track** covers the square one window can search (Z = W × V × T, 4.81 km a
     side), centred on the datum, with its tracks along the predicted drift.
4. **Where will it be when they arrive?** Either *with the current only* (a drifter buoy) or
   *current + 2 % of the wind* (a person). **This sets the
   datum**, where the helicopter is sent and which way the first leg runs. It does not give the
   marker any wind: the marker always drifts with the current alone, and the real buoy goes
   where it really went.

Then **Fly the search**:
1. The helicopter waits 30 minutes, then flies out at 125 kt to the datum.
2. It drops a marker there.
3. It flies the pattern about the marker at 90 kt.

The map labels every mark with a small chip: *last known position*, *datum · where drift predicts
it*, *marker* and *real buoy*. The rest of the pattern is drawn faint ahead of the helicopter, and
a Parallel Track's area is outlined (#75). It draws three lines that tell the story:
- the **predicted drift**, white dots from the last known position to the datum, labelled with how
  long it covers;
- **where the buoy really went**, a pink line that grows as the search plays;
- the **datum error**, dotted, from the datum to where the buoy really was on arrival; the datum's
  label then says how far that was. The marker is dropped at the datum, not on the buoy, because
  the Coast Guard does not know where the buoy is. That gap is the drift model's error, and it is what this project
measures.

The strip the helicopter sees is drawn at its true 185.2 m. The map allows zoom 13 while the view
is open so the strip can be seen.

**One clock.** While a search is loaded it owns the time bar at the bottom:
- the slider scrubs it, and Play or Space plays and pauses it;
- the label gives the time since the call as T+h:mm:ss (its tooltip says so), the phase and
  the playback rate (e.g. "45 s per second"); the panel says the same time in words;
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

**One control** (#73). One button runs a search's whole life: *Fly the search → Pause → Resume →
Replay*. In the Search view, the time bar's ▶ and Space do exactly the same, and they never play
the site's hours there. *Change the set-up* clears the loaded search, because a new set-up is a new
search.

**The compass** (#73; bottom right above the legends since #76, while a search or flight is
loaded) shows:
- north;
- the helicopter's heading (orange);
- the current (cyan) and the wind (amber) where the helicopter is, each pointing the way it is
  going.

Its readout gives the speeds, where the wind comes from (in m/s and knots, since #81), and the
Beaufort force. Above 15 kt it says, in plain words, that whitecaps hide a person, so the Coast
Guard assumes spotters see half as far to each side, and that this view does not, so finding is
easier here than it would really be (limitation L19). The Addendum citation (Table H-10: ×0.5 over
15 kt or 3 ft seas, ×0.25 over 25 kt) is the caveat's tooltip.

**Close up** (#79, replacing #73's texture). Flying yourself follows the helicopter at zoom 15 by
default; the Search view allows zoom 16, where the strip is about 90 px wide. From zoom 13 the view
fades into a close-up of its own, fully on from 13.5 (`closeUp.js` has the rules, `closeUpLayer.js`
draws them).

*Why not arrows or streaks close up.* At the 1 km scale bar the screen is about 17 km wide: two
current cells (0.08° × 0.04°, about 8 × 4.5 km) and less than one wind cell. The forcing is the
same everywhere on screen, so arrows would all run parallel. What differs close up is how three
things move, and the view shows those:

| What | Moves with | Clock |
|---|---|---|
| the whole textured surface, and its whitecaps | the real current | search time |
| waves and whitecaps | set by the wind: roughness peaking at the Pierson–Moskowitz wavelength, whitecap cover from Monahan & O'Muircheartaigh (1980), about 1 % of the sea at 10 m/s | real time |
| Sargassum, in windrows along the wind (Langmuir cells), from about zoom 15.5 | current + 2 % of wind: the drift model's own step (`floaterStep`) | search time |
| now and then an animal: flying fish, dolphins, a turtle, a humpback (December to April only) | its own swimming, carried by the water | real time |

- **The sea** (#83) is ten octaves of gradient noise, 2 m to 1 km, each drawn only while it is a
  few pixels to a screen long, so there is texture at every zoom and no shimmer. Octaves are
  stretched along their crests and travel with the wind at the deep-water speed of a wave that
  long. Faces towards the light are teal, faces away deep blue, with sun glitter in patches.
  The first version faded out every wave under a few pixels and went flat and dark at the
  zooms a search is watched at.
- **Light** follows the real sun at that moment and place. At night it is the same sea, a
  little dimmer and cooler, with a faint moon glitter; the close-up key says the sweep width is
  a daylight figure.
- **The page around it** changes while close up: the basemap goes to satellite (real near a
  coast; offshore the drawn sea covers it), the colour rasters hide (one flat value at that
  scale), and the *Close up* key replaces the Surface current key. Zooming out restores the
  basemap the viewer had.
- **Near land** the sea thins out within a current-model cell of the coast, where the
  satellite photo is the real thing.
- **The swept strip** stays at true width but is fainter close up, so the sea shows through,
  and the faint dashed plan shows only the part of the pattern still to fly (#83).
- **The key** is at most four short rows: what moves the sea, what the golden weed is, whether
  it is night, and that none of it is data.
- **Speed.** The sea is one WebGL pass at 60 % of the screen's pixels; weed and animals are a
  2-D canvas. Measured 25 Sep at 1440×900 on the development laptop (#83's sea): one whole
  close-up frame, forced to finish on the GPU (`readPixels`), takes a median 0.6–0.9 ms and at
  worst 1.7 ms at zooms 13.75–16, against a 16 ms frame. Where WebGL is missing, #73's 2-D
  texture stands in.

Waves, weed and animals are drawn for the eye and never feed detection; animals are not to
scale, like the helicopter icon. `window.__closeUp.summon('dolphins')` calls one up in the
development build.

**Why it left the prediction** (#80, `whyMissed.js`). After a search the result says why the buoy
did not go where the model said. The buoy's own motion (from its positions) is compared with the
model's (current + the target's share of wind) at the same places and times, along the buoy's
real path. The difference is what the model is missing, and its direction against the wind is the
clue:

| The gap | Verdict |
|---|---|
| under 0.03 m/s | the model had it about right |
| along the wind, downwind | the wind pushed harder than the model allows; the windage that would have fitted is given |
| along the wind, upwind, drogued buoy, wind in the model | a drogued buoy barely feels the wind: *current only* fits it better |
| across the wind, or the wind under 2 m/s | the water moved differently from the ocean model: an eddy smaller than a cell, or out of place |

The panel shows three lines (buoy, model, missing) and the cause; the summary's tooltip says what
it cannot see: the waves' own push on anything floating, tides, and the difference between the
surface current and the current 15 m down. It is a clue, not a proof. (#80 also drew three
arrows at the buoy close up; #83 removed them. They were not read as intended, and the buoy's
pink path against the white predicted drift already shows the same thing on the map.)

**The drogue** (#73, reworded in #76). When a buoy is chosen, step 4 says whether it still had
its drogue at the report time, and what a drogue is: the underwater sail, 15 m down, that keeps a
buoy with the water. A drogued buoy follows the water, so *current only* fits it. An undrogued one
feels the wind too, but less than a person.

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
