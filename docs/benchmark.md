# The benchmark: how every searcher is scored

`scripts/score_scenarios.py` (the scorer), `scripts/pick_scenario_buoys.py` (the scenario
tables), `sar.search.greedy` (greedy and the random floor), `sar.search.scenario.
doctrinal_searcher` (how each Coast Guard pattern is laid out). Issues #48, #49.

## In one paragraph

A real drifting buoy is the person in the water. From the buoy's real position and time, the
drift engine releases 10,000 possible "versions" of the person (particles) and drifts them
with that day's real current and wind. A helicopter arrives 1, 2 or 3 hours after the call
and searches for 45 minutes. Every particle that passes within 92.6 m of its track (half the
185 m strip a helicopter crew can see) counts as found. Each particle is worth 1/10,000 of
the probability, so the share found is the search's score. Every searcher flies the same
cloud, so a difference in score can only be the searcher.

## The words

| Word | What it means | How to picture it |
|---|---|---|
| **POS** (probability of success) | The share of all the places the person could be that the helicopter looked at. 0.53 means 53 %. | The part of the probability map the helicopter has drained |
| **Drain rate** | How fast a searcher empties the map: the share of the probability it clears in one minute, in % per minute. POS is the drain rate added up over the 45 minutes. | The live curve under the map. A good start drains fast early. Our own term; "sweep rate" is taken (it means strip width × speed) |
| **Found** | Whether the helicopter passed within 92.6 m of the *real* buoy. | One coin toss per scenario. POS is the chance of a find; "found" is whether it came up |
| **Time to detection** | When, on average, the probability was found, in minutes after arrival. | Earlier is better |
| **Group** | Buoys that drifted within 10 km of each other at the same time. They ride the same water, so they count as one. | See below |
| **Straightness** | How far the buoy got in 4 hours divided by how far it actually travelled. | See below |

### Why buoys are grouped, and what it changes

Buoys are often released together. The largest group in our data is 18 buoys that rode the
same patch of sea. If each counted on its own, that one patch of sea would get 18 votes and
a buoy released alone would get one. So **every number is averaged within each group first,
and then across groups**. The same goes for a long-lived buoy, which gives many scenarios:
they are averaged inside its group before they count.

The range given with each number (the 95 % interval) comes from reshuffling **whole groups**
1,000 times and recomputing the average each time. It is where the average lands in 95 of
100 reshuffles. Reshuffling single scenarios instead would pretend that 18 buoys in one
patch are 18 independent tests, and give a range that is too narrow.

### Straightness, with shapes you can picture

| Straightness | What the buoy's 4 hours looked like |
|---|---|
| **1.00** | a straight line |
| **0.90** | a gentle bend, like a quarter of a circle |
| **0.71** | a sharp right-angle turn halfway |
| **0.64** | a U-shaped half circle |
| **0** | it came back to where it started |

No buoy is removed for turning. A turn is either a data error (a GPS glitch, a buoy on a
ship's deck, a buoy aground), which GDP's quality control and the scenario rules already
remove, or real ocean (an eddy, a front, a wind shift), which is exactly what a person in
the water goes through. Instead every result is also reported for **straight against
turning** buoys, split at the median straightness of the dev buoys.

## The searchers

| Searcher | What it does |
|---|---|
| **Expanding Square** | The Coast Guard's square spiral outward from the datum, legs one strip apart, the first leg along the drift. The baseline every other searcher is compared with (D004) |
| **Sector** | The Coast Guard's "clover" of passes through the datum |
| **Parallel Track** | Back-and-forth legs one strip apart over a square centred on the datum, laid along the datum line (start to predicted position) |
| **Trackline** | Up one side of the datum line and down the other, widening each pass |
| **Greedy** | No pattern: it reads the probability map and flies wherever the most probability lies just ahead. The control a learning searcher must beat (D015) |
| **Random** | A random heading every minute. The floor: anything worth reporting beats it |

All are flown about the datum marker, which drifts with the current, as the Coast Guard flies
its patterns (ADR003). The patterns are laid out by one function,
`sar.search.scenario.doctrinal_searcher`, the same rule the site uses.

### Greedy's settings

At one decision a minute a heading is a 2.78 km straight leg. With eight headings greedy
could not lay its legs side by side and kept re-flying one line: 0.37 on S01 at 1 h against
the Expanding Square's 0.83. More headings, or deciding more often than once a minute,
fixed it (S01: 16 headings 0.78; 8 headings every 10 s 0.84). The settings were chosen on the
55 dev scenarios (7 Oct, jobs 59002–59007), group-first POS with memory:

| Greedy | 1 h | 2 h | 3 h | Mean |
|---|---|---|---|---|
| 8 headings, decide every 60 s | 0.581 | 0.397 | 0.279 | 0.419 |
| 16 headings, 60 s | 0.773 | 0.472 | 0.303 | 0.516 |
| **36 headings, 60 s (the default)** | 0.792 | **0.491** | **0.319** | 0.534 |
| 8 headings, every 20 s | 0.805 | 0.482 | 0.303 | 0.530 |
| 8 headings, every 10 s | 0.815 | 0.480 | 0.283 | 0.526 |
| 16 headings, every 10 s | **0.821** | 0.489 | 0.302 | **0.538** |

**36 headings, one decision a minute**: within 0.4 points of the best on average, the best at
2 h and 3 h (and under the old noise at every arrival), and it keeps the referee's own step,
one heading a minute. So a learning searcher with the same action differs from it only by
learning (D023). It is frozen with the rest before any test set is opened.

## The scenarios

| Table | Rows | What it is for |
|---|---|---|
| `scenarios.csv` | 55 | One start per group, stratified by water speed (15 jet, 20 moderate, 15 quiet, 5 spare). The site, and the ML training rows |
| `bench_dev_48h.csv` | 9,538 | Every dev undrogued buoy, a new start every 48 h until its track ends: 139 buoys, 107 groups. The development benchmark |
| sealed | not built yet | The test. Opened once, after the freeze list below |
| holdout-2023 | not built yet | POS only, in a year nobody developed on. GDP's hourly product ends on 31 Oct 2022, so 2023 has a position only every 6 h: enough to start a cloud, not to say where the buoy was during the search |

**Why 48 h between starts:** the drift model's error lasts about a day (D030, 25.7 h). Two
starts 48 h apart share about 15 % of it, two starts 4 h apart about 86 %. Grouping handles
what remains.

The dev table is mostly quiet water (5,666 quiet, 3,774 moderate, 98 jet rows): buoys cross
the Gulf Stream in hours and spend days outside it. The 55 oversample the jet on purpose.

## Run it

```bash
python scripts/pick_scenario_buoys.py --data DATA --land DATA/raw/hycom_<box>_<dates>.nc \
    --every 48h --out bench_dev_48h.csv
python scripts/score_scenarios.py score --csv bench_dev_48h.csv --forcing-dir DATA \
    --rows 1-9538 --noise rv rw --arrival-h 1 2 3 --out RESULTS/bench_dev_48h_N10000
python scripts/score_scenarios.py summary RESULTS/bench_dev_48h_N10000
# on the cluster, 4 processes per node, 25 rows each:
CSV=.../bench_dev_48h.csv PER=25 sbatch --array=1-96 scripts/score_scenarios.sbatch
```

One folder is one experiment. `score` writes `manifest.json` the first time (the code's
commit, the table's sha256, N, the noise settings, arrivals, searchers, greedy's settings)
and refuses to add flights made with anything different. Flights go to
`flights/<scenario>.jsonl`, one line per flight.

`summary` writes `flights.parquet` (every flight) and `summary.csv`:

| Column | Meaning |
|---|---|
| `measure` | `pos`, `found`, `ttd_min`, `pos_15m`, or `pos difference` |
| `slice` | `all`; `water: quiet / moderate / jet`; `path: straight / turning` |
| `noise` | `rv` (with memory, D030), `rw` (without, D028), or `rv - rw` |
| `searcher` | a searcher, or `X - expanding-square` |
| `n_rows`, `n_groups` | scenarios and groups behind the number |
| `mean`, `lo`, `hi` | the group-first average and its 95 % interval |
| `groups_better` | for differences: in how many groups the first beat the second |

## Before the sealed set is opened

The sealed buoys are the test. They are opened **once**, after these are written down and
dated, and are not changed after:

- the drift model's pass thresholds (D025);
- the noise: σ_u = 0.226 m/s, T_L = 25.7 h (D030);
- N = 10,000, and arrivals at 1, 2 and 3 h;
- each pattern's layout rule, and greedy's settings;
- the scorer, and the straightness split (the dev median).

A bug fix after opening is allowed and logged. Retuning is not.
