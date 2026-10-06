# The search episode

`sar.search.episode` and `sar.search.scenario` (issue #47). This is the one piece of code
every searcher is scored by: the Expanding Square, greedy, a random walk, PPO, and a person
flying on the site. The reasons for each choice are in [ADR004](ADR004.md); this page is how
to use it.

## What it does, in one paragraph

An episode holds a scenario's particle cloud for the 45-minute search, one frame a minute
from the helicopter's arrival. Each particle carries a weight, its share of the probability.
Every minute the searcher says where to fly next: a heading, or timed waypoints for a
pattern's short legs. The helicopter flies it at 90 kt **relative to the datum marker**,
which drifts with the current. Every particle it comes within 92.6 m of loses its weight,
tested by closest approach while both move. The weight removed, summed, is **POS**, the
probability the search found the target.

## Run one

```bash
# a synthetic 2 km cloud; the Expanding Square, Sector Search, a fixed heading, or a replay
python -m sar.search.episode --spread-km 2 --particles 10000 --seed 1 --policy expanding-square
python -m sar.search.episode --spread-km 2 --particles 10000 --seed 1 --policy expanding-square --current 1.8 0
python -m sar.search.episode --policy heading --heading 90
python -m sar.search.episode --policy replay --replay flight.json

# a real scenario: the row's cloud re-run at its seed, the marker, and the buoy as target
python -m sar.search.episode --csv /home/26p67/data/derived/scenarios/scenarios.csv \
    --scenario S01 --forcing-dir /home/26p67/data --arrival-h 2 --particles 10000 \
    --policy expanding-square
```

A pattern's first leg runs along the target's drift at the datum unless `--first-bearing` is
given. That is the current plus 2 % of the wind on a real scenario, and `--current` on a
synthetic one. `--arrival-h` is hours from the call, in whole minutes. The window must end by
4 h, where σ is matched.

## What it prints

| Field | Meaning |
|---|---|
| `pos` | Probability removed over the window, on the raw weights (never renormalised) |
| `expected_ttd_s` | Σ t·Δm / Σ Δm: when, on average, the probability was found. Seconds since arrival, each sub-leg's mass credited at its end. `null` if nothing was found |
| `distance_m` | Path flown about the marker: 125,010 m for every searcher at 90 kt for 45 min. A check, not a comparison |
| `removed_per_step` | The 45 per-minute amounts; the RL reward is one of these |
| `remaining`, `initial_mass` | What is left, and what there was (1) |
| `target` | Scenario runs only: `found`, `found_s`, `closest_m`, `closest_s`, `gaps` against the real buoy, by the site's closest-approach rule |
| `datum`, `first_bearing_deg` | Where the marker was dropped (the cloud's centroid at arrival) and the first leg |

On a 2 km synthetic cloud (seed 1, N = 10⁴) the Expanding Square clears **59.9 %**, the
Sector Search **25.5 %**, and one straight heading **1.8 %**, all in 125,010 m of flight.

## Use it from code

```python
from sar.search.episode import SearchEpisode, run, pattern_policy, heading_policy, replay_policy
from sar.search.patterns import expanding_square
from sar.search.scenario import scenario_row, scenario_search

setup = scenario_search(scenario_row(csv, "S01"), "/home/26p67/data", 2 * 3600, 10_000)
episode = SearchEpisode(setup.window, setup.marker, target=setup.target)
metrics = run(pattern_policy(expanding_square(first_bearing_deg=setup.drift_bearing_deg)), episode)
```

A **searcher** is any function that takes the episode and returns a heading in degrees, or a
`Waypoints` for the coming minute. It can read:
- `episode.position` (the helicopter's lat, lon on the ground) and `episode.offset` (metres
  about the marker);
- `episode.particles()` and `episode.weight` (read-only arrays) and `episode.remaining`;
- `episode.t_s`, `episode.k` and `episode.done`.

`run(policy, episode, decide_every=n)` asks a heading searcher every n minutes and holds its
heading in between.

**For a Gymnasium environment:**
- `reset` builds a `SearchEpisode`.
- `step(action)` calls `episode.step(heading)` and returns what it removed as the reward.
- The observation is built from `episode.particles()`, `episode.weight` and the helicopter's
  state.

Nothing in `sar.search` imports Gymnasium, and a test holds that.

A **flight record** for `replay_policy` is JSON, in one of two forms:
- `{"headings_deg": [...]}`, one heading per minute;
- `{"t_s": [0, 12.5, ...], "heading_deg": [...]}`, each heading from that second on.

It replays headings, not positions, because a ground path already contains the current.

## The cloud every minute

The scenario runs save every 5 minutes, which is too seldom to sweep (ADR004 row 4).
`search_window` therefore runs the scenario again **at its own seed** and keeps every minute
of the window. Because the random pushes are drawn once per step whether saved or not, this
is the same cloud, and a test checks it bit for bit against a 5-minute run.

Memory is the whole run kept every minute: 39 MB at 10⁴ particles for 4 h, 386 MB at 10⁵.

## Cost

Measured on the laptop, 4 Oct 2026, steady synthetic cloud with a 1.8 m/s current:

| N | Expanding Square (88 sub-legs) | one heading a minute (45 sub-legs) |
|---|---|---|
| 10⁴ | 0.04 s | 0.02 s |
| 10⁵ | 0.62 s | 0.38 s |
| 10⁶ | 8.9 s | 4.9 s |

So a heading step costs about 8.5 ms at 10⁵ particles. At that rate 10⁶ training steps take
about 2.4 h; at N = 1,000 the cost is negligible. Nearly all the time is the sweep's
distance test over every particle. A box filter around each sub-leg would cut it, but it is
not built.
