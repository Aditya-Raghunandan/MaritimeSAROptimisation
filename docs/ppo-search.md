# The PPO searcher

`sar.rl` trains a Stable-Baselines3 PPO policy to fly the 45-minute search, and flies it through the same referee as every other searcher (`sar.search.episode.SearchEpisode`, D029), so its score can be put beside the Expanding Square's, greedy's and the random floor's without any change of rules. The policy picks one compass heading a minute; the helicopter flies it at 90 kt about the drifting datum marker; every particle it passes within 92.6 m of is found.

This replaces the prototype on `feat/ppo-search`. That version read episode files made by a scenario table, scored with its own copy of the sweep, and paid the plain probability found. This one is built on the referee, the benchmark's searchers and the scenario bundles that arrived with #98, and it is trained on Monte Carlo runs at five fixed places.

## What it is trained on

### Five places

| Place | Position | Why it is there |
|---|---|---|
| Florida Current | 25.16 N, 80.06 W | The Gulf Stream's source in the Straits of Florida. The channel holds it in place, so this point is in the jet every week of the five years. |
| Gulf Stream off Jacksonville | 30.16 N, 79.82 W | The stream along the shelf break, still steered by the slope, so again in the jet nearly every week. |
| Bahamas, east of Eleuthera | 25.46 N, 75.60 W | Deep water off the islands on the Antilles Current side: weaker, more variable flow. |
| Ring zone | 34.86 N, 71.70 W | South of the stream after it leaves the coast at Cape Hatteras, where its meanders and cold-core rings pass: water that turns. |
| Sargasso eddy field | 29.07 N, 68.88 W | Open-ocean mesoscale eddies: slow, curving flow. |

Two are in the Gulf Stream and three are in eddying water, one of them in the Bahamas. Each is the start of a development-set drifter in `scripts/scenario_buoys.txt` (rows 51, 3, 52, 16 and 18), so each lies at least 20 km from HYCOM land and 50 km inside the forcing box (`scripts/pick_scenario_buoys.py`). The list is `scripts/rl_locations.txt`, one `lat;lon;name` per line, and array task K reads line K.

### One Monte Carlo a week for five years

Every 7 days from 1 January 2019 to 31 December 2023, at 17:00 UTC, each place gets one Monte Carlo: 10,000 particles released at the position with no spread (D026), 60 s steps (D009), 4 hours, saved every 5 minutes, all with the same seed, 20261008. That is 261 runs per place and 1,305 in all. The five years are the whole years inside HYCOM's coverage, 4 December 2018 to 5 September 2024 (`docs/current-fetch.md`).

The random term is the engine's default, the random velocity with memory (D030, σ_u = 0.226 m/s, T_L = 25.7 h), not the σ = 26.3 random walk the earlier script passed; that flag is now kept only for comparisons. One seed for every run means every run draws the same sequence of random pushes, but each week's currents and winds differ, so the clouds differ.

### Three transit times from each run

The helicopter's transit is the time from the call to arriving on scene: 30, 45 and 60 minutes. Each one cuts a different 45-minute window out of the same run, from 30 to 75, 45 to 90 and 60 to 105 minutes after the call, so every Monte Carlo gives three training searches. For each window the datum is the cloud's centroid at arrival (D004) and the marker drifts from there with the current alone (`sar.search.datum`), exactly as `sar.search.scenario` lays out the benchmark's searches.

The referee needs the cloud every 60 s. The run is saved every 5 minutes, so the minutes between saves are straight lines between them. Under the random velocity that is close: against the same run saved every minute, it is 0.84 m off rms, 2.1 m at the 99th percentile and 4.3 m at worst, against a 92.6 m half-strip (`tests/rl/test_windows.py`, measured on 2,000 particles at the Florida Current with a 1.5 m/s current).

### Test: the last week of every month

| Split | Weeks | Windows | Use |
|---|---|---|---|
| Train | every other week | about 2,625 | PPO learns from these |
| Validation | every 8th week, unless it is a test week | about 390 | picks `best_model.zip` |
| Test | the last 7 days of every month, all five years | about 900 | the reported score, flown once at the end |

The test set is the run whose call falls in the last 7 days of its month (`sar.rl.windows.is_test`). The runs are 7 days apart, so each month has exactly one, 12 a year per place. Taking one from every month of every year puts every season and every year's weather into the test score. Holding out a whole year would only test that one year.

Validation counts Monday-to-Sunday weeks from 5 January 1970 and keeps every 8th that is not already a test week. `train.py` never opens a test file. It only counts them into `config.json`.

## What the policy sees, does and is paid

**It sees 1,289 numbers.** A 32 by 32 map of the probability still unfound, in 1 km cells about the marker (R5a), and a 16 by 16 map in 250 m cells about the helicopter (D007's cell, the one greedy reads), each scaled so its fullest cell reads 1. Then nine numbers: the helicopter's offset from the marker, the last known position from the helicopter (both in tens of km), the share of the window left, the probability left, the transit time in hours, and the sine and cosine of the target's drift at the datum. All of it is read from the referee's own episode (`sar.rl.features.observe`), so training and flying see the same thing.

**It does one thing a minute:** pick one of `--headings` compass headings, evenly spaced from north. 36 is greedy's frozen setting (D031); with 8, legs on a 45-degree lattice cannot lie side by side.

**It is paid particles found times time remaining.** If $n_k$ of the $N$ particles are found in minute $k$ (counting from 0), the reward for that minute is

$$r_k = \frac{n_k}{N} \cdot \frac{45 - k}{45}$$

which is the number found times the minutes left, divided by the constant $N \times 45$ so that a perfect first minute pays 1. Writing $m_k = n_k / N$ for the probability found in minute $k$, the whole search's return is

$$R = \sum_{k=0}^{44} m_k \frac{45 - k}{45} = \mathrm{POS} \left(1 - \frac{\bar k}{45}\right), \qquad \bar k = \frac{\sum_k k \, m_k}{\mathrm{POS}}$$

so it rewards finding more and finding it sooner: two searches with the same POS are ranked by when, on average, they found it. POS itself is still what the referee reports and what the comparison uses.

## Running it

All three stages run on the cluster from the repository checkout. The first needs the forcing files; the other two need only the windows.

**1. The Monte Carlo runs and their windows.** One array task per place:

```
sbatch scripts/training_montes.sh
```

Every setting has a default that can be overridden from the environment: `LOCATIONS`, `FIRST`, `LAST`, `N`, `SEED`, `ARRIVALS` (default `30 45 60`), `OUT` (default `$DATA/derived/rl`) and `KEEP_CSV`. For each week it runs `python -m sar.pipeline.ensemble` unless that run's CSV already exists, then

```
python -m sar.rl.windows --run <the run's CSV> --forcing-dir /home/26p67/data --arrival-min 30 45 60 --out $OUT/windows
```

which reads the CSV and its JSON sidecar, drifts the three markers on the real forcing and writes the windows file. A week whose forcing has a gap (D011) prints a line and the loop moves on. Rerunning the job skips what is already there.

**2. Training.**

```
sbatch scripts/train_ppo.sbatch --windows /home/26p67/data/derived/rl/windows --validation-every 8 --arrival-min 30 45 60 --headings 36 --timesteps 3000000 --envs 4 --seed 1 --eval-every 100000 --out /home/26p67/data/derived/rl/runs
```

Every argument is required, because each one says what the run is. `--envs` should match the job's `--cpus-per-task`. PPO's own settings are Stable-Baselines3 2.9's defaults.

**3. The test weeks, and the site.**

```
python -m sar.rl.fly windows --windows /home/26p67/data/derived/rl/windows --arrival-min 30 45 60 --searchers ppo expanding-square greedy random --model <run>/best_model.zip --out <run>/test
python -m sar.rl.fly bundle --model <run>/best_model.zip --bundle /home/26p67/data/published/scenarios/v1/*/rv_*.json
```

The first flies every searcher over every test window and prints the summary. The second adds a PPO flight to published scenario windows in place.

## What it writes

| Stage | File | Holds |
|---|---|---|
| 1 | `$OUT/montes/derived/ensemble_2516N08006W_20190101T1700_N10000_dt60s_T4h_seed20261008.csv` and `.json` | the run, as `sar.pipeline.ensemble` always writes it; about 40 MB a run, 52 GB for all 1,305 |
| 1 | `$OUT/windows/windows_2516N08006W_20190101T1700_N10000_dt60s_T4h_seed20261008_arr30-45-60m.npz` | the saved frames from the first arrival to the last window's end, the LKP, the seed, and per transit the marker every minute and the drift bearing; 2.8 MB, 3.7 GB for all |
| 2 | `<out>/ppo_h36_seed1_T3000000/` | `config.json` (every argument, the commit, the observation, the reward, the training and validation file names), `model.zip`, `best_model.zip`, `monitor/`, `eval/evaluations.npz` |
| 3 | `<out>/flights.csv`, `summary.csv` | one row per flight (window, place, start, transit, searcher, POS, time-weighted return, expected time to detection), and the means per transit and searcher with PPO's paired difference from the Expanding Square and its win rate |
| 3 | `<out>/flights/<windows name>_<transit>m.json` | one window: marker, LKP, datum, drift bearing and every searcher's flight |
| 3 | `flights.ppo` inside a scenario bundle's `<S01>/rv_2h.json` | the PPO flight beside the others |

A windows name is the run's name with `ensemble_` replaced by `windows_` and the transit times added, so `sar.rl.windows.parse_windows_name` reads every run parameter back out of it, and the training split reads the date from it without opening the file. Set `KEEP_CSV=0` to delete each CSV once its windows file is written; the windows hold everything training needs.

## What reads it

**The site.** Every flight is written in the format `scripts/export_scenario_bundles.py` writes for the other searchers: `layout`, `first_bearing_deg`, `steps` (45 lists of `t_s`, `east_m`, `north_m`, the waypoints the referee was given) and `python` (`pos`, `drain_rate`, `expected_ttd_s`, `found`, `found_s`, `closest_m`, `closest_s`). The scenario page (`frontend/src/bench.js`) replays exactly that through `frontend/src/referee.js`. To show PPO there, add a `ppo` entry to `SEARCHER_INFO` in `frontend/src/scenarioData.js` and remove the "ML agent (coming)" placeholder in `bench.js`, once every published bundle has a `ppo` flight: the chart reads `flights[key]` for every searcher it lists.

The published bundles' arrivals are 1, 2 and 3 hours, and this policy trains on 30 to 60 minutes, so on those windows it is working outside its training range: its transit input reads 1 to 3 where it has only seen 0.5 to 1. Train with the bundle arrivals too before reading much into those numbers.

**The paper.** `summary.csv` from the test windows is the result: mean POS per searcher and transit, and PPO against the Expanding Square window by window.

`tests/rl/test_train_fly.py` holds the format to the published fixture's, and rebuilds a published window from its `.f32` cloud to the paper's own Expanding Square and greedy scores to 1e-6.

## How long it takes

Measured on 9 October 2026 on the development laptop (WSL, 10,000 particles), and scaled to a compute node by 1.4, the ratio of the two machines' forcing step at 10,000 particles in `docs/gridded-forcing.md` (2.8 ms against 2.1 ms).

| Stage | Laptop | Node, estimated |
|---|---|---|
| One week's Monte Carlo and windows | about 12 s (0.4 s engine with steady forcing, 3.8 s writing the CSV, 7.2 s reading it back, 0.1 s windows) plus opening the forcing twice | about 20 to 25 s |
| Stage 1, 261 weeks per place, five places in parallel | | about 1.5 to 2 h |
| One training step, 4 environments, through `train.py` | 2.0 ms (400,000 steps in 781 s) | about 2.7 ms |
| One validation pass, 390 windows | about 50 s | about 70 s |

Training time is then the budget times 2.7 ms, plus one validation pass every `--eval-every` steps:

| `--timesteps` | Passes over the training windows | Node, 4 environments |
|---|---|---|
| 1,000,000 | 8 | about 1 h |
| 3,000,000 | 25 | about 3 h |
| 10,000,000 | 85 | about 10 h |

Three seeds run as three jobs at once, so they take the same wall time. Where the time goes: in a training step, about 2.4 ms is the referee's sweep and the observation over 10,000 particles, and well under 1 ms is PPO itself.
