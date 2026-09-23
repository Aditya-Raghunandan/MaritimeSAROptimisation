# The sealed drifter validation set

`sar.validate.split` decides which drifter tracks the drift engine may be debugged on and which
are kept back for the reported results (issue #51). It exists because of one sentence in vault
D018: *"There is no excuse for reporting on tracks that were used while debugging."* A track
anyone has looked at while tuning the engine is no longer an independent test of it, so the
split is drawn **before the first engine run** and recorded, with its seed, in the vault.

## Running it

```
python -m sar.validate.split --data C:/maritime-data
```

It reads both drifter products and their buoy tables from `<data>/raw/` (see
[drifter-fetch.md](drifter-fetch.md)) and writes `<data>/derived/validation_split.csv`, one row
per validation unit. It takes about 20 s. The same data and the same seed always give the same
split.

| Column | Meaning |
|---|---|
| `unit` | `ID@start`: the buoy and the first fix of the unit, stable across re-pulls |
| `ID`, `product` | buoy, and `hourly` or `6-hourly` |
| `start`, `end`, `hours`, `fixes` | the unit's extent; `hours` is duration, never fix count |
| `lat0`, `lon0` | the first fix, where a hindcast would seed its ensemble |
| `tier` | `drogued`, `undrogued` or `uncertain` (the buoy's drogue records disagree) |
| `group` | the shared-water group the buoy belongs to |
| `split` | `dev`, `sealed` or `holdout-2023` |

## The three splits

| Split | What it is | Use |
|---|---|---|
| `dev` | hourly units, 2019 – Oct 2022 | debug and tune on these freely |
| `sealed` | hourly units, 2019 – Oct 2022 | reported only, never debugged on |
| `holdout-2023` | every 6-hourly unit, Nov 2022 – 2023 | a year never developed on, reported separately |

## How units and groups are made

**A unit** is a contiguous in-box run of **48 h or more, measured as duration**: 48 hourly fixes
span 47 h, and a 47 h track cannot verify a 48 h forecast. It has **one drogue tier**. A run that
loses its drogue part-way is cut at the loss, because D018 compares drogued with undrogued and a
mixed run is neither.

**A group** is a set of buoys joined by any chain of **shared water**: two buoys within 10 km of
each other at the same timestamp. 10 km is about one current-grid cell diagonal (√(8² + 4.5²) ≈
9.2 km), so two buoys that close are driven by the same current values, and debugging on one is
debugging on the other. A whole group goes to `dev` or to `sealed`, never both.

**The split** shuffles the groups with the seed, then sends each to `sealed` while its main tier
has fewer than 25 % of that tier's units sealed. That keeps both tiers represented rather than
letting the larger one fill the sealed set.

`check_split` then refuses any split where a group straddles dev and sealed, where fewer than 10
units are sealed (R6d), where either tier is missing from the sealed set, or where a 2023 unit is
outside the holdout.

## Measured on 2026-09-23 (seed 20260923)

| | drogued | undrogued | uncertain | units | buoys |
|---|---|---|---|---|---|
| dev | 122 | 329 | 9 | 460 | 181 |
| sealed | 60 | 122 | 12 | 194 | 74 |
| holdout-2023 | 38 | 108 | 0 | 146 | 94 |
| all | 220 | 559 | 21 | 800 | 314 |

255 shared-water groups, 28 with more than one buoy, the largest 18. Because groups move
whole, the sealed share lands near the 25 % target rather than on it: **33 % of drogued units and
27 % of undrogued**. The buoy columns overlap by 35, because a buoy can have hourly units in 2019–22
and 6-hourly units in 2023; the one-week buffer between the products keeps those apart in time.
