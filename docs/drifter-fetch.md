# Drifter fetch and load

`sar.fetch.drifters` pulls NOAA Global Drifter Program (GDP) tracks for the study box and
window; `sar.utils.data_io.load_drifters` reads them into the project's conventions. Issue
#55 added the quality fields, the per-buoy metadata and the 6-hourly product. Every number
here was measured against the server on **2026-09-23** unless it says otherwise.

These drifters are the **validation set** (vault D018, requirement R2): real trajectories the
drift engine is scored against. They are not a model output and nothing here is invented.

## Running it

```
python -m sar.fetch.drifters --product hourly   --start 2019-01-01 --end 2024-01-01 --out C:/maritime-data
python -m sar.fetch.drifters --product 6-hourly --start 2022-11-01 --end 2024-01-01 --out C:/maritime-data
```

`--out` is required, with no default. The old default was `data` relative to the working
directory, which is how raw CSVs end up inside a synced folder. Dates are **start inclusive,
end exclusive**, like every other fetcher in the repository.

Each run writes two files under `<out>/raw/`:

| File | Rows | What it holds |
|---|---|---|
| `gdp_<product>_<box>_<start>-<end>.csv` | one per fix | position, velocity, SST, drogue loss date, and for hourly the fix gap |
| `gdp_<product>_buoys_<box>_<start>-<end>.csv` | one per buoy | tracking method, buoy type, drogue depth, drogue sensor, how the track ended, deploy and end dates |

**Size.** The hourly file for the five-year window is **124.5 MB** (108.8 MB before the `gap`
column was added). A single file over 100 MB cannot be pushed to GitHub at all, so these stay
in the gitignored data tree. The 6-hourly file for Nov 2022 – Dec 2023 is 5.2 MB.

Reading them back:

```python
from sar.utils.data_io import load_drifters, combine_products

hourly = load_drifters("C:/maritime-data/raw/gdp_hourly_17-36N_82-63W_20190101-20240101.csv",
                       buoys="C:/maritime-data/raw/gdp_hourly_buoys_17-36N_82-63W_20190101-20240101.csv")
later = load_drifters("C:/maritime-data/raw/gdp_6hour_17-36N_82-63W_20221101-20240101.csv",
                      buoys="C:/maritime-data/raw/gdp_6hour_buoys_17-36N_82-63W_20221101-20240101.csv")
drifters = combine_products(hourly, later)
```

## What each loaded field means

| Field | Units / values | Notes |
|---|---|---|
| `ID` | string | the GDP buoy identifier |
| `time` | UTC | |
| `lat`, `lon` | degrees; `lon` 0–360 | D020's storage convention, as for every grid in the project |
| `ve`, `vn` | m/s, eastward and northward | the buoy's own velocity estimate |
| `sst_c` | °C, NaN outside [−2, 40] | **the products differ**: hourly serves Kelvin, 6-hourly serves °C. Both are converted here. Fill values (−999999, and 1000 °C-type values in the hourly Kelvin) are masked, not trusted |
| `undrogued` | bool, **per fix** | a buoy is drogued before its loss date and undrogued after, so this is never a per-buoy flag. A buoy with a 0 m drogue is undrogued at every fix |
| `tier_uncertain` | bool | the metadata disagrees about the drogue (see below). Leave these out of any drogued-versus-undrogued comparison |
| `fix_gap_h` | hours, NaN if not served | hours between the real position fixes either side of this estimate. **Hourly only** |
| `drogue_depth_m` | m, NaN if blank | from the server's `DrogueCenterDepth` string: 15 m for a standard drogue, 0 m for none |
| `location_type` | `GPS` / `Argos` / NaN | NaN for the 6-hourly product, which does not serve the field |
| `product` | `hourly` / `6-hourly` | |
| `segment_id` | int | a contiguous in-box run. A new segment starts after a gap longer than 3 h (hourly) or 7 h (6-hourly) |

## The rules the loader applies, and why

**GPS only.** 3 of 268 buoys in the box are Argos-tracked; the rest are GPS. Argos positions
are far less accurate. A buoy with no `location_type`, which is every 6-hourly buoy, is kept,
not assumed to be Argos.

**Points far from a real fix are flagged, not dropped.** The hourly product is fitted between
raw position fixes and fills gaps of up to 12 h without leaving a hole in the series. `gap` says
how far apart the real fixes were: median 1.0 h, 95th percentile 3.0 h, 99th 5.0 h, maximum 12 h.
**1.49 % of hourly points sit more than 3 h from a real fix**, inside 168 of the 640 segments of
48 h or more. Dropping them would cut 168 real tracks, so a scorer skips them as truth instead,
using `fix_gap_h > 3`, the same 3 h standard the segment rule uses.

**Drogue tier from metadata.** 52 buoys were deployed with a 0 m drogue, meaning no drogue at
all (48 SVPV, 4 SVPBV). For 43 the loss date equals the deployment date, so the loss-date rule
alone already gets them right. `tier_uncertain` is set where the records disagree:

- 0 m drogue, but a loss date 1–9 days after deployment (7 buoys);
- 0 m drogue with no deployment date at all (2 buoys);
- a loss date of 1970-01-01, the server's code for "drogue status uncertain from the
  beginning" (none in the box on 23 Sep).

**Two products, joined with a buffer.** `combine_products` keeps the hourly record and adds the
6-hourly one only from **2022-11-07**, one week after the hourly product ends. A buoy that
crosses from one product to the other then has a gap in its record rather than a seam, so its
two halves cannot sit either side of a dev/sealed split while sharing water (#51).

## What is not used, and why

**`err_lat` and `err_lon`.** They are labelled "95 % confidence interval" and measured a median
**under 1 cm**, maximum 0.2 m. No GPS drifter position is known to a centimetre, so these values
are not what their label says. They are not pulled.

## Where the data comes from

The GDP deploys surface drifting buoys worldwide: a float at the surface tethered to a
**drogue**, a sail centred at 15 m depth, so the buoy moves with the water at that depth rather
than being pushed by the wind. Drogues are often lost, usually in storms. After that the float
drifts at the surface and feels about 1–1.5 % of the wind speed (Pazan & Niiler 2001; Lumpkin et
al. 2013). A person in the water feels 1.9–2.7 % (Allen 2000). That is why both kinds are needed:
the leeway term should improve predictions for undrogued buoys and not for drogued ones (D018).

Both products are **quality-controlled and interpolated**, not raw satellite fixes. The server's
own metadata says how:

| Product | Coverage in the box | Method (server metadata) | Citation |
|---|---|---|---|
| `drifter_hourly_qc` | to **2022-10-31** | "interpolation via mathematical model fitting" | Elipot, Sykulski, Lumpkin, Centurioni, Pazos (2022), v2.01, NOAA NCEI, doi:10.25921/x46c-3620 |
| `drifter_6hour_qc` | to 2025-06-18 | "krigged (interpolation method)" | Lumpkin & Centurioni (2019), NOAA NCEI, doi:10.25921/7ntx-z961 |

Both describe themselves as updated quarterly. **The hourly one has not moved past 2022-10-31
since at least August 2023**, which is why the 6-hourly product is pulled as well: it is the only
drifter truth for Nov 2022 – Dec 2023 inside the study window.

**What that implies for trust.** A position here is an interpolation between real fixes, which is
why `fix_gap_h` matters: close to a fix it is nearly an observation, 12 h from one it is the
fitting method's guess. The 6-hourly product is smoother and coarser in time, so it is scored at
6-hour steps only.

## Measured on 2026-09-23

As fetched:

| | Hourly, 2019 – Oct 2022 | 6-hourly, Nov 2022 – 2023 |
|---|---|---|
| fixes | 926,533 | 48,861 (from 2022-11-01) |
| buoys | 268 (265 GPS, 3 Argos) | 96 (60 not in the hourly set) |
| drogue depth | 215 at 15 m, 52 at 0 m, 1 blank | 88 at 15 m, 8 at 0 m |
| segments ≥ 48 h | 640 | 141 |
| points > 3 h from a real fix | 1.49 % | not served |
| file | 124.5 MB | 5.2 MB |

As loaded, with the buoy tables, by `python -m sar.utils.data_io --drifters ... --buoys ...`:

| | Hourly | 6-hourly | Combined (`combine_products`) |
|---|---|---|---|
| fixes | 923,819 (Argos removed 2,714) | 48,861 | 971,885 (6-hourly from 2022-11-07: 48,066) |
| buoys | 265, all GPS | 96 | 326 |
| undrogued fixes | 77.9 % | 82.4 % | |
| tier-uncertain | 9 buoys, 11,144 fixes | 0 | |
| segments ≥ 48 h (duration) | **636** (640 less the Argos buoys' 4) | 141 | 636 + **139** after the buffer, from 94 buoys |
