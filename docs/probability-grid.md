# The probability map grid (`sar.model.grid`)

Closes #7. Turns a cloud of particle positions into cells: where the cells are, which one a
position falls in, and how counts on them are normalised and coarsened for the agent.

**Why it looks nothing like the issue asked for** is [ADR002](ADR002.md). Short version:
`grid` was denoting three different objects, a single `resolution` argument cannot describe
a grid whose two steps differ, and the 1/12° default matches no grid this project holds.

## Two constructors, and they are not interchangeable

```python
from sar.model.grid import ProbabilityGrid, normalise

# The one an episode runs in. Fixed size, anchored on the cloud at arrival.
grid = ProbabilityGrid.centred_on(lat=26.5, lon=-76.0, box_km=100, cell_m=500,
                                  cells_multiple_of=32)

# The one that tells you what box_km should be. Fits to whatever cloud it is given.
grid = ProbabilityGrid.from_envelope(lats, lons, cell_m=250, margin_km=20)
```

`centred_on` is square in metres, so square in cells, so it coarsens to a square
observation. Use it for anything the policy sees: a constant box size is what makes one
agent cell mean the same distance in every episode.

`from_envelope` sizes the box to the cloud, which is useful for the archive map and for
deciding `box_km` — but it gives a different size per scenario, which the RL environment
must not have.

**`cell_m` is required and has no default.** That is the decision, not an oversight — and the
defensible answer is a *window*, not a number.

## How to pick `cell_m`

Not from the sweep width. W is how wide a strip the searcher clears; a cell here is how
finely the *target's position distribution* is described. Coverage lives on the particles
([ADR002](ADR002.md) §4), so the reward is exact at any cell size and the two are decoupled.
What actually bounds it, over a 100 km box:

| | |
|---|---|
| **Floor, ~200 m** | sampling noise is 1/√k per cell, so it scales as 1/cell: **15.5 %** at 250 m and **39 %** at 100 m, both at N = 10⁶ |
| **Memory** | scales as 1/cell². uint16 counts are lossless here and half the bytes of float32: **45 MB/scenario at 250 m, 11 MB at 500 m** |
| **The browser** | a published chunk is 48 frames — measured at 1.11 MB / 45 ms for the forcing store, against **14.6 MB at 250 m** and **3.7 MB at 500 m**. This is why the archive is 500 m |
| **Ceiling, ~400 m** | a cell comparable to the whole distribution gives a blob, not a shape. **Rests on an ensemble spread nobody has measured yet** |

So: **500 m for anything published**, because of the chunk budget; **250 m is fine for the
episode map**, which is binned on demand in memory from particles the environment already
holds and is never written to disk.

## The cycle

```python
counts, lost = grid.bin(lats, lons)     # lost = particles outside the box -> lost_mass
p = normalise(counts)                    # sums to 1 over the grid, per timestep
obs_grid, obs = grid.coarsen(counts, 8)  # block-sum; mass preserved exactly
spec = grid.to_spec()                    # the six fields the manifest and layers.js speak
```

- `bin` **returns** the outside count rather than raising, because D016 requires lost mass
  to be logged per timestep. `position_to_cell` raises for the same situation, because a
  single position outside the box is a caller's bug.
- Beached particles are frozen in place and **keep their mass**, so they are inside
  `counts` already. `normalise(counts, beached_mass=...)` asserts that rather than adding it.
- Accumulate reward on raw counts **before** `normalise`, or rewards are not comparable
  between timesteps.
- `coarsen` returns the coarser grid with the array, because an array whose grid was not
  coarsened with it is a map drawn in the wrong place.

## Conventions

| | |
|---|---|
| `(lat0, lon0)` | the **centre** of cell (0, 0), as `sar.viz.export.grid_spec` also means it |
| Longitude in | either — indexing unwraps around `lon0`, so `-76` and `284` give the same cell |
| Longitude out | `to_spec()` and `cell_to_position()` return display (−180..180); `lon0` is stored 0–360 per D020 |
| Array shape | `(nlat, nlon)`, rows ascending north |
| Ties | a position exactly on a boundary goes to the even index, matching `interpolate.nearest_index` |

## Guards worth knowing about

`from_envelope` refuses a box wider than `max_box_km` (default 2100 km, the study domain's
own width) or larger than `max_cells` (default 20 million). Both exist because **the map is
not the domain**: fitting to a whole run rather than to the cloud at arrival is the
difference between 23 MB and 9.9 GB per scenario, and neither guard alone catches every
shape of that mistake.

`coarsen` refuses a factor that does not divide the grid rather than trimming or padding,
since either would move mass. Build the grid with `cells_multiple_of=factor` instead.

## Not here, and deliberately

- **Sweep width.** It has no citation yet (ADR002 §Outstanding), and nothing in this module
  needs it — `cell_m` is a parameter precisely so that W is not baked in anywhere.
- **Detection and coverage.** Swept on the particles, not on these cells — R4c and
  ADR002 §4.
- **cos φ in the integrator.** `sar.utils.geo.metres_per_degree_lon(lat)` is the shared
  conversion, and it takes latitude as an argument so that `drift.py` cannot freeze it.
  Freezing it is an 18.2 % eastward error that looks like physics.
