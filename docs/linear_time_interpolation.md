# Linear time interpolation (`sar.utils.interpolation`)

Why the drift model needs this module, copied and trimmed from the project vault decision
`decisions/D009 Integration scheme and engine architecture.md` (Aditya, decided
2026-09-03, design only, no code written at the time of the decision). See the vault for
the full options comparison, evidence plan and consequences.

## Why interpolation is needed

The drift model integrator advances the particle state with Euler-Maruyama at
`dt = 60 s` (one step per minute):

$$\vec x_{k+1} = \vec x_k + \big(\vec v_c + \alpha \vec v_{w10}\big)\Delta t + \sigma\sqrt{\Delta t}\,\vec Z_k,\qquad \vec Z_k\sim\mathcal N(0,I)$$

The forcing fields it reads do not arrive at that cadence:

- Wind: hourly.
- Current: 3-hourly.

Both fields therefore need to be broken down into per-minute pieces between two known
values before the integrator can consume them at every step; this is what
`sar.utils.interpolation` provides.

Forcing access is meant to go through one `ForcingProvider.sample(x, y, t) -> (u, v)`
interface with three backends (cached local NetCDF, live forecast, analytic fields for
tests), fields loaded once into raw NumPy arrays so nothing calls xarray or OPeNDAP
inside the time loop; on a regular grid, interpolation is index arithmetic.

## Reproducibility

`np.random.SeedSequence(base).spawn(n)` for per-worker streams, never `seed(42)` then
fork. Every run logs its seed and parameters beside its output.
