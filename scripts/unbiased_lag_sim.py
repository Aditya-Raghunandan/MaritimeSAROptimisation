"""unbiased_lag_sim.py: an unbiased but noisy drift forecast shows a "lag" and a stretch (18 Oct, D034).

Each true displacement T has a random direction and a fixed length; the forecast is D = T + E with
E isotropic, zero-mean Gaussian: no bias at all. Measured the way scripts/cloud_shape.py measures
real buoys, along the forecast's own direction D/|D|, the truth sits behind the forecast and its
error is wider along than across whenever |E| is comparable to |T|, because E helps choose the
direction it is measured along. So L41's lag and stretch need not mean the current is too fast.

    python scripts/unbiased_lag_sim.py
"""

import numpy as np


def simulate(true_km: float, error_km: float, n: int = 200_000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0.0, 2.0 * np.pi, n)
    truth = true_km * np.column_stack([np.cos(angle), np.sin(angle)])
    forecast = truth + rng.normal(0.0, error_km, (n, 2))
    u = forecast / np.linalg.norm(forecast, axis=1)[:, None]
    v = np.column_stack([-u[:, 1], u[:, 0]])
    offset = truth - forecast
    along, across = (offset * u).sum(axis=1), (offset * v).sum(axis=1)
    return {"true_km": true_km, "error_km": error_km, "mean_along_km": float(along.mean()),
            "rms_ratio": float(np.sqrt((along ** 2).mean() / (across ** 2).mean())),
            "best_scale": float((truth * u).sum() / (forecast * u).sum())}


if __name__ == "__main__":
    for t, e in ((1.0, 0.7), (2.0, 1.4), (3.0, 2.1), (3.0, 1.0)):
        r = simulate(t, e)
        print(f"true drift {t} km, error {e} km per axis: mean along {r['mean_along_km']:+.2f} km, "
              f"along/across rms {r['rms_ratio']:.2f}, best scale of the forecast {r['best_scale']:.2f}")
