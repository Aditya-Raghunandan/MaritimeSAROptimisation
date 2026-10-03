"""region.py: is the truth inside an ensemble's 90 % region? (#89, and #52's R2c)

R2c scores "the fraction of true positions inside the 90 % probability contour": the
smallest region that holds 90 % of the probability, called the highest-density region
(HDR). This finds it from the particles themselves, with no grid and no cell size.

THE METHOD (Hyndman 1996, "Computing and graphing highest density regions",
The American Statistician 50(2)):
  1. Smooth the particles into a density f with a Gaussian kernel (a small blob on every
     particle). The blob is shaped like the cloud: its covariance is the cloud's,
     scaled by Scott's factor N^(-1/6) and by `bandwidth`.
  2. Evaluate f at every particle, leaving that particle's own blob out, which would
     otherwise make every particle look denser than it is.
  3. The level-L region is {x : f(x) >= the (1 - L) quantile of those densities}. It
     holds a fraction L of the particles by construction.
  4. The truth is inside it if f(truth) is at least that threshold.
Step 3 corrects itself for the smoothing: blurring the blobs moves the density at the
particles and the threshold together, so for an elliptical cloud the region does not
grow with the bandwidth. tests/validate/test_region.py checks this on a cloud whose
answer is known, and #89 varies the bandwidth on the real runs.

WHY NOT A GRID. At 500 m cells, a 90 % region 24 h out covers ~10^4 cells, so a stable
grid region needs ~10^5 particles per window. The densities here need ~10^3.

WHY NOT A CIRCLE OR AN ELLIPSE. Shear stretches and bends the cloud; a fixed shape would
call a buoy inside that sits in the cloud's empty corner.

`rank` is the probability content of the smallest HDR that contains the truth. It is
uniform on (0, 1) for a calibrated ensemble, so one number per window gives the coverage
at every level: inside at level L exactly when rank <= L.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sar.utils.geo import M_PER_DEG_LAT, metres_per_degree_lon

LEVELS = (0.5, 0.68, 0.8, 0.9, 0.95)
# A cloud with no spread (sigma = 0) has a singular covariance. One metre per axis is far
# below anything a search could resolve, and keeps the kernel defined.
MIN_SPREAD_M = 1.0


def local_metres(lats, lons, ref_lat, ref_lon) -> np.ndarray:
    """[east, north] metres from a reference point on a local flat earth; (..., 2)."""
    dlon = (np.asarray(lons, float) - ref_lon + 180.0) % 360.0 - 180.0
    east = dlon * metres_per_degree_lon(ref_lat)
    north = (np.asarray(lats, float) - ref_lat) * M_PER_DEG_LAT
    return np.stack([east, north], axis=-1)


@dataclass(frozen=True)
class CloudScore:
    """How one ensemble did against one true position."""

    rank: float              # probability content of the smallest HDR holding the truth
    area_m2: dict            # level -> area of that HDR, m^2
    energy: float            # energy score, m: lower is better
    centroid_error_m: float  # distance from the cloud's mean to the truth
    spread_m: float          # rms distance of the particles from their mean

    def inside(self, level: float) -> bool:
        return self.rank <= level


def _kernel_whitener(points: np.ndarray, bandwidth: float) -> tuple[np.ndarray, float]:
    """The matrix that makes the kernel a unit circle, and the kernel's normaliser."""
    n = points.shape[0]
    cov = np.cov(points, rowvar=False) + MIN_SPREAD_M ** 2 * np.eye(2)
    h = (bandwidth * n ** (-1.0 / 6.0)) ** 2 * cov
    chol = np.linalg.cholesky(h)
    return np.linalg.inv(chol), 1.0 / (2.0 * np.pi * float(np.prod(np.diag(chol))))


def densities(points: np.ndarray, truth: np.ndarray, bandwidth: float = 1.0):
    """Leave-one-out density at every particle, and the density at the truth (per m^2)."""
    points = np.asarray(points, float)
    n = points.shape[0]
    if n < 10:
        raise ValueError(f"need at least 10 particles for a density, got {n}")
    white, norm = _kernel_whitener(points, bandwidth)
    p = points @ white.T
    y = np.asarray(truth, float) @ white.T
    sq = np.sum(p * p, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (p @ p.T)
    np.maximum(d2, 0.0, out=d2)
    k = np.exp(-0.5 * d2)
    f = norm * (k.sum(axis=1) - 1.0) / (n - 1)          # minus each particle's own blob
    f_truth = norm * np.exp(-0.5 * np.sum((p - y) ** 2, axis=1)).mean()
    return f, f_truth


def energy_score(points: np.ndarray, truth: np.ndarray) -> float:
    """E|X - y| - E|X - X'| / 2 over the particles, in metres.

    A proper score: on average it is lowest when the ensemble is the distribution the
    truth came from. It punishes missing the truth (the first term) and being wider
    than needed, since a wider cloud raises the first term more than the second.
    """
    points = np.asarray(points, float)
    n = points.shape[0]
    to_truth = np.hypot(*(points - np.asarray(truth, float)).T).mean()
    sq = np.sum(points * points, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (points @ points.T)
    pair = np.sqrt(np.maximum(d2, 0.0)).sum() / (n * (n - 1))
    return float(to_truth - 0.5 * pair)


def score_cloud(points, truth, levels=LEVELS, bandwidth: float = 1.0) -> CloudScore:
    """Score one ensemble, given as (N, 2) metres, against one truth, (2,) metres."""
    points = np.asarray(points, float)
    truth = np.asarray(truth, float)
    f, f_truth = densities(points, truth, bandwidth)
    rank = float(np.mean(f > f_truth))
    area = {}
    for level in levels:
        threshold = np.quantile(f, 1.0 - level)
        # Area of {f >= threshold}: E over the particles of 1{f >= t} / f, since the
        # particles are draws from the density being integrated.
        # A lone particle can have a leave-one-out density of 0; it is never in the region
        # unless the threshold is 0 too, and then the region is unbounded anyway.
        inv = np.divide(1.0, f, out=np.full_like(f, np.inf), where=f > 0)
        area[level] = float(np.mean(np.where(f >= threshold, inv, 0.0)))
    centre = points.mean(axis=0)
    return CloudScore(rank=rank, area_m2=area, energy=energy_score(points, truth),
                      centroid_error_m=float(np.hypot(*(centre - truth))),
                      spread_m=float(np.sqrt(np.mean(np.sum((points - centre) ** 2, axis=1)))))
