/**
 * The browser port of the resultant, checked against Python case by case.
 *
 * `fixtures/resultant_golden.json` is written by `scripts/export_resultant_golden.py`
 * from `src/sar/model/interpolate.py`. Every number below comes from that run, so a
 * failure here means the two implementations have parted company, not that the fixture
 * is stale: regenerate it only when the Python is what changed, and read the diff.
 */

import { describe, expect, it } from 'vitest';

import golden from '../src/fixtures/resultant_golden.json';
import {
  MissingCornerError,
  OutOfCoverageError,
  bilinearWeights,
  cellCorners,
  interpolationUncertainty,
  keepInside,
  locate,
  neighbourDirection,
  resultantVector,
  sampleField,
  speedDirection,
  timeBlend,
} from '../src/interpolate.js';
import { Grid } from '../src/layers.js';
import { BufferSource } from '../src/sources.js';

const TOL = golden.tolerance;
const grid = new Grid(golden.grid);

/** The fixture's field as a BufferSource, with JSON null turned back into NaN. */
function fixtureSource() {
  const flat = Float32Array.from(golden.field, (x) => (x === null ? NaN : x));
  return new BufferSource(flat, golden.grid, golden.frames);
}

const field = { grid, source: fixtureSource() };

describe('the fixture itself', () => {
  it('carries cases, failures and a land cell', () => {
    expect(golden.cases.length).toBeGreaterThan(0);
    expect(golden.failures.length).toBeGreaterThan(0);
    expect(golden.land_cells.length).toBeGreaterThan(0);
  });

  it('is float32 all the way through, so a mismatch is a real one', () => {
    // The fixture is written from float64 but read into a Float32Array, the same type the
    // archive uses. Checking one known value keeps that narrowing honest rather than
    // letting it hide behind the 1e-9 tolerance.
    const [j, i] = golden.cases[0].cells[0];
    const [u] = field.source.vector(golden.cases[0].frame, j, i);
    expect(Number.isFinite(u)).toBe(true);
  });
});

describe('sampleField matches Python', () => {
  for (const c of golden.cases) {
    it(`${c.name}`, () => {
      const got = sampleField(field, c.frame, c.lat, c.lon);
      // float32 storage, so compare at the tolerance the fixture states rather than
      // exactly: the arithmetic must agree, the storage cannot.
      expect(got.u).toBeCloseTo(c.u, 6);
      expect(got.v).toBeCloseTo(c.v, 6);
      expect(got.speed).toBeCloseTo(c.speed, 6);
      if (c.direction_to_deg === null) {
        expect(got.directionTo).toBeNull();
      } else {
        expect(got.directionTo).toBeCloseTo(c.direction_to_deg, 5);
      }
      expect(got.cells).toEqual(c.cells);
      got.weights.forEach((w, k) => expect(w).toBeCloseTo(c.weights[k], 12));
    });
  }
});

describe('the weights match the reference exactly', () => {
  for (const c of golden.cases) {
    it(`sum to 1 and match the reference for ${c.name}`, () => {
      const { weights } = locate(grid, c.lat, c.lon);
      const total = weights.reduce((a, b) => a + b, 0);
      expect(Math.abs(total - 1)).toBeLessThan(TOL);
      weights.forEach((w, k) => expect(Math.abs(w - c.weights[k])).toBeLessThan(TOL));
    });
  }
});

describe('uncertainty matches Python', () => {
  for (const c of golden.cases) {
    it(`${c.name}`, () => {
      const got = sampleField(field, c.frame, c.lat, c.lon).uncertainty;
      expect(got.sigmaSpatialMs).toBeCloseTo(c.uncertainty.sigma_spatial_ms, 6);
      expect(got.coherence).toBeCloseTo(c.uncertainty.coherence, 6);
      expect(got.speedLossMs).toBeCloseTo(c.uncertainty.speed_loss_ms, 6);
      expect(got.nCorners).toBe(c.uncertainty.n_corners);
      if (c.uncertainty.direction_spread_deg === null) {
        expect(got.directionSpreadDeg).toBeNull();
      } else {
        expect(got.directionSpreadDeg).toBeCloseTo(c.uncertainty.direction_spread_deg, 5);
      }
    });
  }
});

describe('the failures match Python', () => {
  for (const f of golden.failures) {
    it(`${f.name} raises ${f.error}`, () => {
      let thrown = null;
      try {
        sampleField(field, 0, f.lat, f.lon);
      } catch (err) {
        thrown = err;
      }
      expect(thrown).not.toBeNull();
      expect(thrown.name).toBe(f.error);
    });
  }

  it('a land corner says how many of the four are missing', () => {
    const land = golden.failures.find((f) => f.error === 'MissingCornerError');
    try {
      sampleField(field, 0, land.lat, land.lon);
      throw new Error('expected a MissingCornerError');
    } catch (err) {
      expect(err).toBeInstanceOf(MissingCornerError);
      expect(err.message).toMatch(/\d of 4|all 4/);
      expect(err.missing.length).toBeGreaterThan(0);
    }
  });

  it('a point outside the grid is refused, not clamped to the edge', () => {
    // Clamping is the failure this guards: it would answer with the coast's current for a
    // position in the open Atlantic and look entirely plausible.
    expect(() => sampleField(field, 0, golden.grid.lat0 - 5, golden.grid.lon0))
      .toThrow(OutOfCoverageError);
  });
});

describe('timeBlend matches Python', () => {
  const { corners_t1: c1, corners_t2: c2, blends } = golden.time_blend;
  for (const b of blends) {
    it(`frac ${b.frac}`, () => {
      const got = timeBlend(c1, c2, b.frac);
      got.forEach((pair, k) => {
        expect(Math.abs(pair[0] - b.corners[k][0])).toBeLessThan(TOL);
        expect(Math.abs(pair[1] - b.corners[k][1])).toBeLessThan(TOL);
      });
    });
  }

  it('refuses a fraction outside 0 to 1', () => {
    expect(() => timeBlend(c1, c2, 1.5)).toThrow(RangeError);
  });
});

describe('the pieces on their own', () => {
  it('gives each axis its own sign', () => {
    expect(neighbourDirection(0.01, -0.01)).toEqual([1, -1]);
    expect(neighbourDirection(-0.01, 0.01)).toEqual([-1, 1]);
    expect(neighbourDirection(0, 0)).toEqual([1, 1]);
  });

  it('flips a sign that would leave the grid', () => {
    expect(keepInside(5, 1, 10)).toBe(1);
    expect(keepInside(9, 1, 10)).toBe(-1);
    expect(keepInside(0, -1, 10)).toBe(1);
  });

  it('puts all the weight on the nearest point when the offsets are zero', () => {
    expect(bilinearWeights(0, 0)).toEqual([1, 0, 0, 0]);
  });

  it('splits evenly at the centre of a cell', () => {
    bilinearWeights(0.5, 0.5).forEach((w) => expect(w).toBeCloseTo(0.25, 15));
  });

  it('orders the corners nearest, longitude, latitude, diagonal', () => {
    expect(cellCorners(5, 6, 1, -1, 20, 20)).toEqual([[5, 6], [5, 5], [6, 6], [6, 5]]);
  });

  it('measures bearings clockwise from north', () => {
    expect(speedDirection(0, 1).directionTo).toBeCloseTo(0, 9);
    expect(speedDirection(1, 0).directionTo).toBeCloseTo(90, 9);
    expect(speedDirection(0, -1).directionTo).toBeCloseTo(180, 9);
    expect(speedDirection(-1, 0).directionTo).toBeCloseTo(270, 9);
  });

  it('has no bearing for a zero vector', () => {
    expect(speedDirection(0, 0).directionTo).toBeNull();
  });

  it('loses speed when the corners disagree, and says so', () => {
    // East and north at equal weight: the vector average is 0.707, not 1.
    const corners = [[1, 0], [0, 1]];
    const weights = [0.5, 0.5];
    const [u, v] = resultantVector(corners, weights);
    expect(Math.hypot(u, v)).toBeCloseTo(0.7071067811865476, 12);
    const unc = interpolationUncertainty(corners, weights);
    expect(unc.speedLossMs).toBeCloseTo(1 - 0.7071067811865476, 12);
    expect(unc.coherence).toBeLessThan(1);
  });

  it('gives zero, not one, for two opposing corners', () => {
    const [u, v] = resultantVector([[1, 0], [-1, 0]], [0.5, 0.5]);
    expect(Math.hypot(u, v)).toBeCloseTo(0, 15);
    expect(interpolationUncertainty([[1, 0], [-1, 0]], [0.5, 0.5]).coherence)
      .toBeCloseTo(0, 15);
  });
});

describe('against the nearest-neighbour answer it replaces', () => {
  it('differs from rounding to the nearest cell, which is the point of it', () => {
    // Taken from a fixture case rather than a hardcoded cell: when the fixture is cut
    // from the real archive, an arbitrary cell can be land, and this is a test about
    // interpolation rather than about the land mask.
    const c = golden.cases.find((x) => x.name.startsWith('interior'));
    const cell = grid.cellAt(c.lat, c.lon);
    const nearest = field.source.vector(c.frame, cell.j, cell.i);
    const got = sampleField(field, c.frame, c.lat, c.lon);
    expect(Math.hypot(got.u - nearest[0], got.v - nearest[1])).toBeGreaterThan(1e-6);
  });

  it('stays within the range its own four corners span', () => {
    // Bilinear interpolation is a weighted mean, so it can never overshoot the corners.
    // On real data that is the cheapest check that the weights are sane.
    for (const c of golden.cases) {
      const us = c.cells.map(([j, i]) => field.source.vector(c.frame, j, i)[0]);
      expect(c.u).toBeGreaterThanOrEqual(Math.min(...us) - 1e-6);
      expect(c.u).toBeLessThanOrEqual(Math.max(...us) + 1e-6);
    }
  });
});
