/**
 * Tests for the resultant drift field.
 *
 * The arithmetic is trivial; what matters is that the thing refuses to claim
 * more than it has. D002 has three terms, a field rendering can carry at most
 * two, and the one it can never carry is the one that makes the model
 * stochastic.
 */

import { describe, expect, it } from 'vitest';

import { ALPHA, ResultantSource, resultantScale } from '../src/resultant.js';
import { Grid } from '../src/layers.js';

const WIND_GRID = new Grid({ lat0: 17, dlat: 0.25, nlat: 77, lon0: -82, dlon: 0.25, nlon: 77 });
const CUR_GRID = new Grid({ lat0: 17, dlat: 0.04, nlat: 476, lon0: -82, dlon: 0.08, nlon: 238 });

const axis = (stepSeconds, frames) => ({
  start: new Date('2021-01-01T00:00:00Z'), stepSeconds, frames,
});

function src(vector, frames = 48) {
  return { frames, vector, isResident: () => true, ensure: async () => {} };
}

const wind = (u, v) => ({
  source: src(() => [u, v]), grid: WIND_GRID, axis: axis(3600, 48),
});
const current = (u, v) => ({
  source: src(() => [u, v], 16), grid: CUR_GRID, axis: axis(10800, 16),
});

describe('the leeway-only case, which is where we are today', () => {
  const r = new ResultantSource(wind(10, 0), null);

  it('is 2 % of the wind', () => {
    expect(r.vector(0, 0, 0)).toEqual([0.2, 0]);
    expect(ALPHA).toBe(0.02);
  });

  it('knows it is partial', () => {
    expect(r.isPartial).toBe(true);
  });

  it('says so in its own label, not only in a footnote', () => {
    expect(r.describe().label).toMatch(/leeway only/i);
  });

  it('names the current as missing, and says it is usually the larger term', () => {
    const d = r.describe();
    expect(d.missing).toContain('surface current');
    expect(d.caveat).toMatch(/larger term/);
    expect(d.caveat).toMatch(/not yet where a drifter would go/);
  });
});

describe('with a current, which is where we are going', () => {
  const r = new ResultantSource(wind(10, 0), current(1.5, 0.5));

  it('adds the current to the leeway', () => {
    expect(r.vector(0, 0, 0)).toEqual([1.7, 0.5]);   // 0.2 + 1.5, 0 + 0.5
  });

  it('is no longer partial', () => {
    expect(r.isPartial).toBe(false);
    expect(r.describe().label).toBe('Resultant drift');
  });

  it('STILL names eta as missing, because a field can never carry it', () => {
    // This is the point of the whole file. eta is a per-particle draw with no
    // value at a location; a resultant arrow is not where a drifter goes.
    const d = r.describe();
    expect(d.missing.join(' ')).toMatch(/η/);
    expect(d.caveat).toMatch(/not a\s+field/);
    expect(d.caveat).toMatch(/will not\s+follow the same path/);
  });

  it('a land cell in the current does not blank the arrow', () => {
    // HYCOM is NaN over land. Adding it would wipe the leeway term too, and an
    // absent arrow reads as "no wind" rather than "no sea".
    const withLand = new ResultantSource(wind(10, 0), current(NaN, NaN));
    expect(withLand.vector(0, 0, 0)).toEqual([0.2, 0]);
  });
});

describe('the two grids', () => {
  it('output is on the wind grid, the coarser of the two', () => {
    const r = new ResultantSource(wind(10, 0), current(1, 0));
    expect(r.grid.nlat).toBe(77);
    expect(r.grid.dlat).toBe(0.25);
  });

  it('maps a wind cell onto the right current cell', () => {
    // Wind cell (4, 8) is 18.00 N, -80.00 E. On the current grid that is
    // j = (18-17)/0.04 = 25, i = (-80+82)/0.08 = 25.
    const seen = [];
    const r = new ResultantSource(
      wind(10, 0),
      { source: { frames: 16, isResident: () => true, ensure: async () => {},
                  vector: (f, j, i) => { seen.push([j, i]); return [1, 0]; } },
        grid: CUR_GRID, axis: axis(10800, 16) },
    );
    r.vector(0, 4, 8);
    expect(seen).toEqual([[25, 25]]);
  });
});

describe('frame alignment between an hourly and a 3-hourly field', () => {
  const r = new ResultantSource(wind(10, 0), current(1, 0));

  it('snaps to the NEAREST current frame, not the most recent', () => {
    // 01:30 is hour 1.5; the 3-hourly field should go forward to 03:00 rather
    // than hold 00:00, or the current visibly lags the wind on the same map.
    expect(r._currentFrame(0)).toBe(0);
    expect(r._currentFrame(1)).toBe(0);     // 01:00 -> 00:00
    expect(r._currentFrame(2)).toBe(1);     // 02:00 -> 03:00
    expect(r._currentFrame(3)).toBe(1);     // 03:00 -> 03:00
  });

  it('clamps rather than running off the end of the shorter field', () => {
    expect(r._currentFrame(47)).toBe(15);
    expect(r._currentFrame(1e6)).toBe(15);
  });
});

describe('resultantScale', () => {
  it('is not the wind scale, or every arrow would be invisible', () => {
    // Leeway is 2 % of wind, so on a 20 m/s scale the resultant maxes at 0.4.
    expect(resultantScale(20, false)).toBeCloseTo(0.4);
    expect(resultantScale(20, false)).toBeLessThan(20);
  });

  it('is set by the current once there is one', () => {
    expect(resultantScale(20, true, 1.8)).toBeCloseTo(2.52);
  });

  it('never returns zero, which would divide by nothing', () => {
    expect(resultantScale(0, false)).toBeGreaterThan(0);
  });
});
