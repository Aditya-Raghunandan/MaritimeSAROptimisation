import { describe, it, expect } from 'vitest';

import {
  mercatorY, mercatorLat, rowsFor, mercatorRowMap, linearStretchError,
} from '../src/mercator.js';

// The current grid, as the published manifest describes it.
const CURRENT = { lat0: 17.0, dlat: 0.04, nlat: 476 };
// The wind grid. Same box, a quarter degree.
const WIND = { lat0: 17.0, dlat: 0.25, nlat: 77 };

describe('mercatorY / mercatorLat', () => {
  it('is zero at the equator and rises northward', () => {
    expect(mercatorY(0)).toBeCloseTo(0, 12);
    expect(mercatorY(45)).toBeGreaterThan(mercatorY(30));
    expect(mercatorY(-30)).toBeLessThan(0);
  });

  it('round-trips across the study box and well beyond it', () => {
    for (const lat of [-60, -17, 0, 17, 26.5, 36, 60, 80]) {
      expect(mercatorLat(mercatorY(lat))).toBeCloseTo(lat, 9);
    }
  });

  it('stretches the north: equal latitude steps are not equal y steps', () => {
    const low = mercatorY(20) - mercatorY(17);
    const high = mercatorY(36) - mercatorY(33);
    expect(high).toBeGreaterThan(low);
    // This inequality IS the bug. If it ever became an equality the fix would
    // be unnecessary, and the test would be telling us the projection changed.
    expect(high / low).toBeGreaterThan(1.1);
  });
});

/*
  The measurement that motivated the fix, pinned so it cannot quietly stop
  being true. Taken from the live site on 2026-09-21 by locating each coastline
  in the rendered raster and comparing it with the same coastline in the HYCOM
  land mask. Agreement was +-0.015 deg at both latitudes, so the tolerance here
  is 0.02.
*/
describe('linearStretchError -- what the uncorrected draw did', () => {
  it('reproduces the two displacements measured on the live site', () => {
    expect(linearStretchError(CURRENT, 23.060)).toBeCloseTo(0.352, 1);
    expect(linearStretchError(CURRENT, 18.220)).toBeCloseTo(0.105, 1);
  });

  it('peaks near the middle of the box and vanishes at its edges', () => {
    const mid = linearStretchError(CURRENT, 26.5);
    expect(mid).toBeGreaterThan(0.39);
    expect(mid).toBeLessThan(0.41);
    expect(Math.abs(linearStretchError(CURRENT, 17.02))).toBeLessThan(0.02);
    expect(Math.abs(linearStretchError(CURRENT, 35.98))).toBeLessThan(0.02);
  });

  it('hits the wind grid just as hard, which is why wind was never innocent', () => {
    /*
      Wind never SHOWED the error, because a field painted over land and sea
      alike offers nothing to notice a displacement against. It was drawn just
      as wrongly as the current for as long as it has existed.

      The two are not identical, and the reason is worth keeping: each grid's
      painted extent runs half a cell past its outer centres, so wind's box
      reaches 36.125 N against the current's 36.020 N. A wider span in the
      stretch means a slightly larger error -- 0.409 against 0.400.
    */
    const wind = linearStretchError(WIND, 26.5);
    const current = linearStretchError(CURRENT, 26.5);
    expect(wind).toBeGreaterThan(0.39);
    expect(Math.abs(wind - current)).toBeLessThan(0.02);
  });
});

describe('rowsFor', () => {
  it('never gives the buffer fewer rows than the grid has', () => {
    expect(rowsFor(CURRENT)).toBeGreaterThanOrEqual(CURRENT.nlat);
    expect(rowsFor(WIND)).toBeGreaterThanOrEqual(WIND.nlat);
  });

  it('asks for a few more rows than the grid, not a multiple of it', () => {
    // Enough that the south is not undersampled; not so many that the buffer
    // becomes a memory decision.
    expect(rowsFor(CURRENT)).toBeLessThan(2 * CURRENT.nlat);
    expect(rowsFor(WIND)).toBeLessThan(2 * WIND.nlat);
  });
});

describe('mercatorRowMap', () => {
  const rows = rowsFor(CURRENT);
  const map = mercatorRowMap(CURRENT, rows);

  it('runs north to south: row 0 is the highest latitude index', () => {
    expect(map[0]).toBe(CURRENT.nlat - 1);
    expect(map[rows - 1]).toBe(0);
  });

  it('never leaves the grid', () => {
    for (const j of map) {
      expect(j).toBeGreaterThanOrEqual(0);
      expect(j).toBeLessThan(CURRENT.nlat);
    }
  });

  it('is monotonic -- it may repeat a row, never reorder one', () => {
    for (let r = 1; r < rows; r += 1) expect(map[r]).toBeLessThanOrEqual(map[r - 1]);
  });

  it('skips no grid row, which is what rowsFor is sized for', () => {
    // A skipped row is a piece of coastline that simply is not drawn.
    const seen = new Set(map);
    expect(seen.size).toBe(CURRENT.nlat);
  });

  /*
    The property the whole file exists for: a row placed at its own fraction of
    the image must land at the latitude it holds, once the image is stretched
    linearly onto a Mercator axis. Checked by inverting the stretch.
  */
  it('places every row where that latitude actually projects', () => {
    const south = CURRENT.lat0 - CURRENT.dlat / 2;
    const north = CURRENT.lat0 + (CURRENT.nlat - 1) * CURRENT.dlat + CURRENT.dlat / 2;
    const yN = mercatorY(north);
    const yS = mercatorY(south);

    let worst = 0;
    for (let r = 0; r < rows; r += 1) {
      const heldLat = CURRENT.lat0 + map[r] * CURRENT.dlat;      // the data in this row
      const drawnAt = mercatorLat(yN - ((r + 0.5) / rows) * (yN - yS));  // where it lands
      worst = Math.max(worst, Math.abs(drawnAt - heldLat));
    }
    // Within half a cell: the residual is the grid's own resolution, not a
    // projection error. Before the fix this figure was 0.40 deg, ten cells.
    expect(worst).toBeLessThanOrEqual(CURRENT.dlat / 2 + 1e-9);
  });
});
