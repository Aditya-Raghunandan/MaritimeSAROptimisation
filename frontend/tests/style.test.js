/**
 * Tests for the quiver renderer's pure parts.
 *
 * The drawing itself needs a canvas and a live map, so what is pinned here is
 * the arithmetic that decides what gets drawn and in what colour -- and, above
 * all, the property that made us replace leaflet-velocity: a cell is drawn
 * once, where it is, no matter what the map is showing.
 */

import { describe, expect, it } from 'vitest';

import { SPEED_RAMP, decimation, speedColour } from '../src/style.js';
import { Grid } from '../src/layers.js';

describe('speedColour', () => {
  it('maps zero to the lightest step and the maximum to the darkest', () => {
    expect(speedColour(0, 20)).toBe(SPEED_RAMP[0]);
    expect(speedColour(20, 20)).toBe(SPEED_RAMP[SPEED_RAMP.length - 1]);
  });

  it('is monotonic: faster never gets a lighter step', () => {
    let last = -1;
    for (let s = 0; s <= 20; s += 0.5) {
      const idx = SPEED_RAMP.indexOf(speedColour(s, 20));
      expect(idx).toBeGreaterThanOrEqual(last);
      last = idx;
    }
  });

  it('clamps above the maximum rather than running off the ramp', () => {
    expect(speedColour(1e6, 20)).toBe(SPEED_RAMP[SPEED_RAMP.length - 1]);
  });

  it('survives a NaN rather than producing undefined', () => {
    // A NaN reaches here from a masked cell. Returning undefined would set
    // strokeStyle to garbage and silently stop drawing.
    expect(SPEED_RAMP).toContain(speedColour(NaN, 20));
  });

  it('survives a zero or negative maximum', () => {
    expect(SPEED_RAMP).toContain(speedColour(5, 0));
  });

  it('is one hue light to dark, not a rainbow', () => {
    // The old library ramped white -> yellow -> orange -> red, which is the
    // thing a magnitude encoding must not do. Check the hue stays put and the
    // lightness falls.
    const toLab = (hex) => {
      const lin = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
      const [r, g, b] = [1, 3, 5].map((i) => lin(parseInt(hex.slice(i, i + 2), 16) / 255));
      const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
      const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
      const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
      const A = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s;
      const B = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s;
      return {
        L: 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        h: ((Math.atan2(B, A) * 180) / Math.PI + 360) % 360,
      };
    };
    const lab = SPEED_RAMP.map(toLab);
    for (let k = 1; k < lab.length; k += 1) {
      expect(lab[k].L).toBeLessThan(lab[k - 1].L);        // strictly darker
    }
    const hues = lab.map((x) => x.h);
    expect(Math.max(...hues) - Math.min(...hues)).toBeLessThan(30);   // one hue
  });
});

describe('decimation', () => {
  it('draws every cell when they are already far enough apart', () => {
    expect(decimation(30, 26)).toBe(1);
    expect(decimation(26, 26)).toBe(1);
  });

  it('skips cells as they crowd together', () => {
    expect(decimation(13, 26)).toBe(2);
    expect(decimation(2, 26)).toBe(13);
  });

  it('never returns zero or a negative, which would hang or skip everything', () => {
    for (const spacing of [0, -5, 1e-9, NaN, Infinity]) {
      expect(decimation(spacing)).toBeGreaterThanOrEqual(1);
    }
  });

  it('keeps on-screen arrow density roughly constant across zooms', () => {
    // Each zoom level doubles the pixel spacing, so the step should roughly
    // halve and the product stay near the target.
    for (const spacing of [1, 2, 4, 8, 16, 32]) {
      const drawn = spacing * decimation(spacing, 26);
      expect(drawn).toBeGreaterThan(13);
      expect(drawn).toBeLessThan(52);
    }
  });
});

describe('the bug this renderer exists to not have', () => {
  /**
   * leaflet-velocity indexed its grid with floorMod(lon - lon0, 360) / dlon,
   * fed from raw map bounds. Leaflet hands out longitudes outside [-180, 180]
   * once the world repeats, and those wrapped back into our window, so copies
   * of the box appeared over the Pacific.
   *
   * This renderer iterates the grid and projects each cell, so the question
   * "which cell is at longitude 278?" is never asked. These pin the difference.
   */
  const grid = new Grid({ lat0: 17, dlat: 0.25, nlat: 77, lon0: -82, dlon: 0.25, nlon: 77 });

  it('the old indexing really did wrap a Pacific longitude into our box', () => {
    const floorMod = (a, n) => a - n * Math.floor(a / n);
    const i = floorMod(278 - grid.lon0, 360) / grid.dlon;
    expect(i).toBe(0);                    // 278 E resolves to our first column
    expect(i).toBeLessThan(grid.nlon);    // ...so the library happily drew it
  });

  it('our cell lookup has no wrapping at all', () => {
    // Every drawn cell comes from an index, and an index maps to exactly one
    // longitude inside the box. There is no inverse lookup to get wrong.
    for (let i = 0; i < grid.nlon; i += 1) {
      expect(grid.lon(i)).toBeGreaterThanOrEqual(-82);
      expect(grid.lon(i)).toBeLessThanOrEqual(-63);
    }
  });

  it('cellAt refuses a point outside the grid instead of wrapping it', () => {
    expect(grid.cellAt(25, 278)).toBeNull();      // the Pacific copy
    expect(grid.cellAt(25, -140)).toBeNull();     // the real Pacific
    expect(grid.cellAt(0, -70)).toBeNull();       // the Amazon latitude
    expect(grid.cellAt(25, -70)).not.toBeNull();  // and the box still works
  });
});
