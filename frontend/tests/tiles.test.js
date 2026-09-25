/**
 * A missing basemap tile drawn from the zoom above (#76): which parent, and where.
 */

import { describe, expect, it } from 'vitest';

import { fallbackPlacement } from '../src/tileFallback.js';
import { overlapPx } from '../src/cornerGuard.js';

describe('fallbackPlacement', () => {
  it('draws a tile from the right quarter of its parent', () => {
    // x odd, y even: the right-hand, top quarter of the tile one zoom out.
    const p = fallbackPlacement({ x: 4585, y: 6702, z: 14 }, 1);
    expect(p.parent).toEqual({ x: 2292, y: 3351, z: 13 });
    expect(p.size).toBe(512);
    expect([p.marginLeft, p.marginTop]).toEqual([-256, 0]);
    // top, right, bottom, left: only the top-right square shows.
    expect(p.clip).toEqual([0, 0, 256, 256]);
  });

  it('reaches further out one zoom at a time, and always shows exactly one tile', () => {
    const c = { x: 37, y: 90, z: 16 };
    for (let up = 1; up <= 6; up += 1) {
      const p = fallbackPlacement(c, up);
      expect(p.parent.z).toBe(16 - up);
      expect(p.size).toBe(256 * 2 ** up);
      const [top, right, bottom, left] = p.clip;
      expect(p.size - left - right).toBe(256);
      expect(p.size - top - bottom).toBe(256);
      // The visible square lands on the tile's own place.
      expect(p.marginLeft + left).toBe(0);
      expect(p.marginTop + top).toBe(0);
    }
  });
});

describe('overlapPx', () => {
  it('is positive only when the stacks come within the gap of each other', () => {
    expect(overlapPx(400, 500)).toBeLessThan(0);
    expect(overlapPx(495, 500)).toBe(3);
    expect(overlapPx(600, 500, 0)).toBe(100);
  });
});
