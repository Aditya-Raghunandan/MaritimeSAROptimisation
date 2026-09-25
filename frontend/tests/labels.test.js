/**
 * Map labels never sit on each other (issue #86).
 */

import { describe, expect, it } from 'vitest';

import { candidates, overlap, placeLabels } from '../src/labels.js';

const box = (id, placement, l) => ({ x: l.x + placement[id].dx, y: l.y + placement[id].dy, w: l.w, h: l.h });

describe('placeLabels', () => {
  it('puts a lone label to the right of its point', () => {
    const p = placeLabels([{ id: 'a', x: 100, y: 100, w: 60, h: 18, priority: 1 }]);
    expect(p.a).toEqual({ dx: 11, dy: -9, index: 0, hidden: false });
  });

  it('separates labels whose points coincide, as the buoy, marker and datum can', () => {
    const labels = [
      { id: 'buoy', x: 200, y: 200, w: 60, h: 18, priority: 1 },
      { id: 'datum', x: 202, y: 201, w: 150, h: 31, priority: 2 },
      { id: 'marker', x: 199, y: 203, w: 50, h: 18, priority: 3 },
      { id: 'lkp', x: 205, y: 198, w: 110, h: 18, priority: 4 },
    ];
    const p = placeLabels(labels);
    const boxes = labels.map((l) => box(l.id, p, l));
    for (let i = 0; i < boxes.length; i += 1) {
      for (let j = i + 1; j < boxes.length; j += 1) expect(overlap(boxes[i], boxes[j])).toBe(0);
    }
  });

  it('keeps labels off the markers themselves', () => {
    const heli = { x: 105, y: 85, w: 26, h: 26 };
    const p = placeLabels([{ id: 'a', x: 100, y: 100, w: 60, h: 18, priority: 1 }], [heli]);
    expect(overlap(box('a', p, { x: 100, y: 100, w: 60, h: 18 }), heli)).toBe(0);
  });

  it('places the most important label first, where it wants to be', () => {
    const labels = [
      { id: 'minor', x: 100, y: 100, w: 60, h: 18, priority: 5 },
      { id: 'major', x: 100, y: 100, w: 60, h: 18, priority: 1 },
    ];
    expect(placeLabels(labels).major.index).toBe(0);
  });

  it('keeps last frame\'s spot while it is clear, so labels do not flicker', () => {
    const l = { id: 'a', x: 100, y: 100, w: 60, h: 18, priority: 1 };
    expect(placeLabels([l], [], null, { a: 3 }).a.index).toBe(3);
  });

  it('prefers a spot on screen', () => {
    const p = placeLabels([{ id: 'a', x: 780, y: 300, w: 80, h: 18, priority: 1 }], [], { x: 0, y: 0, w: 800, h: 600 });
    expect(780 + p.a.dx).toBeLessThan(780);                 // it went left, not off the edge
  });

  it('offers four rings of eight positions', () => {
    expect(candidates(60, 18)).toHaveLength(32);
  });

  it('spares an unimportant label with nowhere clear to go, rather than overlap', () => {
    // Boxed in: every spot round the point is covered.
    const wall = [{ x: -1000, y: -1000, w: 2000, h: 990 }, { x: -1000, y: 10, w: 2000, h: 990 },
      { x: -1000, y: -1000, w: 990, h: 2000 }, { x: 10, y: -1000, w: 990, h: 2000 }];
    const p = placeLabels([{ id: 'minor', x: 0, y: 0, w: 60, h: 18, priority: 5 }], wall, null, {}, 5);
    expect(p.minor.hidden).toBe(true);
    const q = placeLabels([{ id: 'major', x: 0, y: 0, w: 60, h: 18, priority: 1 }], wall, null, {}, 5);
    expect(q.major.hidden).toBe(false);
  });
});
