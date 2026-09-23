/**
 * Tests for the drifter layer's logic (issue #50). No map, no DOM, no network.
 *
 * The rules these pin down are the ones a reader would notice being wrong: a buoy
 * drawn across a week-long gap, a trail the wrong colour after its drogue is lost,
 * a search that cannot find a buoy because the window is elsewhere, a click that
 * lands the clock on a window that cuts the track in half.
 */

import { describe, expect, it } from 'vitest';

import {
  decodeTrack, describeBuoy, inWindow, lastFixAtOrBefore, lifetimeBar, positionAt,
  prepareIndex, searchBuoys, spanCovering, spanningNow, trailUpTo, undroguedAt,
} from '../src/drifters.js';

const H = 3600 * 1000;
const T0 = Date.parse('2021-06-01T00:00:00Z');

/** A track file as the exporter writes it. */
function file(overrides = {}) {
  return {
    id: 'A', t0: '2021-06-01T00:00:00Z',
    h: [0, 1, 2, 3, 4],
    lat: [28.0, 28.1, 28.2, 28.3, 28.4],
    lon: [-75.0, -74.9, -74.8, -74.7, -74.6],
    seg: [0, 0, 0, 0, 0],
    und: [0, 0, 0, 0, 0],
    far: [0, 0, 0, 0, 0],
    ...overrides,
  };
}

const entry = (id, start, end, extra = {}) => ({
  id, start, end, lat0: 28, lon0: -75, tier: 'undrogued', sealed: false, ...extra,
});

describe('decodeTrack', () => {
  it('turns whole hours since t0 into epoch milliseconds', () => {
    const t = decodeTrack(file());
    expect(Array.from(t.t)).toEqual([0, 1, 2, 3, 4].map((h) => T0 + h * H));
  });

  it('refuses a file whose columns disagree in length, naming the column', () => {
    expect(() => decodeTrack(file({ lat: [28] }))).toThrow(/lat has 1 values for 5 fixes/);
  });
});

describe('prepareIndex, inWindow and spanningNow', () => {
  const index = prepareIndex({
    buoys: [
      entry('B', '2021-06-03T00:00:00Z', '2021-06-05T00:00:00Z'),
      entry('A', '2021-06-01T00:00:00Z', '2021-06-02T00:00:00Z'),
    ],
  });

  it('sorts by first fix and parses the times once', () => {
    expect(index.map((e) => e.id)).toEqual(['A', 'B']);
    expect(index[0].startMs).toBe(T0);
  });

  it('keeps a buoy whose record overlaps the window at all', () => {
    const win = [Date.parse('2021-06-01T12:00:00Z'), Date.parse('2021-06-03T06:00:00Z')];
    expect(inWindow(index, ...win).map((e) => e.id)).toEqual(['A', 'B']);
  });

  it('drops a buoy entirely outside the window, and the window end is exclusive', () => {
    const win = [Date.parse('2021-06-02T06:00:00Z'), Date.parse('2021-06-03T00:00:00Z')];
    expect(inWindow(index, ...win)).toEqual([]);
  });

  it('knows which buoys span a moment', () => {
    expect(spanningNow(index, Date.parse('2021-06-04T00:00:00Z')).map((e) => e.id)).toEqual(['B']);
  });
});

describe('searchBuoys', () => {
  const index = prepareIndex({
    buoys: [
      entry('300234061391930', '2019-01-01T00:00:00Z', '2019-03-02T00:00:00Z'),
      entry('300534061605940', '2021-03-10T00:00:00Z', '2021-04-02T00:00:00Z'),
    ],
  });

  it('finds a buoy by a fragment of its ID', () => {
    expect(searchBuoys(index, '605940').map((e) => e.id)).toEqual(['300534061605940']);
  });

  it('finds a buoy by a month its record overlaps', () => {
    expect(searchBuoys(index, '2021-03').map((e) => e.id)).toEqual(['300534061605940']);
    expect(searchBuoys(index, '2019-02-14').map((e) => e.id)).toEqual(['300234061391930']);
  });

  it('for a date, lists buoys first seen then before long-lived ones passing through', () => {
    const more = prepareIndex({
      buoys: [
        entry('OLD', '2019-01-01T00:00:00Z', '2021-03-31T00:00:00Z'),
        entry('NEW', '2021-03-10T00:00:00Z', '2021-04-02T00:00:00Z'),
      ],
    });
    expect(searchBuoys(more, '2021-03').map((e) => e.id)).toEqual(['NEW', 'OLD']);
  });

  it('an empty query finds nothing rather than everything', () => {
    expect(searchBuoys(index, '   ')).toEqual([]);
  });
});

describe('spanCovering', () => {
  const DAY = 86400;
  const SPANS = [DAY, 3 * DAY, 7 * DAY, 30 * DAY, 365 * DAY];

  it('picks a day for a track inside one UTC day', () => {
    expect(spanCovering(T0 + 2 * H, T0 + 20 * H, SPANS)).toBe(DAY);
  });

  it('steps up when the track crosses the aligned boundary', () => {
    // 20:00 to 04:00 next day: 8 hours, but two UTC days.
    expect(spanCovering(T0 + 20 * H, T0 + 28 * H, SPANS)).not.toBe(DAY);
  });

  it('the window it picks really does contain the whole track', () => {
    const start = Date.parse('2021-03-05T14:00:00Z');
    const end = Date.parse('2021-03-18T02:00:00Z');
    const s = spanCovering(start, end, SPANS) * 1000;
    const from = Math.floor(start / s) * s;
    expect(start >= from && end < from + s).toBe(true);
  });

  it('falls back to the whole archive when nothing smaller holds it', () => {
    expect(spanCovering(Date.parse('2019-06-01'), Date.parse('2021-06-01'), SPANS)).toBeNull();
  });
});

describe('positionAt', () => {
  it('is the fix itself at a fix time', () => {
    expect(positionAt(decodeTrack(file()), T0 + 2 * H)).toEqual([28.2, -74.8]);
  });

  it('interpolates between two fixes of one segment', () => {
    const [lat, lon] = positionAt(decodeTrack(file()), T0 + 1.5 * H);
    expect(lat).toBeCloseTo(28.15, 10);
    expect(lon).toBeCloseTo(-74.85, 10);
  });

  it('is null inside a gap between segments, not a point drawn across it', () => {
    const t = decodeTrack(file({ h: [0, 1, 50, 51, 52], seg: [0, 0, 1, 1, 1] }));
    expect(positionAt(t, T0 + 10 * H)).toBeNull();
  });

  it('is null before the first fix and after the last', () => {
    const t = decodeTrack(file());
    expect(positionAt(t, T0 - H)).toBeNull();
    expect(positionAt(t, T0 + 9 * H)).toBeNull();
  });

  it('binary search finds the last fix at or before a moment', () => {
    const t = decodeTrack(file());
    expect(lastFixAtOrBefore(t, T0 + 2.9 * H)).toBe(2);
    expect(lastFixAtOrBefore(t, T0 - 1)).toBe(-1);
  });
});

describe('trailUpTo', () => {
  it('changes colour at the fix where the drogue is lost', () => {
    const t = decodeTrack(file({ und: [0, 0, 1, 1, 1] }));
    const pieces = trailUpTo(t, T0 + 4 * H);
    expect(pieces.map((p) => p.undrogued)).toEqual([false, true]);
    // The second piece starts where the first ends: one continuous line, two colours.
    expect(pieces[1].points[0]).toEqual(pieces[0].points[pieces[0].points.length - 1]);
  });

  it('dashes the steps that touch a fix far from a real GPS fix', () => {
    const t = decodeTrack(file({ far: [0, 0, 0, 1, 0] }));
    expect(trailUpTo(t, T0 + 4 * H).map((p) => p.far)).toEqual([false, true]);
  });

  it('never draws a step across a gap between segments', () => {
    const t = decodeTrack(file({ h: [0, 1, 50, 51, 52], seg: [0, 0, 1, 1, 1] }));
    const pieces = trailUpTo(t, T0 + 52 * H);
    expect(pieces).toHaveLength(2);
    expect(pieces[0].points).toHaveLength(2);     // fixes 0-1 only
  });

  it('grows with time and ends at the interpolated position now', () => {
    const t = decodeTrack(file());
    const early = trailUpTo(t, T0 + 1 * H)[0].points.length;
    const later = trailUpTo(t, T0 + 2.5 * H)[0].points;
    expect(later.length).toBeGreaterThan(early);
    expect(later[later.length - 1][0]).toBeCloseTo(28.25, 10);
  });

  it('is empty before the buoy is in the water', () => {
    expect(trailUpTo(decodeTrack(file()), T0 - H)).toEqual([]);
  });

  it('knows the drogue state at a moment', () => {
    const t = decodeTrack(file({ und: [0, 0, 1, 1, 1] }));
    expect(undroguedAt(t, T0 + 1.5 * H)).toBe(false);
    expect(undroguedAt(t, T0 + 2 * H)).toBe(true);
  });
});

describe('lifetimeBar', () => {
  const win = [T0, T0 + 10 * H];

  it('places the record as fractions of the window', () => {
    const bar = lifetimeBar(T0 + 2 * H, T0 + 7 * H, ...win);
    expect(bar.left).toBeCloseTo(0.2, 12);
    expect(bar.width).toBeCloseTo(0.5, 12);
  });

  it('clips a record that runs past either edge', () => {
    expect(lifetimeBar(T0 - 5 * H, T0 + 20 * H, ...win)).toEqual({ left: 0, width: 1 });
  });

  it('is null when the record is outside the window', () => {
    expect(lifetimeBar(T0 + 20 * H, T0 + 30 * H, ...win)).toBeNull();
  });
});

describe('describeBuoy', () => {
  it('says the dates, the tier in words, and whether it is sealed', () => {
    const [e] = prepareIndex({
      buoys: [entry('A', '2021-06-01T00:00:00Z', '2021-06-11T00:00:00Z', { sealed: true, tier: 'mixed' })],
    });
    const d = describeBuoy(e);
    expect(d.dates).toBe('2021-06-01 → 2021-06-11 · 10 days');
    expect(d.tier).toMatch(/loses its drogue/);
    expect(d.sealed).toMatch(/do not debug the engine/);
  });
});
