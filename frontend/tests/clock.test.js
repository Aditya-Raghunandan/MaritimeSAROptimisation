/**
 * The clock's job is to hold ONE moment that layers of different cadences can
 * all answer to. These tests are mostly about that: an hourly wind field and a
 * 3-hourly current field asked for the same instant must land on frames that
 * represent the same instant, or the current will visibly lag the wind across
 * the same map.
 */

import { describe, expect, it, vi } from 'vitest';
import { Clock } from '../src/clock.js';

const START = new Date('2021-01-01T00:00:00Z');
const END = new Date('2021-01-03T00:00:00Z');

const hourly = { start: START, stepSeconds: 3600, frames: 48 };
const threeHourly = { start: START, stepSeconds: 10800, frames: 16 };

describe('Clock', () => {
  it('counts steps with the end exclusive', () => {
    // 48 hours at one hour each: 48 positions, 00:00 on the 1st to 23:00 on the 2nd.
    expect(new Clock(START, END, 3600).steps).toBe(48);
  });

  it('builds from a manifest', () => {
    const c = Clock.fromManifest({
      clock: { start: '2021-01-01T00:00:00Z', end: '2021-01-03T00:00:00Z', step_seconds: 3600 },
    });
    expect(c.steps).toBe(48);
    expect(c.index).toBe(0);
  });

  it('clamps an index past the end rather than running off the buffer', () => {
    const c = new Clock(START, END, 3600);
    c.setIndex(999);
    expect(c.index).toBe(47);
  });

  it('clamps a negative index', () => {
    const c = new Clock(START, END, 3600);
    c.setIndex(-5);
    expect(c.index).toBe(0);
  });

  it('notifies listeners once per real change', () => {
    const c = new Clock(START, END, 3600);
    const seen = vi.fn();
    c.onChange(seen);

    c.setIndex(3);
    c.setIndex(3); // same moment -- must not fire again
    c.setIndex(4);

    expect(seen).toHaveBeenCalledTimes(2);
  });

  it('stops notifying after the listener is removed', () => {
    const c = new Clock(START, END, 3600);
    const seen = vi.fn();
    const off = c.onChange(seen);
    off();
    c.setIndex(2);
    expect(seen).not.toHaveBeenCalled();
  });

  describe('frameOf, across mismatched cadences', () => {
    it('gives each layer its own frame for the same instant', () => {
      const c = new Clock(START, END, 3600);
      c.setIndex(6); // 06:00

      expect(c.frameOf(hourly)).toBe(6); // hour 6
      expect(c.frameOf(threeHourly)).toBe(2); // 06:00 is the third 3-hourly stamp
    });

    it('snaps to the NEAREST frame, not the most recent one', () => {
      // 01:30 is closer to 03:00 than to 00:00. Holding 00:00 for another
      // ninety minutes would make the current lag the wind on the same map.
      const c = new Clock(START, END, 1800);
      c.setTime(new Date('2021-01-01T01:30:00Z'));
      expect(c.frameOf(threeHourly)).toBe(1);

      c.setTime(new Date('2021-01-01T01:00:00Z'));
      expect(c.frameOf(threeHourly)).toBe(0);
    });

    it('clamps to a layer that ends before the window does', () => {
      const short = { start: START, stepSeconds: 3600, frames: 4 };
      const c = new Clock(START, END, 3600);
      c.setIndex(40);
      expect(c.frameOf(short)).toBe(3);
    });

    it('clamps to a layer that starts after the window does', () => {
      const late = { start: new Date('2021-01-02T00:00:00Z'), stepSeconds: 3600, frames: 24 };
      const c = new Clock(START, END, 3600);
      c.setIndex(0);
      expect(c.frameOf(late)).toBe(0);
    });
  });

  it('labels in UTC and says so', () => {
    const c = new Clock(START, END, 3600);
    c.setIndex(14);
    expect(c.label()).toBe('2021-01-01 14:00 UTC');
  });
});

/*
  Switching published tier changes the stride, not the moment. This is the
  property that makes the time-resolution control a two-line change rather
  than a re-index of every layer, and it only holds because the clock stores a
  timestamp.
*/
describe('setStep', () => {
  const start = new Date('2021-01-01T00:00:00Z');
  const end = new Date('2021-01-11T00:00:00Z');

  it('keeps the moment when the granularity changes', () => {
    const clock = new Clock(start, end, 86400);
    clock.setIndex(3);                       // 2021-01-04, daily
    const before = clock.t.getTime();

    clock.setStep(3600);                     // to hourly
    expect(clock.t.getTime()).toBe(before);
    expect(clock.label()).toContain('2021-01-04');
  });

  it('re-expresses the same moment as a finer index', () => {
    const clock = new Clock(start, end, 86400);
    clock.setIndex(3);
    expect(clock.index).toBe(3);

    clock.setStep(3600);
    expect(clock.index).toBe(72);            // 3 days = 72 hours
    expect(clock.steps).toBe(240);           // 10 days of hours
  });

  it('notifies listeners, because the slider has to be rescaled', () => {
    const clock = new Clock(start, end, 86400);
    let calls = 0;
    clock.onChange(() => { calls += 1; });
    clock.setStep(3600);
    expect(calls).toBe(1);
  });

  it('ignores a no-op or a nonsense step rather than firing listeners', () => {
    const clock = new Clock(start, end, 86400);
    let calls = 0;
    clock.onChange(() => { calls += 1; });
    clock.setStep(86400);
    clock.setStep(0);
    clock.setStep(-60);
    expect(calls).toBe(0);
    expect(clock.stepSeconds).toBe(86400);
  });
});

/*
  The slider spans a WINDOW, not the archive. Spanning five years at hourly is
  43,824 positions -- one pixel of travel is several hours and a single day
  cannot be stepped through, which made the fine tiers useless the moment they
  became reachable.
*/
describe('window', () => {
  const ARCHIVE_START = new Date('2019-01-01T00:00:00Z');
  const ARCHIVE_END = new Date('2024-01-01T00:00:00Z');
  const at = (step, when) => {
    const c = new Clock(ARCHIVE_START, ARCHIVE_END, step);
    c.setTime(new Date(when));
    return c;
  };

  it('defaults to the whole archive, as it always did', () => {
    const c = new Clock(ARCHIVE_START, ARCHIVE_END, 3600);
    expect(c.steps).toBe(43824);
    expect(c.windowSpanSeconds).toBeNull();
  });

  it('an hourly day is 24 positions', () => {
    const c = at(3600, '2019-05-06T09:00:00Z');
    c.setWindowSpan(86400);
    expect(c.steps).toBe(24);
    expect(c.index).toBe(9);
  });

  it('aligns to the UTC day rather than centring on the cursor', () => {
    const c = at(3600, '2019-05-06T09:00:00Z');
    c.setWindowSpan(86400);
    expect(c.winStart.toISOString()).toBe('2019-05-06T00:00:00.000Z');
    expect(c.winEnd.toISOString()).toBe('2019-05-07T00:00:00.000Z');
    expect(c.windowLabel()).toBe('2019-05-06 UTC');
  });

  it('keeps the moment when the window is applied', () => {
    const c = at(3600, '2019-05-06T09:00:00Z');
    c.setWindowSpan(86400);
    expect(c.t.toISOString()).toBe('2019-05-06T09:00:00.000Z');
  });

  it('shifts a whole window and carries the cursor offset with it', () => {
    const c = at(3600, '2019-05-06T09:00:00Z');
    c.setWindowSpan(86400);
    expect(c.shiftWindow(1)).toBe(true);
    expect(c.t.toISOString()).toBe('2019-05-07T09:00:00.000Z');
    expect(c.index).toBe(9);
    expect(c.windowLabel()).toBe('2019-05-07 UTC');
  });

  it('refuses to shift past the archive, and says so before you try', () => {
    const c = at(3600, '2019-01-01T00:00:00Z');
    c.setWindowSpan(86400);
    expect(c.canShift(-1)).toBe(false);
    expect(c.shiftWindow(-1)).toBe(false);
    expect(c.canShift(1)).toBe(true);
  });

  it('null span returns to the whole archive', () => {
    const c = at(86400, '2020-06-01T00:00:00Z');
    c.setWindowSpan(86400 * 7);
    expect(c.steps).toBe(7);
    c.setWindowSpan(null);
    expect(c.steps).toBe(1826);
    expect(c.windowSpanSeconds).toBeNull();
  });

  it('a multi-day window names both ends', () => {
    const c = at(21600, '2019-05-06T09:00:00Z');
    c.setWindowSpan(7 * 86400);
    expect(c.steps).toBe(28);
    expect(c.windowLabel()).toMatch(/^2019-05-\d\d → 2019-05-\d\d UTC$/);
  });

  it('clamps the cursor into the window rather than leaving it outside', () => {
    const c = at(3600, '2019-05-06T09:00:00Z');
    c.setWindowSpan(86400);
    c.setIndex(999);
    expect(c.index).toBe(23);
    expect(c.t.toISOString()).toBe('2019-05-06T23:00:00.000Z');
  });
});
