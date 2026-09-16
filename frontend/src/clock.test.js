/**
 * The clock's job is to hold ONE moment that layers of different cadences can
 * all answer to. These tests are mostly about that: an hourly wind field and a
 * 3-hourly current field asked for the same instant must land on frames that
 * represent the same instant, or the current will visibly lag the wind across
 * the same map.
 */

import { describe, expect, it, vi } from 'vitest';
import { Clock } from './clock.js';

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
