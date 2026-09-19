/**
 * Tests for the Beaufort mapping shown in the point panel.
 *
 * The band boundaries are the WMO scale, so getting one wrong misreports the
 * sea a searcher would be looking at -- and the panel draws a detection
 * caveat off the same number.
 */

import { describe, expect, it } from 'vitest';

import { BEAUFORT, beaufort, describe as describeWind, detectionOutlook } from '../src/beaufort.js';

describe('the scale itself', () => {
  it('runs 0 to 12 with no gaps', () => {
    expect(BEAUFORT.map((b) => b.force)).toEqual([...Array(13).keys()]);
  });

  it('has strictly increasing bounds and wave heights', () => {
    for (let k = 1; k < BEAUFORT.length; k += 1) {
      expect(BEAUFORT[k].max).toBeGreaterThan(BEAUFORT[k - 1].max);
      expect(BEAUFORT[k].waveM).toBeGreaterThanOrEqual(BEAUFORT[k - 1].waveM);
    }
  });

  it('is open-ended at the top so no wind falls off the end', () => {
    expect(BEAUFORT[BEAUFORT.length - 1].max).toBe(Infinity);
  });
});

describe('beaufort', () => {
  it('puts the WMO boundaries in the right band', () => {
    expect(beaufort(0).force).toBe(0);
    expect(beaufort(0.4).force).toBe(0);
    expect(beaufort(0.5).force).toBe(1);
    expect(beaufort(3.3).force).toBe(2);
    expect(beaufort(3.4).force).toBe(3);
    expect(beaufort(5.4).force).toBe(3);
    expect(beaufort(5.5).force).toBe(4);
    expect(beaufort(10.7).force).toBe(5);
    expect(beaufort(10.8).force).toBe(6);    // strong breeze starts
    expect(beaufort(13.8).force).toBe(6);
    expect(beaufort(13.9).force).toBe(7);    // near gale starts
    expect(beaufort(17.1).force).toBe(7);
    expect(beaufort(17.2).force).toBe(8);    // gale starts
    expect(beaufort(32.7).force).toBe(12);
  });

  it('clamps rather than returning undefined for nonsense', () => {
    // A NaN here would put "undefined" in the panel.
    expect(beaufort(NaN).force).toBe(0);
    expect(beaufort(-5).force).toBe(0);
    expect(beaufort(1e6).force).toBe(12);
  });

  it('the screenshot case reads as a gentle breeze', () => {
    // 3.8 m/s at 28.25 N, 70.00 W, 2021-01-02 22:00 UTC.
    expect(describeWind(3.8)).toBe('Force 3 · gentle breeze');
  });
});

describe('detectionOutlook', () => {
  it('is good in a calm sea and poor in a rough one', () => {
    expect(detectionOutlook(2).level).toBe('good');
    expect(detectionOutlook(9).level).toBe('degraded');
    expect(detectionOutlook(20).level).toBe('poor');
  });

  it('never gets better as the wind gets up', () => {
    const rank = { good: 0, degraded: 1, poor: 2 };
    let worst = -1;
    for (let s = 0; s <= 40; s += 0.5) {
      const r = rank[detectionOutlook(s).level];
      expect(r).toBeGreaterThanOrEqual(worst);
      worst = r;
    }
  });

  it('says something for every band', () => {
    for (const b of BEAUFORT) {
      const o = detectionOutlook(b.max === Infinity ? 40 : b.max - 0.1);
      expect(o.text.length).toBeGreaterThan(10);
    }
  });
});
