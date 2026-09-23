/**
 * Tests for the leeway arithmetic shown in the point panel.
 *
 * These matter more than they look: the panel states a physical claim about
 * where something would drift, and a sign error in the bearing sends it the
 * opposite way while still reading as a perfectly sensible number.
 */

import { describe, expect, it } from 'vitest';

import {
  ALPHA, bearingFrom, bearingTowards, compass, currentBand, driftBand, explain, leewayDistance, leewayFractionOfCurrent, leewaySpeed, speed, summarise, sweepWidthMinutes,
} from '../src/drift.js';

describe('ALPHA', () => {
  it('is 2 %, matching the Python side', () => {
    // src/sar/fetch/wind.py ALPHA_MID = 0.02, cited to Allen (2000). If these
    // two ever disagree, the map and the engine are modelling different things.
    expect(ALPHA).toBe(0.02);
  });
});

describe('bearings', () => {
  // Meteorological convention: the direction the wind comes FROM.
  it('a wind blowing towards the east comes from the west', () => {
    expect(bearingFrom(5, 0)).toBeCloseTo(270);
    expect(compass(bearingFrom(5, 0))).toBe('W');
  });

  it('a wind blowing towards the north comes from the south', () => {
    expect(bearingFrom(0, 5)).toBeCloseTo(180);
    expect(compass(bearingFrom(0, 5))).toBe('S');
  });

  it('a drifter goes the way the wind is going, not the way it is named', () => {
    // The whole point of having both: "a westerly" moves things EAST.
    expect(bearingTowards(5, 0)).toBeCloseTo(90);
    expect(compass(bearingTowards(5, 0))).toBe('E');
  });

  it('towards is always 180 degrees from', () => {
    for (const [u, v] of [[3, 4], [-3, 4], [-3, -4], [3, -4], [1, 0], [0, -1]]) {
      const diff = Math.abs(bearingTowards(u, v) - bearingFrom(u, v));
      expect(Math.min(diff, 360 - diff)).toBeCloseTo(180);
    }
  });

  it('stays in [0, 360) and never returns exactly 360', () => {
    for (const [u, v] of [[0, 1], [0, -1], [1, 0], [-1, 0], [1e-12, 1]]) {
      const b = bearingFrom(u, v);
      expect(b).toBeGreaterThanOrEqual(0);
      expect(b).toBeLessThan(360);
    }
  });

  it('compass wraps to N rather than falling off the end', () => {
    expect(compass(0)).toBe('N');
    expect(compass(359)).toBe('N');
    expect(compass(360)).toBe('N');
    expect(compass(-1)).toBe('N');
  });
});

describe('leeway', () => {
  it('is 2 % of the wind speed', () => {
    expect(leewaySpeed(10)).toBeCloseTo(0.2);
    expect(speed(3, 4)).toBe(5);
  });

  it('carries a metre per second 3.6 km in an hour', () => {
    expect(leewayDistance(50, 1)).toBeCloseTo(3600);   // alpha*50 = 1 m/s
  });

  it('reproduces the number that justifies the three-term model', () => {
    // D002/R1: against a 1.8 m/s Gulf Stream, a 10 m/s wind contributes about
    // 11 % -- not negligible -- while a 3 m/s wind contributes about 3 %.
    expect(leewayFractionOfCurrent(10, 1.8) * 100).toBeCloseTo(11.1, 1);
    expect(leewayFractionOfCurrent(3, 1.8) * 100).toBeCloseTo(3.3, 1);
  });

  it('refuses to divide by a zero current instead of returning Infinity', () => {
    expect(leewayFractionOfCurrent(10, 0)).toBeNull();
    expect(leewayFractionOfCurrent(10, -1)).toBeNull();
  });
});

describe('summarise', () => {
  it('ignores the NaNs that mark unloaded frames', () => {
    const s = summarise([1, NaN, 3, NaN, 5]);
    expect(s.n).toBe(3);
    expect(s.mean).toBe(3);
    expect(s.min).toBe(1);
    expect(s.max).toBe(5);
  });

  it('reports nothing rather than NaN when nothing is loaded', () => {
    // A panel showing "NaN m/s" is worse than one showing a dash.
    const s = summarise([NaN, NaN]);
    expect(s).toEqual({ n: 0, mean: null, min: null, max: null });
    expect(summarise([]).mean).toBeNull();
  });

  it('handles a Float32Array, which is what actually arrives', () => {
    expect(summarise(new Float32Array([2, 4, 6])).mean).toBe(4);
  });
});

describe('currentBand', () => {
  it('names the bands a drift argument is made in, not Beaufort', () => {
    // Beaufort is a wind scale. A current legend that reads "F4 moderate
    // breeze" is describing the wrong fluid.
    expect(currentBand(0.1)).toMatch(/weak/);
    expect(currentBand(0.35)).toBe('moderate');
    expect(currentBand(0.8)).toBe('strong');
    expect(currentBand(1.3)).toMatch(/swift/);
    expect(currentBand(2.0)).toBe('Gulf Stream core');
  });

  it('puts the Gulf Stream core where the Gulf Stream actually is', () => {
    // D001 gives the core as about 1.8 m/s; the band has to contain it or the
    // legend disagrees with the region the project is about.
    expect(currentBand(1.8)).toBe('Gulf Stream core');
  });

  it('does not crash on a land cell', () => {
    expect(currentBand(NaN)).toBe('—');
    expect(currentBand(undefined)).toBe('—');
  });
});

/*
  The sweep-width comparison is the argument the whole project rests on: an
  effective visual sweep width for a person in the water is about 185 m, and the
  Gulf Stream runs 1-2.5 m/s, so a target crosses the full detectable width of a
  search track in a minute or two. These helpers put that in the panel, which is
  why they are phrased in minutes rather than in metres per second.
*/
describe('sweepWidthMinutes', () => {
  it('is the time to cross one sweep width', () => {
    // 185 m at 1 m/s is 185 s, which is 3.08 minutes.
    expect(sweepWidthMinutes(1)).toBeCloseTo(185 / 60, 6);
    // The Gulf Stream core: the number that motivates the project.
    expect(sweepWidthMinutes(1.8)).toBeLessThan(2);
  });

  it('is null where the question has no answer', () => {
    expect(sweepWidthMinutes(0)).toBeNull();
    expect(sweepWidthMinutes(-1)).toBeNull();
    expect(sweepWidthMinutes(NaN)).toBeNull();
  });

  it('takes a different sweep width when one is given', () => {
    expect(sweepWidthMinutes(1, 370)).toBeCloseTo(2 * sweepWidthMinutes(1), 6);
  });
});

describe('driftBand', () => {
  it('cuts its bands where the search consequence changes', () => {
    expect(driftBand(0.02)).toMatch(/stationary/);
    expect(driftBand(0.5)).toMatch(/moderate/);
    expect(driftBand(2.0)).toMatch(/jet speed/);
  });

  it('names the sweep width once the target is actually moving', () => {
    // A number is not an argument. "Out of the swept lane in four minutes" is.
    expect(driftBand(0.8)).toMatch(/sweep width every/);
    expect(driftBand(0.02)).not.toMatch(/sweep width/);
  });

  it('has an answer for a value that is not a number', () => {
    expect(driftBand(NaN)).toBe('—');
  });
});

describe('explain, over land', () => {
  it('describes the wind as what it would do offshore, not here', () => {
    const got = explain(10, 45, 1.8, 185, { overLand: true });
    expect(got.lead).toMatch(/^Over open water/);
    expect(got.caveat).toMatch(/no sea at this point/);
    // No "current here" comparison: there is no current here.
    expect(got.caveat).not.toMatch(/current/);
  });

  it('is unchanged at sea, so nothing else on the panel moves', () => {
    expect(explain(10, 45, 1.8, 185).lead).toMatch(/^Wind alone would carry a drifter/);
  });
});

