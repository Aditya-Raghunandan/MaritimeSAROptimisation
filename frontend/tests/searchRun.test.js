/**
 * One doctrinal search, planned and flown with no map (issue #64).
 *
 * Constant forcing throughout, so every expected number can be worked out by hand.
 */

import { describe, expect, it } from 'vitest';

import { SWEEP_WIDTH_M } from '../src/geo.js';
import { M_PER_DEG_LAT, offsetPosition } from '../src/patterns.js';
import { M_PER_DEG, constantSampler } from '../src/pointDrift.js';
import { NM_M } from '../src/platform.js';
import {
  bearingOf, closestApproach, datumErrorM, detect, formatElapsed, helicopterAt,
  markerPositionAt, planSearch, searchPath,
} from '../src/searchRun.js';

const REPORT = Date.parse('2019-06-01T06:00Z');
const LKP = { lat: 26.5, lon: -79.0 };
const W2 = SWEEP_WIDTH_M / 2;

/** A base `nm` nautical miles due west of the LKP. */
function baseWest(nm) {
  return { lat: LKP.lat, lon: LKP.lon - (nm * NM_M) / (M_PER_DEG * Math.cos(LKP.lat * Math.PI / 180)) };
}

function still() {
  return planSearch({
    base: baseWest(100), lkp: LKP, reportMs: REPORT, sample: constantSampler([0, 0]),
    firstBearingDeg: 0,
  });
}

describe('planSearch', () => {
  it('launches after 30 minutes and flies 100 NM at 125 kt', () => {
    const plan = still();
    expect(plan.launchS).toBe(1800);
    // 100 NM measured along the parallel; the great circle is a touch shorter.
    expect(Math.abs(plan.arriveS - (1800 + 2880))).toBeLessThan(1);
    expect(plan.endS).toBeCloseTo(plan.arriveS + 2700, 6);
  });

  it('puts the datum on the LKP when nothing is moving', () => {
    const plan = still();
    expect(plan.datum.lat).toBeCloseTo(LKP.lat, 12);
    expect(plan.datum.lon).toBeCloseTo(LKP.lon, 12);
  });

  it('carries the datum with current and leeway, and agrees with its own arrival time', () => {
    const plan = planSearch({
      base: baseWest(100), lkp: LKP, reportMs: REPORT, targetLeeway: 0.02,
      sample: constantSampler([1.0, 0.0], [0.0, 10.0]),
    });
    // East at 1 m/s, north at 0.2 m/s, for the whole time to arrival.
    const northM = (plan.datum.lat - LKP.lat) * M_PER_DEG;
    expect(northM).toBeCloseTo(0.2 * plan.arriveS, 0);
    // Flying further east to meet it makes the trip longer than to the LKP.
    expect(plan.arriveS).toBeGreaterThan(1800 + 2880);
  });

  it('flies the first leg along the drift unless told otherwise', () => {
    const plan = planSearch({
      base: baseWest(50), lkp: LKP, reportMs: REPORT, sample: constantSampler([1, 1]),
    });
    expect(plan.firstBearingDeg).toBeCloseTo(45, 9);
    expect(plan.bearingSource).toMatch(/drift/);
  });

  it('refuses a datum beyond the radius of action', () => {
    const plan = planSearch({
      base: baseWest(301), lkp: LKP, reportMs: REPORT, sample: constantSampler([0, 0]),
    });
    expect(plan.error).toMatch(/300 NM/);
  });

  it('stops the datum where the current runs out, and says so', () => {
    let calls = 0;
    const sample = () => (calls++ < 3 ? { current: [1, 0], wind: [0, 0] } : { current: null, reason: 'reached land' });
    const plan = planSearch({ base: baseWest(20), lkp: LKP, reportMs: REPORT, sample, firstBearingDeg: 0 });
    expect(plan.notes.join(' ')).toMatch(/land/);
  });
});

describe('the helicopter through the search', () => {
  it('waits at the base, flies out, then searches about the marker', () => {
    const plan = still();
    expect(helicopterAt(plan, 0).phase).toBe('ready');
    expect(helicopterAt(plan, 1800 + 1).phase).toBe('transit');
    const arrive = helicopterAt(plan, plan.arriveS);
    expect(arrive.phase).toBe('search');
    expect(arrive.lat).toBeCloseTo(LKP.lat, 9);
    expect(arrive.lon).toBeCloseTo(LKP.lon, 9);
    expect(helicopterAt(plan, plan.endS + 60).phase).toBe('done');
  });

  it('carries the whole pattern with a drifting marker', () => {
    const plan = planSearch({
      base: baseWest(50), lkp: LKP, reportMs: REPORT, sample: constantSampler([1, 0]),
      firstBearingDeg: 0,
    });
    const m0 = markerPositionAt(plan, plan.arriveS);
    const m1 = markerPositionAt(plan, plan.endS);
    const eastM = (m1.lon - m0.lon) * M_PER_DEG * Math.cos(m0.lat * Math.PI / 180);
    expect(eastM).toBeCloseTo(2700, 0);
    expect(markerPositionAt(plan, plan.arriveS - 1)).toBeNull();
  });

  it('holds every waypoint in its ground path', () => {
    const plan = still();
    const path = searchPath(plan);
    expect(path.tS[0]).toBeCloseTo(plan.arriveS, 9);
    expect(path.tS.length).toBeGreaterThan(plan.pattern.tS.length);
  });
});

describe('closestApproach', () => {
  it('catches a target that passes straight through between two samples', () => {
    // 100 m away at both ends -- outside W/2 each time -- but through zero in between.
    const hit = closestApproach([-100, 0], [100, 0], W2);
    expect(hit.closest).toBeCloseTo(0, 9);
    expect(hit.contact).toBeCloseTo((100 - W2) / 200, 9);
  });

  it('reports the closest pass of a miss', () => {
    const hit = closestApproach([-100, 150], [100, 150], W2);
    expect(hit.contact).toBeNull();
    expect(hit.closest).toBeCloseTo(150, 9);
  });

  it('is in contact from the start when already inside', () => {
    expect(closestApproach([10, 0], [500, 0], W2).contact).toBe(0);
  });
});

describe('detect', () => {
  it('finds a still target on the first leg', () => {
    const plan = still();   // first leg north from the LKP, S = W long
    const target = offsetPosition(LKP.lat, LKP.lon, 0, 100);
    const res = detect(plan, () => target);
    expect(res.found).toBe(true);
    expect(res.foundS - plan.arriveS).toBeLessThan(5);
  });

  it('misses a target far outside the pattern, and says by how much', () => {
    const plan = still();
    const target = offsetPosition(LKP.lat, LKP.lon, 20000, 0);
    const res = detect(plan, () => target);
    expect(res.found).toBe(false);
    expect(res.closestM).toBeGreaterThan(15000);
  });

  it('catches a fast target crossing a leg between path points', () => {
    const plan = still();
    // Crosses the first leg's line (east = 0) eastward at 20 m/s, level with the
    // helicopter as it passes, starting 100 m west.
    const t0 = plan.reportMs + plan.arriveS * 1000;
    const target = (ms) => {
      const s = (ms - t0) / 1000;
      return offsetPosition(LKP.lat, LKP.lon, -100 + 20 * s, 46.3 * 2);
    };
    expect(detect(plan, target).found).toBe(true);
  });

  it('counts intervals where the target has no position as gaps, not misses', () => {
    const res = detect(still(), () => null);
    expect(res.found).toBe(false);
    expect(res.gaps).toBeGreaterThan(0);
  });

  it('measures the datum error against where the target really was', () => {
    const plan = still();
    const target = offsetPosition(LKP.lat, LKP.lon, 0, 1000);
    expect(datumErrorM(plan, () => target)).toBeCloseTo(1000 * (M_PER_DEG / M_PER_DEG_LAT), -1);
  });
});

describe('helpers', () => {
  it('reads a velocity as a compass bearing', () => {
    expect(bearingOf(1, 0)).toBeCloseTo(90, 12);
    expect(bearingOf(0, -1)).toBeCloseTo(180, 12);
    expect(bearingOf(0, 0)).toBeNull();
  });

  it('formats elapsed time for the read-out', () => {
    expect(formatElapsed(65)).toBe('T+1:05');
    expect(formatElapsed(4620)).toBe('T+1:17:00');
  });
});
