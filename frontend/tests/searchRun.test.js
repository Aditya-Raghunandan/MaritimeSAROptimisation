/**
 * One doctrinal search, planned and flown with no map (issue #64).
 *
 * Constant forcing throughout, so every expected number can be worked out by hand.
 */

import { describe, expect, it } from 'vitest';

import { SWEEP_WIDTH_M } from '../src/geo.js';
import { M_PER_DEG_LAT, offsetPosition } from '../src/patterns.js';
import { M_PER_DEG, constantSampler } from '../src/pointDrift.js';
import { NM_M, searchEffortM2, sectorRadiusM } from '../src/platform.js';
import {
  ManualFlight, bearingOf, closestApproach, datumErrorM, detect, formatDuration, formatElapsed, freePlan,
  headingFromKeys, helicopterAt, keyDirection, markerPositionAt, planSearch, searchPath,
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

/*
  THE DATUM LINE (#75). Current east for the first hour, then north: the buoy's net drift
  (LKP to datum) points east-north-east while the current AT the datum points north, so the
  two layout rules give different bearings and the test can tell them apart.
*/
describe('Parallel Track and Trackline, laid along the predicted drift', () => {
  const turning = (ms) => ({ current: ms < REPORT + 3600e3 ? [1, 0] : [0, 1], wind: [0, 0] });
  const plan = (patternKind, extra = {}) => planSearch({
    base: baseWest(100), lkp: LKP, reportMs: REPORT, sample: turning, patternKind, ...extra,
  });

  it('orients both along the datum line, not along the drift at the datum', () => {
    const square = plan('expanding_square');
    expect(square.firstBearingDeg).toBeCloseTo(0, 6);
    for (const kind of ['parallel_track', 'trackline_return']) {
      const p = plan(kind);
      expect(p.datumLine.bearingDeg).toBeGreaterThan(60);
      expect(p.firstBearingDeg).toBeCloseTo(p.datumLine.bearingDeg, 9);
      expect(p.bearingSource).toMatch(/datum line/);
    }
  });

  it('runs the trackline from the last known position, through the datum, and as far beyond', () => {
    const p = plan('trackline_return');
    expect(p.datumLine.lengthM).toBeGreaterThan(sectorRadiusM());   // so the line, not the floor
    expect(p.patternArgs.halfLengthM).toBeCloseTo(p.datumLine.lengthM, 9);
  });

  it('gives the parallel track the area one window covers, drawn along the same line', () => {
    const p = plan('parallel_track');
    expect(p.pattern.area.bearingDeg).toBeCloseTo(p.datumLine.bearingDeg, 9);
    expect(p.pattern.area.lengthM).toBeCloseTo(Math.sqrt(searchEffortM2()), 6);
  });

  it('falls back to the drift at the datum, and a minute each way, when the buoy barely moved', () => {
    const p = planSearch({
      base: baseWest(100), lkp: LKP, reportMs: REPORT, sample: constantSampler([0, 0]),
      patternKind: 'trackline_return',
    });
    expect(p.datumLine.lengthM).toBeLessThan(50);
    expect(p.bearingSource).not.toMatch(/datum line/);
    expect(p.patternArgs.halfLengthM).toBeCloseTo(sectorRadiusM(), 9);
  });

  it('lets an explicit first bearing win', () => {
    expect(plan('parallel_track', { firstBearingDeg: 200 }).firstBearingDeg).toBe(200);
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

  it('says a duration in words for a sentence', () => {
    expect(formatDuration(8285)).toBe('2 h 18 min');
    expect(formatDuration(7200)).toBe('2 h');
    expect(formatDuration(1314)).toBe('21 min 54 s');
    expect(formatDuration(600)).toBe('10 min');
    expect(formatDuration(39.4)).toBe('39 s');
    expect(formatDuration(-5)).toBe('0 s');
  });
});

describe('the keyboard', () => {
  it('reads WASD and the arrow keys alike', () => {
    expect(keyDirection('KeyW')).toBe('up');
    expect(keyDirection('ArrowUp')).toBe('up');
    expect(keyDirection('KeyA')).toBe('left');
    expect(keyDirection('ArrowRight')).toBe('right');
    expect(keyDirection('KeyQ')).toBeNull();
  });

  it('turns held keys into a heading, two keys a diagonal', () => {
    expect(headingFromKeys(new Set(['up']))).toBeCloseTo(0, 9);
    expect(headingFromKeys(new Set(['right']))).toBeCloseTo(90, 9);
    expect(headingFromKeys(new Set(['up', 'right']))).toBeCloseTo(45, 9);
    expect(headingFromKeys(new Set(['down', 'left']))).toBeCloseTo(225, 9);
  });

  it('keeps the heading when nothing is held or opposite keys cancel', () => {
    expect(headingFromKeys(new Set())).toBeNull();
    expect(headingFromKeys(new Set(['up', 'down']))).toBeNull();
  });
});

describe('a helicopter flown by hand', () => {
  const START = Date.parse('2019-06-01T06:00Z');
  const spawn = { lat: 26.5, lon: -79.0 };

  function flight(targetAt = () => null) {
    return new ManualFlight(freePlan({ spawn, startMs: START }), targetAt);
  }

  it('starts on the spot, with no base, transit or marker', () => {
    const plan = freePlan({ spawn, startMs: START });
    expect(plan.free).toBe(true);
    expect(plan.arriveS).toBe(0);
    expect(plan.endS).toBe(2700);
    expect(plan.marker).toBeNull();
  });

  it('flies 90 kt along its heading', () => {
    const f = flight();
    f.advance(60, 90);
    const p = f.position();
    const eastM = (p.lon - spawn.lon) * M_PER_DEG_LAT * Math.cos(spawn.lat * Math.PI / 180);
    expect(eastM).toBeCloseTo(46.3 * 60, 0);
    expect(f.lengthM).toBeCloseTo(46.3 * 60, 6);
  });

  it('keeps flying the last heading when given none', () => {
    const f = flight();
    f.advance(30, 0);
    f.advance(30, null);
    expect(f.heading).toBe(0);
    expect((f.position().lat - spawn.lat) * M_PER_DEG_LAT).toBeCloseTo(46.3 * 60, 0);
  });

  it('stores a straight run as one segment, and a turn as a new one', () => {
    const f = flight();
    for (let k = 0; k < 20; k += 1) f.advance(3, 90);
    expect(f.tS).toHaveLength(2);
    f.advance(3, 180);
    expect(f.tS).toHaveLength(3);
  });

  it('never flies past the window', () => {
    const f = flight();
    f.advance(10000, 90);
    expect(f.s).toBeCloseTo(2700, 9);
    expect(f.done).toBe(true);
  });

  it('finds a target it passes over, and stops there', () => {
    const buoy = offsetPosition(spawn.lat, spawn.lon, 1000, 0);
    const f = flight(() => buoy);
    f.advance(600, 90);
    expect(f.found).toBe(true);
    expect(f.foundS).toBeCloseTo((1000 - SWEEP_WIDTH_M / 2) / 46.3, 0);
    expect(f.done).toBe(true);
  });

  it('reports the closest pass of a miss', () => {
    const buoy = offsetPosition(spawn.lat, spawn.lon, 1000, 500);
    const f = flight(() => buoy);
    f.advance(120, 90);
    expect(f.found).toBe(false);
    expect(f.closestM).toBeCloseTo(500, -1);
  });

  it('replays: where it was, and the path so far, at any earlier moment', () => {
    const f = flight();
    f.advance(60, 90);
    f.advance(60, 0);
    const mid = f.positionAt(30);
    const eastM = (mid.lon - spawn.lon) * M_PER_DEG_LAT * Math.cos(spawn.lat * Math.PI / 180);
    expect(eastM).toBeCloseTo(46.3 * 30, 0);
    expect(mid.heading).toBeCloseTo(90, 3);
    expect(f.pathUpTo(90)).toHaveLength(3);   // spawn, the turn, and where it was at 90 s
  });

  it('refuses a plan that has a pattern', () => {
    expect(() => new ManualFlight(still(), () => null)).toThrow(/no pattern/);
  });
});
