/**
 * The browser's doctrinal search, checked against Python case by case (issues #48, #63, #64).
 *
 * `fixtures/search_golden.json` is written by `scripts/export_search_golden.py` from
 * `src/sar/search/`. A failure here means the two implementations have parted company;
 * regenerate the fixture only when the Python is what changed, and read the diff.
 */

import { describe, expect, it } from 'vitest';

import golden from '../src/fixtures/search_golden.json';
import {
  PATTERNS, SECTOR_LEGS, eastNorth, expandingSquare, headingAt, offsetAt, onGround,
  sectorSearch,
} from '../src/patterns.js';
import { constantSampler, driftTrack } from '../src/pointDrift.js';
import { distanceM, transitTimeS } from '../src/platform.js';

const TOL_M = golden.tolerance_m;
const TOL_DEG = golden.tolerance_deg;
const TOL_S = golden.tolerance_s;

function close(got, want, tol) {
  expect(got.length).toBe(want.length);
  got.forEach((g, k) => expect(Math.abs(g - want[k])).toBeLessThanOrEqual(tol));
}

function build(kind, args) {
  const a = args;
  return PATTERNS[kind].build({
    spacingM: a.spacing_m,
    radiusM: a.radius_m ?? null,
    lengthM: a.length_m ?? null,
    widthM: a.width_m ?? null,
    halfLengthM: a.half_length_m,
    firstBearingDeg: a.first_bearing_deg,
    speedMs: a.speed_ms,
    durationS: a.duration_s,
  });
}

describe('patterns match Python', () => {
  golden.patterns.forEach((c, n) => {
    it(`${c.kind} case ${n}: waypoints, offsets and headings`, () => {
      const p = build(c.kind, c.args);
      close(p.eastM, c.waypoints.east_m, TOL_M);
      close(p.northM, c.waypoints.north_m, TOL_M);
      close(p.tS, c.waypoints.t_s, TOL_S);
      close(p.headingDeg, c.waypoints.heading_deg, 1e-9);
      c.samples.t_s.forEach((t, k) => {
        const [e, nn] = offsetAt(p, t);
        expect(Math.abs(e - c.samples.east_m[k])).toBeLessThanOrEqual(TOL_M);
        expect(Math.abs(nn - c.samples.north_m[k])).toBeLessThanOrEqual(TOL_M);
        expect(headingAt(p, t)).toBeCloseTo(c.samples.heading_deg[k], 9);
      });
    });
  });

  golden.on_ground.forEach((c) => {
    it(`on the ground at ${c.marker.lat[0]} N, carried by a drifting marker`, () => {
      const p = build(c.pattern.kind, c.pattern.args);
      const marker = { tS: c.marker.t_s, lat: c.marker.lat, lon: c.marker.lon };
      c.t_s.forEach((t, k) => {
        const [lat, lon] = onGround(p, marker, t);
        expect(Math.abs(lat - c.lat[k])).toBeLessThanOrEqual(TOL_DEG);
        expect(Math.abs(lon - c.lon[k])).toBeLessThanOrEqual(TOL_DEG);
      });
    });
  });
});

describe('point drift matches Python', () => {
  golden.drift.forEach((c) => {
    it(`${c.duration_s} s at leeway ${c.leeway}`, () => {
      const track = driftTrack({
        lat: c.start.lat,
        lon: c.start.lon,
        startMs: Date.parse('2019-06-01T06:00Z'),
        durationS: c.duration_s,
        leeway: c.leeway,
        sample: constantSampler(c.current_ms, c.wind_ms),
      });
      close(track.tS, c.track.t_s, TOL_S);
      close(track.lat, c.track.lat, TOL_DEG);
      close(track.lon, c.track.lon, TOL_DEG);
    });
  });
});

describe('transit matches Python', () => {
  golden.transit.forEach((c) => {
    it(`base to ${c.datum.lat} N ${c.datum.lon}`, () => {
      const d = distanceM(c.base, c.datum);
      expect(Math.abs(d - c.distance_m)).toBeLessThanOrEqual(1e-6);
      expect(Math.abs(transitTimeS(d) - c.time_s)).toBeLessThanOrEqual(1e-6);
    });
  });
});

describe('pattern guards and shape', () => {
  it('refuses non-positive inputs', () => {
    expect(() => expandingSquare({ spacingM: 0 })).toThrow(RangeError);
    expect(() => sectorSearch({ speedMs: -1 })).toThrow(RangeError);
  });

  it('is null outside its window', () => {
    const p = expandingSquare({ durationS: 60 });
    expect(offsetAt(p, 61)).toBeNull();
    expect(headingAt(p, -1)).toBeNull();
  });

  it('has nine sector legs and a compass convention clockwise from north', () => {
    expect(SECTOR_LEGS).toHaveLength(9);
    const [e, n] = eastNorth(90, 1);
    expect(e).toBeCloseTo(1, 12);
    expect(n).toBeCloseTo(0, 12);
  });
});
