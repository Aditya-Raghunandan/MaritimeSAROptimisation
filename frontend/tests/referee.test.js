import { describe, expect, it } from 'vitest';
import golden from '../src/fixtures/referee_golden.json';
import { M_PER_DEG_LAT } from '../src/geo.js';
import { SEARCH_SPEED_MS } from '../src/platform.js';
import { TURN_RADIUS_M, TURN_RATE_DEG_S } from '../src/platform.js';
import {
  Episode, HALF_WIDTH_M, STEPS, closestApproachM, cloudFromOffsets, interp, relativeM,
} from '../src/referee.js';

const cloud = () => ({
  frames: golden.window.frames,
  particles: golden.window.particles,
  lat: Float64Array.from(golden.window.lat),
  lon: Float64Array.from(golden.window.lon),
  weight: golden.window.weight,
});
const track = (t) => ({ tS: t.t_s, lat: t.lat, lon: t.lon });
const steps = (flight) => flight.steps.map((s) => ({
  tS: s.t_s, eastM: s.east_m, northM: s.north_m, headingDeg: s.heading_deg ?? undefined,
}));
const HELI = {
  turning: { turnRateDegS: golden.constants.turn_rate_deg_s, headingDeg: golden.constants.arrival_heading_deg },
  instant: { turnRateDegS: Infinity, headingDeg: golden.constants.arrival_heading_deg },
};

function replay(name, set = 'flights') {
  const heli = set === 'flights' ? HELI.turning : HELI.instant;
  const ep = new Episode(cloud(), track(golden.marker), { target: track(golden.target), ...heli });
  for (const s of steps(golden[set][name])) ep.fly(s);
  return ep;
}

describe('the referee holds the same numbers as Python', () => {
  it('has the same constants', () => {
    expect(SEARCH_SPEED_MS).toBeCloseTo(golden.constants.speed_ms, 12);
    expect(HALF_WIDTH_M * 2).toBeCloseTo(golden.constants.sweep_width_m, 12);
    expect(M_PER_DEG_LAT).toBeCloseTo(golden.constants.m_per_deg_lat, 6);
    expect(STEPS).toBe(golden.constants.steps);
    expect(TURN_RATE_DEG_S).toBeCloseTo(golden.constants.turn_rate_deg_s, 12);
  });

  for (const name of Object.keys(golden.instant)) {
    it(`scores ${name} as the old helicopter that turned at once flew it`, () => {
      const m = replay(name, 'instant').metrics();
      m.drainRate.forEach((d, k) => expect(d).toBeCloseTo(golden.instant[name].removed_per_step[k], 12));
      expect(m.turnRateDegS).toBeNull();
    });
  }

  for (const name of Object.keys(golden.flights)) {
    it(`scores ${name} minute by minute`, () => {
      const m = replay(name).metrics();
      const want = golden.flights[name];
      expect(m.drainRate).toHaveLength(45);
      m.drainRate.forEach((d, k) => expect(d).toBeCloseTo(want.removed_per_step[k], 12));
      expect(m.pos).toBeCloseTo(want.pos, 12);
      if (want.expected_ttd_s === null) expect(m.expectedTtdS).toBeNull();
      else expect(m.expectedTtdS).toBeCloseTo(want.expected_ttd_s, 6);
    });

    it(`judges the real buoy for ${name} the same way`, () => {
      const t = replay(name).metrics().target;
      const want = golden.flights[name].target;
      expect(t.found).toBe(want.found);
      expect(t.gaps).toBe(want.gaps);
      expect(t.closestM).toBeCloseTo(want.closest_m, 6);
      expect(t.closestS).toBeCloseTo(want.closest_s, 6);
      if (want.found_s === null) expect(t.foundS).toBeNull();
      else expect(t.foundS).toBeCloseTo(want.found_s, 6);
    });
  }

  it('finds the buoy at least once in the fixture, so contact is exercised', () => {
    expect(Object.values(golden.flights).some((f) => f.target.found)).toBe(true);
  });
});

describe('the bundle decoder', () => {
  it('rebuilds positions from float32 metres exactly as Python does', () => {
    const e = golden.encoded;
    const n = e.east_f32.length;
    const f32 = new Float32Array(2 * n);
    for (let j = 0; j < n; j += 1) {
      f32[2 * j] = e.east_f32[j];
      f32[2 * j + 1] = e.north_f32[j];
    }
    const marker = { tS: [0], lat: [e.marker_lat], lon: [e.marker_lon] };
    const { lat, lon } = cloudFromOffsets(f32, 1, n, marker);
    for (let j = 0; j < n; j += 1) {
      expect(lat[j]).toBeCloseTo(e.lat[j], 11);
      expect(lon[j]).toBeCloseTo(e.lon[j], 11);
    }
  });
});

describe('the pieces', () => {
  it('wraps longitude as Python does', () => {
    const [e1] = relativeM(26.5, 0.1, 26.5, 359.9);
    const [e2] = relativeM(26.5, 359.9, 26.5, 0.1);
    expect(e1).toBeCloseTo(-e2, 9);
    expect(e1).toBeGreaterThan(0);
  });

  it('measures closest approach to the segment, not the line', () => {
    expect(closestApproachM(100, 0, 300, 0)).toBeCloseTo(100, 12);
    expect(closestApproachM(-100, 50, 100, 50)).toBeCloseTo(50, 12);
  });

  it('interpolates like numpy and clamps at the ends', () => {
    expect(interp(5, [0, 10], [0, 1])).toBe(0.5);
    expect(interp(-1, [0, 10], [0, 1])).toBe(0);
    expect(interp(11, [0, 10], [0, 1])).toBe(1);
  });

  it('refuses a path faster than the helicopter', () => {
    const ep = new Episode(cloud(), track(golden.marker));
    expect(() => ep.fly({ tS: [60], eastM: [SEARCH_SPEED_MS * 61], northM: [0] })).toThrow(/faster/);
  });

  it('flies a heading as one straight sub-leg when it turns at once', () => {
    const ep = new Episode(cloud(), track(golden.marker), { turnRateDegS: Infinity });
    ep.step(90);
    expect(ep.offset[0]).toBeCloseTo(SEARCH_SPEED_MS * 60, 9);
    expect(ep.k).toBe(1);
  });

  it('flies a right angle as a quarter circle, then straight (D032)', () => {
    const ep = new Episode(cloud(), track(golden.marker));
    ep.step(90);
    const tau = 90 / TURN_RATE_DEG_S;
    expect(ep.offset[0]).toBeCloseTo(TURN_RADIUS_M + SEARCH_SPEED_MS * (60 - tau), 6);
    expect(ep.offset[1]).toBeCloseTo(TURN_RADIUS_M, 6);
    expect(ep.headingDeg).toBeCloseTo(90, 9);
  });

  it('takes a turn command, right positive', () => {
    const ep = new Episode(cloud(), track(golden.marker), { headingDeg: 90 });
    ep.turn(-45);
    expect(ep.headingDeg).toBeCloseTo(45, 9);
  });

  it('refuses a square corner, as Python does', () => {
    const ep = new Episode(cloud(), track(golden.marker), { headingDeg: 90 });
    const half = SEARCH_SPEED_MS * 30;
    expect(() => ep.fly({ tS: [30, 60], eastM: [half, half], northM: [0, half], headingDeg: [90, 90] }))
      .toThrow(/corner/);
    expect(() => ep.fly({ tS: [60], eastM: [SEARCH_SPEED_MS * 60], northM: [0] })).toThrow(/heading at every waypoint/);
  });
});
