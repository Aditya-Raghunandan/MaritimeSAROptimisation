/**
 * The browser's turn model and autopilot, held to `sar.search.kinematics` (D032).
 *
 * `fixtures/search_golden.json` holds turns flown by Python's `steer` and ten minutes of
 * each pattern flown by its L1 autopilot, every tenth tick; kinematics.js must fly the same.
 */

import { describe, expect, it } from 'vitest';

import golden from '../src/fixtures/search_golden.json';
import {
  ARC_PIECE_DEG, checkTurns, flownPattern, follow, normalise, shortestTurn, steer,
} from '../src/kinematics.js';
import { PATTERNS } from '../src/patterns.js';
import {
  MAX_BANK_DEG, SEARCH_SPEED_MS, STANDARD_RATE_DEG_S, TURN_RADIUS_M, TURN_RATE_DEG_S,
} from '../src/platform.js';

const P = golden.platform;
const camel = (args) => Object.fromEntries(Object.entries(args).map(([k, v]) => [
  k.replace(/_([a-z])/g, (_, c) => c.toUpperCase()).replace(/M$/, 'M').replace(/Deg$/, 'Deg'), v,
]));

describe('the turn constants', () => {
  it('agree with Python', () => {
    expect(MAX_BANK_DEG).toBe(P.max_bank_deg);
    expect(STANDARD_RATE_DEG_S).toBe(P.standard_rate_deg_s);
    expect(TURN_RATE_DEG_S).toBeCloseTo(P.turn_rate_deg_s, 12);
    expect(TURN_RADIUS_M).toBeCloseTo(P.turn_radius_m, 9);
  });

  it('are a 30 degree bank at 90 kt: 7.0 deg/s on 379 m', () => {
    expect(TURN_RATE_DEG_S).toBeCloseTo(7.0065, 4);
    expect(TURN_RADIUS_M).toBeCloseTo(378.62, 2);
  });
});

describe('a turn, flown as Python flies it', () => {
  for (const c of golden.steer) {
    const a = c.args;
    it(`turns ${a.turn_deg} from ${a.heading_deg} for ${a.duration_s} s at ${a.turn_rate_deg_s ?? 'once'}`, () => {
      const p = steer(a.east_m, a.north_m, a.heading_deg, a.turn_deg, a.duration_s, a.speed_ms,
        a.turn_rate_deg_s ?? Infinity);
      expect(p.tS).toHaveLength(c.t_s.length);
      p.tS.forEach((t, i) => {
        expect(t).toBeCloseTo(c.t_s[i], 9);
        expect(p.eastM[i]).toBeCloseTo(c.east_m[i], 9);
        expect(p.northM[i]).toBeCloseTo(c.north_m[i], 9);
        expect(p.headingDeg[i]).toBeCloseTo(c.heading_deg[i], 9);
      });
    });
  }

  it('turns the short way round, and right at exactly 180', () => {
    expect(shortestTurn(350, 10)).toBeCloseTo(20, 12);
    expect(shortestTurn(10, 350)).toBeCloseTo(-20, 12);
    expect(shortestTurn(0, 180)).toBe(180);
    expect(normalise(-1e-17)).toBe(0);
  });

  it('cuts an arc into pieces of at most 10 degrees, each of which the referee accepts', () => {
    const p = steer(0, 0, 0, 175, 60, SEARCH_SPEED_MS, TURN_RATE_DEG_S);
    let h = 0;
    for (const hd of p.headingDeg) {
      expect(Math.abs(shortestTurn(h, hd))).toBeLessThanOrEqual(ARC_PIECE_DEG + 1e-9);
      h = hd;
    }
    expect(() => checkTurns(p.tS, p.eastM, p.northM, p.headingDeg, [0, 0, 0], SEARCH_SPEED_MS, TURN_RATE_DEG_S))
      .not.toThrow();
  });
});

describe('the autopilot, flown as Python flies it', () => {
  for (const c of golden.flown) {
    it(`flies ${c.kind} tick for tick`, () => {
      const drawn = PATTERNS[c.kind].build(camel(c.args));
      const f = follow(drawn.eastM, drawn.northM, c.heading_deg, c.duration_s, SEARCH_SPEED_MS, c.turn_rate_deg_s);
      c.t_s.forEach((t, j) => {
        const i = j * c.every;
        expect(f.tS[i]).toBeCloseTo(t, 9);
        expect(f.eastM[i]).toBeCloseTo(c.east_m[j], 6);
        expect(f.northM[i]).toBeCloseTo(c.north_m[j], 6);
        expect(f.headingDeg[i]).toBeCloseTo(c.heading[j], 6);
      });
      const m = f.path(60, 120);
      expect(m.tS).toEqual(c.minute_2.t_s);
      m.eastM.forEach((e, i) => expect(e).toBeCloseTo(c.minute_2.east_m[i], 6));
      m.headingDeg.forEach((h, i) => expect(h).toBeCloseTo(c.minute_2.heading_deg[i], 6));
    });
  }

  it('never asks for more than the turn rate, and every minute passes the referee', () => {
    const drawn = PATTERNS.parallel_track.build({ firstBearingDeg: 30, durationS: 2700 });
    const f = follow(drawn.eastM, drawn.northM, 30, 600, SEARCH_SPEED_MS, TURN_RATE_DEG_S);
    expect(Math.max(...f.rateDegS.map(Math.abs))).toBeLessThanOrEqual(TURN_RATE_DEG_S + 1e-12);
    for (let k = 0; k < 10; k += 1) {
      const m = f.path(60 * k, 60 * (k + 1));
      const i = 120 * k;
      expect(() => checkTurns(m.tS.map((t) => t - 60 * k), m.eastM, m.northM, m.headingDeg,
        [f.eastM[i], f.northM[i], f.headingDeg[i]], SEARCH_SPEED_MS, TURN_RATE_DEG_S)).not.toThrow();
    }
  });

  it('gives a pattern the same shape of object, keeping the drawing and its area', () => {
    const drawn = PATTERNS.parallel_track.build({ firstBearingDeg: 30, durationS: 2700 });
    const p = flownPattern(drawn, { headingDeg: 30, durationS: 2700, turnRateDegS: TURN_RATE_DEG_S });
    expect(p.drawn).toBe(drawn);
    expect(p.area).toEqual(drawn.area);
    expect(p.headingDeg).toHaveLength(p.tS.length - 1);
    expect(p.tS[p.tS.length - 1]).toBeCloseTo(2700, 9);
    expect(flownPattern(drawn, { headingDeg: 30, durationS: 2700, turnRateDegS: Infinity })).toBe(drawn);
  });
});
