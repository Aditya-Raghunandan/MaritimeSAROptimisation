import { describe, expect, it } from 'vitest';
import {
  GridField, SIGMA_TRUE, bilinearWeights, cloudOffsets, driftOf, heliStep, judgeCloud, radius90,
  segmentDistance, toLatLon, toMetres, trialBuoys,
} from '../src/tutorialMaths.js';
import { SEARCH_SPEED_MS, TURN_RATE_DEG_S } from '../src/platform.js';

describe('mission 1: drift and the helicopter', () => {
  it('drift is the current plus 2 % of the wind', () => {
    const [e, n] = driftOf([0, 1.2], [-8, 0]);
    expect(e).toBeCloseTo(-0.16, 12);
    expect(n).toBeCloseTo(1.2, 12);
  });

  it('turns no faster than 7 degrees a second, then flies on at 90 kt', () => {
    const s = heliStep({ east: 0, north: 0, heading: 0 }, 180, 1);
    expect(s.heading).toBeCloseTo(TURN_RATE_DEG_S, 9);
    expect(Math.hypot(s.east, s.north)).toBeCloseTo(SEARCH_SPEED_MS, 9);
    const u = heliStep({ east: 0, north: 0, heading: 350 }, 10, 1);
    expect(u.heading).toBeCloseTo(350 + TURN_RATE_DEG_S, 9);     // the short way, through north
    expect(heliStep({ east: 0, north: 0, heading: 90 }, null, 1).heading).toBe(90);
  });

  it('measures the closest pass to a buoy along a leg', () => {
    expect(segmentDistance([50, 50], [0, 0], [100, 0])).toBeCloseTo(50, 12);
    expect(segmentDistance([-30, 40], [0, 0], [100, 0])).toBeCloseTo(50, 12);
  });

  it('metres and degrees go both ways', () => {
    const [lat, lon] = toLatLon(25.75, -79.7, 3000, -4000);
    const [e, n] = toMetres(25.75, -79.7, lat, lon);
    expect(e).toBeCloseTo(3000, 6);
    expect(n).toBeCloseTo(-4000, 6);
  });
});

describe('mission 2: between the dots', () => {
  const still = (u, v) => [[[u, v], [u, v], [u, v]], [[u, v], [u, v], [u, v]], [[u, v], [u, v], [u, v]]];

  it('bilinear weights sum to one and favour the nearest corner', () => {
    const w = bilinearWeights(0.25, 0.1);
    expect(w.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(Math.max(...w)).toBe(w[0]);
  });

  it('blends linearly in time between two frames 3 hours apart', () => {
    const f = new GridField({ frames: [still(0.2, 0), still(0.4, 0)] });
    expect(f.at(1000, 500, 0)[0]).toBeCloseTo(0.2, 12);
    expect(f.at(1000, 500, 1.5)[0]).toBeCloseTo(0.3, 12);
    expect(f.at(1000, 500, 3)[0]).toBeCloseTo(0.4, 12);
  });

  it('is exact at a grid point and in between in space', () => {
    const frame = [[[0, 0], [1, 0], [2, 0]], [[0, 0], [1, 0], [2, 0]], [[0, 0], [1, 0], [2, 0]]];
    const f = new GridField({ frames: [frame, frame] });
    const [x] = f.points()[1];
    expect(f.at(x, 0, 0)[0]).toBeCloseTo(1, 12);
    expect(f.at(x + f.dx / 2, 0, 0)[0]).toBeCloseTo(1.5, 12);
  });

  it('a path in a steady field goes where the current says', () => {
    const f = new GridField({ frames: [still(0.5, 0), still(0.5, 0)] });
    const p = f.path(0, 0, 1);
    expect(p[p.length - 1][0]).toBeCloseTo(1800, 6);
  });
});

describe('mission 3: the honest cloud', () => {
  it('a cloud after 2 h spreads by sigma times the time', () => {
    const [e] = cloudOffsets(4000, SIGMA_TRUE, 7200);
    const sd = Math.sqrt(e.reduce((s, x) => s + x * x, 0) / e.length);
    expect(sd / (SIGMA_TRUE * 7200)).toBeGreaterThan(0.95);
    expect(sd / (SIGMA_TRUE * 7200)).toBeLessThan(1.05);
  });

  it('radius90 holds 90 % of the cloud', () => {
    const c = cloudOffsets(2000, 0.2, 3600);
    const r = radius90(c);
    const inside = Array.from(c[0], (x, k) => Math.hypot(x, c[1][k]) <= r).filter(Boolean).length;
    expect(inside / 2000).toBeCloseTo(0.9, 2);
  });

  it('too small misses, the true size catches four of five, too big is too big to search', () => {
    expect(trialBuoys(7200)).toHaveLength(5);
    expect(judgeCloud(0.08, 7200).verdict).toBe('small');
    const honest = judgeCloud(SIGMA_TRUE, 7200);
    expect(honest.verdict).toBe('right');
    expect(honest.hits).toBe(4);
    expect(honest.areaKm2).toBeGreaterThan(25);
    expect(honest.areaKm2).toBeLessThan(55);
    const vague = judgeCloud(0.6, 7200);
    expect(vague.hits).toBe(5);
    expect(vague.verdict).toBe('big');
  });
});
