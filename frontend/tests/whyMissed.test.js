/**
 * Why the buoy left the prediction (issue #80), against buoys whose motion is made up so
 * the right answer is known.
 */

import { describe, expect, it } from 'vitest';

import { constantSampler, M_PER_DEG } from '../src/pointDrift.js';
import {
  CLOSE_MS, buoyVelocityAt, cellSize, explainMiss, velocityBetween,
} from '../src/whyMissed.js';

const T0 = Date.UTC(2023, 2, 10, 0, 0);
const HOUR = 3600 * 1000;
const LAT0 = 28.6;
const LON0 = -64.4;

/** A buoy moving at a steady [u, v] m/s from (LAT0, LON0) at T0, for `hours`. */
function steadyBuoy([u, v], hours = 4) {
  const cos = Math.cos((LAT0 * Math.PI) / 180);
  return (ms) => {
    const t = (ms - T0) / 1000;
    if (t < 0 || t > hours * 3600) return null;
    return [LAT0 + (v * t) / M_PER_DEG, LON0 + (u * t) / (M_PER_DEG * cos)];
  };
}

const explain = (buoyVel, current, wind, leeway, undrogued) => explainMiss({
  targetAt: steadyBuoy(buoyVel),
  sample: constantSampler(current, wind),
  fromMs: T0,
  toMs: T0 + 2 * HOUR,
  leeway,
  undrogued,
});

describe('velocityBetween and buoyVelocityAt', () => {
  it('recovers a steady velocity from two positions', () => {
    const at = steadyBuoy([0.3, -0.2]);
    const v = velocityBetween(at(T0), at(T0 + HOUR), 3600);
    expect(v[0]).toBeCloseTo(0.3, 3);
    expect(v[1]).toBeCloseTo(-0.2, 6);
  });

  it('uses both sides where it can and one side at the ends of the record', () => {
    const at = steadyBuoy([0.1, 0.4], 2);
    expect(buoyVelocityAt(at, T0 + HOUR)[1]).toBeCloseTo(0.4, 6);    // centred
    expect(buoyVelocityAt(at, T0)[1]).toBeCloseTo(0.4, 6);           // start: forward only
    expect(buoyVelocityAt(at, T0 + 2 * HOUR)[1]).toBeCloseTo(0.4, 6);// end: backward only
    expect(buoyVelocityAt(() => null, T0)).toBeNull();
  });
});

describe('explainMiss', () => {
  it('finds no gap when the buoy moves exactly as the model says', () => {
    const current = [0.2, 0.1];
    const wind = [-9.6, 0];
    const r = explain([0.2 + 0.02 * -9.6, 0.1], current, wind, 0.02, true);
    expect(r.gapSpeed).toBeLessThan(1e-3);
    expect(r.verdict).toBe('close');
    expect(r.headline).toMatch(/about right/);
  });

  it('blames the wind when an undrogued buoy runs further downwind, and says what would fit', () => {
    // Wind 10 m/s towards the east; the buoy takes 3.5 % of it, the model 2 %.
    const r = explain([0.35, 0.2], [0, 0.2], [10, 0], 0.02, true);
    expect(r.verdict).toBe('downwind');
    expect(r.downwind).toBeCloseTo(0.15, 3);
    expect(r.fitLeeway).toBeCloseTo(0.035, 3);
    expect(r.detail).toMatch(/3\.5 % of the wind would have fitted, where the model used 2\.0 %/);
    expect(r.detail).toMatch(/Without its drogue/);
  });

  it('blames the current when the gap runs across the wind', () => {
    // Model: current north 0.2 + 2 % of a 10 m/s easterly-going wind. The buoy also goes
    // 0.2 m/s further north: across the wind.
    const r = explain([0.2, 0.4], [0, 0.2], [10, 0], 0.02, true);
    expect(r.verdict).toBe('current');
    expect(Math.abs(r.crosswind)).toBeCloseTo(0.2, 3);
    expect(r.detail).toMatch(/does not run along the wind/);
    expect(r.detail).toMatch(/cells here are about 7\.8 × 4\.4 km/);
  });

  it('says a drogued buoy was wrongly given the wind', () => {
    // The model added 2 % of a 10 m/s wind; the drogued buoy rode the current alone.
    const r = explain([0, 0.2], [0, 0.2], [10, 0], 0.02, false);
    expect(r.verdict).toBe('upwind-drogued');
    expect(r.downwind).toBeCloseTo(-0.2, 3);
    expect(r.detail).toMatch(/"current only" fits this buoy better/);
  });

  it('blames the current when the wind is too light to matter, whatever the direction', () => {
    const r = explain([0.3, 0], [0, 0], [1, 0], 0.02, true);
    expect(r.verdict).toBe('current');
    expect(r.detail).toMatch(/The wind was light \(1\.0 m\/s\)/);
  });

  it('gives a short form for the panel: three labelled lines and a one-line cause', () => {
    const r = explain([0.2, 0.4], [0, 0.2], [10, 0], 0.02, true);
    expect(r.table.map(([k]) => k)).toEqual(['Buoy', 'Model', 'Missing']);
    expect(r.table[2][1]).toMatch(/0\.20 m\/s towards N/);
    expect(r.cause).toMatch(/^Not along the wind, so likelier the model's current: 7\.8 × 4\.4 km cells/);
    for (const args of [[[0.2, 0.4], [0, 0.2], [10, 0], 0.02, true], [[0.35, 0.2], [0, 0.2], [10, 0], 0.02, true],
      [[0, 0.2], [0, 0.2], [10, 0], 0.02, false]]) {
      expect(explain(...args).cause.length).toBeLessThan(110);
    }
  });

  it('always says what it cannot see', () => {
    const r = explain([0.35, 0.2], [0, 0.2], [10, 0], 0.02, true);
    expect(r.limits).toMatch(/15 m down/);
  });

  it('gives nothing without the buoy or the forcing', () => {
    expect(explainMiss({
      targetAt: () => null, sample: constantSampler([0, 0]), fromMs: T0, toMs: T0 + HOUR, leeway: 0,
    })).toBeNull();
    expect(explainMiss({
      targetAt: steadyBuoy([0, 0]), sample: () => ({ current: null }), fromMs: T0, toMs: T0 + HOUR, leeway: 0,
    })).toBeNull();
  });

  it('keeps the close threshold at a few centimetres a second', () => {
    expect(CLOSE_MS).toBeLessThan(0.05);
  });
});

describe('cellSize', () => {
  it('sizes the current model cell by latitude', () => {
    expect(cellSize(28.6)).toBe('7.8 × 4.4 km');
    expect(cellSize(0)).toBe('8.9 × 4.4 km');
  });
});
