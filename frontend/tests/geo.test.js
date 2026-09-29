/**
 * Bearings and distance formatting.
 *
 * `bearing` is tested against known great-circle answers rather than recorded
 * output, and specifically at due north -- where a plain `% 360` returns
 * exactly 360, the same bug the Python side hit in `_bearing`. A bearing of
 * 360 breaks any consumer binning into sectors.
 */

import { describe, expect, it } from 'vitest';
import golden from '../src/fixtures/search_golden.json';
import { M_PER_DEG_LAT, R_EARTH_M, bearing, formatDistance } from '../src/geo.js';
import { M_PER_DEG_LAT as PATTERNS_M_PER_DEG } from '../src/patterns.js';
import { EARTH_RADIUS_M, M_PER_DEG } from '../src/pointDrift.js';

const at = (lat, lng) => ({ lat, lng });

describe('bearing', () => {
  it('is 0 due north', () => {
    expect(bearing(at(30, -70), at(31, -70))).toBeCloseTo(0, 6);
  });

  it('never returns 360 for due north', () => {
    // The whole point. Round-tripping through a modulo can land on exactly 360,
    // which is outside the half-open range every consumer assumes.
    const b = bearing(at(30, -70), at(31, -70));
    expect(b).toBeGreaterThanOrEqual(0);
    expect(b).toBeLessThan(360);
  });

  it('is 180 due south', () => {
    expect(bearing(at(31, -70), at(30, -70))).toBeCloseTo(180, 6);
  });

  it('is 90 due east on the equator', () => {
    expect(bearing(at(0, -70), at(0, -69))).toBeCloseTo(90, 6);
  });

  it('is 270 due west on the equator', () => {
    expect(bearing(at(0, -69), at(0, -70))).toBeCloseTo(270, 6);
  });

  it('bends away from due east at latitude, as a great circle does', () => {
    // A rhumb line due east stays at 90; a great circle does not. At 35 N a
    // 10 degree eastward hop starts north of east. This is the difference that
    // makes geodesic distance the right answer over the whole 19 degree box.
    const b = bearing(at(35, -75), at(35, -65));
    expect(b).toBeLessThan(90);
    expect(b).toBeGreaterThan(85);
  });

  it('wraps into [0, 360) across the antimeridian', () => {
    const b = bearing(at(0, 179), at(0, -179));
    expect(b).toBeCloseTo(90, 3);
  });
});

describe('formatDistance', () => {
  it('uses metres below a kilometre, which sweep width needs', () => {
    // Sweep width is ~185 m. Rendering that as "0.19 km" makes the number that
    // sets the probability map's cell size unreadable.
    expect(formatDistance(185)).toBe('185 m');
  });

  it('uses two decimals of a kilometre in the mid range', () => {
    expect(formatDistance(47_300)).toBe('47.30 km');
  });

  it('drops the decimals once they stop meaning anything', () => {
    expect(formatDistance(155_000)).toBe('155 km');
  });
});


describe('one Earth radius', () => {
  // Until 29 Sep 2026 the patterns, the close-up sea and the particles used 111,320 m per
  // degree while the drift and the ruler used this sphere: 0.11 % apart.
  it('is the same number as Python', () => {
    expect(R_EARTH_M).toBe(golden.geo.earth_radius_m);
    expect(M_PER_DEG_LAT).toBeCloseTo(golden.geo.m_per_deg_lat, 6);
  });

  it('is the one every module converts with', () => {
    expect(PATTERNS_M_PER_DEG).toBe(M_PER_DEG_LAT);
    expect(M_PER_DEG).toBe(M_PER_DEG_LAT);
    expect(EARTH_RADIUS_M).toBe(R_EARTH_M);
  });
});
