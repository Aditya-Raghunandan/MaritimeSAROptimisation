/**
 * Tests for the raster colour ramp.
 *
 * The gate that applies to a sequential ramp is monotonic lightness — that is
 * what separates viridis from the rainbow ramps it resembles, and the whole
 * reason it is defensible here.
 */

import { describe, expect, it } from 'vitest';

import { MAGMA, VIRIDIS, normaliseSpeed, ramp, rampCss, viridis, viridisCss } from '../src/colormap.js';

const lin = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
function lightness([r, g, b]) {
  const R = lin(r / 255);
  const G = lin(g / 255);
  const B = lin(b / 255);
  const l = Math.cbrt(0.4122214708 * R + 0.5363325363 * G + 0.0514459929 * B);
  const m = Math.cbrt(0.2119034982 * R + 0.6806995451 * G + 0.1073969566 * B);
  const s = Math.cbrt(0.0883024619 * R + 0.2817188376 * G + 0.6299787005 * B);
  return 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s;
}

describe('viridis', () => {
  it('rises monotonically in lightness across the whole ramp', () => {
    // The gate. A naive HSV rainbow fails this and invents banding where the
    // data is smooth; viridis exists precisely to pass it.
    let prev = -1;
    for (let t = 0; t <= 1.0001; t += 0.01) {
      const L = lightness(viridis(t));
      expect(L).toBeGreaterThan(prev - 1e-9);
      prev = L;
    }
  });

  it('survives greyscale, which the printed report will need', () => {
    expect(lightness(viridis(1)) - lightness(viridis(0))).toBeGreaterThan(0.5);
  });

  it('clamps outside [0, 1] rather than wrapping', () => {
    // A wrapped scale makes the strongest wind look like the calmest.
    expect(viridis(-5)).toEqual(viridis(0));
    expect(viridis(99)).toEqual(viridis(1));
  });

  it('returns null for a NaN so the caller can draw nothing', () => {
    // Land in the current field is NaN. Painting it as the calmest water would
    // put a false dead-calm patch over every coastline.
    expect(viridis(NaN)).toBeNull();
    expect(viridisCss(NaN)).toBe('transparent');
  });

  it('gives back well-formed channels', () => {
    for (let t = 0; t <= 1; t += 0.05) {
      for (const c of viridis(t)) {
        expect(Number.isInteger(c)).toBe(true);
        expect(c).toBeGreaterThanOrEqual(0);
        expect(c).toBeLessThanOrEqual(255);
      }
    }
  });
});

describe('normaliseSpeed', () => {
  it('is square root, not linear', () => {
    // Ocean wind is skewed: most of the box sits low most of the time, so a
    // linear ramp renders the ordinary sea as one flat dark field.
    expect(normaliseSpeed(5, 20)).toBeCloseTo(0.5);      // sqrt(0.25)
    expect(normaliseSpeed(5, 20)).toBeGreaterThan(5 / 20);
  });

  it('maps the ends to the ends', () => {
    expect(normaliseSpeed(0, 20)).toBe(0);
    expect(normaliseSpeed(20, 20)).toBe(1);
  });

  it('clamps above the maximum', () => {
    expect(normaliseSpeed(60, 20)).toBe(1);
  });

  it('is NaN for absent data or a zero scale, not 0', () => {
    expect(normaliseSpeed(NaN, 20)).toBeNaN();
    expect(normaliseSpeed(5, 0)).toBeNaN();
  });
});

/*
  Two ramps, because there are two fields. Wind and current were painted
  identically and could not be told apart at a glance, which on a projector
  across a room is the only glance anyone gets.
*/
describe('magma, the current ramp', () => {
  // OKLab lightness is the gate a sequential ramp has to pass; a cheap proxy
  // that catches the same failure is that perceived luminance rises at every
  // anchor. A non-monotonic ramp invents banding where the data is smooth.
  const luma = ([r, g, b]) => 0.2126 * r + 0.7152 * g + 0.0722 * b;

  it('rises monotonically in luminance, like viridis', () => {
    for (const anchors of [VIRIDIS, MAGMA]) {
      for (let i = 1; i < anchors.length; i += 1) {
        expect(luma(anchors[i])).toBeGreaterThan(luma(anchors[i - 1]));
      }
    }
  });

  it('is genuinely different from viridis at every stop', () => {
    // If the two ramps were close, splitting the products by colour would buy
    // nothing. Checked rather than assumed.
    for (let k = 1; k <= 9; k += 1) {
      const a = ramp(VIRIDIS, k / 10);
      const b = ramp(MAGMA, k / 10);
      const dist = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
      expect(dist).toBeGreaterThan(40);
    }
  });

  it('starts near black so the slow ocean recedes and the jet glows', () => {
    expect(luma(MAGMA[0])).toBeLessThan(10);
  });

  it('clamps rather than wraps, so the fastest is never drawn as the slowest', () => {
    expect(ramp(MAGMA, 5)).toEqual(MAGMA[MAGMA.length - 1]);
    expect(ramp(MAGMA, -5)).toEqual(MAGMA[0]);
  });

  it('returns null for NaN, which is absent rather than zero', () => {
    expect(ramp(MAGMA, NaN)).toBeNull();
    expect(rampCss(MAGMA, NaN)).toBe('transparent');
  });

  it('viridis is still exactly what it was', () => {
    expect(viridis(0.5)).toEqual(ramp(VIRIDIS, 0.5));
    expect(viridisCss(0.5)).toBe(rampCss(VIRIDIS, 0.5));
  });
});
