/**
 * The rules of the close-up view (issue #79): when it is on, where the sun is, what the
 * wind does to the sea, and how floating weed moves.
 */

import { describe, expect, it } from 'vitest';

import {
  CLOSE_ZOOM, closeUpWeight, floaterStep, isCloseUp, metresPerPixel, nightness, peakWavelengthM,
  WEED_TILE_M, seededRandom, sunPosition, toKnots, weedRows, whitecapFraction,
} from '../src/closeUp.js';
import { stepPosition } from '../src/pointDrift.js';

describe('when the close-up is on', () => {
  it('is off below 13, fading in to 13.5, and fully on from there', () => {
    expect(closeUpWeight(12)).toBe(0);
    expect(closeUpWeight(13)).toBe(0);
    expect(closeUpWeight(13.25)).toBeGreaterThan(0);
    expect(closeUpWeight(13.25)).toBeLessThan(1);
    expect(closeUpWeight(CLOSE_ZOOM)).toBe(1);
    expect(closeUpWeight(16)).toBe(1);
  });

  it('switches the view at 13.5 exactly', () => {
    expect(isCloseUp(13.375)).toBe(false);
    expect(isCloseUp(13.5)).toBe(true);
  });

  it('agrees with the search layer on metres per pixel', () => {
    // Zoom 0 at the equator is the whole circumference over 256 px.
    expect(metresPerPixel(0, 0)).toBeCloseTo(156543.03, 1);
    expect(metresPerPixel(60, 1)).toBeCloseTo(156543.03 / 4, 1);
  });

  it('converts m/s to knots', () => {
    expect(toKnots(1852 / 3600)).toBeCloseTo(1, 12);
    expect(toKnots(9.6)).toBeCloseTo(18.66, 2);
  });
});

describe('sunPosition', () => {
  it('puts the midsummer noon sun at 90 - lat + 23.44 at Greenwich', () => {
    const s = sunPosition(Date.UTC(2024, 5, 21, 12, 0), 51.4769, 0);
    expect(s.elevationDeg).toBeCloseTo(90 - 51.4769 + 23.44, 0);
    expect(Math.abs(s.azimuthDeg - 180)).toBeLessThan(3);        // due south
  });

  it('has the equinox sun nearly overhead at noon on the equator', () => {
    expect(sunPosition(Date.UTC(2023, 2, 20, 12, 0), 0, 0).elevationDeg).toBeGreaterThan(87);
  });

  it('rises in the east and is far below the horizon at midnight', () => {
    const dawn = sunPosition(Date.UTC(2023, 2, 20, 6, 30), 0, 0);
    expect(dawn.elevationDeg).toBeGreaterThan(0);
    expect(dawn.elevationDeg).toBeLessThan(15);
    expect(Math.abs(dawn.azimuthDeg - 90)).toBeLessThan(3);
    expect(sunPosition(Date.UTC(2023, 2, 20, 0, 0), 0, 0).elevationDeg).toBeLessThan(-60);
  });

  it('reads 2 a.m. UTC in the Gulf Stream as night, as the screenshots were', () => {
    // 2023-03-10 02:30 UTC at 28.6 N, 64.4 W: about 22:15 local solar time.
    const s = sunPosition(Date.UTC(2023, 2, 10, 2, 30), 28.6, -64.4);
    expect(s.elevationDeg).toBeLessThan(-30);
    expect(nightness(s.elevationDeg)).toBe(1);
  });

  it('turns day into night through civil twilight', () => {
    expect(nightness(10)).toBe(0);
    expect(nightness(2)).toBe(0);
    expect(nightness(-2)).toBeGreaterThan(0);
    expect(nightness(-2)).toBeLessThan(1);
    expect(nightness(-6)).toBe(1);
  });
});

describe('what the wind does to the sea', () => {
  it('has no whitecaps below force 3', () => {
    expect(whitecapFraction(0)).toBe(0);
    expect(whitecapFraction(3.3)).toBe(0);                       // force 2
  });

  it('follows Monahan and O\'Muircheartaigh above it: about 1 % at 10 m/s', () => {
    expect(whitecapFraction(10)).toBeCloseTo(3.84e-6 * 10 ** 3.41, 12);
    expect(whitecapFraction(10)).toBeGreaterThan(0.009);
    expect(whitecapFraction(10)).toBeLessThan(0.011);
    expect(whitecapFraction(15)).toBeGreaterThan(whitecapFraction(10));
    expect(whitecapFraction(60)).toBe(0.25);
  });

  it('makes the Pierson-Moskowitz peak wave about 83 m at 10 m/s, and longer in more wind', () => {
    expect(peakWavelengthM(10)).toBeCloseTo((2 * Math.PI * 100) / (0.877 ** 2 * 9.81), 6);
    expect(peakWavelengthM(10)).toBeGreaterThan(80);
    expect(peakWavelengthM(10)).toBeLessThan(86);
    expect(peakWavelengthM(15)).toBeGreaterThan(peakWavelengthM(10));
    expect(peakWavelengthM(0)).toBeGreaterThan(0);
  });

  it('repeats a seeded random sequence', () => {
    expect(seededRandom(5)()).toBe(seededRandom(5)());
  });
});

describe('floating weed', () => {
  it('lays out the same windrows for the same water, inside their tile', () => {
    expect(weedRows(3, -7)).toEqual(weedRows(3, -7));
    for (let tx = -5; tx < 5; tx += 1) {
      for (const row of weedRows(tx, 2)) {
        expect(row.x).toBeGreaterThanOrEqual(tx * WEED_TILE_M);
        expect(row.x).toBeLessThan((tx + 1) * WEED_TILE_M);
        expect(row.lengthM).toBeGreaterThan(0);
      }
    }
  });

  it('leaves most of the sea without weed, so it comes in patches', () => {
    let empty = 0;
    for (let tx = 0; tx < 40; tx += 1) for (let ty = 0; ty < 40; ty += 1) if (weedRows(tx, ty).length === 0) empty += 1;
    expect(empty / 1600).toBeGreaterThan(0.55);
    expect(empty / 1600).toBeLessThan(0.75);
  });

  it('moves exactly as the drift model moves a person: current + 2 % of wind', () => {
    const current = [0.3, -0.1];
    const wind = [-9.6, 0];
    const got = floaterStep(28.6, -64.4, current, wind, 0.02, 60);
    const want = stepPosition(28.6, -64.4, 0.3 + 0.02 * -9.6, -0.1, 60);
    expect(got[0]).toBe(want[0]);
    expect(got[1]).toBe(want[1]);
  });

  it('rides the current alone with no leeway', () => {
    const got = floaterStep(20, -70, [0.5, 0.5], [10, 10], 0, 120);
    expect(got).toEqual(stepPosition(20, -70, 0.5, 0.5, 120));
  });
});
