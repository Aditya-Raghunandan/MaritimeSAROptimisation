/**
 * What the compass says, and how rough the close-up sea is drawn (issue #73).
 */

import { describe, expect, it } from 'vitest';

import { compassReadout, whitecapCount } from '../src/seaState.js';

describe('compassReadout', () => {
  it('reads a heading as three digits and a compass point', () => {
    const r = compassReadout({ heading: 45 });
    expect(r.heading).toBe(45);
    expect(r.lines[0]).toBe('Heading 045° NE');
  });

  it('wraps a heading into 0-360', () => {
    expect(compassReadout({ heading: -90 }).heading).toBe(270);
  });

  it('points the current the way the water goes', () => {
    const r = compassReadout({ current: [0, 1.5] });   // flowing north
    expect(r.current.towards).toBeCloseTo(0, 9);
    expect(r.current.speed).toBeCloseTo(1.5, 9);
    expect(r.lines[1]).toBe('Current 1.50 m/s towards N');
  });

  it('says where the wind comes from, and its Beaufort force', () => {
    const r = compassReadout({ wind: [-5, 0] });       // blowing towards the west, from the east
    expect(r.wind.from).toBeCloseTo(90, 9);
    expect(r.wind.towards).toBeCloseTo(270, 9);
    expect(r.lines[2]).toMatch(/from E · Force 3/);
  });

  it('carries the Addendum caveat above 15 kt, and not below', () => {
    expect(compassReadout({ wind: [7, 0] }).caveat).toBeNull();          // 13.6 kt
    expect(compassReadout({ wind: [8, 0] }).caveat).toMatch(/Table H-10/); // 15.6 kt
  });

  it('shows a dash for anything not yet loaded', () => {
    const r = compassReadout({});
    expect(r.lines).toEqual(['Heading —', 'Current —', 'Wind —']);
    expect(compassReadout({ current: [NaN, 1] }).current).toBeNull();
  });
});

describe('whitecapCount', () => {
  it('draws none in a calm or a light breeze', () => {
    expect(whitecapCount(0)).toBe(0);
    expect(whitecapCount(2)).toBe(0);
  });

  it('grows with the force, and levels off in a gale', () => {
    expect(whitecapCount(3)).toBeGreaterThan(0);
    expect(whitecapCount(5)).toBeGreaterThan(whitecapCount(4));
    expect(whitecapCount(12)).toBe(whitecapCount(7));
  });
});
