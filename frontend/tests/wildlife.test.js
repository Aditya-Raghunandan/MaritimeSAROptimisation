/**
 * The close-up wildlife is rare, one at a time, and in season (issue #79).
 */

import { describe, expect, it } from 'vitest';

import { seededRandom } from '../src/closeUp.js';
import { MIN_GAP_S, SPECIES, Wildlife, pickSpecies } from '../src/wildlife.js';

/** Run the scheduler for `seconds` of real time and return every sighting. */
function watch(seconds, { seed = 1, month = 2, dt = 0.1 } = {}) {
  const w = new Wildlife({ seed });
  const seen = [];
  for (let t = 0; t < seconds; t += dt) {
    const s = w.step(dt, { month });
    if (s) seen.push({ ...s });
  }
  return seen;
}

describe('Wildlife', () => {
  it('is rare: under one sighting a minute over an hour, whatever the seed', () => {
    for (const seed of [1, 2, 3, 4, 5]) {
      const seen = watch(3600, { seed });
      expect(seen.length).toBeLessThanOrEqual(60);
      expect(seen.length).toBeGreaterThanOrEqual(15);   // rare, not absent
    }
  });

  it('never shows two at once, and rests at least the minimum between them', () => {
    const seen = watch(3600, { seed: 9 });
    for (let k = 1; k < seen.length; k += 1) {
      const prev = seen[k - 1];
      expect(seen[k].bornS).toBeGreaterThanOrEqual(prev.bornS + prev.durationS + MIN_GAP_S - 0.11);
    }
  });

  it('shows the first one within the first minute close up', () => {
    const seen = watch(60, { seed: 4 });
    expect(seen.length).toBeGreaterThanOrEqual(1);
  });

  it('starts sightings clear of the side panels', () => {
    for (const s of watch(3600, { seed: 7 })) {
      expect(s.x).toBeGreaterThanOrEqual(0.34);
      expect(s.x).toBeLessThanOrEqual(0.7);
      expect(s.y).toBeGreaterThanOrEqual(0.28);
      expect(s.y).toBeLessThanOrEqual(0.68);
    }
  });

  it('is the same sequence for the same seed', () => {
    expect(watch(600, { seed: 3 })).toEqual(watch(600, { seed: 3 }));
  });
});

describe('pickSpecies', () => {
  it('never offers a humpback outside December to April', () => {
    const rand = seededRandom(11);
    for (let k = 0; k < 5000; k += 1) expect(pickSpecies(rand, { month: 6 })).not.toBe('humpback');
  });

  it('offers every animal in winter, flying fish most often', () => {
    const rand = seededRandom(12);
    const counts = {};
    for (let k = 0; k < 20000; k += 1) {
      const kind = pickSpecies(rand, { month: 1 });
      counts[kind] = (counts[kind] ?? 0) + 1;
    }
    for (const kind of Object.keys(SPECIES)) expect(counts[kind]).toBeGreaterThan(0);
    expect(counts.flyingFish).toBeGreaterThan(counts.dolphins);
    expect(counts.dolphins).toBeGreaterThan(counts.humpback);
  });
});

describe('summon', () => {
  it('starts the asked-for animal on the next step, out of season or not', () => {
    const w = new Wildlife({ seed: 2 });
    expect(w.summon('humpback')).toBe('humpback');
    const s = w.step(0.1, { month: 6 });
    expect(s.kind).toBe('humpback');
    expect(w.summon('kraken')).toBeNull();
  });
});
