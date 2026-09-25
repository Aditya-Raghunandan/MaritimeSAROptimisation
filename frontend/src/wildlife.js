/**
 * wildlife.js -- what turns up in the close-up sea, and when (issue #79).
 *
 * Decoration, and the key says so: nothing here is data, nothing here touches detection.
 * But it is this sea's life, not any sea's: flying fish, which the Sargasso is full of;
 * bottlenose and spotted dolphins; loggerhead and green turtles, whose young live in the
 * Sargassum; and humpbacks, which winter in the Caribbean from December to April, so they
 * only appear then.
 *
 * RARE, AND ONE AT A TIME. A sighting is a surprise, not wallpaper: the first after a
 * quarter to half a minute close up, then on average about one every minute and a half,
 * never two at once. Times are REAL seconds, like the waves: animals move at the speed
 * animals move, whatever speed the search is played at.
 *
 * Pure -- the drawing is wildlifeDraw.js -- so the rarity is tested, not hoped for.
 */

import { seededRandom } from './closeUp.js';

export const SPECIES = {
  flyingFish: { label: 'flying fish', weight: 5, durationS: [4.5, 6.5], count: [4, 8] },
  dolphins: { label: 'dolphins', weight: 3, durationS: [11, 16], count: [3, 6] },
  turtle: { label: 'a sea turtle', weight: 2, durationS: [12, 17], count: [1, 1] },
  humpback: { label: 'a humpback whale', weight: 0.9, durationS: [14, 19], count: [1, 1], months: [11, 0, 1, 2, 3] },
};

/** Average quiet time between sightings, and the least, in real seconds. */
export const MEAN_GAP_S = 75;
export const MIN_GAP_S = 25;
const FIRST_GAP_S = [15, 35];

function between(rand, [lo, hi]) {
  return lo + (hi - lo) * rand();
}

/** Which animal, by weight, among those in season. `month` is 0-11, UTC. */
export function pickSpecies(rand, { month = 0 } = {}) {
  const ok = Object.entries(SPECIES).filter(([, s]) => !s.months || s.months.includes(month));
  const total = ok.reduce((sum, [, s]) => sum + s.weight, 0);
  let r = rand() * total;
  for (const [kind, s] of ok) {
    r -= s.weight;
    if (r <= 0) return kind;
  }
  return ok[ok.length - 1][0];
}

/**
 * The scheduler. `step(dtS, ctx)` advances real time and returns a new sighting when one
 * starts, else null; `active` is the one on screen, if any. A sighting says where on the
 * SCREEN it starts (x, y as fractions, kept clear of the side panels) and which way it
 * goes; the layer pins it to the water there.
 */
export class Wildlife {
  constructor({ seed = 20260925, meanGapS = MEAN_GAP_S, minGapS = MIN_GAP_S } = {}) {
    this._rand = seededRandom(seed);
    this._meanGapS = meanGapS;
    this._minGapS = minGapS;
    this.clock = 0;
    this.active = null;
    this._next = between(this._rand, FIRST_GAP_S);
  }

  /** Start the wait again, as when the view comes back close up. */
  rest() {
    this.active = null;
    this._next = this.clock + between(this._rand, FIRST_GAP_S);
  }

  /** Start a sighting of `kind` now, replacing any on screen. Null for an unknown kind. */
  summon(kind) {
    if (!SPECIES[kind]) return null;
    this.active = null;
    this._forced = kind;
    this._next = this.clock;
    return kind;
  }

  step(dtS, { month = 0 } = {}) {
    this.clock += Math.max(0, dtS);
    if (this.active) {
      if (this.clock - this.active.bornS < this.active.durationS) return null;
      this.active = null;
      // Exponential, so sightings do not arrive like a metronome, above a floor.
      const extra = -Math.log(1 - this._rand() * 0.999) * (this._meanGapS - this._minGapS);
      this._next = this.clock + this._minGapS + extra;
      return null;
    }
    if (this.clock < this._next) return null;
    const rand = this._rand;
    const kind = this._forced ?? pickSpecies(rand, { month });
    this._forced = null;
    const s = SPECIES[kind];
    this.active = {
      kind,
      label: s.label,
      bornS: this.clock,
      durationS: between(rand, s.durationS),
      count: Math.round(between(rand, s.count)),
      x: between(rand, [0.34, 0.7]),
      y: between(rand, [0.28, 0.68]),
      headingDeg: rand() * 360,
      seed: Math.floor(rand() * 2 ** 31),
    };
    return this.active;
  }
}
