/**
 * leaderboard.js -- the open-day game's scores, kept on the machine it is played on (D031).
 *
 * No server: the site is static (D021) and an open-day kiosk should not depend on the
 * venue's Wi-Fi. Scores live in this browser's localStorage, every access wrapped so a
 * private window or blocked storage simply means no saved board, and can be exported as a
 * file. Nicknames only; nothing else about a visitor is kept.
 *
 * THE SCORE (12 Oct). POS alone mostly measures which buoy was picked: across the 55
 * published windows (rvc, 2 h), the Expanding Square's POS falls with the cloud's size almost
 * exactly (r = -0.99 against log area): 0.68 on a 23 km² cloud, 0.14 on a 149 km² one. So the
 * score is POS against what the Coast Guard's pattern gets on a cloud that big: 1,000 points
 * matches it, more beats it. The size is the smallest area holding 90 % of the cloud at
 * arrival, on 250 m cells; the curve PAR is fitted to the Square's POS on those 55 windows,
 * and on them the Square scores 871-1,066 on every buoy (the raw POS score ran 139-678).
 * Finding the real buoy is shown, not scored: one buoy is one coin toss.
 *
 * WINNING (sweets). Beat both the Coast Guard pattern and the AI on POS, over the same
 * cloud. One constant, so it can be changed on the day.
 */

// v2: points by cloud size (12 Oct). The v1 board is brought over once, rescored (load).
export const STORAGE_KEY = 'sar-open-day-leaderboard-v2';
export const MAX_ENTRIES = 500;
/** The Expanding Square's POS on a cloud of A km²: PAR.k * A^PAR.power. */
export const PAR = {
  k: 8.4453, power: -0.7934, minKm2: 22.4, maxKm2: 148.8,
  fittedOn: 'scenarios/v2, rvc, 2 h, the 55 windows, 12 Oct 2026',
};
export const AREA_SHARE = 0.9;
export const AREA_CELL_M = 250;

/**
 * What the Coast Guard's pattern searches, on average, of a cloud of `areaKm2`. Outside the
 * sizes it was fitted on, the nearest end: the curve is not stretched past what was measured.
 */
export function par(areaKm2) {
  const a = Math.min(PAR.maxKm2, Math.max(PAR.minKm2, areaKm2));
  return PAR.k * a ** PAR.power;
}

/**
 * The smallest area holding `share` of the probability, km², from a grid's cell masses:
 * the likeliest cells first, as many as it takes.
 */
export function cloudAreaKm2(mass, cellM = AREA_CELL_M, share = AREA_SHARE) {
  const sorted = Float64Array.from(mass).sort().reverse();
  const total = sorted.reduce((s, m) => s + m, 0);
  let held = 0;
  let cells = 0;
  while (cells < sorted.length && held < share * total - 1e-12) held += sorted[cells++];
  return (cells * cellM * cellM) / 1e6;
}

/** Points for a flight: 1,000 x POS over what the Coast Guard's pattern gets on a cloud that big. */
export function score(pos, areaKm2) {
  return Math.round((1000 * pos) / par(areaKm2));
}

/** Did the player win? They must clear more probability than every rival. */
export function verdict(you, rivals) {
  const beaten = rivals.filter((r) => you.pos > r.pos).map((r) => r.name);
  return { won: beaten.length === rivals.length, beaten };
}

// The board before 12 Oct, scored 1,000 x POS + 500 if found. Its entries keep their POS and
// their buoy, so they are rescored by the new rule rather than dropped.
export const OLD_STORAGE_KEY = 'sar-open-day-leaderboard-v1';
/**
 * Each buoy's cloud size where the game flies it (rvc, 2 h), km²: measured on the published v2
 * windows on 12 Oct (scratch `window_sizes.py`). Only for rescoring the old board.
 */
export const GAME_AREA_KM2 = {
  S01: 111.4, S02: 67.3, S03: 139.5, S04: 42.6, S05: 68.3, S06: 42.1, S07: 37.0, S08: 91.8,
  S09: 97.9, S10: 32.9, S11: 148.8, S12: 36.3, S13: 90.4, S14: 23.6, S15: 34.9, S16: 26.8,
  S17: 28.5, S18: 30.4, S19: 27.6, S20: 35.0, S21: 28.1, S22: 30.4, S23: 38.4, S24: 42.0,
  S25: 23.9, S26: 24.3, S27: 44.8, S28: 88.1, S29: 28.5, S30: 74.9, S31: 27.7, S32: 23.4,
  S33: 29.1, S34: 32.6, S35: 35.9, S36: 24.1, S37: 26.9, S38: 24.1, S39: 23.3, S40: 23.6,
  S41: 63.7, S42: 60.7, S43: 91.8, S44: 41.8, S45: 29.6, S46: 22.4, S47: 60.9, S48: 26.2,
  S49: 24.1, S50: 45.6, S51: 111.7, S52: 36.1, S53: 27.8, S54: 22.9, S55: 25.1,
};

/** Old entries on the new scale: same POS, same buoy, points by its cloud's size. */
export function rescore(entries) {
  return entries
    .filter((e) => Number.isFinite(e?.pos) && GAME_AREA_KM2[e.scenario])
    .map((e) => ({ ...e, area_km2: GAME_AREA_KM2[e.scenario], score: score(e.pos, GAME_AREA_KM2[e.scenario]),
      old_score: e.score, rescored: true }))
    .sort((a, b) => b.score - a.score || String(a.when).localeCompare(String(b.when)));
}

export function load(storage = globalThis.localStorage) {
  try {
    const held = storage?.getItem(STORAGE_KEY);
    if (held == null) {
      // First load on the new scale: bring the old board over, rescored, once.
      const old = JSON.parse(storage?.getItem(OLD_STORAGE_KEY) ?? '[]');
      const list = Array.isArray(old) ? rescore(old) : [];
      if (list.length) storage.setItem(STORAGE_KEY, JSON.stringify(list));
      return list;
    }
    const list = JSON.parse(held);
    return Array.isArray(list) ? list : [];
  } catch {
    return [];
  }
}

/** Add an entry; returns the board, best first, and whether it could be saved. */
export function add(entry, storage = globalThis.localStorage) {
  const list = [...load(storage), entry]
    .sort((a, b) => b.score - a.score || a.when.localeCompare(b.when))
    .slice(0, MAX_ENTRIES);
  try {
    storage.setItem(STORAGE_KEY, JSON.stringify(list));
    return { list, saved: true };
  } catch {
    return { list, saved: false };
  }
}

/**
 * The best n players for one mode: one line per name (the same nickname, ignoring case and
 * spaces at the ends, is the same player), with that player's highest score. Every flight is
 * still kept in storage and in the exported file.
 */
export function top(list, mode, n = 8) {
  const best = new Map();
  for (const e of list) {
    if (e.mode !== mode) continue;
    const key = String(e.name ?? '').trim().toLowerCase();
    const held = best.get(key);
    if (!held || e.score > held.score || (e.score === held.score && e.when < held.when)) {
      best.set(key, e);
    }
  }
  return [...best.values()]
    .sort((a, b) => b.score - a.score || String(a.when).localeCompare(String(b.when)))
    .slice(0, n);
}

/** A nickname fit for a public screen: trimmed, short, plain characters only. */
export function nickname(text) {
  const clean = String(text ?? '').replace(/[^\p{L}\p{N} _.-]/gu, '').trim().slice(0, 16);
  return clean || 'Player';
}
