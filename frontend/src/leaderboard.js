/**
 * leaderboard.js -- the open-day game's scores, kept on the machine it is played on (D031).
 *
 * No server: the site is static (D021) and an open-day kiosk should not depend on the
 * venue's Wi-Fi. Scores live in this browser's localStorage, every access wrapped so a
 * private window or blocked storage simply means no saved board, and can be exported as a
 * file. Nicknames only; nothing else about a visitor is kept.
 *
 * THE SCORE. POS is the measurement, so it is most of the score: 1,000 points for clearing
 * all the probability. Finding the real buoy is the moment people remember, so it is worth a
 * bonus, but it is one coin toss, so the bonus is half of a perfect POS, not more.
 *
 * WINNING (sweets). Beat both the Coast Guard pattern and the AI on POS, over the same
 * cloud. One constant, so it can be changed on the day.
 */

export const STORAGE_KEY = 'sar-open-day-leaderboard-v1';
export const FOUND_BONUS = 500;
export const MAX_ENTRIES = 500;

/** Points for a flight: 1,000 x POS, plus the bonus for finding the real buoy. */
export function score(pos, found) {
  return Math.round(1000 * pos) + (found ? FOUND_BONUS : 0);
}

/** Did the player win? They must clear more probability than every rival. */
export function verdict(you, rivals) {
  const beaten = rivals.filter((r) => you.pos > r.pos).map((r) => r.name);
  return { won: beaten.length === rivals.length, beaten };
}

export function load(storage = globalThis.localStorage) {
  try {
    const list = JSON.parse(storage?.getItem(STORAGE_KEY) ?? '[]');
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
