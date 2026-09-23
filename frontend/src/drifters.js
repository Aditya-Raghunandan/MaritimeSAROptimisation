/**
 * drifters.js -- the logic behind the drifter layer, with no map and no DOM (issue #50).
 *
 * The drifters are the project's TEST DATA: real buoys the drift engine is scored
 * against (vault D018, D025). The layer does two jobs, kept apart on purpose:
 *
 *   FINDING   follows the span control. Every buoy seen in the current window is a
 *             faint dot where it was first seen; the list and its search hold the same
 *             buoys, and the search reaches every buoy whatever the window.
 *   WATCHING  follows the clock. Buoys in the water at the current moment are solid,
 *             at their position now, and move on play. The selected buoy's path draws
 *             behind it as time passes.
 *
 * Everything here is a plain function of plain data, so it is tested without a
 * browser, the way geo.js and drift.js are.
 */

const HOUR_MS = 3600 * 1000;

/**
 * A track file as typed arrays: times in epoch ms, and per-fix flags.
 *
 * The file stores whole hours since `t0`, which is exact for both GDP products
 * (every fix is on the hour). Lengths must agree, or the file is refused by name.
 */
export function decodeTrack(json) {
  const n = json.h.length;
  for (const key of ['lat', 'lon', 'seg', 'und', 'far']) {
    if (json[key].length !== n) {
      throw new Error(`track ${json.id}: ${key} has ${json[key].length} values for ${n} fixes`);
    }
  }
  const t0 = Date.parse(json.t0);
  const t = new Float64Array(n);
  for (let k = 0; k < n; k += 1) t[k] = t0 + json.h[k] * HOUR_MS;
  return {
    id: json.id,
    t,
    lat: Float64Array.from(json.lat),
    lon: Float64Array.from(json.lon),
    seg: Int32Array.from(json.seg),
    und: Uint8Array.from(json.und),
    far: Uint8Array.from(json.far),
  };
}

/** The index's buoys with their times parsed once, sorted by first fix. */
export function prepareIndex(manifest) {
  return manifest.buoys
    .map((b) => ({ ...b, startMs: Date.parse(b.start), endMs: Date.parse(b.end) }))
    .sort((a, b) => a.startMs - b.startMs || (a.id < b.id ? -1 : 1));
}

/** Buoys whose record overlaps the window `[winStart, winEnd)`, in ms. */
export function inWindow(entries, winStart, winEnd) {
  return entries.filter((e) => e.startMs < winEnd && e.endMs >= winStart);
}

/**
 * Buoys whose record spans the moment `when`. A buoy can still be in a gap between
 * segments at that moment; `positionAt` says so by returning null.
 */
export function spanningNow(entries, when) {
  return entries.filter((e) => e.startMs <= when && e.endMs >= when);
}

const DATE_QUERY = /^\d{4}(-\d{2}(-\d{2})?)?$/;

/** The UTC period a date query names, as `[start, end)` in ms. */
function periodOf(query) {
  const [y, m, d] = query.split('-').map(Number);
  if (d) return [Date.UTC(y, m - 1, d), Date.UTC(y, m - 1, d + 1)];
  if (m) return [Date.UTC(y, m - 1, 1), Date.UTC(y, m, 1)];
  return [Date.UTC(y, 0, 1), Date.UTC(y + 1, 0, 1)];
}

/**
 * Search EVERY buoy, whatever the window: this is the one way out of the span filter,
 * so a buoy named in the paper can always be found.
 *
 * Matches an ID fragment, or a date (`2021`, `2021-03`, `2021-03-05`) that the buoy's
 * record overlaps. A bare year could be either, so it matches both.
 *
 * For a date, buoys FIRST SEEN in that period come first. A long-lived buoy that only
 * passes through it still matches, but listed first it sent a search for March 2021 to
 * a buoy first seen on 1 January 2019, which is correct and not what anyone meant.
 */
export function searchBuoys(entries, query) {
  const q = (query ?? '').trim();
  if (!q) return [];
  const byDate = DATE_QUERY.test(q) ? periodOf(q) : null;
  const hits = entries.filter((e) => e.id.includes(q)
    || (byDate && e.startMs < byDate[1] && e.endMs >= byDate[0]));
  if (!byDate) return hits;
  const startsIn = (e) => e.startMs >= byDate[0] && e.startMs < byDate[1];
  return hits.sort((a, b) => (startsIn(b) - startsIn(a)) || (a.startMs - b.startMs));
}

/**
 * The smallest span whose window holds the buoy's whole record, or null for the
 * whole archive.
 *
 * The clock aligns a window to a whole multiple of its span from the epoch (a day is a
 * UTC day), so "covers the track" means the ALIGNED window containing the first fix
 * also contains the last one. A track that crosses a boundary needs the next span up.
 */
export function spanCovering(startMs, endMs, spansSeconds) {
  const spans = spansSeconds.filter((s) => s > 0).sort((a, b) => a - b);
  for (const s of spans) {
    const ms = s * 1000;
    const from = Math.floor(startMs / ms) * ms;
    if (endMs < from + ms) return s;
  }
  return null;
}

/** Index of the last fix at or before `when`, or -1 if `when` is before the first. */
export function lastFixAtOrBefore(track, when) {
  const { t } = track;
  let lo = 0;
  let hi = t.length - 1;
  if (t.length === 0 || when < t[0]) return -1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (t[mid] <= when) lo = mid; else hi = mid - 1;
  }
  return lo;
}

/**
 * Where the buoy is at `when`: on a fix, or interpolated between two fixes of the SAME
 * segment. Null before the first fix, after the last, and inside a gap between
 * segments, because a buoy with no fixes for days is not somewhere we can draw.
 */
export function positionAt(track, when) {
  const k = lastFixAtOrBefore(track, when);
  if (k < 0) return null;
  if (track.t[k] === when) return [track.lat[k], track.lon[k]];
  if (k === track.t.length - 1 || track.seg[k] !== track.seg[k + 1]) return null;
  const w = (when - track.t[k]) / (track.t[k + 1] - track.t[k]);
  return [
    track.lat[k] + w * (track.lat[k + 1] - track.lat[k]),
    track.lon[k] + w * (track.lon[k + 1] - track.lon[k]),
  ];
}

/** The drogue state at `when`: that of the last fix at or before it, or null. */
export function undroguedAt(track, when) {
  const k = lastFixAtOrBefore(track, when);
  return k < 0 ? null : track.und[k] === 1;
}

/**
 * The path drawn so far, up to `when`, as pieces with one style each.
 *
 * Each step between two consecutive fixes of one segment gets the drogue state of
 * the fix it arrives at, and is `far` if either end is more than 3 h from a real
 * GPS fix. Consecutive steps with the same style join into one piece. No step is
 * drawn across a gap between segments. The last piece ends at the interpolated
 * position now, so the trail stays attached to the marker between fixes.
 */
export function trailUpTo(track, when) {
  const pieces = [];
  const last = lastFixAtOrBefore(track, when);
  let current = null;
  for (let k = 1; k <= last; k += 1) {
    if (track.seg[k] !== track.seg[k - 1]) { current = null; continue; }
    const undrogued = track.und[k] === 1;
    const far = track.far[k] === 1 || track.far[k - 1] === 1;
    if (!current || current.undrogued !== undrogued || current.far !== far) {
      current = { undrogued, far, points: [[track.lat[k - 1], track.lon[k - 1]]] };
      pieces.push(current);
    }
    current.points.push([track.lat[k], track.lon[k]]);
  }
  const now = positionAt(track, when);
  if (now && last >= 0 && track.t[last] !== when) {
    const undrogued = track.und[last] === 1;
    const far = track.far[last] === 1 || track.far[last + 1] === 1;
    if (!current || current.undrogued !== undrogued || current.far !== far) {
      current = { undrogued, far, points: [[track.lat[last], track.lon[last]]] };
      pieces.push(current);
    }
    current.points.push(now);
  }
  return pieces;
}

/**
 * Where the selected buoy's record sits on the slider, as fractions of the window:
 * `{ left, width }`, clipped to the window, or null if it does not overlap.
 */
export function lifetimeBar(startMs, endMs, winStart, winEnd) {
  const span = winEnd - winStart;
  if (!(span > 0) || endMs < winStart || startMs >= winEnd) return null;
  const left = Math.max(0, (startMs - winStart) / span);
  const right = Math.min(1, (endMs - winStart) / span);
  return { left, width: Math.max(right - left, 0.004) };
}

const TIER_WORDS = {
  drogued: 'drogued: follows the water',
  undrogued: 'undrogued: the wind pushes it too',
  mixed: 'loses its drogue part-way',
  uncertain: 'drogue records disagree',
};

/** `2019-03-02`: a date as a reader names it. Every source here is UTC. */
export function day(ms) {
  return new Date(ms).toISOString().slice(0, 10);
}

/** Whole days a record spans, at least one. */
export function lengthDays(entry) {
  return Math.max(1, Math.round((entry.endMs - entry.startMs) / (24 * HOUR_MS)));
}

/** The words a tooltip or a list row shows for one buoy. */
export function describeBuoy(entry) {
  const days = lengthDays(entry);
  return {
    title: `Buoy ${entry.id}`,
    dates: `${day(entry.startMs)} → ${day(entry.endMs)} · ${days} day${days === 1 ? '' : 's'}`,
    tier: TIER_WORDS[entry.tier] ?? entry.tier,
    sealed: entry.sealed
      ? 'sealed: look at the track, but do not debug the engine on it (D025)'
      : null,
  };
}
