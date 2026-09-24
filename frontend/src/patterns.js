/**
 * patterns.js -- the Coast Guard's Expanding Square and Sector Search (issues #48, #63).
 *
 * A port of `src/sar/search/patterns.py`, which is the reference; the geometry and why
 * it is flown about a drifting marker are argued there and in docs/ADR003.md, cited to
 * the USCG Addendum by page. `tests/patterns.test.js` holds every waypoint, offset and
 * heading here to Python through `fixtures/search_golden.json`.
 *
 * A pattern is waypoints in metres EAST and NORTH OF THE MARKER, each with the time the
 * helicopter reaches it. Never 60 s samples: the Expanding Square's first leg takes 4 s
 * at 0.1 NM and 90 kt, so joining samples a minute apart would cut across the pattern.
 *
 * Pure arithmetic, no DOM.
 */

import {
  ON_SCENE_WINDOW_S, SEARCH_SPEED_MS, SWEEP_WIDTH_M, sectorRadiusM,
} from './platform.js';

const TO_RAD = Math.PI / 180;

/** Metres per degree of latitude, as `sar.utils.geo.M_PER_DEG_LAT`. */
export const M_PER_DEG_LAT = 111320;

/** One Sector Search pattern as turns from its first leg (Addendum pp. 3-27 to 3-28). */
export const SECTOR_LEGS = [0, 120, 240, 240, 0, 120, 120, 240, 0];
export const SECTOR_ROTATION_DEG = 30;

const TIME_TOLERANCE_S = 1e-9;

/** A distance along a compass bearing as [east, north] metres. */
export function eastNorth(bearingDeg, distanceM) {
  const b = bearingDeg * TO_RAD;
  return [distanceM * Math.sin(b), distanceM * Math.cos(b)];
}

/** The point [lat, lon] that is (east, north) metres from (lat, lon), cos(lat) at the start. */
export function offsetPosition(lat, lon, eastM, northM) {
  return [
    lat + northM / M_PER_DEG_LAT,
    lon + eastM / (M_PER_DEG_LAT * Math.cos(lat * TO_RAD)),
  ];
}

function check(name, value) {
  if (!Number.isFinite(value) || value <= 0) {
    throw new RangeError(`${name} must be a positive number, got ${value}`);
  }
  return value;
}

/** Fly (heading, length) legs from the marker until the window ends, mid-leg if need be. */
function fly(kind, legs, speedMs, durationS) {
  const eastM = [0];
  const northM = [0];
  const tS = [0];
  const headingDeg = [];
  for (const [heading, length] of legs) {
    const left = durationS - tS[tS.length - 1];
    if (left <= TIME_TOLERANCE_S) break;
    const flown = Math.min(length, left * speedMs);
    const [de, dn] = eastNorth(heading, flown);
    eastM.push(eastM[eastM.length - 1] + de);
    northM.push(northM[northM.length - 1] + dn);
    tS.push(tS[tS.length - 1] + flown / speedMs);
    headingDeg.push(((heading % 360) + 360) % 360);
  }
  return { kind, eastM, northM, tS, headingDeg, speedMs, durationS: tS[tS.length - 1] };
}

/** The Expanding Square (SS): legs S, S, 2S, 2S, ..., every turn 90 degrees right. */
export function expandingSquare({
  spacingM = SWEEP_WIDTH_M, firstBearingDeg = 0, speedMs = SEARCH_SPEED_MS,
  durationS = ON_SCENE_WINDOW_S,
} = {}) {
  check('spacingM', spacingM);
  check('speedMs', speedMs);
  check('durationS', durationS);
  function* legs() {
    for (let k = 0; ; k += 1) yield [firstBearingDeg + 90 * k, (Math.floor(k / 2) + 1) * spacingM];
  }
  return fly('expanding_square', legs(), speedMs, durationS);
}

/** The Sector Search (VS): nine legs of R per pattern, each pattern 30 degrees right. */
export function sectorSearch({
  radiusM = null, firstBearingDeg = 0, speedMs = SEARCH_SPEED_MS,
  durationS = ON_SCENE_WINDOW_S,
} = {}) {
  check('speedMs', speedMs);
  const r = check('radiusM', radiusM ?? sectorRadiusM(speedMs));
  check('durationS', durationS);
  function* legs() {
    for (let p = 0; ; p += 1) {
      const base = firstBearingDeg + SECTOR_ROTATION_DEG * p;
      for (const turn of SECTOR_LEGS) yield [base + turn, r];
    }
  }
  return fly('sector_search', legs(), speedMs, durationS);
}

export const PATTERNS = {
  expanding_square: { label: 'Expanding Square', build: expandingSquare },
  sector_search: { label: 'Sector Search', build: sectorSearch },
};

/** Index of the leg in progress at t: the last waypoint at or before it, clamped. */
function legAt(times, t) {
  let lo = 0;
  let hi = times.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (times[mid] <= t) lo = mid; else hi = mid - 1;
  }
  return lo;
}

/** Linear interpolation over a strictly increasing axis, as `np.interp`. */
function interp(t, xs, ys) {
  if (t <= xs[0]) return ys[0];
  const n = xs.length - 1;
  if (t >= xs[n]) return ys[n];
  const k = legAt(xs, t);
  const w = (t - xs[k]) / (xs[k + 1] - xs[k]);
  return ys[k] + w * (ys[k + 1] - ys[k]);
}

/** [east, north] metres from the marker at t seconds, or null outside the pattern. */
export function offsetAt(pattern, t) {
  if (t < -TIME_TOLERANCE_S || t > pattern.durationS + TIME_TOLERANCE_S) return null;
  return [interp(t, pattern.tS, pattern.eastM), interp(t, pattern.tS, pattern.northM)];
}

/** The heading of the leg being flown at t. At a waypoint, the leg leaving it. */
export function headingAt(pattern, t) {
  if (t < -TIME_TOLERANCE_S || t > pattern.durationS + TIME_TOLERANCE_S) return null;
  const k = Math.min(legAt(pattern.tS, t), pattern.headingDeg.length - 1);
  return pattern.headingDeg[k];
}

/** The marker's [lat, lon] at t seconds after the drop, or null outside its track. */
export function markerAt(marker, t) {
  const { tS } = marker;
  if (t < tS[0] - TIME_TOLERANCE_S || t > tS[tS.length - 1] + TIME_TOLERANCE_S) return null;
  return [interp(t, tS, marker.lat), interp(t, tS, marker.lon)];
}

/** The helicopter's [lat, lon] at t seconds into the pattern: the marker plus the offset. */
export function onGround(pattern, marker, t) {
  const m = markerAt(marker, t);
  const o = offsetAt(pattern, t);
  if (!m || !o) return null;
  return offsetPosition(m[0], m[1], o[0], o[1]);
}
