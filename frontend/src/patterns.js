/**
 * patterns.js -- the Coast Guard's search patterns (issues #48, #63, #75).
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
  ON_SCENE_WINDOW_S, SEARCH_SPEED_MS, SWEEP_WIDTH_M, searchEffortM2, sectorRadiusM,
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

/** (heading, length) legs visiting each point in turn, starting from the marker. */
function* legsThrough(points) {
  let here = [0, 0];
  for (const [e, n] of points) {
    const de = e - here[0];
    const dn = n - here[1];
    const length = Math.hypot(de, dn);
    if (length > 1e-9) yield [((Math.atan2(de, dn) / TO_RAD) % 360 + 360) % 360, length];
    here = [e, n];
  }
}

/** Unit vectors along a bearing and across it to the right, as [east, north]. */
function axes(bearingDeg) {
  return [eastNorth(bearingDeg, 1), eastNorth(bearingDeg + 90, 1)];
}

function point(u, v, along, across) {
  return [along * u[0] + across * v[0], along * u[1] + across * v[1]];
}

/**
 * Parallel Track (PS): legs along the major axis (`firstBearingDeg`), S apart, over a
 * rectangle centred on the marker, the first starting 1/2 S inside a corner (Figure H-34).
 * The default is the square one window covers at coverage 1: Z = W x V x T.
 */
export function parallelTrack({
  lengthM = null, widthM = null, spacingM = SWEEP_WIDTH_M, firstBearingDeg = 0,
  speedMs = SEARCH_SPEED_MS, durationS = ON_SCENE_WINDOW_S,
} = {}) {
  check('spacingM', spacingM);
  check('speedMs', speedMs);
  check('durationS', durationS);
  const side = Math.sqrt(searchEffortM2(spacingM, speedMs, durationS));
  const len = check('lengthM', lengthM ?? side);
  const wid = check('widthM', widthM ?? side);
  if (len < spacingM || wid < spacingM) {
    throw new RangeError('a parallel track area must be at least one track spacing each way');
  }
  const [u, v] = axes(firstBearingDeg);
  const legs = Math.max(1, Math.floor(wid / spacingM + 0.5));
  const near = -len / 2 + spacingM / 2;
  const far = len / 2 - spacingM / 2;
  function* points() {
    for (let k = 0; k < legs; k += 1) {
      const across = -wid / 2 + spacingM / 2 + k * spacingM;
      const [a0, a1] = k % 2 === 0 ? [near, far] : [far, near];
      yield point(u, v, a0, across);
      yield point(u, v, a1, across);
    }
  }
  const pattern = fly('parallel_track', legsThrough(points()), speedMs, durationS);
  // The area it covers, for drawing: the map outlines it around the marker.
  pattern.area = { lengthM: len, widthM: wid, bearingDeg: firstBearingDeg };
  return pattern;
}

/**
 * Trackline Return (TSR): up one side of a line through the marker and down the other,
 * 1/2 S off it (Figure H-31); each later pass widens by S (a project extension).
 */
export function tracklineReturn({
  halfLengthM, spacingM = SWEEP_WIDTH_M, firstBearingDeg = 0, speedMs = SEARCH_SPEED_MS,
  durationS = ON_SCENE_WINDOW_S,
} = {}) {
  const h = check('halfLengthM', halfLengthM);
  check('spacingM', spacingM);
  check('speedMs', speedMs);
  check('durationS', durationS);
  const [u, v] = axes(firstBearingDeg);
  function* points() {
    for (let k = 0; ; k += 1) {
      const side = k % 2 === 0 ? 1 : -1;
      const off = (k + 0.5) * spacingM;
      yield point(u, v, -h, side * off);
      yield point(u, v, h, side * off);
      yield point(u, v, h, -side * off);
      yield point(u, v, -h, -side * off);
    }
  }
  return fly('trackline_return', legsThrough(points()), speedMs, durationS);
}

/**
 * The patterns the panel offers. `short` is the button; `help` is its one line. The last
 * two are laid out by the drift model: along its predicted path, and over the area around
 * its datum aligned with the drift, as §H.7.3.9 directs in a current.
 */
export const PATTERNS = {
  expanding_square: {
    label: 'Expanding Square', short: 'Square', build: expandingSquare,
    help: 'A square spiral out from the marker: even coverage, centre first.',
  },
  sector_search: {
    label: 'Sector Search', short: 'Sector', build: sectorSearch,
    help: 'Spokes through the marker: densest right beside it.',
  },
  parallel_track: {
    label: 'Parallel Track', short: 'Parallel', build: parallelTrack,
    help: 'Straight legs over the datum\'s area, laid along the predicted drift.',
  },
  trackline_return: {
    label: 'Trackline', short: 'Trackline', build: tracklineReturn,
    help: 'Up and down the drift model\'s predicted path, wider each pass.',
  },
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
