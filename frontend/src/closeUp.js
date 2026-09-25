/**
 * closeUp.js -- the rules of the close-up view, with nothing drawn (issue #79).
 *
 * Close up, the forcing is the same everywhere on screen. At the 1 km scale bar the view
 * is about 17 km wide: two current cells (0.08 x 0.04 deg, about 8 x 4.5 km) and less
 * than one wind cell (about 25 km). Arrows or streaks there would all run parallel. What
 * DOES differ close up is how three things move, so the view shows those:
 *
 *   the water     its texture and whitecaps are carried by the current;
 *   the wind      sets the waves (their scale, from Pierson-Moskowitz) and how much of
 *                 the sea is white (Monahan & O'Muircheartaigh 1980);
 *   floating weed drifts at current + 2 % of wind: the drift model (D002), made visible.
 *
 * The light follows the real sun at that time and place, so a search at night looks
 * like night. Everything here is pure, so it is tested without a browser; seaGL.js and
 * closeUpLayer.js only draw what it returns.
 */

import { beaufort } from './beaufort.js';
import { stepPosition } from './pointDrift.js';

const TO_RAD = Math.PI / 180;
const G = 9.81;

/** The zoom the close-up is fully on from; it fades in over the half zoom below. */
export const CLOSE_ZOOM = 13.5;
const FADE_ZOOMS = 0.5;

/** How much of the close-up to draw at a zoom: 0 below 13, 1 from 13.5. */
export function closeUpWeight(zoom) {
  const x = (zoom - (CLOSE_ZOOM - FADE_ZOOMS)) / FADE_ZOOMS;
  if (!(x > 0)) return 0;
  if (x >= 1) return 1;
  return x * x * (3 - 2 * x);
}

/** Whether the view counts as close up: the basemap, legends and key switch here. */
export function isCloseUp(zoom) {
  return zoom >= CLOSE_ZOOM;
}

/** Web Mercator metres per screen pixel at a latitude and (fractional) zoom. */
export function metresPerPixel(lat, zoom) {
  return (40075016.686 * Math.cos(lat * TO_RAD)) / (256 * 2 ** zoom);
}

/** m/s to knots. */
export function toKnots(ms) {
  return (ms * 3600) / 1852;
}

/**
 * Where the sun is, in degrees: elevation above the horizon and azimuth clockwise from
 * north. The low-precision solar position of the Astronomical Almanac (about 0.01 deg in
 * declination), which is far finer than the light it sets.
 */
export function sunPosition(ms, lat, lon) {
  const d = ms / 86400000 + 2440587.5 - 2451545.0;          // days since J2000.0
  const g = (357.529 + 0.98560028 * d) * TO_RAD;           // mean anomaly
  const q = 280.459 + 0.98564736 * d;                       // mean longitude, deg
  const L = (q + 1.915 * Math.sin(g) + 0.020 * Math.sin(2 * g)) * TO_RAD;
  const e = (23.439 - 0.00000036 * d) * TO_RAD;            // obliquity
  const ra = Math.atan2(Math.cos(e) * Math.sin(L), Math.cos(L));
  const dec = Math.asin(Math.sin(e) * Math.sin(L));
  const gmstHours = 18.697374558 + 24.06570982441908 * d;
  const hourAngle = ((gmstHours * 15 + lon) * TO_RAD) - ra;
  const phi = lat * TO_RAD;
  const sinEl = Math.sin(phi) * Math.sin(dec) + Math.cos(phi) * Math.cos(dec) * Math.cos(hourAngle);
  const el = Math.asin(Math.max(-1, Math.min(1, sinEl)));
  const az = Math.atan2(-Math.sin(hourAngle), Math.cos(phi) * Math.tan(dec) - Math.sin(phi) * Math.cos(hourAngle));
  return { elevationDeg: el / TO_RAD, azimuthDeg: ((az / TO_RAD) % 360 + 360) % 360 };
}

/** How dark it is, 0 in daylight to 1 at night, through civil twilight (0 to -6 deg). */
export function nightness(elevationDeg) {
  const x = (2 - elevationDeg) / 8;          // 2 deg above the horizon .. 6 below
  if (!(x > 0)) return 0;
  if (x >= 1) return 1;
  return x * x * (3 - 2 * x);
}

/**
 * The fraction of the sea surface that is white, from the 10 m wind: Monahan &
 * O'Muircheartaigh (1980), W = 3.84e-6 U^3.41. Zero below force 3, where the WMO sea-state
 * wording has no whitecaps, and capped at a quarter, past the winds this study meets.
 */
export function whitecapFraction(windMs) {
  if (!(windMs > 0) || beaufort(windMs).force < 3) return 0;
  return Math.min(0.25, 3.84e-6 * windMs ** 3.41);
}

/**
 * The peak wavelength of a fully developed wind sea, from Pierson-Moskowitz: the peak
 * frequency is 0.877 g / U, and deep water has lambda = 2 pi g / omega^2. About 83 m at
 * 10 m/s. A fully developed sea needs a long fetch and hours of wind; out here there
 * usually is both, and it is decoration either way.
 */
export function peakWavelengthM(windMs) {
  const u = Math.max(0, windMs || 0);
  const omega = (0.877 * G) / Math.max(u, 0.5);
  return Math.min(250, Math.max(0.6, (2 * Math.PI * G) / (omega * omega)));
}

/** A small, fast, seeded random source (mulberry32), so a scene is repeatable. */
export function seededRandom(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** The side of one square of floating weed's own frame, metres. */
export const WEED_TILE_M = 700;

/**
 * The Sargassum windrows in one tile of the weed's frame, as { x, y, lengthM, widthM,
 * clumpM, seed } with the centre in metres. Wind over the sea sets up Langmuir cells --
 * counter-rotating rolls lying along the wind -- that sweep floating weed into lines
 * parallel to it, so the drawing lays each row along the wind. Deterministic by tile, so
 * the same water always carries the same weed.
 */
export function weedRows(tx, ty) {
  const rand = seededRandom(((tx * 73856093) ^ (ty * 19349663) ^ 0x5bd1e995) >>> 0);
  const r = rand();
  const n = r < 0.65 ? 0 : r < 0.93 ? 1 : 2;
  const rows = [];
  for (let k = 0; k < n; k += 1) {
    rows.push({
      x: (tx + rand()) * WEED_TILE_M,
      y: (ty + rand()) * WEED_TILE_M,
      lengthM: 70 + 330 * rand(),
      widthM: 3 + 9 * rand(),
      clumpM: 2 + 4 * rand(),
      seed: Math.floor(rand() * 2 ** 31),
    });
  }
  return rows;
}

/**
 * One step of something floating on the surface: carried by the current plus `leeway`
 * of the wind -- the drift model's step (pointDrift.stepPosition), nothing else.
 */
export function floaterStep(lat, lon, current, wind, leeway, dtS) {
  const [cu, cv] = current;
  const [wu, wv] = wind ?? [0, 0];
  return stepPosition(lat, lon, cu + leeway * wu, cv + leeway * wv, dtS);
}
