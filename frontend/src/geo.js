/**
 * geo.js -- distance, bearing and the project's scale constants.
 *
 * Pure arithmetic, importing nothing. It was originally inside `measure.js`,
 * which imports Leaflet, and Leaflet needs a `window` -- so the geodesy could
 * not be tested without a DOM. That was the wrong dependency anyway: a bearing
 * has no business needing a map library, and the KML export (R8c) will want
 * these same functions with no map anywhere near it.
 *
 * WHY GEODESIC AND NOT PIXELS. Leaflet draws in Web Mercator, which stretches
 * east-west with latitude. Across this project's 17-36 N box the same screen
 * distance is about 18 % fewer kilometres at the top than the bottom
 * (sec 36 / sec 17 = 1.18), so anything measured on screen is wrong by up to a
 * fifth depending where you measure it.
 */

const R_EARTH_M = 6371008.8; // IUGG mean radius
const TO_RAD = Math.PI / 180;

/** Gulf Stream surface average -- what the project sizes drift against. */
export const TYPICAL_CURRENT_MS = 1.8;

/**
 * Sweep width for a person in the water, seen from a helicopter: 0.1 NM.
 *
 * USCG Addendum COMDTINST M16130.2F (2013), App. H Tables H-15/H-16, p. H-44, at
 * 300-1000 ft, visibility 3 NM or more, winds up to 15 kt, 90 kt, no lifejacket.
 * `sar.search.platform.SWEEP_WIDTH_M` is the reference, and `tests/platform.test.js`
 * holds this to it through `fixtures/search_golden.json` (issue #44).
 */
export const SWEEP_WIDTH_M = 185.2;

export const RANGE_RINGS_KM = [10, 25, 50, 100];

/**
 * Wrap into [0, 360) -- closed at the bottom, OPEN at the top.
 *
 * `x % 360` is not enough. For a tiny negative x, which is exactly what
 * `atan2` gives for due north, the true answer is just under 360 and floating
 * point rounds it to 360 itself. A bearing of 360 then breaks any consumer
 * binning into 36 sectors as `Math.floor(d / 10)`, which yields index 36 in a
 * length-36 array. The Python side hit the identical bug in `_bearing`.
 */
export function wrapBearing(degrees) {
  const wrapped = ((degrees % 360) + 360) % 360;
  return wrapped >= 360 ? 0 : wrapped;
}

/** Initial great-circle bearing from a to b, degrees true. */
export function bearing(a, b) {
  const φ1 = a.lat * TO_RAD;
  const φ2 = b.lat * TO_RAD;
  const Δλ = (b.lng - a.lng) * TO_RAD;
  const y = Math.sin(Δλ) * Math.cos(φ2);
  const x = Math.cos(φ1) * Math.sin(φ2) - Math.sin(φ1) * Math.cos(φ2) * Math.cos(Δλ);
  return wrapBearing((Math.atan2(y, x) * 180) / Math.PI);
}

/**
 * Great-circle distance in metres, by the haversine formula.
 *
 * Leaflet's `map.distance()` gives the same answer and is what the live map
 * uses; this exists so the figure is computable without a map instance, and so
 * the two can be checked against each other.
 */
export function distance(a, b) {
  const φ1 = a.lat * TO_RAD;
  const φ2 = b.lat * TO_RAD;
  const Δφ = (b.lat - a.lat) * TO_RAD;
  const Δλ = (b.lng - a.lng) * TO_RAD;
  const h = Math.sin(Δφ / 2) ** 2 + Math.cos(φ1) * Math.cos(φ2) * Math.sin(Δλ / 2) ** 2;
  return 2 * R_EARTH_M * Math.asin(Math.min(1, Math.sqrt(h)));
}

/**
 * Metres for sub-kilometre distances, because sweep width is ~185 m and
 * rendering that as "0.19 km" hides the number that sets the map's cell size.
 */
export function formatDistance(metres) {
  if (metres < 1000) return `${metres.toFixed(0)} m`;
  if (metres < 100000) return `${(metres / 1000).toFixed(2)} km`;
  return `${(metres / 1000).toFixed(0)} km`;
}

/** How long a distance takes at the Gulf Stream's typical speed, in hours. */
export function driftHours(metres, speedMs = TYPICAL_CURRENT_MS) {
  return metres / speedMs / 3600;
}
