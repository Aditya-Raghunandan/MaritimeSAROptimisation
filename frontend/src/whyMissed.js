/**
 * whyMissed.js -- why the buoy did not go where the drift model said (issue #80).
 *
 * The prediction is the current plus a share of the wind. The buoy's own positions say
 * how it really moved. Sampled at the same places and times -- along the buoy's REAL
 * path, so a prediction that went astray early is not blamed twice -- the difference
 * between the two is exactly the motion the model is missing: the GAP.
 *
 * The gap's direction against the wind is the clue to which part of the model to blame:
 *
 *   small                 the model had the motion about right;
 *   along the wind,       the wind pushed harder than the model allows: typical of a buoy
 *   downwind              that has lost its drogue and sits high (the windage that would
 *                         have fitted is worked out);
 *   along the wind,       the model pushed it with the wind, but a drogued buoy barely
 *   upwind                feels the wind: "current only" fits it better;
 *   across the wind,      the water itself moved differently from the ocean model -- an
 *   or light wind         eddy smaller than a model cell, or one slightly out of place.
 *
 * It is a clue, not a proof, and it says what it cannot see: the waves' own push on
 * anything floating, tides, and the difference between the surface current and the
 * current 15 m down where a drogue sits. Pure: tested with made-up buoys and forcing.
 */

import { bearingTowards, compass, speed } from './drift.js';
import { M_PER_DEG } from './pointDrift.js';

const TO_RAD = Math.PI / 180;

/** Below this the gap is within the noise of hourly positions and nearest-frame forcing. */
export const CLOSE_MS = 0.03;

/** Below this wind, 2 % of it is under 0.04 m/s and cannot explain much of anything. */
export const WIND_MIN_MS = 2;

/** A gap whose along-wind part is at least this share of it runs along the wind (~45 deg). */
const ALONG_SHARE = 0.7;

/** The current model's cell, degrees: HYCOM GLBy0.08 is 0.08 lon x 0.04 lat (D019). */
const CELL_LON_DEG = 0.08;
const CELL_LAT_DEG = 0.04;

/** What no comparison of currents and winds can see. Said with every verdict. */
export const CANNOT_SEE = 'What this cannot see: the waves\' own push on anything floating, tides, '
  + 'and the difference between the surface current and the current 15 m down, where a drogue sits.';

/** Velocity [u, v] m/s from position a to b, [lat, lon] each, `dtS` seconds apart. */
export function velocityBetween(a, b, dtS) {
  const lat = (a[0] + b[0]) / 2;
  return [
    ((b[1] - a[1]) * M_PER_DEG * Math.cos(lat * TO_RAD)) / dtS,
    ((b[0] - a[0]) * M_PER_DEG) / dtS,
  ];
}

/**
 * The buoy's velocity at `ms`, from its positions `halfS` either side: centred where both
 * exist, one-sided at the ends of the record, null with neither.
 */
export function buoyVelocityAt(targetAt, ms, halfS = 1800) {
  const before = targetAt(ms - halfS * 1000);
  const after = targetAt(ms + halfS * 1000);
  const now = targetAt(ms);
  if (before && after) return velocityBetween(before, after, 2 * halfS);
  if (now && after) return velocityBetween(now, after, halfS);
  if (before && now) return velocityBetween(before, now, halfS);
  return null;
}

function add(a, b, k = 1) { return [a[0] + k * b[0], a[1] + k * b[1]]; }
function scale(a, k) { return [a[0] * k, a[1] * k]; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1]; }

function says(v) {
  return `${speed(v[0], v[1]).toFixed(2)} m/s towards ${compass(bearingTowards(v[0], v[1]))}`;
}

function pct(fraction) {
  const p = fraction * 100;
  return `${p < 10 ? p.toFixed(1) : p.toFixed(0)} %`;
}

/** The current model's cell at a latitude, as "8 x 4.5 km". */
export function cellSize(lat) {
  const ew = (CELL_LON_DEG * M_PER_DEG * Math.cos(lat * TO_RAD)) / 1000;
  const ns = (CELL_LAT_DEG * M_PER_DEG) / 1000;
  return `${ew.toFixed(ew < 10 ? 1 : 0).replace(/\.0$/, '')} × ${ns.toFixed(1).replace(/\.0$/, '')} km`;
}

/**
 * Why the buoy left the prediction between `fromMs` and `toMs`.
 *
 * @param {object} o
 * @param {(ms) => [number, number]|null} o.targetAt  the buoy's real position
 * @param {(ms, lat, lon) => {current, wind}} o.sample the forcing the prediction used
 * @param {number} o.fromMs, o.toMs                   the stretch to explain (report to arrival)
 * @param {number} o.leeway                           the share of wind the prediction used
 * @param {boolean|null} o.undrogued                  the buoy's drogue state, if known
 * @returns {object|null} null when the buoy or the forcing is missing over the stretch
 */
export function explainMiss({ targetAt, sample, fromMs, toMs, leeway, undrogued = null, stepS = 600 }) {
  const dtS = (toMs - fromMs) / 1000;
  if (!(dtS > 0)) return null;
  const start = targetAt(fromMs);
  const end = targetAt(toMs);
  if (!start || !end) return null;
  const actual = velocityBetween(start, end, dtS);

  // The forcing along where the buoy really was, averaged over the stretch.
  let n = 0;
  let current = [0, 0];
  let wind = [0, 0];
  for (let t = fromMs; t <= toMs; t += stepS * 1000) {
    const p = targetAt(t);
    const s = p ? sample(t, p[0], p[1]) : null;
    if (!s || !s.current || !s.current.every(Number.isFinite)) continue;
    current = add(current, s.current);
    wind = add(wind, s.wind && s.wind.every(Number.isFinite) ? s.wind : [0, 0]);
    n += 1;
  }
  if (n === 0) return null;
  current = scale(current, 1 / n);
  wind = scale(wind, 1 / n);

  const model = add(current, wind, leeway);
  const gap = add(actual, model, -1);
  const gapSpeed = speed(gap[0], gap[1]);
  const windSpeed = speed(wind[0], wind[1]);
  const down = windSpeed > 0 ? scale(wind, 1 / windSpeed) : [0, 0];
  const right = [down[1], -down[0]];
  const downwind = dot(gap, down);
  const crosswind = dot(gap, right);
  const fitLeeway = windSpeed > 0 ? leeway + downwind / windSpeed : null;
  const lat = (start[0] + end[0]) / 2;

  const base = {
    actual, model, current, wind, gap, gapSpeed, downwind, crosswind, fitLeeway, leeway, windSpeed,
  };
  const facts = `The buoy went ${says(actual)}; the model said ${says(model)}.`;
  // The same, short enough for the search panel: three labelled lines and a cause (#80).
  base.table = [['Buoy', says(actual)], ['Model', says(model)], ['Missing', says(gap)]];

  if (gapSpeed < CLOSE_MS) {
    return {
      ...base,
      verdict: 'close',
      headline: 'The model had its motion about right.',
      cause: 'They agree to within a few centimetres a second.',
      detail: `${facts} They differ by under ${CLOSE_MS} m/s.`,
      limits: CANNOT_SEE,
    };
  }

  const alongWind = windSpeed >= WIND_MIN_MS && Math.abs(downwind) >= ALONG_SHARE * gapSpeed;
  if (alongWind && downwind > 0) {
    const why = undrogued
      ? 'Without its drogue it sits high and catches more wind, so that is the likelier cause.'
      : 'It still had its drogue, so the buoy itself barely feels the wind; the waves\' push, '
        + 'which the model leaves out, is one explanation.';
    return {
      ...base,
      verdict: 'downwind',
      headline: 'The wind pushed it harder than the model allows.',
      cause: `Further downwind: ${pct(fitLeeway)} of the wind fits, the model used ${pct(leeway)}. `
        + (undrogued ? 'No drogue, so it catches more wind.' : 'It had its drogue; the push of the waves may explain it.'),
      detail: `${facts} It drifted ${downwind.toFixed(2)} m/s further downwind: about ${pct(fitLeeway)} `
        + `of the wind would have fitted, where the model used ${pct(leeway)}. ${why}`,
      limits: CANNOT_SEE,
    };
  }
  if (alongWind && downwind < 0 && leeway > 0 && undrogued === false) {
    return {
      ...base,
      verdict: 'upwind-drogued',
      headline: 'The model pushed it with the wind, but a drogued buoy barely feels the wind.',
      cause: 'Less downwind than modelled, and it had its drogue: "current only" fits it better.',
      detail: `${facts} It drifted ${(-downwind).toFixed(2)} m/s less downwind than predicted. `
        + 'It still had its drogue, the underwater sail that keeps it with the water, so '
        + '"current only" fits this buoy better.',
      limits: CANNOT_SEE,
    };
  }
  if (alongWind && downwind < 0 && fitLeeway !== null && fitLeeway >= 0) {
    return {
      ...base,
      verdict: 'upwind',
      headline: 'The wind pushed it less than the model allows.',
      cause: `Less downwind: ${pct(fitLeeway)} of the wind fits, the model used ${pct(leeway)}.`,
      detail: `${facts} It drifted ${(-downwind).toFixed(2)} m/s less downwind: about ${pct(fitLeeway)} `
        + `of the wind would have fitted, where the model used ${pct(leeway)}.`,
      limits: CANNOT_SEE,
    };
  }
  const notWind = windSpeed < WIND_MIN_MS
    ? `The wind was light (${windSpeed.toFixed(1)} m/s), so it cannot explain a gap that size`
    : 'That difference does not run along the wind';
  return {
    ...base,
    verdict: 'current',
    headline: 'The water itself moved differently from the ocean model.',
    cause: `${windSpeed < WIND_MIN_MS ? 'The wind was too light to matter' : 'Not along the wind'}, so likelier `
      + `the model's current: ${cellSize(lat)} cells, every 3 hours.`,
    detail: `${facts} The difference, ${says(gap)}, is what the model is missing. ${notWind}, so `
      + `the likelier culprit is the model's current: its cells here are about ${cellSize(lat)} and it `
      + 'steps every 3 hours, so an eddy smaller than that, or one slightly out of place, is not in it.',
    limits: CANNOT_SEE,
  };
}
