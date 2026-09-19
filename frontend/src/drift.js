/**
 * drift.js -- what the wind at a point does to something floating in it.
 *
 * Pure arithmetic, no imports, so it is testable without a DOM -- the same
 * reason geo.js and style.js are separate from the map code.
 *
 * THIS IS THE LEEWAY TERM ONLY, AND THAT IS THE WHOLE POINT OF SAYING SO.
 * D002 freezes the drift model at three terms:
 *
 *     v_d = v_c(x,t) + alpha * v_w10(x,t) + eta(t)
 *            current      leeway            stochastic
 *
 * The map can currently answer for exactly one of them. The current field is
 * not fetched yet and the stochastic term belongs to the engine, which does
 * not exist. So this module computes the leeway and the panel labels the other
 * two as pending rather than quietly presenting a third of the physics as the
 * answer. A number with a missing term is worse than no number, because it
 * looks like an answer.
 */

/**
 * Leeway coefficient, the fraction of wind speed a floating object makes
 * downwind.
 *
 * 0.02, matching `ALPHA_MID` in src/sar/fetch/wind.py so the browser and the
 * engine cannot drift apart. Cited to Allen (2000), "The Leeway of
 * Persons-In-Water and Three Small Craft", USCG R&D Center, DTIC ADA376479:
 * a downwind slope of 1.93 % for a PIW, 2.7 % for a PIW in a survival suit.
 * R1c's 1-4 % range is right; the 3 % midpoint an earlier version of this
 * project used was not.
 */
export const ALPHA = 0.02;

/** Wind speed from components, m/s. */
export function speed(u, v) {
  return Math.hypot(u, v);
}

/**
 * Direction the wind is blowing FROM, degrees clockwise from north.
 *
 * Meteorological convention, matching the `dir_from_deg` column
 * `sar.fetch.wind` writes. "A westerly" means from the west, and getting this
 * backwards inverts every statement about where something drifts.
 */
export function bearingFrom(u, v) {
  return (270 - (Math.atan2(v, u) * 180) / Math.PI + 360) % 360;
}

/** Direction the wind is blowing TOWARDS -- which is the way a drifter goes. */
export function bearingTowards(u, v) {
  return (bearingFrom(u, v) + 180) % 360;
}

/** The 16-point compass name for a bearing, for reading aloud. */
export function compass(deg) {
  const points = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
  return points[Math.round(((deg % 360) + 360) % 360 / 22.5) % 16];
}

/** Downwind speed imparted by the wind, m/s. The alpha term, and only that. */
export function leewaySpeed(windSpeed) {
  return ALPHA * windSpeed;
}

/** How far that leeway carries something in `hours`, in metres. */
export function leewayDistance(windSpeed, hours) {
  return leewaySpeed(windSpeed) * hours * 3600;
}

/**
 * Leeway as a percentage of a typical Gulf Stream current.
 *
 * This is the number that decides whether the leeway term earns its place in
 * the model at all (D002, R1). Against 1.8 m/s, a 10 m/s wind contributes
 * 0.2 m/s -- about 11 %, not negligible; a 3 m/s wind contributes 3 %, which
 * is. The regime where it stops being negligible is the one that justifies the
 * three-term model empirically.
 */
export function leewayFractionOfCurrent(windSpeed, currentSpeed) {
  if (!(currentSpeed > 0)) return null;
  return leewaySpeed(windSpeed) / currentSpeed;
}

/** Mean, min and max of a series, ignoring the NaNs that mark unloaded frames. */
export function summarise(series) {
  let n = 0;
  let sum = 0;
  let min = Infinity;
  let max = -Infinity;
  for (const x of series) {
    if (!Number.isFinite(x)) continue;
    n += 1;
    sum += x;
    if (x < min) min = x;
    if (x > max) max = x;
  }
  if (n === 0) return { n: 0, mean: null, min: null, max: null };
  return { n, mean: sum / n, min, max };
}

/**
 * One sentence saying what the wind at this cell would actually do.
 *
 * The panel was a column of correct numbers that meant nothing unless you
 * already knew the model. This turns them into the claim they support -- and,
 * importantly, into the claim they DO NOT support: the leeway term is one of
 * three, so the sentence says whose contribution it is describing every time,
 * rather than relying on a footnote nobody reads.
 *
 * @returns {{lead: string, caveat: string}}
 */
export function explain(windSpeed, towardsDeg, currentSpeed, sweepWidthM) {
  const km = leewayDistance(windSpeed, 24) / 1000;
  // Two decimals, ALWAYS. This used to interpolate currentSpeed raw, which was
  // fine while it was the constant 1.8 and produced
  // "38 % of a 0.376257897392351 m/s current" the moment a measured value was
  // passed in. A number with fifteen decimal places in a sentence reads as a
  // bug even when the value is right, which on a projector is the only thing
  // anyone will take away from the line.
  const cs = Number.isFinite(currentSpeed) ? currentSpeed.toFixed(2) : '—';
  const frac = leewayFractionOfCurrent(windSpeed, currentSpeed);
  const pct = frac === null ? null : frac * 100;
  const dir = compass(towardsDeg);

  const lead = `Wind alone would carry a drifter about `
    + `${km < 1 ? `${Math.round(km * 1000)} m` : `${km.toFixed(1)} km`} ${dir} in a day.`;

  let caveat;
  if (pct === null) {
    caveat = 'No current figure to compare it against.';
  } else if (pct < 5) {
    caveat = `That is only ${pct.toFixed(0)} % of what a 
      ${cs} m/s current does here, so the current would dominate `
      + 'and the search datum follows the water, not the weather.';
  } else if (pct < 15) {
    caveat = `That is ${pct.toFixed(0)} % of a ${cs} m/s current -- `
      + 'enough to matter, which is the regime that earns the leeway term its '
      + 'place in the model.';
  } else {
    caveat = `That is ${pct.toFixed(0)} % of a ${cs} m/s current, so `
      + 'wind and water are comparable here and the two terms have to be added '
      + 'as vectors, not ranked.';
  }

  // Tie the displacement to the thing a searcher actually flies.
  if (sweepWidthM > 0) {
    const lanes = (km * 1000) / sweepWidthM;
    caveat += ` For scale, that displacement is about ${Math.round(lanes)} `
      + `sweep widths at ${sweepWidthM} m.`;
  }

  return { lead, caveat: caveat.replace(/\s+/g, ' ').trim() };
}

/**
 * Speed bands for a surface current, where Beaufort sits for wind.
 *
 * Lives here rather than in legend.js because legend.js imports Leaflet, and
 * the point panel needs this too -- routing it through the legend would drag a
 * map library into a module that only does arithmetic, which is the coupling
 * this project splits files to avoid. drift.js imports nothing.
 *
 * Beaufort is a WIND scale and means nothing for water. These are the bands a
 * drift argument is actually made in: whether the current dominates the leeway
 * term, and whether you are in the jet.
 */
export function currentBand(speedMs) {
  if (!Number.isFinite(speedMs)) return '—';
  if (speedMs < 0.25) return 'weak · leeway competes';
  if (speedMs < 0.5) return 'moderate';
  if (speedMs < 1.0) return 'strong';
  if (speedMs < 1.6) return 'swift · jet edge';
  return 'Gulf Stream core';
}
