/**
 * pointDrift.js -- one point carried by the current and the wind (issue #64).
 *
 * The browser's copy of `sar.search.datum.drift_track`: Euler steps of 60 s with no
 * random term, the same arithmetic as `sar.model.position.calculate_position` at
 * sigma = 0, then one short step for a ragged remainder. It is used twice by the
 * search, and the difference is the doctrine (docs/ADR003.md):
 *
 *   the datum    the last known position, carried by current + the TARGET's leeway
 *                until the helicopter arrives;
 *   the marker   dropped at the datum, carried by the current alone.
 *
 * The forcing comes in through `sample(tMs, lat, lon) -> {current, wind}`, so this file
 * never touches a data store and is tested against Python with constant forcing.
 *
 * WHERE THE WATER RUNS OUT. A position with no current -- land in HYCOM, or off the
 * published grid -- stops the point there and says why, as D016 freezes a beached
 * particle. Drifting on the leeway alone would draw a path the model cannot vouch for.
 */

const TO_RAD = Math.PI / 180;

/** GRS80 mean radius, as `sar.model.position.EARTH_RADIUS_M` and geo.js. */
export const EARTH_RADIUS_M = 6371008.8;

/** Metres per degree on that sphere -- `calculate_position`'s own conversion. */
export const M_PER_DEG = (EARTH_RADIUS_M * Math.PI) / 180;

const TIME_TOLERANCE_S = 1e-9;

/** One step of `u, v` m/s for `dtS` seconds, flat earth, cos(lat) at the start. */
export function stepPosition(lat, lon, u, v, dtS) {
  return [
    lat + (v * dtS) / M_PER_DEG,
    lon + (u * dtS) / (M_PER_DEG * Math.cos(lat * TO_RAD)),
  ];
}

/**
 * A point drifted for `durationS` from `startMs`, as {tS, lat, lon, stoppedS, reason}.
 *
 * `leeway` is the fraction of the 10 m wind added to the current: 0.02 for a person in
 * water (D002), 0 for the marker and for a drogued drifter buoy.
 */
export function driftTrack({
  lat, lon, startMs, durationS, leeway, sample, stepS = 60,
}) {
  if (!Number.isFinite(durationS) || durationS <= 0) {
    throw new RangeError(`durationS must be a positive number of seconds, got ${durationS}`);
  }
  if (!(leeway >= 0)) throw new RangeError(`leeway must not be negative, got ${leeway}`);

  const tS = [0];
  const lats = [lat];
  const lons = [lon];
  let stoppedS = null;
  let reason = null;
  const whole = Math.floor(durationS / stepS + TIME_TOLERANCE_S);
  const steps = [];
  for (let k = 0; k < whole; k += 1) steps.push(stepS);
  const rest = durationS - whole * stepS;
  if (rest > TIME_TOLERANCE_S) steps.push(rest);

  let t = 0;
  let here = [lat, lon];
  for (const dt of steps) {
    if (stoppedS === null) {
      const s = sample(startMs + t * 1000, here[0], here[1]);
      if (!s || !s.current || !s.current.every(Number.isFinite)) {
        stoppedS = t;
        reason = (s && s.reason) || 'no surface current here';
      } else {
        const [wu, wv] = s.wind && s.wind.every(Number.isFinite) ? s.wind : [0, 0];
        here = stepPosition(here[0], here[1], s.current[0] + leeway * wu,
          s.current[1] + leeway * wv, dt);
      }
    }
    t += dt;
    tS.push(t);
    lats.push(here[0]);
    lons.push(here[1]);
  }
  return { tS, lat: lats, lon: lons, stoppedS, reason };
}

/** A sampler that answers the same current and wind everywhere, for tests and demos. */
export function constantSampler(current, wind = [0, 0]) {
  return () => ({ current, wind });
}

/**
 * A sampler over the site's `ResultantSource`: the nearest wind frame to the moment,
 * the current it pairs with, and the wind recovered from the leeway term.
 *
 * NEAREST FRAME, NOT BLENDED. The site's forcing is hourly wind and 3-hourly current,
 * and a search lasts a few hours, so this is a stated simplification of the time
 * interpolation the engine does. `frameOf` maps epoch ms to a frame of the tier in use.
 */
export function resultantSampler(source, frameOf) {
  return (tMs, lat, lon) => {
    const frame = frameOf(tMs);
    if (frame === null || !source.isResident(frame)) return { current: null, reason: 'forcing not loaded' };
    const s = source.sampleAt(frame, lat, lon);
    if (s.onLand) return { current: null, reason: 'reached land in the current model' };
    if (!s.current) return { current: null, reason: s.currentReason };
    const wind = source.alpha > 0 ? [s.leeway[0] / source.alpha, s.leeway[1] / source.alpha] : [0, 0];
    return { current: s.current, wind };
  };
}
