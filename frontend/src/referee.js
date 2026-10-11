/**
 * referee.js -- the search episode's referee, in the browser (D031, Stage 3).
 *
 * The paper scores every searcher with one referee, `sar.search.episode` (D029). The site
 * replays the paper's flights over the paper's clouds and scores a player's flight the
 * same way, so this is that referee again, line for line, held to it by a golden fixture
 * (scripts/export_referee_golden.py -> fixtures/referee_golden.json). Nothing here is a
 * simplification: if the two disagree, the browser is wrong.
 *
 * WHAT AN EPISODE IS. The scenario's cloud every 60 s through the 45-minute window, each
 * particle carrying its share of the probability. Each step the searcher flies a short path
 * (Waypoints: times inside the step, and positions in metres east and north of the datum
 * marker, which drifts with the current). Every particle that comes within W/2 = 92.6 m of
 * the helicopter loses (pod x) its weight. The weight taken in a step is the DRAIN RATE for
 * that minute (the referee's `removed_per_step`); POS is the drain rate summed over the
 * window. Nothing is renormalised.
 *
 * DETECTION BY CLOSEST APPROACH. A step's path is cut into sub-legs at its waypoints. Over
 * a sub-leg both the helicopter and each particle move in straight lines (the particle
 * interpolated between the two frames), so the closest they come is the distance from the
 * origin to the segment r0 -> r1 of their separation.
 *
 * IT CANNOT TURN ON THE SPOT (D032). The helicopter turns at most `turnRateDegS`, 7.0 deg/s
 * by default (kinematics.js), arriving on `headingDeg`. Every waypoint carries the heading
 * there, and `fly` refuses a path that turns faster or changes speed. A rate of Infinity is
 * the referee before D032, which turned at once: format 1 bundles were flown by it.
 *
 * THE SAME ARITHMETIC AS PYTHON. Positions are latitude and longitude in the store
 * convention (0 to 360), as the engine keeps them; distances are metres on a local flat
 * earth with cos(lat) at the reference point (`relativeM`, Python's `relative_m`), and a
 * position plus an offset uses cos(lat) at the start (`offsetPosition`). Longitude
 * differences wrap with a floor modulo, as Python's % does.
 */

import { M_PER_DEG_LAT, SWEEP_WIDTH_M } from './geo.js';
import { bearing, checkTurns, isInstant, normalise, shortestTurn, steer } from './kinematics.js';
import { ON_SCENE_WINDOW_S, SEARCH_SPEED_MS, STEP_S, TURN_RATE_DEG_S } from './platform.js';

export { STEP_S };
export const STEPS = Math.round(ON_SCENE_WINDOW_S / STEP_S);
export const HALF_WIDTH_M = SWEEP_WIDTH_M / 2;

const TO_RAD = Math.PI / 180;
const TIME_TOLERANCE_S = 1e-6;

/** Python's floor modulo: the sign of the result follows the divisor. */
function floorMod(x, m) {
  return ((x % m) + m) % m;
}

function mPerDegLon(lat) {
  return M_PER_DEG_LAT * Math.cos(lat * TO_RAD);
}

/** (lat, lon) plus (east, north) metres, cos(lat) at the start: Python's offset_position. */
export function offsetPosition(lat, lon, eastM, northM) {
  return [lat + northM / M_PER_DEG_LAT, lon + eastM / mPerDegLon(lat)];
}

/** (east, north) metres from (refLat, refLon) to (lat, lon), cos at the reference. */
export function relativeM(lat, lon, refLat, refLon) {
  const dlon = floorMod(lon - refLon + 180, 360) - 180;
  return [dlon * mPerDegLon(refLat), (lat - refLat) * M_PER_DEG_LAT];
}

/** The least distance from the origin to the segment r0 -> r1: Python's closest_approach_m. */
export function closestApproachM(r0e, r0n, r1e, r1n) {
  const de = r1e - r0e;
  const dn = r1n - r0n;
  const dd = de * de + dn * dn;
  let tau = dd > 0 ? -(r0e * de + r0n * dn) / dd : 0;
  tau = Math.min(1, Math.max(0, tau));
  return Math.hypot(r0e + tau * de, r0n + tau * dn);
}

/** When two straight motions first come within radius: Python's first_contact. */
export function firstContact(r0e, r0n, r1e, r1n, radius) {
  const de = r1e - r0e;
  const dn = r1n - r0n;
  const a = de * de + dn * dn;
  const b = 2 * (r0e * de + r0n * dn);
  const c = r0e * r0e + r0n * r0n;
  const tau = a === 0 ? 0 : Math.min(1, Math.max(0, -b / (2 * a)));
  const closest = Math.hypot(r0e + tau * de, r0n + tau * dn);
  let contact = null;
  if (c <= radius * radius) contact = 0;
  else if (closest <= radius) {
    const disc = Math.max(0, b * b - 4 * a * (c - radius * radius));
    contact = Math.max(0, (-b - Math.sqrt(disc)) / (2 * a));
  }
  return { contact, closest, tau };
}

/** numpy's interp on a sorted axis, clamped at the ends. */
export function interp(x, xp, fp) {
  const n = xp.length;
  if (x <= xp[0]) return fp[0];
  if (x >= xp[n - 1]) return fp[n - 1];
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xp[mid] <= x) lo = mid; else hi = mid;
  }
  const slope = (fp[lo + 1] - fp[lo]) / (xp[lo + 1] - xp[lo]);
  return slope * (x - xp[lo]) + fp[lo];
}

/** A track {tS, lat, lon} at t seconds, refusing a time outside it, as MarkerTrack.at does. */
export function trackAt(track, t) {
  const { tS } = track;
  if (t < tS[0] - TIME_TOLERANCE_S || t > tS[tS.length - 1] + TIME_TOLERANCE_S) {
    throw new RangeError(`t = ${t} s lies outside the track, ${tS[0]} to ${tS[tS.length - 1]} s`);
  }
  return [interp(t, tS, track.lat), interp(t, tS, track.lon)];
}

/**
 * A scenario bundle's cloud (float32 metres from the marker at each frame) as positions:
 * {lat, lon}, Float64Arrays of frames x particles, frame-major. The exact inverse of how
 * scripts/export_scenario_bundles.py wrote it.
 */
export function cloudFromOffsets(f32, frames, particles, marker) {
  const lat = new Float64Array(frames * particles);
  const lon = new Float64Array(frames * particles);
  for (let k = 0; k < frames; k += 1) {
    const [mlat, mlon] = trackAt(marker, k * STEP_S);
    const scale = mPerDegLon(mlat);
    for (let j = 0; j < particles; j += 1) {
      const i = k * particles + j;
      lat[i] = mlat + f32[2 * i + 1] / M_PER_DEG_LAT;
      lon[i] = mlon + f32[2 * i] / scale;
    }
  }
  return { lat, lon };
}

/**
 * One search over one cloud. `cloud` is {frames, particles, lat, lon, weight} with lat and
 * lon frame-major Float64Arrays; `marker` and `target` are {tS, lat, lon} tracks in seconds
 * since arrival. `target` may be null.
 */
export class Episode {
  constructor(cloud, marker, {
    target = null, speedMs = SEARCH_SPEED_MS, halfWidthM = HALF_WIDTH_M, pod = 1, steps = STEPS,
    headingDeg = 0, turnRateDegS = TURN_RATE_DEG_S,
  } = {}) {
    if (!(turnRateDegS > 0)) throw new RangeError(`the turn rate must be positive, got ${turnRateDegS}`);
    if (!Number.isFinite(headingDeg)) throw new RangeError(`the arrival heading must be finite, got ${headingDeg}`);
    if (cloud.frames !== steps + 1) {
      throw new RangeError(`an episode of ${steps} steps needs ${steps + 1} frames, got ${cloud.frames}`);
    }
    this.cloud = cloud;
    this.n = cloud.particles;
    this.marker = marker;
    this.target = target;
    this.speedMs = speedMs;
    this.halfWidthM = halfWidthM;
    this.pod = pod;
    this.turnRateDegS = turnRateDegS;
    this.headingDeg = normalise(headingDeg);
    this.steps = steps;
    this.k = 0;
    this.weight = Float64Array.from(cloud.weight);
    this.initial = this.weight.reduce((s, w) => s + w, 0);
    this.offset = [0, 0];
    this.removed = [];
    this.legs = [];
    this.track = [[0, ...this.position]];
    this.distance = 0;
    this.foundS = null;
    this.closest = [Infinity, null];
    this.targetGaps = 0;
  }

  get tS() { return this.k * STEP_S; }

  get done() { return this.k >= this.steps; }

  get markerPosition() { return trackAt(this.marker, this.tS); }

  get position() {
    const [mlat, mlon] = this.markerPosition;
    return offsetPosition(mlat, mlon, this.offset[0], this.offset[1]);
  }

  get remaining() { return this.weight.reduce((s, w) => s + w, 0); }

  /** True for the referee before D032, whose helicopter turns at once. */
  get instant() { return isInstant(this.turnRateDegS); }

  /**
   * The path one step flies for a command: a heading, or {turnDeg} (right positive). The
   * turn at the turn rate, then straight; with an instant turn, one straight leg as before.
   */
  waypointsFor(command) {
    const turn = typeof command === 'object' && command !== null ? command.turnDeg : null;
    const heading = turn === null ? command : null;
    if (!Number.isFinite(turn ?? heading)) throw new RangeError(`a heading or turn must be finite degrees, got ${command}`);
    if (this.instant && heading !== null) {
      const d = this.speedMs * STEP_S;
      const b = heading * TO_RAD;
      return {
        tS: [STEP_S], eastM: [this.offset[0] + d * Math.sin(b)], northM: [this.offset[1] + d * Math.cos(b)],
        headingDeg: [normalise(heading)],
      };
    }
    const by = turn ?? shortestTurn(this.headingDeg, heading);
    return steer(this.offset[0], this.offset[1], this.headingDeg, by, STEP_S, this.speedMs, this.turnRateDegS);
  }

  /** Turn to a heading, relative to the marker, and fly on along it, for one step. */
  step(headingDeg) {
    return this.fly(this.waypointsFor(headingDeg));
  }

  /** Turn by `turnDeg`, right positive, then fly straight, for one step (D032). */
  turn(turnDeg) {
    return this.fly(this.waypointsFor({ turnDeg }));
  }

  /** Fly one step through waypoints {tS, eastM, northM}; returns the drain rate of the step. */
  fly(waypoints) {
    if (this.done) throw new RangeError(`the episode is over: all ${this.steps} steps have been flown`);
    const wt = waypoints.tS;
    if (Math.abs(wt[wt.length - 1] - STEP_S) > TIME_TOLERANCE_S) {
      throw new RangeError(`waypoints must end at the end of the step, ${STEP_S} s`);
    }
    const t = [0, ...wt.slice(0, -1), STEP_S];
    const east = [this.offset[0], ...waypoints.eastM];
    const north = [this.offset[1], ...waypoints.northM];
    let flown = 0;
    for (let i = 0; i + 1 < t.length; i += 1) {
      const leg = Math.hypot(east[i + 1] - east[i], north[i + 1] - north[i]);
      const allowed = this.speedMs * (t[i + 1] - t[i]);
      if (leg > allowed * (1 + 1e-9) + 1e-6) {
        throw new RangeError(`the path flies faster than the helicopter's ${this.speedMs} m/s`);
      }
      flown += leg;
    }
    if (!this.instant) {
      if (!waypoints.headingDeg) throw new RangeError('a referee that limits the turn rate needs the heading at every waypoint (D032)');
      checkTurns(wt, waypoints.eastM, waypoints.northM, waypoints.headingDeg,
        [this.offset[0], this.offset[1], this.headingDeg], this.speedMs, this.turnRateDegS);
    }

    const t0 = this.tS;
    const ground = t.map((ti, i) => {
      const [mlat, mlon] = trackAt(this.marker, t0 + ti);
      return offsetPosition(mlat, mlon, east[i], north[i]);
    });
    const { lat, lon } = this.cloud;
    const n = this.n;
    const base = this.k * n;
    const next = base + n;
    const aLat = new Float64Array(n);
    const aLon = new Float64Array(n);
    for (let j = 0; j < n; j += 1) {
      aLat[j] = lat[base + j];
      aLon[j] = lon[base + j];
    }

    let removedStep = 0;
    for (let i = 0; i + 1 < t.length; i += 1) {
      const f = t[i + 1] / STEP_S;
      const [hLat0, hLon0] = ground[i];
      const [hLat1, hLon1] = ground[i + 1];
      let removed = 0;
      for (let j = 0; j < n; j += 1) {
        const lat0 = lat[base + j];
        const lon0 = lon[base + j];
        const bLat = lat0 + f * (lat[next + j] - lat0);
        const bLon = lon0 + f * (floorMod(lon[next + j] - lon0 + 180, 360) - 180);
        const w = this.weight[j];
        if (w > 0) {
          const [r0e, r0n] = relativeM(hLat0, hLon0, aLat[j], aLon[j]);
          const [r1e, r1n] = relativeM(hLat1, hLon1, bLat, bLon);
          if (closestApproachM(r0e, r0n, r1e, r1n) <= this.halfWidthM) {
            const after = w * (1 - this.pod);
            removed += w - after;
            this.weight[j] = after;
          }
        }
        aLat[j] = bLat;
        aLon[j] = bLon;
      }
      this.legs.push([t0 + t[i + 1], removed]);
      removedStep += removed;
      if (this.target) this.scoreTarget(t0 + t[i], t0 + t[i + 1], ground[i], ground[i + 1]);
    }

    // A turning referee has just checked the path is flown at full speed; the chords of its
    // arcs are a little shorter than the arcs.
    this.distance += this.instant ? flown : this.speedMs * STEP_S;
    const last = east.length - 1;
    this.offset = [east[last], north[last]];
    if (waypoints.headingDeg) this.headingDeg = normalise(waypoints.headingDeg[waypoints.headingDeg.length - 1]);
    else if (Math.hypot(east[last] - east[last - 1], north[last] - north[last - 1]) > 0) {
      this.headingDeg = bearing(east[last] - east[last - 1], north[last] - north[last - 1]);
    }
    for (let i = 1; i < t.length; i += 1) this.track.push([t0 + t[i], ...ground[i]]);
    this.removed.push(removedStep);
    this.k += 1;
    return removedStep;
  }

  /** The closest-approach test against the one real target, for one sub-leg. */
  scoreTarget(ta, tb, ha, hb) {
    const { tS } = this.target;
    const lo = tS[0];
    const hi = tS[tS.length - 1];
    if (ta < lo - TIME_TOLERANCE_S || tb > hi + TIME_TOLERANCE_S) {
      this.targetGaps += 1;
      return;
    }
    const [tLatA, tLonA] = trackAt(this.target, Math.min(hi, Math.max(lo, ta)));
    const [tLatB, tLonB] = trackAt(this.target, Math.min(hi, Math.max(lo, tb)));
    const [r0e, r0n] = relativeM(ha[0], ha[1], tLatA, tLonA);
    const [r1e, r1n] = relativeM(hb[0], hb[1], tLatB, tLonB);
    const { contact, closest, tau } = firstContact(r0e, r0n, r1e, r1n, this.halfWidthM);
    if (closest < this.closest[0]) this.closest = [closest, ta + tau * (tb - ta)];
    if (contact !== null && this.foundS === null) this.foundS = ta + contact * (tb - ta);
  }

  /** Each particle's position at t seconds since arrival, between frames: for drawing. */
  particlesAt(t) {
    const k = Math.min(this.steps - 1, Math.max(0, Math.floor(t / STEP_S)));
    const f = Math.min(1, Math.max(0, t / STEP_S - k));
    const { lat, lon } = this.cloud;
    const n = this.n;
    const outLat = new Float64Array(n);
    const outLon = new Float64Array(n);
    for (let j = 0; j < n; j += 1) {
      const a = k * n + j;
      outLat[j] = lat[a] + f * (lat[a + n] - lat[a]);
      outLon[j] = lon[a] + f * (floorMod(lon[a + n] - lon[a] + 180, 360) - 180);
    }
    return { lat: outLat, lon: outLon };
  }

  /** POS, the drain rate per minute, time to detection, distance; and the target, if any. */
  metrics() {
    const pos = this.removed.reduce((s, m) => s + m, 0);
    const found = this.legs.reduce((s, [, m]) => s + m, 0);
    const ttd = found > 0 ? this.legs.reduce((s, [t, m]) => s + t * m, 0) / found : null;
    const out = {
      steps: this.k,
      elapsedS: this.tS,
      pos,
      initialMass: this.initial,
      remaining: this.remaining,
      expectedTtdS: ttd,
      distanceM: this.distance,
      drainRate: [...this.removed],
      turnRateDegS: this.instant ? null : this.turnRateDegS,
    };
    if (this.target) {
      const [closest, when] = this.closest;
      out.target = {
        found: this.foundS !== null,
        foundS: this.foundS,
        closestM: Number.isFinite(closest) ? closest : null,
        closestS: when,
        gaps: this.targetGaps,
      };
    }
    return out;
  }
}
