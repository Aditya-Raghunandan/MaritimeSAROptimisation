/**
 * searchRun.js -- one doctrinal search, planned and flown, with no map (issue #64).
 *
 * The sequence is the Coast Guard's own (docs/ADR003.md, USCG Addendum by page):
 *
 *   1. the call comes in: the target's last known position (LKP) at the report time;
 *   2. the helicopter is airborne within 30 minutes (B-0, p. PPO-7);
 *   3. it flies at 125 kt (Table 5-3) to the DATUM -- the LKP carried on by the current
 *      and the target's leeway for as long as all that took (p. 3-21);
 *   4. it drops a marker there, which drifts with the current alone (p. 3-21);
 *   5. it flies the pattern relative to that marker at 90 kt (p. 3-24) for the
 *      45-minute window, seeing everything within W/2 = 92.6 m of its track.
 *
 * The datum depends on the arrival time and the arrival time on the datum, so step 3
 * is iterated until the two agree to a second; it converges in two or three passes,
 * because the datum moves kilometres while the helicopter covers one in 16 s.
 *
 * DETECTION BY CLOSEST APPROACH. The target keeps moving while the helicopter flies:
 * at 1.8 m/s a target moves 108 m in a minute, more than W/2. So each short interval
 * of the flight is checked as two straight-line motions, the helicopter's and the
 * target's, and the target is found at the first moment the gap between them closes
 * to W/2 -- not by comparing two positions a step apart, which can miss a target the
 * helicopter flew straight over (ADR003 section 3).
 *
 * Only the search counts. The helicopter is not searching in transit, and doctrine
 * does not credit it with detections there.
 */

import { SWEEP_WIDTH_M } from './geo.js';
import {
  M_PER_DEG_LAT, PATTERNS, headingAt, markerAt, onGround,
} from './patterns.js';
import { driftTrack } from './pointDrift.js';
import {
  LAUNCH_DELAY_S, ON_SCENE_WINDOW_S, distanceM, transitTimeS,
} from './platform.js';

const TO_RAD = Math.PI / 180;

/** Leeway for the target types the panel offers. */
export const TARGETS = {
  drifter: { label: 'drogued drifter (current only)', leeway: 0 },
  person: { label: 'person in water (2 % of wind)', leeway: 0.02 },
};

/** Compass bearing, degrees true, of a velocity [u, v]; null for no motion. */
export function bearingOf(u, v) {
  if (!Number.isFinite(u) || !Number.isFinite(v) || (u === 0 && v === 0)) return null;
  const b = Math.atan2(u, v) / TO_RAD;
  return ((b % 360) + 360) % 360;
}

/** Initial bearing from a to b on a local flat earth, for the transit heading. */
function flatBearing(a, b) {
  const e = (b.lon - a.lon) * Math.cos(((a.lat + b.lat) / 2) * TO_RAD);
  return bearingOf(e, b.lat - a.lat) ?? 0;
}

/**
 * Plan a search. Returns the plan, or {error} saying why it cannot be flown.
 *
 * `sample(tMs, lat, lon) -> {current, wind}` is the forcing (pointDrift.js). `firstBearingDeg`
 * of null flies the first leg along the target's drift at the datum, as doctrine does
 * (p. 3-26); a number overrides it.
 */
export function planSearch({
  base, lkp, reportMs, patternKind = 'expanding_square', patternArgs = {},
  targetLeeway = 0, sample, firstBearingDeg = null, windowS = ON_SCENE_WINDOW_S,
}) {
  if (!PATTERNS[patternKind]) return { error: `unknown pattern ${patternKind}` };
  let elapsed = transitTimeS(distanceM(base, lkp));
  if (elapsed === null) return { error: 'the datum is beyond the helicopter\'s 300 NM radius of action' };

  let datumTrack = null;
  let datum = lkp;
  for (let pass = 0; pass < 6; pass += 1) {
    datumTrack = driftTrack({
      lat: lkp.lat, lon: lkp.lon, startMs: reportMs, durationS: elapsed, leeway: targetLeeway, sample,
    });
    const n = datumTrack.tS.length - 1;
    datum = { lat: datumTrack.lat[n], lon: datumTrack.lon[n] };
    const next = transitTimeS(distanceM(base, datum));
    if (next === null) return { error: 'the datum drifts beyond the helicopter\'s 300 NM radius of action' };
    const settled = Math.abs(next - elapsed) < 1;
    elapsed = next;
    if (settled) break;
  }

  const dropMs = reportMs + elapsed * 1000;
  const marker = driftTrack({
    lat: datum.lat, lon: datum.lon, startMs: dropMs, durationS: windowS, leeway: 0, sample,
  });

  let bearing = firstBearingDeg;
  let bearingSource = 'as set';
  if (bearing === null) {
    const s = sample(dropMs, datum.lat, datum.lon);
    const w = s && s.wind ? s.wind : [0, 0];
    bearing = s && s.current
      ? bearingOf(s.current[0] + targetLeeway * w[0], s.current[1] + targetLeeway * w[1])
      : null;
    bearingSource = bearing === null ? 'no drift at the datum; north' : 'along the drift (doctrine)';
    if (bearing === null) bearing = 0;
  }

  const pattern = PATTERNS[patternKind].build({ ...patternArgs, firstBearingDeg: bearing, durationS: windowS });
  return {
    base, lkp, reportMs, datum, datumTrack, marker, pattern, patternKind,
    targetLeeway, firstBearingDeg: bearing, bearingSource,
    launchS: LAUNCH_DELAY_S,
    arriveS: elapsed,
    dropMs,
    distanceM: distanceM(base, datum),
    endS: elapsed + pattern.durationS,
    notes: [datumTrack.reason, marker.reason].filter(Boolean),
  };
}

/** Where the helicopter is `s` seconds after the report, and what it is doing. */
export function helicopterAt(plan, s) {
  const { base, datum, launchS, arriveS, endS } = plan;
  if (s < launchS) return { lat: base.lat, lon: base.lon, heading: flatBearing(base, datum), phase: 'ready' };
  if (s < arriveS) {
    const f = (s - launchS) / (arriveS - launchS);
    return {
      lat: base.lat + f * (datum.lat - base.lat),
      lon: base.lon + f * (datum.lon - base.lon),
      heading: flatBearing(base, datum),
      phase: 'transit',
    };
  }
  const t = Math.min(s, endS) - arriveS;
  const p = onGround(plan.pattern, plan.marker, t);
  return { lat: p[0], lon: p[1], heading: headingAt(plan.pattern, t), phase: s <= endS ? 'search' : 'done' };
}

/** The marker's position `s` seconds after the report, or null before it is dropped. */
export function markerPositionAt(plan, s) {
  if (s < plan.arriveS) return null;
  const p = markerAt(plan.marker, Math.min(s, plan.endS) - plan.arriveS);
  return p ? { lat: p[0], lon: p[1] } : null;
}

/**
 * The search as a ground path: seconds after the report, lat, lon. Every waypoint of the
 * pattern, plus a point every `everyS`, because the marker keeps drifting between them.
 */
export function searchPath(plan, everyS = 10) {
  const times = new Set(plan.pattern.tS.map((t) => Math.round(t * 1e6) / 1e6));
  for (let t = 0; t < plan.pattern.durationS; t += everyS) times.add(t);
  const tS = [...times].sort((a, b) => a - b);
  const lat = [];
  const lon = [];
  for (const t of tS) {
    const p = onGround(plan.pattern, plan.marker, t);
    lat.push(p[0]);
    lon.push(p[1]);
  }
  return { tS: tS.map((t) => t + plan.arriveS), lat, lon };
}

/**
 * The first moment in [0, 1] at which two straight-line motions come within `radius`,
 * given their separation at the start (r0) and at the end (r1), in metres. Null if they
 * never do. Also returns the closest approach, which is what a miss reports.
 */
export function closestApproach(r0, r1, radius) {
  const d = [r1[0] - r0[0], r1[1] - r0[1]];
  const a = d[0] * d[0] + d[1] * d[1];
  const b = 2 * (r0[0] * d[0] + r0[1] * d[1]);
  const c = r0[0] * r0[0] + r0[1] * r0[1];
  const tau = a === 0 ? 0 : Math.min(1, Math.max(0, -b / (2 * a)));
  const closest = Math.hypot(r0[0] + tau * d[0], r0[1] + tau * d[1]);
  let contact = null;
  if (c <= radius * radius) contact = 0;
  else if (closest <= radius) {
    const disc = b * b - 4 * a * (c - radius * radius);
    contact = Math.max(0, (-b - Math.sqrt(Math.max(0, disc))) / (2 * a));
  }
  return { contact, closest, tau };
}

/** A target position relative to the helicopter, in local metres about the helicopter. */
function relative(heli, target) {
  const cos = Math.cos(heli[0] * TO_RAD);
  return [(target[1] - heli[1]) * M_PER_DEG_LAT * cos, (target[0] - heli[0]) * M_PER_DEG_LAT];
}

/**
 * Was the target found? `targetAt(ms) -> [lat, lon] | null` is where it really was.
 * Returns {found, foundS, closestM, closestS, gaps}, times in seconds after the report.
 */
export function detect(plan, targetAt, { halfWidthM = SWEEP_WIDTH_M / 2, everyS = 10 } = {}) {
  const path = searchPath(plan, everyS);
  let closestM = Infinity;
  let closestS = null;
  let gaps = 0;
  for (let k = 0; k + 1 < path.tS.length; k += 1) {
    const sa = path.tS[k];
    const sb = path.tS[k + 1];
    const ta = targetAt(plan.reportMs + sa * 1000);
    const tb = targetAt(plan.reportMs + sb * 1000);
    if (!ta || !tb) { gaps += 1; continue; }
    const ha = [path.lat[k], path.lon[k]];
    const hb = [path.lat[k + 1], path.lon[k + 1]];
    const hit = closestApproach(relative(ha, ta), relative(hb, tb), halfWidthM);
    if (hit.closest < closestM) {
      closestM = hit.closest;
      closestS = sa + hit.tau * (sb - sa);
    }
    if (hit.contact !== null) {
      return { found: true, foundS: sa + hit.contact * (sb - sa), closestM: 0, closestS, gaps };
    }
  }
  return { found: false, foundS: null, closestM, closestS, gaps };
}

/** How far the datum was from where the target really was when the helicopter arrived. */
export function datumErrorM(plan, targetAt) {
  const t = targetAt(plan.dropMs);
  return t ? distanceM(plan.datum, { lat: t[0], lon: t[1] }) : null;
}

/** "T+1:17:05": seconds after the report, for the read-out. */
export function formatElapsed(s) {
  const sign = s < 0 ? '-' : '+';
  const a = Math.abs(Math.round(s));
  const h = Math.floor(a / 3600);
  const m = Math.floor((a % 3600) / 60);
  const sec = a % 60;
  const mm = String(m).padStart(2, '0');
  const ss = String(sec).padStart(2, '0');
  return h > 0 ? `T${sign}${h}:${mm}:${ss}` : `T${sign}${m}:${ss}`;
}
