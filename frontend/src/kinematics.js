/**
 * kinematics.js -- how the helicopter moves: it cannot turn on the spot (D032, ADR006).
 *
 * The browser's copy of `sar.search.kinematics`, line for line, held to it by
 * `fixtures/search_golden.json` (steer and flown cases) and `fixtures/referee_golden.json`
 * (every searcher, and a player). If the two disagree, the browser is wrong.
 *
 * THE MODEL: A DUBINS VEHICLE. Constant speed about the datum marker; the heading changes
 * at most TURN_RATE_DEG_S, 7.0 deg/s (a 30 degree bank at 90 kt), so the tightest path is an
 * arc of radius 379 m. An infinite rate is the old helicopter that turned at once.
 *
 *   steer      a command: turn by so many degrees (right positive), or to a heading the
 *              short way round, at the turn rate, then straight
 *   follow     a drawn path (a pattern's legs) flown by the L1 autopilot (Park, Deyst and
 *              How 2004): aim at where the path leaves a circle of radius L1 = r about the
 *              helicopter, turning at 2 V sin(eta) / L1, never faster than the rate
 *   checkTurns the referee's check that a path could have been flown
 */

export const ARC_PIECE_DEG = 10;
export const AUTOPILOT_DT_S = 0.5;
export const MAX_CHORD_TURN_DEG = 30;
const STRAIGHT_RAD_S = 1e-12;
const REL_TOL = 1e-9;
const ABS_TOL_DEG = 1e-6;
const ABS_TOL_M = 1e-6;
const TO_RAD = Math.PI / 180;
const TO_DEG = 180 / Math.PI;

/** Python's float %: the result takes the divisor's sign. */
function floorMod(x, m) {
  const r = x % m;
  return r !== 0 && (r < 0) !== (m < 0) ? r + m : r;
}

export function isInstant(turnRateDegS) {
  return turnRateDegS === Infinity;
}

/** A heading in [0, 360). */
export function normalise(headingDeg) {
  const h = floorMod(headingDeg, 360);
  return h === 360 ? 0 : h;
}

/** The turn from one heading to another the short way, in (-180, 180]: right positive. */
export function shortestTurn(fromDeg, toDeg) {
  const d = floorMod(toDeg - fromDeg, 360);
  return d > 180 ? d - 360 : d;
}

/** The compass bearing of an (east, north) step, degrees in [0, 360). */
export function bearing(de, dn) {
  return normalise(Math.atan2(de, dn) * TO_DEG);
}

/** Where a helicopter turning at a constant rate is after t seconds: [east, north, heading]. */
export function arcPoint(east, north, headingDeg, rateDegS, t, speedMs) {
  const psi0 = headingDeg * TO_RAD;
  const half = (rateDegS * TO_RAD * t) / 2;
  const length = speedMs * t * (half !== 0 ? Math.sin(half) / half : 1);
  const along = psi0 + half;
  return [east + length * Math.sin(along), north + length * Math.cos(along),
    normalise(headingDeg + rateDegS * t)];
}

/**
 * Turn by `turnDeg` (right positive) at the turn rate, then fly straight, for `durationS`.
 * Returns the chord ends after the start: {tS, eastM, northM, headingDeg}.
 */
export function steer(east, north, headingDeg, turnDeg, durationS, speedMs, turnRateDegS) {
  if (!(durationS > 0)) throw new RangeError(`a steer needs a positive duration, got ${durationS}`);
  if (!Number.isFinite(turnDeg)) throw new RangeError(`a turn must be finite degrees, got ${turnDeg}`);
  if (isInstant(turnRateDegS)) {
    const h = normalise(headingDeg + turnDeg);
    const d = speedMs * durationS;
    const b = h * TO_RAD;
    return { tS: [durationS], eastM: [east + d * Math.sin(b)], northM: [north + d * Math.cos(b)], headingDeg: [h] };
  }
  const omega = turnRateDegS;
  const sign = turnDeg > 0 ? 1 : -1;
  const full = Math.abs(turnDeg) / omega;
  const tau = Math.min(full, durationS);
  const out = { tS: [], eastM: [], northM: [], headingDeg: [] };
  if (tau > 0) {
    const turned = full <= durationS ? Math.abs(turnDeg) : omega * durationS;
    const pieces = Math.max(1, Math.ceil(turned / ARC_PIECE_DEG - 1e-9));
    for (let k = 1; k <= pieces; k += 1) {
      const tk = (tau * k) / pieces;
      const [e, n, h] = arcPoint(east, north, headingDeg, sign * omega, tk, speedMs);
      out.tS.push(tk);
      out.eastM.push(e);
      out.northM.push(n);
      out.headingDeg.push(h);
    }
    if (full <= durationS) out.headingDeg[out.headingDeg.length - 1] = normalise(headingDeg + turnDeg);
  }
  if (tau < durationS) {
    const last = out.tS.length - 1;
    const [e0, n0, h0] = last >= 0
      ? [out.eastM[last], out.northM[last], out.headingDeg[last]]
      : [east, north, normalise(headingDeg)];
    const [e, n, h] = arcPoint(e0, n0, h0, 0, durationS - tau, speedMs);
    out.tS.push(durationS);
    out.eastM.push(e);
    out.northM.push(n);
    out.headingDeg.push(h);
  }
  return out;
}

/** Turn to a heading the short way round, then straight: a heading command. */
export function steerTo(east, north, headingDeg, toHeadingDeg, durationS, speedMs, turnRateDegS) {
  return steer(east, north, headingDeg, shortestTurn(headingDeg, toHeadingDeg), durationS, speedMs,
    turnRateDegS);
}

/* ------------------------------------------------------------------------- the autopilot */

function sign(rateDegS) {
  if (Math.abs(rateDegS * TO_RAD) <= STRAIGHT_RAD_S) return 0;
  return rateDegS > 0 ? 1 : -1;
}

/** Which ticks after `a` up to `b` to keep, so each chord turns one way, at most a piece. */
export function chordEnds(rateDegS, a, b, dtS) {
  const keep = [];
  let s0 = 0;
  let turned = 0;
  for (let k = a; k < b; k += 1) {
    const s = sign(rateDegS[k]);
    const d = Math.abs(rateDegS[k]) * dtS;
    if (k > a && ((s !== 0 && s0 !== 0 && s !== s0) || turned + d > ARC_PIECE_DEG + 1e-9)) {
      keep.push(k);
      s0 = 0;
      turned = 0;
    }
    if (s !== 0) s0 = s;
    turned += d;
  }
  keep.push(b);
  return keep;
}

class Polyline {
  constructor(eastM, northM) {
    this.p = [[eastM[0], northM[0]]];
    for (let i = 1; i < eastM.length; i += 1) {
      const last = this.p[this.p.length - 1];
      if (Math.hypot(eastM[i] - last[0], northM[i] - last[1]) > 1e-9) this.p.push([eastM[i], northM[i]]);
    }
    if (this.p.length < 2) throw new RangeError('a path to follow needs at least two distinct points');
    this.d = [];
    this.length = [];
    this.cum = [0];
    for (let i = 0; i + 1 < this.p.length; i += 1) {
      const [ax, ay] = this.p[i];
      const [bx, by] = this.p[i + 1];
      const L = Math.hypot(bx - ax, by - ay);
      this.d.push([(bx - ax) / L, (by - ay) / L]);
      this.length.push(L);
      this.cum.push(this.cum[this.cum.length - 1] + L);
    }
    this.total = this.cum[this.cum.length - 1];
  }

  /** The segment holding distance s: the last whose start is at or before it. */
  segmentAt(s) {
    let lo = 0;
    let hi = this.cum.length;           // bisect_right(cum, s)
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (s < this.cum[mid]) hi = mid; else lo = mid + 1;
    }
    return Math.min(Math.max(lo - 1, 0), this.length.length - 1);
  }

  point(s) {
    const c = Math.min(Math.max(s, 0), this.total);
    const i = this.segmentAt(c);
    const u = c - this.cum[i];
    return [this.p[i][0] + u * this.d[i][0], this.p[i][1] + u * this.d[i][1]];
  }

  /** The nearest point to (x, y) between sLo and sHi along the path: [s, distance]. */
  project(x, y, sLo, sHi) {
    const hiS = Math.min(sHi, this.total);
    let bestS = sLo;
    let bestD = Infinity;
    let i = this.segmentAt(sLo);
    while (i < this.length.length && this.cum[i] <= hiS) {
      const lo = Math.max(sLo, this.cum[i]) - this.cum[i];
      const hi = Math.min(hiS, this.cum[i + 1]) - this.cum[i];
      const [ax, ay] = this.p[i];
      const [dx, dy] = this.d[i];
      const u = Math.min(Math.max((x - ax) * dx + (y - ay) * dy, lo), hi);
      const dist = Math.hypot(x - (ax + u * dx), y - (ay + u * dy));
      if (dist < bestD) {
        bestS = this.cum[i] + u;
        bestD = dist;
      }
      i += 1;
    }
    return [bestS, bestD];
  }

  /** The first point after s0 where the path leaves the circle of `radius` about (x, y). */
  exit(x, y, s0, radius) {
    let i = this.segmentAt(s0);
    let u0 = s0 - this.cum[i];
    while (i < this.length.length) {
      const [ax, ay] = this.p[i];
      const [dx, dy] = this.d[i];
      const b = (ax - x) * dx + (ay - y) * dy;
      const c = (ax - x) ** 2 + (ay - y) ** 2 - radius * radius;
      const disc = b * b - c;
      if (disc < 0) return this.cum[i] + u0;
      const uOut = -b + Math.sqrt(disc);
      if (uOut < u0) return this.cum[i] + u0;
      if (uOut <= this.length[i]) return this.cum[i] + uOut;
      i += 1;
      u0 = 0;
    }
    return this.total;
  }
}

/**
 * Fly a drawn path from its first point with the L1 autopilot, for `durationS`.
 * Returns {tS, eastM, northM, headingDeg, rateDegS} every dtS, and path(t0, t1) for the
 * chords flown in (t0, t1].
 */
export function follow(eastM, northM, headingDeg, durationS, speedMs, turnRateDegS,
  { l1M = null, dtS = AUTOPILOT_DT_S } = {}) {
  if (isInstant(turnRateDegS)) throw new RangeError('an instantly turning helicopter flies the drawn path itself');
  if (!(turnRateDegS > 0)) throw new RangeError(`the turn rate must be positive, got ${turnRateDegS}`);
  if (!(speedMs > 0)) throw new RangeError(`the autopilot needs a positive speed, got ${speedMs}`);
  const omega = turnRateDegS;
  const line = new Polyline(eastM, northM);
  const l1 = l1M === null ? speedMs / (omega * TO_RAD) : l1M;
  const ticks = Math.round(durationS / dtS);
  if (ticks < 1 || Math.abs(ticks * dtS - durationS) > 1e-9) {
    throw new RangeError(`the duration must be whole ticks of ${dtS} s, got ${durationS}`);
  }
  let [x, y] = line.p[0];
  let psi = normalise(headingDeg);
  let s = 0;
  const window = 2 * l1 + speedMs * dtS;
  const out = { tS: [0], eastM: [x], northM: [y], headingDeg: [psi], rateDegS: [] };
  for (let k = 0; k < ticks; k += 1) {
    let dist;
    [s, dist] = line.project(x, y, s, s + window);
    const ref = dist < l1 ? line.point(line.exit(x, y, s, l1)) : line.point(s);
    const de = ref[0] - x;
    const dn = ref[1] - y;
    let rate = 0;
    if (Math.hypot(de, dn) > 1e-9) {
      const eta = shortestTurn(psi, bearing(de, dn)) * TO_RAD;
      const push = Math.abs(eta) <= Math.PI / 2 ? Math.sin(eta) : Math.sign(eta);
      rate = ((2 * speedMs * push) / l1) * TO_DEG;
      rate = Math.max(-omega, Math.min(omega, rate));
    }
    [x, y, psi] = arcPoint(x, y, psi, rate, dtS, speedMs);
    out.tS.push((k + 1) * dtS);
    out.eastM.push(x);
    out.northM.push(y);
    out.headingDeg.push(psi);
    out.rateDegS.push(rate);
  }
  out.rateDegS.push(0);
  out.dtS = dtS;
  out.path = (t0, t1) => {
    const a = Math.round(t0 / dtS);
    const b = Math.round(t1 / dtS);
    const keep = chordEnds(out.rateDegS, a, b, dtS);
    return {
      tS: keep.map((i) => out.tS[i]),
      eastM: keep.map((i) => out.eastM[i]),
      northM: keep.map((i) => out.northM[i]),
      headingDeg: keep.map((i) => out.headingDeg[i]),
    };
  };
  return out;
}

/* ---------------------------------------------------------------------- the referee's check */

/**
 * Refuse a path the helicopter could not fly at its speed and turn rate: throws RangeError.
 * `start` is [east, north, heading] at time 0. The same three tests as Python's
 * check_turns: the turn rate, the chord's length at constant speed, and its direction.
 */
export function checkTurns(tS, eastM, northM, headingDeg, start, speedMs, turnRateDegS) {
  const omega = turnRateDegS;
  const w = omega * TO_RAD;
  const v = speedMs;
  let [e0, n0, h0] = start;
  let t0 = 0;
  for (let i = 0; i < tS.length; i += 1) {
    const t = tS[i];
    const e = eastM[i];
    const n = northM[i];
    const h = headingDeg[i];
    const dt = t - t0;
    const dh = shortestTurn(h0, h);
    const a = Math.abs(dh);
    if (a > MAX_CHORD_TURN_DEG + ABS_TOL_DEG) {
      throw new RangeError(`a chord ending at ${t} s turns ${a.toFixed(1)} degrees; cut turns into pieces of at most ${MAX_CHORD_TURN_DEG}`);
    }
    if (a > omega * dt * (1 + REL_TOL) + ABS_TOL_DEG) {
      throw new RangeError(`the path turns ${a.toFixed(2)} degrees in ${dt} s ending at ${t} s, faster than the helicopter's ${omega.toFixed(2)} deg/s`);
    }
    const aRad = a * TO_RAD;
    const tau = Math.min(aRad / w, dt);
    const arc = aRad > 0 ? ((2 * v) / w) * Math.sin(aRad / 2) : 0;
    const shortest = v * (dt - tau) * Math.cos(aRad / 2) + arc;
    const longest = v * (dt - tau) + arc;
    const c = Math.hypot(e - e0, n - n0);
    if (c < shortest * (1 - REL_TOL) - ABS_TOL_M || c > longest * (1 + REL_TOL) + ABS_TOL_M) {
      throw new RangeError(`the chord ending at ${t} s is ${c.toFixed(1)} m; turning ${a.toFixed(1)} degrees it must be ${shortest.toFixed(1)} to ${longest.toFixed(1)} m`);
    }
    if (c > ABS_TOL_M) {
      const side = dh >= 0 ? 1 : -1;
      const off = side * shortestTurn(h0, bearing(e - e0, n - n0));
      const frac = dt > 0 ? tau / (2 * dt) : 0.5;
      const slack = (aRad ** 3 / 5) * TO_DEG + ABS_TOL_DEG;
      if (!(a * frac - slack <= off && off <= a * (1 - frac) + slack)) {
        throw new RangeError(`the chord ending at ${t} s points ${off.toFixed(2)} degrees into a ${a.toFixed(2)} degree turn: a corner, not a turn`);
      }
    }
    [e0, n0, h0, t0] = [e, n, h, t];
  }
}

/* -------------------------------------------------------------------- a pattern as flown */

/**
 * A drawn pattern (patterns.js) as the helicopter flies it with the autopilot: the same
 * shape of object, {kind, eastM, northM, tS, headingDeg (each leg), speedMs, durationS}, so
 * everything that draws or times a pattern takes it unchanged; its legs are the chords of
 * the flown path. `drawn` keeps the manual's drawing. An instant helicopter flies the
 * drawing itself.
 */
export function flownPattern(pattern, { headingDeg, durationS, turnRateDegS }) {
  if (isInstant(turnRateDegS)) return pattern;
  const dt = AUTOPILOT_DT_S;
  const duration = Math.floor(durationS / dt + 1e-9) * dt;
  const f = follow(pattern.eastM, pattern.northM, headingDeg, duration, pattern.speedMs, turnRateDegS);
  const p = f.path(0, duration);
  const eastM = [f.eastM[0], ...p.eastM];
  const northM = [f.northM[0], ...p.northM];
  const legs = [];
  for (let i = 0; i + 1 < eastM.length; i += 1) legs.push(bearing(eastM[i + 1] - eastM[i], northM[i + 1] - northM[i]));
  return {
    kind: pattern.kind, eastM, northM, tS: [0, ...p.tS], headingDeg: legs, speedMs: pattern.speedMs,
    durationS: duration, drawn: pattern, turnRateDegS,
    ...(pattern.area ? { area: pattern.area } : {}),
  };
}
