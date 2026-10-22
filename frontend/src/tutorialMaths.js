/**
 * tutorialMaths.js -- the numbers behind the three missions before the real search, kept
 * apart from the drawing so they can be tested.
 *
 *   1. Drift = current + 2 % of the wind (D002, alpha = 0.02, Allen 2000), and a helicopter
 *      that cannot turn on the spot (D032: 7 degrees a second at 90 kt).
 *   2. The current is only known on a grid, 8 km by 4.5 km, every 3 hours (HYCOM, D019); in
 *      between it is blended, bilinear in space and linear in time (D009, interpolate.js).
 *   3. The model's small error persists (D030: a velocity error per drifter, sigma_u = 0.226
 *      m/s, remembered for 25.7 h), so one forecast becomes 10,000 and the cloud must be the
 *      honest size: too small misses the buoy, too big cannot be searched.
 *
 * The missions are teaching toys on made-up but realistic numbers; the game after them is
 * the real thing.
 */

import { SEARCH_SPEED_MS, TURN_RATE_DEG_S } from './platform.js';
import { M_PER_DEG_LAT } from './geo.js';
import { shortestTurn, normalise } from './kinematics.js';

export const LEEWAY = 0.02;
export const SIGMA_TRUE = 0.226;        // D030, m/s
export const HALF_STRIP_M = 92.6;       // half of the 185.2 m sweep width (D027)

/** Drift velocity [east, north] m/s from the current and the wind (both [east, north] m/s). */
export function driftOf(current, wind, leeway = LEEWAY) {
  return [current[0] + leeway * wind[0], current[1] + leeway * wind[1]];
}

/** Metres [east, north] about (lat0, lon0) to [lat, lon]. */
export function toLatLon(lat0, lon0, east, north) {
  return [lat0 + north / M_PER_DEG_LAT, lon0 + east / (M_PER_DEG_LAT * Math.cos((lat0 * Math.PI) / 180))];
}

/** [lat, lon] to metres [east, north] about (lat0, lon0). */
export function toMetres(lat0, lon0, lat, lon) {
  return [(lon - lon0) * M_PER_DEG_LAT * Math.cos((lat0 * Math.PI) / 180), (lat - lat0) * M_PER_DEG_LAT];
}

/**
 * The helicopter, one tick of `dt` seconds: it turns toward `wanted` (a compass heading, or
 * null to hold) at no more than the turn rate, then flies straight on. Returns the new state.
 */
export function heliStep({ east, north, heading }, wanted, dt, { speed = SEARCH_SPEED_MS, rate = TURN_RATE_DEG_S } = {}) {
  let h = heading;
  if (wanted !== null && wanted !== undefined) {
    const turn = shortestTurn(heading, wanted);
    const most = rate * dt;
    h = normalise(heading + Math.max(-most, Math.min(most, turn)));
  }
  const b = (h * Math.PI) / 180;
  return { east: east + speed * dt * Math.sin(b), north: north + speed * dt * Math.cos(b), heading: h };
}

/** The shortest distance from point p to the segment a-b, all [east, north]. */
export function segmentDistance(p, a, b) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len2 = dx * dx + dy * dy;
  const u = len2 > 0 ? Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2)) : 0;
  return Math.hypot(p[0] - (a[0] + u * dx), p[1] - (a[1] + u * dy));
}

/** Bilinear weights of the four corners (sw, se, nw, ne) for a point fx, fy in [0, 1] across a cell. */
export function bilinearWeights(fx, fy) {
  return [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy];
}

/**
 * Mission 2's field: a smooth current on a 3 x 3 grid, as HYCOM would give it, at two times
 * 3 hours apart. `at(east, north, hours)` blends it, bilinear in space and linear in time,
 * exactly as the engine reads the real one.
 */
export class GridField {
  constructor({ dx = 8000, dy = 4500, nx = 3, ny = 3, frames }) {
    Object.assign(this, { dx, dy, nx, ny, frames });   // frames: [t0, t1], each [ny][nx] of [u, v]
    this.x0 = -dx * (nx - 1) / 2;
    this.y0 = -dy * (ny - 1) / 2;
  }

  points() {
    const out = [];
    for (let j = 0; j < this.ny; j += 1) for (let i = 0; i < this.nx; i += 1) out.push([this.x0 + i * this.dx, this.y0 + j * this.dy, i, j]);
    return out;
  }

  /** The cell holding (east, north), its corners and the fractions across it. */
  cell(east, north) {
    const gx = Math.max(0, Math.min(this.nx - 1 - 1e-9, (east - this.x0) / this.dx));
    const gy = Math.max(0, Math.min(this.ny - 1 - 1e-9, (north - this.y0) / this.dy));
    const i = Math.floor(gx);
    const j = Math.floor(gy);
    return { i, j, fx: gx - i, fy: gy - j };
  }

  /** The current [u, v] at a grid point at `hours` (0 to 3), linear in time. */
  node(i, j, hours) {
    const f = Math.max(0, Math.min(1, hours / 3));
    const [a, b] = [this.frames[0][j][i], this.frames[1][j][i]];
    return [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])];
  }

  at(east, north, hours) {
    const { i, j, fx, fy } = this.cell(east, north);
    const w = bilinearWeights(fx, fy);
    const c = [this.node(i, j, hours), this.node(i + 1, j, hours), this.node(i, j + 1, hours), this.node(i + 1, j + 1, hours)];
    return [w.reduce((s, wk, k) => s + wk * c[k][0], 0), w.reduce((s, wk, k) => s + wk * c[k][1], 0)];
  }

  /** A path through the field from (east, north) for `hours`, `stepS` seconds a step: [[e, n], ...]. */
  path(east, north, hours, { stepS = 60, extra = [0, 0] } = {}) {
    const out = [[east, north]];
    let [e, n] = [east, north];
    for (let s = 0; s < hours * 3600; s += stepS) {
      const [u, v] = this.at(e, n, s / 3600);
      e += (u + extra[0]) * stepS;
      n += (v + extra[1]) * stepS;
      out.push([e, n]);
    }
    return out;
  }
}

/** A small seeded generator (mulberry32) and standard normals from it, so a mission replays the same. */
export function rng(seed) {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const normal = () => Math.sqrt(-2 * Math.log(1 - next())) * Math.cos(2 * Math.PI * next());
  return { next, normal };
}

/**
 * Mission 3's cloud: n drifters after `seconds`, each carried by the drift plus its own error
 * velocity, drawn once (with a 25.7 h memory the error barely changes in 2 h). Offsets
 * [east[], north[]] from where the drift alone would put it.
 */
export function cloudOffsets(n, sigma, seconds, seed = 7) {
  const r = rng(seed);
  const e = new Float64Array(n);
  const nn = new Float64Array(n);
  for (let k = 0; k < n; k += 1) {
    e[k] = sigma * r.normal() * seconds;
    nn[k] = sigma * r.normal() * seconds;
  }
  return [e, nn];
}

/** The radius of the circle holding 90 % of a cloud's offsets, metres. */
export function radius90([e, n]) {
  const d = Array.from(e, (x, k) => Math.hypot(x, n[k])).sort((a, b) => a - b);
  return d[Math.min(d.length - 1, Math.floor(0.9 * d.length))];
}

/**
 * The five "real buoys" of mission 3: offsets from the forecast, drawn from the true error
 * (sigma_u = 0.226 m/s) at radii chosen so that the honest cloud catches four of the five,
 * as the calibrated model catches nine in ten.
 */
export const BUOY_RADII = [0.55, 1.05, 1.45, 1.85, 2.7];   // in units of the true spread per axis
export function trialBuoys(seconds, sigmaTrue = SIGMA_TRUE) {
  const spread = sigmaTrue * seconds;
  return BUOY_RADII.map((r, k) => {
    const a = (k * 137.5 + 20) * (Math.PI / 180);
    return [r * spread * Math.sin(a), r * spread * Math.cos(a)];
  });
}

/** Mission 3's verdict on a chosen error size: how many buoys the 90 % circle catches, and its area. */
export function judgeCloud(sigma, seconds, { n = 2000, seed = 7 } = {}) {
  const cloud = cloudOffsets(n, sigma, seconds, seed);
  const r90 = radius90(cloud);
  const buoys = trialBuoys(seconds);
  const caught = buoys.map(([e, nn]) => Math.hypot(e, nn) <= r90);
  const areaKm2 = (Math.PI * r90 * r90) / 1e6;
  const hits = caught.filter(Boolean).length;
  let verdict = 'right';
  if (hits < 4) verdict = 'small';
  else if (areaKm2 > 80) verdict = 'big';
  return { cloud, r90, buoys, caught, hits, areaKm2, verdict };
}
