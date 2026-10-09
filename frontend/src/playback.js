/**
 * playback.js -- a flight played through the referee over time (D031, Stages 4 and 5).
 *
 * Two kinds of flight, one interface, so the scenario page and the game treat them alike:
 *
 *   RecordedFlight  a benchmark searcher's flight from a scenario bundle: the Waypoints the
 *                   paper's referee was given, step by step. Replayed through referee.js, it
 *                   gives the paper's numbers (held to 1e-12 by the golden fixture).
 *   PlayerFlight    a person steering. The heading they ask for is read every SAMPLE_S
 *                   seconds of search time and held for that long; the helicopter turns to
 *                   it at the turn rate, the short way round, then flies straight (D032).
 *                   That is exactly a heading record that Python's `replay_policy`
 *                   (sar.search.episode) flies the same way, which is how a player is judged
 *                   by the paper's own referee.
 *
 * Both fly a whole minute at a time, when it has finished, as the referee does; the map
 * drains minute by minute. Between, `position(t)` says where the helicopter is drawn.
 *
 * WHICH HELICOPTER. A window from a format 2 bundle carries the turn rate and the heading
 * every searcher arrived on, and both flights use them. A format 1 window (scenarios/v1)
 * was flown by the old helicopter that turned at once: its recorded flights replay that
 * way, while a player always flies the D032 helicopter.
 */

import { Episode, STEP_S, offsetPosition, trackAt } from './referee.js';
import { bearing, normalise, shortestTurn, steerTo } from './kinematics.js';
import { TURN_RATE_DEG_S } from './platform.js';

/** Seconds of search time a player's heading is held for: 12 commands a minute. */
export const SAMPLE_S = 5;

/** The offset at s seconds into a step, from where the step started, along its waypoints. */
export function offsetWithin(start, wp, s) {
  let t0 = 0;
  let [e0, n0] = start;
  for (let i = 0; i < wp.tS.length; i += 1) {
    const t1 = wp.tS[i];
    const e1 = wp.eastM[i];
    const n1 = wp.northM[i];
    if (s <= t1) {
      const f = t1 > t0 ? (s - t0) / (t1 - t0) : 1;
      return [e0 + f * (e1 - e0), n0 + f * (n1 - n0)];
    }
    t0 = t1;
    e0 = e1;
    n0 = n1;
  }
  return [e0, n0];
}

/** The turn rate and arrival heading a window's recorded flights were flown with. */
export function helicopterOf(window) {
  const meta = window.meta ?? {};
  return {
    turnRateDegS: window.turnRateDegS ?? meta.turn_rate_deg_s ?? Infinity,
    headingDeg: window.arrivalHeadingDeg ?? meta.arrival_heading_deg ?? meta.drift_bearing_deg ?? 0,
  };
}

class Flight {
  constructor(window, { turnRateDegS, headingDeg } = helicopterOf(window)) {
    this.window = window;
    this.ep = new Episode(window.cloud, window.marker, { target: window.target, turnRateDegS, headingDeg });
  }

  get done() { return this.ep.done; }

  get duration() { return this.ep.steps * STEP_S; }

  /** Ground position [lat, lon] (store longitude) for an offset at t. */
  ground(t, offset) {
    const [mlat, mlon] = trackAt(this.window.marker, Math.min(this.duration, Math.max(0, t)));
    return offsetPosition(mlat, mlon, offset[0], offset[1]);
  }

  /** POS so far, and the drain rate of every finished minute. */
  metrics() { return this.ep.metrics(); }

  /**
   * The ground track so far, [[t, lat, lon], ...]: every finished minute as the referee flew
   * it, then the minute in progress along its own path up to t, then the helicopter. The
   * referee only takes a minute once it has finished, so without the middle part the trail
   * would cut a straight chord across a turn and snap to the curve when the minute ended.
   */
  trail(t) {
    return [...this.ep.track, ...this.flownThisMinute(t), [t, ...this.position(t)]];
  }

  /** The points of the minute in progress already passed by t, on the ground: [[t, lat, lon]]. */
  flownThisMinute() {
    return [];
  }
}

export class RecordedFlight extends Flight {
  constructor(window, searcher) {
    super(window);
    const flight = window.flights[searcher];
    if (!flight) throw new Error(`no ${searcher} flight in this window`);
    this.searcher = searcher;
    this.steps = flight.steps;
    this.python = flight.python;
  }

  /** Fly every minute that has finished by t seconds after arrival. */
  advanceTo(t) {
    while (!this.ep.done && (this.ep.k + 1) * STEP_S <= t + 1e-9) {
      this.ep.fly(this.steps[this.ep.k]);
    }
  }

  /** Where the helicopter is drawn at t (it must have been advanced to t). */
  position(t) {
    const k = this.ep.k;
    const offset = k >= this.steps.length
      ? this.ep.offset
      : offsetWithin(this.ep.offset, this.steps[k], t - k * STEP_S);
    return this.ground(t, offset);
  }

  flownThisMinute(t) {
    const k = this.ep.k;
    if (k >= this.steps.length) return [];
    const wp = this.steps[k];
    const s = t - k * STEP_S;
    const out = [];
    for (let i = 0; i < wp.tS.length && wp.tS[i] < s; i += 1) {
      const at = k * STEP_S + wp.tS[i];
      out.push([at, ...this.ground(at, [wp.eastM[i], wp.northM[i]])]);
    }
    return out;
  }

  /** Which way it points at t: between the headings at the waypoints either side. */
  headingAt(t) {
    const k = this.ep.k;
    if (k >= this.steps.length) return this.ep.headingDeg;
    const wp = this.steps[k];
    const s = t - k * STEP_S;
    let t0 = 0;
    let h0 = this.ep.headingDeg;
    let [e0, n0] = this.ep.offset;
    for (let i = 0; i < wp.tS.length; i += 1) {
      const h1 = wp.headingDeg ? wp.headingDeg[i] : bearing(wp.eastM[i] - e0, wp.northM[i] - n0);
      if (s <= wp.tS[i]) {
        const f = wp.tS[i] > t0 ? (s - t0) / (wp.tS[i] - t0) : 1;
        return normalise(h0 + f * shortestTurn(h0, h1));
      }
      [t0, h0, e0, n0] = [wp.tS[i], h1, wp.eastM[i], wp.northM[i]];
    }
    return h0;
  }
}

export class PlayerFlight extends Flight {
  /**
   * `heading` is where the helicopter points on arrival (the window's, normally), and the
   * first command; `turnRateDegS` the helicopter's, TURN_RATE_DEG_S unless a test says so.
   */
  constructor(window, { heading = helicopterOf(window).headingDeg, turnRateDegS = TURN_RATE_DEG_S } = {}) {
    super(window, { turnRateDegS, headingDeg: heading });
    this.heading = normalise(heading);
    this.command = this.heading;
    this.t = 0;
    this.offset = [0, 0];
    this.pending = { tS: [], eastM: [], northM: [], headingDeg: [] };
    this.headings = { tS: [], headingDeg: [] };
  }

  get turnRateDegS() { return this.ep.turnRateDegS; }

  /** The heading the helicopter should turn to and hold, degrees true, about the marker. */
  steer(headingDeg) {
    this.command = normalise(headingDeg);
  }

  /** Fly on to t seconds after arrival, one held command per SAMPLE_S. */
  advanceTo(t) {
    const end = Math.min(t, this.duration);
    while (this.t + SAMPLE_S <= end + 1e-9 && !this.ep.done) {
      this.headings.tS.push(this.t);
      this.headings.headingDeg.push(this.command);
      const p = steerTo(this.offset[0], this.offset[1], this.heading, this.command, SAMPLE_S,
        this.ep.speedMs, this.turnRateDegS);
      const startInStep = this.t - this.ep.k * STEP_S;
      for (let i = 0; i < p.tS.length; i += 1) {
        this.pending.tS.push(startInStep + p.tS[i]);
        this.pending.eastM.push(p.eastM[i]);
        this.pending.northM.push(p.northM[i]);
        this.pending.headingDeg.push(p.headingDeg[i]);
      }
      const last = p.tS.length - 1;
      this.offset = [p.eastM[last], p.northM[last]];
      this.heading = p.headingDeg[last];
      this.t += SAMPLE_S;
      if (Math.abs(this.t - (this.ep.k + 1) * STEP_S) < 1e-9) {
        this.pending.tS[this.pending.tS.length - 1] = STEP_S;
        this.ep.fly(this.pending);
        this.pending = { tS: [], eastM: [], northM: [], headingDeg: [] };
      }
    }
  }

  /** Where it is and which way it points at t >= this.t: on its turn, if it is turning. */
  pose(t) {
    const ahead = Math.max(0, Math.min(t, this.duration) - this.t);
    if (ahead <= 0) return { offset: this.offset, heading: this.heading };
    const p = steerTo(this.offset[0], this.offset[1], this.heading, this.command, ahead,
      this.ep.speedMs, this.turnRateDegS);
    const last = p.tS.length - 1;
    return { offset: [p.eastM[last], p.northM[last]], heading: p.headingDeg[last] };
  }

  /** Where the helicopter is drawn at t. */
  position(t) {
    return this.ground(t, this.pose(t).offset);
  }

  /** Which way it points at t. */
  headingAt(t) {
    return this.pose(t).heading;
  }

  flownThisMinute(t) {
    const start = this.ep.k * STEP_S;
    const out = this.pending.tS.map((s, i) => {
      const at = start + s;
      return [at, ...this.ground(at, [this.pending.eastM[i], this.pending.northM[i]])];
    });
    // The arc from the last held command to t, piece by piece, so a turn is drawn as one.
    const ahead = Math.min(t, this.duration) - this.t;
    if (ahead > 0) {
      const p = steerTo(this.offset[0], this.offset[1], this.heading, this.command, ahead,
        this.ep.speedMs, this.turnRateDegS);
      for (let i = 0; i + 1 < p.tS.length; i += 1) {
        const at = this.t + p.tS[i];
        out.push([at, ...this.ground(at, [p.eastM[i], p.northM[i]])]);
      }
    }
    return out;
  }

  /**
   * Where it will go if the command holds: the next `seconds` of its turn and the straight
   * after, as ground points, so a player sees the bend coming before it happens.
   */
  preview(t, seconds = 30) {
    const { offset, heading } = this.pose(t);
    const p = steerTo(offset[0], offset[1], heading, this.command, seconds, this.ep.speedMs,
      this.turnRateDegS);
    return [this.ground(t, offset), ...p.tS.map((_, i) => this.ground(t, [p.eastM[i], p.northM[i]]))];
  }

  /** The flight as Python's replay_policy reads it: {t_s, heading_deg, turn_rate_deg_s}. */
  record() {
    return {
      t_s: [...this.headings.tS],
      heading_deg: [...this.headings.headingDeg],
      turn_rate_deg_s: Number.isFinite(this.turnRateDegS) ? this.turnRateDegS : null,
    };
  }
}
