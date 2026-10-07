/**
 * playback.js -- a flight played through the referee over time (D031, Stages 4 and 5).
 *
 * Two kinds of flight, one interface, so the scenario page and the game treat them alike:
 *
 *   RecordedFlight  a benchmark searcher's flight from a scenario bundle: the Waypoints the
 *                   paper's referee was given, step by step. Replayed through referee.js, it
 *                   gives the paper's numbers (held to 1e-12 by the golden fixture).
 *   PlayerFlight    a person steering. The heading is read every SAMPLE_S seconds of search
 *                   time and held for that long, so the flight is a sequence of straight
 *                   legs at full speed: exactly a heading record that Python's
 *                   `replay_policy` (sar.search.episode) can score again, which is how a
 *                   player is judged by the paper's own referee.
 *
 * Both fly a whole minute at a time, when it has finished, as the referee does; the map
 * drains minute by minute. Between, `position(t)` says where the helicopter is drawn.
 */

import { Episode, STEP_S, offsetPosition, trackAt } from './referee.js';

/** Seconds of search time a player's heading is held for: 12 legs a minute. */
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

class Flight {
  constructor(window) {
    this.window = window;
    this.ep = new Episode(window.cloud, window.marker, { target: window.target });
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

  /** The ground track so far, sub-leg ends: [[t, lat, lon], ...], plus the helicopter now. */
  trail(t) {
    return [...this.ep.track, [t, ...this.position(t)]];
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
}

export class PlayerFlight extends Flight {
  constructor(window, { heading = 0 } = {}) {
    super(window);
    this.heading = heading;
    this.t = 0;
    this.offset = [0, 0];
    this.pending = { tS: [], eastM: [], northM: [] };
    this.headings = { tS: [], headingDeg: [] };
  }

  /** The heading the next leg will hold, degrees true, relative to the marker. */
  steer(headingDeg) {
    this.heading = ((headingDeg % 360) + 360) % 360;
  }

  /** Fly on to t seconds after arrival, a held heading per SAMPLE_S leg. */
  advanceTo(t) {
    const end = Math.min(t, this.duration);
    while (this.t + SAMPLE_S <= end + 1e-9 && !this.ep.done) {
      const b = (this.heading * Math.PI) / 180;
      const d = this.ep.speedMs * SAMPLE_S;
      this.headings.tS.push(this.t);
      this.headings.headingDeg.push(this.heading);
      this.offset = [this.offset[0] + d * Math.sin(b), this.offset[1] + d * Math.cos(b)];
      this.t += SAMPLE_S;
      const inStep = this.t - this.ep.k * STEP_S;
      this.pending.tS.push(inStep);
      this.pending.eastM.push(this.offset[0]);
      this.pending.northM.push(this.offset[1]);
      if (Math.abs(inStep - STEP_S) < 1e-9) {
        this.ep.fly(this.pending);
        this.pending = { tS: [], eastM: [], northM: [] };
      }
    }
  }

  /** Where the helicopter is drawn at t: on the current leg, at the held heading. */
  position(t) {
    const ahead = Math.max(0, Math.min(t, this.duration) - this.t);
    const b = (this.heading * Math.PI) / 180;
    const d = this.ep.speedMs * ahead;
    return this.ground(t, [this.offset[0] + d * Math.sin(b), this.offset[1] + d * Math.cos(b)]);
  }

  /** The flight as Python's replay_policy reads it: {t_s, heading_deg}, from 0. */
  record() {
    return { t_s: [...this.headings.tS], heading_deg: [...this.headings.headingDeg] };
  }
}
