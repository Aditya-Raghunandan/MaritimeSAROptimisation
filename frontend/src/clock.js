/**
 * clock.js -- the one clock every layer reads.
 *
 * IT HOLDS A TIMESTAMP, NOT A FRAME INDEX, and that is the whole point.
 *
 * The layers this map has to carry run at different cadences and always will:
 * wind is hourly, HYCOM current is 3-hourly, the drift engine emits roughly
 * every 15 minutes (D009), a drifter reports on its own irregular schedule, and
 * a searching helicopter moves continuously. If the slider held "frame 17"
 * there would be no answer to the question "frame 17 of what", and every layer
 * added later would need the others re-indexed.
 *
 * So the clock holds a moment in time, and each layer answers `at(t)` in its
 * own units. Adding the probability map later is then a matter of that layer
 * knowing its own cadence, and nothing else changing.
 */

export class Clock {
  /**
   * @param {Date} start  first moment, inclusive
   * @param {Date} end    last moment, EXCLUSIVE -- the project-wide convention
   * @param {number} stepSeconds  granularity the slider moves in
   */
  constructor(start, end, stepSeconds) {
    this.start = start;
    this.end = end;
    this.stepSeconds = stepSeconds;
    this.t = new Date(start.getTime());
    this._listeners = new Set();
  }

  static fromManifest(manifest) {
    const c = manifest.clock;
    return new Clock(new Date(c.start), new Date(c.end), c.step_seconds);
  }

  /** Number of slider positions. End is exclusive, so this is a count of steps. */
  get steps() {
    return Math.max(1, Math.round((this.end - this.start) / (this.stepSeconds * 1000)));
  }

  get index() {
    return Math.round((this.t - this.start) / (this.stepSeconds * 1000));
  }

  /** Move to a slider position, clamped into range. */
  setIndex(i) {
    const clamped = Math.min(Math.max(i, 0), this.steps - 1);
    this.setTime(new Date(this.start.getTime() + clamped * this.stepSeconds * 1000));
  }

  /**
   * Change the granularity the slider steps in, keeping the moment.
   *
   * Switching published tier changes how finely the archive can be stepped,
   * not what time it is. Because this clock holds a timestamp rather than a
   * frame index, the current moment survives the change and every layer
   * re-derives its own frame from it -- which is the entire reason it holds a
   * timestamp. Swapping an index-based clock between a daily and an hourly
   * tier would land on 1/24th of the intended date.
   */
  setStep(stepSeconds) {
    if (!(stepSeconds > 0) || stepSeconds === this.stepSeconds) return;
    this.stepSeconds = stepSeconds;
    for (const fn of this._listeners) fn(this.t);
  }

  setTime(when) {
    if (when.getTime() === this.t.getTime()) return;
    this.t = when;
    for (const fn of this._listeners) fn(this.t);
  }

  onChange(fn) {
    this._listeners.add(fn);
    return () => this._listeners.delete(fn);
  }

  /**
   * The frame of a layer's own time axis that is current.
   *
   * NEAREST, not most-recent: a 3-hourly current field shown at 01:30 should
   * snap forward to 03:00 rather than hold 00:00 for the next ninety minutes,
   * which would make the current visibly lag the wind across the same map.
   *
   * @param {{start: Date, stepSeconds: number, frames: number}} axis
   */
  frameOf(axis) {
    const offset = (this.t - axis.start) / (axis.stepSeconds * 1000);
    return Math.min(Math.max(Math.round(offset), 0), axis.frames - 1);
  }

  /** UTC, always, and labelled as such. Every source in this project is UTC. */
  label() {
    const iso = this.t.toISOString();
    return `${iso.slice(0, 10)} ${iso.slice(11, 16)} UTC`;
  }
}
