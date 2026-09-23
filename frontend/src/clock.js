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
    // The slider spans a WINDOW, not the whole archive. Defaults to the whole
    // archive so a caller that never sets one behaves as before.
    this.winStart = new Date(start.getTime());
    this.winEnd = new Date(end.getTime());
    this._listeners = new Set();
  }

  static fromManifest(manifest) {
    const c = manifest.clock;
    return new Clock(new Date(c.start), new Date(c.end), c.step_seconds);
  }

  /**
   * Number of slider positions, ACROSS THE WINDOW.
   *
   * The slider used to span the whole archive at whatever stride was chosen,
   * which made the finer tiers useless: hourly over five years is 43,824
   * positions, so one pixel of travel was several hours and there was no way
   * to step through a single day. A slider is a fixed number of pixels, so
   * what it spans has to be bounded by something other than the archive.
   */
  get steps() {
    return Math.max(1, Math.round((this.winEnd - this.winStart) / (this.stepSeconds * 1000)));
  }

  get index() {
    return Math.round((this.t - this.winStart) / (this.stepSeconds * 1000));
  }

  /** Move to a slider position, clamped into the window. */
  setIndex(i) {
    const clamped = Math.min(Math.max(i, 0), this.steps - 1);
    this.setTime(new Date(this.winStart.getTime() + clamped * this.stepSeconds * 1000));
  }

  /**
   * Put a window of `spanSeconds` around the current moment.
   *
   * Aligned to a whole multiple of the span from the epoch rather than centred
   * on `t`, so a 24 h window is a UTC DAY -- 00:00 to 00:00 -- rather than an
   * arbitrary window that happens to contain the cursor. A window whose edges
   * move every time you touch the slider is impossible to reason about, and
   * "the 6th of May" is a thing a reader can name.
   *
   * `null` or 0 means the whole archive, which is what the coarsest tier
   * wants: at daily resolution the archive IS the overview.
   */
  setWindowSpan(spanSeconds) {
    if (!spanSeconds) {
      this.winStart = new Date(this.start.getTime());
      this.winEnd = new Date(this.end.getTime());
    } else {
      const ms = spanSeconds * 1000;
      const aligned = Math.floor(this.t.getTime() / ms) * ms;
      this.winStart = new Date(Math.max(aligned, this.start.getTime()));
      this.winEnd = new Date(Math.min(aligned + ms, this.end.getTime()));
    }
    this._clampIntoWindow();
    for (const fn of this._listeners) fn(this.t);
    return this.windowSpanSeconds;
  }

  /** Window length in seconds, or null when it is the whole archive. */
  get windowSpanSeconds() {
    const whole = this.winStart.getTime() === this.start.getTime()
      && this.winEnd.getTime() === this.end.getTime();
    return whole ? null : (this.winEnd - this.winStart) / 1000;
  }

  /**
   * Slide the window by `n` of its own lengths, carrying the cursor with it.
   *
   * The cursor keeps its offset into the window, so stepping forward a day at
   * 09:00 lands on 09:00 the next day rather than snapping to midnight.
   * Returns false when there is nothing that way, so the caller can grey out
   * the button rather than offering a move that does nothing.
   */
  shiftWindow(n) {
    const span = this.winEnd - this.winStart;
    if (!span) return false;
    const offset = this.t - this.winStart;
    const wantStart = this.winStart.getTime() + n * span;
    const maxStart = this.end.getTime() - span;
    const clamped = Math.min(Math.max(wantStart, this.start.getTime()), maxStart);
    if (clamped === this.winStart.getTime()) return false;

    this.winStart = new Date(clamped);
    this.winEnd = new Date(clamped + span);
    this.t = new Date(clamped + offset);
    this._clampIntoWindow();
    for (const fn of this._listeners) fn(this.t);
    return true;
  }

  /** Can the window move that way at all? For enabling the arrows. */
  canShift(n) {
    const span = this.winEnd - this.winStart;
    if (!span) return false;
    const maxStart = this.end.getTime() - span;
    const want = Math.min(Math.max(this.winStart.getTime() + n * span, this.start.getTime()), maxStart);
    return want !== this.winStart.getTime();
  }

  _clampIntoWindow() {
    const lo = this.winStart.getTime();
    const hi = this.winEnd.getTime() - this.stepSeconds * 1000;
    const t = Math.min(Math.max(this.t.getTime(), lo), Math.max(lo, hi));
    this.t = new Date(t);
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
    this._clampIntoWindow();
    for (const fn of this._listeners) fn(this.t);
  }

  setTime(when) {
    if (when.getTime() === this.t.getTime()) return;
    this.t = when;
    for (const fn of this._listeners) fn(this.t);
  }

  /**
   * Go to a moment anywhere in the archive, and bring a window of `spanSeconds` with
   * it, in one move.
   *
   * `setTime` alone leaves the window where it was, so a moment outside it is the
   * next thing anything clamps: a tier switch calls `setStep`, which pulls `t` back
   * into the OLD window before the new one is set. Clicking a buoy first seen in
   * March 2021 from a view of 1 January 2019 landed the clock on 1 January 2019.
   * Found 23 Sep.
   */
  jumpTo(when, spanSeconds) {
    const t = Math.min(Math.max(when.getTime(), this.start.getTime()), this.end.getTime() - 1);
    this.t = new Date(t);
    return this.setWindowSpan(spanSeconds);
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

  /** The window as a readable range, for the label beside the slider. */
  windowLabel() {
    const day = (d) => d.toISOString().slice(0, 10);
    const hm = (d) => d.toISOString().slice(11, 16);
    const span = this.windowSpanSeconds;
    if (span === null) return `${day(this.winStart)} → ${day(this.winEnd)}`;

    // `winEnd` is exclusive, so the last moment inside the window is one
    // millisecond before it. Using winEnd itself printed a whole day as
    // "00:00-00:00", which reads as an empty range rather than as a full one.
    const last = new Date(this.winEnd.getTime() - 1);
    const wholeDays = span % 86400 === 0
      && this.winStart.getTime() % 86400000 === 0;

    if (wholeDays && span === 86400) return `${day(this.winStart)} UTC`;
    if (wholeDays) return `${day(this.winStart)} → ${day(last)} UTC`;
    if (day(this.winStart) === day(last)) {
      return `${day(this.winStart)} ${hm(this.winStart)}–${hm(this.winEnd)} UTC`;
    }
    return `${day(this.winStart)} ${hm(this.winStart)} → ${day(last)} ${hm(this.winEnd)} UTC`;
  }
}
