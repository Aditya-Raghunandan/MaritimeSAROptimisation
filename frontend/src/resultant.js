/**
 * resultant.js -- the vector a person in the water would actually follow.
 *
 * The map draws the wind. The wind is not where anyone drifts. This composes
 * the forcing fields into the drift model's own left-hand side, so a cell shows
 * the thing the project is about:
 *
 *     v_d = v_c(x,t) + alpha * v_w10(x,t) + eta(t)        (D002, frozen)
 *            current      leeway            stochastic
 *
 * THIS COMPUTES THE FIRST TWO TERMS AND SAYS SO. Issue #10 calls it the
 * "resultant vector method"; it is two thirds of one.
 *
 * eta is not a field. It is a per-particle random draw at each timestep, so it
 * has no value at a location and cannot be drawn as an arrow at all. Showing
 * `v_c + alpha*v_w10` and calling it "drift" would quietly promise a
 * determinism the model does not claim -- the whole reason the model is
 * stochastic is that two people in the same cell do not end up in the same
 * place. `describe()` returns the wording the UI must show, and `isPartial`
 * is true whenever a term is missing, so a caller cannot forget.
 *
 * WHICH GRID. The output is on the WIND grid. Wind is 0.25 deg (77 x 77) and
 * current is 0.08 x 0.04 deg (476 x 238), so one of them has to be resampled
 * and the coarser one is the honest choice: interpolating wind up to the
 * current grid would invent structure at 4 km that a 28 km field does not
 * contain. The current is sampled by NEAREST NEIGHBOUR at each wind cell
 * centre -- no averaging, so the value shown is one the archive really holds.
 *
 * Until the current archive is published this runs with `current = null` and
 * produces the leeway term alone, labelled as such. It is the same code path
 * the moment currents land: nothing here changes, the layer simply stops
 * being partial.
 */

/** D002 / Allen (2000), matching ALPHA in drift.js and ALPHA_MID in wind.py. */
export const ALPHA = 0.02;

export class ResultantSource {
  /**
   * @param {{source, grid}} wind     field source and its grid (the output grid)
   * @param {{source, grid}|null} current  field source and its grid, or null
   * @param {{alpha?: number}} opts
   */
  constructor(wind, current, opts = {}) {
    this.wind = wind;
    this.current = current ?? null;
    this.alpha = opts.alpha ?? ALPHA;
    this.grid = wind.grid;
    this.frames = wind.source.frames;
  }

  /** True while any term of D002 is missing from what is drawn. */
  get isPartial() {
    return this.current === null;
  }

  /** The terms actually included, and the ones that are not. */
  describe() {
    const have = this.current
      ? ['surface current', `leeway (α = ${(this.alpha * 100).toFixed(0)} %)`]
      : [`leeway (α = ${(this.alpha * 100).toFixed(0)} %)`];
    const missing = this.current ? [] : ['surface current'];
    // eta is ALWAYS missing from a field rendering, even once the engine runs.
    missing.push('stochastic η');
    return {
      label: this.current ? 'Resultant drift' : 'Resultant drift (leeway only)',
      have,
      missing,
      caveat: this.current
        ? 'Current + leeway. The stochastic term η is a per-particle draw, not a '
          + 'field, so it cannot be drawn here — two drifters in this cell will not '
          + 'follow the same path.'
        : 'Leeway only — the surface current is not published yet, and it is usually '
          + 'the larger term in the Gulf Stream. This is not yet where a drifter would go.',
    };
  }

  isResident(frame) {
    if (!this.wind.source.isResident(frame)) return false;
    if (!this.current) return true;
    return this.current.source.isResident(this._currentFrame(frame));
  }

  async ensure(frame) {
    const jobs = [this.wind.source.ensure(frame)];
    if (this.current) jobs.push(this.current.source.ensure(this._currentFrame(frame)));
    return Promise.all(jobs);
  }

  /**
   * Map a wind frame to the nearest current frame.
   *
   * Wind is hourly and current 3-hourly, so they do not share an index. NEAREST
   * rather than most-recent: at 01:30 a 3-hourly field should snap forward to
   * 03:00 rather than hold 00:00 for another ninety minutes, which would make
   * the current visibly lag the wind on the same map. Same rule the clock uses.
   */
  _currentFrame(windFrame) {
    const w = this.wind.axis;
    const c = this.current.axis;
    const when = w.start.getTime() + windFrame * w.stepSeconds * 1000;
    const k = Math.round((when - c.start.getTime()) / (c.stepSeconds * 1000));
    return Math.min(Math.max(k, 0), this.current.source.frames - 1);
  }

  /** [u, v] of the resultant at a wind cell, m/s. */
  vector(frame, j, i) {
    const [uw, vw] = this.wind.source.vector(frame, j, i);
    let u = this.alpha * uw;
    let v = this.alpha * vw;

    if (this.current) {
      const cell = this.current.grid.cellAt(this.grid.lat(j), this.grid.lon(i));
      if (cell) {
        const [uc, vc] = this.current.source.vector(this._currentFrame(frame), cell.j, cell.i);
        // A land cell is NaN in HYCOM. Adding it would wipe out the leeway term
        // and blank the arrow, which reads as "no wind" rather than "no sea".
        if (Number.isFinite(uc) && Number.isFinite(vc)) {
          u += uc;
          v += vc;
        }
      }
    }
    return [u, v];
  }
}

/**
 * A sensible top-of-scale for the resultant, in m/s.
 *
 * Not the wind's scale: leeway is 2 % of wind, so on the wind's scale every
 * resultant arrow would be invisible. With currents present the Gulf Stream's
 * ~1.8 m/s dominates and sets the range; without them the scale is the largest
 * leeway the wind can produce.
 */
export function resultantScale(windMaxSpeed, hasCurrent, typicalCurrent = 1.8) {
  const leeway = ALPHA * windMaxSpeed;
  return hasCurrent ? Math.max(typicalCurrent * 1.4, leeway) : Math.max(leeway, 0.05);
}
