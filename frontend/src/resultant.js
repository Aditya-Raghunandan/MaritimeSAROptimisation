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

import { sampleField } from './interpolate.js';

/** D002 / Allen (2000), matching ALPHA in drift.js and ALPHA_MID in wind.py. */
export const ALPHA = 0.02;

/**
 * Whether the current is sampled bilinearly rather than by nearest neighbour.
 *
 * OFF, deliberately. `interpolate.js` is the browser port of the engine's own
 * `sar.model.interpolate`, and it is the better answer at a point: nearest neighbour
 * moves a value by up to half a cell, about 4 km east-west on the current grid. But the
 * DRAWN field is a different question from a queried point. Every arrow becomes four
 * reads instead of one, and the argument written above about resampling still stands:
 * smoothing the current onto the 28 km wind grid invents detail the picture cannot
 * honestly carry.
 *
 * So the flag exists to be switched on once the archive is published and the cost has
 * been measured on a real frame, not before. `sampleAt` below ignores it entirely and is
 * always bilinear, because a number shown beside a clicked position is exactly the case
 * where half a cell matters and the cost does not.
 */
export const BILINEAR_FIELD = false;

export class ResultantSource {
  /**
   * @param {{source, grid}} wind     field source and its grid (the output grid)
   * @param {{source, grid}|null} current  field source and its grid, or null
   * @param {{alpha?: number, bilinear?: boolean}} opts
   *
   * `wind.source` and `current.source` are READ AT EVERY CALL, never copied. Changing
   * the span swaps each field onto a different published tier -- a different store,
   * a different object -- and mutates the shared axis in place. A copy taken here kept
   * reading the tier the page opened on (daily) with frame numbers from the tier it
   * switched to (hourly), so the drift arrows painted the wrong day and then froze at
   * the first chunk nobody was fetching: 2019-01-03 01:00, hour 49, on the live data.
   */
  constructor(wind, current, opts = {}) {
    this.wind = wind;
    this.current = current ?? null;
    this.alpha = opts.alpha ?? ALPHA;
    this.bilinear = opts.bilinear ?? BILINEAR_FIELD;
    this.grid = wind.grid;
  }

  /** Frames in the wind tier in use now, which a span change can alter. */
  get frames() {
    return this.wind.source.frames;
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
      const [uc, vc] = this._currentAt(frame, this.grid.lat(j), this.grid.lon(i));
      /*
        NO CURRENT MEANS NO SEA, SO NO ARROW. HYCOM writes land as NaN. This used
        to keep the leeway term on its own, on the worry that a blank would read
        as "no wind". What it drew instead was drift arrows and streaks over
        Florida, Cuba and Hispaniola, each saying a person there would be carried
        a couple of hundred metres an hour. Wind over land is the wind layer's
        job. This layer answers "where would a person in the water go", and on
        land the answer is nowhere. Found by eye on the live site, 23 Sep.
      */
      if (!Number.isFinite(uc) || !Number.isFinite(vc)) return [NaN, NaN];
      u += uc;
      v += vc;
    }
    return [u, v];
  }

  /**
   * The current at a position, by whichever method the flag selects.
   *
   * Returns [NaN, NaN] rather than throwing, because this is on the draw path: one
   * coastal cell must not take the whole frame down. `sampleAt` handles the same cases
   * with words instead.
   */
  _currentAt(frame, lat, lon) {
    const k = this._currentFrame(frame);
    if (this.bilinear) {
      try {
        const s = sampleField(this.current, k, lat, lon);
        return [s.u, s.v];
      } catch {
        // Off the current grid, or within one cell of land. Either way there is no
        // current to add here, so `vector` draws no arrow.
        return [NaN, NaN];
      }
    }
    const cell = this.current.grid.cellAt(lat, lon);
    if (!cell) return [NaN, NaN];
    return this.current.source.vector(k, cell.j, cell.i);
  }

  /**
   * The resultant at an exact position rather than at a cell: what a click should show.
   *
   * ALWAYS BILINEAR, whatever `bilinear` says, and this is the point of the whole port.
   * Rounding a clicked position to the nearest grid point moves the answer by up to half
   * a cell, and the number is being shown beside the very coordinates it disagrees with.
   * `interpolate.js` computes it exactly as `sar.model.interpolate` does, and
   * `fixtures/resultant_golden.json` is what holds the two to that.
   *
   * The current is reported separately from the leeway rather than only summed, so the
   * caller can show which term dominates. `current` is null when the position has no
   * usable current, with `currentReason` saying why in words; the leeway term is still
   * returned, exactly as the leeway-only layer is still drawn.
   *
   * ON LAND the answer is different in kind, not a partial one: `onLand` is true and
   * `u`, `v` are NaN, because nothing drifts from there. The leeway term is still
   * reported, since the wind is real, but it is not a drift.
   */
  sampleAt(frame, lat, lon) {
    const [uw, vw] = this._windAt(frame, lat, lon);
    const leeway = [this.alpha * uw, this.alpha * vw];

    let current = null;
    let currentReason = 'the surface current is not published yet';
    let uncertainty = null;
    let onLand = false;

    if (this.current) {
      const k = this._currentFrame(frame);
      try {
        const s = sampleField(this.current, k, lat, lon);
        current = [s.u, s.v];
        uncertainty = s.uncertainty;
        currentReason = null;
      } catch (err) {
        if (err.name === 'MissingCornerError' && this._landAt(k, lat, lon)) {
          onLand = true;
          currentReason = 'on land in the current model';
        } else {
          currentReason = err.name === 'MissingCornerError'
            ? 'within one cell of land, so the current cannot be interpolated here'
            : 'outside the published current grid';
        }
      }
    }

    const u = onLand ? NaN : leeway[0] + (current ? current[0] : 0);
    const v = onLand ? NaN : leeway[1] + (current ? current[1] : 0);
    return {
      lat, lon, frame, u, v, leeway, current, currentReason, uncertainty,
      isPartial: current === null,
      onLand,
      ...this.describe(),
    };
  }

  /**
   * Whether the point itself is on HYCOM's land, not merely beside it.
   *
   * A MissingCornerError says one of the four surrounding grid points is land, which is
   * as true of a point 1 km offshore as of one 100 km inland. The nearest cell tells the
   * two apart. The coastline is HYCOM's, at 0.04 x 0.08 deg (about 4.5 x 8 km), so a
   * point right at the shore can count as land: the model's coastline, not the map's.
   */
  _landAt(k, lat, lon) {
    const cell = this.current.grid.cellAt(lat, lon);
    if (!cell) return false;
    const [u, v] = this.current.source.vector(k, cell.j, cell.i);
    return !Number.isFinite(u) || !Number.isFinite(v);
  }

  /** The wind at a position, bilinear on its own grid, falling back to the cell. */
  _windAt(frame, lat, lon) {
    try {
      const s = sampleField(this.wind, frame, lat, lon);
      return [s.u, s.v];
    } catch {
      const cell = this.grid.cellAt(lat, lon);
      if (!cell) return [NaN, NaN];
      return this.wind.source.vector(frame, cell.j, cell.i);
    }
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
