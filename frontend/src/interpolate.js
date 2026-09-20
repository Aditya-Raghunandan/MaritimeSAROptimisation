/**
 * interpolate.js -- the backend's resultant, computed in the browser (issue #10).
 *
 * A port of `src/sar/model/interpolate.py`. Python is the reference; this is the copy.
 *
 * WHY A COPY AND NOT A FETCH. The published site is static: it is a Vite bundle on
 * GitHub Pages reading a Zarr store on Hugging Face, and there is no server anywhere to
 * ask for a number. The browser already holds the chunk a click lands in, so the honest
 * options were to duplicate the arithmetic or to give up on interpolated point queries.
 * This duplicates it, and pays for that with `fixtures/resultant_golden.json`: Python
 * generates it, `interpolate.test.js` replays every case through this file, and the two
 * must agree to 1e-9. Edit one side alone and that test fails, naming the case. Without
 * the fixture the copies drift apart quietly and a wrong map still looks reasonable.
 *
 * WHAT IT IS FOR. `Grid.cellAt` rounds a click to the nearest grid point, which moves the
 * answer by up to half a cell: about 4 km east-west and 2.2 km north-south on the current
 * grid, about 14 km on the wind grid. That is fine for drawing a field and wrong for a
 * number shown next to a position. This weights the four surrounding grid points instead.
 *
 * INDEX NAMING IS THE OPPOSITE OF PYTHON'S, AND IT IS THE FRONTEND'S THAT WINS HERE. In
 * `layers.js`, `j` indexes LATITUDE and `i` indexes LONGITUDE (`grid.lat(j)`,
 * `grid.lon(i)`). The Python module uses `i` for the latitude row and `j` for the
 * longitude column. Every name below follows the frontend, so a reader of this file is
 * never reading against its neighbours. The fixture stores cells as [j, i], frontend
 * order.
 *
 * LONGITUDE IS DISPLAY LONGITUDE HERE. The manifest hands the browser -180 to 180 (D020
 * converts at the presentation boundary), so nothing in this file wraps to 0-360. Passing
 * a stored 0-360 longitude in would land outside the grid and raise, which is the right
 * failure.
 */

/** Asking for a position or frame the grid does not cover. */
export class OutOfCoverageError extends Error {
  constructor(message) {
    super(message);
    this.name = 'OutOfCoverageError';
  }
}

/** One or more of the four surrounding grid points is land. */
export class MissingCornerError extends Error {
  constructor(message, missing) {
    super(message);
    this.name = 'MissingCornerError';
    this.missing = missing;
  }
}

/** Below this coherence the corners nearly cancel, so a direction spread means nothing. */
export const COHERENCE_FLOOR = 0.1;

/**
 * Index of the nearest grid line on one axis, and the signed offset to it in degrees.
 *
 * Mirrors `nearest_index`. `origin` and `step` are the grid's `lat0`/`dlat` or
 * `lon0`/`dlon`; `n` is how many lines the axis has.
 */
export function nearestIndex(origin, step, n, value, name = 'axis') {
  const last = origin + (n - 1) * step;
  const lo = Math.min(origin, last);
  const hi = Math.max(origin, last);
  // The same 1e-9 slack as the Python EDGE_EPS: a coordinate that arrives as
  // 26.999999999 must still count as the edge it plainly is.
  if (value < lo - 1e-9 || value > hi + 1e-9) {
    throw new OutOfCoverageError(
      `${name} ${value} is outside the grid, which covers ${lo} to ${hi}`,
    );
  }
  const index = Math.min(Math.max(Math.round((value - origin) / step), 0), n - 1);
  return { index, offset: value - (origin + index * step) };
}

/**
 * Which side of the nearest grid point the position is on, one sign per axis.
 *
 * Two axes, so two signs: +1 toward the higher index, -1 toward the lower. Zero counts as
 * +1, and the far corner then carries weight zero, so the choice cannot matter.
 */
export function neighbourDirection(dy, dx) {
  return [dy >= 0 ? 1 : -1, dx >= 0 ? 1 : -1];
}

/**
 * Flip a direction sign inward when it would point off the end of an axis of n lines.
 *
 * Only reachable for a position exactly on the first or last line, where the offset is
 * zero and the far weight is zero, so flipping changes no number.
 */
export function keepInside(index, sign, n) {
  return index + sign >= 0 && index + sign < n ? sign : -sign;
}

/**
 * The four [j, i] cells around a position: nearest, its neighbour in longitude, its
 * neighbour in latitude, then the diagonal. Same order as the weights.
 */
export function cellCorners(j, i, sy, sx, nlat, nlon) {
  const cells = [[j, i], [j, i + sx], [j + sy, i], [j + sy, i + sx]];
  for (const [cj, ci] of cells) {
    if (cj < 0 || cj >= nlat || ci < 0 || ci >= nlon) {
      throw new OutOfCoverageError(
        `the cell around grid point (${j}, ${i}) leaves a grid of ${nlat} by ${nlon}`,
      );
    }
  }
  return cells;
}

/**
 * The four corner weights from the fractional offsets, summing to 1.
 *
 * `fy` and `fx` are |offset| as a fraction of one step, so each is 0 to 0.5. Each weight
 * is the area of the sub-rectangle opposite its corner.
 */
export function bilinearWeights(fy, fx) {
  if (!(fy >= 0 && fy <= 1 && fx >= 0 && fx <= 1)) {
    throw new RangeError(`fractional offsets must be within 0 to 1, got fy=${fy}, fx=${fx}`);
  }
  return [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy];
}

/** The weighted sum of the corner [u, v] pairs, component by component. */
export function resultantVector(corners, weights) {
  let u = 0;
  let v = 0;
  for (let k = 0; k < corners.length; k += 1) {
    u += weights[k] * corners[k][0];
    v += weights[k] * corners[k][1];
  }
  return [u, v];
}

/**
 * Speed in m/s and the compass bearing the vector points TOWARD, in degrees.
 *
 * Clockwise from north, so `atan2(u, v)` and not `atan2(v, u)`. Null for a zero vector,
 * where a bearing does not exist. For wind, add 180 for the meteorological "from".
 */
export function speedDirection(u, v) {
  const speed = Math.hypot(u, v);
  if (speed === 0) return { speed: 0, directionTo: null };
  const deg = (Math.atan2(u, v) * 180) / Math.PI;
  return { speed, directionTo: ((deg % 360) + 360) % 360 };
}

/** Linear blend of corner values between two frames. Mirrors `time_blend`. */
export function timeBlend(cornersT1, cornersT2, frac) {
  if (!(frac >= 0 && frac <= 1)) {
    throw new RangeError(`frac must be within 0 to 1, got ${frac}`);
  }
  return cornersT1.map((c, k) => [
    c[0] + frac * (cornersT2[k][0] - c[0]),
    c[1] + frac * (cornersT2[k][1] - c[1]),
  ]);
}

/**
 * How far the interpolated vector can be trusted, from the four corners alone.
 *
 * These measure variability inside the cell, not a guaranteed bound: see
 * docs/resultant-vector.md. `productSigma` is the source model's own error in m/s, which
 * this project has not measured, so it defaults to 0 rather than to an invented number.
 */
export function interpolationUncertainty(corners, weights, productSigma = 0) {
  const [up, vp] = resultantVector(corners, weights);

  let variance = 0;
  let meanSpeed = 0;
  for (let k = 0; k < corners.length; k += 1) {
    const du = corners[k][0] - up;
    const dv = corners[k][1] - vp;
    variance += weights[k] * (du * du + dv * dv);
    meanSpeed += weights[k] * Math.hypot(corners[k][0], corners[k][1]);
  }
  const sigmaSpatial = Math.sqrt(variance);
  const speedP = Math.hypot(up, vp);
  const coherence = meanSpeed === 0 ? 1 : Math.min(speedP / meanSpeed, 1);

  return {
    sigmaSpatialMs: sigmaSpatial,
    coherence,
    speedLossMs: meanSpeed - speedP,
    directionSpreadDeg: coherence >= COHERENCE_FLOOR
      ? (Math.sqrt(-2 * Math.log(coherence)) * 180) / Math.PI
      : null,
    sigmaTotalMs: Math.hypot(sigmaSpatial, productSigma),
    nCorners: corners.length,
  };
}

/**
 * Locate the four cells around a position on a grid, with their weights.
 *
 * `grid` is a `layers.js` Grid (lat0, dlat, nlat, lon0, dlon, nlon). Pure index
 * arithmetic: no data is read here, so it can be tested without a source.
 */
export function locate(grid, lat, lon) {
  const { index: j, offset: dy } = nearestIndex(grid.lat0, grid.dlat, grid.nlat, lat, 'lat');
  const { index: i, offset: dx } = nearestIndex(grid.lon0, grid.dlon, grid.nlon, lon, 'lon');

  let [sy, sx] = neighbourDirection(dy, dx);
  sy = keepInside(j, sy, grid.nlat);
  sx = keepInside(i, sx, grid.nlon);

  const fy = Math.abs(dy / grid.dlat);
  const fx = Math.abs(dx / grid.dlon);
  return {
    j, i, sy, sx, dy, dx, fy, fx,
    cells: cellCorners(j, i, sy, sx, grid.nlat, grid.nlon),
    weights: bilinearWeights(fy, fx),
  };
}

/**
 * The four corner vectors at one frame, or a MissingCornerError if any is land.
 *
 * `source` is anything with `vector(frame, j, i)`: a BufferSource or a ZarrSource. HYCOM
 * writes land as NaN, and a NaN that is allowed through turns the whole resultant into
 * NaN, which draws as nothing at all and reads on screen as calm water rather than as
 * coast. So it stops here, saying how many corners were missing.
 */
export function cornersAt(source, cells, frame) {
  const corners = cells.map(([cj, ci]) => source.vector(frame, cj, ci));
  const missing = cells.filter((_, k) => !Number.isFinite(corners[k][0])
    || !Number.isFinite(corners[k][1]));
  if (missing.length > 0) {
    const howMany = missing.length === 4 ? 'all 4' : `${missing.length} of 4`;
    throw new MissingCornerError(
      `${howMany} surrounding grid points have no data at frame ${frame}, at [j, i] `
      + `${JSON.stringify(missing)}: the point is within one cell of land.`,
      missing,
    );
  }
  return corners;
}

/**
 * The bilinear resultant of one field at a position and frame.
 *
 * Returns the vector, its speed and bearing, the cells and weights used, and the
 * uncertainty. Throws OutOfCoverageError outside the grid and MissingCornerError on land,
 * exactly as the Python does, rather than returning a plausible wrong number.
 */
export function sampleField({ grid, source }, frame, lat, lon, productSigma = 0) {
  const at = locate(grid, lat, lon);
  const corners = cornersAt(source, at.cells, frame);
  const [u, v] = resultantVector(corners, at.weights);
  const { speed, directionTo } = speedDirection(u, v);
  return {
    u,
    v,
    speed,
    directionTo,
    cells: at.cells,
    weights: at.weights,
    uncertainty: interpolationUncertainty(corners, at.weights, productSigma),
  };
}
