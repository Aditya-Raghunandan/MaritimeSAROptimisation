/**
 * mercator.js -- the vertical mapping between a latitude grid and the screen.
 *
 * WHY THIS FILE EXISTS. The raster is drawn as one `drawImage` stretched
 * between the projected north and south edges of the grid. `drawImage` scales
 * LINEARLY IN PIXELS; Leaflet's map is Web Mercator, in which latitude is not
 * linear in pixels. So pinning the two edges correctly is not enough -- it
 * leaves everything between them displaced, by zero at the edges and by a
 * maximum near the middle.
 *
 * MEASURED, on the live site, 2026-09-21, over the study box 17-36 N:
 *
 *   coastline                 data edge   rendered   displacement
 *   Cuba north coast, 80 W      23.060     23.412       +0.352 deg
 *   Hispaniola south, 71 W      18.220     18.325       +0.105 deg
 *
 * and the model below predicts +0.337 and +0.090 for those two, so the cause
 * is settled rather than guessed. The worst case is +0.40 deg -- about 44 km --
 * at 26.5 N, which is the middle of the box, the Gulf Stream, and the subject
 * of the entire project.
 *
 * It was invisible for as long as the only raster was wind, because WIND HAS NO
 * LAND MASK: a smooth field over land and sea alike offers nothing to notice a
 * displacement against. The current has coastlines in it, so the error became
 * visible the day the current layer landed, having been there all along.
 *
 * THE FIX IS TO MOVE THE PROBLEM INTO THE SOURCE IMAGE. Rather than correcting
 * the draw -- one `drawImage` per row would be hundreds of calls per repaint
 * and would seam -- the offscreen buffer is built with its rows uniform in
 * MERCATOR Y instead of uniform in latitude. A linear stretch of a
 * Mercator-uniform image onto a Mercator axis is exact, so `_draw` needs no
 * knowledge of any of this, and the mapping depends only on the grid, not on
 * the view, so it is computed once and cached.
 *
 * Everything here is pure arithmetic and imports nothing, so it is tested
 * without a DOM -- the same reason `geo.js` is its own file.
 */

const DEG = Math.PI / 180;

/**
 * The Web Mercator y of a latitude, in the projection's own units.
 *
 * Only the RATIOS of differences matter here, so the radius factor Leaflet
 * applies is deliberately left out; putting it in would change nothing and
 * invite the reader to think this is Leaflet's projection rather than the
 * shape of it.
 */
export function mercatorY(latDeg) {
  return Math.log(Math.tan(Math.PI / 4 + (latDeg * DEG) / 2));
}

/** The inverse: the latitude at a Mercator y. */
export function mercatorLat(y) {
  return (2 * (Math.atan(Math.exp(y)) - Math.PI / 4)) / DEG;
}

/**
 * How many image rows the buffer needs so that no grid row is skipped.
 *
 * Sampling uniformly in y walks latitude in steps of `dy * cos(lat)`, which is
 * LARGEST at the lowest latitude in the box. Fewer rows than this and whole
 * rows of data are stepped over -- which on a coastline means a piece of it
 * simply is not drawn, and that is exactly the kind of plausible-looking
 * omission this project keeps finding.
 *
 * The result is a little larger than `nlat` (512 against 476 for the current
 * grid, 82 against 77 for wind), because Mercator stretches the north and the
 * buffer has to afford the south the resolution it needs.
 */
export function rowsFor(grid) {
  const south = grid.lat0 - grid.dlat / 2;
  const north = grid.lat0 + (grid.nlat - 1) * grid.dlat + grid.dlat / 2;
  const span = mercatorY(north) - mercatorY(south);
  const lowest = Math.min(Math.abs(south), Math.abs(north));
  const needed = Math.ceil((span * Math.cos(lowest * DEG)) / (grid.dlat * DEG));
  // Never fewer rows than the grid has, and never absurdly many: 4096 rows of
  // a 500-wide field is 8 MB of buffer, which is the point to stop.
  return Math.max(grid.nlat, Math.min(needed, 4096));
}

/**
 * Image row -> grid latitude index, for a buffer whose rows are uniform in
 * Mercator y. Row 0 is the TOP of the image, which is the NORTH edge.
 *
 * `+ 0.5` samples the middle of each row's slab rather than its upper edge.
 * Without it the whole image is half a row too far north -- a small version of
 * the very bug this file exists to remove.
 *
 * Indices are clamped rather than left out of range: the first and last slabs
 * straddle the grid's outer half-cells, where the nearest real row is the edge
 * row and there is nothing better to use.
 */
export function mercatorRowMap(grid, rows) {
  const south = grid.lat0 - grid.dlat / 2;
  const north = grid.lat0 + (grid.nlat - 1) * grid.dlat + grid.dlat / 2;
  const yNorth = mercatorY(north);
  const ySouth = mercatorY(south);

  const map = new Int32Array(rows);
  for (let r = 0; r < rows; r += 1) {
    const y = yNorth - ((r + 0.5) / rows) * (yNorth - ySouth);
    const lat = mercatorLat(y);
    const j = Math.round((lat - grid.lat0) / grid.dlat);
    map[r] = Math.max(0, Math.min(grid.nlat - 1, j));
  }
  return map;
}

/**
 * The displacement the uncorrected draw produces at a latitude, in degrees.
 *
 * Not used by the renderer. It is here because it is the measurement that
 * justifies the rest of the file, and a number that can be re-derived is worth
 * more than a comment claiming it -- the test pins it against what was measured
 * on the live site.
 */
export function linearStretchError(grid, latDeg) {
  const south = grid.lat0 - grid.dlat / 2;
  const north = grid.lat0 + (grid.nlat - 1) * grid.dlat + grid.dlat / 2;
  const fraction = (north - latDeg) / (north - south);          // where a linear draw puts it
  const y = mercatorY(north) - fraction * (mercatorY(north) - mercatorY(south));
  return mercatorLat(y) - latDeg;                                // where that pixel really is
}
