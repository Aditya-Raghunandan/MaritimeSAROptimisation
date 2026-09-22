/**
 * raster.js -- speed as a continuous painted field, under everything else.
 *
 * This is the layer that changes what the map looks like. A grid of arrows
 * tells you the value at 5,929 points and leaves the eye to infer the shape
 * between them; a painted field shows the shape directly -- where the jet is,
 * where the calm sits, where a front runs -- which is what anyone actually
 * reads a wind map for. Zoom Earth, nullschool and Windy all lead with it.
 *
 * HOW THE SMOOTHNESS IS PRODUCED, AND WHAT IT IS NOT.
 *
 * The field is drawn into an offscreen canvas at roughly grid resolution --
 * one pixel per cell across, and one row per equal step of MERCATOR Y down --
 * and then blitted to the map scaled up, with the browser's own bilinear
 * filtering doing the interpolation. That is fast (one putImageData and one
 * drawImage per frame, whatever the zoom) and it is exactly what the GPU is
 * for.
 *
 * The rows are spaced in Mercator y rather than in latitude because the blit
 * is a LINEAR stretch and the map is not linear in latitude. With uniform
 * latitude rows the field was painted up to 44 km north of where it belongs in
 * the middle of this box -- pinned correctly at 17 N and 36 N and wrong
 * everywhere between. Measured on the live site against two coastlines; the
 * numbers, and the model that predicts them, are in mercator.js.
 *
 * It is also INTERPOLATION FOR DISPLAY, and that distinction matters enough
 * that the vault has a note on it: "Two things called interpolation - sampling
 * vs inventing". The smooth picture does not mean the data is smooth. ERA5 is
 * 0.25 deg, about 28 km, and nothing between two cells was measured or
 * modelled at any finer scale. Anyone reading a value takes it from the point
 * panel, which does nearest-neighbour onto a real cell and shows which cell it
 * used. The raster is for pattern, never for value -- and the arrows on top
 * still land on real cell centres, so the honest resolution stays visible.
 *
 * LAND IS LEFT TRANSPARENT rather than painted zero. A NaN in the current field
 * is "no sea here", and colouring it as the calmest possible water would put a
 * false dead-calm patch over every coastline.
 */

import L from 'leaflet';

import { VIRIDIS, normaliseSpeed, ramp } from './colormap.js';
import { mercatorRowMap, rowsFor } from './mercator.js';
import { isFrameReady } from './sources.js';

export const RasterLayer = L.Layer.extend({
  /**
   * @param {object} field  anything with .grid and .vector(frame, j, i)
   * @param {object} opts   { maxSpeed, opacity }
   */
  initialize(field, opts = {}) {
    this._field = field;
    this._frame = 0;
    this._maxSpeed = opts.maxSpeed ?? 20;
    // Lower than it was. The raster is the background the streaks and arrows
    // are read against, and at 0.72 it was competing with them rather than
    // sitting behind them -- especially over the satellite basemap, where the
    // sea already carries texture.
    /*
      Lower than it was, because two of these can now be on at once.

      At 0.55 each, wind over current is mud: neither field is readable and the
      basemap is gone too. The painted raster is the BACKGROUND the streaks and
      arrows are read against, so it can afford to be faint -- it carries the
      pattern, not the value.
    */
    this._opacity = opts.opacity ?? 0.42;
    // Per product. Wind is viridis; current is magma, whose near-black low end
    // lets the two thirds of the box under 0.3 m/s recede so the Gulf Stream
    // is the only bright thing on the map.
    this._ramp = opts.ramp ?? VIRIDIS;

    /*
      The vertical mapping for the offscreen buffer, worked out HERE rather
      than in `onAdd` because the grid is already known and `setFrame` can
      legitimately arrive before the layer is on a map -- `main.js` warms the
      first frame before any `addTo`, precisely because of the first-paint race
      that #19 was. Computing this in `onAdd` would make that warm-up throw.

      Its rows are uniform in MERCATOR Y, not in latitude, so there are a few
      more of them than the grid has: 512 against 476 for the current, 82
      against 77 for wind. That is what makes the single stretched `drawImage`
      in `_draw` correct. Columns need no such treatment -- longitude IS linear
      in Web Mercator. See mercator.js for the measurement behind all of it.
    */
    this._rows = rowsFor(field.grid);
    this._rowMap = mercatorRowMap(field.grid, this._rows);
  },

  onAdd(map) {
    this._map = map;
    this._canvas = L.DomUtil.create('canvas', 'leaflet-raster-layer leaflet-layer');
    L.DomUtil.addClass(this._canvas, 'leaflet-zoom-hide');
    this._canvas.style.opacity = String(this._opacity);
    // Under the arrows and the tracks, over the basemap.
    map.getPanes().overlayPane.appendChild(this._canvas);
    this._canvas.style.zIndex = '150';

    /*
      The buffer. Allocated once; only its contents change.

      Its rows are uniform in MERCATOR Y, not in latitude, so there are a few
      more of them than the grid has (512 against 476 for the current). That is
      what makes the single stretched `drawImage` in `_draw` correct -- see
      mercator.js for the measurement that made it necessary. Columns are still
      one per cell, because longitude IS linear in Web Mercator.
    */
    this._src = document.createElement('canvas');
    this._src.width = this._field.grid.nlon;
    this._src.height = this._rows;

    map.on('move', this._reset, this);
    map.on('zoomend viewreset resize', this._reset, this);
    this._reset();
    return this;
  },

  onRemove(map) {
    map.off('move', this._reset, this);
    map.off('zoomend viewreset resize', this._reset, this);
    if (this._canvas && this._canvas.parentNode) this._canvas.parentNode.removeChild(this._canvas);
    this._canvas = null;
    return this;
  },

  setFrame(frame) {
    this._frame = frame;
    this._paintSource();
    this._draw();
  },

  setMaxSpeed(maxSpeed) {
    this._maxSpeed = maxSpeed;
    this._paintSource();
    this._draw();
  },

  _reset() {
    if (!this._map || !this._canvas) return;
    const size = this._map.getSize();
    const dpr = window.devicePixelRatio || 1;
    if (this._canvas.width !== size.x * dpr || this._canvas.height !== size.y * dpr) {
      this._canvas.width = size.x * dpr;
      this._canvas.height = size.y * dpr;
      this._canvas.style.width = `${size.x}px`;
      this._canvas.style.height = `${size.y}px`;
    }
    L.DomUtil.setPosition(this._canvas, this._map.containerPointToLayerPoint([0, 0]));
    this._draw();
  },

  /** Fill the grid-resolution buffer: one pixel per cell. */
  _paintSource() {
    // Nothing is fetched when Leaflet calls onAdd -> _reset -> _draw -> here.
    // Painting anyway would throw out of addTo(map) and unwire the rest of the
    // page; painting zeros would be worse still, because a calm field is a
    // plausible picture. Leave the buffer transparent and come back on the
    // next setFrame, which redraw() issues as soon as the chunk lands.
    if (!isFrameReady(this._field, this._frame)) return;
    const g = this._field.grid;
    const ctx = this._src.getContext('2d');
    const img = ctx.createImageData(g.nlon, this._rows);

    for (let r = 0; r < this._rows; r += 1) {
      /*
        Row 0 is the TOP of the image, which is the NORTH edge, while the grid
        ascends in latitude (D020) -- so the rows are read from the far end.
        Getting that backwards flips the field about the equator and still
        looks like weather.

        The inversion now lives inside `_rowMap`, which also does the part that
        matters: rows are spaced evenly in MERCATOR Y rather than in latitude,
        so the linear stretch in `_draw` lands every row where that latitude
        actually is. Uniform-in-latitude rows put the field up to 44 km north
        of the truth in the middle of this box -- mercator.js has the numbers.
      */
      const row = this._rowMap[r];
      for (let i = 0; i < g.nlon; i += 1) {
        const [u, v] = this._field.vector(this._frame, row, i);
        const at = (r * g.nlon + i) * 4;
        const c = ramp(this._ramp, normaliseSpeed(Math.hypot(u, v), this._maxSpeed));
        if (!c) { img.data[at + 3] = 0; continue; }   // NaN -> transparent, not calm
        img.data[at] = c[0];
        img.data[at + 1] = c[1];
        img.data[at + 2] = c[2];
        img.data[at + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    this._painted = true;
  },

  _draw() {
    if (!this._map || !this._canvas) return;
    if (!this._painted) this._paintSource();

    const ctx = this._canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const size = this._map.getSize();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.x, size.y);

    // Where the grid's outer edges land on screen right now. Half a cell is
    // added on each side because a cell's value belongs to its CENTRE, and the
    // painted square should cover the cell rather than start at its middle.
    const g = this._field.grid;
    const nw = this._map.latLngToContainerPoint([
      g.lat(g.nlat - 1) + g.dlat / 2, g.lon(0) - g.dlon / 2]);
    const se = this._map.latLngToContainerPoint([
      g.lat(0) - g.dlat / 2, g.lon(g.nlon - 1) + g.dlon / 2]);

    ctx.imageSmoothingEnabled = true;      // the bilinear fill, for free
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(this._src, nw.x, nw.y, se.x - nw.x, se.y - nw.y);
  },
});

export function rasterLayer(field, opts) {
  return new RasterLayer(field, opts);
}
