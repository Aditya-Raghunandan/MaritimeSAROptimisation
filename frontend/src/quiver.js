/**
 * quiver.js -- our own canvas renderer for a u/v field on a Leaflet map.
 *
 * WHY WE OWN THIS. D021 chose `leaflet-velocity` with the fallback written in
 * advance: "unmaintained; fallback is a custom canvas quiver, ~80 lines, which
 * we own." It fought us, so this is that fallback. Two faults, both in the
 * library and neither fixable from outside it (1.9.2, `dist/leaflet-velocity.js`):
 *
 *   1. LONGITUDE WRAPPING. Line 641 indexes the grid with
 *      `floorMod(λ - λ0, 360) / Δλ`, and it is fed `map.getBounds()` raw.
 *      Leaflet does not normalise longitude once the world repeats, so at low
 *      zoom it is asked about λ = 278 or λ = -442; those wrap modulo 360 back
 *      into our -82..-63 window and it paints phantom copies of the box across
 *      the Pacific. Observed 2026-09-18.
 *
 *   2. A STALE CANVAS. `_startWindy` is debounced 750 ms and rebuilds the whole
 *      field from scratch on every move, so between the move and the rebuild
 *      the previous painting sits at the new screen position. That is the
 *      smearing while panning, and the box landing over the Amazon.
 *
 * THIS RENDERER CANNOT HAVE EITHER BUG, BY CONSTRUCTION.
 *
 *   - It iterates the DATA GRID and projects each cell to the screen, rather
 *     than iterating the screen and asking the grid what is there. A cell has
 *     exactly one latitude and longitude, so it is drawn exactly once, in the
 *     right place. No modulo, no wrapping, nothing to get wrong at low zoom.
 *   - It redraws SYNCHRONOUSLY on every `move`, from the current projection.
 *     There is no debounce and no cached image, so there is no window in which
 *     what is on screen disagrees with where the map is.
 *
 * COLOUR. One hue, light to dark, which is what a magnitude encoding takes --
 * the old library's white-yellow-orange-red rainbow is the thing to avoid.
 * Orange rather than the default blue because the default basemap is Esri
 * Ocean and a blue ramp on blue bathymetry is invisible; orange is both the
 * documented second sequential hue and the complement of the surface.
 * Anchored on the documented categorical orange `#eb6834`, and checked on the
 * gate that applies to a sequential ramp -- monotonic lightness (OKLab L
 * 0.933 -> 0.446, strictly decreasing) over a 14.9 deg hue spread.
 *
 * Speed is encoded TWICE, as colour and as arrow length. That redundancy is
 * deliberate: it survives colour-blindness, greyscale printing in the report,
 * and the basemap showing through.
 */

import L from 'leaflet';

import { MAX_ARROW_PX, arrowLength, decimation, speedColour } from './style.js';

/** Drawn under every arrow so it stays legible over any basemap tile. */
const OUTLINE = 'rgba(20, 20, 20, 0.55)';

export const QuiverLayer = L.Layer.extend({
  /**
   * @param {object} field  anything with .grid (a Grid) and .vector(frame,j,i)
   * @param {object} opts   { maxSpeed }
   */
  initialize(field, opts = {}) {
    this._field = field;
    this._frame = 0;
    this._maxSpeed = opts.maxSpeed ?? 20;
  },

  onAdd(map) {
    this._map = map;
    this._canvas = L.DomUtil.create('canvas', 'leaflet-quiver-layer leaflet-layer');
    // leaflet-zoom-hide makes Leaflet hide this during the zoom ANIMATION.
    // That is the honest behaviour: mid-animation the map is mid-transform and
    // any projection we computed is already wrong. Hiding for ~250 ms and
    // redrawing correctly on zoomend is what "clean" looks like; the
    // alternative is the smear the previous library produced.
    L.DomUtil.addClass(this._canvas, 'leaflet-zoom-hide');
    map.getPanes().overlayPane.appendChild(this._canvas);

    map.on('move', this._reset, this);
    map.on('zoomend viewreset resize', this._reset, this);
    this._reset();
    return this;
  },

  onRemove(map) {
    map.off('move', this._reset, this);
    map.off('zoomend viewreset resize', this._reset, this);
    if (this._canvas && this._canvas.parentNode) {
      this._canvas.parentNode.removeChild(this._canvas);
    }
    this._canvas = null;
    return this;
  },

  /** Show a different timestep. */
  setFrame(frame) {
    this._frame = frame;
    this._draw();
  },

  setMaxSpeed(maxSpeed) {
    this._maxSpeed = maxSpeed;
    this._draw();
  },

  /** Re-anchor the canvas to the viewport, then repaint. Called on every move. */
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
    // The canvas covers the viewport, so it is pinned to the viewport's
    // top-left expressed in layer coordinates. Recomputed every move, which is
    // why it cannot drift.
    L.DomUtil.setPosition(this._canvas, this._map.containerPointToLayerPoint([0, 0]));
    this._draw();
  },

  /** Screen distance between two adjacent grid cells, for the decimation. */
  _cellSpacingPx() {
    const g = this._field.grid;
    const a = this._map.latLngToContainerPoint([g.lat(0), g.lon(0)]);
    const b = this._map.latLngToContainerPoint([g.lat(0), g.lon(1)]);
    return Math.abs(b.x - a.x);
  },

  _draw() {
    if (!this._map || !this._canvas) return;
    const ctx = this._canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const size = this._map.getSize();

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.x, size.y);

    const g = this._field.grid;
    const step = decimation(this._cellSpacingPx());
    const pad = MAX_ARROW_PX;

    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    for (let j = 0; j < g.nlat; j += step) {
      for (let i = 0; i < g.nlon; i += step) {
        // The projection is of a real cell coordinate, so it lands where the
        // cell is. Nothing here can produce a copy somewhere else.
        const p = this._map.latLngToContainerPoint([g.lat(j), g.lon(i)]);
        if (p.x < -pad || p.y < -pad || p.x > size.x + pad || p.y > size.y + pad) continue;

        const [u, v] = this._field.vector(this._frame, j, i);
        if (!Number.isFinite(u) || !Number.isFinite(v)) continue;

        const speed = Math.hypot(u, v);
        if (speed < 1e-6) continue;

        const len = arrowLength(speed, this._maxSpeed, MAX_ARROW_PX);
        // Screen y grows downward while northward v grows upward, so v is
        // negated here. Getting this wrong mirrors the whole field about the
        // horizontal and still looks entirely plausible.
        const dx = (u / speed) * len;
        const dy = -(v / speed) * len;

        this._arrow(ctx, p.x - dx / 2, p.y - dy / 2, dx, dy,
                    speedColour(speed, this._maxSpeed));
      }
    }
  },

  /** One arrow: a dark outline pass, then the coloured pass on top of it. */
  _arrow(ctx, x, y, dx, dy, colour) {
    const hx = x + dx;
    const hy = y + dy;
    const head = Math.min(5, Math.hypot(dx, dy) * 0.5);
    const ang = Math.atan2(dy, dx);
    const spread = 0.45;

    for (const [stroke, width] of [[OUTLINE, 3.2], [colour, 1.6]]) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(hx, hy);
      ctx.moveTo(hx, hy);
      ctx.lineTo(hx - head * Math.cos(ang - spread), hy - head * Math.sin(ang - spread));
      ctx.moveTo(hx, hy);
      ctx.lineTo(hx - head * Math.cos(ang + spread), hy - head * Math.sin(ang + spread));
      ctx.stroke();
    }
  },
});

export function quiverLayer(field, opts) {
  return new QuiverLayer(field, opts);
}
