/**
 * probabilityLayer.js -- the probability map, draining as the helicopter searches (D031).
 *
 * WHAT IS DRAWN. What is LEFT of the probability: every particle's remaining weight, binned
 * into 250 m cells (the episode map's cell, D007) in the frame of the datum marker, which
 * drifts with the current. A cell's colour is its share of the probability on the viridis
 * ramp the rest of the site uses; empty sea is not painted at all. As the strip passes, the
 * particles it reaches lose their weight and their cells go dark: the map drains.
 *
 * ONE SCALE FOR THE WHOLE SEARCH. Colours are scaled to the fullest cell at the moment the
 * helicopter arrives, and never rescaled, so a drained map looks drained instead of being
 * stretched back to full brightness.
 *
 * Positions are in the store convention for the arithmetic (referee.js) and converted to
 * Leaflet's -180..180 only to place a cell on screen.
 */

import L from 'leaflet';
import { viridisCss } from './colormap.js';
import { offsetPosition, relativeM, trackAt } from './referee.js';

export const CELL_M = 250;

const display = (lon) => (lon > 180 ? lon - 360 : lon);

/** Remaining probability per cell about the marker: Map "ix,iy" -> mass. */
export function binCells(lat, lon, weight, mlat, mlon, cellM = CELL_M) {
  const cells = new Map();
  for (let j = 0; j < weight.length; j += 1) {
    const w = weight[j];
    if (w > 0 && Number.isFinite(lat[j])) {
      const [e, n] = relativeM(lat[j], lon[j], mlat, mlon);
      const key = `${Math.floor(e / cellM)},${Math.floor(n / cellM)}`;
      cells.set(key, (cells.get(key) ?? 0) + w);
    }
  }
  return cells;
}

export const ProbabilityLayer = L.Layer.extend({
  initialize({ cellM = CELL_M, opacity = 0.8 } = {}) {
    this.cellM = cellM;
    this.opacity = opacity;
    this.state = null;
    this.scale = null;
  },

  onAdd(map) {
    this.map = map;
    this.canvas = L.DomUtil.create('canvas', 'leaflet-zoom-hide probability-layer');
    this.canvas.style.pointerEvents = 'none';
    map.getPanes().overlayPane.appendChild(this.canvas);
    map.on('moveend zoomend resize', this.redraw, this);
    this.redraw();
  },

  onRemove(map) {
    map.off('moveend zoomend resize', this.redraw, this);
    this.canvas.remove();
    this.map = null;
  },

  /**
   * What to draw: {lat, lon, weight} particles (store longitude), and the marker's track
   * and time, which fix the cells. `resetScale` takes the colour scale from this state.
   */
  setState(state, { resetScale = false } = {}) {
    this.state = state;
    if (resetScale || this.scale === null) {
      const [mlat, mlon] = trackAt(state.marker, state.t);
      const cells = binCells(state.lat, state.lon, state.weight, mlat, mlon, this.cellM);
      this.scale = Math.max(1e-12, ...cells.values());
    }
    this.redraw();
  },

  clear() {
    this.state = null;
    this.scale = null;
    this.redraw();
  },

  redraw() {
    if (!this.map) return;
    const size = this.map.getSize();
    const ratio = window.devicePixelRatio || 1;
    const canvas = this.canvas;
    canvas.width = size.x * ratio;
    canvas.height = size.y * ratio;
    canvas.style.width = `${size.x}px`;
    canvas.style.height = `${size.y}px`;
    L.DomUtil.setPosition(canvas, this.map.containerPointToLayerPoint([0, 0]));
    const ctx = canvas.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, size.x, size.y);
    if (!this.state) return;

    const { lat, lon, weight, marker, t } = this.state;
    const [mlat, mlon] = trackAt(marker, t);
    const cells = binCells(lat, lon, weight, mlat, mlon, this.cellM);
    const c = this.cellM;
    ctx.globalAlpha = this.opacity;
    for (const [key, mass] of cells) {
      const [ix, iy] = key.split(',').map(Number);
      const [la0, lo0] = offsetPosition(mlat, mlon, ix * c, iy * c);
      const [la1, lo1] = offsetPosition(mlat, mlon, (ix + 1) * c, (iy + 1) * c);
      const p0 = this.map.latLngToContainerPoint([la0, display(lo0)]);
      const p1 = this.map.latLngToContainerPoint([la1, display(lo1)]);
      const share = Math.min(1, mass / this.scale);
      ctx.fillStyle = viridisCss(0.15 + 0.85 * Math.sqrt(share));
      ctx.fillRect(Math.min(p0.x, p1.x), Math.min(p0.y, p1.y),
        Math.max(1, Math.abs(p1.x - p0.x)), Math.max(1, Math.abs(p1.y - p0.y)));
    }
    ctx.globalAlpha = 1;
  },
});
