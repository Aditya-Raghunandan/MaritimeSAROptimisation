/**
 * oceanLayer.js -- a moving sea for the close-up view (issue #73).
 *
 * Flying a helicopter over a flat colour gives no sense of motion or of the sea state,
 * so at close zoom the Search view draws a texture on the water. It is DECORATION, and
 * the key says so, but it is driven by the real forcing where the helicopter is, so it
 * shows true things rather than invented ones:
 *
 *   wave marks   short crests lying ACROSS the wind, pinned to the water and carried by
 *                the real surface current -- sped up by the playback rate, so at 45 s
 *                per second a 1 m/s current moves them 45 m each real second
 *   whitecaps    brief white flecks whose number follows the Beaufort force: none in a
 *                calm, "scattered" at force 3, "frequent" at 4, "many" from 5 (the
 *                WMO wording in beaufort.js)
 *
 * It never feeds detection: the sweep width stays the Addendum's calm-water figure, and
 * the compass carries the caveat about wind (Table H-10, limitation L19).
 *
 * Positions are kept in lat/lon so the marks stay on the water while the map follows the
 * helicopter, and are re-seeded when they leave the view.
 */

import L from 'leaflet';

import { beaufort } from './beaufort.js';
import { whitecapCount } from './seaState.js';

/** Below this zoom the texture would be noise, not water. */
export const OCEAN_MIN_ZOOM = 13.5;

const M_PER_DEG = 111320;
const MARKS = 260;

export const OceanLayer = L.Layer.extend({
  initialize() {
    this._marks = [];
    this._caps = [];
    this._current = [0, 0];
    this._wind = [0, 0];
    this._rate = 0;
    this._raf = null;
    this._last = null;
  },

  onAdd(map) {
    this._map = map;
    const pane = map.getPane('oceanPane') || map.createPane('oceanPane');
    pane.style.zIndex = 390;          // over the rasters, under the search's marks
    pane.style.pointerEvents = 'none';
    this._canvas = L.DomUtil.create('canvas', 'ocean-canvas', pane);
    this._ctx = this._canvas.getContext('2d');
    this._reset = () => this._resize();
    map.on('resize zoomend', this._reset);
    this._resize();
    this._start();
  },

  onRemove(map) {
    map.off('resize zoomend', this._reset);
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = null;
    this._canvas.remove();
    this._map = null;
  },

  /** The forcing where the helicopter is, and how fast search time is passing (0 = paused). */
  setConditions({ current, wind, rate }) {
    if (current && current.every(Number.isFinite)) this._current = current;
    if (wind && wind.every(Number.isFinite)) this._wind = wind;
    if (Number.isFinite(rate)) this._rate = rate;
  },

  _resize() {
    const size = this._map.getSize();
    this._canvas.width = size.x;
    this._canvas.height = size.y;
    this._marks = [];
    this._caps = [];
  },

  _seed() {
    const b = this._map.getBounds().pad(0.05);
    return {
      lat: b.getSouth() + Math.random() * (b.getNorth() - b.getSouth()),
      lon: b.getWest() + Math.random() * (b.getEast() - b.getWest()),
      phase: Math.random() * Math.PI * 2,
      size: 0.6 + Math.random() * 0.8,
    };
  },

  _start() {
    const frame = (now) => {
      if (!this._map) return;
      const dt = this._last === null ? 0 : Math.min((now - this._last) / 1000, 0.25);
      this._last = now;
      this._draw(dt);
      this._raf = requestAnimationFrame(frame);
    };
    this._raf = requestAnimationFrame(frame);
  },

  _draw(dt) {
    const map = this._map;
    const ctx = this._ctx;
    // Panes move with the map as it pans; keep the canvas on the container's corner.
    L.DomUtil.setPosition(this._canvas, map.containerPointToLayerPoint([0, 0]));
    ctx.clearRect(0, 0, this._canvas.width, this._canvas.height);
    const zoom = map.getZoom();
    if (zoom < OCEAN_MIN_ZOOM) return;
    const fade = Math.min(1, (zoom - OCEAN_MIN_ZOOM) / 1);

    const bounds = map.getBounds().pad(0.08);
    const lat0 = map.getCenter().lat;
    const cos = Math.cos((lat0 * Math.PI) / 180);
    // Search seconds this frame: the texture moves with search time, not wall time.
    const simS = dt * this._rate;
    const dLat = (this._current[1] * simS) / M_PER_DEG;
    const dLon = (this._current[0] * simS) / (M_PER_DEG * cos);

    const wind = Math.hypot(this._wind[0], this._wind[1]);
    const force = beaufort(wind).force;
    // Crests lie across the wind: a line perpendicular to the direction it blows.
    const along = wind > 0.1 ? Math.atan2(-this._wind[1], this._wind[0]) + Math.PI / 2 : 0;
    const len = 6 + force * 2.2;

    while (this._marks.length < MARKS) this._marks.push(this._seed());
    ctx.lineCap = 'round';
    for (const m of this._marks) {
      m.lat += dLat;
      m.lon += dLon;
      m.phase += dt * 1.6;
      if (!bounds.contains([m.lat, m.lon])) Object.assign(m, this._seed());
      const p = map.latLngToContainerPoint([m.lat, m.lon]);
      // Faint enough to read as texture, strong enough to see over the dimmed current.
      const a = 0.16 + 0.16 * (0.5 + 0.5 * Math.sin(m.phase));
      const half = (len * m.size) / 2;
      const dx = Math.cos(along) * half;
      const dy = Math.sin(along) * half;
      ctx.strokeStyle = `rgba(210, 240, 255, ${(a * fade).toFixed(3)})`;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(p.x - dx, p.y - dy);
      ctx.quadraticCurveTo(p.x + dy * 0.35, p.y - dx * 0.35, p.x + dx, p.y + dy);
      ctx.stroke();
    }

    const want = whitecapCount(force);
    while (this._caps.length < want) this._caps.push({ ...this._seed(), life: Math.random() });
    this._caps.length = Math.min(this._caps.length, want);
    for (const c of this._caps) {
      c.lat += dLat;
      c.lon += dLon;
      c.life += dt * 0.5;
      if (c.life > 1 || !bounds.contains([c.lat, c.lon])) Object.assign(c, this._seed(), { life: 0 });
      const p = map.latLngToContainerPoint([c.lat, c.lon]);
      const a = Math.sin(Math.PI * c.life) * 0.75 * fade;
      ctx.fillStyle = `rgba(255, 255, 255, ${a.toFixed(3)})`;
      ctx.beginPath();
      ctx.ellipse(p.x, p.y, 2.2 * c.size + 1, 1.1 * c.size + 0.6, along, 0, Math.PI * 2);
      ctx.fill();
    }
  },
});
