/**
 * fieldLayer.js -- the drift a person in the water feels, as arrows (D031, the game's hard mode).
 *
 * In "currents only" mode the player is not shown the probability map, only what a pilot
 * could read off a chart: the water's current plus 2 % of the wind (the leeway the drift
 * model gives a person, D002), on the scenario bundle's 21 x 21 grid, 2 km apart about the
 * datum, every 5 minutes. Arrows point where the water would carry a person; their length
 * and colour are the speed.
 */

import L from 'leaflet';
import { offsetPosition } from './referee.js';
import { DRIFT_RAMP } from './style.js';

export const LEEWAY = 0.02;
const MAX_SPEED_MS = 1.5;
const MAX_PX = 34;

const display = (lon) => (lon > 180 ? lon - 360 : lon);

/** The resultant (u, v) at a grid point, linear in time between the grid's frames. */
export function resultantAt(field, row, col, t) {
  const times = field.times;
  const i = Math.max(0, Math.min(times.length - 2, Math.floor(t / (times[1] - times[0]))));
  const f = Math.max(0, Math.min(1, (t - times[i]) / (times[i + 1] - times[i])));
  const a = field.at(i, row, col);
  const b = field.at(i + 1, row, col);
  const mix = (k) => a[k] + f * (b[k] - a[k]);
  return [mix(0) + LEEWAY * mix(2), mix(1) + LEEWAY * mix(3)];
}

export const FieldLayer = L.Layer.extend({
  initialize() {
    this.state = null;
  },

  onAdd(map) {
    this.map = map;
    this.canvas = L.DomUtil.create('canvas', 'leaflet-zoom-hide field-layer');
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

  /** {field, centre: [lat, lon] (store longitude), t} */
  setState(state) {
    this.state = state;
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

    const { field, centre, t } = this.state;
    const k = (field.size - 1) / 2;
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    for (let row = 0; row < field.size; row += 1) {
      for (let col = 0; col < field.size; col += 1) {
        const [u, v] = resultantAt(field, row, col, t);
        const speed = Math.hypot(u, v);
        if (!Number.isFinite(speed)) continue;
        const [lat, lon] = offsetPosition(centre[0], centre[1], (col - k) * field.stepM, (row - k) * field.stepM);
        const p = this.map.latLngToContainerPoint([lat, display(lon)]);
        const len = Math.max(4, (Math.min(speed, MAX_SPEED_MS) / MAX_SPEED_MS) * MAX_PX);
        const dx = (u / (speed || 1)) * len;
        const dy = -(v / (speed || 1)) * len;
        const shade = DRIFT_RAMP[Math.min(DRIFT_RAMP.length - 1, Math.floor((speed / MAX_SPEED_MS) * DRIFT_RAMP.length))];
        ctx.strokeStyle = shade;
        ctx.beginPath();
        ctx.moveTo(p.x - dx / 2, p.y - dy / 2);
        ctx.lineTo(p.x + dx / 2, p.y + dy / 2);
        const ang = Math.atan2(dy, dx);
        ctx.moveTo(p.x + dx / 2, p.y + dy / 2);
        ctx.lineTo(p.x + dx / 2 - 6 * Math.cos(ang - 0.5), p.y + dy / 2 - 6 * Math.sin(ang - 0.5));
        ctx.moveTo(p.x + dx / 2, p.y + dy / 2);
        ctx.lineTo(p.x + dx / 2 - 6 * Math.cos(ang + 0.5), p.y + dy / 2 - 6 * Math.sin(ang + 0.5));
        ctx.stroke();
      }
    }
  },
});
