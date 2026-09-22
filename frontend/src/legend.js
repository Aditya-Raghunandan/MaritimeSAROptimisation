/**
 * legend.js -- the key for the wind arrows.
 *
 * Without it the field is decorative: the arrows encode speed twice, in colour
 * and in length, and neither could be read off the map. A viewer had to click a
 * cell to learn that orange meant anything at all.
 *
 * The swatches are drawn with the SAME functions the map uses -- `speedColour`
 * for the colour and the same length formula for the arrow -- so the key cannot
 * drift away from the thing it is describing. A legend maintained separately
 * from its renderer is a legend that eventually lies.
 *
 * Beaufort names rather than only numbers, for the reason beaufort.js exists:
 * "Force 6, strong breeze" is a sea somebody can picture, and picturing the sea
 * is how a reader judges whether the search is plausible.
 */

import L from 'leaflet';

import { beaufort } from './beaufort.js';
import { currentBand, driftBand } from './drift.js';
import {
  CURRENT_RAMP, DRIFT_RAMP, MAX_ARROW_PX, SPEED_RAMP, arrowLength, speedColour,
} from './style.js';
import { MAGMA, VIRIDIS, normaliseSpeed, rampCss } from './colormap.js';

/**
 * Speeds to key, derived from the scale rather than fixed.
 *
 * Fixed samples let the key advertise 22 m/s while the map's scale topped out
 * at 15, so the last three rows drew identically and promised a distinction
 * the map could not make. Even fractions of maxSpeed always land on the scale.
 */
function samplesFor(maxSpeed) {
  return [0.12, 0.3, 0.5, 0.7, 0.85, 1].map((f) => Math.round(f * maxSpeed * 10) / 10);
}

export const WindLegend = L.Control.extend({
  options: { position: 'bottomright' },

  initialize(opts = {}) {
    L.Util.setOptions(this, opts);
    this._maxSpeed = opts.maxSpeed ?? 20;
    this._maxArrowPx = opts.maxArrowPx ?? MAX_ARROW_PX;
    // Everything a legend needs to describe a field is now an option, because
    // there are two fields and one of them is not wind. A legend that says
    // "10 m wind" over a current ramp is worse than no legend.
    this._title = opts.title ?? '10 m wind';
    this._ramp = opts.ramp ?? VIRIDIS;
    this._arrowRamp = opts.arrowRamp ?? SPEED_RAMP;
    this._weight = opts.weight ?? 1;
    // A bar keys a PAINTED field. The resultant is drawn only as arrows and
    // streaks, so showing one would explain something that is not on the map.
    this._showBar = opts.showBar ?? true;
    this._foot = opts.foot
      ?? 'bar: painted speed · arrows: length and colour, at real cell centres';
    this._describe = opts.describe ?? ((v) => {
      const b = beaufort(v);
      return `F${b.force} ${b.name.toLowerCase()}`;
    });
  },

  onAdd() {
    const box = L.DomUtil.create('div', 'wind-legend');
    // Otherwise dragging across the key pans the map underneath it.
    L.DomEvent.disableClickPropagation(box);
    L.DomEvent.disableScrollPropagation(box);

    const head = L.DomUtil.create('div', 'legend-head', box);
    head.innerHTML = `<span>${this._title}</span><button class="legend-toggle" `
      + 'aria-label="Collapse">−</button>';

    const body = L.DomUtil.create('div', 'legend-body', box);

    // The colour bar for the painted field. It is the dominant thing on the
    // map now, so keying the arrows alone would leave most of the picture
    // unexplained. Built from the same normaliseSpeed + viridis the raster
    // uses, so it cannot describe a different scale from the one on screen.
    if (this._showBar) {
      const barWrap = L.DomUtil.create('div', 'legend-bar-wrap', body);
      const stops = [];
      for (let k = 0; k <= 10; k += 1) {
        stops.push(`${rampCss(this._ramp, normaliseSpeed((k / 10) * this._maxSpeed, this._maxSpeed))} ${k * 10}%`);
      }
      barWrap.innerHTML =
        `<div class="legend-bar" style="background:linear-gradient(to right,${stops.join(',')})"></div>`
        + `<div class="legend-bar-ends"><span>0</span>`
        + `<span>${Math.round(this._maxSpeed)} m/s</span></div>`;
    }

    for (const s of samplesFor(this._maxSpeed)) {
      const row = L.DomUtil.create('div', 'legend-row', body);
      row.innerHTML =
        `<canvas width="46" height="14"></canvas>`
        + `<span class="legend-speed">${s}</span>`
        + `<span class="legend-name">${this._describe(s)}</span>`;
      this._arrow(row.querySelector('canvas'), s);
    }
    const foot = L.DomUtil.create('div', 'legend-foot', body);
    foot.textContent = this._foot;

    const toggle = head.querySelector('.legend-toggle');
    toggle.addEventListener('click', () => {
      const hidden = box.classList.toggle('collapsed');
      toggle.textContent = hidden ? '+' : '−';
      toggle.setAttribute('aria-label', hidden ? 'Expand' : 'Collapse');
    });

    this._box = box;
    return box;
  },

  /** Same colour and the same length rule as quiver.js, on a 46 px strip. */
  _arrow(canvas, speed) {
    const ctx = canvas.getContext('2d');
    const len = arrowLength(speed, this._maxSpeed, this._maxArrowPx);
    const y = 7;
    const x0 = 2;
    const x1 = x0 + len;

    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    for (const [stroke, width] of [['rgba(20,20,20,0.55)', 3.2 * this._weight],
                                   [speedColour(speed, this._maxSpeed, this._arrowRamp),
                                    1.6 * this._weight]]) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(x0, y);
      ctx.lineTo(x1, y);
      ctx.moveTo(x1, y);
      ctx.lineTo(x1 - 4, y - 2.6);
      ctx.moveTo(x1, y);
      ctx.lineTo(x1 - 4, y + 2.6);
      ctx.stroke();
    }
  },
});

export function windLegend(opts) {
  return new WindLegend(opts);
}

/**
 * The legend for the RESULTANT: green arrows, drift bands, its own scale.
 *
 * It exists because the alternative was actively misleading rather than merely
 * missing. The Drift view draws resultant arrows scaled by `resultantScale`,
 * which tops out near 2.5 m/s, and the only key on screen was the wind's,
 * keyed to 25 m/s. Anyone reading arrow length off it was wrong by a factor of
 * ten, in the one view this project exists to produce.
 *
 * No colour bar, and that is deliberate: the resultant is drawn as arrows and
 * streaks over whatever raster is underneath, so it never paints a field of its
 * own. A bar would key a thing that is not on the map. `showBar: false` is
 * honoured by the control for exactly this case.
 */
export function resultantLegend(opts = {}) {
  return new WindLegend({
    title: 'Drift — where a person goes',
    arrowRamp: DRIFT_RAMP,
    weight: 1.7,
    describe: driftBand,
    maxSpeed: 2.5,
    showBar: false,
    foot: 'current + leeway, at wind cell centres · η is per-particle and cannot be drawn',
    position: 'bottomright',
    ...opts,
  });
}

/** The legend for the surface current: magma bar, cyan arrows, water bands. */
export function currentLegend(opts = {}) {
  return new WindLegend({
    title: 'Surface current',
    ramp: MAGMA,
    arrowRamp: CURRENT_RAMP,
    weight: 1.45,
    describe: currentBand,
    maxSpeed: 2.5,
    position: 'bottomright',
    ...opts,
  });
}
