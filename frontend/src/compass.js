/**
 * compass.js -- which way is north, which way the helicopter is going, and what the
 * water and the air are doing where it is (issue #73).
 *
 * Three needles on one rose, because the three together are the search problem in a
 * glance: the helicopter's HEADING (orange), the CURRENT the marker and the buoy ride on
 * (cyan), and the WIND that pushes a person but not a drogued buoy (amber). The current
 * and the wind point the way they are GOING, like everything else on the search map;
 * the wind's text also says where it comes FROM, because that is how wind is spoken of.
 *
 * `compassReadout` (seaState.js) is pure and tested; the control only draws what it returns.
 *
 * Bottom-right, on top of the legends, rose beside its read-out (#76): top-right, under
 * the drifter list, it ran down over the Surface current key. In the same stack as the
 * legends it cannot overlap them, and cornerGuard.js keeps that stack clear of the top.
 */

import L from 'leaflet';

import { compassReadout } from './seaState.js';

const RING = `
  <circle cx="0" cy="0" r="46" fill="rgba(18,18,17,.82)" stroke="#3a3a37" stroke-width="1.5"/>
  ${Array.from({ length: 36 }, (_, k) => {
    const long = k % 9 === 0;
    const a = (k * 10 * Math.PI) / 180;
    const r0 = long ? 36 : 41;
    return `<line x1="${(r0 * Math.sin(a)).toFixed(2)}" y1="${(-r0 * Math.cos(a)).toFixed(2)}"
      x2="${(45 * Math.sin(a)).toFixed(2)}" y2="${(-45 * Math.cos(a)).toFixed(2)}"
      stroke="${long ? '#c3c2b7' : '#6f6f66'}" stroke-width="${long ? 1.6 : 1}"/>`;
  }).join('')}
  <text x="0" y="-26" text-anchor="middle" font-size="11" font-weight="700" fill="#ff5f5f">N</text>
  <text x="27" y="4" text-anchor="middle" font-size="9" fill="#c3c2b7">E</text>
  <text x="0" y="33" text-anchor="middle" font-size="9" fill="#c3c2b7">S</text>
  <text x="-27" y="4" text-anchor="middle" font-size="9" fill="#c3c2b7">W</text>`;

/** An arrow from the centre along +y-up, rotated later; `len` in px, `w` its weight. */
function needle(cls, colour, len, w) {
  return `<g class="${cls}" style="display:none">
    <line x1="0" y1="0" x2="0" y2="${-len}" stroke="${colour}" stroke-width="${w}" stroke-linecap="round"/>
    <polygon points="0,${-len - 6} 4,${-len + 2} -4,${-len + 2}" fill="${colour}"/></g>`;
}

export const Compass = L.Control.extend({
  options: { position: 'bottomright' },

  onAdd() {
    if (this._root) return this._root;
    const root = L.DomUtil.create('div', 'search-compass');
    root.innerHTML = `
      <svg viewBox="-50 -50 100 100" width="84" height="84" aria-hidden="true">
        ${RING}
        ${needle('cp-wind', '#f0b04a', 28, 2.2)}
        ${needle('cp-current', '#48b1e3', 32, 2.6)}
        ${needle('cp-heading', '#ff6b35', 38, 3.2)}
        <circle cx="0" cy="0" r="3" fill="#ffffff"/>
      </svg>
      <div class="cp-text">
        <div class="cp-lines"></div>
        <div class="cp-caveat" hidden></div>
        <div class="cp-key"><span><i style="background:#ff6b35"></i>heading</span>
          <span><i style="background:#48b1e3"></i>current</span>
          <span><i style="background:#f0b04a"></i>wind, going to</span></div>
      </div>`;
    L.DomEvent.disableClickPropagation(root);
    this._root = root;
    this.update({});
    return root;
  },

  /** Redraw from the helicopter's heading and the current and wind at it. */
  update(state) {
    if (!this._root) return;
    const r = compassReadout(state);
    const turn = (cls, deg) => {
      const g = this._root.querySelector(cls);
      g.style.display = deg === null ? 'none' : '';
      if (deg !== null) g.setAttribute('transform', `rotate(${deg.toFixed(1)})`);
    };
    turn('.cp-heading', r.heading);
    turn('.cp-current', r.current ? r.current.towards : null);
    turn('.cp-wind', r.wind ? r.wind.towards : null);
    this._root.querySelector('.cp-lines').innerHTML = r.lines.map((l) => `<div>${l}</div>`).join('');
    const caveat = this._root.querySelector('.cp-caveat');
    caveat.hidden = !r.caveat;
    caveat.textContent = r.caveat ?? '';
  },
});
