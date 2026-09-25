/**
 * closeUpKey.js -- what the close-up view shows, said where the legends are (issue #79).
 *
 * Close up, the Surface current key describes nothing on screen (the colour is one flat
 * value there), so it goes and this takes its place in the same stack, built the same
 * way so the corner guard can fold it for room like any legend. It says what each moving
 * thing is, which of them are data and which are drawn for the eye, and -- the one fact
 * the picture itself cannot -- whether it is night there, since the sweep width is a
 * daylight figure.
 */

import L from 'leaflet';

import { FLOWS } from './closeUp.js';

const CHOICES = ['drift', 'current', 'wind', 'off'];

export const CloseUpKey = L.Control.extend({
  options: { position: 'bottomright', onFlow: null },

  onAdd() {
    if (this._root) return this._root;
    const root = L.DomUtil.create('div', 'wind-legend closeup-key');
    L.DomEvent.disableClickPropagation(root);
    L.DomEvent.disableScrollPropagation(root);
    root.innerHTML = `
      <div class="legend-head"><span>Close up</span>
        <button class="legend-toggle" aria-label="Collapse">−</button></div>
      <div class="legend-body ck-body"></div>`;
    const toggle = root.querySelector('.legend-toggle');
    toggle.addEventListener('click', () => {
      const folded = root.classList.toggle('collapsed');
      toggle.textContent = folded ? '+' : '−';
      toggle.setAttribute('aria-label', folded ? 'Expand' : 'Collapse');
    });
    // The streak switch (#85): one flow at a time, so the view stays readable.
    root.addEventListener('click', (e) => {
      const b = e.target.closest('button[data-flow]');
      if (b && this.options.onFlow) this.options.onFlow(b.dataset.flow);
    });
    this._root = root;
    this.update(this._state ?? {});
    return root;
  },

  /**
   * At most four short rows (#83, #85): the streak switch, what moves the sea, whether it is
   * night there, and that none of it is data.
   * @param {object} s
   * @param {string} s.flow              the flow streaked: 'drift', 'current', 'wind' or 'off'
   * @param {number|null} s.force        Beaufort force where the view is
   * @param {number} s.whitecaps         the share of the sea drawn white
   * @param {number|null} s.sunElevationDeg
   * @param {string|null} s.sighting     the animal in view, if any
   */
  update(s) {
    const sig = JSON.stringify(s);
    if (sig === this._sig && this._root && this._root.querySelector('.ck-body').childElementCount) return;
    this._sig = sig;
    this._state = s;
    if (!this._root) return;
    const flow = s.flow ?? 'drift';
    const pills = CHOICES.map((k) => {
      const f = FLOWS[k];
      const dot = f ? `<i style="background:rgb(${f.rgb.join(',')})"></i>` : '';
      return `<button type="button" data-flow="${k}" class="${k === flow ? 'on' : ''}" `
        + `title="${f ? f.tip : 'No streaks'}">${dot}${f ? f.label : 'Off'}</button>`;
    }).join('');
    const wind = s.force === null || s.force === undefined ? 'the wind' : `the wind, force ${s.force}`;
    const rows = [
      `<span class="ck-flow">Streaks <span class="ck-pills">${pills}</span></span>`,
      `<span>The sea moves with the current; its waves${s.whitecaps > 0 ? ' and whitecaps' : ''} follow ${wind}; `
        + 'golden weed, close in, drifts like a person.</span>',
    ];
    if (Number.isFinite(s.sunElevationDeg) && s.sunElevationDeg < -6) {
      rows.push('<span class="ck-night">Night here: the sweep width assumes daylight, so finding is easier than for real.</span>');
    }
    rows.push('<span class="ck-foot">One model cell spans this view, so streaks run parallel. None of this affects '
      + `detection.${s.sighting ? ` Now: ${s.sighting}.` : ''}</span>`);
    this._root.querySelector('.ck-body').innerHTML = rows.join('');
  },
});
