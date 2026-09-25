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

import { SEARCH_COLOURS, WHY_COLOUR } from './searchLayer.js';

function pct(f) {
  const p = f * 100;
  return p < 1 ? p.toFixed(1) : p.toFixed(0);
}

export const CloseUpKey = L.Control.extend({
  options: { position: 'bottomright' },

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
    this._root = root;
    this.update(this._state ?? {});
    return root;
  },

  /**
   * @param {object} s
   * @param {number|null} s.force        Beaufort force where the view is
   * @param {number} s.whitecaps         the share of the sea drawn white
   * @param {number|null} s.sunElevationDeg
   * @param {boolean} s.arrows           whether the buoy's arrows are drawn
   * @param {string|null} s.sighting     the animal in view, if any
   */
  update(s) {
    this._state = s;
    if (!this._root) return;
    const caps = s.force === null || s.force === undefined
      ? 'whitecaps: set by the wind'
      : s.whitecaps > 0
        ? `whitecaps: the wind, force ${s.force}, about ${pct(s.whitecaps)} % of the sea white`
        : `waves: the wind, force ${s.force}, too light for whitecaps`;
    const rows = [
      '<span><i class="ck-sw ck-water"></i>water: its patches move with the real current</span>',
      `<span><i class="ck-sw ck-caps"></i>${caps}</span>`,
      '<span><i class="ck-sw ck-weed"></i>Sargassum: drifts as the model drifts a person, '
        + 'current + 2 % of wind</span>',
    ];
    if (s.arrows) {
      rows.push('<span><i class="ck-sw ck-arrows"></i>at the buoy, 15 min of motion: '
        + '<b style="color:#fff">model</b> · '
        + `<b style="color:${SEARCH_COLOURS.target}">buoy</b> · `
        + `<b style="color:${WHY_COLOUR}">what the model missed</b></span>`);
    }
    if (Number.isFinite(s.sunElevationDeg) && s.sunElevationDeg < -6) {
      rows.push(`<span class="ck-night">Night here: the sun is ${Math.round(-s.sunElevationDeg)}° below the horizon. `
        + 'The sweep width is a daylight figure, so finding is easier here than it would really be.</span>');
    }
    const now = s.sighting ? ` Now: ${s.sighting}.` : '';
    rows.push(`<span class="ck-foot">Waves, weed and animals are drawn for the eye and never affect `
      + `detection; animals are not to scale.${now}</span>`);
    this._root.querySelector('.ck-body').innerHTML = rows.join('');
  },
});
