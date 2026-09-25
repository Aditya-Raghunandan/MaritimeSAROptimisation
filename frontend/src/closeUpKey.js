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
   * At most four short rows (#83): what moves the sea, what the golden weed is, whether it
   * is night there, and that none of it is data.
   * @param {object} s
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
    const wind = s.force === null || s.force === undefined ? 'the wind' : `the wind, force ${s.force}`;
    const rows = [
      `<span><i class="ck-sw ck-water"></i>The sea moves with the real current; its waves${s.whitecaps > 0 ? ' and whitecaps' : ''} follow ${wind}.</span>`,
      '<span><i class="ck-sw ck-weed"></i>Golden weed, close in, drifts as the model drifts a person.</span>',
    ];
    if (Number.isFinite(s.sunElevationDeg) && s.sunElevationDeg < -6) {
      rows.push('<span class="ck-night">Night here: the sweep width assumes daylight, so finding is easier than for real.</span>');
    }
    rows.push(`<span class="ck-foot">Drawn for the eye; never affects detection.${s.sighting ? ` Now: ${s.sighting}.` : ''}</span>`);
    this._root.querySelector('.ck-body').innerHTML = rows.join('');
  },
});
