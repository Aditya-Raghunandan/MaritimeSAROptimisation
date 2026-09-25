/**
 * northArrow.js -- a small north arrow, always on the map (#75).
 *
 * The Search view's compass (compass.js) is an instrument: it shows the helicopter's
 * heading and the current and wind where it is, and it is there only while a search
 * runs. This is the chart convention instead -- a plain mark in the corner, in every
 * view, saying which way is north. The map never rotates, so it always points up; its
 * job is to say so to someone who has not assumed it.
 *
 * Bottom-left, beside the scale bar: the two things a chart puts in its corner.
 */

import L from 'leaflet';

const ARROW_SVG = `
<svg viewBox="0 0 24 34" width="18" height="26" aria-hidden="true">
  <text x="12" y="8.5" text-anchor="middle" font-size="9" font-weight="700" fill="currentColor">N</text>
  <polygon points="12,11 6.5,31 12,26.5" fill="#ff5f5f"/>
  <polygon points="12,11 17.5,31 12,26.5" fill="#8a8a80"/>
  <polygon points="12,11 6.5,31 12,26.5 17.5,31" fill="none" stroke="#121211" stroke-width=".8" stroke-linejoin="round"/>
</svg>`;

export const NorthArrow = L.Control.extend({
  options: { position: 'bottomleft' },

  onAdd() {
    const el = L.DomUtil.create('div', 'north-arrow');
    el.innerHTML = ARROW_SVG;
    el.title = 'North is up: this map never rotates';
    el.setAttribute('role', 'img');
    el.setAttribute('aria-label', 'North arrow: north is up');
    L.DomEvent.disableClickPropagation(el);
    return el;
  },
});
