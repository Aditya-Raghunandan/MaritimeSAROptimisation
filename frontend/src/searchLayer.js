/**
 * searchLayer.js -- a doctrinal search drawn on the map (issue #64).
 *
 * Everything decided -- where the helicopter is, where the marker is, whether the
 * target was found -- is computed in searchRun.js and tested there. This file draws it:
 *
 *   base        where the helicopter launches from, placed by a click
 *   LKP         the target's last known position at the report time
 *   datum       the LKP carried on by drift to the moment of arrival; the transit line
 *               runs from the base to here
 *   marker      the datum marker buoy, drifting with the current; its path so far
 *   swept strip the path flown so far, drawn at its TRUE width, 185.2 m, re-scaled on
 *               every zoom -- ADR003 and the vault's open-day note both refuse a
 *               strip drawn wider than the physics
 *   helicopter  an icon at the current moment, pointing along its heading
 *   target      where the real buoy actually was at that moment
 */

import L from 'leaflet';

import { SWEEP_WIDTH_M } from './geo.js';
import { helicopterAt, markerPositionAt, searchPath } from './searchRun.js';

/**
 * Rescue orange for the searcher: not the wind's amber, the current's cyan, the drift's
 * green or either drogue colour, so a helicopter never reads as a forcing field.
 */
export const SEARCH_COLOURS = {
  helicopter: '#ff6b35',
  marker: '#c38bff',
  datum: '#ffffff',
  base: '#9ad1ff',
  target: '#ff5fa2',
  found: '#5cff9d',
};

/** Web Mercator metres per screen pixel at a latitude and (fractional) zoom. */
export function metresPerPixel(lat, zoom) {
  return (40075016.686 * Math.cos((lat * Math.PI) / 180)) / (256 * 2 ** zoom);
}

const HELI_SVG = `
<svg viewBox="-12 -12 24 24" width="26" height="26" aria-hidden="true">
  <line x1="-10" y1="-10" x2="10" y2="10" stroke="#121211" stroke-width="3.2"/>
  <line x1="10" y1="-10" x2="-10" y2="10" stroke="#121211" stroke-width="3.2"/>
  <line x1="-10" y1="-10" x2="10" y2="10" stroke="#fff" stroke-width="1.4"/>
  <line x1="10" y1="-10" x2="-10" y2="10" stroke="#fff" stroke-width="1.4"/>
  <ellipse cx="0" cy="-1" rx="3.6" ry="5.4" fill="${SEARCH_COLOURS.helicopter}" stroke="#121211" stroke-width="1"/>
  <rect x="-0.9" y="3.5" width="1.8" height="7" fill="${SEARCH_COLOURS.helicopter}" stroke="#121211" stroke-width=".6"/>
</svg>`;

export const SearchLayer = L.Layer.extend({
  initialize() {
    this._plan = null;
    this._path = null;
    this._result = null;
    this._targetAt = null;
    this._s = 0;
    this._base = null;
    this._lkp = null;
  },

  onAdd(map) {
    this._map = map;
    this._group = L.layerGroup().addTo(map);
    this._restyle = () => this._render();
    map.on('zoomend', this._restyle);
    this._render();
  },

  onRemove(map) {
    map.off('zoomend', this._restyle);
    this._group.remove();
    this._map = null;
  },

  /** A base placed before a plan exists, so the click has visible effect. */
  setBase(base) {
    this._base = base;
    if (this._map) this._render();
  },

  /** The target's last known position, shown as soon as it is chosen. */
  setTarget(lkp) {
    this._lkp = lkp;
    if (this._map) this._render();
  },

  /** Load a planned search. `targetAt(ms)` is where the real target was. */
  setPlan(plan, targetAt, result) {
    this._plan = plan;
    this._path = plan ? searchPath(plan) : null;
    this._targetAt = targetAt;
    this._result = result;
    this._s = 0;
    if (this._map) this._render();
  },

  /** Seconds after the report. */
  setTime(s) {
    this._s = s;
    if (this._map) this._render();
  },

  clear() {
    this.setPlan(null, null, null);
  },

  _render() {
    const g = this._group;
    g.clearLayers();
    const plan = this._plan;
    const base = plan ? plan.base : this._base;
    if (base) {
      L.marker([base.lat, base.lon], {
        icon: L.divIcon({ className: 'search-base', html: 'BASE', iconSize: [38, 16] }),
        interactive: false,
      }).addTo(g);
    }
    const lkp = plan ? plan.lkp : this._lkp;
    if (lkp) {
      L.circleMarker([lkp.lat, lkp.lon], {
        radius: 6, color: SEARCH_COLOURS.target, weight: 1.5, fill: false, dashArray: '2 3',
        className: 'search-lkp', interactive: false,
      }).addTo(g);
    }
    if (!plan) return;

    const s = this._s;
    const { datum } = plan;
    L.polyline([[base.lat, base.lon], [datum.lat, datum.lon]], {
      color: SEARCH_COLOURS.helicopter, weight: 1.2, opacity: 0.7, dashArray: '6 6',
      className: 'search-transit', interactive: false,
    }).addTo(g);
    L.circleMarker([datum.lat, datum.lon], {
      radius: 7, color: SEARCH_COLOURS.datum, weight: 1.5, fill: false, dashArray: '3 3',
      className: 'search-datum', interactive: false,
    }).addTo(g);

    // The marker's drift so far, then the marker itself.
    if (s >= plan.arriveS) {
      const m = plan.marker;
      const upto = Math.min(s, plan.endS) - plan.arriveS;
      const pts = [];
      for (let k = 0; k < m.tS.length && m.tS[k] <= upto; k += 1) pts.push([m.lat[k], m.lon[k]]);
      const now = markerPositionAt(plan, s);
      if (now) pts.push([now.lat, now.lon]);
      L.polyline(pts, { color: SEARCH_COLOURS.marker, weight: 1.5, opacity: 0.8, interactive: false }).addTo(g);
      if (now) {
        L.circleMarker([now.lat, now.lon], {
          radius: 4, color: '#121211', weight: 1, fillColor: SEARCH_COLOURS.marker, fillOpacity: 1,
          className: 'search-marker', interactive: false,
        }).addTo(g);
      }
    }

    // The swept strip and the trail, up to now.
    const flown = this._flownUpTo(s);
    if (flown.length > 1) {
      const zoom = this._map.getZoom();
      const px = SWEEP_WIDTH_M / metresPerPixel(datum.lat, zoom);
      L.polyline(flown, {
        color: SEARCH_COLOURS.helicopter, weight: Math.max(px, 1), opacity: 0.28,
        lineCap: 'butt', lineJoin: 'miter', className: 'search-strip', interactive: false,
      }).addTo(g);
      L.polyline(flown, {
        color: SEARCH_COLOURS.helicopter, weight: 1.4, opacity: 0.95,
        className: 'search-trail', interactive: false,
      }).addTo(g);
    }

    // The real target, where it actually was at this moment.
    const t = this._targetAt ? this._targetAt(plan.reportMs + s * 1000) : null;
    if (t) {
      L.circleMarker(t, {
        radius: 5, color: '#121211', weight: 1, fillColor: SEARCH_COLOURS.target, fillOpacity: 1,
        className: 'search-target', interactive: false,
      }).addTo(g);
    }

    const r = this._result;
    if (r && r.found && s >= r.foundS) {
      const at = this._targetAt(plan.reportMs + r.foundS * 1000);
      if (at) {
        L.circleMarker(at, {
          radius: 12, color: SEARCH_COLOURS.found, weight: 2.5, fill: false,
          className: 'search-found', interactive: false,
        }).addTo(g);
      }
    }

    const h = helicopterAt(plan, Math.min(s, r && r.found ? r.foundS : plan.endS));
    L.marker([h.lat, h.lon], {
      icon: L.divIcon({
        className: 'search-heli',
        html: `<div style="transform: rotate(${h.heading}deg)">${HELI_SVG}</div>`,
        iconSize: [26, 26],
        iconAnchor: [13, 13],
      }),
      interactive: false,
      zIndexOffset: 1000,
    }).addTo(g);
  },

  /** The ground path from arrival to `s`, ending exactly at the helicopter's position now. */
  _flownUpTo(s) {
    const plan = this._plan;
    if (!plan || s < plan.arriveS) return [];
    const stopS = this._result && this._result.found ? Math.min(s, this._result.foundS) : s;
    const p = this._path;
    const pts = [];
    for (let k = 0; k < p.tS.length && p.tS[k] <= stopS; k += 1) pts.push([p.lat[k], p.lon[k]]);
    const h = helicopterAt(plan, Math.min(stopS, plan.endS));
    pts.push([h.lat, h.lon]);
    return pts;
  },
});
