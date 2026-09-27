/**
 * searchLayer.js -- a search drawn on the map (issues #64, #68, #69).
 *
 * Everything decided -- where the helicopter is, where the marker is, whether the
 * target was found -- is computed in searchRun.js and tested there. This file draws it,
 * and LABELS it, because on 24 Sep the first question about the picture was "why does
 * it drop the marker next to the buoy and not on it?". The answer is the point of the
 * project: the Coast Guard does not know where the buoy is. It drops the marker where
 * the drift model PREDICTS it will be, the datum, and the gap between the datum and the
 * real buoy is the model's error. So both are named on the map and the gap is drawn:
 *
 *   base         where the helicopter launches from
 *   last known   the buoy's position at the report time
 *   datum        where drift predicts it will be when the helicopter arrives
 *   predicted    the drift model's path from the last known position to the datum: why
 *                the marker lands so far from where the buoy started (#71)
 *   datum error  a dotted line from the datum to where the buoy REALLY was then
 *   real path    where the buoy actually went, growing as the search plays (#71)
 *   marker       dropped at the datum, drifting with the current; the pattern follows it
 *   planned      the part of the pattern not yet flown, faint, carried with the marker (only
 *                ahead of the helicopter since #83: behind it, the flown track says it); a Parallel
 *                Track's area is outlined too (#75)
 *   swept strip  the path flown, drawn at its TRUE width, 185.2 m, re-scaled on zoom
 *   helicopter   an icon at the current moment, pointing along its heading
 *   real buoy    where the buoy actually was at this moment
 *
 * A helicopter spawned by hand (#68) has no base, datum or marker: only its own path.
 */

import L from 'leaflet';

import { isCloseUp, metresPerPixel as mpp } from './closeUp.js';
import { SWEEP_WIDTH_M, formatDistance } from './geo.js';
import { placeLabels } from './labels.js';
import { offsetAt, offsetPosition } from './patterns.js';
import { formatDuration, helicopterAt, markerPositionAt, searchPath } from './searchRun.js';

/** How often the buoy's real path is sampled for drawing, in seconds. */
const TRACE_STEP_S = 300;

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
export const metresPerPixel = mpp;


const HELI_SVG = `
<svg viewBox="-12 -12 24 24" width="26" height="26" aria-hidden="true">
  <line x1="-10" y1="-10" x2="10" y2="10" stroke="#121211" stroke-width="3.2"/>
  <line x1="10" y1="-10" x2="-10" y2="10" stroke="#121211" stroke-width="3.2"/>
  <line x1="-10" y1="-10" x2="10" y2="10" stroke="#fff" stroke-width="1.4"/>
  <line x1="10" y1="-10" x2="-10" y2="10" stroke="#fff" stroke-width="1.4"/>
  <ellipse cx="0" cy="-1" rx="3.6" ry="5.4" fill="${SEARCH_COLOURS.helicopter}" stroke="#121211" stroke-width="1"/>
  <rect x="-0.9" y="3.5" width="1.8" height="7" fill="${SEARCH_COLOURS.helicopter}" stroke="#121211" stroke-width=".6"/>
</svg>`;

/**
 * A small label beside a point on the map, never in the way of a click: a dark chip in
 * the point's colour, so it reads over any basemap (italic white text on a halo did not).
 * `anchor` is where the chip's corner sits relative to the point, as Leaflet counts it;
 * labels.js chooses it so no two chips overlap (#86).
 */
function tag(latlng, html, colour, cls, anchor) {
  return L.marker(latlng, {
    icon: L.divIcon({
      className: `search-tag ${cls}`,
      html: `<span style="color:${colour}">${html}</span>`,
      iconSize: null,
      iconAnchor: anchor,
    }),
    interactive: false,
    keyboard: false,
  });
}

/*
  A chip's size without drawing it: its text measured in the chip's own fonts, plus its
  padding and borders (index.html, .search-tag span: 600 10.5px, line-height 1.3, padding
  2px 7px 2px 6px, a 1 px border and a 2 px left rule; a second line in 500 10px).
*/
let measure = null;
const widths = new Map();
// Measured before the web font has loaded, a width is the fallback font's: start again once
// it has, and the chips' own measured sizes (below) take over after the first draw anyway.
if (typeof document !== 'undefined' && document.fonts) document.fonts.ready.then(() => widths.clear());
function textWidth(text, font) {
  const key = `${font}|${text}`;
  if (widths.has(key)) return widths.get(key);
  if (!measure) measure = document.createElement('canvas').getContext('2d');
  const family = getComputedStyle(document.documentElement).getPropertyValue('--font').trim() || 'Inter, sans-serif';
  measure.font = `${font} ${family}`;
  const w = measure.measureText(text).width;
  widths.set(key, w);
  return w;
}
function chipSize(text, sub) {
  const w = Math.max(textWidth(text, '600 10.5px'), sub ? textWidth(sub, '500 10px') : 0);
  return [Math.ceil(w + 16), sub ? 34 : 20];
}

export const SearchLayer = L.Layer.extend({
  initialize() {
    this._plan = null;
    this._path = null;
    this._result = null;
    this._targetAt = null;
    this._s = 0;
    this._base = null;
    this._lkp = null;
    this._flight = null;
    this._labelsAt = {};     // id -> the spot each label used last frame (labels.js)
    this._drawn = new Map(); // label text -> its chip's size as actually drawn, [w, h]
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

  /**
   * Load a search. `targetAt(ms)` is where the real target was, or null for none. For a
   * helicopter flown by hand, `opts.flight` is its ManualFlight.
   */
  setPlan(plan, targetAt, result, opts = {}) {
    this._flight = opts.flight ?? null;
    this._plan = plan;
    this._path = plan && plan.pattern ? searchPath(plan) : null;
    this._targetAt = targetAt;
    this._result = result;
    this._s = 0;
    if (this._map) this._render();
  },

  /** Seconds after the report (or after the spawn). */
  setTime(s) {
    this._s = s;
    if (this._map) this._render();
  },

  clear() {
    this.setPlan(null, null, null);
  },

  /** A label to place once everything is drawn, so no two overlap (#86). */
  _label(id, latlng, text, colour, priority, cls = '', sub = null) {
    this._pending.push({ id, latlng, text, sub, colour, priority, cls });
  },

  /** A marker's footprint, px: labels keep off it. */
  _occupy(latlng, w, h = w) {
    const p = this._map.latLngToContainerPoint(latlng);
    this._obstacles.push({ x: p.x - w / 2, y: p.y - h / 2, w, h });
  },

  _render() {
    this._pending = [];
    this._obstacles = [];
    this._draw();
    this._placeLabels();
  },

  /**
   * The panels over the map -- the Search panel, the presets, the drifter list, the legends,
   * the compass -- as boxes in map pixels: a label under one is as unread as one under
   * another label (the last known position hid under the Search panel).
   */
  _panelBoxes() {
    const c = this._map.getContainer().getBoundingClientRect();
    const boxes = [];
    for (const el of this._map.getContainer().querySelectorAll('.leaflet-control')) {
      const r = el.getBoundingClientRect();
      if (r.width > 0 && r.height > 0) boxes.push({ x: r.left - c.left, y: r.top - c.top, w: r.width, h: r.height });
    }
    return boxes;
  },

  _placeLabels() {
    const size = this._map.getSize();
    const items = this._pending.map((l) => {
      const p = this._map.latLngToContainerPoint(l.latlng);
      const key = `${l.text}|${l.sub ?? ''}`;
      const [w, h] = this._drawn.get(key) ?? chipSize(l.text, l.sub);
      return { ...l, key, x: p.x, y: p.y, w, h };
    });
    // The predicted drift's label (priority 5) is the one to spare: its dotted path says it.
    const at = placeLabels(items, [...this._obstacles, ...this._panelBoxes()], { x: 0, y: 0, w: size.x, h: size.y },
      this._labelsAt, 5);
    this._labelsAt = {};
    for (const l of items) {
      const spot = at[l.id];
      this._labelsAt[l.id] = spot.index;
      if (spot.hidden) continue;
      const html = l.sub ? `${l.text}<small>${l.sub}</small>` : l.text;
      const marker = tag(l.latlng, html, l.colour, l.cls, [-spot.dx, -spot.dy]).addTo(this._group);
      // The chip as drawn is the truth: remember its size, and if the estimate was short,
      // place everything again with it (once per text, so this settles at once).
      const span = marker.getElement() && marker.getElement().firstElementChild;
      if (span && !this._drawn.has(l.key)) {
        const r = span.getBoundingClientRect();
        this._drawn.set(l.key, [Math.ceil(r.width), Math.ceil(r.height)]);
        if (r.width > l.w + 1 || r.height > l.h + 1) this._replace = true;
      }
    }
    if (this._replace) {
      this._replace = false;
      for (const m of this._group.getLayers()) if (m.options.icon && /search-tag/.test(m.options.icon.options.className)) this._group.removeLayer(m);
      this._placeLabels();
    }
  },

  _draw() {
    const g = this._group;
    g.clearLayers();
    const plan = this._plan;
    const s = this._s;

    const base = plan ? (plan.free ? null : plan.base) : this._base;
    if (base) {
      L.marker([base.lat, base.lon], {
        icon: L.divIcon({ className: 'search-base', html: 'BASE', iconSize: [38, 16] }),
        interactive: false,
      }).addTo(g);
      this._occupy([base.lat, base.lon], 40, 18);
    }
    const lkp = plan ? plan.lkp : this._lkp;
    if (lkp) {
      L.circleMarker([lkp.lat, lkp.lon], {
        radius: 6, color: SEARCH_COLOURS.target, weight: 1.5, fill: false, dashArray: '2 3',
        className: 'search-lkp', interactive: false,
      }).addTo(g);
      this._occupy([lkp.lat, lkp.lon], 14);
      this._label('lkp', [lkp.lat, lkp.lon], 'last known position', SEARCH_COLOURS.target, 4);
    }
    if (!plan) return;

    if (!plan.free) this._renderPrediction(g, plan);
    this._renderBuoyPath(g, plan, s);
    if (!plan.free) this._renderDoctrine(g, plan, s);
    if (!plan.free) this._renderPlanned(g, plan, s);
    this._renderFlown(g, plan, s);

    // The real target, where it actually was at this moment.
    const t = this._targetAt ? this._targetAt(plan.reportMs + s * 1000) : null;
    if (t) {
      L.circleMarker(t, {
        radius: 5, color: '#121211', weight: 1, fillColor: SEARCH_COLOURS.target, fillOpacity: 1,
        className: 'search-target', interactive: false,
      }).addTo(g);
      this._occupy(t, 12);
      this._label('buoy', t, 'real buoy', SEARCH_COLOURS.target, 1);
    }

    const r = this._flight ? this._flight.result() : this._result;
    if (r && r.found && s >= r.foundS) {
      const at = this._targetAt(plan.reportMs + r.foundS * 1000);
      if (at) {
        L.circleMarker(at, {
          radius: 12, color: SEARCH_COLOURS.found, weight: 2.5, fill: false,
          className: 'search-found', interactive: false,
        }).addTo(g);
        this._occupy(at, 28);
      }
    }

    const h = this._helicopter(plan, s, r);
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
    this._occupy([h.lat, h.lon], 26);
  },

  /**
   * The drift model's prediction, drawn as the path it is: from the last known position
   * to the datum, over the time the helicopter took to get there. That is the answer to
   * "why is the marker so far from where the buoy started?".
   */
  _renderPrediction(g, plan) {
    const d = plan.datumTrack;
    if (!d || d.tS.length < 2) return;
    const pts = d.tS.map((_, k) => [d.lat[k], d.lon[k]]);
    L.polyline(pts, {
      color: SEARCH_COLOURS.datum, weight: 1.6, opacity: 0.85, dashArray: '2 5',
      className: 'search-predicted', interactive: false,
    }).addTo(g);
    const mid = pts[Math.floor(pts.length / 2)];
    this._label('predicted', mid, `predicted drift · ${formatDuration(plan.arriveS)}`, SEARCH_COLOURS.datum,
      5, 'search-tag-predicted');
  },

  /** Where the buoy really went, from the report to now: the trace the prediction is judged by. */
  _renderBuoyPath(g, plan, s) {
    if (!this._targetAt) return;
    const pts = [];
    for (let t = 0; t < s; t += TRACE_STEP_S) {
      const p = this._targetAt(plan.reportMs + t * 1000);
      if (p) pts.push(p);
    }
    const now = this._targetAt(plan.reportMs + s * 1000);
    if (now) pts.push(now);
    if (pts.length < 2) return;
    L.polyline(pts, {
      color: SEARCH_COLOURS.target, weight: 2, opacity: 0.9,
      className: 'search-buoy-path', interactive: false,
    }).addTo(g);
  },

  /** The doctrinal marks: flight out, datum, the datum's error, and the marker. */
  _renderDoctrine(g, plan, s) {
    const { base, datum } = plan;
    L.polyline([[base.lat, base.lon], [datum.lat, datum.lon]], {
      color: SEARCH_COLOURS.helicopter, weight: 1.2, opacity: 0.7, dashArray: '6 6',
      className: 'search-transit', interactive: false,
    }).addTo(g);
    L.circleMarker([datum.lat, datum.lon], {
      radius: 7, color: SEARCH_COLOURS.datum, weight: 1.5, fill: false, dashArray: '3 3',
      className: 'search-datum', interactive: false,
    }).addTo(g);
    // How wrong the prediction was: the datum against where the buoy really was on arrival.
    // Said in the datum's own label, once there is an answer: a second label halfway along
    // the error line ran into the found ring (screenshots, 25 Sep).
    const truth = s >= plan.arriveS && this._targetAt ? this._targetAt(plan.dropMs) : null;
    const miss = truth ? this._map.distance([datum.lat, datum.lon], truth) : null;
    this._occupy([datum.lat, datum.lon], 16);
    this._label('datum', [datum.lat, datum.lon], 'datum · where drift predicts it', SEARCH_COLOURS.datum, 2,
      'search-tag-datum', miss === null ? null : `${formatDistance(miss)} from where the buoy really was`);

    if (s < plan.arriveS) return;
    if (truth) {
      L.polyline([[datum.lat, datum.lon], truth], {
        color: SEARCH_COLOURS.datum, weight: 1.2, opacity: 0.8, dashArray: '1 4',
        className: 'search-datum-error', interactive: false,
      }).addTo(g);
    }

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
      this._occupy([now.lat, now.lon], 10);
      this._label('marker', [now.lat, now.lon], 'marker', SEARCH_COLOURS.marker, 3);
    }
  },

  /**
   * The pattern still to fly, faint and dashed, where the marker is now -- so the shape is
   * legible before the helicopter has drawn it. A Parallel Track also shows its area.
   */
  _renderPlanned(g, plan, s) {
    // Before the drop the pattern sits on the datum; after it, it rides the marker.
    const m = markerPositionAt(plan, Math.max(s, plan.arriveS)) ?? plan.datum;
    const p = plan.pattern;
    // Only what is still to fly (#83): behind the helicopter the flown track says it, and
    // the two drawn over each other were a mesh of orange lines.
    const r = this._result;
    const stopS = r && r.found ? r.foundS : plan.endS;
    const t = Math.min(Math.max(Math.min(s, stopS) - plan.arriveS, 0), p.durationS);
    const here = offsetAt(p, t);
    const ahead = here ? [here] : [];
    for (let k = 0; k < p.tS.length; k += 1) if (p.tS[k] > t) ahead.push([p.eastM[k], p.northM[k]]);
    if (ahead.length >= 2) {
      L.polyline(ahead.map(([e, n]) => offsetPosition(m.lat, m.lon, e, n)), {
        color: SEARCH_COLOURS.helicopter, weight: 1, opacity: 0.35, dashArray: '3 6',
        className: 'search-planned', interactive: false,
      }).addTo(g);
    }
    if (p.area) {
      const len = p.area.lengthM;
      const wid = p.area.widthM;
      const b = (p.area.bearingDeg * Math.PI) / 180;
      const u = [Math.sin(b), Math.cos(b)];
      const v = [Math.cos(b), -Math.sin(b)];
      const corner = (x, y) => offsetPosition(m.lat, m.lon, x * u[0] + y * v[0], x * u[1] + y * v[1]);
      L.polygon([corner(-len / 2, -wid / 2), corner(len / 2, -wid / 2), corner(len / 2, wid / 2),
        corner(-len / 2, wid / 2)], {
        color: SEARCH_COLOURS.helicopter, weight: 1, opacity: 0.5, dashArray: '6 4', fill: false,
        className: 'search-area', interactive: false,
      }).addTo(g);
    }
  },

  /** The swept strip at true width, and the track down its middle. */
  _renderFlown(g, plan, s) {
    const flown = this._flight ? this._flight.pathUpTo(s) : this._flownUpTo(s);
    if (flown.length < 2) return;
    const zoom = this._map.getZoom();
    const px = SWEEP_WIDTH_M / metresPerPixel(flown[0][0], zoom);
    // Still at true width close up, but fainter: side by side the strips cover the whole
    // pattern, and at 0.28 that was an orange block over the sea it was searching (#79).
    L.polyline(flown, {
      color: SEARCH_COLOURS.helicopter, weight: Math.max(px, 1), opacity: isCloseUp(zoom) ? 0.13 : 0.28,
      lineCap: 'butt', lineJoin: 'miter', className: 'search-strip', interactive: false,
    }).addTo(g);
    L.polyline(flown, {
      color: SEARCH_COLOURS.helicopter, weight: 1.4, opacity: 0.95,
      className: 'search-trail', interactive: false,
    }).addTo(g);
  },

  _helicopter(plan, s, r) {
    if (this._flight) {
      return s >= this._flight.s ? this._flight.position() : this._flight.positionAt(s);
    }
    return helicopterAt(plan, Math.min(s, r && r.found ? r.foundS : plan.endS));
  },

  /** A pattern's ground path from arrival to `s`, ending exactly at the helicopter now. */
  _flownUpTo(s) {
    const plan = this._plan;
    if (!plan || !this._path || s < plan.arriveS) return [];
    const stopS = this._result && this._result.found ? Math.min(s, this._result.foundS) : s;
    const p = this._path;
    const pts = [];
    for (let k = 0; k < p.tS.length && p.tS[k] <= stopS; k += 1) pts.push([p.lat[k], p.lon[k]]);
    const h = helicopterAt(plan, Math.min(stopS, plan.endS));
    pts.push([h.lat, h.lon]);
    return pts;
  },
});
