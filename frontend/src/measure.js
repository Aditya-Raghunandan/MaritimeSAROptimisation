/**
 * measure.js -- scale bar, ruler and range rings on the live map.
 *
 * The arithmetic lives in `geo.js`, which imports nothing: a bearing has no
 * business needing a map library, and splitting them is what lets the geodesy
 * be tested without a DOM. This file is only the Leaflet wiring.
 *
 * These tools matter more here than on an ordinary map because of the scales:
 * sweep width is ~185 m, the probability map's cells are 200-500 m (D007), and
 * the search patterns are kilometres across. Without a ruler nobody can judge
 * by eye whether a track spacing is plausible.
 *
 * Distance comes from `map.distance()` -- great circle, from the coordinates --
 * never from pixels. See `geo.js` for why that is not a nicety.
 */

import L from 'leaflet';

import {
  RANGE_RINGS_KM, TYPICAL_CURRENT_MS, bearing, driftHours, formatDistance,
} from './geo.js';

/**
 * Click-to-measure ruler.
 *
 * Reports each leg's distance and bearing, a running total, and how long that
 * distance takes at the Gulf Stream's typical speed -- which is the question
 * this project exists to answer, so it may as well be readable off the map.
 */
export class Ruler {
  constructor(map, opts = {}) {
    this.map = map;
    this.active = false;
    this.points = [];
    this.layer = L.layerGroup().addTo(map);
    this.onChange = opts.onChange ?? (() => {});
    // The ruler owns the whole chain: click, point, label, readout. It used to
    // add the point here and leave the labelling and the readout to a separate
    // handler in main.js, which meant two listeners on one click and a readout
    // that silently stopped happening when another tool's handler returned
    // first. A tool that owns its click cannot be half-disabled by a sibling.
    this._onClick = (e) => {
      const summary = this.addPoint(e.latlng);
      this.label(summary.legs, summary.total);
      this.onChange(summary);
    };
  }

  toggle() {
    this.active = !this.active;
    if (this.active) {
      this.map.on('click', this._onClick);
      L.DomUtil.addClass(this.map.getContainer(), 'measuring');
    } else {
      this.map.off('click', this._onClick);
      L.DomUtil.removeClass(this.map.getContainer(), 'measuring');
      this.clear();
    }
    return this.active;
  }

  clear() {
    this.points = [];
    this.layer.clearLayers();
    this._labels = null;   // cleared with the layer it lived in
  }

  addPoint(latlng) {
    this.points.push(latlng);
    L.circleMarker(latlng, { radius: 4, color: '#ff3b30', fillOpacity: 1 }).addTo(this.layer);
    if (this.points.length > 1) {
      L.polyline(this.points, { color: '#ff3b30', weight: 2, dashArray: '5,4' }).addTo(this.layer);
    }
    return this.summary();
  }

  /**
   * Write each leg's distance onto the map, plus a running total at the end.
   *
   * A divIcon rather than a Leaflet tooltip: tooltips are tied to a marker's
   * anchor and fight for space, while this sits at the leg's midpoint where
   * the measurement actually belongs.
   */
  label(legs, total) {
    if (this._labels) this._labels.clearLayers();
    else this._labels = L.layerGroup().addTo(this.layer);

    for (let k = 0; k < legs.length; k += 1) {
      const a = this.points[k];
      const b = this.points[k + 1];
      const mid = L.latLng((a.lat + b.lat) / 2, (a.lng + b.lng) / 2);
      L.marker(mid, {
        interactive: false,
        icon: L.divIcon({
          className: '',
          html: `<span class="leg-label">${formatDistance(legs[k].distance)}`
            + ` · ${legs[k].bearing.toFixed(0)}°</span>`,
          iconSize: null,
        }),
      }).addTo(this._labels);
    }

    // The total belongs at the far end, where the eye finishes the line.
    if (legs.length > 1) {
      L.marker(this.points[this.points.length - 1], {
        interactive: false,
        icon: L.divIcon({
          className: '',
          html: `<span class="leg-label total">total ${formatDistance(total)}`
            + ` · ${driftHours(total).toFixed(1)} h at ${TYPICAL_CURRENT_MS} m/s</span>`,
          iconSize: null,
        }),
      }).addTo(this._labels);
    }
  }

  summary() {
    const legs = [];
    let total = 0;
    for (let k = 1; k < this.points.length; k += 1) {
      const d = this.map.distance(this.points[k - 1], this.points[k]); // metres, geodesic
      total += d;
      legs.push({ distance: d, bearing: bearing(this.points[k - 1], this.points[k]) });
    }
    return { legs, total, driftHours: driftHours(total) };
  }
}

/**
 * Range rings around a datum, in the units a SAR planner thinks in.
 *
 * `L.circle` takes a radius in metres and draws it correctly on the projection,
 * so these are true distances rather than a fixed pixel radius. The fastest
 * sanity check there is on a drift ensemble: the Gulf Stream covers about
 * 155 km a day, so a 24 h particle cloud reaching outside that ring means the
 * physics is wrong, not the map.
 */
/**
 * Range rings around a datum you can actually move.
 *
 * The first version dropped a fixed set of rings at the map centre and that was
 * all: no way to put them where the incident was, no way to change the radii,
 * no way to get rid of them except toggling the button. A datum you cannot
 * place is not a datum -- the whole point is "how far could they have gone from
 * HERE".
 *
 * So: click the map to place it, drag the centre to move it, and scroll or use
 * the +/- keys over the map to scale the whole set. `L.circle` takes metres and
 * draws them correctly on the projection, so these stay true distances at any
 * latitude rather than a fixed pixel radius -- which matters across a 19 deg
 * box where Mercator stretches by 18 % top to bottom.
 */
export class RangeRings {
  constructor(map, opts = {}) {
    this.map = map;
    this.radiiKm = [...(opts.radiiKm ?? RANGE_RINGS_KM)];
    this.baseKm = [...this.radiiKm];
    this.scale = 1;
    this.active = false;
    this.centre = null;
    this.layer = L.layerGroup();
    this.onChange = opts.onChange ?? (() => {});

    this._onClick = (e) => this.placeAt(e.latlng);
    // NO WHEEL HANDLER, deliberately. Scroll already means zoom on a map, and
    // taking it over made the rings resize while the map refused to zoom at
    // all -- trading one gesture for another rather than adding one. Resizing
    // is on explicit buttons and on +/-, which collide with nothing.
    this._onKey = (e) => {
      if (!this.active || !this.centre) return;
      if (e.key === '+' || e.key === '=') this.rescale(1.25);
      else if (e.key === '-' || e.key === '_') this.rescale(0.8);
    };
  }

  toggle() {
    this.active = !this.active;
    if (this.active) {
      this.layer.addTo(this.map);
      this.map.on('click', this._onClick);
      document.addEventListener('keydown', this._onKey);
      // Somewhere to start, so the tool is visibly on before the first click.
      this.placeAt(this.map.getCenter());
    } else {
      this.map.off('click', this._onClick);
      document.removeEventListener('keydown', this._onKey);
      this.clear();
      this.map.removeLayer(this.layer);
    }
    return this.active;
  }

  clear() {
    this.layer.clearLayers();
    this.centre = null;
  }

  placeAt(latlng) {
    this.centre = latlng;
    this._draw();
    this.onChange(this.summary());
  }

  /** Grow or shrink every ring together, keeping their ratios. */
  rescale(by) {
    this.scale = Math.min(20, Math.max(0.05, this.scale * by));
    this.radiiKm = this.baseKm.map((km) => {
      const v = km * this.scale;
      // Round to something a planner would say out loud rather than 23.44 km.
      return v >= 100 ? Math.round(v / 10) * 10 : v >= 10 ? Math.round(v) : Math.round(v * 10) / 10;
    });
    this._draw();
    this.onChange(this.summary());
  }

  summary() {
    return { centre: this.centre, radiiKm: [...this.radiiKm], scale: this.scale };
  }

  _draw() {
    this.layer.clearLayers();
    if (!this.centre) return;
    const c = this.centre;

    for (const km of this.radiiKm) {
      L.circle(c, {
        radius: km * 1000, color: '#ffcc00', weight: 1, fill: false, dashArray: '4,4',
        interactive: false,
      }).addTo(this.layer);
      // Label north of the ring. 111 km per degree of latitude is exact enough
      // for placing a label and wrong enough to never use for a measurement.
      L.marker([c.lat + km / 111, c.lng], {
        interactive: false,
        icon: L.divIcon({ className: 'ring-label', html: `${km} km`, iconSize: [46, 14] }),
      }).addTo(this.layer);
    }

    // The draggable datum, added last so it sits on top of the rings.
    const handle = L.circleMarker(c, {
      radius: 6, color: '#fff', weight: 2, fillColor: '#ffcc00', fillOpacity: 1,
      className: 'ring-datum',
    }).addTo(this.layer);
    handle.on('mousedown', () => this._startDrag());
    handle.bindTooltip('Drag to move · scroll to resize', { direction: 'top' });
  }

  /** Drag on the datum moves the whole set; the map must not pan underneath. */
  _startDrag() {
    this.map.dragging.disable();
    const move = (e) => this.placeAt(e.latlng);
    const stop = () => {
      this.map.off('mousemove', move);
      this.map.off('mouseup', stop);
      this.map.dragging.enable();
    };
    this.map.on('mousemove', move);
    this.map.on('mouseup', stop);
  }
}

export function rangeRings(map, latlng, radiiKm = RANGE_RINGS_KM) {
  const rings = new RangeRings(map, { radiiKm });
  rings.layer.addTo(map);
  rings.placeAt(latlng);
  return rings.layer;
}

export function addScaleBar(map) {
  return L.control.scale({ metric: true, imperial: false, maxWidth: 180 }).addTo(map);
}

export { RANGE_RINGS_KM, SWEEP_WIDTH_M, TYPICAL_CURRENT_MS, formatDistance } from './geo.js';
