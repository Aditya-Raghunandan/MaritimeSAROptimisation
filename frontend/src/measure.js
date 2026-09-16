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

import { RANGE_RINGS_KM, TYPICAL_CURRENT_MS, bearing, driftHours } from './geo.js';

/**
 * Click-to-measure ruler.
 *
 * Reports each leg's distance and bearing, a running total, and how long that
 * distance takes at the Gulf Stream's typical speed -- which is the question
 * this project exists to answer, so it may as well be readable off the map.
 */
export class Ruler {
  constructor(map) {
    this.map = map;
    this.active = false;
    this.points = [];
    this.layer = L.layerGroup().addTo(map);
    this._onClick = (e) => this.addPoint(e.latlng);
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
  }

  addPoint(latlng) {
    this.points.push(latlng);
    L.circleMarker(latlng, { radius: 4, color: '#ff3b30', fillOpacity: 1 }).addTo(this.layer);
    if (this.points.length > 1) {
      L.polyline(this.points, { color: '#ff3b30', weight: 2, dashArray: '5,4' }).addTo(this.layer);
    }
    return this.summary();
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
export function rangeRings(map, latlng, radiiKm = RANGE_RINGS_KM) {
  const group = L.layerGroup();
  L.circleMarker(latlng, { radius: 5, color: '#ffcc00', fillOpacity: 1 }).addTo(group);
  for (const km of radiiKm) {
    L.circle(latlng, {
      radius: km * 1000,
      color: '#ffcc00',
      weight: 1,
      fill: false,
      dashArray: '4,4',
    }).addTo(group);
    L.marker(
      [latlng.lat + km / 111, latlng.lng],
      {
        icon: L.divIcon({
          className: 'ring-label',
          html: `${km} km`,
          iconSize: [40, 14],
        }),
      },
    ).addTo(group);
  }
  group.addTo(map);
  return group;
}

/** Metric scale bar, redrawn by Leaflet on every pan -- which Mercator requires. */
export function addScaleBar(map) {
  return L.control.scale({ metric: true, imperial: false, maxWidth: 180 }).addTo(map);
}

export { RANGE_RINGS_KM, SWEEP_WIDTH_M, TYPICAL_CURRENT_MS, formatDistance } from './geo.js';
