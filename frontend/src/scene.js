/**
 * scene.js -- one search drawn on a Leaflet map: the draining probability map, the strip
 * the helicopter has swept at its true 185 m width, the helicopter, the datum marker, the
 * last known position and, when revealed, the real buoy (D031, Stages 4 and 5).
 *
 * The scenario page draws one of these and the game's reveal draws three side by side; the
 * flight (playback.js) decides everything, this only draws it. The helicopter points the
 * way it is flying, and a flight being steered by a person shows, dashed, the turn it is
 * about to make (D032: it cannot turn on the spot, so the bend is worth seeing coming).
 */

import L from 'leaflet';
import { SWEEP_WIDTH_M } from './geo.js';
import { ProbabilityLayer } from './probabilityLayer.js';
import { trackAt } from './referee.js';

const display = (lon) => (lon > 180 ? lon - 360 : lon);
const ll = (lat, lon) => [lat, display(lon)];

const HELI = (colour) => L.divIcon({
  className: 'scene-heli',
  html: `<svg viewBox="-12 -12 24 24" width="22" height="22"><circle r="9" fill="${colour}" `
    + 'stroke="#fff" stroke-width="2"/><path d="M0 -6 L4 4 L0 2 L-4 4 Z" fill="#fff"/></svg>',
  iconSize: [22, 22],
  iconAnchor: [11, 11],
});

const DOT = (colour, size = 12, label = '') => L.divIcon({
  className: 'scene-dot',
  html: `<span style="display:block;width:${size}px;height:${size}px;border-radius:50%;`
    + `background:${colour};border:2px solid #fff;box-shadow:0 0 0 1px #0006"></span>`
    + (label ? `<span class="scene-label">${label}</span>` : ''),
  iconSize: [size, size],
  iconAnchor: [size / 2, size / 2],
});

/** Metres one screen pixel covers at this latitude and zoom (Web Mercator). */
function metresPerPixel(map, lat) {
  return (40075016.686 * Math.cos((lat * Math.PI) / 180)) / (256 * 2 ** map.getZoom());
}

export class Scene {
  constructor(map, { colour = '#3987e5', showCloud = true } = {}) {
    this.map = map;
    this.colour = colour;
    this.probability = new ProbabilityLayer();
    if (showCloud) this.probability.addTo(map);
    this.strip = L.polyline([], { color: colour, opacity: 0.45, lineCap: 'round', lineJoin: 'round', interactive: false }).addTo(map);
    this.path = L.polyline([], { color: '#fff', weight: 1, opacity: 0.9, interactive: false }).addTo(map);
    this.heli = L.marker([0, 0], { icon: HELI(colour), interactive: false });
    this.ahead = L.polyline([], {
      color: '#ffcd8d', weight: 2, opacity: 0.85, dashArray: '6 6', interactive: false,
    });
    this.marker = L.marker([0, 0], { icon: DOT('#ffcd8d', 10), interactive: false });
    this.lkp = L.marker([0, 0], { icon: DOT('#c3c2b7', 9, 'last known position'), interactive: false });
    this.buoy = L.marker([0, 0], { icon: DOT('#ff4d6d', 14, 'the real buoy'), interactive: false });
    this.buoyTrail = L.polyline([], { color: '#ff4d6d', weight: 2, dashArray: '4 4', interactive: false });
    map.on('zoomend', this.restyle, this);
  }

  /** A new window: the cloud at arrival sets the view and the colour scale. */
  setWindow(window, { fit = true } = {}) {
    this.window = window;
    const { cloud, marker } = window;
    const n = cloud.particles;
    const lat = cloud.lat.subarray(0, n);
    const lon = cloud.lon.subarray(0, n);
    this.probability.setState({ lat, lon, weight: cloud.weight, marker, t: 0 }, { resetScale: true });
    this.lkp.setLatLng(ll(window.meta.lkp.lat, window.meta.lkp.lon)).addTo(this.map);
    this.marker.setLatLng(ll(...trackAt(marker, 0))).addTo(this.map);
    this.strip.setLatLngs([]);
    this.path.setLatLngs([]);
    this.heli.remove();
    this.hideBuoy();
    if (fit) {
      const last = cloud.lat.length - n;
      const pts = [];
      for (let j = 0; j < n; j += Math.max(1, Math.floor(n / 400))) {
        pts.push(ll(cloud.lat[j], cloud.lon[j]), ll(cloud.lat[last + j], cloud.lon[last + j]));
      }
      this.map.fitBounds(L.latLngBounds(pts).pad(0.05), { animate: false });
    }
    this.restyle();
  }

  /** The search at t seconds after arrival, as `flight` has flown it so far. */
  update(time, flight) {
    // requestAnimationFrame's clock can start a frame a hair before the one a loop stored.
    const t = Math.min(flight.duration, Math.max(0, time));
    const { lat, lon } = flight.ep.particlesAt(t);
    this.probability.setState({ lat, lon, weight: flight.ep.weight, marker: this.window.marker, t });
    const trail = flight.trail(t).map(([, la, lo]) => ll(la, lo));
    this.strip.setLatLngs(trail);
    this.path.setLatLngs(trail);
    this.heli.setLatLng(trail[trail.length - 1]);
    if (!this.map.hasLayer(this.heli)) this.heli.addTo(this.map);
    const svg = this.heli.getElement()?.querySelector('svg');
    if (svg && flight.headingAt) svg.style.transform = `rotate(${flight.headingAt(t)}deg)`;
    if (flight.preview && t < flight.duration) {
      this.ahead.setLatLngs(flight.preview(t).map(([la, lo]) => ll(la, lo)));
      if (!this.map.hasLayer(this.ahead)) this.ahead.addTo(this.map);
    } else this.ahead.remove();
    this.marker.setLatLng(ll(...trackAt(this.window.marker, Math.min(t, 2700))));
    if (this.map.hasLayer(this.buoy)) this.placeBuoy(t);
  }

  /** The real buoy at t, and where it went, if the window has one. */
  showBuoy(t) {
    if (!this.window?.target) return;
    this.buoy.addTo(this.map);
    this.buoyTrail.addTo(this.map);
    this.placeBuoy(t);
  }

  placeBuoy(t) {
    const target = this.window.target;
    const end = Math.min(t, target.tS[target.tS.length - 1]);
    const start = Math.max(target.tS[0], 0);
    this.buoy.setLatLng(ll(...trackAt(target, Math.max(start, end))));
    const pts = [];
    for (let s = start; s <= end; s += 60) pts.push(ll(...trackAt(target, s)));
    pts.push(ll(...trackAt(target, Math.max(start, end))));
    this.buoyTrail.setLatLngs(pts);
  }

  hideBuoy() {
    this.buoy.remove();
    this.buoyTrail.remove();
  }

  /** The strip at its true width on screen, never thinner than 2 px. */
  restyle() {
    const lat = this.map.getCenter().lat;
    this.strip.setStyle({ weight: Math.max(2, SWEEP_WIDTH_M / metresPerPixel(this.map, lat)) });
  }

  destroy() {
    this.map.off('zoomend', this.restyle, this);
    for (const layer of [this.probability, this.strip, this.path, this.heli, this.ahead, this.marker,
      this.lkp, this.buoy, this.buoyTrail]) layer.remove();
  }
}
