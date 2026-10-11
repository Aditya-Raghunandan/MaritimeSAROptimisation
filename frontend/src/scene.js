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

const STILL = typeof globalThis.matchMedia === 'function' && globalThis.matchMedia('(prefers-reduced-motion: reduce)').matches;
let heliCount = 0;

/**
 * A Coast Guard search helicopter seen from above, nose up, its rotor hub at the middle so the
 * rotation by heading turns it about the right point: white with an orange nose and band, the
 * main and tail rotors turning, the searchlight ahead, the red (left) and green (right)
 * navigation lights, the red beacon and the white strobes. A ring in the searcher's colour
 * says whose it is. Still, with no animation, for a visitor who asks for reduced motion.
 */
function heliSvg(colour, size) {
  const id = `heli${heliCount += 1}`;
  const anim = (body) => (STILL ? '' : body);
  const spin = (dur, cx = 0, cy = 0) => anim(`<animateTransform attributeName="transform" type="rotate" `
    + `from="0 ${cx} ${cy}" to="360 ${cx} ${cy}" dur="${dur}" repeatCount="indefinite"/>`);
  const blink = (values, dur, begin = '0s') => anim(`<animate attributeName="opacity" values="${values}" `
    + `dur="${dur}" begin="${begin}" repeatCount="indefinite"/>`);
  const blade = (deg) => `<rect x="-1.1" y="-23" width="2.2" height="21" rx="1.1" transform="rotate(${deg})"/>`;
  return `<svg viewBox="-32 -32 64 64" width="${size}" height="${size}" style="overflow:visible">
  <defs>
    <linearGradient id="${id}b" gradientUnits="userSpaceOnUse" x1="0" y1="-12" x2="0" y2="-32">
      <stop offset="0" stop-color="#fff4c2" stop-opacity="0.75"/><stop offset="1" stop-color="#fff4c2" stop-opacity="0"/>
    </linearGradient>
    <filter id="${id}g" x="-1" y="-1" width="3" height="3"><feGaussianBlur stdDeviation="2.4"/></filter>
  </defs>
  <path d="M-2 -12 L-12 -32 L12 -32 L2 -12 Z" fill="url(#${id}b)">${blink('0.8;1;0.85;1;0.8', '1.7s')}</path>
  <circle r="24" fill="none" stroke="${colour}" stroke-width="1.1" stroke-opacity="0.4"/>
  <circle cy="2" r="11" fill="#ff2a2a" filter="url(#${id}g)" opacity="0">${blink('0;0.75;0', '1.1s')}</circle>
  <g stroke="#151515" stroke-width="0.7">
    <rect x="-1.7" y="5" width="3.4" height="19" rx="1.5" fill="#fff"/>
    <rect x="-1.7" y="14" width="3.4" height="2.6" fill="#ff5a1f" stroke="none"/>
    <rect x="-5.5" y="20.5" width="11" height="2.4" rx="1.1" fill="#fff"/>
    <rect x="-9.5" y="-0.2" width="19" height="2.6" rx="1.3" fill="#fff"/>
    <ellipse cy="-1" rx="6" ry="11" fill="#fff"/>
    <rect x="-5.3" y="2.6" width="10.6" height="2.6" rx="1.2" fill="#ff5a1f" stroke="none"/>
    <ellipse cy="-8.6" rx="4.2" ry="3.2" fill="#ff5a1f"/>
    <path d="M-4.6 -5.6 Q0 -9.4 4.6 -5.6 L4 -3.6 Q0 -5.6 -4 -3.6 Z" fill="#1c2f4a"/>
  </g>
  <circle cx="-9.6" cy="1.1" r="1.5" fill="#ff3030"/><circle cx="-9.6" cy="1.1" r="4" fill="#ff3030" filter="url(#${id}g)" opacity="0.7"/>
  <circle cx="9.6" cy="1.1" r="1.5" fill="#33ff77"/><circle cx="9.6" cy="1.1" r="4" fill="#33ff77" filter="url(#${id}g)" opacity="0.7"/>
  <g fill="#fff" filter="url(#${id}g)" opacity="0">
    <circle cx="-10" cy="1.1" r="3.4"/><circle cx="10" cy="1.1" r="3.4"/><circle cy="23" r="3"/>
    ${blink('0;0;1;0;1;0;0', '1.6s', '0.3s')}
  </g>
  <g fill="#e9eef2" fill-opacity="0.9" stroke="#0007" stroke-width="0.5">
    <circle cx="3.6" cy="21.6" r="3.6" fill="#fff" fill-opacity="0.12" stroke="none"/>
    <g><rect x="3.1" y="18" width="1" height="7.2" rx="0.5"/><rect x="0" y="21.1" width="7.2" height="1" rx="0.5"/>${spin('0.25s', 3.6, 21.6)}</g>
  </g>
  <circle r="23" fill="#fff" fill-opacity="0.07"/>
  <g fill="#e9eef2" fill-opacity="0.92" stroke="#0008" stroke-width="0.5">${[0, 90, 180, 270].map(blade).join('')}${spin('0.6s')}</g>
  <circle r="2" fill="#2a2a2a" stroke="#fff" stroke-width="0.8"/>
  <circle cy="3.6" r="1.4" fill="#ff2a2a">${blink('0.35;1;0.35', '1.1s')}</circle>
</svg>`;
}

const HELI = (colour, size = 52) => L.divIcon({
  className: 'scene-heli',
  html: heliSvg(colour, size),
  iconSize: [size, size],
  iconAnchor: [size / 2, size / 2],
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
  constructor(map, { colour = '#3987e5', showCloud = true, heliSize = 52 } = {}) {
    this.map = map;
    this.colour = colour;
    this.probability = new ProbabilityLayer();
    if (showCloud) this.probability.addTo(map);
    this.strip = L.polyline([], { color: colour, opacity: 0.45, lineCap: 'round', lineJoin: 'round', interactive: false }).addTo(map);
    this.path = L.polyline([], { color: '#fff', weight: 1, opacity: 0.9, interactive: false }).addTo(map);
    this.heli = L.marker([0, 0], { icon: HELI(colour, heliSize), interactive: false, zIndexOffset: 1000 });
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
  /**
   * About 400 particles at arrival and at the end of the window, for fitting the view: a
   * sample, so the few far outliers of 10,000 do not set the zoom.
   */
  cloudPoints({ startOnly = false } = {}) {
    const { cloud } = this.window;
    const n = cloud.particles;
    const last = cloud.lat.length - n;
    const pts = [];
    for (let j = 0; j < n; j += Math.max(1, Math.floor(n / 400))) {
      pts.push(ll(cloud.lat[j], cloud.lon[j]));
      if (!startOnly) pts.push(ll(cloud.lat[last + j], cloud.lon[last + j]));
    }
    return pts;
  }

  /**
   * A new window: the cloud at arrival sets the view and the colour scale. The view fits the
   * cloud at arrival and where it has drifted to by the end; `startOnly` fits the arrival
   * alone, and `zoomOut` steps back that many whole zoom levels: the game flies with room to
   * turn in, and its camera follows the drift.
   */
  setWindow(window, { fit = true, zoomOut = 0, startOnly = false } = {}) {
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
      this.map.fitBounds(L.latLngBounds(this.cloudPoints({ startOnly })).pad(0.05), { animate: false });
      if (zoomOut) this.map.setZoom(this.map.getZoom() - zoomOut, { animate: false });
    }
    this.restyle();
  }

  /** Zoom out to the cloud, the flown path and where the real buoy went up to t. */
  fitAll(t, { pad = 0.08 } = {}) {
    const pts = [...this.cloudPoints(), ...this.path.getLatLngs()];
    const target = this.window?.target;
    if (target) {
      const end = Math.min(t, target.tS[target.tS.length - 1]);
      for (let s = Math.max(target.tS[0], 0); s < end; s += 300) pts.push(ll(...trackAt(target, s)));
      pts.push(ll(...trackAt(target, Math.max(Math.max(target.tS[0], 0), end))));
    }
    this.map.fitBounds(L.latLngBounds(pts).pad(pad), { animate: true });
  }

  /** The search at t seconds after arrival, as `flight` has flown it so far. */
  update(time, flight) {
    // requestAnimationFrame's clock can start a frame a hair before the one a loop stored.
    const t = Math.min(flight.duration, Math.max(0, time));
    const { lat, lon } = flight.ep.particlesAt(t);
    this.probability.setState({
      lat, lon, weight: flight.ep.weight, prior: this.window.cloud.weight, marker: this.window.marker, t,
    });
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
