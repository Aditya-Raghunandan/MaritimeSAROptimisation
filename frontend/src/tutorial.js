/**
 * tutorial.js -- three one-minute missions before the real search, each one idea the project
 * had to work out, played on the real ocean with its real places labelled.
 *
 *   1. Go with the flow   drift = current + 2 % of the wind, then catch the drifting buoy
 *                         with a helicopter that turns like a real one (7 degrees a second)
 *   2. Between the dots   the current is known only every 8 km and every 3 hours, and blended
 *                         in between; then guess where the buoy will be, against the model
 *   3. 10,000 guesses     the model's small error persists, so one forecast becomes a cloud;
 *                         pick the error size that catches the real buoys without a cloud too
 *                         big to search
 *
 * The numbers are in tutorialMaths.js (tested); this file only draws and listens. Every
 * mission can be skipped, and so can all three.
 */

import L from 'leaflet';
import { heliIcon } from './scene.js';
import { bearing } from './kinematics.js';
import { esriTiles } from './tiles.js';
import { SEARCH_SPEED_MS, TURN_RATE_DEG_S } from './platform.js';
import {
  GridField, HALF_STRIP_M, SIGMA_TRUE, bilinearWeights, driftOf, heliStep, judgeCloud,
  segmentDistance, toLatLon, toMetres,
} from './tutorialMaths.js';
import { headingFromKeys, keyDirection } from './searchRun.js';

const $ = (id) => document.getElementById(id);
/** Metres one screen pixel covers at this latitude and zoom (Web Mercator), as scene.js. */
const metresPerPixel = (map, lat) => (40075016.686 * Math.cos((lat * Math.PI) / 180)) / (256 * 2 ** map.getZoom());
const ease = (u) => { const x = Math.max(0, Math.min(1, u)); return x * x * (3 - 2 * x); };
const km = (m) => `${(m / 1000).toFixed(1)} km`;

/** Places to orient by, drawn over the ocean (the tile labels name the land). */
const SEA_LABELS = [
  [27.3, -79.75, 'Gulf Stream ↑', 'stream'], [31.0, -78.6, 'Gulf Stream ↗', 'stream'],
  [25.65, -79.75, 'Florida Current ↑', 'stream'], [28.0, -66.0, 'Sargasso Sea', 'sea'],
  [24.2, -74.0, 'The Bahamas', 'sea'], [32.6, -64.6, 'Bermuda', 'sea'], [26.0, -70.5, 'Bermuda Triangle', 'tri'],
];
const TRIANGLE = [[25.76, -80.19], [32.3, -64.78], [18.47, -66.11], [25.76, -80.19]];

function label(lat, lon, text, kind) {
  return L.marker([lat, lon], {
    interactive: false,
    icon: L.divIcon({ className: `tut-label tut-${kind}`, html: text, iconSize: null }),
  });
}

/** An arrow from a to b (both [lat, lon]), with a head and an optional label at its tip. */
function arrow(a, b, colour, text = '', weight = 5, side = 'right') {
  const g = L.layerGroup();
  L.polyline([a, b], { color: colour, weight, opacity: 0.95, interactive: false }).addTo(g);
  const lat0 = a[0];
  const [de, dn] = toMetres(a[0], a[1], b[0], b[1]);
  const len = Math.hypot(de, dn);
  if (len > 1) {
    const h = Math.min(0.28 * len, 900);
    const ang = Math.atan2(dn, de);
    for (const s of [-1, 1]) {
      const back = ang + Math.PI + s * 0.45;
      const tip = toLatLon(lat0, a[1], de + h * Math.cos(back), dn + h * Math.sin(back));
      L.polyline([b, tip], { color: colour, weight, opacity: 0.95, interactive: false }).addTo(g);
    }
  }
  if (text) {
    L.marker(b, { interactive: false, icon: L.divIcon({ className: `tut-arrow-label ${side}`, html: `<span style="color:${colour}">${text}</span>`, iconSize: null }) }).addTo(g);
  }
  return g;
}

const DOT = (colour, size, text = '') => L.divIcon({
  className: 'tut-dot',
  html: `<span style="display:block;width:${size}px;height:${size}px;border-radius:50%;background:${colour};`
    + `border:2px solid #fff;box-shadow:0 0 12px ${colour}"></span>${text ? `<span class="tut-dot-label">${text}</span>` : ''}`,
  iconSize: [size, size],
  iconAnchor: [size / 2, size / 2],
});

export class Tutorial {
  /** `speedUp` (the game's ?speed=) makes the scripted parts faster for tests and rehearsal. */
  constructor({ speedUp = 1, onDone }) {
    Object.assign(this, { speedUp, onDone, map: null, layers: null, raf: 0, timers: [] });
    this.held = new Set();
    this.pointer = null;
    this.keys = (e) => {
      const d = keyDirection(e.code);
      if (!d || !this.flying) return;
      e.preventDefault();
      if (e.type === 'keydown') { this.held.add(d); this.input = 'keys'; } else this.held.delete(d);
    };
  }

  /* ------------------------------------------------------------ frame */

  setup() {
    if (!this.map) {
      this.map = L.map('tutMap', {
        zoomSnap: 0.25, attributionControl: false, keyboard: false, doubleClickZoom: false, zoomControl: false,
      });
      esriTiles('World_Imagery', { maxNativeZoom: 13, maxZoom: 18 }).addTo(this.map);
      L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
        { maxNativeZoom: 13, maxZoom: 18, opacity: 0.9 }).addTo(this.map);
      L.polyline(TRIANGLE, { color: '#ff3b47', weight: 2, dashArray: '8 8', opacity: 0.7, interactive: false }).addTo(this.map);
      for (const l of SEA_LABELS) label(...l).addTo(this.map);
      this.map.on('mousemove', (e) => { this.pointer = e.latlng; if (this.flying) this.input = 'mouse'; });
      this.map.on('click', (e) => this.click?.(e.latlng));
      window.addEventListener('keydown', this.keys);
      window.addEventListener('keyup', this.keys);
    }
    this.map.invalidateSize();
    this.layers?.remove();
    this.layers = L.layerGroup().addTo(this.map);
  }

  clear() {
    cancelAnimationFrame(this.raf);
    for (const t of this.timers) clearTimeout(t);
    this.timers = [];
    this.flying = false;
    this.click = null;
    this.layers?.clearLayers();
    $('tutControls').innerHTML = '';
    $('tutResult').textContent = '';
    $('tutResult').className = 'tut-result';
    $('tutClock').textContent = '';
  }

  later(ms, fn) { this.timers.push(setTimeout(fn, ms / this.speedUp)); }

  panel(step, title, text) {
    $('tutStep').textContent = `Mission ${step} of 3`;
    $('tutTitle').textContent = title;
    $('tutText').innerHTML = text;
    for (const [k, el] of [...document.querySelectorAll('.tut-pips i')].entries()) el.className = k < step ? 'on' : '';
  }

  say(text, kind = '') {
    $('tutResult').innerHTML = text;
    $('tutResult').className = `tut-result ${kind}`;
  }

  next(label, fn) {
    const b = document.createElement('button');
    b.className = 'big';
    b.textContent = label;
    b.addEventListener('click', fn);
    $('tutControls').append(b);
    return b;
  }

  /** From the whole Triangle down to the mission's patch of sea: you are here. */
  arrive(lat, lon, zoom, then) {
    this.map.setView([26.5, -72.5], 5, { animate: false });
    const shiftM = 0.17 * this.map.getSize().x * metresPerPixel({ getZoom: () => zoom }, lat);
    const centre = toLatLon(lat, lon, -shiftM, 0);
    this.later(400, () => this.map.flyTo(centre, zoom, { duration: 2.4 / this.speedUp }));
    this.later(2900, then);
  }

  start() {
    this.setup();
    this.mission1();
  }

  stop() {
    this.clear();
    window.removeEventListener('keydown', this.keys);
    window.removeEventListener('keyup', this.keys);
    this.map?.remove();
    this.map = null;
  }

  skip() {
    if (this.step === 1) this.mission2();
    else if (this.step === 2) this.mission3();
    else this.finish();
  }

  finish() {
    this.clear();
    this.onDone?.();
  }

  /* ------------------------------------------------------------ mission 1 */

  mission1() {
    this.clear();
    this.step = 1;
    const [lat0, lon0] = [25.8, -79.62];                // in the Florida Current, off Miami
    const current = [0, 1.2];
    const wind = [-8, 0];
    const drift = driftOf(current, wind);
    const at = (e, n) => toLatLon(lat0, lon0, e, n);
    this.panel(1, 'Go with the flow',
      'Someone is in the water off Miami, in the <b>Florida Current</b>, the start of the Gulf Stream. '
      + 'What moves them? The <b style="color:#5aa9ff">current</b>, and a little of the <b style="color:#9ff0e0">wind</b>: '
      + 'a person floats low, so only about <b>2 %</b> of the wind’s speed pushes them.');
    this.arrive(lat0 + 0.035, lon0, 12.5, () => {
      const buoy = L.marker(at(0, 0), { icon: DOT('#ff4d6d', 16, 'them'), interactive: false }).addTo(this.layers);
      const T = 2 * 3600;
      this.later(300, () => arrow(at(0, 0), at(current[0] * T, current[1] * T), '#5aa9ff', 'current 1.2 m/s: 8.6 km in 2 h').addTo(this.layers));
      this.later(1800, () => arrow(at(0, 0), at(-4000, 0), '#9ff0e0', 'wind 8 m/s from the east', 3, 'below').addTo(this.layers));
      this.later(3300, () => arrow(at(0, 0), at(drift[0] * T, drift[1] * T), '#ff8a3d', '<b>drift = current + 2 % of the wind</b>', 5, 'left').addTo(this.layers));
      this.later(4800, () => {
        const trail = L.polyline([], { color: '#ff8a3d', weight: 3, dashArray: '2 6', interactive: false }).addTo(this.layers);
        const t0 = performance.now();
        const run = (now) => {
          const u = Math.min(1, ((now - t0) / 4000) * this.speedUp);
          const p = at(drift[0] * T * u, drift[1] * T * u);
          buoy.setLatLng(p);
          trail.addLatLng(p);
          if (u < 1) this.raf = requestAnimationFrame(run);
          else {
            this.say(`In 2 hours the sea carried them <b>${km(Math.hypot(...drift) * T)}</b>. Now go and get them.`);
            this.next('Fly the helicopter →', () => this.mission1Fly());
          }
        };
        this.raf = requestAnimationFrame(run);
      });
    });
  }

  mission1Fly() {
    this.clear();
    const [lat0, lon0] = [25.8, -79.62];
    const drift = driftOf([0, 1.2], [-8, 0]);
    const at = (e, n) => toLatLon(lat0, lon0, e, n);
    this.panel(1, 'Catch the drifting buoy',
      'Fly over them: point the mouse where you want to go (or use the arrow keys). '
      + `<b>It flies like a real Coast Guard helicopter:</b> ${Math.round(SEARCH_SPEED_MS * 3.6)} km/h, `
      + `and it banks round at ${Math.round(TURN_RATE_DEG_S)}° a second, so it cannot turn on the spot. Aim early, and pass right over them: it only sees a strip 185 m wide.`);
    const SPEED = 20 * this.speedUp;              // sim seconds per real second
    const LIMIT_S = 12 * 60;
    let heli = { east: -2500, north: -3500, heading: 45 };
    let target = [0, 0];
    let t = 0;
    const buoy = L.marker(at(0, 0), { icon: DOT('#ff4d6d', 16, 'them'), interactive: false }).addTo(this.layers);
    const strip = L.polyline([at(heli.east, heli.north)], { color: '#ffffff', opacity: 0.4, lineCap: 'round', lineJoin: 'round', interactive: false }).addTo(this.layers);
    const icon = L.marker(at(heli.east, heli.north), { icon: heliIcon('#ffffff', 52), interactive: false, zIndexOffset: 1000 }).addTo(this.layers);
    const restyle = () => strip.setStyle({ weight: Math.max(2, 185.2 / metresPerPixel(this.map, lat0)) });
    this.map.setView(at(-1200 - 0.17 * this.map.getSize().x * metresPerPixel({ getZoom: () => 13.5 }, lat0), -1500), 13.5, { animate: false });
    restyle();
    this.map.on('zoomend', restyle);
    this.flying = true;
    this.input = 'keys';
    this.say('Move the mouse to steer. Find them before the clock runs out.');
    let last = performance.now();
    const tick = (now) => {
      if (!this.flying) return;
      const dt = Math.min(0.1, (now - last) / 1000) * SPEED;
      last = now;
      let wanted = null;
      if (this.input === 'mouse' && this.pointer) {
        const [e, n] = toMetres(lat0, lon0, this.pointer.lat, this.pointer.lng);
        if (Math.hypot(e - heli.east, n - heli.north) > 30) wanted = bearing(e - heli.east, n - heli.north);
      } else wanted = headingFromKeys(this.held);
      const prev = [heli.east, heli.north];
      heli = heliStep(heli, wanted, dt);
      t += dt;
      target = [drift[0] * t, drift[1] * t];
      buoy.setLatLng(at(...target));
      const p = at(heli.east, heli.north);
      strip.addLatLng(p);
      icon.setLatLng(p);
      const svg = icon.getElement()?.querySelector('svg');
      if (svg) svg.style.transform = `rotate(${heli.heading}deg)`;
      this.map.panInside(p, { padding: [80, 80], animate: false });
      const left = Math.max(0, LIMIT_S - t);
      $('tutClock').textContent = `${Math.floor(left / 60)}:${String(Math.floor(left % 60)).padStart(2, '0')}`;
      if (segmentDistance(target, prev, [heli.east, heli.north]) <= HALF_STRIP_M) {
        this.flying = false;
        this.map.off('zoomend', restyle);
        L.circle(at(...target), { radius: 300, color: '#04c951', weight: 3, fill: false }).addTo(this.layers);
        this.say(`<b>Found them</b>, ${Math.round(t / 60)} minutes after you took off. That was the easy part: `
          + 'you knew where they were. Next: how we work out where someone is.', 'ok');
        this.next('Mission 2 →', () => this.mission2());
        return;
      }
      if (t >= LIMIT_S) {
        this.flying = false;
        this.map.off('zoomend', restyle);
        this.say('Out of time. A helicopter cannot turn on the spot: point well ahead and let it swing round.', 'bad');
        this.next('Try again', () => this.mission1Fly());
        this.next('Mission 2 →', () => this.mission2());
        return;
      }
      this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }

  /* ------------------------------------------------------------ mission 2 */

  mission2() {
    this.clear();
    this.step = 2;
    const [lat0, lon0] = [25.42, -75.72];               // east of Eleuthera, the Bahamas
    const at = (e, n) => toLatLon(lat0, lon0, e, n);
    const frame0 = [0, 1, 2].map((j) => [0, 1, 2].map((i) => [0.16 + 0.09 * i, 0.06 + 0.1 * j - 0.03 * i]));
    const rot = (u, v, deg) => {
      const a = (deg * Math.PI) / 180;
      return [1.15 * (u * Math.cos(a) + v * Math.sin(a)), 1.15 * (-u * Math.sin(a) + v * Math.cos(a))];
    };
    const frame3 = frame0.map((row) => row.map(([u, v]) => rot(u, v, -55)));
    const field = new GridField({ frames: [frame0, frame3] });
    const start = [2600, -1300];
    this.panel(2, 'Read the sea between the dots',
      'We never know the current everywhere. The US Navy’s ocean model gives it on a grid, '
      + '<b>8 km apart</b>, and only <b>every 3 hours</b>. The buoy is between the dots and between the hours, '
      + 'so we blend: <b>nearer dots count more</b>, and the hour in between is a mix of the two times.');
    this.arrive(lat0, lon0, 12, () => {
      const nodes = L.layerGroup().addTo(this.layers);
      const blend = L.layerGroup().addTo(this.layers);
      L.marker(at(...start), { icon: DOT('#ff4d6d', 14, 'the buoy'), interactive: false }).addTo(this.layers);
      const ARROW_S = 3 * 3600;
      const t0 = performance.now();
      const DUR = 7000;
      const draw = (hours) => {
        nodes.clearLayers();
        blend.clearLayers();
        for (const [x, y, i, j] of field.points()) {
          const [u, v] = field.node(i, j, hours);
          L.circleMarker(at(x, y), { radius: 6, color: '#fff', weight: 2, fillColor: '#5aa9ff', fillOpacity: 1, interactive: false }).addTo(nodes);
          arrow(at(x, y), at(x + u * ARROW_S, y + v * ARROW_S), '#5aa9ff', '', 3).addTo(nodes);
        }
        const { i, j, fx, fy } = field.cell(...start);
        const w = bilinearWeights(fx, fy);
        const corners = [[i, j], [i + 1, j], [i, j + 1], [i + 1, j + 1]];
        corners.forEach(([ci, cj], k) => {
          const c = at(field.x0 + ci * field.dx, field.y0 + cj * field.dy);
          L.polyline([at(...start), c], { color: '#fff', weight: 1 + 7 * w[k], opacity: 0.25 + 0.6 * w[k], dashArray: '4 6', interactive: false }).addTo(blend);
          L.marker([(c[0] + at(...start)[0]) / 2, (c[1] + at(...start)[1]) / 2], { interactive: false, icon: L.divIcon({ className: 'tut-weight', html: `${Math.round(100 * w[k])} %`, iconSize: null }) }).addTo(blend);
        });
        const [u, v] = field.at(...start, hours);
        arrow(at(...start), at(start[0] + u * ARROW_S, start[1] + v * ARROW_S), '#ff8a3d', 'blended here', 5).addTo(blend);
        const hh = 12 + hours;
        $('tutClock').textContent = `${Math.floor(hh)}:${String(Math.round((hh % 1) * 60) % 60).padStart(2, '0')}`;
        this.say(`It is ${Math.floor(hh)}:${String(Math.round((hh % 1) * 60) % 60).padStart(2, '0')}: the 12:00 map counts `
          + `<b>${Math.round(100 * (1 - hours / 3))} %</b>, the 15:00 map <b>${Math.round((100 * hours) / 3)} %</b>. `
          + 'The lines show how much each dot counts at the buoy.');
      };
      const run = (now) => {
        const u = Math.min(1, ((now - t0) / DUR) * this.speedUp);
        draw(3 * ease(u));
        if (u < 1) this.raf = requestAnimationFrame(run);
        else this.mission2Guess(field, start, at);
      };
      this.raf = requestAnimationFrame(run);
    });
  }

  mission2Guess(field, start, at) {
    $('tutText').innerHTML = 'Your turn to forecast. The current turns as the hours pass. '
      + '<b>Click on the map where you think the buoy will be at 15:00</b>, 3 hours from now.';
    this.say('Click where it will be at 15:00.');
    this.click = (ll) => {
      this.click = null;
      const guess = toMetres(...at(0, 0), ll.lat, ll.lng);
      L.marker(ll, { icon: DOT('#ffffff', 12, 'your guess'), interactive: false }).addTo(this.layers);
      const model = field.path(...start, 3);
      // the real buoy: the same field plus a small error that keeps pushing one way (D030's size)
      const real = field.path(...start, 3, { extra: [0.13, -0.18] });
      const modelLine = L.polyline([], { color: '#ffffff', weight: 3, dashArray: '6 6', interactive: false }).addTo(this.layers);
      const realLine = L.polyline([], { color: '#ff4d6d', weight: 4, interactive: false }).addTo(this.layers);
      const t0 = performance.now();
      const run = (now) => {
        const u = Math.min(1, ((now - t0) / 3500) * this.speedUp);
        const k = Math.round(u * (model.length - 1));
        modelLine.setLatLngs(model.slice(0, k + 1).map((p) => at(...p)));
        realLine.setLatLngs(real.slice(0, k + 1).map((p) => at(...p)));
        if (u < 1) {
          this.raf = requestAnimationFrame(run);
          return;
        }
        const end = real[real.length - 1];
        const mEnd = model[model.length - 1];
        L.marker(at(...mEnd), { icon: DOT('#ffffff', 10, 'the model'), interactive: false }).addTo(this.layers);
        L.marker(at(...end), { icon: DOT('#ff4d6d', 16, 'where it really went'), interactive: false }).addTo(this.layers);
        const you = Math.hypot(guess[0] - end[0], guess[1] - end[1]);
        const them = Math.hypot(mEnd[0] - end[0], mEnd[1] - end[1]);
        this.say(`You were <b>${km(you)}</b> off; the model was <b>${km(them)}</b> off. `
          + (you < them ? 'You beat the model! ' : '')
          + 'Even with every blend done right, the model is wrong by a small speed, about 0.2 m/s, '
          + 'that keeps pushing the same way for hours. That is the last thing to deal with.', you < them ? 'ok' : '');
        this.next('Mission 3 →', () => this.mission3());
      };
      this.raf = requestAnimationFrame(run);
    };
  }

  /* ------------------------------------------------------------ mission 3 */

  mission3() {
    this.clear();
    this.step = 3;
    const [lat0, lon0] = [32.18, -64.98];               // south-west of Bermuda
    const at = (e, n) => toLatLon(lat0, lon0, e, n);
    const T = 2 * 3600;
    this.panel(3, '10,000 guesses',
      'So the forecast is a little wrong, and we do not know which way. Instead of one guess we make thousands, '
      + 'each with its own small error, and search the <b>cloud</b>. But how big an error? '
      + '<b>Set it with the slider</b>: catch at least 4 of the 5 real buoys, with a cloud small enough to search.');
    this.arrive(lat0, lon0, 11.75, () => {
      L.marker(at(0, 0), { icon: DOT('#ffffff', 12, 'the forecast, 2 h after the call'), interactive: false }).addTo(this.layers);
      const cloudLayer = L.layerGroup().addTo(this.layers);
      const renderer = L.canvas({ padding: 0.2 });
      const wrap = document.createElement('label');
      wrap.className = 'tut-slider';
      wrap.innerHTML = '<span>How wrong could the model be?</span><input type="range" id="tutSigma" min="0" max="100" value="12" />'
        + '<b id="tutSigmaVal"></b>';
      $('tutControls').append(wrap);
      const go = this.next('Release the drifters', () => show(true));
      const sigmaOf = () => 0.02 * (40 ** (Number($('tutSigma').value) / 100));   // 0.02 to 0.8 m/s
      const label = () => {
        const s = sigmaOf();
        $('tutSigmaVal').textContent = `${s.toFixed(2)} m/s, so up to about ${km(2.1 * s * T)} after 2 h`;
      };
      $('tutSigma').addEventListener('input', () => { label(); show(false); });
      label();
      const show = (reveal) => {
        cloudLayer.clearLayers();
        const j = judgeCloud(sigmaOf(), T);
        const [e, n] = j.cloud;
        for (let k = 0; k < e.length; k += 1) {
          L.circleMarker(at(e[k], n[k]), { renderer, radius: 1.6, stroke: false, fillColor: '#7fe6ff', fillOpacity: 0.55, interactive: false }).addTo(cloudLayer);
        }
        L.circle(at(0, 0), { radius: j.r90, color: '#fff', weight: 2, dashArray: '6 6', fill: false, interactive: false }).addTo(cloudLayer);
        if (!reveal) {
          this.say(`Cloud: <b>${Math.round(j.areaKm2)} km²</b> holds 90 % of the guesses. Release them to see the real buoys.`);
          return;
        }
        j.buoys.forEach(([be, bn], k) => {
          this.later(350 * (k + 1), () => {
            L.marker(at(be, bn), { icon: DOT(j.caught[k] ? '#04c951' : '#ff4d6d', 14, j.caught[k] ? 'caught' : 'missed'), interactive: false }).addTo(cloudLayer);
          });
        });
        this.later(350 * 6, () => {
          if (j.verdict === 'small') {
            this.say(`<b>Too sure of itself.</b> The cloud caught only ${j.hits} of the 5 buoys. A helicopter would search the wrong sea.`, 'bad');
          } else if (j.verdict === 'big') {
            this.say(`<b>Too vague.</b> It catches all ${j.hits}, but ${Math.round(j.areaKm2)} km² is far too much sea for one helicopter in 45 minutes.`, 'bad');
          } else {
            this.say(`<b>Just right:</b> ${j.hits} of 5 caught in ${Math.round(j.areaKm2)} km². We sized ours the same way, on `
              + `thousands of real drifting buoys: <b>${SIGMA_TRUE} m/s</b>, and 9 in 10 land inside the cloud.`, 'ok');
            go.remove();
            this.next('Now the real search →', () => this.finish());
          }
        });
      };
      show(false);
    });
  }
}
