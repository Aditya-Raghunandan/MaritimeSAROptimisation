/**
 * tutorial.js -- three one-minute missions before the real search, each one idea the project
 * had to work out, played on the real ocean with its real places labelled.
 *
 *   1. Go with the flow   drift = current + 2 % of the wind; then catch the buoy, which keeps
 *                         drifting the way the arrows said, with a helicopter that turns like a
 *                         real one (7 degrees a second)
 *   2. Between the dots   the current is known only every 8 km and every 3 hours; move the mouse
 *                         to see how the dots around a spot are blended, drop the buoy, and
 *                         watch the forecast and the real buoy drift apart
 *   3. 10,000 guesses     the model's small error persists, so one forecast becomes a cloud;
 *                         size the error with a slider until the cloud catches the real buoys
 *                         without growing too big to search
 *
 * Reworked 14 Oct after open-day testers found the missions confusing: the buoy now flies on
 * from where mission 1 left it, mission 2 is explored with the mouse instead of guessed, and
 * mission 3 shows the real buoys while the slider moves. The sea is the game's own crisp dark
 * map at whole zoom levels, and each mission cuts in from the Triangle instead of a blurry
 * zoom across seven levels of half-loaded tiles.
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
  segmentDistance, toLatLon, toMetres, trialBuoys,
} from './tutorialMaths.js';
import { headingFromKeys, keyDirection } from './searchRun.js';

const $ = (id) => document.getElementById(id);
/** Metres one screen pixel covers at this latitude and zoom (Web Mercator), as scene.js. */
const metresPerPixel = (map, lat) => (40075016.686 * Math.cos((lat * Math.PI) / 180)) / (256 * 2 ** map.getZoom());
const ease = (u) => { const x = Math.max(0, Math.min(1, u)); return x * x * (3 - 2 * x); };
const km = (m) => `${(m / 1000).toFixed(1)} km`;
const hhmm = (hours) => {
  const m = Math.round((12 + hours) * 60);
  return `${Math.floor(m / 60)}:${String(m % 60).padStart(2, '0')}`;
};

/** Places to orient by, drawn over the ocean (the tile labels name the land). */
const SEA_LABELS = [
  [27.3, -79.75, 'Gulf Stream ↑', 'stream'], [31.0, -78.6, 'Gulf Stream ↗', 'stream'],
  [25.65, -79.75, 'Florida Current ↑', 'stream'], [28.0, -66.0, 'Sargasso Sea', 'sea'],
  [24.2, -74.0, 'The Bahamas', 'sea'], [32.6, -64.6, 'Bermuda', 'sea'], [26.0, -70.5, 'Bermuda Triangle', 'tri'],
];
const TRIANGLE = [[25.76, -80.19], [32.3, -64.78], [18.47, -66.11], [25.76, -80.19]];

// Mission 1's sea: the Florida Current off Miami, 1.2 m/s north, and 8 m/s of wind from the east.
const M1 = { lat0: 25.8, lon0: -79.62, current: [0, 1.2], wind: [-8, 0], T: 2 * 3600 };

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
      // The game's own dark sea, at whole zoom levels only: a fractional zoom scales every tile.
      this.map = L.map('tutMap', {
        zoomSnap: 1, attributionControl: false, keyboard: false, doubleClickZoom: false, zoomControl: false,
      });
      // The sea drawn from zoom 10 at most: over open water Esri holds some close-up tiles and not
      // others, and a mix of the two zooms showed as a patchwork of greys (Bermuda, 14 Oct).
      esriTiles('Canvas/World_Dark_Gray_Base', { maxNativeZoom: 10, maxZoom: 18 }).addTo(this.map);
      esriTiles('Canvas/World_Dark_Gray_Reference', { maxNativeZoom: 16, maxZoom: 18, opacity: 0.9 }).addTo(this.map);
      L.polyline(TRIANGLE, { color: '#ff3b47', weight: 2, dashArray: '8 8', opacity: 0.7, interactive: false }).addTo(this.map);
      for (const l of SEA_LABELS) label(...l).addTo(this.map);
      this.map.on('mousemove', (e) => {
        this.pointer = e.latlng;
        if (this.flying) this.input = 'mouse';
        this.move?.(e.latlng);
      });
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
    this.move = null;
    this.map?.getContainer().classList.remove('tut-fade');
    this.layers?.clearLayers();
    $('tutControls').innerHTML = '';
    $('tutResult').textContent = '';
    $('tutResult').className = 'tut-result';
    $('tutClock').textContent = '';
  }

  later(ms, fn) { this.timers.push(setTimeout(fn, ms / this.speedUp)); }

  /**
   * How far through an animation of `ms` begun at `t0` a frame at `now` is, 0 to 1. A frame's
   * time can be a moment before t0 (it is when the frame began), so it is held at 0, not below.
   */
  progress(now, t0, ms) { return Math.max(0, Math.min(1, ((now - t0) / ms) * this.speedUp)); }

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

  /** The map centred so that (lat, lon) sits right of the panel, at a whole zoom. */
  frame(lat, lon, zoom) {
    const shiftM = 0.17 * this.map.getSize().x * metresPerPixel({ getZoom: () => zoom }, lat);
    this.map.setView(toLatLon(lat, lon, -shiftM, 0), zoom, { animate: false });
  }

  /**
   * You are here: the whole Triangle with a pulsing mark where the mission is, then a quick
   * cut to it. A cut, not a fly-in: zooming across seven levels draws every tile blown up
   * before its own has loaded, which is what made the old fly-in blurry.
   */
  arrive(lat, lon, zoom, then) {
    this.map.setView([26.5, -72.5], 5, { animate: false });
    const pin = L.marker([lat, lon], {
      interactive: false, icon: L.divIcon({ className: 'tut-pin', html: '<span></span>', iconSize: [30, 30], iconAnchor: [15, 15] }),
    }).addTo(this.layers);
    const box = this.map.getContainer();
    this.later(1400, () => box.classList.add('tut-fade'));
    this.later(1750, () => {
      pin.remove();
      this.frame(lat, lon, zoom);
    });
    this.later(2100, () => box.classList.remove('tut-fade'));
    this.later(2300, then);
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
    const { lat0, lon0, current, wind, T } = M1;
    const drift = driftOf(current, wind);
    const at = (e, n) => toLatLon(lat0, lon0, e, n);
    this.panel(1, 'Go with the flow',
      'Someone is in the water off Miami, in the <b>Florida Current</b>, the start of the Gulf Stream. '
      + 'What moves them? The <b style="color:#5aa9ff">current</b>, and a little of the <b style="color:#9ff0e0">wind</b>: '
      + 'a person floats low, so only about <b>2 %</b> of the wind’s speed pushes them.');
    this.arrive(lat0 + 0.039, lon0, 13, () => {
      const buoy = L.marker(at(0, 0), { icon: DOT('#ff4d6d', 16, 'them'), interactive: false }).addTo(this.layers);
      this.later(300, () => arrow(at(0, 0), at(current[0] * T, current[1] * T), '#5aa9ff', 'current 1.2 m/s: 8.6 km in 2 h').addTo(this.layers));
      this.later(1800, () => arrow(at(0, 0), at(-4000, 0), '#9ff0e0', 'wind 8 m/s from the east', 3, 'below').addTo(this.layers));
      this.later(3300, () => arrow(at(0, 0), at(drift[0] * T, drift[1] * T), '#ff8a3d', '<b>drift = current + 2 % of the wind</b>', 5, 'left').addTo(this.layers));
      this.later(4800, () => {
        const trail = L.polyline([], { color: '#ff8a3d', weight: 3, dashArray: '2 6', interactive: false }).addTo(this.layers);
        const t0 = performance.now();
        const run = (now) => {
          const u = this.progress(now, t0, 4000);
          const p = at(drift[0] * T * u, drift[1] * T * u);
          buoy.setLatLng(p);
          trail.addLatLng(p);
          if (u < 1) this.raf = requestAnimationFrame(run);
          else {
            this.say(`In 2 hours the sea carried them <b>${km(Math.hypot(...drift) * T)}</b>, right to the tip of the orange arrow. `
              + 'The helicopter arrives now, and they are still drifting the same way. Go and get them.');
            this.next('Fly the helicopter →', () => this.mission1Fly());
          }
        };
        this.raf = requestAnimationFrame(run);
      });
    });
  }

  mission1Fly() {
    this.clear();
    const { lat0, lon0, current, wind, T } = M1;
    const drift = driftOf(current, wind);
    const at = (e, n) => toLatLon(lat0, lon0, e, n);
    // They are where mission 1 left them, 2 h on, and keep drifting the same way.
    const from = [drift[0] * T, drift[1] * T];
    this.panel(1, 'Catch the drifting buoy',
      'They are where the drift took them, and still drifting along the <b style="color:#ff8a3d">orange line</b>. '
      + 'Fly over them: point the mouse where you want to go (or use the arrow keys). '
      + `<b>It flies like a real Coast Guard helicopter:</b> ${Math.round(SEARCH_SPEED_MS * 3.6)} km/h, `
      + `and it banks round at ${Math.round(TURN_RATE_DEG_S)}° a second, so it cannot turn on the spot. Aim ahead of them: it only sees a strip 185 m wide.`);
    const SPEED = 20 * this.speedUp;              // sim seconds per real second
    const LIMIT_S = 12 * 60;
    let heli = { east: from[0] - 2400, north: from[1] - 2800, heading: 20 };
    let target = from;
    let t = 0;
    // the 2 h they have drifted, and where the drift takes them next
    L.polyline([at(0, 0), at(...from)], { color: '#ff8a3d', weight: 3, dashArray: '2 6', opacity: 0.6, interactive: false }).addTo(this.layers);
    L.polyline([at(...from), at(from[0] + drift[0] * LIMIT_S, from[1] + drift[1] * LIMIT_S)],
      { color: '#ff8a3d', weight: 4, opacity: 0.9, interactive: false }).addTo(this.layers);
    const buoy = L.marker(at(...from), { icon: DOT('#ff4d6d', 16, 'them'), interactive: false }).addTo(this.layers);
    const strip = L.polyline([at(heli.east, heli.north)], { color: '#ffffff', opacity: 0.4, lineCap: 'round', lineJoin: 'round', interactive: false }).addTo(this.layers);
    const icon = L.marker(at(heli.east, heli.north), { icon: heliIcon('#ffffff', 52), interactive: false, zIndexOffset: 1000 }).addTo(this.layers);
    const restyle = () => strip.setStyle({ weight: Math.max(2, 185.2 / metresPerPixel(this.map, lat0)) });
    const mid = at(from[0] - 1100, from[1] - 1100);
    this.frame(mid[0], mid[1], 14);
    restyle();
    this.map.on('zoomend', restyle);
    this.flying = true;
    this.input = 'keys';
    this.say('Move the mouse to steer. Catch them before the clock runs out.');
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
      target = [from[0] + drift[0] * t, from[1] + drift[1] * t];
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
        this.say(`<b>Found them</b>, ${Math.round(t / 60)} minutes after you arrived, on the line the drift said they would be. `
          + 'That was the easy part: we knew the current exactly. Next: what if we do not?', 'ok');
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
    this.panel(2, 'Read the sea between the dots',
      'We never know the current everywhere. The US Navy’s ocean model gives it only at these dots, '
      + '<b>8 km apart</b>, and only <b>every 3 hours</b>. Watch the 12:00 map turn into the 15:00 one.');
    this.arrive(lat0, lon0, 12, () => {
      const nodes = L.layerGroup().addTo(this.layers);
      const blend = L.layerGroup().addTo(this.layers);
      const ARROW_S = 3 * 3600;
      const drawNodes = (hours) => {
        nodes.clearLayers();
        for (const [x, y, i, j] of field.points()) {
          const [u, v] = field.node(i, j, hours);
          L.circleMarker(at(x, y), { radius: 6, color: '#fff', weight: 2, fillColor: '#5aa9ff', fillOpacity: 1, interactive: false }).addTo(nodes);
          arrow(at(x, y), at(x + u * ARROW_S, y + v * ARROW_S), '#5aa9ff', '', 3).addTo(nodes);
        }
        $('tutClock').textContent = hhmm(hours);
      };
      /** How the four dots round `pos` are blended at `hours`: a line and a share to each, and the result. */
      const drawBlend = (pos, hours) => {
        blend.clearLayers();
        const { i, j, fx, fy } = field.cell(...pos);
        const w = bilinearWeights(fx, fy);
        [[i, j], [i + 1, j], [i, j + 1], [i + 1, j + 1]].forEach(([ci, cj], k) => {
          const c = at(field.x0 + ci * field.dx, field.y0 + cj * field.dy);
          const p = at(...pos);
          L.polyline([p, c], { color: '#fff', weight: 1 + 7 * w[k], opacity: 0.25 + 0.6 * w[k], dashArray: '4 6', interactive: false }).addTo(blend);
          L.marker([(2 * c[0] + p[0]) / 3, (2 * c[1] + p[1]) / 3], { interactive: false, icon: L.divIcon({ className: 'tut-weight', html: `${Math.round(100 * w[k])} %`, iconSize: null }) }).addTo(blend);
        });
        const [u, v] = field.at(...pos, hours);
        arrow(at(...pos), at(pos[0] + u * ARROW_S, pos[1] + v * ARROW_S), '#ff8a3d', 'the blend', 5).addTo(blend);
      };
      // Keep the buoy inside the dots, half a km clear of the outer row.
      const inside = ([e, n]) => [
        Math.max(field.x0 + 500, Math.min(field.x0 + 2 * field.dx - 500, e)),
        Math.max(field.y0 + 500, Math.min(field.y0 + 2 * field.dy - 500, n)),
      ];
      const t0 = performance.now();
      const run = (now) => {
        const u = this.progress(now, t0, 4500);
        drawNodes(3 * ease(u));
        if (u < 1) this.raf = requestAnimationFrame(run);
        else this.later(600, explore);
      };
      const explore = () => {
        drawNodes(0);
        $('tutText').innerHTML = 'So what is the current <b>between</b> the dots? We blend the four around the spot: '
          + '<b>the nearer a dot, the more it counts</b>. <b>Move your mouse</b> between the dots and watch the '
          + '<b style="color:#ff8a3d">orange arrow</b>. Then <b>click to drop the buoy</b>.';
        this.say('The % on each line is how much that dot counts here.');
        let pos = [2600, -1300];
        const buoy = L.marker(at(...pos), { icon: DOT('#ff4d6d', 14, 'the buoy'), interactive: false }).addTo(this.layers);
        drawBlend(pos, 0);
        this.move = (ll) => {
          pos = inside(toMetres(lat0, lon0, ll.lat, ll.lng));
          buoy.setLatLng(at(...pos));
          drawBlend(pos, 0);
        };
        this.click = (ll) => {
          this.move = null;
          this.click = null;
          pos = inside(toMetres(lat0, lon0, ll.lat, ll.lng));
          buoy.remove();
          driftAway(pos);
        };
      };
      const driftAway = (pos) => {
        $('tutText').innerHTML = 'The forecast follows the blend for 3 hours, the dots turning as the hours pass. '
          + 'The real buoy feels the same sea, plus the small error every forecast has.';
        const model = field.path(...pos, 3);
        // the real buoy: the same field plus a small error that keeps pushing one way (D030's size)
        const real = field.path(...pos, 3, { extra: [0.13, -0.18] });
        const modelLine = L.polyline([], { color: '#ffffff', weight: 3, dashArray: '6 6', interactive: false }).addTo(this.layers);
        const realLine = L.polyline([], { color: '#ff4d6d', weight: 4, interactive: false }).addTo(this.layers);
        const fc = L.marker(at(...pos), { icon: DOT('#ffffff', 12), interactive: false }).addTo(this.layers);
        const rb = L.marker(at(...pos), { icon: DOT('#ff4d6d', 14), interactive: false }).addTo(this.layers);
        const t1 = performance.now();
        const go = (now) => {
          const u = this.progress(now, t1, 6000);
          const k = Math.round(u * (model.length - 1));
          const hours = (3 * k) / (model.length - 1);
          drawNodes(hours);
          drawBlend(model[k], hours);
          modelLine.setLatLngs(model.slice(0, k + 1).map((p) => at(...p)));
          realLine.setLatLngs(real.slice(0, k + 1).map((p) => at(...p)));
          fc.setLatLng(at(...model[k]));
          rb.setLatLng(at(...real[k]));
          if (u < 1) {
            this.raf = requestAnimationFrame(go);
            return;
          }
          blend.clearLayers();
          fc.setIcon(DOT('#ffffff', 12, 'our forecast'));
          rb.setIcon(DOT('#ff4d6d', 16, 'the real buoy'));
          const apart = Math.hypot(model.at(-1)[0] - real.at(-1)[0], model.at(-1)[1] - real.at(-1)[1]);
          this.say(`Every blend done right, and after 3 hours the real buoy is still <b>${km(apart)}</b> from our forecast. `
            + 'The model itself is a little wrong, by about 0.2 m/s, and the error keeps pushing the same way for hours. '
            + 'Next: what we do about that.', 'ok');
          this.next('Mission 3 →', () => this.mission3());
        };
        this.raf = requestAnimationFrame(go);
      };
      this.raf = requestAnimationFrame(run);
    });
  }

  /* ------------------------------------------------------------ mission 3 */

  mission3() {
    this.clear();
    this.step = 3;
    const [lat0, lon0] = [32.18, -64.98];               // south-west of Bermuda
    const at = (e, n) => toLatLon(lat0, lon0, e, n);
    const T = 2 * 3600;
    this.panel(3, '10,000 guesses',
      'The model is a little wrong, and we never know which way. So we run it 10,000 times, each with its own small '
      + 'error, and search the whole <b>cloud</b>. <b>Drag the slider</b> to size the error: the dashed circle must catch '
      + '<b>at least 4 of these 5 real buoys</b>, but stay small enough for one helicopter to search (<b>80 km²</b>).');
    this.arrive(lat0, lon0, 12, () => {
      const cloudLayer = L.layerGroup().addTo(this.layers);
      const renderer = L.canvas({ padding: 0.2 });
      const buoys = trialBuoys(T).map(([e, n]) => L.marker(at(e, n), { icon: DOT('#d8d8d8', 14), interactive: false }).addTo(this.layers));
      L.marker(at(0, 0), { icon: DOT('#ffffff', 12, 'forecast'), interactive: false, zIndexOffset: 500 }).addTo(this.layers);
      const wrap = document.createElement('label');
      wrap.className = 'tut-slider';
      wrap.innerHTML = '<span>How wrong could the model be?</span><input type="range" id="tutSigma" min="0" max="100" value="12" />'
        + '<b id="tutSigmaVal"></b>';
      $('tutControls').append(wrap);
      let go = null;
      const sigmaOf = () => 0.02 * (40 ** (Number($('tutSigma').value) / 100));   // 0.02 to 0.8 m/s
      const draw = () => {
        const s = sigmaOf();
        $('tutSigmaVal').textContent = `${s.toFixed(2)} m/s, so up to about ${km(2.1 * s * T)} after 2 h`;
        const j = judgeCloud(s, T);
        cloudLayer.clearLayers();
        const [e, n] = j.cloud;
        for (let k = 0; k < e.length; k += 2) {
          L.circleMarker(at(e[k], n[k]), { renderer, radius: 1.6, stroke: false, fillColor: '#7fe6ff', fillOpacity: 0.55, interactive: false }).addTo(cloudLayer);
        }
        L.circle(at(0, 0), { radius: j.r90, color: '#fff', weight: 2, dashArray: '6 6', fill: false, interactive: false }).addTo(cloudLayer);
        buoys.forEach((b, k) => b.setIcon(DOT(j.caught[k] ? '#04c951' : '#ff4d6d', 14)));
        const tally = `Caught <b>${j.hits} of 5</b> (need 4) · cloud <b>${Math.round(j.areaKm2)} km²</b> (limit 80). `;
        if (j.verdict === 'small') {
          this.say(`${tally}<br /><b>Too sure of itself:</b> the real buoys drift outside it. Make the error bigger.`, 'bad');
        } else if (j.verdict === 'big') {
          this.say(`${tally}<br /><b>Too vague:</b> far too much sea for one helicopter. Make it smaller.`, 'bad');
        } else {
          this.say(`${tally}<br /><b>Just right.</b> We sized ours the same way, on thousands of real drifting buoys: `
            + `<b>${SIGMA_TRUE} m/s</b>, and 9 in 10 land inside the cloud.`, 'ok');
        }
        if (j.verdict === 'right' && !go) go = this.next('Now the real search →', () => this.finish());
        if (j.verdict !== 'right' && go) { go.remove(); go = null; }
      };
      let queued = false;
      $('tutSigma').addEventListener('input', () => {
        if (queued) return;
        queued = true;
        requestAnimationFrame(() => { queued = false; draw(); });
      });
      draw();
    });
  }
}
