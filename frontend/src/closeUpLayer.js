/**
 * closeUpLayer.js -- the close-up view on the map (issue #79).
 *
 * From zoom 13 the Search view fades in a sea of its own, and from 13.5 it is the whole
 * view. Two canvases, both pinned to the water in metres:
 *
 *   the sea    seaGL.js on the graphics card: waves, whitecaps, light, the water's own
 *              slow patches, thinning out near land;
 *   the life   streaks of the chosen flow, now and then an animal, and rarely a ship
 *              along an edge (wildlife.js).
 *
 * THREE CLOCKS, on purpose. What the water carries -- the patches, the whitecaps, the
 * animals' starting points -- moves in SEARCH time, at the real current, so its speed on
 * screen matches the helicopter's and the marker's. The waves and the animals themselves move in REAL
 * time, because waves played at 120 times speed look like nothing on earth. And the
 * light follows the SITE's clock: the sun where it was at that moment.
 *
 * Entering and leaving close up is reported through `onEnter` / `onExit`, so the page
 * can swap the basemap and legends (main.js). Where WebGL is missing, the 2-D texture
 * of oceanLayer.js stands in for the sea; streaks, animals and ships are drawn either way.
 */

import L from 'leaflet';

import {
  FLOWS, SEA_RES, closeUpWeight, floaterStep, flowVector, isCloseUp, metresPerPixel, nextSeaRes, nightness,
  peakWavelengthM, streakSpeedPx, sunPosition, whitecapFraction,
} from './closeUp.js';
import { ALPHA } from './drift.js';
import { OceanLayer } from './oceanLayer.js';
import { M_PER_DEG } from './pointDrift.js';
import { createSea } from './seaGL.js';
import { ShipTraffic, Wildlife } from './wildlife.js';
import { drawShip, drawSighting } from './wildlifeDraw.js';

const TO_RAD = Math.PI / 180;
const TAU = Math.PI * 2;


/** The water/land grid the sea is thinned by: this many samples, over the view and more. */
const LAND_COLS = 40;
const LAND_ROWS = 28;
const LAND_PAD = 0.6;

/** Turn angle `a` towards `b` by fraction `k`, the short way round. */
function turnTowards(a, b, k) {
  const d = Math.atan2(Math.sin(b - a), Math.cos(b - a));
  return a + d * k;
}

export const CloseUpLayer = L.Layer.extend({
  options: {
    onEnter: null,          // () => void, on reaching close up
    onExit: null,           // () => void, on leaving it
    waterAt: null,          // (lat, lon) => true | false | null (unknown)
    onSighting: null,       // (label | null) => void, when an animal comes or goes
    seed: 20260925,
  },

  initialize(options) {
    L.setOptions(this, options);
    this._current = [0, 0];
    this._wind = [0, 0];
    this._rate = 0;
    this._timeMs = null;
    this._origin = null;
    this._water = null;
    this._closed = false;
    this._wild = new Wildlife({ seed: this.options.seed });
    this._ships = new ShipTraffic({ seed: this.options.seed + 1 });
    this._avoid = [];        // [lat, lon] a ship must keep clear of: buoy, helicopter, marker, datum
    this._raf = null;
    this._last = null;
    this._clock = 0;
    this._windAngle = null;
    this._land = null;
    this._frameMs = 0;
    this._frames = 0;
    this._res = SEA_RES;     // the sea's share of the screen's pixels; lowered if it lags
    this._flow = 'drift';    // which flow the streaks show (#85)
    this._streaks = [];
    this._tick = this._tick.bind(this);
  },

  onAdd(map) {
    this._map = map;
    const seaPane = map.getPane('closeUpSea') || map.createPane('closeUpSea');
    seaPane.style.zIndex = 380;          // over the basemap, under the search's marks
    seaPane.style.pointerEvents = 'none';
    const lifePane = map.getPane('closeUpLife') || map.createPane('closeUpLife');
    lifePane.style.zIndex = 395;
    lifePane.style.pointerEvents = 'none';
    this._lifeCanvas = L.DomUtil.create('canvas', 'closeup-life leaflet-zoom-animated', lifePane);
    this._ctx = this._lifeCanvas.getContext('2d');
    this._glCanvas = L.DomUtil.create('canvas', 'closeup-sea leaflet-zoom-animated', seaPane);
    this._sea = createSea(this._glCanvas);
    if (this._sea) {
      this._onLost = (e) => { e.preventDefault(); this._fallBack(); };
      this._glCanvas.addEventListener('webglcontextlost', this._onLost);
    } else {
      this._fallBack();
    }
    map.on('zoomanim', this._onZoomAnim, this);
    map.on('zoomend moveend resize viewreset', this._onView, this);
    this._resize();
    this._onView();
  },

  onRemove(map) {
    map.off('zoomanim', this._onZoomAnim, this);
    map.off('zoomend moveend resize viewreset', this._onView, this);
    this._stop();
    if (this._closed) {
      this._closed = false;
      if (this.options.onExit) this.options.onExit();
    }
    if (this._fallback && map.hasLayer(this._fallback)) map.removeLayer(this._fallback);
    if (this._sea) this._sea.dispose();
    this._sea = null;
    if (this._glCanvas) this._glCanvas.remove();
    this._lifeCanvas.remove();
    this._glCanvas = null;
    this._land = null;
    this._map = null;
  },

  /** The 2-D sea instead of the shader: no WebGL, or the context was lost. */
  _fallBack() {
    if (this._sea) { try { this._sea.dispose(); } catch { /* already gone */ } }
    this._sea = null;
    if (this._glCanvas) { this._glCanvas.remove(); this._glCanvas = null; }
    this._fallback = this._fallback ?? new OceanLayer();
    if (this._map && !this._map.hasLayer(this._fallback)) this._map.addLayer(this._fallback);
    this._fallback.setConditions({ current: this._current, wind: this._wind, rate: this._rate });
  },

  /**
   * The forcing where the view is looking, how fast search time is passing (0 when
   * paused), and the site's moment, which sets the sun.
   */
  setConditions({ current, wind, rate, timeMs }) {
    if (current && current.every(Number.isFinite)) this._current = current;
    if (wind && wind.every(Number.isFinite)) this._wind = wind;
    if (Number.isFinite(rate)) this._rate = rate;
    if (Number.isFinite(timeMs)) this._timeMs = timeMs;
    if (this._fallback) this._fallback.setConditions({ current, wind, rate });
  },

  /** Streak `kind` ('drift', 'current', 'wind') or none ('off'). */
  setFlow(kind) {
    this._flow = FLOWS[kind] ? kind : 'off';
    this._streaks = [];
  },

  /** Which flow the streaks show. */
  flow() {
    return this._flow;
  },

  /** The points a passing ship keeps clear of, [lat, lon] each (#86). */
  setAvoid(latlngs) {
    this._avoid = latlngs ?? [];
  },

  /** Call up a ship now, if an edge is free: for looking at it, and for a demonstration. */
  summonShip() {
    if (!this._closed) return false;
    this._ships.summon();
    return true;
  },

  /** Whether the view is close up now. */
  isClose() {
    return this._closed;
  },

  /** Mean time to draw one frame over the last second, ms: measured, for the record. */
  frameMs() {
    return this._frameMs;
  },

  /** What the sea shows: the sun, and the share of the sea that is white. */
  describe() {
    const c = this._map ? this._map.getCenter() : null;
    const sun = c && this._timeMs !== null ? sunPosition(this._timeMs, c.lat, c.lng) : null;
    return {
      sunElevationDeg: sun ? sun.elevationDeg : null,
      night: sun ? nightness(sun.elevationDeg) : 0,
      whitecaps: whitecapFraction(Math.hypot(this._wind[0], this._wind[1])),
      sighting: this._wild.active ? this._wild.active.label : null,
      webgl: Boolean(this._sea),
    };
  },

  /**
   * Call up an animal now, whatever the scheduler says: for looking at each one while
   * working on it, and for a demonstration. Answers false if not close up.
   */
  summon(kind) {
    if (!this._closed) return false;
    return Boolean(this._wild.summon(kind));
  },

  /* ------------------------------------------------------------ view changes */

  _onView() {
    if (!this._map) return;
    this._resize();
    const zoom = this._map.getZoom();
    const close = isCloseUp(zoom);
    if (close !== this._closed) {
      this._closed = close;
      if (!close) this._wild.rest();
      const cb = close ? this.options.onEnter : this.options.onExit;
      if (cb) cb();
    }
    if (closeUpWeight(zoom) > 0) this._start();
    else { this._stop(); this._clear(); }
  },

  /** Scale the canvases with Leaflet's zoom animation, as its own renderers do. */
  _onZoomAnim(e) {
    if (!this._drawCenter) return;
    const map = this._map;
    const scale = map.getZoomScale(e.zoom, this._drawZoom);
    const offset = map.getSize().multiplyBy(-0.5 * scale)
      .add(map.project(this._drawCenter, e.zoom))
      .subtract(map._getNewPixelOrigin(e.center, e.zoom));
    for (const c of [this._glCanvas, this._lifeCanvas]) if (c) L.DomUtil.setTransform(c, offset, scale);
  },

  _resize() {
    const size = this._map.getSize();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (this._glCanvas) {
      const w = Math.max(1, Math.round(size.x * this._res));
      const h = Math.max(1, Math.round(size.y * this._res));
      if (this._glCanvas.width !== w || this._glCanvas.height !== h) {
        this._glCanvas.width = w;
        this._glCanvas.height = h;
      }
      this._glCanvas.style.width = `${size.x}px`;
      this._glCanvas.style.height = `${size.y}px`;
    }
    const lw = Math.round(size.x * dpr);
    const lh = Math.round(size.y * dpr);
    if (this._lifeCanvas.width !== lw || this._lifeCanvas.height !== lh) {
      this._lifeCanvas.width = lw;
      this._lifeCanvas.height = lh;
    }
    this._lifeCanvas.style.width = `${size.x}px`;
    this._lifeCanvas.style.height = `${size.y}px`;
    this._dpr = dpr;
  },

  _start() {
    if (this._raf || !this._map) return;
    this._last = null;
    if (this._glCanvas) this._glCanvas.style.visibility = '';
    this._lifeCanvas.style.visibility = '';
    this._raf = requestAnimationFrame(this._tick);
  },

  _stop() {
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = null;
  },

  _clear() {
    if (this._glCanvas) this._glCanvas.style.visibility = 'hidden';
    this._lifeCanvas.style.visibility = 'hidden';
  },

  _tick(now) {
    this._raf = null;
    if (!this._map) return;
    const dt = this._last === null ? 0 : Math.min((now - this._last) / 1000, 0.1);
    this._last = now;
    this._clock = (this._clock + dt) % 3600;
    // During Leaflet's zoom animation the canvases are scaled with the map instead.
    if (!this._map._animatingZoom) {
      const t0 = performance.now();
      this._draw(dt);
      this._measure(performance.now() - t0, now);
    }
    if (this._map && closeUpWeight(this._map.getZoom()) > 0) this._raf = requestAnimationFrame(this._tick);
  },

  _measure(ms, now) {
    this._frames += 1;
    this._frameSum = (this._frameSum ?? 0) + ms;
    if (!this._frameAt) this._frameAt = now;
    if (now - this._frameAt >= 1000) {
      this._frameMs = this._frameSum / this._frames;
      this._frames = 0;
      this._frameSum = 0;
      this._frameAt = now;
      this._lifeCanvas.dataset.frameMs = this._frameMs.toFixed(2);
      this._lifeCanvas.dataset.seaRes = this._res.toFixed(2);
    }
  },

  /* ------------------------------------------------------------ one frame */

  _toWorld(lat, lon) {
    const [lat0, lon0] = this._origin;
    return [(lon - lon0) * M_PER_DEG * Math.cos(lat0 * TO_RAD), (lat - lat0) * M_PER_DEG];
  },

  _fromWorld(x, y) {
    const [lat0, lon0] = this._origin;
    return [lat0 + y / M_PER_DEG, lon0 + x / (M_PER_DEG * Math.cos(lat0 * TO_RAD))];
  },

  _draw(dt) {
    const map = this._map;
    const zoom = map.getZoom();
    const fade = closeUpWeight(zoom);
    const size = map.getSize();
    const centre = map.getCenter();
    // One origin per scene; start again if the view has gone far from it.
    if (!this._origin || Math.abs(centre.lat - this._origin[0]) > 2 || Math.abs(centre.lng - this._origin[1]) > 2) {
      this._origin = [centre.lat, centre.lng];
      this._water = [...this._origin];
      this._land = null;
    }
    const topLeft = map.containerPointToLayerPoint([0, 0]);
    if (this._glCanvas) L.DomUtil.setPosition(this._glCanvas, topLeft);
    L.DomUtil.setPosition(this._lifeCanvas, topLeft);
    this._drawCenter = centre;
    this._drawZoom = zoom;

    // What the water carries moves in search time.
    const simS = dt * this._rate;
    if (simS > 0) {
      this._water = floaterStep(this._water[0], this._water[1], this._current, this._wind, 0, simS);
    }
    const mpp = metresPerPixel(centre.lat, zoom);
    const c = this._toWorld(centre.lat, centre.lng);
    const waterFlow = this._toWorld(this._water[0], this._water[1]);

    // The wind, turned smoothly so the sea and the streaks swing rather than jump on the hour.
    const ws = Math.hypot(this._wind[0], this._wind[1]);
    const target = ws > 0.2 ? Math.atan2(this._wind[0], this._wind[1]) : (this._windAngle ?? 0);
    this._windAngle = this._windAngle === null ? target : turnTowards(this._windAngle, target, 1 - Math.exp(-dt / 2));
    const wdir = [Math.sin(this._windAngle), Math.cos(this._windAngle)];

    // The sun where the view is, at the site's moment.
    const sun = this._timeMs === null
      ? { elevationDeg: 55, azimuthDeg: 160 }
      : sunPosition(this._timeMs, centre.lat, centre.lng);
    const night = nightness(sun.elevationDeg);
    const el = sun.elevationDeg * TO_RAD;
    const az = sun.azimuthDeg * TO_RAD;

    if (this._sea) {
      // The wind's speed is eased too, so the sea's roughness changes smoothly.
      const eased = this._windSpeed === undefined ? ws : this._windSpeed + (ws - this._windSpeed) * (1 - Math.exp(-dt / 2));
      this._windSpeed = eased;
      this._landFor(c, size, mpp);
      const seaParams = {
        centre: c,
        flow: waterFlow,
        time: this._clock,
        windDir: wdir,
        windSpeed: eased,
        peak: Math.max(8, peakWavelengthM(eased)),
        foam: whitecapFraction(ws),
        sun: [Math.sin(az) * Math.cos(el), Math.cos(az) * Math.cos(el), Math.sin(el)],
        // No ephemeris for the moon: a fixed moonlight across the sky from the sun, faint,
        // so a night search is dark blue with some glitter rather than a black screen.
        moon: [-Math.sin(az) * 0.8, -Math.cos(az) * 0.8, 0.6, 0.45 * night],
        night,
        fade,
      };
      if (!this._calibrated) this._calibrate(seaParams, mpp);
      this._sea.draw({ ...seaParams, mpp: mpp / this._res });
    }

    const ctx = this._ctx;
    ctx.setTransform(this._dpr, 0, 0, this._dpr, 0, 0);
    ctx.clearRect(0, 0, size.x, size.y);
    ctx.globalAlpha = fade;
    const view = { size, c, mpp };
    this._drawStreaks(ctx, view, dt);
    ctx.globalAlpha = fade;
    this._drawShip(ctx, view, dt, zoom, night, el, az);
    ctx.globalAlpha = fade;
    this._drawWildlife(ctx, view, dt, zoom, waterFlow, wdir, night, el, az);
  },

  /** Keep the sea's water/land grid covering the view, re-sampling when it leaves it. */
  _landFor(c, size, mpp) {
    const waterAt = this.options.waterAt;
    if (!waterAt || !this._sea) return;
    const hw = (size.x * mpp) / 2;
    const hh = (size.y * mpp) / 2;
    const view = [c[0] - hw, c[1] - hh, c[0] + hw, c[1] + hh];
    const now = performance.now();
    const b = this._land && this._land.box;
    const covers = b && view[0] >= b[0] && view[1] >= b[1] && view[2] <= b[2] && view[3] <= b[3]
      && (b[2] - b[0]) < 5 * (view[2] - view[0]);
    if (covers && !this._land.pending) return;
    if (this._land && this._land.pending && now - this._land.at < 1000) return;
    const px = hw * (1 + 2 * LAND_PAD);
    const py = hh * (1 + 2 * LAND_PAD);
    const box = [c[0] - px, c[1] - py, c[0] + px, c[1] + py];
    const mask = new Uint8Array(LAND_COLS * LAND_ROWS);
    let known = 0;
    for (let j = 0; j < LAND_ROWS; j += 1) {
      const y = box[1] + ((j + 0.5) / LAND_ROWS) * (box[3] - box[1]);
      for (let i = 0; i < LAND_COLS; i += 1) {
        const x = box[0] + ((i + 0.5) / LAND_COLS) * (box[2] - box[0]);
        const [lat, lon] = this._fromWorld(x, y);
        const w = waterAt(lat, lon);
        if (w !== null) known += 1;
        mask[j * LAND_COLS + i] = w === false ? 0 : 255;
      }
    }
    this._sea.setLand(mask, LAND_COLS, LAND_ROWS, box);
    this._land = { box, pending: known === 0, at: now };
  },

  /**
   * Once, when the sea is first drawn: time three frames forced to finish on the GPU, and
   * step the resolution down while one costs more than 8 ms (closeUp.nextSeaRes). Cost,
   * not frame rate, so a throttled pane or a 30 fps cap cannot make a fast machine blurry.
   */
  _calibrate(params, mpp) {
    this._calibrated = true;
    for (let step = 0; step < 4; step += 1) {
      this._sea.draw({ ...params, mpp: mpp / this._res });
      this._sea.finish();
      const times = [];
      for (let k = 0; k < 3; k += 1) {
        const t0 = performance.now();
        this._sea.draw({ ...params, mpp: mpp / this._res });
        this._sea.finish();
        times.push(performance.now() - t0);
      }
      times.sort((a, b) => a - b);
      const res = nextSeaRes(this._res, times[1]);
      this._lifeCanvas.dataset.seaMs = times[1].toFixed(2);
      if (res === this._res) break;
      this._res = res;
      this._resize();
    }
    this._lifeCanvas.dataset.seaRes = this._res.toFixed(2);
  },

  _toScreen(view, x, y) {
    return [view.size.x / 2 + (x - view.c[0]) / view.mpp, view.size.y / 2 - (y - view.c[1]) / view.mpp];
  },

  /**
   * Streaks of the chosen flow (#85): dense, tapered like comets, pinned to the water so
   * they stay right while the map follows the helicopter. They run at a speed that shows
   * how strong the flow is, as the site's other particles do, not at the playback clock.
   */
  _drawStreaks(ctx, view, dt) {
    const f = FLOWS[this._flow];
    if (!f) return;
    const [u, v] = flowVector(this._flow, this._current, this._wind, ALPHA);
    const speed = Math.hypot(u, v);
    const pxps = streakSpeedPx(speed, f.refMs);
    if (pxps <= 0) return;
    const { size, c, mpp } = view;
    const dir = [u / speed, v / speed];
    const want = Math.round((size.x * size.y) / 3400);
    const spawn = (s) => {
      s.x = c[0] + (Math.random() - 0.5) * size.x * mpp;
      s.y = c[1] + (Math.random() - 0.5) * size.y * mpp;
      s.age = 0;
      s.life = 1.6 + Math.random() * 2.4;
      return s;
    };
    while (this._streaks.length < want) {
      const s = spawn({});
      s.age = Math.random() * s.life;
      this._streaks.push(s);
    }
    this._streaks.length = Math.min(this._streaks.length, want);
    const step = pxps * mpp * dt;
    const tail = Math.min(46, pxps * 0.32);                 // px
    const sdx = dir[0];
    const sdy = -dir[1];
    const heads = [[], []];                                 // fading in or out, and full
    for (const s of this._streaks) {
      s.age += dt;
      s.x += dir[0] * step;
      s.y += dir[1] * step;
      const [hx, hy] = this._toScreen(view, s.x, s.y);
      if (s.age > s.life || hx < -60 || hy < -60 || hx > size.x + 60 || hy > size.y + 60) {
        spawn(s);
        continue;
      }
      const fade = Math.min(1, s.age / 0.4, (s.life - s.age) / 0.6);
      heads[fade > 0.55 ? 1 : 0].push([hx, hy]);
    }
    const [r, g, b] = f.rgb;
    ctx.lineCap = 'round';
    ctx.lineWidth = f.width;
    // Three segments from tail to head, fainter towards the tail, in two strengths.
    for (const [bucket, strength] of [[0, 0.45], [1, 1]]) {
      for (const [from, to, a] of [[1, 0.66, 0.16], [0.66, 0.33, 0.34], [0.33, 0, 0.7]]) {
        ctx.strokeStyle = `rgba(${r},${g},${b},${(a * strength).toFixed(3)})`;
        ctx.beginPath();
        for (const [hx, hy] of heads[bucket]) {
          ctx.moveTo(hx - sdx * tail * from, hy - sdy * tail * from);
          ctx.lineTo(hx - sdx * tail * to, hy - sdy * tail * to);
        }
        ctx.stroke();
      }
    }
  },

  /**
   * Now and then a ship (#86), along the top or bottom edge only -- never through the middle
   * of the view, never through a person. Its lane is in the view, not on the water, so a
   * map following the helicopter cannot carry it inwards; and if the buoy, the helicopter,
   * the marker or the datum comes near it anyway, it leaves early.
   */
  _drawShip(ctx, view, dt, zoom, night, el, az) {
    if (!isCloseUp(zoom)) return;
    const { size } = view;
    const avoid = this._avoid.map((ll) => {
      const p = this._map.latLngToContainerPoint(ll);
      return [p.x, p.y];
    });
    // One clearance for planning the lane and for leaving early: the whole ship stays clear.
    const L = 120 * 2 ** ((zoom - 15) * 0.5);
    const clear = 40 + L * 0.6;
    this._ships.step(dt, { width: size.x, height: size.y, avoid, margin: clear });
    const s = this._ships.active;
    if (!s) return;
    const age = this._ships.clock - s.bornS;
    const f = age / s.crossS;
    const [ax, ay] = [s.a[0] * size.x, s.a[1] * size.y];
    const [bx, by] = [s.b[0] * size.x, s.b[1] * size.y];
    const x = ax + (bx - ax) * f;
    const y = ay + (by - ay) * f;
    if (!s.leaving && avoid.some(([px, py]) => Math.hypot(px - x, py - y) < clear)) {
      s.leaving = true;
      this._ships.leave(1.5);
    }
    const fade = Math.max(0, Math.min(1, age / 2, (s.durationS - age) / 1.5));
    drawShip(ctx, s, age, {
      x, y, heading: Math.atan2(by - ay, bx - ax) + Math.PI / 2, L, fade: fade * ctx.globalAlpha, night,
      sun: { up: Math.max(0, Math.sin(el)) * (1 - night), dx: Math.sin(az), dy: -Math.cos(az) },
    });
  },

  /** An animal now and then, pinned to the water where it appeared. */
  _drawWildlife(ctx, view, dt, zoom, flow, wdir, night, el, az) {
    if (!isCloseUp(zoom)) return;
    const month = new Date(this._timeMs ?? Date.now()).getUTCMonth();
    const born = this._wild.step(dt, { month });
    if (born) {
      const sx = born.x * view.size.x;
      const sy = born.y * view.size.y;
      born.q = [view.c[0] + (sx - view.size.x / 2) * view.mpp - flow[0],
        view.c[1] - (sy - view.size.y / 2) * view.mpp - flow[1]];
    }
    const s = this._wild.active;
    const label = s && s.q ? s.label : null;
    if (label !== this._label) {
      this._label = label;
      if (this.options.onSighting) this.options.onSighting(label);
    }
    if (!s || !s.q) return;
    drawSighting(ctx, s, this._wild.clock - s.bornS, {
      anchor: this._toScreen(view, s.q[0] + flow[0], s.q[1] + flow[1]),
      size: (base) => base * 2 ** ((zoom - 15) * 0.5),
      night,
      sun: { up: Math.max(0, Math.sin(el)) * (1 - night), dx: Math.sin(az), dy: -Math.cos(az) },
      wind: [wdir[0], -wdir[1]],
      fade: ctx.globalAlpha,
    });
  },
});
