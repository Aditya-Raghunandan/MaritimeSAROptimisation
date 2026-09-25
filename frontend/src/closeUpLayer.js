/**
 * closeUpLayer.js -- the close-up view on the map (issue #79).
 *
 * From zoom 13 the Search view fades in a sea of its own, and from 13.5 it is the whole
 * view. Two canvases, both pinned to the water in metres:
 *
 *   the sea    seaGL.js on the graphics card: waves, whitecaps, light, the water's own
 *              slow patches, thinning out near land;
 *   the life   Sargassum in windrows, and now and then an animal (wildlife.js).
 *
 * THREE CLOCKS, on purpose. What the water carries -- the patches, the whitecaps, the
 * weed, the animals' starting points -- moves in SEARCH time, at the real current (and,
 * for the weed, current + 2 % of wind, the drift model), so its speed on screen matches
 * the helicopter's and the marker's. The waves and the animals themselves move in REAL
 * time, because waves played at 120 times speed look like nothing on earth. And the
 * light follows the SITE's clock: the sun where it was at that moment.
 *
 * Entering and leaving close up is reported through `onEnter` / `onExit`, so the page
 * can swap the basemap and legends (main.js). Where WebGL is missing, the 2-D texture
 * of oceanLayer.js stands in for the sea; the weed and the animals are drawn either way.
 */

import L from 'leaflet';

import {
  WEED_TILE_M, closeUpWeight, floaterStep, isCloseUp, metresPerPixel, nightness, seededRandom,
  peakWavelengthM, sunPosition, weedRows, whitecapFraction,
} from './closeUp.js';
import { ALPHA } from './drift.js';
import { OceanLayer } from './oceanLayer.js';
import { M_PER_DEG } from './pointDrift.js';
import { createSea } from './seaGL.js';
import { Wildlife } from './wildlife.js';
import { drawSighting } from './wildlifeDraw.js';

const TO_RAD = Math.PI / 180;
const TAU = Math.PI * 2;

/** The sea's canvas as a share of the screen's pixels: water is smooth. */
const SEA_RES = 0.6;

/** The water/land grid the sea is thinned by: this many samples, over the view and more. */
const LAND_COLS = 40;
const LAND_ROWS = 28;
const LAND_PAD = 0.6;

/*
  Weed only close in, as clumps (#83). Further out a windrow is a few pixels wide, and
  drawn as a line it read as "orange lines going over" -- taken for current or wind.
  About zoom 15.5 at these latitudes, so the default follow zoom (15) shows none.
*/
const WEED_MAX_MPP = 3.4;

/** A few Sargassum clumps, drawn once and stamped many times. */
function weedSprites() {
  const out = [];
  for (let s = 0; s < 6; s += 1) {
    const cv = document.createElement('canvas');
    cv.width = 48;
    cv.height = 48;
    const g = cv.getContext('2d');
    const rand = seededRandom(1000 + s);
    const blobs = [];
    const n = 9 + Math.floor(rand() * 8);
    for (let k = 0; k < n; k += 1) {
      const a = rand() * TAU;
      const r = rand() * 13;
      blobs.push([24 + Math.cos(a) * r, 24 + Math.sin(a) * r * 0.75, 3 + rand() * 5]);
    }
    g.fillStyle = 'rgba(0,18,28,0.28)';
    for (const [x, y, r] of blobs) { g.beginPath(); g.arc(x + 2, y + 2.5, r, 0, TAU); g.fill(); }
    for (const [x, y, r] of blobs) {
      const grad = g.createRadialGradient(x - r * 0.3, y - r * 0.3, 0, x, y, r);
      grad.addColorStop(0, '#e9c26a');
      grad.addColorStop(0.6, '#b98a2e');
      grad.addColorStop(1, '#7d561a');
      g.fillStyle = grad;
      g.beginPath();
      g.arc(x, y, r, 0, TAU);
      g.fill();
    }
    // The little air bladders that keep it afloat.
    g.fillStyle = 'rgba(248,222,140,0.9)';
    for (let k = 0; k < 16; k += 1) {
      const [x, y, r] = blobs[Math.floor(rand() * blobs.length)];
      g.beginPath();
      g.arc(x + (rand() - 0.5) * r, y + (rand() - 0.5) * r, 0.9 + rand() * 0.8, 0, TAU);
      g.fill();
    }
    out.push(cv);
  }
  return out;
}

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
    this._weed = null;
    this._closed = false;
    this._wild = new Wildlife({ seed: this.options.seed });
    this._raf = null;
    this._last = null;
    this._clock = 0;
    this._windAngle = null;
    this._land = null;
    this._sprites = null;
    this._frameMs = 0;
    this._frames = 0;
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
    this._sprites = this._sprites ?? weedSprites();
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
      const w = Math.max(1, Math.round(size.x * SEA_RES));
      const h = Math.max(1, Math.round(size.y * SEA_RES));
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
      this._weed = [...this._origin];
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
      this._weed = floaterStep(this._weed[0], this._weed[1], this._current, this._wind, ALPHA, simS);
    }
    const mpp = metresPerPixel(centre.lat, zoom);
    const c = this._toWorld(centre.lat, centre.lng);
    const waterFlow = this._toWorld(this._water[0], this._water[1]);
    const weedFlow = this._toWorld(this._weed[0], this._weed[1]);

    // The wind, turned smoothly so the weed rows swing rather than jump on the hour.
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
      // The wind is eased like the weed's rows, so the sea's texture turns smoothly too.
      const eased = this._windSpeed === undefined ? ws : this._windSpeed + (ws - this._windSpeed) * (1 - Math.exp(-dt / 2));
      this._windSpeed = eased;
      this._landFor(c, size, mpp);
      this._sea.draw({
        mpp: mpp / SEA_RES,
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
      });
    }

    const ctx = this._ctx;
    ctx.setTransform(this._dpr, 0, 0, this._dpr, 0, 0);
    ctx.clearRect(0, 0, size.x, size.y);
    ctx.globalAlpha = fade;
    const view = { size, c, mpp };
    this._drawWeed(ctx, view, weedFlow, wdir, night);
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

  _toScreen(view, x, y) {
    return [view.size.x / 2 + (x - view.c[0]) / view.mpp, view.size.y / 2 - (y - view.c[1]) / view.mpp];
  },

  /** Sargassum in windrows, carried at current + 2 % of wind. */
  _drawWeed(ctx, view, flow, wdir, night) {
    const { size, c, mpp } = view;
    const reach = 260;                        // a row reaches this far out of its tile
    const x0 = c[0] - flow[0] - (size.x * mpp) / 2 - reach;
    const x1 = c[0] - flow[0] + (size.x * mpp) / 2 + reach;
    const y0 = c[1] - flow[1] - (size.y * mpp) / 2 - reach;
    const y1 = c[1] - flow[1] + (size.y * mpp) / 2 + reach;
    if (mpp > WEED_MAX_MPP) return;
    const perp = [wdir[1], -wdir[0]];
    ctx.globalAlpha *= 1 - 0.5 * night;
    for (let tx = Math.floor(x0 / WEED_TILE_M); tx <= Math.floor(x1 / WEED_TILE_M); tx += 1) {
      for (let ty = Math.floor(y0 / WEED_TILE_M); ty <= Math.floor(y1 / WEED_TILE_M); ty += 1) {
        for (const row of weedRows(tx, ty)) {
          const cx = row.x + flow[0];
          const cy = row.y + flow[1];
          // Clumps along the row, bunched and ragged, with gaps: a windrow is patchy.
          const rand = seededRandom(row.seed);
          const step = Math.max(row.clumpM * 1.1, 2.2 * mpp);
          const n = Math.max(2, Math.floor(row.lengthM / step));
          for (let k = 0; k < n; k += 1) {
            if (rand() < 0.45) continue;                // gaps: a windrow is patchy
            const f = k / (n - 1) - 0.5;
            const along = f * row.lengthM + (rand() - 0.5) * step * 1.4;
            const across = (rand() - 0.5) * row.widthM * 3 * (1 - 3 * f * f);
            const [sx, sy] = this._toScreen(view, cx + wdir[0] * along + perp[0] * across,
              cy + wdir[1] * along + perp[1] * across);
            const px = Math.min(18, Math.max(2.2, (row.clumpM * (0.6 + 0.9 * rand()) * 1.8) / mpp));
            if (sx < -px || sy < -px || sx > size.x + px || sy > size.y + px) continue;
            ctx.drawImage(this._sprites[Math.floor(rand() * this._sprites.length)], sx - px / 2, sy - px / 2, px, px);
          }
        }
      }
    }
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
