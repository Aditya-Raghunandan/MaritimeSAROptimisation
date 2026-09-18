/**
 * particles.js -- animated streaklines, the thing that makes a field read as flow.
 *
 * A painted raster shows where the wind is strong. Arrows show which way it
 * points. Neither shows that it is MOVING, and movement is what makes a
 * circulation legible at a glance -- you see the gyre turn rather than working
 * it out from arrowheads. It is also the single biggest difference between a
 * chart and something worth putting on a screen behind you while you talk.
 *
 * HOW IT WORKS, which is the standard streakline trick:
 *
 *   1. Scatter N particles over the box.
 *   2. Each animation frame, look up (u, v) where each particle is and step it
 *      forward by dt. Draw a short segment from where it was to where it is.
 *   3. Do NOT clear the canvas. Paint it with a translucent black instead, so
 *      previous segments fade over a few frames. That fade is the tail.
 *   4. Respawn each particle after a random lifetime, so the field does not
 *      slowly drain into its convergence zones and leave the rest bare.
 *
 * THE STEP IS IN DEGREES PER SECOND, NOT PIXELS PER FRAME, and that matters
 * here more than on most maps. Converting m/s to degrees divides the eastward
 * component by cos(latitude): across 17-36 N that is a 13 % difference between
 * the bottom of the box and the top. Stepping in pixels would make the same
 * wind appear to blow faster at the top of the screen than the bottom -- an
 * artefact of Mercator that a viewer would read as physics.
 *
 * RESPAWN IS SEEDED AND UNIFORM over the box, not over the visible screen, so
 * panning does not change where particles come from. A particle that leaves the
 * data is retired rather than clamped: clamping piles them against the edges
 * and draws a bright false rim around the domain.
 */

import L from 'leaflet';

const METRES_PER_DEGREE = 111_320;

export const ParticleLayer = L.Layer.extend({
  /**
   * @param {object} field  anything with .grid and .vector(frame, j, i)
   * @param {object} opts
   *   count      how many particles
   *   speed      seconds of simulated drift per animation frame
   *   fade       0-1, how much of the previous frame survives
   *   maxAgeMs   respawn after about this long
   */
  initialize(field, opts = {}) {
    this._field = field;
    this._frame = 0;
    this._count = opts.count ?? 2600;
    this._speed = opts.speed ?? 900;
    this._fade = opts.fade ?? 0.94;
    this._maxAge = opts.maxAgeMs ?? 2600;
    this._colour = opts.colour ?? 'rgba(255, 255, 255, 0.72)';
    this._particles = [];
    this._raf = null;
  },

  onAdd(map) {
    this._map = map;
    this._canvas = L.DomUtil.create('canvas', 'leaflet-particle-layer leaflet-layer');
    L.DomUtil.addClass(this._canvas, 'leaflet-zoom-hide');
    map.getPanes().overlayPane.appendChild(this._canvas);
    this._canvas.style.zIndex = '160';

    this._seed();
    map.on('move', this._reset, this);
    map.on('zoomend viewreset resize', this._reset, this);
    this._reset();
    this._start();
    return this;
  },

  onRemove(map) {
    this._stop();
    map.off('move', this._reset, this);
    map.off('zoomend viewreset resize', this._reset, this);
    if (this._canvas && this._canvas.parentNode) this._canvas.parentNode.removeChild(this._canvas);
    this._canvas = null;
    return this;
  },

  setFrame(frame) { this._frame = frame; },

  _start() {
    if (this._raf) return;
    let last = performance.now();
    const tick = (now) => {
      // Real elapsed time, not a fixed step: a dropped frame should move the
      // particles further rather than slow the whole field down.
      const dt = Math.min((now - last) / 1000, 0.1);
      last = now;
      this._step(dt);
      this._raf = requestAnimationFrame(tick);
    };
    this._raf = requestAnimationFrame(tick);
  },

  _stop() {
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = null;
  },

  /** Uniform over the DATA box, so panning does not change where they come from. */
  _seed() {
    const g = this._field.grid;
    this._particles = Array.from({ length: this._count }, () => this._spawn(g, true));
  },

  _spawn(g, stagger = false) {
    return {
      lat: g.lat(0) + Math.random() * (g.lat(g.nlat - 1) - g.lat(0)),
      lon: g.lon(0) + Math.random() * (g.lon(g.nlon - 1) - g.lon(0)),
      // Staggered ages on the first seed, or every particle respawns together
      // and the whole field blinks in unison.
      age: stagger ? Math.random() * this._maxAge : 0,
    };
  },

  _reset() {
    if (!this._map || !this._canvas) return;
    const size = this._map.getSize();
    const dpr = window.devicePixelRatio || 1;
    if (this._canvas.width !== size.x * dpr || this._canvas.height !== size.y * dpr) {
      this._canvas.width = size.x * dpr;
      this._canvas.height = size.y * dpr;
      this._canvas.style.width = `${size.x}px`;
      this._canvas.style.height = `${size.y}px`;
    }
    L.DomUtil.setPosition(this._canvas, this._map.containerPointToLayerPoint([0, 0]));
    // Old trails are in the wrong place after a move; keeping them would smear.
    this._canvas.getContext('2d').clearRect(0, 0, size.x, size.y);
  },

  _step(dt) {
    if (!this._map || !this._canvas) return;
    const g = this._field.grid;
    const ctx = this._canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const size = this._map.getSize();

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // The fade IS the tail: paint over the last frame instead of clearing it.
    ctx.globalCompositeOperation = 'destination-in';
    ctx.fillStyle = `rgba(0, 0, 0, ${this._fade})`;
    ctx.fillRect(0, 0, size.x, size.y);
    ctx.globalCompositeOperation = 'source-over';

    ctx.strokeStyle = this._colour;
    ctx.lineWidth = 1.1;
    ctx.lineCap = 'round';
    ctx.beginPath();

    const seconds = this._speed * dt;

    for (let k = 0; k < this._particles.length; k += 1) {
      const p = this._particles[k];
      p.age += dt * 1000;

      const cell = g.cellAt(p.lat, p.lon);
      if (!cell || p.age > this._maxAge) {
        // Retired, not clamped. Clamping piles particles against the edges and
        // draws a bright false rim around the domain.
        this._particles[k] = this._spawn(g);
        continue;
      }

      let u;
      let v;
      try {
        [u, v] = this._field.vector(this._frame, cell.j, cell.i);
      } catch {
        // The chunk is not resident yet. Leave the particle where it is; it
        // will move as soon as the data lands.
        continue;
      }
      if (!Number.isFinite(u) || !Number.isFinite(v)) {
        this._particles[k] = this._spawn(g);
        continue;
      }

      const from = this._map.latLngToContainerPoint([p.lat, p.lon]);

      // m/s -> degrees. The eastward component shrinks with cos(latitude);
      // ignoring it makes the same wind look faster at the top of the box.
      const dLat = (v * seconds) / METRES_PER_DEGREE;
      const dLon = (u * seconds) / (METRES_PER_DEGREE * Math.cos((p.lat * Math.PI) / 180));
      p.lat += dLat;
      p.lon += dLon;

      const to = this._map.latLngToContainerPoint([p.lat, p.lon]);
      // A segment longer than this is a particle that jumped, usually because
      // the map moved mid-step. Drawing it puts a stray streak across the view.
      if (Math.abs(to.x - from.x) + Math.abs(to.y - from.y) > 60) continue;

      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
    }
    ctx.stroke();
  },
});

export function particleLayer(field, opts) {
  return new ParticleLayer(field, opts);
}
