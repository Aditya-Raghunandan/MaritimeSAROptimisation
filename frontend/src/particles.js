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
 *   3. Keep the last few positions of each particle and draw them as a
 *      polyline that fades along its length. That is the tail.
 *   4. Respawn each particle after a random lifetime, so the field does not
 *      slowly drain into its convergence zones and leave the rest bare.
 *
 * STEP 3 USED TO BE THE USUAL TRICK -- never clear the canvas, paint it with a
 * translucent black each frame and let old segments fade out. It leaves
 * PERMANENT GHOSTS, and the reason is arithmetic rather than taste. The fade
 * multiplies an 8-bit alpha channel: at 0.965 per frame the alpha falls 255 ->
 * 42 -> 14 and then sticks, because round(14 * 0.965) is 14. It can never
 * reach zero. Every streak the field has ever drawn stays on the canvas at
 * alpha 14 forever, which is exactly the faint debris left behind the real
 * tails.
 *
 * Holding the trail explicitly costs about 820 x 8 short segments a frame,
 * which canvas does not notice, and it makes the tail length an honest
 * parameter instead of something emergent from a decay constant.
 *
 * THE STEP IS SIZED IN PIXELS AND APPLIED IN DEGREES, and it needs both halves.
 *
 * Sized in pixels, because pixels are what the eye integrates. A step fixed in
 * seconds looks different at every zoom, and the first version of this got that
 * so wrong the layer was pointless: 15 s of drift per frame is 0.048 px at zoom
 * 5 for a 14 m/s wind, so nothing moved and the particles read as static white
 * noise. Motion is the whole payload -- streak length comes out proportional to
 * speed for free -- so the step is derived each frame from the map's current
 * metres-per-pixel.
 *
 * Applied in degrees, because converting m/s to degrees divides the eastward
 * component by cos(latitude), and across 17-36 N that is a 13 % difference
 * between the bottom of the box and the top. Skipping that correction would
 * make the same wind appear to blow faster at the top of the screen -- a
 * Mercator artefact a viewer would read as physics.
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
   *   maxSpeed   top of the speed scale, m/s -- sets the step with targetPx
   *   targetPx   pixels per frame the fastest wind should travel
   *   trail      how many past positions each streak keeps
   *   maxAgeMs   respawn after about this long
   */
  initialize(field, opts = {}) {
    this._field = field;
    this._frame = 0;
    this._count = opts.count ?? 820;
    this._maxSpeed = opts.maxSpeed ?? 20;
    // Pixels per frame that the FASTEST wind should travel. The step is
    // derived from this and the map's current scale, never fixed in seconds.
    this._targetPx = opts.targetPx ?? 1.5;
    // Positions kept per particle. The tail is drawn from these, so its
    // length is this number times the per-frame step -- explicit, and it can
    // actually reach zero, which a multiplied alpha cannot.
    this._trail = opts.trail ?? 9;
    this._maxAge = opts.maxAgeMs ?? 4200;
    this._alpha = opts.alpha ?? 0.75;
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
    const lat = g.lat(0) + Math.random() * (g.lat(g.nlat - 1) - g.lat(0));
    const lon = g.lon(0) + Math.random() * (g.lon(g.nlon - 1) - g.lon(0));
    return {
      lat,
      lon,
      // The tail starts empty, so a respawned particle does not draw a line
      // from wherever it died to wherever it reappeared.
      past: [],
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
    // Stored tail positions are SCREEN coordinates, so a move or a zoom makes
    // every one of them wrong. Dropping them costs a few frames of tail and
    // avoids drawing streaks between where a particle was on the old view and
    // where it is on the new one.
    for (const p of this._particles) p.past.length = 0;
    this._canvas.getContext('2d').clearRect(0, 0, size.x, size.y);
  },

  _step(dt) {
    if (!this._map || !this._canvas) return;
    const g = this._field.grid;
    const ctx = this._canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const size = this._map.getSize();

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // Cleared every frame. The tail comes from each particle's stored history,
    // not from an alpha decay that cannot reach zero -- see the module note.
    ctx.clearRect(0, 0, size.x, size.y);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    /*
      THE STEP IS DERIVED FROM THE MAP'S SCALE, NOT FIXED IN SECONDS, and this
      is what the first version got wrong badly enough to make the layer
      pointless. It advanced a fixed 15 s of drift per frame, which at zoom 5
      is 0.048 px for a 14 m/s wind -- the particles did not move at all, and a
      particle that does not move carries no information whatever. It read as
      static white noise over the field.

      Motion is the entire payload here: streak length ends up proportional to
      speed for free, so a fast jet draws long lines and a calm patch draws
      short ones. That only works if the step is measured in PIXELS, because
      pixels are what the eye integrates. Deriving it per frame also means the
      flow looks the same at every zoom instead of freezing as you zoom out.
    */
    const centre = this._map.getCenter();
    const mPerPx = 156543.03392
      * Math.cos((centre.lat * Math.PI) / 180)
      / 2 ** this._map.getZoom();
    const secondsPerSecond = (this._targetPx * mPerPx) / Math.max(this._maxSpeed, 0.1);
    const seconds = secondsPerSecond * dt * 60;   // dt is in seconds; 60 fps nominal

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
      if (Math.abs(to.x - from.x) + Math.abs(to.y - from.y) > 60) {
        p.past.length = 0;
        continue;
      }

      p.past.push([to.x, to.y]);
      if (p.past.length > this._trail) p.past.shift();

      // Fade ALONG the tail: oldest segment faintest, newest brightest. Each
      // segment is its own stroke because each has its own alpha.
      for (let n = 1; n < p.past.length; n += 1) {
        const t = n / (p.past.length - 1);
        ctx.strokeStyle = `rgba(255, 255, 255, ${(this._alpha * t * t).toFixed(3)})`;
        ctx.lineWidth = 0.6 + 0.8 * t;
        ctx.beginPath();
        ctx.moveTo(p.past[n - 1][0], p.past[n - 1][1]);
        ctx.lineTo(p.past[n][0], p.past[n][1]);
        ctx.stroke();
      }
    }
  },
});

export function particleLayer(field, opts) {
  return new ParticleLayer(field, opts);
}
