/**
 * sources.js -- where a field layer's numbers come from.
 *
 * A FieldLayer used to hold one Float32Array covering the whole window, which
 * is right for a 48 h bundle and impossible for five years: the hourly wind
 * tier alone is 1.32 GB. So the layer no longer owns its data. It owns a
 * SOURCE, and there are two:
 *
 *   BufferSource  one flat float32 blob, already in memory, always resident.
 *                 What `sar.viz.export` writes. Still the right thing for a
 *                 single scenario bundle and for local development.
 *
 *   ZarrSource    a published store on Hugging Face, fetched a chunk at a
 *                 time. What `sar.viz.archive` writes. One chunk is 48
 *                 timesteps over the whole box, 1.11 MB, and MEASURED against
 *                 the real store on 2026-09-17: zarrita fetches and decodes
 *                 one in 45 ms, and the values match Python to the last bit.
 *
 * THE ASYNC SEAM IS `ensure()`, AND IT IS DELIBERATELY NARROW.
 *
 * Reading a value stays synchronous -- `vector(frame, j, i)` -- because the
 * render loop, leaflet-velocity and the click-a-point chart all want a number
 * now, not a promise. What is asynchronous is making a frame RESIDENT, and
 * that happens once per chunk boundary rather than once per read. A caller
 * awaits `ensure(frame)` when the clock moves and then draws synchronously,
 * which is why the whole of main.js keeps its shape.
 *
 * Asking for a value that is not resident THROWS rather than returning zero or
 * stale data. A map that silently draws the wrong hour is the failure mode this
 * project keeps meeting, and it looks entirely reasonable on screen.
 */

import * as zarr from 'zarrita';

/** Frames already in memory: the whole window, always. */
export class BufferSource {
  /**
   * @param {Float32Array} buffer  [time][lat][lon][u, v], interleaved
   * @param {{nlat: number, nlon: number}} grid
   * @param {number} frames
   */
  constructor(buffer, grid, frames) {
    this.data = buffer;
    this.nlat = grid.nlat;
    this.nlon = grid.nlon;
    this.frameSize = grid.nlat * grid.nlon * 2;
    this.frames = frames ?? Math.floor(buffer.length / this.frameSize);
  }

  isResident() { return true; }

  async ensure() { /* nothing to fetch: it is all here */ }

  vector(frame, j, i) {
    const at = frame * this.frameSize + (j * this.nlon + i) * 2;
    return [this.data[at], this.data[at + 1]];
  }
}

/**
 * Frames fetched from a published Zarr store, a chunk at a time.
 *
 * The store holds u and v as two separate arrays, each chunked
 * [48, nlat, nlon]. A "chunk" here means both of them for the same time range,
 * because no caller ever wants one without the other.
 */
export class ZarrSource {
  /**
   * @param {string} base   URL of the .zarr store
   * @param {{frames, chunks, variables, grid}} tier  from <product>_archive.json
   * @param {number} cacheChunks  how many to keep. 4 x 1.11 MB x 2 vars = 8.9 MB.
   */
  constructor(base, tier, cacheChunks = 4) {
    this.base = base.replace(/\/$/, '');
    this.frames = tier.frames;
    this.chunkFrames = tier.chunks.time;
    this.nlat = tier.grid.nlat;
    this.nlon = tier.grid.nlon;
    this.variables = tier.variables;
    this.cacheChunks = cacheChunks;
    this._cache = new Map();   // chunk index -> { u, v }; insertion order is the LRU
    this._inflight = new Map();
    this._arrays = null;
  }

  async open() {
    const grp = zarr.root(new zarr.FetchStore(this.base));
    const [u, v] = await Promise.all(
      this.variables.map((name) => zarr.open(grp.resolve(name), { kind: 'array' })),
    );
    this._arrays = { u, v };

    // The manifest and the store must agree. If they do not, one of them was
    // regenerated without the other, and every index below is off.
    const [t, nlat, nlon] = u.shape;
    if (t !== this.frames || nlat !== this.nlat || nlon !== this.nlon) {
      throw new Error(
        `manifest says [${this.frames}, ${this.nlat}, ${this.nlon}] but the store at ` +
        `${this.base} is [${t}, ${nlat}, ${nlon}] -- they were not built together`,
      );
    }
    return this;
  }

  chunkOf(frame) { return Math.floor(frame / this.chunkFrames); }

  isResident(frame) { return this._cache.has(this.chunkOf(frame)); }

  /** Fetch and decode the chunk holding `frame`, unless it is already here. */
  async ensure(frame) {
    if (!this._arrays) throw new Error('ZarrSource.open() was never awaited');
    const c = this.chunkOf(frame);
    if (this._cache.has(c)) {
      // Touch it so the LRU keeps what is being scrubbed through.
      const hit = this._cache.get(c);
      this._cache.delete(c);
      this._cache.set(c, hit);
      return;
    }
    // Two clock moves inside one chunk must not become two fetches.
    if (this._inflight.has(c)) return this._inflight.get(c);

    const lo = c * this.chunkFrames;
    const hi = Math.min(lo + this.chunkFrames, this.frames);
    const job = Promise.all([
      zarr.get(this._arrays.u, [zarr.slice(lo, hi), null, null]),
      zarr.get(this._arrays.v, [zarr.slice(lo, hi), null, null]),
    ]).then(([u, v]) => {
      this._cache.set(c, { lo, frames: hi - lo, u: u.data, v: v.data });
      while (this._cache.size > this.cacheChunks) {
        this._cache.delete(this._cache.keys().next().value);
      }
      this._inflight.delete(c);
    }).catch((err) => {
      this._inflight.delete(c);
      throw err;
    });

    this._inflight.set(c, job);
    return job;
  }

  /**
   * [u, v] at a cell. Synchronous, and throws if the frame is not resident.
   *
   * Returning zeros instead would draw a calm, plausible, wrong map.
   */
  vector(frame, j, i) {
    const chunk = this._cache.get(this.chunkOf(frame));
    if (!chunk) {
      throw new Error(`frame ${frame} is not resident -- await ensure(${frame}) first`);
    }
    // u and v are separate arrays here, not interleaved as in the flat bundle.
    const at = (frame - chunk.lo) * this.nlat * this.nlon + j * this.nlon + i;
    return [chunk.u[at], chunk.v[at]];
  }

  /** Bytes currently held, for the status line. Chunks are ~1.11 MB each. */
  get residentBytes() {
    let n = 0;
    for (const c of this._cache.values()) n += (c.u.length + c.v.length) * 4;
    return n;
  }
}

/**
 * Pick the tier to draw at, from how much time is on screen.
 *
 * Nobody can perceive hourly detail while scrubbing across a year, so nobody
 * should download it: 1.32 GB hourly against 0.07 GB daily for the same five
 * years. The thresholds are in days of visible span, and are deliberately
 * generous -- fetching a slightly finer tier than strictly needed costs one
 * 1.11 MB chunk, while a coarse one that visibly steps is an artefact the
 * viewer will read as physics.
 *
 * Returns the finest tier whose cadence still gives a smooth scrub, falling
 * back to whatever the manifest actually has.
 */
export function pickTier(tiers, spanDays) {
  const order = ['hourly', '3-hourly', '6-hourly', 'daily'];
  const available = order.filter((name) => name in tiers);
  if (available.length === 0) throw new Error('the manifest publishes no tiers');

  let wanted;
  if (spanDays <= 14) wanted = 'hourly';
  else if (spanDays <= 120) wanted = '6-hourly';
  else wanted = 'daily';

  if (available.includes(wanted)) return wanted;

  // The wanted tier is not published. Prefer the next COARSER one, walking
  // away from `wanted`, because that is the direction that keeps the download
  // bounded -- substituting hourly for a requested daily across five years is
  // 1.32 GB instead of 0.07 GB. Only if nothing coarser exists do we go finer.
  //
  // This matters for currents, which publish 3-hourly and daily and no hourly
  // tier at all: a short span asks for `hourly`, and the right answer is
  // 3-hourly, not daily.
  const idx = order.indexOf(wanted);
  for (let k = idx + 1; k < order.length; k += 1) {
    if (available.includes(order[k])) return order[k];
  }
  for (let k = idx - 1; k >= 0; k -= 1) {
    if (available.includes(order[k])) return order[k];
  }
  return available[0];
}
