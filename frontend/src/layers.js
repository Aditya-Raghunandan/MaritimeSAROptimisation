/**
 * layers.js -- the four layer types, and the registry that draws them.
 *
 * NOT "a wind layer". Four TYPES, of which wind and current are merely the
 * first two instances of one of them:
 *
 *   field    u/v vectors over a regular grid   wind, current
 *   raster   scalar heatmap on its own grid    probability map (D007), exceedance
 *   points   animated positions                drift particles (D009, ~500 saved)
 *   track    a path with a marker at t         Expanding Square, the RL agent, drifters
 *
 * Only `field` has real data today. The other three are here, with their
 * interfaces settled and their draw calls stubbed, because retrofitting them
 * would be a rewrite: they differ in what they need from the clock, not just in
 * how they look. A `track` needs a position interpolated between fixes; a
 * `points` layer needs a whole cloud at once; a cumulative layer -- the swept
 * area behind a searcher -- needs everything from the start of the run up to
 * now, not one frame.
 *
 * NO RENDERER MAY ASSUME THE GRID. Each layer carries its own from the
 * manifest. The probability map will arrive at 200-500 m inside a box that
 * follows the drifting ensemble, which is nothing like the 0.25 deg wind grid,
 * and the search tracks carry no grid at all.
 */

/** A regular lat/lon grid described by origin and step, as the manifest gives it. */
export class Grid {
  constructor(spec) {
    Object.assign(this, spec);
  }

  lat(j) { return this.lat0 + j * this.dlat; }
  lon(i) { return this.lon0 + i * this.dlon; }

  /** Nearest cell to a click, or null if the point is outside the grid. */
  cellAt(lat, lon) {
    const j = Math.round((lat - this.lat0) / this.dlat);
    const i = Math.round((lon - this.lon0) / this.dlon);
    if (j < 0 || j >= this.nlat || i < 0 || i >= this.nlon) return null;
    return { i, j };
  }

  get bounds() {
    return [
      [this.lat0, this.lon0],
      [this.lat(this.nlat - 1), this.lon(this.nlon - 1)],
    ];
  }
}

/**
 * A u/v field over a regular grid, one frame per timestep.
 *
 * The buffer is [time][lat][lon][u, v] float32, interleaved so a cell's two
 * components are adjacent -- one read per vector rather than two, a whole plane
 * apart.
 */
export class FieldLayer {
  constructor(spec, buffer) {
    this.id = spec.id;
    this.type = 'field';
    this.label = spec.label;
    this.units = spec.units;
    this.grid = new Grid(spec.grid);
    this.valueRange = spec.value_range;
    this.data = buffer;
    this.frameSize = this.grid.nlat * this.grid.nlon * 2;
  }

  /** [u, v] at a cell in a given frame. */
  vector(frame, j, i) {
    const at = frame * this.frameSize + (j * this.grid.nlon + i) * 2;
    return [this.data[at], this.data[at + 1]];
  }

  speed(frame, j, i) {
    const [u, v] = this.vector(frame, j, i);
    return Math.hypot(u, v);
  }

  /**
   * Every value at one cell across the whole window.
   *
   * This is why clicking a point costs nothing. The buffer already holds each
   * cell's time series -- it is the same array read along a different axis --
   * so a click is arithmetic, not a fetch, and needs no server. The limit is
   * honest and worth stating: it covers the LOADED WINDOW only. Five years at
   * this resolution is 2.03 GB and is not going into a browser; that view comes
   * from the box-mean series instead.
   */
  seriesAt(j, i, frames) {
    const out = new Float32Array(frames);
    for (let f = 0; f < frames; f += 1) out[f] = this.speed(f, j, i);
    return out;
  }

  /** Reshape one frame into leaflet-velocity's expected header/data pair. */
  velocityFrame(frame) {
    const { nlat, nlon, lat0, lon0, dlat, dlon } = this.grid;
    const n = nlat * nlon;
    const u = new Array(n);
    const v = new Array(n);

    // leaflet-velocity reads rows from NORTH down, while our grid is stored
    // ascending in latitude (D020). Flipping here rather than at export keeps
    // the stored convention intact and confines the plugin's assumption to the
    // one function that talks to it.
    for (let j = 0; j < nlat; j += 1) {
      const src = nlat - 1 - j;
      for (let i = 0; i < nlon; i += 1) {
        const [uu, vv] = this.vector(frame, src, i);
        u[j * nlon + i] = uu;
        v[j * nlon + i] = vv;
      }
    }

    const header = {
      parameterUnit: 'm.s-1',
      nx: nlon,
      ny: nlat,
      lo1: lon0,
      la1: lat0 + dlat * (nlat - 1),
      lo2: lon0 + dlon * (nlon - 1),
      la2: lat0,
      dx: dlon,
      dy: dlat,
    };
    return [
      { header: { ...header, parameterCategory: 2, parameterNumber: 2 }, data: u },
      { header: { ...header, parameterCategory: 2, parameterNumber: 3 }, data: v },
    ];
  }
}

/**
 * A scalar field on its own grid, drawn as a heatmap.
 *
 * For the probability map (D007) and for exceedance maps. Deliberately separate
 * from FieldLayer: it has one component rather than two, its own much finer
 * grid, and a grid ORIGIN THAT MOVES, because the map follows the drifting
 * ensemble rather than sitting over a fixed frame. That last point is why the
 * grid is read per frame here and once in FieldLayer.
 */
export class RasterLayer {
  constructor(spec, buffer) {
    this.id = spec.id;
    this.type = 'raster';
    this.label = spec.label;
    this.grids = spec.grids ? spec.grids.map((g) => new Grid(g)) : [new Grid(spec.grid)];
    this.data = buffer;
    this.valueRange = spec.value_range;
  }

  gridAt(frame) {
    return this.grids[Math.min(frame, this.grids.length - 1)];
  }
}

/**
 * A cloud of positions per frame -- the drift ensemble.
 *
 * D009 saves full trajectories for ~500 particles, chosen as the first 500
 * because they are i.i.d. and that makes the subsample unbiased. The remaining
 * N are summarised by the probability map instead, which is why this layer is
 * small and the RasterLayer is the one carrying the full ensemble's weight.
 */
export class PointsLayer {
  constructor(spec, buffer) {
    this.id = spec.id;
    this.type = 'points';
    this.label = spec.label;
    this.count = spec.count;
    this.data = buffer; // [time][particle][lat, lon]
  }

  positions(frame) {
    const out = [];
    const base = frame * this.count * 2;
    for (let p = 0; p < this.count; p += 1) {
      out.push([this.data[base + p * 2], this.data[base + p * 2 + 1]]);
    }
    return out;
  }
}

/**
 * A path with a marker at the current time -- drifters now, searchers later.
 *
 * `cumulative` is what the swept area needs: the corridor behind a helicopter
 * accumulates rather than being replaced each frame, because it is what the RL
 * reward is computed from. Showing it is showing the mechanism of the result.
 */
export class TrackLayer {
  constructor(spec) {
    this.id = spec.id;
    this.type = 'track';
    this.label = spec.label;
    this.cumulative = Boolean(spec.cumulative);
    this.sweptWidthM = spec.swept_width_m ?? null;
    this.fixes = spec.fixes ?? []; // [{ t: epoch ms, lat, lon }]
  }

  /**
   * Position at an arbitrary moment, interpolated between fixes.
   *
   * Interpolated rather than snapped so a helicopter moves smoothly over a
   * current field that only updates every three hours. Linear is right here:
   * these are straight legs between waypoints, and the drift engine's own
   * interpolation (docs/linear_time_interpolation.md) is linear for the same
   * reason.
   */
  positionAt(when) {
    const t = when.getTime();
    const f = this.fixes;
    if (f.length === 0) return null;
    if (t <= f[0].t) return [f[0].lat, f[0].lon];
    if (t >= f[f.length - 1].t) return [f[f.length - 1].lat, f[f.length - 1].lon];

    let k = 0;
    while (k < f.length - 2 && f[k + 1].t < t) k += 1;
    const span = f[k + 1].t - f[k].t;
    const w = span === 0 ? 0 : (t - f[k].t) / span;
    return [
      f[k].lat + w * (f[k + 1].lat - f[k].lat),
      f[k].lon + w * (f[k + 1].lon - f[k].lon),
    ];
  }

  /** The path up to `when` for a cumulative layer, or the whole path otherwise. */
  pathUpTo(when) {
    if (!this.cumulative) return this.fixes.map((p) => [p.lat, p.lon]);
    const t = when.getTime();
    return this.fixes.filter((p) => p.t <= t).map((p) => [p.lat, p.lon]);
  }
}

const BUILDERS = {
  field: (spec, buf) => new FieldLayer(spec, buf),
  raster: (spec, buf) => new RasterLayer(spec, buf),
  points: (spec, buf) => new PointsLayer(spec, buf),
  track: (spec) => new TrackLayer(spec),
};

/**
 * Build a layer from its manifest entry.
 *
 * An unknown type throws rather than being skipped. A layer that silently does
 * not appear is the hardest kind of bug to notice on a map, because the map
 * still looks perfectly reasonable without it.
 */
export function buildLayer(spec, buffer) {
  const make = BUILDERS[spec.type];
  if (!make) {
    throw new Error(
      `unknown layer type "${spec.type}" for layer "${spec.id}"; ` +
      `expected one of ${Object.keys(BUILDERS).join(', ')}`,
    );
  }
  return make(spec, buffer);
}
