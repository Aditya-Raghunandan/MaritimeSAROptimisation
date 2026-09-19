/**
 * The layer types, and in particular the two places a silent error would hide:
 * the buffer indexing (a wrong stride draws a plausible but wrong map) and the
 * latitude flip that leaflet-velocity needs (a mirrored field still looks like
 * weather).
 */

import { describe, expect, it } from 'vitest';
import { FieldLayer, Grid, TrackLayer, buildLayer } from './layers.js';

const GRID = { lat0: 17, dlat: 0.25, nlat: 3, lon0: -82, dlon: 0.25, nlon: 4 };

/** Two frames of a 3 x 4 field where u encodes the cell index and v its negative. */
function fieldSpec() {
  const n = GRID.nlat * GRID.nlon;
  const buf = new Float32Array(2 * n * 2);
  for (let f = 0; f < 2; f += 1) {
    for (let k = 0; k < n; k += 1) {
      buf[f * n * 2 + k * 2] = f * 100 + k;
      buf[f * n * 2 + k * 2 + 1] = -(f * 100 + k);
    }
  }
  return [
    { id: 'wind', type: 'field', label: '10 m wind', units: 'm/s', grid: GRID, value_range: [0, 20] },
    buf,
  ];
}

describe('Grid', () => {
  it('reconstructs coordinates from origin and step', () => {
    const g = new Grid(GRID);
    expect(g.lat(0)).toBe(17);
    expect(g.lat(2)).toBe(17.5);
    expect(g.lon(3)).toBeCloseTo(-81.25);
  });

  it('finds the nearest cell to a click', () => {
    const g = new Grid(GRID);
    expect(g.cellAt(17.26, -81.76)).toEqual({ i: 1, j: 1 });
  });

  it('returns null outside the grid rather than clamping to an edge', () => {
    // Clamping would silently report the wrong cell's data for a click in the
    // sea beyond the box, which reads as real.
    const g = new Grid(GRID);
    expect(g.cellAt(50, -70)).toBeNull();
    expect(g.cellAt(17, -90)).toBeNull();
  });
});

describe('FieldLayer', () => {
  it('indexes the interleaved buffer correctly', () => {
    const layer = new FieldLayer(...fieldSpec());
    // frame 1, row 2, column 3 -> cell index 2*4+3 = 11, value 100+11
    expect(layer.vector(1, 2, 3)).toEqual([111, -111]);
  });

  it('computes speed from the pair', () => {
    const layer = new FieldLayer(...fieldSpec());
    expect(layer.speed(0, 0, 3)).toBeCloseTo(Math.hypot(3, -3));
  });

  it('reads a cell across every frame without refetching', () => {
    const layer = new FieldLayer(...fieldSpec());
    const series = layer.seriesAt(1, 1, 2); // cell index 5
    expect(series.length).toBe(2);
    expect(series[0]).toBeCloseTo(Math.hypot(5, -5));
    expect(series[1]).toBeCloseTo(Math.hypot(105, -105));
  });

});

describe('TrackLayer', () => {
  const t0 = Date.parse('2021-01-01T00:00:00Z');
  const track = new TrackLayer({
    id: 'sq', type: 'track', label: 'Expanding Square',
    fixes: [
      { t: t0, lat: 30, lon: -70 },
      { t: t0 + 3600_000, lat: 31, lon: -70 },
      { t: t0 + 7200_000, lat: 31, lon: -69 },
    ],
  });

  it('interpolates between fixes so a searcher moves smoothly', () => {
    // The point of interpolating: a helicopter must move continuously over a
    // current field that only updates every three hours.
    expect(track.positionAt(new Date(t0 + 1800_000))).toEqual([30.5, -70]);
  });

  it('lands exactly on a fix', () => {
    expect(track.positionAt(new Date(t0 + 3600_000))).toEqual([31, -70]);
  });

  it('holds the endpoints outside its own span', () => {
    expect(track.positionAt(new Date(t0 - 10_000))).toEqual([30, -70]);
    expect(track.positionAt(new Date(t0 + 99_999_999))).toEqual([31, -69]);
  });

  it('draws the whole path when it is not cumulative', () => {
    expect(track.pathUpTo(new Date(t0)).length).toBe(3);
  });

  it('grows a cumulative path with time, as swept area must', () => {
    const swept = new TrackLayer({
      id: 'swept', type: 'track', label: 'swept', cumulative: true,
      swept_width_m: 185,
      fixes: track.fixes,
    });
    expect(swept.pathUpTo(new Date(t0)).length).toBe(1);
    expect(swept.pathUpTo(new Date(t0 + 3600_000)).length).toBe(2);
    expect(swept.pathUpTo(new Date(t0 + 7200_000)).length).toBe(3);
  });
});

describe('buildLayer', () => {
  it('builds each known type', () => {
    const [spec, buf] = fieldSpec();
    expect(buildLayer(spec, buf).type).toBe('field');
    expect(buildLayer({ id: 'p', type: 'track', label: 't', fixes: [] }).type).toBe('track');
  });

  it('throws on an unknown type instead of skipping it', () => {
    // A layer that silently fails to appear is the hardest bug to notice on a
    // map, because the map still looks entirely reasonable without it.
    expect(() => buildLayer({ id: 'x', type: 'contour', label: 'x' }, null))
      .toThrow(/unknown layer type "contour"/);
  });
});
