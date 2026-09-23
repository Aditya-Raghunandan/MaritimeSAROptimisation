/**
 * Tests for the two field sources.
 *
 * zarrita is mocked. What is under test is the chunking, caching and residency
 * logic we wrote -- not zarrita's decoder, which was verified separately
 * against the real store on 2026-09-17 (45 ms per chunk, values byte-identical
 * to Python).
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';

const fetched = [];
// The mocked store's shape. Tests that vary `frames` set this too, because
// ZarrSource deliberately refuses a store that disagrees with its manifest.
const shape = { frames: 96 };
// Flip on to make the mocked store fail, as a dropped connection would.
const failing = { on: false };

vi.mock('zarrita', () => {
  const slice = (start, stop) => ({ start, stop });
  return {
    FetchStore: class { constructor(url) { this.url = url; } },
    root: () => ({ resolve: (name) => ({ name }) }),
    open: async (loc) => ({ name: loc.name, shape: [shape.frames, 2, 3] }),
    slice,
    // One value per cell, encoding (frame, j, i) so a read can be checked
    // exactly: u = frame * 100 + j * 10 + i, and v = -u.
    get: async (arr, sel) => {
      if (failing.on) throw new Error('network down');
      const [{ start, stop }] = sel;
      fetched.push({ name: arr.name, start, stop });
      const n = (stop - start) * 2 * 3;
      const out = new Float32Array(n);
      let k = 0;
      for (let f = start; f < stop; f += 1) {
        for (let j = 0; j < 2; j += 1) {
          for (let i = 0; i < 3; i += 1) {
            const value = f * 100 + j * 10 + i;
            out[k] = arr.name === 'u10' ? value : -value;
            k += 1;
          }
        }
      }
      return { data: out, shape: [stop - start, 2, 3] };
    },
  };
});

const { BufferSource, ZarrSource, isFrameReady, pickTier, residentSpanOf } = await import('../src/sources.js');

const GRID = { nlat: 2, nlon: 3 };

function tier(overrides = {}) {
  return {
    frames: 96,
    chunks: { time: 48, lat: 2, lon: 3 },
    variables: ['u10', 'v10'],
    grid: GRID,
    ...overrides,
  };
}

describe('BufferSource', () => {
  // [time][lat][lon][u, v] interleaved, 2 frames of a 2 x 3 grid.
  const buf = new Float32Array(2 * 2 * 3 * 2);
  for (let f = 0; f < 2; f += 1) {
    for (let j = 0; j < 2; j += 1) {
      for (let i = 0; i < 3; i += 1) {
        const at = f * 12 + (j * 3 + i) * 2;
        buf[at] = f * 100 + j * 10 + i;
        buf[at + 1] = -(f * 100 + j * 10 + i);
      }
    }
  }

  it('reads an interleaved cell', () => {
    const s = new BufferSource(buf, GRID);
    expect(s.vector(1, 1, 2)).toEqual([112, -112]);
  });

  it('infers the frame count from the buffer length', () => {
    expect(new BufferSource(buf, GRID).frames).toBe(2);
  });

  it('is always resident and never fetches', async () => {
    const s = new BufferSource(buf, GRID);
    expect(s.isResident(0)).toBe(true);
    await expect(s.ensure(0)).resolves.toBeUndefined();
  });
});

describe('ZarrSource', () => {
  beforeEach(() => { fetched.length = 0; shape.frames = 96; failing.on = false; });

  it('maps a frame to its chunk', () => {
    const s = new ZarrSource('http://x/w.zarr', tier());
    expect(s.chunkOf(0)).toBe(0);
    expect(s.chunkOf(47)).toBe(0);
    expect(s.chunkOf(48)).toBe(1);
    expect(s.chunkOf(95)).toBe(1);
  });

  it('nothing is resident before ensure()', async () => {
    const s = await new ZarrSource('http://x/w.zarr', tier()).open();
    expect(s.isResident(0)).toBe(false);
  });

  it('reading a frame that is not resident throws rather than returning zeros', async () => {
    const s = await new ZarrSource('http://x/w.zarr', tier()).open();
    expect(() => s.vector(0, 0, 0)).toThrow(/not resident/);
  });

  it('ensure() fetches exactly the chunk holding the frame', async () => {
    const s = await new ZarrSource('http://x/w.zarr', tier()).open();
    await s.ensure(50);
    expect(fetched.map((f) => [f.name, f.start, f.stop])).toEqual([
      ['u10', 48, 96],
      ['v10', 48, 96],
    ]);
  });

  it('reads the right value once resident', async () => {
    const s = await new ZarrSource('http://x/w.zarr', tier()).open();
    await s.ensure(50);
    expect(s.vector(50, 1, 2)).toEqual([5012, -5012]);
  });

  it('indexes correctly at a chunk boundary', async () => {
    const s = await new ZarrSource('http://x/w.zarr', tier()).open();
    await s.ensure(48);
    expect(s.vector(48, 0, 0)).toEqual([4800, -4800]);
    await s.ensure(47);
    expect(s.vector(47, 0, 0)).toEqual([4700, -4700]);
  });

  it('a second frame in the same chunk costs no fetch', async () => {
    const s = await new ZarrSource('http://x/w.zarr', tier()).open();
    await s.ensure(0);
    await s.ensure(1);
    await s.ensure(47);
    expect(fetched).toHaveLength(2);      // u and v, once
  });

  it('two ensures racing on one chunk produce one fetch', async () => {
    const s = await new ZarrSource('http://x/w.zarr', tier()).open();
    await Promise.all([s.ensure(0), s.ensure(10), s.ensure(20)]);
    expect(fetched).toHaveLength(2);
  });

  it('clamps the last chunk to the array rather than over-reading', async () => {
    shape.frames = 60;
    const s = await new ZarrSource('http://x/w.zarr', tier({ frames: 60 })).open();
    await s.ensure(55);
    expect(fetched[0].stop).toBe(60);
  });

  it('evicts the least recently used chunk past the cache limit', async () => {
    shape.frames = 480;
    const s = await new ZarrSource('http://x/w.zarr',
      tier({ frames: 480, chunks: { time: 48, lat: 2, lon: 3 } }), 2).open();
    await s.ensure(0);        // chunk 0
    await s.ensure(48);       // chunk 1
    await s.ensure(96);       // chunk 2 -> evicts 0
    expect(s.isResident(0)).toBe(false);
    expect(s.isResident(48)).toBe(true);
    expect(s.isResident(96)).toBe(true);
  });

  it('a re-touched chunk is not the one evicted', async () => {
    shape.frames = 480;
    const s = await new ZarrSource('http://x/w.zarr',
      tier({ frames: 480 }), 2).open();
    await s.ensure(0);
    await s.ensure(48);
    await s.ensure(0);        // touch chunk 0, so chunk 1 is now the oldest
    await s.ensure(96);
    expect(s.isResident(0)).toBe(true);
    expect(s.isResident(48)).toBe(false);
  });

  it('reports how much it is holding', async () => {
    const s = await new ZarrSource('http://x/w.zarr', tier()).open();
    await s.ensure(0);
    expect(s.residentBytes).toBe(48 * 2 * 3 * 4 * 2);   // u and v
  });

  it('refuses a store whose shape disagrees with the manifest', async () => {
    // The store is mocked at [96, 2, 3]; claim something else.
    await expect(
      new ZarrSource('http://x/w.zarr', tier({ frames: 240 })).open(),
    ).rejects.toThrow(/not built together/);
  });

  it('a NaN frame is refused by name, not by zarrita', async () => {
    // Unguarded this becomes slice(NaN, NaN) and emerges from deep inside the
    // library as "Input contains an empty iterator", naming neither the frame
    // nor the store.
    const s = await new ZarrSource('http://x/w.zarr', tier()).open();
    await expect(s.ensure(NaN)).rejects.toThrow(RangeError);
    await expect(s.ensure(NaN)).rejects.toThrow(/frame NaN is not a valid index/);
  });

  it('refuses a frame past the end and a negative one', async () => {
    const s = await new ZarrSource('http://x/w.zarr', tier()).open();
    await expect(s.ensure(96)).rejects.toThrow(/0\.\.95/);
    await expect(s.ensure(-1)).rejects.toThrow(RangeError);
    await expect(s.ensure(1.5)).rejects.toThrow(RangeError);
  });

  it('using it before open() is a clear error, not undefined', async () => {
    const s = new ZarrSource('http://x/w.zarr', tier());
    await expect(s.ensure(0)).rejects.toThrow(/open\(\) was never awaited/);
  });
});

describe('prefetchNext', () => {
  beforeEach(() => { fetched.length = 0; shape.frames = 96; failing.on = false; });

  it('fetches the chunk AFTER the one holding the frame', async () => {
    const s = await new ZarrSource('http://x/w.zarr', tier()).open();
    s.prefetchNext(10);                 // frame 10 is in chunk 0, so chunk 1
    await Promise.all(s._inflight.values());
    expect(fetched.map((f) => [f.start, f.stop])).toEqual([[48, 96], [48, 96]]);
  });

  it('leaves the next frame resident, so playback does not stall on it', async () => {
    const s = await new ZarrSource('http://x/w.zarr', tier()).open();
    await s.ensure(40);
    s.prefetchNext(40);
    await Promise.all(s._inflight.values());
    const before = fetched.length;
    expect(s.isResident(48)).toBe(true);
    await s.ensure(48);
    expect(fetched).toHaveLength(before);   // no second fetch
  });

  it('does nothing at the last chunk', async () => {
    const s = await new ZarrSource('http://x/w.zarr', tier()).open();
    s.prefetchNext(60);                 // chunk 1 is the last of 96 frames
    expect(fetched).toHaveLength(0);
  });

  it('swallows a failed fetch rather than raising an unhandled rejection', async () => {
    const s = await new ZarrSource('http://x/w.zarr', tier()).open();
    failing.on = true;
    expect(() => s.prefetchNext(0)).not.toThrow();
    await new Promise((r) => setTimeout(r, 0));
    expect(s.isResident(48)).toBe(false);
    failing.on = false;
    await s.ensure(48);                 // and the normal path still works afterwards
    expect(s.isResident(48)).toBe(true);
  });

  it('is a no-op on a flat buffer, which holds everything already', () => {
    const b = new BufferSource(new Float32Array(2 * 3 * 2 * 4), GRID);
    expect(() => b.prefetchNext(0)).not.toThrow();
  });
});

describe('pickTier', () => {
  const all = { hourly: {}, '6-hourly': {}, daily: {} };

  it('uses hourly for a short span', () => {
    expect(pickTier(all, 2)).toBe('hourly');
    expect(pickTier(all, 14)).toBe('hourly');
  });

  it('steps down to 6-hourly for weeks and months', () => {
    expect(pickTier(all, 30)).toBe('6-hourly');
    expect(pickTier(all, 120)).toBe('6-hourly');
  });

  it('steps down to daily across years', () => {
    // 1.32 GB hourly against 0.07 GB daily for the same five years.
    expect(pickTier(all, 365)).toBe('daily');
    expect(pickTier(all, 1826)).toBe('daily');
  });

  it('falls back to a finer published tier when the wanted one is absent', () => {
    expect(pickTier({ hourly: {} }, 365)).toBe('hourly');
  });

  it('handles the current product, which publishes 3-hourly not hourly', () => {
    expect(pickTier({ '3-hourly': {}, daily: {} }, 2)).toBe('3-hourly');
    expect(pickTier({ '3-hourly': {}, daily: {} }, 400)).toBe('daily');
  });

  it('a manifest with no tiers is an error, not a silent blank map', () => {
    expect(() => pickTier({}, 10)).toThrow(/no tiers/);
  });
});

/*
  isFrameReady is the guard that stopped a live site painting nothing and
  unwiring every control below the renderers. Leaflet calls onAdd from
  addTo(map) synchronously, our renderers paint from there, and on the Zarr
  path frame 0 is not resident until ensure() resolves -- so vector() threw,
  the throw escaped start(), and the layer control, slider, ruler and rings
  were never wired.
*/
describe('isFrameReady', () => {
  it('delegates to a field that can answer for itself', () => {
    const field = { isResident: (f) => f === 7 };
    expect(isFrameReady(field, 7)).toBe(true);
    expect(isFrameReady(field, 8)).toBe(false);
  });

  it('asks the wrapped source when the field is a thin wrapper', () => {
    // main.js hands the quiver a { grid, vector, source } literal for the
    // resultant layer; the residency answer lives one level down.
    const field = { source: { isResident: (f) => f === 3 } };
    expect(isFrameReady(field, 3)).toBe(true);
    expect(isFrameReady(field, 4)).toBe(false);
  });

  it("prefers the field's own answer over the wrapped source's", () => {
    const field = { isResident: () => false, source: { isResident: () => true } };
    expect(isFrameReady(field, 0)).toBe(false);
  });

  it('treats a field that cannot answer as READY, not as not-ready', () => {
    // The flat-bundle path is always resident and must keep painting exactly
    // as it did. Defaulting to false here would blank that map instead.
    expect(isFrameReady({ grid: {}, vector: () => [1, 2] }, 0)).toBe(true);
    expect(isFrameReady({}, 0)).toBe(true);
  });

  it('does not throw on a missing field', () => {
    expect(isFrameReady(undefined, 0)).toBe(true);
    expect(isFrameReady(null, 0)).toBe(true);
  });

  it('a BufferSource is ready at once and a fresh ZarrSource is not', async () => {
    const buffer = new BufferSource(new Float32Array(2 * 3 * 2), { nlat: 2, nlon: 3 }, 1);
    expect(isFrameReady(buffer, 0)).toBe(true);

    const zarrSource = new ZarrSource('https://example.invalid/wind.zarr', {
      frames: shape.frames,
      chunks: { time: 48 },
      variables: ['u10', 'v10'],
      grid: { nlat: 2, nlon: 3 },
    });
    // This is the exact state the page was in when it threw: opened, shapes
    // agreed, nothing fetched.
    await zarrSource.open();
    expect(isFrameReady(zarrSource, 0)).toBe(false);
    await zarrSource.ensure(0);
    expect(isFrameReady(zarrSource, 0)).toBe(true);
  });
});

/*
  The click-a-point chart asks for this so it plots the hours it has rather
  than the whole tier. Plotting the tier drew a sliver of data against four
  empty years and reported a mean over an unnamed subset of frames.
*/
describe('residentSpan', () => {
  const tier = () => new ZarrSource('https://example.invalid/wind.zarr', {
    frames: shape.frames,                    // 96 frames, 48 per chunk -> 2 chunks
    chunks: { time: 48 },
    variables: ['u10', 'v10'],
    grid: { nlat: 2, nlon: 3 },
  });

  it('is empty before anything is fetched', async () => {
    const src = await tier().open();
    expect(src.residentSpan(0)).toEqual({ from: 0, to: 0 });
  });

  it('covers the chunk holding the frame once it lands', async () => {
    const src = await tier().open();
    await src.ensure(10);
    expect(src.residentSpan(10)).toEqual({ from: 0, to: 48 });
  });

  it('joins adjacent chunks into one run', async () => {
    const src = await tier().open();
    await src.ensure(10);
    await src.ensure(60);
    expect(src.residentSpan(10)).toEqual({ from: 0, to: 96 });
    expect(src.residentSpan(60)).toEqual({ from: 0, to: 96 });
  });

  it('stops at the end of the axis rather than past it', async () => {
    const src = await tier().open();
    await src.ensure(95);
    expect(src.residentSpan(95).to).toBe(96);
  });

  it('a BufferSource is the whole window', () => {
    const buf = new BufferSource(new Float32Array(2 * 3 * 2 * 4), { nlat: 2, nlon: 3 }, 4);
    expect(buf.residentSpan()).toEqual({ from: 0, to: 4 });
  });
});

describe('residentSpanOf', () => {
  it('unwraps a field that holds a source', async () => {
    const src = await new ZarrSource('https://example.invalid/wind.zarr', {
      frames: shape.frames,
      chunks: { time: 48 },
      variables: ['u10', 'v10'],
      grid: { nlat: 2, nlon: 3 },
    }).open();
    await src.ensure(0);
    expect(residentSpanOf({ source: src }, 0, 96)).toEqual({ from: 0, to: 48 });
  });

  it('falls back to the whole axis for a field that cannot answer', () => {
    expect(residentSpanOf({ vector: () => [0, 0] }, 5, 40)).toEqual({ from: 0, to: 40 });
    expect(residentSpanOf(null, 5, 40)).toEqual({ from: 0, to: 40 });
  });

  it('falls back to the whole axis rather than returning an empty range', async () => {
    // An empty span would make the chart plot nothing at all, which looks
    // like a broken panel rather than like data still arriving.
    const src = await new ZarrSource('https://example.invalid/wind.zarr', {
      frames: shape.frames,
      chunks: { time: 48 },
      variables: ['u10', 'v10'],
      grid: { nlat: 2, nlon: 3 },
    }).open();
    expect(residentSpanOf(src, 0, 96)).toEqual({ from: 0, to: 96 });
  });
});
