/**
 * Tests for the resultant drift field.
 *
 * The arithmetic is trivial; what matters is that the thing refuses to claim
 * more than it has. D002 has three terms, a field rendering can carry at most
 * two, and the one it can never carry is the one that makes the model
 * stochastic.
 */

import { describe, expect, it } from 'vitest'

import {
  ALPHA, BILINEAR_FIELD, ResultantSource, resultantScale
} from '../src/resultant.js'
import { Grid } from '../src/layers.js'

const WIND_GRID = new Grid({ lat0: 17, dlat: 0.25, nlat: 77, lon0: -82, dlon: 0.25, nlon: 77 })
const CUR_GRID = new Grid({ lat0: 17, dlat: 0.04, nlat: 476, lon0: -82, dlon: 0.08, nlon: 238 })

const axis = (stepSeconds, frames) => ({
  start: new Date('2021-01-01T00:00:00Z'), stepSeconds, frames
})

function src (vector, frames = 48) {
  return { frames, vector, isResident: () => true, ensure: async () => {} }
}

const wind = (u, v) => ({
  source: src(() => [u, v]), grid: WIND_GRID, axis: axis(3600, 48)
})
const current = (u, v) => ({
  source: src(() => [u, v], 16), grid: CUR_GRID, axis: axis(10800, 16)
})

describe('the leeway-only case, which is where we are today', () => {
  const r = new ResultantSource(wind(10, 0), null)

  it('is 2 % of the wind', () => {
    expect(r.vector(0, 0, 0)).toEqual([0.2, 0])
    expect(ALPHA).toBe(0.02)
  })

  it('knows it is partial', () => {
    expect(r.isPartial).toBe(true)
  })

  it('says so in its own label, not only in a footnote', () => {
    expect(r.describe().label).toMatch(/leeway only/i)
  })

  it('names the current as missing, and says it is usually the larger term', () => {
    const d = r.describe()
    expect(d.missing).toContain('surface current')
    expect(d.caveat).toMatch(/larger term/)
    expect(d.caveat).toMatch(/not yet where a drifter would go/)
  })
})

describe('with a current, which is where we are going', () => {
  const r = new ResultantSource(wind(10, 0), current(1.5, 0.5))

  it('adds the current to the leeway', () => {
    expect(r.vector(0, 0, 0)).toEqual([1.7, 0.5])   // 0.2 + 1.5, 0 + 0.5
  })

  it('is no longer partial', () => {
    expect(r.isPartial).toBe(false)
    expect(r.describe().label).toBe('Resultant drift')
  })

  it('STILL names eta as missing, because a field can never carry it', () => {
    // This is the point of the whole file. eta is a per-particle draw with no
    // value at a location; a resultant arrow is not where a drifter goes.
    const d = r.describe()
    expect(d.missing.join(' ')).toMatch(/η/)
    expect(d.caveat).toMatch(/not a\s+field/)
    expect(d.caveat).toMatch(/will not\s+follow the same path/)
  })

  it('a land cell in the current draws no arrow at all', () => {
    // HYCOM is NaN over land. This used to keep the leeway term on its own, which
    // drew drift arrows over Florida and Cuba saying a person there would move a
    // couple of hundred metres an hour. Nobody drifts on land. Reversed 23 Sep.
    const withLand = new ResultantSource(wind(10, 0), current(NaN, NaN))
    const [u, v] = withLand.vector(0, 0, 0)
    expect(Number.isNaN(u)).toBe(true)
    expect(Number.isNaN(v)).toBe(true)
  })

  it('without a current at all it is still leeway everywhere, and says so', () => {
    // Land is only knowable from the current. With no current published there is
    // no land mask, so the leeway-only layer keeps drawing and keeps its label.
    const r = new ResultantSource(wind(10, 0), null)
    expect(r.vector(0, 0, 0)).toEqual([0.2, 0])
    expect(r.describe().label).toMatch(/leeway only/i)
  })
})

describe('the two grids', () => {
  it('output is on the wind grid, the coarser of the two', () => {
    const r = new ResultantSource(wind(10, 0), current(1, 0))
    expect(r.grid.nlat).toBe(77)
    expect(r.grid.dlat).toBe(0.25)
  })

  it('maps a wind cell onto the right current cell', () => {
    // Wind cell (4, 8) is 18.00 N, -80.00 E. On the current grid that is
    // j = (18-17)/0.04 = 25, i = (-80+82)/0.08 = 25.
    const seen = []
    const r = new ResultantSource(
      wind(10, 0),
      {
 source: {
 frames: 16,
isResident: () => true,
ensure: async () => {},
        vector: (f, j, i) => { seen.push([j, i]); return [1, 0] } 
},
      grid: CUR_GRID,
axis: axis(10800, 16) 
}
    );
    r.vector(0, 4, 8)
    expect(seen).toEqual([[25, 25]])
  })
})

describe('frame alignment between an hourly and a 3-hourly field', () => {
  const r = new ResultantSource(wind(10, 0), current(1, 0))

  it('snaps to the NEAREST current frame, not the most recent', () => {
    // 01:30 is hour 1.5; the 3-hourly field should go forward to 03:00 rather
    // than hold 00:00, or the current visibly lags the wind on the same map.
    expect(r._currentFrame(0)).toBe(0)
    expect(r._currentFrame(1)).toBe(0)     // 01:00 -> 00:00
    expect(r._currentFrame(2)).toBe(1)     // 02:00 -> 03:00
    expect(r._currentFrame(3)).toBe(1)     // 03:00 -> 03:00
  })

  it('clamps rather than running off the end of the shorter field', () => {
    expect(r._currentFrame(47)).toBe(15)
    expect(r._currentFrame(1e6)).toBe(15)
  })
})

describe('resultantScale', () => {
  it('is not the wind scale, or every arrow would be invisible', () => {
    // Leeway is 2 % of wind, so on a 20 m/s scale the resultant maxes at 0.4.
    expect(resultantScale(20, false)).toBeCloseTo(0.4)
    expect(resultantScale(20, false)).toBeLessThan(20)
  })

  it('is set by the current once there is one', () => {
    expect(resultantScale(20, true, 1.8)).toBeCloseTo(2.52)
  })

  it('never returns zero, which would divide by nothing', () => {
    expect(resultantScale(0, false)).toBeGreaterThan(0)
  })
})

/**
 * The bilinear flag, and the point query behind it.
 *
 * `BILINEAR_FIELD` is off: the drawn field keeps nearest neighbour until the cost has
 * been measured on a real frame. `sampleAt` ignores the flag and is always bilinear,
 * because a number shown beside a clicked position is where half a cell matters.
 */
describe('the bilinear flag', () => {
  // A current that varies cell by cell, so nearest neighbour and bilinear cannot agree
  // by accident. Constant fields, which the helpers above use, hide the difference.
  const varying = (grid) => ({
    grid,
    axis: axis(10800, 16),
    source: {
      frames: 16,
      isResident: () => true,
      ensure: async () => {},
      vector: (f, j, i) => [0.1 * j + 0.2 * i, 0.05 * j - 0.1 * i]
    },
  })

  // A position deliberately between grid lines, where rounding is at its worst.
  const LAT = CUR_GRID.lat0 + 10.5 * CUR_GRID.dlat
  const LON = CUR_GRID.lon0 + 10.5 * CUR_GRID.dlon

  it('is off by default, so nothing drawn changes today', () => {
    expect(BILINEAR_FIELD).toBe(false)
    expect(new ResultantSource(wind(10, 0), current(1, 0)).bilinear).toBe(false)
  })

  it('is switched on per instance, not globally', () => {
    const r = new ResultantSource(wind(10, 0), current(1, 0), { bilinear: true })
    expect(r.bilinear).toBe(true)
    expect(new ResultantSource(wind(10, 0), current(1, 0)).bilinear).toBe(false)
  })

  it('changes the drawn vector once it is on', () => {
    const off = new ResultantSource(wind(0, 0), varying(CUR_GRID))
    const on = new ResultantSource(wind(0, 0), varying(CUR_GRID), { bilinear: true })
    const j = WIND_GRID.cellAt(LAT, LON).j
    const i = WIND_GRID.cellAt(LAT, LON).i
    expect(on.vector(0, j, i)).not.toEqual(off.vector(0, j, i))
  })

  it('never throws on the draw path, even at a land cell', () => {
    const land = varying(CUR_GRID)
    land.source.vector = () => [NaN, NaN]
    const r = new ResultantSource(wind(10, 0), land, { bilinear: true })
    // One land cell must not take the frame down: it answers NaN, which the
    // renderers skip, rather than throwing out of the draw loop.
    expect(() => r.vector(0, 0, 0)).not.toThrow()
    expect(Number.isNaN(r.vector(0, 0, 0)[0])).toBe(true)
  })
})

describe('sampleAt, the point query', () => {
  const varying = {
    grid: CUR_GRID,
    axis: axis(10800, 16),
    source: {
      frames: 16,
      isResident: () => true,
      ensure: async () => {},
      vector: (f, j, i) => [0.1 * j + 0.2 * i, 0.05 * j - 0.1 * i]
    },
  }
  const LAT = CUR_GRID.lat0 + 10.5 * CUR_GRID.dlat
  const LON = CUR_GRID.lon0 + 10.5 * CUR_GRID.dlon

  it('is bilinear even when the field flag is off', () => {
    const off = new ResultantSource(wind(0, 0), varying)
    const cell = CUR_GRID.cellAt(LAT, LON)
    const nearest = varying.source.vector(0, cell.j, cell.i)
    const got = off.sampleAt(0, LAT, LON)
    expect(got.current[0]).not.toBeCloseTo(nearest[0], 6)
  })

  it('reports the current and the leeway separately, and their sum', () => {
    const r = new ResultantSource(wind(10, 0), varying)
    const got = r.sampleAt(0, LAT, LON)
    expect(got.leeway[0]).toBeCloseTo(0.2, 12)
    expect(got.u).toBeCloseTo(got.leeway[0] + got.current[0], 12)
    expect(got.v).toBeCloseTo(got.leeway[1] + got.current[1], 12)
  })

  it('carries the uncertainty the engine computes', () => {
    const got = new ResultantSource(wind(10, 0), varying).sampleAt(0, LAT, LON)
    expect(got.uncertainty.nCorners).toBe(4)
    expect(got.uncertainty.sigmaSpatialMs).toBeGreaterThan(0)
  })

  it('still answers with leeway alone when there is no current at all', () => {
    const got = new ResultantSource(wind(10, 0), null).sampleAt(0, LAT, LON)
    expect(got.current).toBeNull()
    expect(got.isPartial).toBe(true)
    expect(got.currentReason).toMatch(/not published/)
    expect(got.u).toBeCloseTo(0.2, 12)
  })

  // Positions 0.4 of a cell past a grid line, so the nearest cell is unambiguous:
  // j = i = 10, with the four corners at 10 and 11 on each axis.
  const NEAR_LAT = CUR_GRID.lat0 + 10.4 * CUR_GRID.dlat
  const NEAR_LON = CUR_GRID.lon0 + 10.4 * CUR_GRID.dlon
  const landWhere = (isLand) => ({
    ...varying,
    source: {
      ...varying.source,
      vector: (f, j, i) => (isLand(j, i) ? [NaN, NaN] : varying.source.vector(f, j, i)),
    },
  })

  it('says in words why a coastal point has no current, rather than throwing', () => {
    // The nearest cell is water; one corner beside it is land. The point is at sea,
    // so the leeway term still stands and the reason names the coast.
    const coast = landWhere((j, i) => i === 11)
    const got = new ResultantSource(wind(10, 0), coast).sampleAt(0, NEAR_LAT, NEAR_LON)
    expect(got.onLand).toBe(false)
    expect(got.current).toBeNull()
    expect(got.currentReason).toMatch(/within one cell of land/)
    expect(got.u).toBeCloseTo(0.2, 12)
  })

  it('says a point ON land is land, and gives no drift there', () => {
    // The nearest cell itself is land. A leeway-only answer here told people a
    // person in North Carolina would drift 208 m an hour. Found 23 Sep.
    const inland = landWhere((j, i) => i === 10)
    const got = new ResultantSource(wind(10, 0), inland).sampleAt(0, NEAR_LAT, NEAR_LON)
    expect(got.onLand).toBe(true)
    expect(got.currentReason).toMatch(/on land/)
    expect(Number.isNaN(got.u)).toBe(true)
    expect(Number.isNaN(got.v)).toBe(true)
    expect(got.leeway[0]).toBeCloseTo(0.2, 12)   // the wind is still real, and reported
  })

  it('a point far inland, with every cell land, is land too', () => {
    const allLand = landWhere(() => true)
    const got = new ResultantSource(wind(10, 0), allLand).sampleAt(0, NEAR_LAT, NEAR_LON)
    expect(got.onLand).toBe(true)
  })

  it('open water is not land', () => {
    const got = new ResultantSource(wind(10, 0), varying).sampleAt(0, NEAR_LAT, NEAR_LON)
    expect(got.onLand).toBe(false)
    expect(Number.isFinite(got.u)).toBe(true)
  })

  it('says so when the point is off the current grid', () => {
    const got = new ResultantSource(wind(10, 0), varying).sampleAt(0, 16.0, -81.0)
    expect(got.current).toBeNull()
    expect(got.currentReason).toMatch(/outside/)
  })

  it('keeps the caveat wording, so a caller cannot show a number without it', () => {
    const got = new ResultantSource(wind(10, 0), varying).sampleAt(0, LAT, LON)
    expect(got.missing).toContain('stochastic η')
    expect(got.caveat).toBeTruthy()
  })
})
