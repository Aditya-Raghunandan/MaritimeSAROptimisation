/**
 * Tests for the probability map's arithmetic: the grid, the display blur, the containment
 * colouring and the contours (probabilityMap.js, drawn by probabilityLayer.js).
 *
 * The map is smoothed for the eye only, so the gates are that the smoothing moves no
 * probability, that colour order is density order, and that a contour labelled "half"
 * encloses half.
 */

import { describe, expect, it } from 'vitest';

import { ICE, INFERNO, ramp } from '../src/colormap.js';
import {
  CELL_M, PALETTES, alphaFor, blur, blurRadius, containmentOf, containmentTable, contourSegments,
  densityGrid, displayCellM, fineSplit, joinSegments, levelsFor, paletteTable, shownGrid, upsample,
} from '../src/probabilityMap.js';
import { offsetPosition, relativeM } from '../src/referee.js';

const sum = (a) => a.reduce((s, x) => s + x, 0);

/**
 * A seeded Gaussian cloud of n particles about (lat, lon), sd metres; `ratio` makes it narrower
 * north-south, and `turnDeg` then turns it anticlockwise from east.
 */
function cloud(n, sd, lat = 26, lon = 285, seed = 7, { ratio = 1, turnDeg = 0 } = {}) {
  let s = seed;
  const rand = () => {
    s = (s * 16807) % 2147483647;
    return s / 2147483647;
  };
  const gauss = () => Math.sqrt(-2 * Math.log(rand())) * Math.cos(2 * Math.PI * rand());
  const la = new Float64Array(n);
  const lo = new Float64Array(n);
  const [cos, sin] = [Math.cos((turnDeg * Math.PI) / 180), Math.sin((turnDeg * Math.PI) / 180)];
  for (let j = 0; j < n; j += 1) {
    const a = sd * gauss();
    const b = sd * ratio * gauss();
    [la[j], lo[j]] = offsetPosition(lat, lon, a * cos - b * sin, a * sin + b * cos);
  }
  return { lat: la, lon: lo, weight: new Float64Array(n).fill(1 / n) };
}

const lin = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
function lightness([r, g, b]) {
  const R = lin(r / 255);
  const G = lin(g / 255);
  const B = lin(b / 255);
  const l = Math.cbrt(0.4122214708 * R + 0.5363325363 * G + 0.0514459929 * B);
  const m = Math.cbrt(0.2119034982 * R + 0.6806995451 * G + 0.1073969566 * B);
  const s = Math.cbrt(0.0883024619 * R + 0.2817188376 * G + 0.6299787005 * B);
  return 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s;
}

describe('the grid', () => {
  it('holds every particle\'s weight, in its 250 m cell about the marker', () => {
    const c = cloud(2000, 1500);
    const g = densityGrid(c.lat, c.lon, c.weight, 26, 285, { pad: 4 });
    expect(sum(g.mass)).toBeCloseTo(1, 12);
    // A particle 600 m east and 300 m north of the marker is in cell [2, 1].
    const [la, lo] = offsetPosition(26, 285, 600, 300);
    const one = densityGrid([la], [lo], [1], 26, 285);
    expect([one.x0, one.y0, one.nx, one.ny]).toEqual([2, 1, 1, 1]);
  });

  it('leaves out drained particles, and is nothing when nothing is left', () => {
    const c = cloud(100, 500);
    c.weight.fill(0, 0, 50);
    expect(sum(densityGrid(c.lat, c.lon, c.weight, 26, 285).mass)).toBeCloseTo(0.5, 12);
    expect(densityGrid(c.lat, c.lon, new Float64Array(100), 26, 285)).toBeNull();
  });

  it('trims a far stray to keep the grid small, and loses only the stray', () => {
    const c = cloud(5000, 1000);
    const [la, lo] = offsetPosition(26, 285, 900e3, 900e3);
    c.lat[0] = la;
    c.lon[0] = lo;
    const g = densityGrid(c.lat, c.lon, c.weight, 26, 285);
    expect(g.nx * g.ny).toBeLessThan(1 << 20);
    expect(sum(g.mass)).toBeGreaterThan(0.99);
  });
});

describe('the display cell', () => {
  it('is the referee\'s 250 m for an ordinary cloud', () => {
    const c = cloud(10000, 1000);
    expect(displayCellM(c.lat, c.lon, c.weight, 26, 285)).toBe(250);
  });

  it('coarsens with the cloud by Silverman\'s rule, to whole 250 m steps', () => {
    // s * n^(-1/6) = 8000 * 10000^(-1/6) = 1724 m: seven cells of 250 m.
    const c = cloud(10000, 8000);
    expect(displayCellM(c.lat, c.lon, c.weight, 26, 285)).toBe(1750);
  });

  it('follows the narrow axis of a long cloud, whichever way it lies', () => {
    // 3 km across, 15 km along, lying north-east: s = 3000 gives 646 m, three cells.
    const c = cloud(10000, 15000, 26, 285, 7, { ratio: 0.2, turnDeg: 45 });
    expect(displayCellM(c.lat, c.lon, c.weight, 26, 285)).toBe(750);
  });

  it('is not coarsened by a few far strays', () => {
    const c = cloud(10000, 1000);
    for (let j = 0; j < 20; j += 1) [c.lat[j], c.lon[j]] = offsetPosition(26, 285, 200e3 * (j % 2 ? 1 : -1), 150e3);
    expect(displayCellM(c.lat, c.lon, c.weight, 26, 285)).toBe(250);
  });
});

describe('the search taken out at the finer scale', () => {
  it('splits a display cell into 250 m cells, at most three each way', () => {
    expect([250, 500, 750, 1000, 1750].map(fineSplit)).toEqual([1, 2, 3, 3, 3]);
  });

  it('bins the same particles on a grid k times finer over the same rectangle', () => {
    const c = cloud(3000, 2000);
    const left = Float64Array.from(c.weight, (w, j) => (j % 3 ? w : 0));
    const g = densityGrid(c.lat, c.lon, c.weight, 26, 285, { cellM: 750, k: 3, pad: 2, left });
    expect([g.fine.nx, g.fine.ny]).toEqual([3 * g.nx, 3 * g.ny]);
    expect(sum(g.fine.mass)).toBeCloseTo(sum(g.mass), 12);
    expect(sum(g.fine.left)).toBeCloseTo(sum(g.left), 12);
    // Every fine cell's mass is inside its display cell.
    const back = new Float64Array(g.nx * g.ny);
    for (let J = 0; J < g.fine.ny; J += 1) {
      for (let I = 0; I < g.fine.nx; I += 1) back[Math.floor(J / 3) * g.nx + Math.floor(I / 3)] += g.fine.mass[J * g.fine.nx + I];
    }
    back.forEach((m, i) => expect(m).toBeCloseTo(g.mass[i], 12));
  });

  it('reads a smooth field at the finer centres, exactly for a plane', () => {
    const plane = Float64Array.from({ length: 20 }, (_, i) => 2 * (i % 5) + 3 * Math.floor(i / 5));
    const up = upsample(plane, 5, 4, 3);
    // Fine centre (I, J) sits at ((I + 0.5) / 3 - 0.5, (J + 0.5) / 3 - 0.5) in display cells.
    for (const [I, J] of [[4, 4], [7, 2], [10, 9]]) {
      expect(up[J * 15 + I]).toBeCloseTo(2 * ((I + 0.5) / 3 - 0.5) + 3 * ((J + 0.5) / 3 - 0.5), 12);
    }
  });

  it('is the smooth map itself until something is searched', () => {
    const c = cloud(10000, 3000);
    const g = densityGrid(c.lat, c.lon, c.weight, 26, 285, { cellM: 750, k: 3, pad: 4, left: c.weight });
    const shown = shownGrid(g, 1);
    const before = upsample(blur({ nx: g.nx, ny: g.ny, mass: g.mass }, 1).mass, g.nx, g.ny, 3);
    shown.mass.forEach((m, i) => expect(m).toBeCloseTo(before[i] / 9, 15));
    expect(sum(shown.mass)).toBeCloseTo(1, 2);
  });

  it('darkens a searched strip sharply, and leaves the sea beside it alone', () => {
    // Search a north-south strip 500 m wide through the middle of a 3 km cloud.
    const c = cloud(10000, 3000);
    const left = Float64Array.from(c.weight);
    for (let j = 0; j < left.length; j += 1) {
      const [e] = relativeM(c.lat[j], c.lon[j], 26, 285);
      if (Math.abs(e) < 250) left[j] = 0;
    }
    const g = densityGrid(c.lat, c.lon, c.weight, 26, 285, { cellM: 750, k: 3, pad: 4, left });
    const was = shownGrid({ ...g, left: g.mass, fine: { ...g.fine, left: g.fine.mass } }, 1);
    const now = shownGrid(g, 1);
    const at = (grid, eM, nM) => {
      const I = Math.floor(eM / 250) - g.x0 * 3;
      const J = Math.floor(nM / 250) - g.y0 * 3;
      return grid.mass[J * grid.nx + I];
    };
    expect(at(now, 0, 0) / at(was, 0, 0)).toBeLessThan(0.45);   // on the strip
    expect(at(now, 1500, 0) / at(was, 1500, 0)).toBeGreaterThan(0.97);  // 1.5 km beside it
  });
});

describe('the display blur', () => {
  it('moves no probability, given room round the cloud', () => {
    const c = cloud(3000, 1200);
    const g = densityGrid(c.lat, c.lon, c.weight, 26, 285, { pad: blurRadius(1) });
    expect(sum(blur(g, 1).mass)).toBeCloseTo(sum(g.mass), 12);
  });

  it('spreads one cell into a Gaussian one cell wide', () => {
    const mass = new Float64Array(81);
    mass[40] = 1;
    const out = blur({ nx: 9, ny: 9, mass }, 1).mass;
    const k = (i) => Math.exp(-(i * i) / 2);
    const norm = [-3, -2, -1, 0, 1, 2, 3].reduce((s, i) => s + k(i), 0);
    expect(out[40]).toBeCloseTo(1 / norm ** 2, 12);
    expect(out[41] / out[40]).toBeCloseTo(k(1), 12);
    expect(out[40 + 9 * 2] / out[40]).toBeCloseTo(k(2), 12);
  });

  it('takes the speckle out of 10,000 particles', () => {
    // Roughness: the mean squared step between neighbouring cells, relative to the peak.
    const c = cloud(10000, 2000);
    const raw = densityGrid(c.lat, c.lon, c.weight, 26, 285, { pad: 4 });
    const rough = ({ nx, mass }) => {
      let s = 0;
      for (let i = 0; i + 1 < mass.length; i += 1) if ((i + 1) % nx) s += (mass[i + 1] - mass[i]) ** 2;
      return s / Math.max(...mass) ** 2;
    };
    expect(rough(blur(raw, 1))).toBeLessThan(rough(raw) / 5);
  });
});

describe('containment', () => {
  const table = containmentTable(Float64Array.from([0.4, 0, 0.3, 0.2, 0.1]));

  it('is the share of the probability in fuller cells', () => {
    expect(table.total).toBeCloseTo(1, 12);
    expect(containmentOf(table, 0.4)).toBe(0);
    expect(containmentOf(table, 0.3)).toBeCloseTo(0.4, 12);
    expect(containmentOf(table, 0.2)).toBeCloseTo(0.7, 12);
    expect(containmentOf(table, 0.1)).toBeCloseTo(0.9, 12);
    expect(containmentOf(table, 0)).toBe(1);
  });

  it('only ever rises as a cell drains, so a drained cell goes darker, never brighter', () => {
    let prev = -1;
    for (let m = 0.5; m >= 0; m -= 0.005) {
      const c = containmentOf(table, m);
      expect(c).toBeGreaterThanOrEqual(prev);
      expect(c).toBeLessThanOrEqual(1);
      prev = c;
    }
  });
});

describe('the contours', () => {
  it('bound the smallest areas holding half and 90 % of the probability', () => {
    const c = cloud(10000, 2000);
    const g = blur(densityGrid(c.lat, c.lon, c.weight, 26, 285, { pad: 4 }), 1);
    const [l50, l90] = levelsFor(g.mass, [0.5, 0.9]);
    const inside = (level) => sum(g.mass.filter((m) => m >= level));
    // A level is a histogram bin's lower edge: at least the share, and not much more.
    expect(inside(l50)).toBeGreaterThanOrEqual(0.5);
    expect(inside(l50)).toBeLessThan(0.51);
    expect(inside(l90)).toBeGreaterThanOrEqual(0.9);
    expect(inside(l90)).toBeLessThan(0.91);
    expect(l50).toBeGreaterThan(l90);
  });

  it('agree with sorting the cells, which they stand in for every frame', () => {
    const c = cloud(4000, 1500, 26, 285, 11);
    const g = blur(densityGrid(c.lat, c.lon, c.weight, 26, 285, { pad: 4 }), 1);
    const sorted = Float64Array.from(g.mass).sort().reverse();
    let acc = 0;
    let exact = 0;
    for (const m of sorted) {
      acc += m;
      if (acc >= 0.5 * sum(g.mass)) {
        exact = m;
        break;
      }
    }
    expect(levelsFor(g.mass, [0.5])[0] / exact).toBeGreaterThan(0.99);
    expect(levelsFor(g.mass, [0.5])[0]).toBeLessThanOrEqual(exact);
  });

  it('trace a closed ring round a single peak, every point on the level', () => {
    const n = 21;
    const mass = new Float64Array(n * n);
    for (let j = 0; j < n; j += 1) {
      for (let i = 0; i < n; i += 1) mass[j * n + i] = Math.exp(-((i - 10) ** 2 + (j - 10) ** 2) / 18);
    }
    // Not exp(-1): that lands exactly on grid corners ((3, 3) from the peak) and pins the test
    // to a tie, not to the tracing.
    const level = Math.exp(-1.03);
    const segs = contourSegments(mass, n, n, level);
    expect(segs.length).toBeGreaterThan(8);
    // Closed: every end point is shared by exactly two segments.
    const ends = new Map();
    for (const [ax, ay, bx, by] of segs) {
      for (const key of [`${ax.toFixed(9)},${ay.toFixed(9)}`, `${bx.toFixed(9)},${by.toFixed(9)}`]) {
        ends.set(key, (ends.get(key) ?? 0) + 1);
      }
    }
    expect([...ends.values()].every((k) => k === 2)).toBe(true);
    // On the level: radius sqrt(18 * 1.03) cells about the centre, to within the linear
    // interpolation along each cell edge.
    for (const [ax, ay] of segs) expect(Math.abs(Math.hypot(ax - 10, ay - 10) - Math.sqrt(18 * 1.03))).toBeLessThan(0.15);
    // Chained, the ring is one line that ends where it starts, so a dash runs round it.
    const lines = joinSegments(segs);
    expect(lines).toHaveLength(1);
    expect(lines[0]).toHaveLength(segs.length + 1);
    expect(lines[0][0]).toEqual(lines[0][lines[0].length - 1]);
  });

  it('chain two separate peaks into two rings, and an open edge into one open line', () => {
    const n = 30;
    const mass = new Float64Array(n * n);
    for (let j = 0; j < n; j += 1) {
      for (let i = 0; i < n; i += 1) {
        mass[j * n + i] = Math.exp(-((i - 7) ** 2 + (j - 15) ** 2) / 8) + Math.exp(-((i - 22) ** 2 + (j - 15) ** 2) / 8);
      }
    }
    expect(joinSegments(contourSegments(mass, n, n, 0.5))).toHaveLength(2);
    // A ramp rising to the east crosses the level once, bottom edge to top edge.
    const east = Float64Array.from({ length: 25 }, (_, k) => k % 5);
    const open = joinSegments(contourSegments(east, 5, 5, 2.5));
    expect(open).toHaveLength(1);
    expect(open[0].every(([x]) => x === 2.5)).toBe(true);
    expect(open[0].map(([, y]) => y).sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4]);
  });
});

describe('the palettes', () => {
  it('each rise monotonically in lightness, the gate for a sequential ramp', () => {
    for (const anchors of [ICE, INFERNO]) {
      let prev = -1;
      for (let t = 0; t <= 1.0001; t += 0.01) {
        const L = lightness(ramp(anchors, t));
        expect(L).toBeGreaterThan(prev - 1e-9);
        prev = L;
      }
    }
    for (const name of Object.keys(PALETTES)) {
      const lut = paletteTable(name);
      let prev = -1;
      for (let i = 0; i < 256; i += 1) {
        const L = lightness([lut[4 * i], lut[4 * i + 1], lut[4 * i + 2]]);
        expect(L).toBeGreaterThan(prev - 0.004);  // rounding to whole RGB steps
        prev = L;
      }
    }
  });

  it('fade the unlikely tail to nothing and the likeliest sea to nearly solid', () => {
    expect(alphaFor(0)).toBe(0);
    expect(alphaFor(0.01)).toBe(0);
    expect(alphaFor(1)).toBeCloseTo(0.9, 12);
    for (let t = 0.02; t <= 1; t += 0.01) expect(alphaFor(t)).toBeGreaterThan(alphaFor(t - 0.01));
  });

  it('fall back to viridis for a name it does not know', () => {
    expect(paletteTable('nonsense')).toEqual(paletteTable('viridis'));
  });

  it('are drawn on the referee\'s 250 m cell', () => {
    expect(CELL_M).toBe(250);
  });
});
