/**
 * probabilityMap.js -- the arithmetic behind the probability map (probabilityLayer.js draws
 * it). Pure, no Leaflet, so it is tested on its own (tests/probabilityLayer.test.js).
 *
 *   displayCellM     the cell size: 250 m, or coarser for a big cloud (Silverman's rule)
 *   densityGrid      the probability before and after the search, on those cells about the
 *                    marker, and on cells split finer
 *   blur             the display smoothing: a Gaussian one cell wide (a kernel density estimate)
 *   shownGrid        the map as drawn: smooth where unsearched, sharp where searched
 *   containment*     a cell's colour: the share of the probability in likelier cells
 *   levelsFor        the masses that bound the likeliest half and 90 % (the contour levels)
 *   contourSegments  marching squares: the contour lines at those levels
 *   joinSegments     the segments chained into lines, so a dashed line is dashed along it
 *   paletteTable     the colour schemes, with the transparency that fades the tail
 *
 * Why each of these, and why the map is smoothed for the eye but never for the score, is in
 * probabilityLayer.js.
 */

import { ICE, INFERNO, VIRIDIS, ramp } from './colormap.js';
import { relativeM } from './referee.js';

export const CELL_M = 250;

/** The display blur's Gaussian width, in cells: one cell. */
export const BLUR_CELLS = 1;

/**
 * The contours are traced on a smoother grid, blurred two display cells wide: at one cell the
 * edge of "half of what is left" is ragged with chance speckle, and after a pattern has swept
 * the middle it wraps every single track instead of the searched area.
 */
export const CONTOUR_BLUR_CELLS = 2;

/** The contours: inside each, this share of the probability still left. */
export const CONTOURS = [
  { share: 0.5, dash: [7, 4], width: 1.6, label: 'half of what is left' },
  { share: 0.9, dash: [1.5, 3.5], width: 1.4, label: '90 % of what is left' },
];

/** The colour schemes a visitor may choose, in order. */
export const PALETTES = {
  viridis: { label: 'Viridis', anchors: VIRIDIS },
  heat: { label: 'Heat', anchors: INFERNO },
  ice: { label: 'Ice', anchors: ICE },
};

/** A grid bigger than this is trimmed to the bulk of the cloud: a far stray would cost megabytes. */
const MAX_CELLS = 1 << 20;

/** The q-quantile of a sorted list (the nearest rank below). */
function quantile(sorted, q) {
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(q * (sorted.length - 1))))];
}

/**
 * The display cell for a cloud, in metres: 250 m (the referee's cell, D007), or a whole
 * multiple of it when the cloud is big enough that 250 m is finer than its particles can
 * resolve. A cloud 8 km across spreads 10,000 particles over 100,000 cells of 250 m: a tenth of
 * a particle each, all speckle, and a frame that takes 30 ms to draw.
 *
 * The size is Silverman's rule of thumb for a kernel density estimate in two dimensions,
 * h = s * n^(-1/6), with s the cloud's spread across its narrow axis (so a jet's long cloud
 * keeps its detail across the jet), measured robustly as the interquartile range / 1.349 so a
 * few strays do not coarsen the map. The cell is h to the nearest 250 m and the display blur
 * one cell wide, so the smoothing follows the rule and the grid stays a few thousand cells
 * however big the cloud.
 */
export function displayCellM(lat, lon, weight, mlat, mlon) {
  const es = [];
  const ns = [];
  for (let j = 0; j < weight.length; j += 1) {
    if (!(weight[j] > 0) || !Number.isFinite(lat[j])) continue;
    const [e, n] = relativeM(lat[j], lon[j], mlat, mlon);
    es.push(e);
    ns.push(n);
  }
  const n = es.length;
  if (n < 4) return CELL_M;
  const me = es.reduce((a, b) => a + b, 0) / n;
  const mn = ns.reduce((a, b) => a + b, 0) / n;
  let [cee, cnn, cen] = [0, 0, 0];
  for (let j = 0; j < n; j += 1) {
    cee += (es[j] - me) ** 2;
    cnn += (ns[j] - mn) ** 2;
    cen += (es[j] - me) * (ns[j] - mn);
  }
  // The long axis is at theta from east; the narrow one at right angles to it.
  const theta = 0.5 * Math.atan2(2 * cen, cee - cnn);
  const [ue, un] = [-Math.sin(theta), Math.cos(theta)];
  const across = Float64Array.from(es, (e, j) => e * ue + ns[j] * un).sort();
  const s = (quantile(across, 0.75) - quantile(across, 0.25)) / 1.349;
  return CELL_M * Math.max(1, Math.round((s * n ** (-1 / 6)) / CELL_M));
}

/** How many fine cells a display cell splits into each way: down to 250 m, at most 3. */
export const fineSplit = (cellM) => Math.max(1, Math.min(3, Math.round(cellM / CELL_M)));

/**
 * Probability on a dense grid about the marker, `pad` empty cells round the cloud:
 * {x0, y0, nx, ny, mass}, mass[(iy - y0) * nx + (ix - x0)] for the cell [ix, iy], south row
 * first. Null when there is none.
 *
 * `left`, a second set of weights for the same particles (what the search has left of
 * `weight`), is binned alongside as `left`. With `k` > 1 both are binned again on a fine grid,
 * each cell split k by k, as `fine` {k, nx, ny, mass, left}: the fine grid covers exactly the
 * same rectangle, so fine cell [I, J] lies in cell [floor(I / k), floor(J / k)].
 */
export function densityGrid(lat, lon, weight, mlat, mlon, { cellM = CELL_M, pad = 0, k = 1, left = null } = {}) {
  const n = weight.length;
  const fx = new Int32Array(n);
  const fy = new Int32Array(n);
  const keep = new Uint8Array(n);
  let [x0, x1, y0, y1] = [Infinity, -Infinity, Infinity, -Infinity];
  for (let j = 0; j < n; j += 1) {
    if (!(weight[j] > 0) || !Number.isFinite(lat[j])) continue;
    const [e, nn] = relativeM(lat[j], lon[j], mlat, mlon);
    fx[j] = Math.floor((e * k) / cellM);
    fy[j] = Math.floor((nn * k) / cellM);
    keep[j] = 1;
    const [ix, iy] = [Math.floor(fx[j] / k), Math.floor(fy[j] / k)];
    x0 = Math.min(x0, ix);
    x1 = Math.max(x1, ix);
    y0 = Math.min(y0, iy);
    y1 = Math.max(y1, iy);
  }
  if (x0 > x1) return null;
  if ((x1 - x0 + 1 + 2 * pad) * (y1 - y0 + 1 + 2 * pad) > MAX_CELLS) {
    // Trim to the middle 99.8 % each way: the strays left out are not drawn, nothing else.
    const xs = Int32Array.from(fx.filter((_, j) => keep[j]), (v) => Math.floor(v / k)).sort();
    const ys = Int32Array.from(fy.filter((_, j) => keep[j]), (v) => Math.floor(v / k)).sort();
    [x0, x1, y0, y1] = [quantile(xs, 0.001), quantile(xs, 0.999), quantile(ys, 0.001), quantile(ys, 0.999)];
  }
  x0 -= pad;
  y0 -= pad;
  const nx = x1 + pad - x0 + 1;
  const ny = y1 + pad - y0 + 1;
  const grid = { x0, y0, nx, ny, mass: new Float64Array(nx * ny) };
  if (left) grid.left = new Float64Array(nx * ny);
  if (k > 1) {
    grid.fine = { k, nx: nx * k, ny: ny * k, mass: new Float64Array(nx * ny * k * k) };
    if (left) grid.fine.left = new Float64Array(nx * ny * k * k);
  }
  for (let j = 0; j < n; j += 1) {
    if (!keep[j]) continue;
    const I = fx[j] - x0 * k;
    const J = fy[j] - y0 * k;
    if (I < 0 || I >= nx * k || J < 0 || J >= ny * k) continue;
    const c = Math.floor(J / k) * nx + Math.floor(I / k);
    grid.mass[c] += weight[j];
    if (left) grid.left[c] += left[j];
    if (k > 1) {
      grid.fine.mass[J * nx * k + I] += weight[j];
      if (left) grid.fine.left[J * nx * k + I] += left[j];
    }
  }
  return grid;
}

/** The radius, in cells, a blur of width `sigma` reaches. */
export const blurRadius = (sigma) => Math.ceil(3 * sigma);

/** The grid blurred with a Gaussian `sigma` cells wide, one axis then the other. */
export function blur(grid, sigma = BLUR_CELLS) {
  if (!(sigma > 0)) return grid;
  const r = blurRadius(sigma);
  const k = new Float64Array(2 * r + 1);
  for (let i = -r; i <= r; i += 1) k[i + r] = Math.exp(-(i * i) / (2 * sigma * sigma));
  const sum = k.reduce((a, b) => a + b, 0);
  for (let i = 0; i < k.length; i += 1) k[i] /= sum;
  const { nx, ny, mass } = grid;
  const across = new Float64Array(nx * ny);
  const out = new Float64Array(nx * ny);
  for (let y = 0; y < ny; y += 1) {
    for (let x = 0; x < nx; x += 1) {
      let s = 0;
      for (let i = Math.max(-r, -x); i <= Math.min(r, nx - 1 - x); i += 1) s += k[i + r] * mass[y * nx + x + i];
      across[y * nx + x] = s;
    }
  }
  for (let y = 0; y < ny; y += 1) {
    for (let x = 0; x < nx; x += 1) {
      let s = 0;
      for (let i = Math.max(-r, -y); i <= Math.min(r, ny - 1 - y); i += 1) s += k[i + r] * across[(y + i) * nx + x];
      out[y * nx + x] = s;
    }
  }
  return { ...grid, mass: out };
}

/**
 * Values at cell centres, nx by ny, read at the centres of a grid k times finer over the same
 * rectangle: bilinear between the four nearest, held level past the outer centres.
 */
export function upsample(values, nx, ny, k) {
  if (k === 1) return values;
  const out = new Float64Array(nx * k * ny * k);
  const axis = (m, size) => {
    const u = Math.min(size - 1, Math.max(0, (m + 0.5) / k - 0.5));
    const i = Math.min(size - 2, Math.floor(u));
    return size < 2 ? [0, 0, 0] : [i, i + 1, u - i];
  };
  for (let J = 0; J < ny * k; J += 1) {
    const [j0, j1, fj] = axis(J, ny);
    for (let I = 0; I < nx * k; I += 1) {
      const [i0, i1, fi] = axis(I, nx);
      const south = values[j0 * nx + i0] * (1 - fi) + values[j0 * nx + i1] * fi;
      const north = values[j1 * nx + i0] * (1 - fi) + values[j1 * nx + i1] * fi;
      out[J * nx * k + I] = south * (1 - fj) + north * fj;
    }
  }
  return out;
}

/**
 * The map as drawn, from a densityGrid with `left` (and `fine` for a coarse cell): {nx, ny,
 * mass} on the fine grid, mass per fine cell.
 *
 * The density before any search, smoothed at the display cell (Silverman's width), times the
 * share the search has left, measured on the finer grid: left / before, each blurred one fine
 * cell. Where nothing has been searched that share is exactly 1, so the unsearched sea is as
 * smooth as the cloud warrants; where the strip has been it falls to what is left there, at
 * 250 m (or a third of the cell), so a searched patch shows as one instead of being smeared
 * out by a bandwidth chosen for the whole cloud. With one cell per display cell it is simply
 * what is left, smoothed.
 */
export function shownGrid(grid, sigma = BLUR_CELLS) {
  const { nx, ny, fine } = grid;
  const smooth = (mass, w = nx, h = ny) => blur({ nx: w, ny: h, mass }, sigma).mass;
  if (!fine) return { nx, ny, mass: smooth(grid.left ?? grid.mass) };
  const { k } = fine;
  const before = upsample(smooth(grid.mass), nx, ny, k);
  const finePrior = smooth(fine.mass, fine.nx, fine.ny);
  const fineLeft = fine.left ? smooth(fine.left, fine.nx, fine.ny) : finePrior;
  const mass = new Float64Array(fine.nx * fine.ny);
  for (let i = 0; i < mass.length; i += 1) {
    const share = finePrior[i] > 0 ? Math.min(1, fineLeft[i] / finePrior[i]) : 1;
    mass[i] = (before[i] * share) / (k * k);
  }
  return { nx: fine.nx, ny: fine.ny, mass };
}

/**
 * The containment table of a grid: its cells' masses, largest first (`v`), and for each the
 * share of the total in cells holding more (`c`). `total` is the grid's whole mass.
 */
export function containmentTable(mass) {
  const v = Float64Array.from(mass.filter((x) => x > 0)).sort().reverse();
  const c = new Float64Array(v.length);
  let total = 0;
  for (let i = 0; i < v.length; i += 1) {
    c[i] = total;
    total += v[i];
  }
  for (let i = 0; i < v.length; i += 1) c[i] /= total;
  return { v, c, total };
}

/**
 * A cell holding mass m, read on a containment table: the share of the table's probability
 * in cells holding more. 0 at or above the fullest cell, rising to 1 as m falls to nothing.
 */
export function containmentOf(table, m) {
  const { v, c } = table;
  const n = v.length;
  if (!n || m >= v[0]) return 0;
  if (m <= v[n - 1]) return v[n - 1] > 0 ? 1 - (m / v[n - 1]) * (1 - c[n - 1]) : 1;
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (v[mid] > m) lo = mid; else hi = mid;
  }
  const f = (v[lo] - m) / (v[lo] - v[hi]);
  return c[lo] + f * (c[hi] - c[lo]);
}

/**
 * For each share, the least mass a cell must hold to be inside the smallest area holding that
 * share of the grid's total: the contour level. One pass, through a fine logarithmic
 * histogram, because it is retaken every frame; a level is a bin's lower edge, so the area it
 * bounds holds at least the share.
 */
export function levelsFor(mass, shares, bins = 2048) {
  let top = 0;
  let total = 0;
  for (const x of mass) {
    if (x > 0) {
      total += x;
      if (x > top) top = x;
    }
  }
  if (!(total > 0)) return shares.map(() => Infinity);
  const floor = top * 1e-9;
  const per = bins / Math.log(top / floor);
  const hist = new Float64Array(bins);
  for (const x of mass) if (x > floor) hist[Math.min(bins - 1, Math.floor(Math.log(x / floor) * per))] += x;
  const order = shares.map((s, i) => [s, i]).sort((a, b) => a[0] - b[0]);
  const out = shares.map(() => floor);
  let acc = 0;
  let s = 0;
  for (let b = bins - 1; b >= 0 && s < order.length; b -= 1) {
    acc += hist[b];
    while (s < order.length && acc >= order[s][0] * total * (1 - 1e-12)) {
      out[order[s][1]] = floor * Math.exp(b / per);
      s += 1;
    }
  }
  return out;
}

/**
 * Marching squares: the line where a grid crosses `level`, as segments [x0, y0, x1, y1] in
 * cell units, (i, j) being the centre of cell [i, j]. A saddle is split by its centre value.
 */
export function contourSegments(mass, nx, ny, level) {
  const out = [];
  const at = (a, b) => (level - a) / (b - a);
  for (let j = 0; j + 1 < ny; j += 1) {
    for (let i = 0; i + 1 < nx; i += 1) {
      const a = mass[j * nx + i];
      const b = mass[j * nx + i + 1];
      const c = mass[(j + 1) * nx + i + 1];
      const d = mass[(j + 1) * nx + i];
      const code = (a >= level) | ((b >= level) << 1) | ((c >= level) << 2) | ((d >= level) << 3);
      if (code === 0 || code === 15) continue;
      const bottom = () => [i + at(a, b), j];
      const right = () => [i + 1, j + at(b, c)];
      const top = () => [i + at(d, c), j + 1];
      const left = () => [i, j + at(a, d)];
      const seg = (p, q) => out.push([...p(), ...q()]);
      const high = (a + b + c + d) / 4 >= level;
      switch (code) {
        case 1: case 14: seg(left, bottom); break;
        case 2: case 13: seg(bottom, right); break;
        case 3: case 12: seg(left, right); break;
        case 4: case 11: seg(right, top); break;
        case 6: case 9: seg(bottom, top); break;
        case 7: case 8: seg(left, top); break;
        case 5:
          if (high) { seg(bottom, right); seg(left, top); } else { seg(left, bottom); seg(right, top); }
          break;
        case 10:
          if (high) { seg(left, bottom); seg(right, top); } else { seg(bottom, right); seg(left, top); }
          break;
        default:
      }
    }
  }
  return out;
}

/**
 * Marching squares' segments chained into lines, [[x, y], ...]: a closed ring ends where it
 * starts. Two cells compute a shared edge's crossing from the same two values, so an end
 * shared by two segments is the same number in both and can be matched exactly.
 */
export function joinSegments(segs) {
  const key = (x, y) => `${x},${y}`;
  const ends = new Map();
  segs.forEach(([ax, ay, bx, by], i) => {
    for (const [k, end] of [[key(ax, ay), 0], [key(bx, by), 1]]) {
      if (!ends.has(k)) ends.set(k, []);
      ends.get(k).push([i, end]);
    }
  });
  const used = new Uint8Array(segs.length);
  const lines = [];
  for (let s = 0; s < segs.length; s += 1) {
    if (used[s]) continue;
    used[s] = 1;
    const line = [[segs[s][0], segs[s][1]], [segs[s][2], segs[s][3]]];
    for (const forward of [true, false]) {
      for (;;) {
        const [x, y] = forward ? line[line.length - 1] : line[0];
        const next = (ends.get(key(x, y)) ?? []).find(([i]) => !used[i]);
        if (!next) break;
        const [i, end] = next;
        used[i] = 1;
        const far = end === 0 ? [segs[i][2], segs[i][3]] : [segs[i][0], segs[i][1]];
        if (forward) line.push(far); else line.unshift(far);
      }
    }
    lines.push(line);
  }
  return lines;
}

/**
 * Opacity for a place on the ramp: the likeliest sea nearly solid, fading to nothing at the
 * edge of the likeliest 99 % (t = 0.01). Without the cut, the last few strays' cells draw as a
 * staircase of faint dark squares round the cloud.
 */
export const alphaFor = (t) => 0.9 * Math.max(0, Math.min(1, (t - 0.01) / 0.99)) ** 0.6;

/** 256 RGBA entries for a palette: colour and transparency for t = 0..1. */
export function paletteTable(name) {
  const anchors = (PALETTES[name] ?? PALETTES.viridis).anchors;
  const lut = new Uint8ClampedArray(256 * 4);
  for (let i = 0; i < 256; i += 1) {
    const t = i / 255;
    const [r, g, b] = ramp(anchors, 0.08 + 0.92 * t);
    lut.set([r, g, b, Math.round(255 * alphaFor(t))], 4 * i);
  }
  return lut;
}

/** A palette's ramp as a CSS gradient, left least likely, right likeliest. */
export function paletteGradient(name) {
  const lut = paletteTable(name);
  const stops = [];
  for (let i = 0; i <= 10; i += 1) {
    const k = Math.round(25.5 * i) * 4;
    stops.push(`rgba(${lut[k]}, ${lut[k + 1]}, ${lut[k + 2]}, ${(lut[k + 3] / 255).toFixed(2)}) ${i * 10}%`);
  }
  return `linear-gradient(to right, ${stops.join(', ')})`;
}
