/**
 * probabilityLayer.js -- the probability map, draining as the helicopter searches (D031).
 *
 * WHAT IS DRAWN. What is LEFT of the probability: every particle's remaining weight, binned
 * in the frame of the datum marker, which drifts with the current. As the strip passes, the
 * particles it reaches lose their weight and the map drains under it.
 *
 * SMOOTHED FOR THE EYE, NOT FOR THE SCORE. 10,000 particles over a few thousand 250 m cells is
 * a handful per cell, so raw bins are speckle: neighbouring cells differ by chance and the
 * hotspot hides in the noise. So the map is a kernel density estimate: binned on a cell sized
 * to the cloud by Silverman's rule (250 m, the referee's cell (D007), for a small cloud;
 * coarser for a big one), blurred with a Gaussian one cell wide, and drawn with the browser's
 * bilinear smoothing. The cell is chosen once, on arrival (displayCellM in probabilityMap.js).
 *
 * A bandwidth that suits the whole cloud would smear a searched patch away, so the search is
 * taken out at a finer scale: the drawn map is the smooth density before the search times the
 * share left at each place, measured on cells down to 250 m (shownGrid). Unsearched sea keeps
 * a share of exactly 1 and stays smooth; searched sea goes dark where the strip went.
 *
 * Only the picture is smoothed: the referee still scores every particle exactly where it is.
 *
 * COLOUR IS CONTAINMENT, NOT RAW DENSITY. A cell's colour says how much of the probability
 * lies in sea likelier than it: the brightest colour is the likeliest sea, the middle of the
 * ramp is the edge of the smallest area holding half the probability, and the dark end the
 * edge of the area holding 90 %. Search planners read a probability map by those areas, and
 * it spends the colours where the probability is instead of on a few peak cells, which is why
 * the hotspot now stands out. The transparency follows the colour, so the unlikely tail fades
 * into the sea and the hotspot glows.
 *
 * ONE SCALE FOR THE WHOLE SEARCH. The density -> containment table is taken when the
 * helicopter arrives and never retaken, so a drained cell slides down the ramp and goes dark
 * instead of being stretched back to full brightness.
 *
 * THE LINES ARE WHAT IS LEFT. Two contours are retaken every frame: inside the dashed line is
 * half of the probability still left, inside the dotted one 90 %. At arrival they sit near the
 * ramp's middle and dark end; as the search drains the map they move to wherever the rest of
 * the chance is, which is where to fly next. They are traced on a grid blurred twice as wide
 * as the colours (probabilityMap.js says why), so they outline areas, not speckle.
 *
 * Positions are in the store convention for the arithmetic (referee.js) and converted to
 * Leaflet's -180..180 only to place the grid on screen.
 */

import L from 'leaflet';
import {
  BLUR_CELLS, CELL_M, CONTOURS, CONTOUR_BLUR_CELLS, PALETTES, blur, blurRadius, containmentOf,
  containmentTable, contourSegments, densityGrid, displayCellM, fineSplit, joinSegments,
  levelsFor, paletteGradient, paletteTable, shownGrid,
} from './probabilityMap.js';
import { offsetPosition, relativeM, trackAt } from './referee.js';
import './probabilityLayer.css';

export { CELL_M };

const display = (lon) => (lon > 180 ? lon - 360 : lon);

// ---------------------------------------------------------------------------------- palette

const STORE = 'sar.heatmap.palette';

function storedPalette() {
  try {
    const name = localStorage.getItem(STORE);
    return PALETTES[name] ? name : 'viridis';
  } catch {
    return 'viridis';
  }
}

let palette = storedPalette();

/** Every layer and legend on the page, repainted when the palette changes. */
const live = new Set();

/** The palette in use, by name. */
export function heatPalette() {
  return palette;
}

/** Choose the palette for every map on the page, and remember it for the next visit. */
export function setHeatPalette(name) {
  if (!PALETTES[name] || name === palette) return;
  palette = name;
  try {
    localStorage.setItem(STORE, name);
  } catch {
    // A private window: the choice lasts for this page only.
  }
  for (const x of live) x.repaint();
}

// ------------------------------------------------------------------------------------ layer

export const ProbabilityLayer = L.Layer.extend({
  /** `cellM` fixes the cell; left out, each window's cloud chooses its own (displayCellM). */
  initialize({ cellM = null, sigma = BLUR_CELLS, opacity = 1 } = {}) {
    this.fixedCellM = cellM;
    this.cellM = cellM ?? CELL_M;
    this.sigma = sigma;
    this.opacity = opacity;
    this.state = null;
    this.table = null;
    this.frame = null;
  },

  onAdd(map) {
    this.map = map;
    this.canvas = L.DomUtil.create('canvas', 'leaflet-zoom-hide probability-layer');
    this.canvas.style.pointerEvents = 'none';
    this.image = document.createElement('canvas');
    map.getPanes().overlayPane.appendChild(this.canvas);
    map.on('moveend zoomend resize', this.redraw, this);
    live.add(this);
    this.repaint();
  },

  onRemove(map) {
    map.off('moveend zoomend resize', this.redraw, this);
    live.delete(this);
    this.canvas.remove();
    this.map = null;
  },

  /**
   * The grid at the state's time: `shown`, the map as drawn (probabilityMap.js's shownGrid,
   * on cells split `k` ways), and `smooth`, what is left blurred CONTOUR_BLUR_CELLS display
   * cells wide for the contours.
   */
  grid(state) {
    const [mlat, mlon] = trackAt(state.marker, state.t);
    const wide = Math.max(this.sigma, CONTOUR_BLUR_CELLS);
    const k = fineSplit(this.cellM);
    const prior = state.prior ?? state.weight;
    const raw = densityGrid(state.lat, state.lon, prior, mlat, mlon,
      { cellM: this.cellM, pad: blurRadius(wide) + 1, k, left: state.weight });
    if (!raw) return null;
    return {
      x0: raw.x0, y0: raw.y0, nx: raw.nx, ny: raw.ny, k, mlat, mlon,
      shown: shownGrid(raw, this.sigma),
      smooth: blur({ nx: raw.nx, ny: raw.ny, mass: raw.left }, wide).mass,
    };
  },

  /**
   * What to draw: {lat, lon, weight} particles (store longitude), and the marker's track
   * and time, which fix the cells; `prior`, the weights before the search, if the search has
   * begun. `resetScale` takes the cells and the colour scale from this state.
   */
  setState(state, { resetScale = false } = {}) {
    this.state = state;
    const arriving = resetScale || this.table === null;
    if (arriving) {
      const [mlat, mlon] = trackAt(state.marker, state.t);
      this.cellM = this.fixedCellM ?? displayCellM(state.lat, state.lon, state.weight, mlat, mlon);
    }
    this.frame = this.grid(state);
    if (arriving && this.frame) this.table = containmentTable(this.frame.shown.mass);
    this.compose();
    this.redraw();
  },

  clear() {
    this.state = null;
    this.table = null;
    this.frame = null;
    this.redraw();
  },

  /** The palette changed: recolour the grid already worked out. */
  repaint() {
    this.compose();
    this.redraw();
  },

  /** Colour the grid into an image, one pixel per cell, north row first; and the contours. */
  compose() {
    const f = this.frame;
    if (!f || !this.table || !this.image) return;
    const { nx, ny, mass } = f.shown;
    this.image.width = nx;
    this.image.height = ny;
    const ctx = this.image.getContext('2d');
    const img = ctx.createImageData(nx, ny);
    const lut = paletteTable(palette);
    const px = img.data;
    for (let j = 0; j < ny; j += 1) {
      for (let i = 0; i < nx; i += 1) {
        const m = mass[j * nx + i];
        if (!(m > 0)) continue;
        const t = 1 - containmentOf(this.table, m);
        if (t <= 0.01) continue;
        const k = 4 * Math.round(255 * t);
        const o = 4 * ((ny - 1 - j) * nx + i);
        px[o] = lut[k];
        px[o + 1] = lut[k + 1];
        px[o + 2] = lut[k + 2];
        px[o + 3] = lut[k + 3];
      }
    }
    ctx.putImageData(img, 0, 0);
    f.levels = levelsFor(f.smooth, CONTOURS.map((c) => c.share));
    f.lines = f.levels.map((level) => joinSegments(contourSegments(f.smooth, f.nx, f.ny, level)));
  },

  /**
   * The grid's corners on screen: where cell centre (i, j) lands is x(i), y(j). The grid is
   * east-north metres about the marker, so on a map a few tens of kilometres across it is a
   * plain rectangle to well within a cell.
   */
  placement() {
    const f = this.frame;
    const c = this.cellM;
    const [la0, lo0] = offsetPosition(f.mlat, f.mlon, f.x0 * c, f.y0 * c);
    const [la1, lo1] = offsetPosition(f.mlat, f.mlon, (f.x0 + f.nx) * c, (f.y0 + f.ny) * c);
    const sw = this.map.latLngToContainerPoint([la0, display(lo0)]);
    const ne = this.map.latLngToContainerPoint([la1, display(lo1)]);
    return {
      sw,
      ne,
      x: (i) => sw.x + ((i + 0.5) / f.nx) * (ne.x - sw.x),
      y: (j) => sw.y + ((j + 0.5) / f.ny) * (ne.y - sw.y),
    };
  },

  redraw() {
    if (!this.map) return;
    const size = this.map.getSize();
    const ratio = window.devicePixelRatio || 1;
    const canvas = this.canvas;
    canvas.width = size.x * ratio;
    canvas.height = size.y * ratio;
    canvas.style.width = `${size.x}px`;
    canvas.style.height = `${size.y}px`;
    L.DomUtil.setPosition(canvas, this.map.containerPointToLayerPoint([0, 0]));
    const ctx = canvas.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, size.x, size.y);
    if (!this.state || !this.frame || !this.table) return;

    const p = this.placement();
    ctx.globalAlpha = this.opacity;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(this.image, p.sw.x, p.ne.y, p.ne.x - p.sw.x, p.sw.y - p.ne.y);
    ctx.globalAlpha = 1;

    // The contours, with a dark edge under each so they read over the brightest colours.
    ctx.lineCap = 'round';
    CONTOURS.forEach((style, k) => {
      const lines = this.frame.lines?.[k] ?? [];
      if (!lines.length) return;
      ctx.beginPath();
      for (const line of lines) {
        line.forEach(([x, y], i) => (i ? ctx.lineTo(p.x(x), p.y(y)) : ctx.moveTo(p.x(x), p.y(y))));
      }
      ctx.setLineDash([]);
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.45)';
      ctx.lineWidth = style.width + 2;
      ctx.stroke();
      ctx.setLineDash(style.dash);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.92)';
      ctx.lineWidth = style.width;
      ctx.stroke();
    });
    ctx.setLineDash([]);
  },

  /**
   * What the map says at a place, for a hover readout: the share of all the probability in
   * the cell there (smoothed, as drawn), the cell's size, and which contour it is inside, if
   * any (0 for the half, 1 for 90 %, -1 outside both). Null off the grid.
   */
  valueAt(latlng) {
    const f = this.frame;
    if (!f || !this.table) return null;
    const [e, n] = relativeM(latlng.lat, latlng.lng, f.mlat, f.mlon);
    const I = Math.floor((e * f.k) / this.cellM) - f.x0 * f.k;
    const J = Math.floor((n * f.k) / this.cellM) - f.y0 * f.k;
    if (I < 0 || J < 0 || I >= f.shown.nx || J >= f.shown.ny) return null;
    const c = Math.floor(J / f.k) * f.nx + Math.floor(I / f.k);
    const inside = (f.levels ?? []).findIndex((level) => f.smooth[c] >= level);
    return { share: f.shown.mass[J * f.shown.nx + I] / this.table.total, cellM: this.cellM / f.k, inside };
  },
});

// ----------------------------------------------------------------------------------- legend

/**
 * The key, drawn from the same palette table as the map so it cannot drift from it: the
 * ramp with the half and 90 % marks, the two contour styles, and the palette picker.
 */
export const ProbabilityLegend = L.Control.extend({
  options: { position: 'bottomleft', compact: false },

  onAdd() {
    this.el = L.DomUtil.create('div', `prob-legend${this.options.compact ? ' compact' : ''}`);
    L.DomEvent.disableClickPropagation(this.el);
    L.DomEvent.disableScrollPropagation(this.el);
    // The game steers by the mouse: reaching for a colour must not turn the helicopter.
    L.DomEvent.on(this.el, 'mousemove', L.DomEvent.stopPropagation);
    live.add(this);
    this.repaint();
    return this.el;
  },

  onRemove() {
    live.delete(this);
  },

  repaint() {
    if (!this.el) return;
    const line = ({ dash, width }) => `<svg width="26" height="8" aria-hidden="true">`
      + `<line x1="1" y1="4" x2="25" y2="4" stroke="#000a" stroke-width="${width + 2}"/>`
      + `<line x1="1" y1="4" x2="25" y2="4" stroke="#fff" stroke-width="${width}" `
      + `stroke-dasharray="${dash.join(' ')}" stroke-linecap="round"/></svg>`;
    const picks = Object.entries(PALETTES).map(([name, p]) => `<button type="button" data-palette="${name}" `
      + `aria-pressed="${name === palette}" title="${p.label} colours">`
      + `<span class="prob-swatch" style="background:${paletteGradient(name)}"></span>${p.label}</button>`).join('');
    this.el.innerHTML = '<div class="prob-title">Where they could be</div>'
      + `<div class="prob-bar" style="background:${paletteGradient(palette)}"></div>`
      + '<div class="prob-ticks"><span style="left:10%">90 %</span><span style="left:50%">50 %</span>'
      + '<span style="left:90%">10 %</span></div>'
      + '<div class="prob-note">Brighter is likelier: on arrival the brightest colours hold the likeliest '
      + '10 % of the chance. Searched sea goes dark.</div>'
      + `<div class="prob-lines">${CONTOURS.map((c) => `<span>${line(c)}${c.label}</span>`).join('')}</div>`
      + `<div class="prob-picks" role="group" aria-label="colours">${picks}</div>`;
    for (const b of this.el.querySelectorAll('button[data-palette]')) {
      b.addEventListener('click', () => setHeatPalette(b.dataset.palette));
    }
  },
});
