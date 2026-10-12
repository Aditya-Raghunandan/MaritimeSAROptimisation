/**
 * resultCard.js -- the picture a player takes home: the cloud the search started on, and the
 * three flights over it (you, the Coast Guard's pattern, the AI), with the real person's drift
 * drawn where it really went. North up, on true metres about where the search started.
 *
 * The email (mailer/) carries it inline; nothing here sends anything. The geometry is kept
 * apart from the drawing so it can be tested without a canvas.
 */

import { M_PER_DEG_LAT } from './geo.js';
import { trackAt } from './referee.js';

export const CARD_PX = 640;
const HEAT = [[5, 16, 28], [43, 26, 90], [163, 31, 107], [255, 90, 31], [255, 194, 74], [255, 251, 224]];

/** [east, north] metres of (lat, lon) from an origin, on the local flat earth. */
export function toMetres(lat, lon, lat0, lon0) {
  return [(lon - lon0) * M_PER_DEG_LAT * Math.cos((lat0 * Math.PI) / 180), (lat - lat0) * M_PER_DEG_LAT];
}

/**
 * Everything the card draws, in metres about the marker at arrival: the cloud at arrival
 * (x, y, w), each flight's ground track, the real person's drift during the search, and a
 * square view that holds the flights, the person and the middle 96 % of the cloud.
 * `trails` are [[t, lat, lon], ...] as playback.js's trail() gives them.
 */
export function cardGeometry(window, trails, windowS = 2700) {
  const { cloud, marker, target } = window;
  const [lat0, lon0] = trackAt(marker, 0);
  const n = cloud.particles;
  const x = new Float64Array(n);
  const y = new Float64Array(n);
  for (let j = 0; j < n; j += 1) [x[j], y[j]] = toMetres(cloud.lat[j], cloud.lon[j], lat0, lon0);
  const paths = trails.map((tr) => tr.map(([, la, lo]) => toMetres(la, lo, lat0, lon0)));
  const person = [];
  if (target) {
    const start = Math.max(target.tS[0], 0);
    const end = Math.min(windowS, target.tS[target.tS.length - 1]);
    for (let s = start; s < end; s += 60) person.push(toMetres(...trackAt(target, s), lat0, lon0));
    if (end >= start) person.push(toMetres(...trackAt(target, end), lat0, lon0));
  }
  const q = (arr, p) => {
    const s = Float64Array.from(arr).sort();
    return s[Math.min(s.length - 1, Math.max(0, Math.floor(p * (s.length - 1))))];
  };
  const xs = [q(x, 0.02), q(x, 0.98)];
  const ys = [q(y, 0.02), q(y, 0.98)];
  for (const [e, nn] of [...paths.flat(), ...person]) {
    xs.push(e);
    ys.push(nn);
  }
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
  const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const half = 0.56 * Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 2000);
  return { x, y, weight: cloud.weight, paths, person, view: { cx, cy, half } };
}

function heatColour(v) {
  const t = Math.max(0, Math.min(1, v)) * (HEAT.length - 1);
  const i = Math.min(HEAT.length - 2, Math.floor(t));
  const f = t - i;
  return HEAT[i].map((c, k) => Math.round(c + f * (HEAT[i + 1][k] - c)));
}

/** The cloud as a soft heat image, bins x bins, normalised to its fullest cell. */
export function heatGrid(geo, bins = 160) {
  const { x, y, weight, view } = geo;
  const g = new Float64Array(bins * bins);
  const size = (2 * view.half) / bins;
  for (let j = 0; j < x.length; j += 1) {
    const c = Math.floor((x[j] - (view.cx - view.half)) / size);
    const r = Math.floor((view.cy + view.half - y[j]) / size);
    if (c >= 0 && c < bins && r >= 0 && r < bins) g[r * bins + c] += weight ? weight[j] : 1;
  }
  // three box blurs make a near-Gaussian
  let a = g;
  for (let pass = 0; pass < 3; pass += 1) {
    const b = new Float64Array(bins * bins);
    for (let r = 0; r < bins; r += 1) {
      for (let c = 0; c < bins; c += 1) {
        let s = 0;
        let m = 0;
        for (let dr = -2; dr <= 2; dr += 1) {
          for (let dc = -2; dc <= 2; dc += 1) {
            const rr = r + dr;
            const cc = c + dc;
            if (rr >= 0 && rr < bins && cc >= 0 && cc < bins) {
              s += a[rr * bins + cc];
              m += 1;
            }
          }
        }
        b[r * bins + c] = s / m;
      }
    }
    a = b;
  }
  let max = 0;
  for (const v of a) max = Math.max(max, v);
  if (max > 0) for (let i = 0; i < a.length; i += 1) a[i] /= max;
  return a;
}

/**
 * Draw the card on a canvas, CARD_PX square: the heat, the three flights, the person, a scale
 * bar and a key. `flights` is [{label, colour, trail}] with the player first.
 */
export function drawResultCard(canvas, window, flights, windowS = 2700) {
  const W = CARD_PX;
  canvas.width = W;
  canvas.height = W;
  const ctx = canvas.getContext('2d');
  const geo = cardGeometry(window, flights.map((f) => f.trail), windowS);
  const { cx, cy, half } = geo.view;
  const px = ([e, nn]) => [((e - (cx - half)) / (2 * half)) * W, ((cy + half - nn) / (2 * half)) * W];
  ctx.fillStyle = '#05101c';
  ctx.fillRect(0, 0, W, W);

  const bins = 160;
  const heat = heatGrid(geo, bins);
  const img = ctx.createImageData(bins, bins);
  for (let i = 0; i < heat.length; i += 1) {
    const v = heat[i] ** 0.7;
    const [r, g, b] = heatColour(v);
    img.data.set([r, g, b, Math.round(255 * Math.min(1, v / 0.12) * 0.95)], 4 * i);
  }
  const off = document.createElement('canvas');
  off.width = bins;
  off.height = bins;
  off.getContext('2d').putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, 0, 0, W, W);

  // the flights: the strip each one saw, then its centre line; the player's on top
  const metresPerPx = (2 * half) / W;
  for (const f of [...flights].reverse()) {
    const pts = geo.paths[flights.indexOf(f)].map(px);
    if (pts.length < 2) continue;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = f.colour;
    ctx.globalAlpha = 0.35;
    ctx.lineWidth = Math.max(3, 185.2 / metresPerPx);
    ctx.beginPath();
    pts.forEach(([a, b], i) => (i ? ctx.lineTo(a, b) : ctx.moveTo(a, b)));
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.lineWidth = f === flights[0] ? 2.5 : 1.6;
    ctx.stroke();
  }
  // the real person: where they drifted while the helicopters searched
  if (geo.person.length) {
    const pts = geo.person.map(px);
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = '#ff4d6d';
    ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach(([a, b], i) => (i ? ctx.lineTo(a, b) : ctx.moveTo(a, b)));
    ctx.stroke();
    ctx.setLineDash([]);
    const [a, b] = pts[pts.length - 1];
    ctx.fillStyle = '#ff4d6d';
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(a, b, 7, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  }

  // the key and a scale bar
  ctx.font = '600 15px system-ui, Segoe UI, Arial, sans-serif';
  const key = [...flights.map((f) => [f.colour, f.label]), ['#ff4d6d', 'where they really were']];
  ctx.fillStyle = 'rgba(0,0,0,0.6)';
  ctx.fillRect(12, W - 20 - 22 * key.length, 230, 22 * key.length + 10);
  key.forEach(([colour, label], i) => {
    const yy = W - 8 - 22 * (key.length - i);
    ctx.fillStyle = colour;
    ctx.fillRect(22, yy - 4, 18, 5);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, 48, yy + 2);
  });
  const step = [500, 1000, 2000, 5000, 10000].find((s) => s / metresPerPx > 70) ?? 10000;
  const len = step / metresPerPx;
  ctx.fillStyle = '#fff';
  ctx.fillRect(W - 20 - len, W - 24, len, 3);
  ctx.textAlign = 'right';
  ctx.fillText(step >= 1000 ? `${step / 1000} km` : `${step} m`, W - 20, W - 32);
  ctx.textAlign = 'left';
  ctx.fillText('N ↑', W - 44, 28);
  return canvas;
}
