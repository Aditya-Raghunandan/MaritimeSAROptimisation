/**
 * chart.js -- the floating panel for a clicked cell: tiles, series, drift.
 *
 * uPlot rather than Chart.js: about 40 KB, built for exactly this, and it
 * redraws fast enough to follow the time slider without stuttering.
 *
 * THE SIZING BUG THIS FILE USED TO HAVE, because it is easy to reintroduce.
 * The panel is hidden until a cell is clicked, and the previous version built
 * the plot with `width: this.el.clientWidth` while it was still hidden. A
 * hidden element measures 0, so uPlot was constructed 0 px wide and stayed
 * that way -- the title wrapped to one word per line, the legend stacked
 * vertically, and the panel filled the page with nothing. Observed 2026-09-18.
 *
 * So: SHOW FIRST, MEASURE SECOND, and never trust a measurement of something
 * that is not laid out. `_width()` refuses a zero and a ResizeObserver keeps
 * the plot matched to the panel afterwards.
 *
 * The series covers the LOADED WINDOW only, which is a real limit and is said
 * out loud rather than left to be discovered. Five years at hourly resolution
 * for one cell is not going into a browser; the long view comes from the
 * box-mean series instead.
 */

import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';

import { SWEEP_WIDTH_M, formatDistance } from './geo.js';
import { beaufort, describe, detectionOutlook } from './beaufort.js';
import {
  ALPHA, bearingFrom, bearingTowards, compass, currentBand, explain, leewayDistance,
  leewayFractionOfCurrent, leewaySpeed, summarise,
} from './drift.js';

const CHART_HEIGHT = 120;
const MIN_WIDTH = 240;

/** "33.00 N, 76.25 W" -- signed degrees are unreadable on a map. */
export function formatLatLon(lat, lon) {
  const ns = lat >= 0 ? 'N' : 'S';
  const ew = lon >= 0 ? 'E' : 'W';
  return `${Math.abs(lat).toFixed(2)} ${ns}, ${Math.abs(lon).toFixed(2)} ${ew}`;
}

export class PointPanel {
  /** @param {{root, chart, onClose}} els */
  constructor(els) {
    this.root = els.root;
    this.chartEl = els.chart;
    this.plot = null;
    this._onClose = els.onClose ?? (() => {});

    this.root.querySelector('#point-close').addEventListener('click', () => this.hide());
    // Esc closes it. The previous panel had no way out at all short of
    // reloading the page.
    this._keydown = (e) => { if (e.key === 'Escape' && this.isOpen) this.hide(); };
    document.addEventListener('keydown', this._keydown);

    if (typeof ResizeObserver !== 'undefined') {
      this._ro = new ResizeObserver(() => {
        if (this.plot && this.isOpen) this.plot.setSize({ width: this._width(), height: CHART_HEIGHT });
      });
      this._ro.observe(this.chartEl);
    }
  }

  get isOpen() {
    return this.root.classList.contains('visible');
  }

  hide() {
    this.root.classList.remove('visible');
    this._onClose();
  }

  /** Never returns 0: a zero-width plot is the bug described at the top. */
  _width() {
    return Math.max(MIN_WIDTH, this.chartEl.clientWidth || this.root.clientWidth - 16);
  }

  /**
   * @param {object} p
   *   lat, lon      the cell's centre
   *   u, v          components at the current frame, m/s
   *   series        speed per frame across the loaded window (NaN where absent)
   *   axis          { start, stepSeconds, frames }
   *   cursor        frame index the clock is on
   *   when          label for the current timestamp
   *   currentSpeed  current to compare leeway against, m/s -- the MEASURED
   *                 value at this cell when the current archive is loaded,
   *                 otherwise the Gulf Stream typical of 1.8
   *   currentAt     { u, v, measured } at this cell, or null if not loaded
   */
  show(p) {
    // Visible BEFORE anything is measured. Everything below depends on this.
    this.root.classList.add('visible');

    this.root.querySelector('#point-where').textContent = formatLatLon(p.lat, p.lon);
    this.root.querySelector('#point-when').textContent = p.when;

    const speed = Math.hypot(p.u, p.v);
    const from = bearingFrom(p.u, p.v);
    const towards = bearingTowards(p.u, p.v);
    const stats = summarise(p.series);

    const set = (id, text) => { this.root.querySelector(id).textContent = text; };
    const unit = (id, value, u) => {
      this.root.querySelector(id).innerHTML =
        `${value}<span class="unit">${u}</span>`;
    };

    const b = beaufort(speed);
    const outlook = detectionOutlook(speed);
    const story = explain(speed, towards, p.currentSpeed, SWEEP_WIDTH_M);

    unit('#t-speed', speed.toFixed(1), 'm/s');
    set('#t-force', describe(speed));
    set('#t-dir', `${from.toFixed(0)}° ${compass(from)}`);
    set('#t-towards', `towards ${compass(towards)}`);
    unit('#t-mean', stats.mean === null ? '—' : stats.mean.toFixed(1), 'm/s');
    set('#t-range', stats.n === 0 ? 'nothing loaded'
      : `${stats.min.toFixed(1)}–${stats.max.toFixed(1)} over ${stats.n} frames`);

    // The sentence, not the table, is what a reader takes away. See
    // drift.js::explain on why it names whose contribution it is describing.
    set('#d-lead', story.lead);
    set('#d-caveat', story.caveat);
    const badge = this.root.querySelector('#d-outlook');
    badge.className = `outlook ${outlook.level}`;
    badge.textContent = outlook.text;

    // The leeway term, and only the leeway term. See drift.js on why the other
    // two terms of D002's model are named as pending rather than left out.
    const lee = leewaySpeed(speed);
    const frac = leewayFractionOfCurrent(speed, p.currentSpeed);
    set('#d-sea', `${b.sea} · about ${b.waveM} m`);
    set('#d-leeway', `${lee.toFixed(3)} m/s`);
    set('#d-6h', formatDistance(leewayDistance(speed, 6)));
    set('#d-24h', formatDistance(leewayDistance(speed, 24)));
    set('#d-set', `${towards.toFixed(0)}° ${compass(towards)}`);
    set('#d-vscurrent', frac === null ? '—' : `${(frac * 100).toFixed(1)} %`);

    /*
      THE OCEAN SECTION.

      The panel was entirely about wind, because for a week wind was all there
      was. It now has a measured current at the same cell and can answer the
      question the project is actually about: where does this person go, and
      how big is the box you would have to search to find them.

      Every row here is arithmetic on two vectors we already hold. None of it
      is a model run -- the stochastic term is the engine's and is named as
      pending rather than estimated, because a spread is exactly the number
      somebody would quote.
    */
    const cur = p.currentAt;
    const hasCurrent = Boolean(cur) && Number.isFinite(cur.u) && Number.isFinite(cur.v);
    const oceanSection = this.root.querySelector('#ocean-section');
    if (oceanSection) {
      if (!hasCurrent) {
        unit('#o-speed', '—', '');
        set('#o-band', cur === null ? 'current layer is off' : 'land, or not loaded');
        set('#o-dir', '—');
        set('#o-carry', '—');
        set('#o-lead', cur === null
          ? 'Switch on a current layer and click again to see what the water does here.'
          : 'No current value at this cell — HYCOM has land or no data here.');
        for (const id of ['#o-6h', '#o-24h', '#o-ratio', '#o-resultant', '#o-area']) set(id, '—');
      } else {
        const cs = Math.hypot(cur.u, cur.v);
        const cTowards = bearingTowards(cur.u, cur.v);

        // The resultant is the first two terms of D002, computed the same way
        // ResultantSource computes them, so the panel and the arrow on the map
        // cannot disagree about this cell.
        const ru = cur.u + ALPHA * p.u;
        const rv = cur.v + ALPHA * p.v;
        const rs = Math.hypot(ru, rv);
        const rTowards = bearingTowards(ru, rv);

        const km24 = (rs * 86400) / 1000;
        // A datum displaced this far, with no spread term, still has to be
        // searched as an AREA rather than a point -- this is the lower bound
        // on that area, and it is a lower bound precisely because eta is
        // missing. Circle of radius = one hour of resultant travel.
        const radiusKm = (rs * 3600) / 1000;
        const areaKm2 = Math.PI * radiusKm * radiusKm;

        unit('#o-speed', cs.toFixed(2), 'm/s');
        set('#o-band', currentBand(cs));
        set('#o-dir', `${cTowards.toFixed(0)}° ${compass(cTowards)}`);
        set('#o-carry', `${formatDistance(cs * 86400)} in a day`);

        set('#o-6h', formatDistance(cs * 6 * 3600));
        set('#o-24h', formatDistance(cs * 24 * 3600));

        const lee2 = leewaySpeed(speed);
        set('#o-ratio', lee2 > 0 ? `${(cs / lee2).toFixed(1)} : 1` : '—');
        set('#o-resultant', `${rs.toFixed(2)} m/s towards ${rTowards.toFixed(0)}° ${compass(rTowards)}`);
        set('#o-area', `≥ ${areaKm2 < 10 ? areaKm2.toFixed(1) : Math.round(areaKm2)} km² · ${km24.toFixed(0)} km downstream`);

        // The sentence, because a column of numbers is not an argument.
        const dominant = cs > lee2 * 2 ? 'the water'
          : (lee2 > cs * 2 ? 'the wind' : 'neither');
        set('#o-lead', dominant === 'the water'
          ? `The current dominates here: ${(cs / lee2).toFixed(1)}× the leeway, so the datum follows the water and a wind-only estimate would send searchers to the wrong place.`
          : (dominant === 'the wind'
            ? `Unusually, leeway is the larger term here — ${(lee2 / cs).toFixed(1)}× the current — so a drifter tracks the weather more than the sea.`
            : 'Wind and water are comparable here, so the two terms must be added as vectors rather than ranked. This is the regime that earns the leeway term its place in the model.'));
      }
    }

    /*
      The surface current row was hard-coded to "awaiting HYCOM pull" in the
      markup, which stopped being true the moment the archive was published.
      A pending label that outlives the thing it was waiting for is worse than
      no label: it tells the reader the system is less finished than it is.

      Still labelled pending when there is genuinely nothing loaded, because
      the current layer is off by default and a blank row would be ambiguous.
    */
    const curRow = this.root.querySelector('#d-current');
    if (curRow) {
      const row = curRow.closest('.row');
      if (cur && Number.isFinite(cur.u) && Number.isFinite(cur.v)) {
        const cs = Math.hypot(cur.u, cur.v);
        set('#d-current', `${cs.toFixed(2)} m/s towards ${compass(bearingTowards(cur.u, cur.v))}`);
        if (row) row.classList.remove('pending');
      } else {
        set('#d-current', cur === null ? 'switch on the current layer' : 'land or no data here');
        if (row) row.classList.add('pending');
      }
    }

    this._drawSeries(p.series, p.axis, p.cursor);
  }

  _drawSeries(series, axis, cursor) {
    const xs = new Array(series.length);
    for (let f = 0; f < series.length; f += 1) {
      xs[f] = axis.start.getTime() / 1000 + f * axis.stepSeconds;
    }
    const data = [xs, Array.from(series, (x) => (Number.isFinite(x) ? x : null))];

    if (!this.plot) {
      this.plot = new uPlot(
        {
          width: this._width(),
          height: CHART_HEIGHT,
          padding: [8, 8, 0, 0],
          scales: { x: { time: true } },
          axes: [
            { stroke: '#8a8a80', grid: { stroke: '#2b2b28' }, ticks: { stroke: '#2b2b28' } },
            {
              stroke: '#8a8a80', grid: { stroke: '#2b2b28' }, ticks: { stroke: '#2b2b28' },
              size: 38,
            },
          ],
          series: [
            { label: 'time' },
            {
              label: 'm/s',
              stroke: '#eb6834',
              width: 2,
              fill: 'rgba(235, 104, 52, 0.14)',
              // A gap is drawn as a gap. Joining across unloaded frames would
              // invent weather that was never fetched.
              spanGaps: false,
              points: { show: false },
            },
          ],
          cursor: { show: true, y: false },
          legend: { live: true },
        },
        data,
        this.chartEl,
      );
    } else {
      this.plot.setSize({ width: this._width(), height: CHART_HEIGHT });
      this.plot.setData(data);
    }

    // Park the cursor on the frame the map is showing, so the chart and the
    // map always agree about which moment is on screen.
    const at = this.plot.valToPos(xs[Math.min(cursor, xs.length - 1)], 'x');
    if (Number.isFinite(at)) this.plot.setCursor({ left: at, top: 0 });
  }
}
