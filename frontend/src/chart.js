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
  bearingFrom, bearingTowards, compass, explain, leewayDistance,
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
   *   currentSpeed  typical current to compare leeway against, m/s
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
