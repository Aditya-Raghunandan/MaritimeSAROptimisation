/**
 * chart.js -- the time series for a clicked cell.
 *
 * uPlot rather than Chart.js: about 40 KB, built for exactly this, and it
 * redraws fast enough to follow the time slider without stuttering.
 *
 * The series covers the LOADED WINDOW only, which is a real limit and is
 * labelled on the chart rather than left to be discovered. Five years at hourly
 * resolution per cell is 2.03 GB and is not going into a browser; the long view
 * comes from the box-mean series, which is one number per hour for the whole
 * box and about 2 MB for five years.
 */

import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';

export class PointChart {
  constructor(el) {
    this.el = el;
    this.plot = null;
  }

  /**
   * @param {Float32Array} series  one value per frame of the layer's axis
   * @param {{start: Date, stepSeconds: number, frames: number}} axis
   * @param {number} cursor        frame the clock is currently on
   */
  show(series, axis, cursor) {
    const xs = new Array(series.length);
    for (let f = 0; f < series.length; f += 1) {
      xs[f] = axis.start.getTime() / 1000 + f * axis.stepSeconds;
    }
    const data = [xs, Array.from(series)];

    if (!this.plot) {
      this.plot = new uPlot(
        {
          width: this.el.clientWidth,
          height: 150,
          title: 'wind speed at the clicked cell (loaded window only)',
          scales: { x: { time: true } },
          axes: [{}, { label: 'm/s' }],
          series: [
            {},
            { label: 'speed', stroke: '#1f77b4', width: 2, fill: 'rgba(31,119,180,0.15)' },
          ],
          cursor: { show: true },
        },
        data,
        this.el,
      );
    } else {
      this.plot.setData(data);
    }

    // Park the cursor on the frame the map is showing, so the chart and the map
    // always agree about which moment is on screen.
    this.plot.setCursor({ left: this.plot.valToPos(xs[cursor], 'x'), top: 0 });
    this.el.classList.add('visible');
  }
}
