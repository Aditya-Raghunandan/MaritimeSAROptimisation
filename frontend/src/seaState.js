/**
 * seaState.js -- what the compass says and how rough the drawn sea is (issue #73).
 *
 * Pure: no Leaflet, no DOM, so the words the compass shows and the whitecap rule the
 * close-up sea follows are tested in tests/seaState.test.js. compass.js and oceanLayer.js
 * only draw what these return.
 */

import { describe as beaufortName, beaufort } from './beaufort.js';
import { bearingFrom, bearingTowards, compass as point, speed } from './drift.js';

/** Addendum Table H-10: above 15 kt of wind the PIW sweep width is halved. */
const WEATHER_FACTOR_MS = 15 * (1852 / 3600);

/**
 * What the compass says, from the helicopter's heading and the current and wind at it.
 * `current` and `wind` are [u, v] m/s, or null while the forcing is loading or absent.
 */
export function compassReadout({ heading = null, current = null, wind = null } = {}) {
  const out = { heading: null, current: null, wind: null, lines: [], caveat: null };
  if (Number.isFinite(heading)) {
    const h = ((heading % 360) + 360) % 360;
    out.heading = h;
    out.lines.push(`Heading ${String(Math.round(h)).padStart(3, '0')}° ${point(h)}`);
  } else {
    out.lines.push('Heading —');
  }
  if (current && current.every(Number.isFinite)) {
    const s = speed(current[0], current[1]);
    const to = bearingTowards(current[0], current[1]);
    out.current = { speed: s, towards: to };
    out.lines.push(`Current ${s.toFixed(2)} m/s towards ${point(to)}`);
  } else {
    out.lines.push('Current —');
  }
  if (wind && wind.every(Number.isFinite)) {
    const s = speed(wind[0], wind[1]);
    out.wind = { speed: s, towards: bearingTowards(wind[0], wind[1]), from: bearingFrom(wind[0], wind[1]), force: beaufort(s).force };
    out.lines.push(`Wind ${s.toFixed(1)} m/s from ${point(out.wind.from)} · ${beaufortName(s)}`);
    if (s > WEATHER_FACTOR_MS) {
      out.caveat = 'Over 15 kt: the Coast Guard halves the sweep width here (Table H-10). '
        + 'Not applied in this view.';
    }
  } else {
    out.lines.push('Wind —');
  }
  return out;
}

/** How many whitecaps are live at once, per Beaufort force (WMO sea-state wording). */
export function whitecapCount(force) {
  if (force <= 2) return 0;
  return [0, 0, 0, 6, 16, 30, 48, 70][Math.min(force, 7)] ?? 70;
}
