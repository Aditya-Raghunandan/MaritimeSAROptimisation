/**
 * seaState.js -- what the compass says and how rough the drawn sea is (issue #73).
 *
 * Pure: no Leaflet, no DOM, so the words the compass shows and the whitecap rule the
 * close-up sea follows are tested in tests/seaState.test.js. compass.js and oceanLayer.js
 * only draw what these return.
 */

import { describe as beaufortName, beaufort } from './beaufort.js';
import { toKnots } from './closeUp.js';
import { bearingFrom, bearingTowards, compass as point, speed } from './drift.js';

/** Addendum Table H-10: above 15 kt of wind the PIW sweep width is halved. */
const WEATHER_FACTOR_MS = 15 * (1852 / 3600);

/*
  THE ROUGH-SEA CAVEAT, IN PLAIN WORDS (#81). It used to read "Over 15 kt: the Coast Guard
  halves the sweep width here (Table H-10). Not applied in this view." -- knots beside a
  wind given in m/s, a term the page never explains, a table number, and no word on what
  "not applied" means for the search being watched. The rule is unchanged (limitation L19);
  the citation moves to the tooltip, and the wind line gives knots so the 15 kt can be seen.
*/
export const ROUGH_SEA = 'Rough sea: whitecaps hide a person in the water, so the Coast Guard '
  + 'assumes spotters see only half as far to each side. This view does not, so finding here is '
  + 'easier than it would really be.';
export const ROUGH_SEA_SOURCE = 'USCG Addendum (COMDTINST M16130.2F), Table H-10: in winds over '
  + '15 kt or seas over 3 ft, the sweep width for a person in the water is multiplied by 0.5; '
  + 'over 25 kt, by 0.25.';

/**
 * What the compass says, from the helicopter's heading and the current and wind at it.
 * `current` and `wind` are [u, v] m/s, or null while the forcing is loading or absent.
 */
export function compassReadout({ heading = null, current = null, wind = null } = {}) {
  const out = { heading: null, current: null, wind: null, lines: [], caveat: null, caveatSource: null };
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
    out.lines.push(`Wind ${s.toFixed(1)} m/s (${Math.round(toKnots(s))} kt) from ${point(out.wind.from)} · ${beaufortName(s)}`);
    if (s > WEATHER_FACTOR_MS) {
      out.caveat = ROUGH_SEA;
      out.caveatSource = ROUGH_SEA_SOURCE;
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
