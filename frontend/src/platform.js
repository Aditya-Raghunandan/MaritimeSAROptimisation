/**
 * platform.js -- the search helicopter's numbers, as the site uses them (issue #64).
 *
 * The reference is `src/sar/search/platform.py`, where each number is cited to the
 * USCG Addendum (COMDTINST M16130.2F) by table and page -- see docs/ADR003.md. These
 * are copies, and `tests/platform.test.js` holds every one to Python through
 * `fixtures/search_golden.json`, so neither side can change a number alone.
 *
 * Pure arithmetic, no imports beyond geo.js, so it is testable without a DOM.
 */

import { SWEEP_WIDTH_M, distance } from './geo.js';

export { SWEEP_WIDTH_M };

/** The international nautical mile, and the knot it defines. */
export const NM_M = 1852;
export const KNOT_MS = NM_M / 3600;

/** 90 kt: the speed the Addendum's helicopter sweep widths are stated at (Table H-9). */
export const SEARCH_SPEED_MS = 90 * KNOT_MS;

/** H-60 cruise, radius of action and endurance (Addendum Table 5-3, p. 5-17). */
export const TRANSIT_SPEED_MS = 125 * KNOT_MS;
export const RADIUS_OF_ACTION_M = 300 * NM_M;
export const MAX_ENDURANCE_S = 6 * 3600;

/** B-0 readiness: airborne within 30 minutes of the call (Addendum p. PPO-7). */
export const LAUNCH_DELAY_S = 30 * 60;

/** A project requirement (R7a/R7b), not a published figure. */
export const ON_SCENE_WINDOW_S = 45 * 60;

/** The integration step and the agent's decision interval (D009, ADR002). */
export const STEP_S = 60;

/** One minute at search speed or twice the sweep width, whichever is larger (p. 3-23). */
export function sectorRadiusM(speedMs = SEARCH_SPEED_MS, sweepWidthM = SWEEP_WIDTH_M) {
  return Math.max(speedMs * 60, 2 * sweepWidthM);
}

/**
 * Seconds from the call to arriving on scene: the launch delay, then the flight out.
 * Null beyond the radius of action -- a search the helicopter could not fly is not one
 * to draw -- so the caller can say why rather than catching.
 */
export function transitTimeS(distanceM, speedMs = TRANSIT_SPEED_MS, launchDelayS = LAUNCH_DELAY_S) {
  if (!Number.isFinite(distanceM) || distanceM < 0 || distanceM > RADIUS_OF_ACTION_M) return null;
  return launchDelayS + distanceM / speedMs;
}

/** Great-circle metres between two {lat, lon} points (geo.js takes {lat, lng}). */
export function distanceM(a, b) {
  return distance({ lat: a.lat, lng: a.lon }, { lat: b.lat, lng: b.lon });
}
