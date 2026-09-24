/**
 * The site's search constants, held to Python's (issues #44, #64).
 *
 * `sar.search.platform` is the one home for these numbers (ADR002, ADR003).
 * `fixtures/search_golden.json` is written from it by
 * `scripts/export_search_golden.py`, so if either side changes one alone, this
 * fails and names it.
 */

import { describe, expect, it } from 'vitest';

import golden from '../src/fixtures/search_golden.json';
import { SWEEP_WIDTH_M } from '../src/geo.js';
import {
  KNOT_MS, LAUNCH_DELAY_S, MAX_ENDURANCE_S, NM_M, ON_SCENE_WINDOW_S, RADIUS_OF_ACTION_M,
  SEARCH_SPEED_MS, STEP_S, TRANSIT_SPEED_MS, searchEffortM2, sectorRadiusM, transitTimeS,
} from '../src/platform.js';

const P = golden.platform;

describe('platform constants', () => {
  it('quotes the same sweep width as Python', () => {
    expect(SWEEP_WIDTH_M).toBeCloseTo(P.sweep_width_m, 9);
  });

  it('is a tenth of a nautical mile', () => {
    expect(SWEEP_WIDTH_M).toBeCloseTo(0.1 * P.nm_m, 9);
  });

  it('agrees with Python on every other number', () => {
    const js = {
      nm_m: NM_M,
      knot_ms: KNOT_MS,
      search_speed_ms: SEARCH_SPEED_MS,
      transit_speed_ms: TRANSIT_SPEED_MS,
      radius_of_action_m: RADIUS_OF_ACTION_M,
      max_endurance_s: MAX_ENDURANCE_S,
      launch_delay_s: LAUNCH_DELAY_S,
      on_scene_window_s: ON_SCENE_WINDOW_S,
      step_s: STEP_S,
    };
    for (const [name, value] of Object.entries(js)) {
      expect(value, name).toBeCloseTo(P[name], 9);
    }
  });

  it('agrees with Python on the search effort, Z = W x V x T', () => {
    expect(searchEffortM2()).toBeCloseTo(P.search_effort_m2, 6);
  });

  it('turns a 45-minute window into 45 steps', () => {
    expect(P.on_scene_window_s / P.step_s).toBe(45);
  });
});

describe('derived sizes', () => {
  it('flies a sector radius of 1.5 NM at 90 kt, the Addendum\'s example', () => {
    expect(sectorRadiusM()).toBeCloseTo(1.5 * NM_M, 9);
  });

  it('adds the launch delay to the flight out, and refuses beyond 300 NM', () => {
    expect(transitTimeS(100 * NM_M)).toBeCloseTo(1800 + 2880, 6);
    expect(transitTimeS(RADIUS_OF_ACTION_M + 1)).toBeNull();
    expect(transitTimeS(-1)).toBeNull();
  });
});
