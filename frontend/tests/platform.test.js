/**
 * The site's search constants, held to Python's (issue #44).
 *
 * `sar.search.platform` is the one home for the sweep width (ADR002, ADR003).
 * `fixtures/search_golden.json` is written from it by
 * `scripts/export_search_golden.py`, so if either side changes the sweep width
 * alone, this fails and names it.
 */

import { describe, expect, it } from 'vitest';

import golden from '../src/fixtures/search_golden.json';
import { SWEEP_WIDTH_M } from '../src/geo.js';

describe('platform constants', () => {
  it('quotes the same sweep width as Python', () => {
    expect(SWEEP_WIDTH_M).toBeCloseTo(golden.platform.sweep_width_m, 9);
  });

  it('is a tenth of a nautical mile', () => {
    expect(SWEEP_WIDTH_M).toBeCloseTo(0.1 * golden.platform.nm_m, 9);
  });

  it('turns a 45-minute window into 45 steps', () => {
    expect(golden.platform.on_scene_window_s / golden.platform.step_s).toBe(45);
  });
});
