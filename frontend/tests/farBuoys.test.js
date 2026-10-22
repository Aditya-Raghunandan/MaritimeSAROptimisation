import { expect, it } from 'vitest';
import { FAR_WINDOWS, playable } from '../src/farBuoys.js';
import { arrivalsOf } from '../src/scenarioData.js';

const all = ['rvc_0.25h', 'rvc_0.5h', 'rvc_0.75h', 'rvc_1h', 'rvc_2h', 'rvc_3h'];
const index = { format: 2, scenarios: [{ scenario: 'S09', windows: all }, { scenario: 'S10', windows: all }] };

it('offers S10 only at 15 min, where its buoy stays by the cloud, and leaves the others alone', () => {
  const p = playable(index);
  expect(arrivalsOf(p.scenarios.find((s) => s.scenario === 'S10'), 'rvc')).toEqual([0.25]);
  expect(p.scenarios.find((s) => s.scenario === 'S09').windows).toEqual(all);
  expect(p.format).toBe(2);
  expect(index.scenarios[1].windows).toEqual(all);                   // the index itself is untouched
});

it('drops a buoy with no window left', () => {
  const p = playable(index, { ...FAR_WINDOWS, S09: all });
  expect(p.scenarios.map((s) => s.scenario)).toEqual(['S10']);
});

it('on v6 leaves out S10 from 30 min, all of S12 and S02 from 2 h', () => {
  const v6 = ['rvsc_0.25h', 'rvsc_0.5h', 'rvsc_0.75h', 'rvsc_1h', 'rvsc_2h', 'rvsc_3h'];
  const p = playable({ scenarios: ['S02', 'S10', 'S12', 'S13'].map((scenario) => ({ scenario, windows: v6 })) });
  expect(p.scenarios.map((s) => s.scenario)).toEqual(['S02', 'S10', 'S13']);
  expect(arrivalsOf(p.scenarios.find((s) => s.scenario === 'S10'), 'rvsc')).toEqual([0.25]);
  expect(arrivalsOf(p.scenarios.find((s) => s.scenario === 'S02'), 'rvsc')).toEqual([0.25, 0.5, 0.75, 1]);
  expect(p.scenarios.find((s) => s.scenario === 'S13').windows).toEqual(v6);
});
