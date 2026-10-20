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
