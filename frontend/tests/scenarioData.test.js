import { describe, expect, it } from 'vitest';
import { SCENARIO_VERSIONS, aiOf, arrivalLabel, arrivalsOf, searchersOf } from '../src/scenarioData.js';

describe('which searchers a version carries', () => {
  it('lists the trained agent only when the index says the files have it', () => {
    const six = ['expanding-square', 'sector', 'parallel', 'trackline', 'greedy', 'random'];
    expect(searchersOf({ searchers: six })).not.toContain('ppo');
    expect(searchersOf({ searchers: [...six, 'ppo'] })).toContain('ppo');
  });

  it('leaves the agent out of an index that does not say', () => {
    expect(searchersOf({})).not.toContain('ppo');
    expect(searchersOf({})).toContain('greedy');
  });

  it('keeps the page order, whatever order the index lists them in', () => {
    expect(searchersOf({ searchers: ['random', 'ppo', 'greedy'] })).toEqual(['greedy', 'ppo', 'random']);
  });
});

describe("who plays the computer's side", () => {
  it('is the trained model where the window has its flight and the helicopter arrives by 45 min', () => {
    const both = { flights: { greedy: {}, ppo: {} } };
    expect(aiOf(both, 0.25)).toBe('ppo');
    expect(aiOf(both, 0.75)).toBe('ppo');
    expect(aiOf(both, 1)).toBe('greedy');
    expect(aiOf(both, 3)).toBe('greedy');
    expect(aiOf({ flights: { greedy: {} } }, 0.5)).toBe('greedy');
  });

  it('is Telemachus S, up to 1 h, where the window carries it', () => {
    const all = { flights: { greedy: {}, ppo: {}, telemachus: {} } };
    expect(aiOf(all, 0.25)).toBe('telemachus');
    expect(aiOf(all, 1)).toBe('telemachus');
    expect(aiOf(all, 2)).toBe('greedy');
  });

  it('looks for v5 first', () => {
    expect(SCENARIO_VERSIONS[0]).toBe('v5');
  });
});

describe('arrival times', () => {
  it('are read from the window names for one drift model, soonest first', () => {
    const s = { windows: ['rvc_2h', 'rvc_0.25h', 'rv_1h', 'rvc_0.5h', 'rvc_3h'] };
    expect(arrivalsOf(s, 'rvc')).toEqual([0.25, 0.5, 2, 3]);
    expect(arrivalsOf(s, 'rv')).toEqual([1]);
    expect(arrivalsOf({}, 'rvc')).toEqual([]);
  });

  it('read as people say them', () => {
    expect([0.25, 0.5, 0.75, 1, 2].map(arrivalLabel)).toEqual(['15 min', '30 min', '45 min', '1 h', '2 h']);
  });
});
