import { describe, expect, it } from 'vitest';
import { SCENARIO_VERSIONS, aiOf, searchersOf } from '../src/scenarioData.js';

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

describe('who plays the AI', () => {
  it('is the trained agent where the window has its flight, else greedy', () => {
    expect(aiOf({ flights: { greedy: {}, ppo: {} } })).toBe('ppo');
    expect(aiOf({ flights: { greedy: {} } })).toBe('greedy');
  });

  it('looks for v3 first', () => {
    expect(SCENARIO_VERSIONS[0]).toBe('v3');
  });
});
