import { describe, expect, it } from 'vitest';
import { FOUND_BONUS, add, load, nickname, score, top, verdict } from '../src/leaderboard.js';

function memory() {
  const store = new Map();
  return {
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, String(v)),
  };
}

describe('the score', () => {
  it('is mostly POS, with a bonus for finding the buoy', () => {
    expect(score(0.531, false)).toBe(531);
    expect(score(0.531, true)).toBe(531 + FOUND_BONUS);
  });

  it('wins only by beating every rival on POS', () => {
    const rivals = [{ name: 'Coast Guard', pos: 0.53 }, { name: 'AI', pos: 0.49 }];
    expect(verdict({ pos: 0.6 }, rivals)).toEqual({ won: true, beaten: ['Coast Guard', 'AI'] });
    expect(verdict({ pos: 0.5 }, rivals)).toEqual({ won: false, beaten: ['AI'] });
  });
});

describe('the board', () => {
  it('keeps the best first, per mode', () => {
    const s = memory();
    add({ name: 'a', mode: 'map', score: 300, when: '1' }, s);
    add({ name: 'b', mode: 'map', score: 700, when: '2' }, s);
    add({ name: 'c', mode: 'currents', score: 900, when: '3' }, s);
    expect(top(load(s), 'map').map((e) => e.name)).toEqual(['b', 'a']);
    expect(top(load(s), 'currents').map((e) => e.name)).toEqual(['c']);
  });

  it('survives storage that refuses or holds rubbish', () => {
    const broken = { getItem: () => '{not json', setItem: () => { throw new Error('quota'); } };
    expect(load(broken)).toEqual([]);
    expect(add({ name: 'a', mode: 'map', score: 1, when: '1' }, broken).saved).toBe(false);
    expect(load(undefined)).toEqual([]);
  });

  it('cleans nicknames for a public screen', () => {
    expect(nickname('  Zanele  ')).toBe('Zanele');
    expect(nickname('<b>Zan</b>')).toBe('bZanb');
    expect(nickname('')).toBe('Player');
    expect(nickname('a'.repeat(40))).toHaveLength(16);
    expect(nickname('<script>')).toBe('script');
  });
});
