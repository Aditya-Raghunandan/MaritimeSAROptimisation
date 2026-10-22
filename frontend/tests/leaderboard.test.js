import { describe, expect, it } from 'vitest';
import {
  GAME_AREA_KM2, OLD_STORAGE_KEY, STORAGE_KEY, add, cloudAreaKm2, load, nickname, par, rescore, score, standing, top,
  verdict,
} from '../src/leaderboard.js';

function memory() {
  const store = new Map();
  return {
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, String(v)),
  };
}

describe('the score', () => {
  it('is 1,000 for what the Coast Guard gets on a cloud that size', () => {
    // Two published windows (rvc, 2 h): S32, 23.4 km², and S11, 148.8 km².
    expect(par(23.4)).toBeCloseTo(0.69, 2);
    expect(par(148.8)).toBeCloseTo(0.16, 2);
    expect(score(par(60), 60)).toBe(1000);
  });

  it('holds a cloud outside the fitted sizes to the nearest end', () => {
    expect(par(5)).toBe(par(22.4));
    expect(par(900)).toBe(par(148.8));
    expect(par(5)).toBeLessThan(1);
  });

  it('gives the same share more points on a bigger cloud', () => {
    expect(score(0.3, 140)).toBeGreaterThan(score(0.3, 25));
    // The Square's own POS on S32 and S11, 0.678 and 0.139, score about alike.
    expect(Math.abs(score(0.678, 23.4) - score(0.139, 148.8))).toBeLessThan(150);
  });

  it('measures the cloud as the smallest area holding 90 %', () => {
    // Ten cells: one with half the probability, four with a tenth each, five with 0.02.
    const mass = [0.5, 0.1, 0.1, 0.1, 0.1, 0.02, 0.02, 0.02, 0.02, 0.02];
    expect(cloudAreaKm2(mass, 250)).toBeCloseTo(5 * 0.0625, 10);
    expect(cloudAreaKm2(mass, 250, 0.5)).toBeCloseTo(0.0625, 10);
    expect(cloudAreaKm2([2, 2], 1000)).toBe(2);
  });

  it('wins only by beating every rival on POS', () => {
    const rivals = [{ name: 'Coast Guard', pos: 0.53 }, { name: 'AI', pos: 0.49 }];
    expect(verdict({ pos: 0.6 }, rivals)).toEqual({ won: true, beaten: ['Coast Guard', 'AI'] });
    expect(verdict({ pos: 0.5 }, rivals)).toEqual({ won: false, beaten: ['AI'] });
  });
});

describe('the board from before 12 Oct', () => {
  it('comes over once, rescored by cloud size from its own POS and buoy', () => {
    const s = memory();
    // A small-cloud buoy used to score more for less: S32 is 23.4 km², S11 148.8 km².
    s.setItem(OLD_STORAGE_KEY, JSON.stringify([
      { name: 'Aditya', mode: 'map', scenario: 'S32', pos: 0.30, score: 300, when: '1' },
      { name: 'Her', mode: 'map', scenario: 'S11', pos: 0.12, score: 620, when: '2' },
      { name: 'Lost', mode: 'map', scenario: 'S99', pos: 0.5, score: 500, when: '3' },
    ]));
    const list = load(s);
    expect(list.map((e) => e.name)).toEqual(['Her', 'Aditya']);
    expect(list[0].score).toBe(score(0.12, 148.8));
    expect(list[0].old_score).toBe(620);
    expect(list.every((e) => e.rescored)).toBe(true);
    // Saved on the new board, so it is not rescored again.
    expect(JSON.parse(s.getItem(STORAGE_KEY)).length).toBe(2);
    add({ name: 'new', mode: 'map', score: 1, when: '4' }, s);
    expect(load(s).length).toBe(3);
  });

  it('knows every one of the 55 buoys', () => {
    expect(Object.keys(GAME_AREA_KM2)).toHaveLength(55);
    expect(rescore([{ scenario: 'S01', pos: 0.2, when: '1' }])[0].area_km2).toBe(111.4);
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

  it('shows one line per name, with that name\'s highest score', () => {
    const list = [
      { name: 'Aditya', mode: 'map', score: 406, when: '2026-10-09T20:00:00Z' },
      { name: 'aditya ', mode: 'map', score: 437, when: '2026-10-09T20:05:00Z' },
      { name: 'Sune', mode: 'map', score: 420, when: '2026-10-09T20:06:00Z' },
      { name: 'Aditya', mode: 'currents', score: 300, when: '2026-10-09T20:07:00Z' },
    ];
    expect(top(list, 'map').map((e) => [e.name, e.score])).toEqual([['aditya ', 437], ['Sune', 420]]);
    expect(top(list, 'currents')).toHaveLength(1);
  });

  it('says where a player stands, for the email', () => {
    const list = [
      { name: 'Ada', mode: 'map', score: 1200, when: '1' },
      { name: 'Bo', mode: 'map', score: 900, when: '2' },
      { name: 'ada', mode: 'map', score: 1000, when: '3' },
      { name: 'Cy', mode: 'map', score: 800, when: '4' },
      { name: 'Di', mode: 'currents', score: 2000, when: '5' },
    ];
    const first = standing(list, 'map', 'ADA ');
    expect([first.rank, first.of]).toEqual([1, 3]);
    expect(first.lines.map((l) => [l.rank, l.name, l.score, l.you]))
      .toEqual([[1, 'Ada', 1200, true], [2, 'Bo', 900, false], [3, 'Cy', 800, false]]);
    const low = standing(list, 'map', 'Cy', 2);
    expect(low.rank).toBe(3);
    expect(low.lines.map((l) => [l.rank, l.you])).toEqual([[1, false], [2, false], [3, true]]);
    expect(standing(list, 'currents', 'Di').of).toBe(1);
    expect(standing([], 'map', 'Nobody')).toEqual({ rank: 0, of: 0, lines: [] });
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
