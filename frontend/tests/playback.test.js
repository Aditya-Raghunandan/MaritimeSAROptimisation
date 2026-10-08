import { describe, expect, it } from 'vitest';
import golden from '../src/fixtures/referee_golden.json';
import { PlayerFlight, RecordedFlight, SAMPLE_S, offsetWithin } from '../src/playback.js';

function window() {
  const flights = {};
  for (const [name, f] of Object.entries(golden.flights)) {
    flights[name] = {
      steps: f.steps.map((s) => ({ tS: s.t_s, eastM: s.east_m, northM: s.north_m })),
      python: { pos: f.pos },
    };
  }
  return {
    cloud: {
      frames: golden.window.frames,
      particles: golden.window.particles,
      lat: Float64Array.from(golden.window.lat),
      lon: Float64Array.from(golden.window.lon),
      weight: golden.window.weight,
    },
    marker: { tS: golden.marker.t_s, lat: golden.marker.lat, lon: golden.marker.lon },
    target: { tS: golden.target.t_s, lat: golden.target.lat, lon: golden.target.lon },
    flights,
  };
}

describe('a recorded flight', () => {
  it('flies only the minutes that have finished', () => {
    const f = new RecordedFlight(window(), 'expanding-square');
    f.advanceTo(59.9);
    expect(f.ep.k).toBe(0);
    f.advanceTo(125);
    expect(f.ep.k).toBe(2);
  });

  it('ends with the number the paper gives', () => {
    const f = new RecordedFlight(window(), 'parallel');
    f.advanceTo(45 * 60);
    expect(f.done).toBe(true);
    expect(f.metrics().pos).toBeCloseTo(golden.flights.parallel.pos, 12);
  });

  it('is drawn on its path between waypoints', () => {
    expect(offsetWithin([0, 0], { tS: [30, 60], eastM: [100, 100], northM: [0, 200] }, 15))
      .toEqual([50, 0]);
    expect(offsetWithin([0, 0], { tS: [30, 60], eastM: [100, 100], northM: [0, 200] }, 45))
      .toEqual([100, 100]);
  });
});

describe('a player', () => {
  it('is scored by the browser exactly as Python scores the same headings', () => {
    const { record } = golden.player;
    const f = new PlayerFlight(window());
    for (let i = 0; i < record.t_s.length; i += 1) {
      f.advanceTo(record.t_s[i]);
      f.steer(record.heading_deg[i]);
    }
    f.advanceTo(45 * 60);
    const m = f.metrics();
    m.drainRate.forEach((d, k) => expect(d).toBeCloseTo(golden.player.removed_per_step[k], 12));
    expect(m.target.found).toBe(golden.player.target.found);
    expect(f.record().t_s).toEqual(record.t_s);
    expect(f.record().heading_deg.map((h) => Math.round(h * 1e9) / 1e9))
      .toEqual(record.heading_deg.map((h) => Math.round(h * 1e9) / 1e9));
  });

  it('holds a heading for one leg, and flies a minute at a time', () => {
    const f = new PlayerFlight(window(), { heading: 90 });
    f.advanceTo(SAMPLE_S * 3);
    expect(f.headings.tS).toEqual([0, 5, 10]);
    expect(f.ep.k).toBe(0);
    f.advanceTo(60);
    expect(f.ep.k).toBe(1);
    expect(f.offset[0]).toBeCloseTo(f.ep.speedMs * 60, 6);
  });
});
