/**
 * scenarioData.js -- the published scenario bundles, loaded for the browser (D031, Stage 2).
 *
 * scripts/export_scenario_bundles.py writes, for every scenario x noise model x arrival
 * (a "window"), the cloud through the 45-minute search, every benchmark searcher's flight
 * and the paper's numbers for it, the real buoy, and a current and wind grid. This turns
 * those files into what referee.js and the layers take: tracks as {tS, lat, lon}, the cloud
 * as frame-major Float64Arrays, weights as a Float64Array.
 *
 * Longitudes stay in the store convention (0 to 360) for the referee, which does its
 * arithmetic exactly as Python does; `display` converts one for Leaflet.
 *
 * Where the files live: VITE_DATA_BASE/scenarios/<version> (Hugging Face in the deployed
 * site, ./data/scenarios/<version> under public/ when developing, or a local copy for the
 * offline open-day build).
 *
 * TWO VERSIONS. v2 (format 2) was flown by the D032 helicopter, which turns at most 7.0 deg/s
 * and carries the heading at every waypoint; v1 (format 1) by the old one that turned at
 * once. The newest version that is published is used, so publishing v2 switches the site
 * over without a deploy; each window says which helicopter flew it (`turnRateDegS`).
 */

import { cloudFromOffsets } from './referee.js';

export const DATA_BASE = String(import.meta.env.VITE_DATA_BASE ?? './data').replace(/\/$/, '');
/** Newest first. */
export const SCENARIO_VERSIONS = ['v2', 'v1'];
export const SCENARIO_ROOT = `${DATA_BASE}/scenarios/${SCENARIO_VERSIONS[SCENARIO_VERSIONS.length - 1]}`;
/** The 3D loops, which do not change with the helicopter. */
export const SHOWCASE_ROOT = `${DATA_BASE}/scenarios/showcase`;

/** The searchers in the order every page lists them, with words for people. */
export const SEARCHER_INFO = {
  'expanding-square': { label: 'Expanding Square', kind: 'Coast Guard pattern', colour: '#3987e5' },
  sector: { label: 'Sector', kind: 'Coast Guard pattern', colour: '#a77bf2' },
  parallel: { label: 'Parallel Track', kind: 'Coast Guard pattern', colour: '#e8b23d' },
  trackline: { label: 'Trackline', kind: 'Coast Guard pattern', colour: '#e8735a' },
  greedy: { label: 'Greedy (AI for now)', kind: 'reads the map', colour: '#04c951' },
  random: { label: 'Random', kind: 'the floor', colour: '#8a8a80' },
};

export const NOISE_INFO = {
  rv: { label: 'with memory', note: 'the calibrated drift model (errors persist about a day)' },
  rw: { label: 'without memory', note: 'the old model (a fresh random push every minute)' },
};

/** A store longitude (0 to 360) as Leaflet wants it (-180 to 180). */
export function display(lon) {
  return lon > 180 ? lon - 360 : lon;
}

const cache = new Map();

async function fetchOnce(url, kind) {
  if (!cache.has(url)) {
    cache.set(url, fetch(url).then((r) => {
      if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
      return kind === 'json' ? r.json() : r.arrayBuffer();
    }).catch((err) => {
      cache.delete(url);
      throw err;
    }));
  }
  return cache.get(url);
}

let resolved = null;

/** The newest published version's folder: the first whose index.json answers. */
export function scenarioRoot() {
  if (!resolved) {
    resolved = (async () => {
      for (const v of SCENARIO_VERSIONS.slice(0, -1)) {
        const root = `${DATA_BASE}/scenarios/${v}`;
        try {
          await fetchOnce(`${root}/index.json`, 'json');
          return root;
        } catch {
          // not published yet: try the one before
        }
      }
      return SCENARIO_ROOT;
    })();
  }
  return resolved;
}

/** Every scenario: start, water type, straightness, the buoy's first hours, its windows. */
export async function loadIndex(root) {
  return fetchOnce(`${root ?? await scenarioRoot()}/index.json`, 'json');
}

/** The benchmark across every scenario, if it has been published. */
export async function loadBenchmark(root) {
  return fetchOnce(`${root ?? await scenarioRoot()}/benchmark.json`, 'json');
}

export function windowName(noise, hours) {
  return `${noise}_${hours}h`;
}

/**
 * One window, ready for referee.Episode: {meta, cloud, marker, target, flights,
 * turnRateDegS, arrivalHeadingDeg}. A v1 window's helicopter turned at once (Infinity).
 */
export async function loadWindow(scenario, noise, hours, root) {
  root = root ?? await scenarioRoot();
  const name = windowName(noise, hours);
  const [meta, buffer] = await Promise.all([
    fetchOnce(`${root}/${scenario}/${name}.json`, 'json'),
    fetchOnce(`${root}/${scenario}/${name}.f32`, 'bin'),
  ]);
  const f32 = new Float32Array(buffer);
  if (f32.length !== meta.frames * meta.particles * 2) {
    throw new Error(`${scenario} ${name}: the cloud holds ${f32.length} numbers, expected `
      + `${meta.frames * meta.particles * 2}`);
  }
  const marker = { tS: meta.marker.t_s, lat: meta.marker.lat, lon: meta.marker.lon };
  const target = meta.target
    ? { tS: meta.target.t_s, lat: meta.target.lat, lon: meta.target.lon } : null;
  const { lat, lon } = cloudFromOffsets(f32, meta.frames, meta.particles, marker);
  const cloud = {
    frames: meta.frames,
    particles: meta.particles,
    lat,
    lon,
    weight: new Float64Array(meta.particles).fill(meta.weight),
  };
  const flights = {};
  for (const [key, f] of Object.entries(meta.flights)) {
    flights[key] = {
      ...f,
      steps: f.steps.map((s) => ({
        tS: s.t_s, eastM: s.east_m, northM: s.north_m, headingDeg: s.heading_deg ?? undefined,
      })),
    };
  }
  return {
    meta, cloud, marker, target, flights,
    turnRateDegS: meta.turn_rate_deg_s ?? Infinity,
    arrivalHeadingDeg: meta.arrival_heading_deg ?? meta.drift_bearing_deg ?? 0,
  };
}

/**
 * The current and wind grid about the datum, every few minutes of the window:
 * {times (s since arrival), size, stepM, at(i, row, col) -> [cu, cv, wu, wv]}.
 */
export async function loadField(scenario, hours, index, root) {
  root = root ?? await scenarioRoot();
  const buffer = await fetchOnce(`${root}/${scenario}/field_${hours}h.f32`, 'bin');
  const f32 = new Float32Array(buffer);
  const { half_km: halfKm, step_km: stepKm, every_min: everyMin } = index.field;
  const size = 2 * Math.round(halfKm / stepKm) + 1;
  const count = f32.length / (size * size * 4);
  return {
    size,
    stepM: stepKm * 1000,
    times: Array.from({ length: count }, (_, i) => i * everyMin * 60),
    at(i, row, col) {
      const o = ((i * size + row) * size + col) * 4;
      return [f32[o], f32[o + 1], f32[o + 2], f32[o + 3]];
    },
  };
}
