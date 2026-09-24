/**
 * searchView.js -- the Search view: a Coast Guard pattern flown to find a real buoy (issue #64).
 *
 * The flow, in the order the Coast Guard works (docs/ADR003.md):
 *
 *   1. TARGET   the buoy selected in the drifter list, at the clock's current time. That
 *               moment is the report time and the buoy's position then is the LKP.
 *   2. BASE     a click on the map: where the helicopter launches from.
 *   3. FLY      the forcing for the next few hours is loaded; the datum is drifted from
 *               the LKP with the target's leeway; the helicopter waits 30 minutes, flies
 *               out at 125 kt, drops a marker, and flies the pattern about it at 90 kt.
 *   4. RESULT   found, and when -- or the closest pass -- judged against where the real
 *               buoy actually went, not where the model said it would.
 *
 * The search runs on ITS OWN clock, in seconds after the report. The site's clock steps
 * a whole hour at a time, and a helicopter covers 167 km in an hour; this one plays the
 * 45-minute window in about a minute and leaves the site's clock where it was.
 *
 * Everything that decides anything is in searchRun.js and tested there; this file only
 * sequences it, loads data and animates.
 */

import { positionAt } from './drifters.js';
import { formatDistance } from './geo.js';
import { PATTERNS } from './patterns.js';
import { NM_M, distanceM, transitTimeS } from './platform.js';
import { resultantSampler } from './pointDrift.js';
import { SearchLayer } from './searchLayer.js';
import { SearchPanel } from './searchPanel.js';
import {
  TARGETS, datumErrorM, detect, formatElapsed, helicopterAt, planSearch, searchPath,
} from './searchRun.js';

/** Forcing to load past the report: launch, a 300 NM transit, the window, and slack. */
const LOOKAHEAD_MS = 4 * 3600 * 1000;

/** Redraw at most this often while animating; the map does not need 60 fps. */
const FRAME_MS = 40;

/*
  CLOSE ENOUGH TO SEE THE STRIP. The site stops at zoom 11, where 185 m is under three
  pixels; the search lifts that to 13 while it is open, where the strip is about eleven
  pixels wide. 13, not more: the ocean basemap has no tiles past it.
*/
const SEARCH_MAX_ZOOM = 13;

const PHASES = {
  ready: 'on the ground: 30 minutes to launch (B-0)',
  transit: 'in transit at 125 kt',
  search: 'searching at 90 kt',
  done: 'window over',
};

function coord(p) {
  const ns = p.lat >= 0 ? 'N' : 'S';
  const ew = p.lon >= 0 ? 'E' : 'W';
  return `${Math.abs(p.lat).toFixed(3)} ${ns} ${Math.abs(p.lon).toFixed(3)} ${ew}`;
}

function utc(ms) {
  return `${new Date(ms).toISOString().slice(0, 16).replace('T', ' ')} UTC`;
}

/**
 * @param {object} deps
 * @param {L.Map} deps.map
 * @param {() => object|null} deps.getSelectedBuoy   the drifter index entry, or null
 * @param {(buoy) => Promise<object>} deps.getTrack  the decoded track
 * @param {() => number} deps.getTime                the site clock, epoch ms
 * @param {object} deps.resultantSource              the site's ResultantSource
 * @param {(ms) => number|null} deps.frameOf         epoch ms to a frame of the wind tier in use
 * @param {(ms) => Promise<void>} deps.prepareHourly switch the site to its hourly tier near ms
 * @param {(text) => void} deps.setStatus
 */
export function createSearchView(deps) {
  const { map, setStatus } = deps;
  const layer = new SearchLayer();
  let placing = false;
  let base = null;
  let target = null;
  let plan = null;
  let result = null;
  const anim = { s: 0, raf: null, last: null, drawn: 0, state: 'idle', zoomed: false };

  const panel = new SearchPanel({ onUseBuoy, onPlaceBase, onFly, onPause, onReset });

  // The panel is part of the layer: it shows exactly when the layer does.
  const siteMaxZoom = map.getMaxZoom();
  map.on('overlayadd overlayremove', () => {
    const on = map.hasLayer(layer);
    if (on && !panel._map) {
      panel.addTo(map);
      map.setMaxZoom(Math.max(siteMaxZoom, SEARCH_MAX_ZOOM));
    } else if (!on && panel._map) {
      map.removeControl(panel);
      map.setMaxZoom(siteMaxZoom);
      stop();
      placing = false;
    }
  });

  function describeBase() {
    if (!base) { panel.setBase('Base: not placed'); return; }
    let text = `Base: ${coord(base)}`;
    if (target) {
      const d = distanceM(base, target.lkp);
      const t = transitTimeS(d);
      text += ` · ${(d / NM_M).toFixed(0)} NM to the LKP · `
        + (t === null ? 'beyond the H-60\'s 300 NM radius of action'
          : `on scene about ${formatElapsed(t).slice(2)} after the call`);
    }
    panel.setBase(text);
  }

  async function onUseBuoy() {
    const buoy = deps.getSelectedBuoy();
    if (!buoy) {
      panel.setTarget('No buoy selected: click one in the drifter list or on the map first.');
      return;
    }
    const reportMs = deps.getTime();
    let track;
    try {
      track = await deps.getTrack(buoy);
    } catch (err) {
      panel.setTarget(`Could not load buoy ${buoy.id}'s track: ${err.message}`);
      return;
    }
    const at = positionAt(track, reportMs);
    if (!at) {
      panel.setTarget(`Buoy ${buoy.id} has no fix at ${utc(reportMs)}: move the clock inside its record.`);
      return;
    }
    target = { buoy, track, reportMs, lkp: { lat: at[0], lon: at[1] } };
    panel.setTarget(`Buoy ${buoy.id}, reported ${utc(reportMs)} at ${coord(target.lkp)}`
      + (buoy.sealed ? ' · sealed: fine to watch, not to tune on' : ''));
    describeBase();
    reset();
    layer.setTarget(target.lkp);
    // The list has done its job; folding it leaves the search panel room to show its result.
    if (deps.collapseDrifterList) deps.collapseDrifterList();
  }

  function onPlaceBase() {
    placing = !placing;
    panel.setPlacing(placing);
    setStatus(placing ? 'Click the map where the helicopter launches from.' : 'Search.');
  }

  /** The map's click, offered here first. True if the search took it. */
  function consumeClick(latlng) {
    if (!placing || !map.hasLayer(layer)) return false;
    base = { lat: latlng.lat, lon: latlng.lng };
    placing = false;
    panel.setPlacing(false);
    layer.setBase(base);
    describeBase();
    setStatus('Base placed. Choose a pattern and press Fly.');
    return true;
  }

  async function loadForcing(reportMs) {
    await deps.prepareHourly(reportMs);
    const frames = new Set();
    for (let ms = reportMs; ms <= reportMs + LOOKAHEAD_MS; ms += 1800 * 1000) {
      const f = deps.frameOf(ms);
      if (f !== null) frames.add(f);
    }
    await Promise.all([...frames].map((f) => deps.resultantSource.ensure(f)));
  }

  async function onFly(choices) {
    if (!target) { panel.setPhase('Step 1 first: choose the target buoy.'); return; }
    if (!base) { panel.setPhase('Step 2 first: place the base.'); return; }
    stop();
    panel.setResult('');
    panel.setPhase('Loading the current and wind for the next few hours…');
    try {
      await loadForcing(target.reportMs);
    } catch (err) {
      panel.setPhase(`Could not load the forcing: ${err.message}`);
      return;
    }
    const sample = resultantSampler(deps.resultantSource, deps.frameOf);
    plan = planSearch({
      base,
      lkp: target.lkp,
      reportMs: target.reportMs,
      patternKind: choices.patternKind,
      targetLeeway: TARGETS[choices.targetKind].leeway,
      sample,
    });
    if (plan.error) { panel.setPhase(plan.error); plan = null; return; }

    const targetAt = (ms) => positionAt(target.track, ms);
    result = detect(plan, targetAt);
    result.datumErrorM = datumErrorM(plan, targetAt);
    layer.setPlan(plan, targetAt, result);
    map.fitBounds([[base.lat, base.lon], [plan.datum.lat, plan.datum.lon]], { padding: [60, 60] });

    anim.s = 0;
    anim.zoomed = false;
    anim.state = 'flying';
    panel.setFlying('flying');
    anim.last = null;
    anim.raf = requestAnimationFrame(tick);
  }

  function stopAt() {
    return result && result.found ? result.foundS : plan.endS;
  }

  function tick(now) {
    if (anim.state !== 'flying' || !plan) return;
    if (anim.last === null) anim.last = now;
    const dt = (now - anim.last) / 1000;
    anim.last = now;
    // Launch and transit fast-forward to about eight seconds (three at the quick-look
    // speed); the search itself plays at the chosen rate.
    const speed = panel.speedX();
    const rate = anim.s < plan.arriveS ? Math.max(300, plan.arriveS / (speed >= 600 ? 3 : 8)) : speed;
    anim.s = Math.min(anim.s + dt * rate, stopAt());

    if (!anim.zoomed && anim.s >= plan.arriveS) {
      anim.zoomed = true;
      const p = searchPath(plan);
      const lats = p.lat;
      const lons = p.lon;
      map.fitBounds([[Math.min(...lats), Math.min(...lons)], [Math.max(...lats), Math.max(...lons)]],
        { padding: [40, 40], maxZoom: 15 });
    }

    if (now - anim.drawn >= FRAME_MS || anim.s >= stopAt()) {
      anim.drawn = now;
      layer.setTime(anim.s);
      const h = helicopterAt(plan, anim.s);
      panel.setPhase(`${formatElapsed(anim.s)} · ${PHASES[h.phase]}`);
    }

    if (anim.s >= stopAt()) {
      finish();
      return;
    }
    anim.raf = requestAnimationFrame(tick);
  }

  function finish() {
    anim.state = 'done';
    panel.setFlying('done');
    const label = PATTERNS[plan.patternKind].label;
    const lines = [];
    if (result.found) {
      lines.push(`<b class="sp-ok">Found</b> at ${formatElapsed(result.foundS)}, `
        + `${formatElapsed(result.foundS - plan.arriveS).slice(2)} into the ${label}.`);
    } else {
      const pass = Number.isFinite(result.closestM) ? formatDistance(result.closestM) : 'none';
      lines.push(`<b class="sp-miss">Not found</b> in the 45-minute window. Closest pass ${pass}`
        + (result.closestS !== null ? ` at ${formatElapsed(result.closestS)}.` : '.'));
    }
    if (result.datumErrorM !== null) {
      lines.push(`The datum was ${formatDistance(result.datumErrorM)} from where the buoy really `
        + 'was when the helicopter arrived.');
    }
    lines.push(`First leg ${plan.firstBearingDeg.toFixed(0)}°, ${plan.bearingSource}.`);
    for (const note of plan.notes) lines.push(`Note: ${note}.`);
    panel.setResult(lines.map((l) => `<p>${l}</p>`).join(''));
    panel.setPhase(`${formatElapsed(anim.s)} · ${result.found ? 'search over: found' : PHASES.done}`);
    setStatus(result.found ? 'Found. Reset to fly again.' : 'Not found. Reset to fly again.');
  }

  function onPause() {
    if (anim.state === 'flying') {
      anim.state = 'paused';
      if (anim.raf) cancelAnimationFrame(anim.raf);
      panel.setFlying('paused');
    } else if (anim.state === 'paused') {
      anim.state = 'flying';
      anim.last = null;
      panel.setFlying('flying');
      anim.raf = requestAnimationFrame(tick);
    }
  }

  function stop() {
    if (anim.raf) cancelAnimationFrame(anim.raf);
    anim.raf = null;
    if (anim.state !== 'idle') anim.state = 'idle';
  }

  function reset() {
    stop();
    plan = null;
    result = null;
    layer.clear();
    layer.setBase(base);
    panel.setFlying('idle');
    panel.setPhase('');
    panel.setResult('');
  }

  function onReset() {
    reset();
    setStatus('Search reset.');
  }

  return { layer, panel, consumeClick };
}
