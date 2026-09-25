/**
 * searchView.js -- the Search view: a Coast Guard search, or a helicopter flown by hand
 * (issues #64, #68, #69).
 *
 * A DOCTRINAL SEARCH, in the order the Coast Guard works (docs/ADR003.md):
 *
 *   1. TARGET   the buoy selected in the drifter list, at the clock's current time. That
 *               moment is the report time and the buoy's position then is the LKP. Once it
 *               is chosen the drifter list gets out of the way.
 *   2. BASE     a click on the map: where the helicopter launches from.
 *   3. FLY      the forcing for the next few hours is loaded; the datum is drifted from
 *               the LKP; the helicopter waits 30 minutes, flies out at 125 kt, drops a
 *               marker, and flies the pattern about it at 90 kt.
 *   4. RESULT   found, and when -- or the closest pass -- judged against where the real
 *               buoy actually went, not where the model said it would.
 *
 * FLOWN BY HAND (#68): "Spawn a helicopter", click the map, and steer with WASD or the
 * arrow keys. It appears on the spot, takes off on the first key, and flies one on-scene
 * window at 90 kt; a chosen buoy is found by passing over it.
 *
 * ONE CLOCK (#69). A search used to run a clock of its own beside the site's, and the two
 * disagreed on screen: the time bar stood still while the helicopter flew, Play moved the
 * site's copy of the buoy while the search's copy stayed put, and nothing said how fast
 * search time was passing. Now, while a search is loaded, it OWNS the time bar: the slider
 * scrubs the search, Play plays and pauses it, the label says where it is and how fast
 * it is going, and the site clock is moved to the search's moment, so the header, the
 * wind and current fields, and the buoy all show the same instant. Reset, or leaving the
 * view, hands the bar back.
 *
 * Everything that decides anything is in searchRun.js and tested there; this file only
 * sequences it, loads data and animates.
 */

import { Compass } from './compass.js';
import { positionAt, undroguedAt } from './drifters.js';
import { SWEEP_WIDTH_M, formatDistance } from './geo.js';
import { PATTERNS } from './patterns.js';
import { NM_M, ON_SCENE_WINDOW_S, distanceM, transitTimeS } from './platform.js';
import { OceanLayer } from './oceanLayer.js';
import { resultantSampler } from './pointDrift.js';
import { SearchLayer } from './searchLayer.js';
import { SearchPanel, speedLabel } from './searchPanel.js';
import {
  ManualFlight, TARGETS, datumErrorM, detect, formatDuration, formatElapsed, freePlan, headingFromKeys,
  helicopterAt, keyDirection, planSearch, searchPath,
} from './searchRun.js';

/** Forcing to load past the report: launch, a 300 NM transit, the window, and slack. */
const LOOKAHEAD_MS = 4 * 3600 * 1000;

/** Redraw at most this often while animating; the map does not need 60 fps. */
const FRAME_MS = 40;

/** Move the site clock (and so repaint the fields) at most this often, in real ms. */
const MOMENT_MS = 250;

/*
  CLOSE ENOUGH TO SEE THE STRIP. The site stops at zoom 11, where 185 m is under three
  pixels; the search lifts that to 16 while it is open -- the strip is about 11 px wide at
  13 and 90 px at 16. Basemaps up-scale their last native zoom past that rather than going
  blank (main.js). Flying yourself follows the helicopter at 15 (#73).
*/
const SEARCH_MAX_ZOOM = 16;
const FOLLOW_ZOOM = 15;

const PHASES = {
  ready: 'on the ground: airborne within 30 minutes (B-0)',
  transit: 'flying out at 125 kt (fast-forwarded)',
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
 * @param {(on: boolean) => void} deps.showDrifters  show or hide the drifter layer and list
 * @param {object} deps.timeBar                      take / update / release the bottom bar
 * @param {(ms) => void} deps.setMoment              move the site clock to a moment
 * @param {(text) => void} deps.setStatus
 */
export function createSearchView(deps) {
  const { map, setStatus, timeBar } = deps;
  const layer = new SearchLayer();
  const compass = new Compass();
  const ocean = new OceanLayer();
  // The forcing where the helicopter is, for the compass and the sea. Reads the site's
  // store at every call, so it follows a tier switch, and answers null until loaded.
  const sampleHere = resultantSampler(deps.resultantSource, deps.frameOf);

  let placing = null;          // 'base' | 'spawn' | null
  let base = null;
  let target = null;           // { buoy, track, reportMs, lkp }
  let mode = null;             // 'pattern' | 'free' | null
  let plan = null;
  let result = null;
  let flight = null;
  let datumErr = null;
  let targetAt = null;
  const held = new Set();
  const anim = { s: 0, raf: null, last: null, drawn: 0, moment: 0, state: 'idle', zoomed: false, rate: 0 };

  const panel = new SearchPanel({
    onUseBuoy, onChangeBuoy, onPlaceBase, onPrimary, onSpawn, onEdit: () => reset(), onReset,
  });

  // The panel is part of the layer: it shows exactly when the layer does.
  const siteMaxZoom = map.getMaxZoom();
  map.on('overlayadd overlayremove', () => {
    const on = map.hasLayer(layer);
    if (on && !panel._map) {
      panel.addTo(map);
      map.addLayer(ocean);
      map.setMaxZoom(Math.max(siteMaxZoom, SEARCH_MAX_ZOOM));
    } else if (!on && panel._map) {
      reset();
      map.removeControl(panel);
      if (map.hasLayer(ocean)) map.removeLayer(ocean);
      map.setMaxZoom(siteMaxZoom);
      setPlacing(null);
    }
  });

  /* ---------------------------------------------------------------- the keyboard */

  function onKeyDown(e) {
    if (mode !== 'free' || !flight || flight.done) return;
    const d = keyDirection(e.code);
    if (!d) return;
    e.preventDefault();
    held.add(d);
    // The first steering key is the take-off.
    if (anim.state === 'armed') play();
  }
  function onKeyUp(e) {
    const d = keyDirection(e.code);
    if (!d) return;
    held.delete(d);
    if (mode === 'free') e.preventDefault();
  }
  function onBlur() { held.clear(); }
  function grabKeys() {
    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', onBlur);
    // The map pans on the arrow keys; while flying, the keys steer and nothing else.
    if (map.keyboard) map.keyboard.disable();
    // Off the buttons and menus, so an arrow key cannot change a choice instead.
    if (document.activeElement && document.activeElement.blur) document.activeElement.blur();
  }
  function releaseKeys() {
    document.removeEventListener('keydown', onKeyDown);
    document.removeEventListener('keyup', onKeyUp);
    window.removeEventListener('blur', onBlur);
    held.clear();
    if (map.keyboard) map.keyboard.enable();
  }

  /* ---------------------------------------------------------------- choices */

  function setPlacing(what) {
    placing = what;
    panel.setPlacing(what);
  }

  function describeBase() {
    if (!base) { panel.setBase('Base: not placed', false); return; }
    let text = `Base ${coord(base)}`;
    if (target) {
      const d = distanceM(base, target.lkp);
      const t = transitTimeS(d);
      text += ` · ${(d / NM_M).toFixed(0)} NM out · `
        + (t === null ? 'beyond the H-60\'s 300 NM range'
          : `on scene ${formatDuration(t)} after the call`);
    }
    panel.setBase(text, true);
  }

  async function onUseBuoy() {
    const buoy = deps.getSelectedBuoy();
    if (!buoy) {
      panel.setTarget('No buoy selected: click one in the Drifters list or on the map first.', false);
      return;
    }
    const reportMs = deps.getTime();
    let track;
    try {
      track = await deps.getTrack(buoy);
    } catch (err) {
      panel.setTarget(`Could not load buoy ${buoy.id}'s track: ${err.message}`, false);
      return;
    }
    const at = positionAt(track, reportMs);
    if (!at) {
      panel.setTarget(`Buoy ${buoy.id} has no fix at ${utc(reportMs)}: move the clock inside its record.`, false);
      return;
    }
    reset();
    target = { buoy, track, reportMs, lkp: { lat: at[0], lon: at[1] } };
    panel.setTarget(`Buoy ${buoy.id}, reported missing ${utc(reportMs)} at ${coord(target.lkp)}`
      + (buoy.sealed ? ' · sealed: fine to watch, not to tune on' : ''), true);
    describeBase();
    layer.setTarget(target.lkp);
    // Whether it still has its drogue then: that decides how much the wind moves it.
    const und = undroguedAt(track, reportMs);
    // Said in plain words (#76): "Drogue lost by then" assumed the reader knew what one was.
    panel.setDrogueHint(und === null ? null : (und
      ? 'It had lost its drogue by then, the underwater sail that keeps a buoy with the water, '
        + 'so the wind pushes it too, though less than a person.'
      : 'It still had its drogue then, an underwater sail 15 m down that keeps it with the '
        + 'water, so "current only" fits.'), und ? 'undrogued' : 'drogued');
    // The list has done its job. Hiding the drifter layer also removes its second copy of
    // the buoy, which moved on the site clock while the search drew its own.
    deps.showDrifters(false);
    setStatus('Target chosen. Place the base, then Fly the search.');
  }

  function onChangeBuoy() {
    reset();
    target = null;
    layer.setTarget(null);
    panel.setDrogueHint(null);
    panel.setTarget('Pick a buoy in the Drifters list and move the clock to when it is reported missing.', false);
    describeBase();
    deps.showDrifters(true);
  }

  function onPlaceBase() {
    setPlacing(placing === 'base' ? null : 'base');
    setStatus(placing ? 'Click the map where the helicopter launches from.' : 'Search.');
  }

  function onSpawn() {
    setPlacing(placing === 'spawn' ? null : 'spawn');
    setStatus(placing ? 'Click the map where the helicopter should appear.' : 'Search.');
  }

  /**
   * The map's click, offered here first. In the Search view a click only ever places
   * something, so it is always taken; the point panel belongs to the other views.
   */
  function consumeClick(latlng) {
    if (!map.hasLayer(layer)) return false;
    if (placing === 'base') {
      base = { lat: latlng.lat, lon: latlng.lng };
      setPlacing(null);
      if (mode === 'pattern') reset();
      layer.setBase(base);
      describeBase();
      setStatus('Base placed. Choose how the Coast Guard searches, and Fly the search.');
    } else if (placing === 'spawn') {
      setPlacing(null);
      spawn({ lat: latlng.lat, lon: latlng.lng });
      setStatus('Helicopter ready: steer with W A S D or the arrow keys.');
    }
    return true;
  }

  /**
   * The primary button: fly a search when none is loaded, else play, pause or replay it.
   * The time bar's ▶ comes here too while the Search view is open.
   */
  function onPrimary(choices) {
    if (plan) { togglePlay(); return; }
    if (panel.currentTab() === 'fly') {
      setStatus('Spawn a helicopter first: press "Spawn a helicopter on the map", then click the map.');
      return;
    }
    onFly(choices);
  }

  /* ---------------------------------------------------------------- a doctrinal search */

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
    if (!target) { panel.setPhase('Step 1 first: choose who is missing.'); return; }
    if (!base) { panel.setPhase('Step 2 first: place the base.'); return; }
    reset();
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

    mode = 'pattern';
    targetAt = (ms) => positionAt(target.track, ms);
    result = detect(plan, targetAt);
    datumErr = datumErrorM(plan, targetAt);
    layer.setPlan(plan, targetAt, result);
    // The set-up has done its job: fold it to one line so the result fits (#71).
    const drift = choices.targetKind === 'person' ? 'current + 2 % of wind' : 'current only';
    panel.collapseSetup([
      ['Buoy', target.buoy.id],
      ['Base', `${(plan.distanceM / NM_M).toFixed(0)} NM out`],
      ['Pattern', PATTERNS[plan.patternKind].label],
      ['Datum', `drifted with ${drift}`],
    ]);
    map.fitBounds([[base.lat, base.lon], [plan.datum.lat, plan.datum.lon]], { padding: [60, 60] });
    ownTime([
      { toS: plan.launchS, label: 'call to launch', cls: 'ph-ready' },
      { toS: plan.arriveS, label: 'flying out', cls: 'ph-transit' },
      { toS: plan.endS, label: `on scene · ${PATTERNS[plan.patternKind].short}`, cls: 'ph-search' },
    ]);
    play();
  }

  /* ---------------------------------------------------------------- flown by hand */

  function spawn(at) {
    reset();
    mode = 'free';
    plan = freePlan({ spawn: at, startMs: deps.getTime(), lkp: target ? target.lkp : null });
    targetAt = target ? (ms) => positionAt(target.track, ms) : () => null;
    flight = new ManualFlight(plan, targetAt);
    layer.setPlan(plan, targetAt, null, { flight });
    panel.showTab('fly');
    // Close up and following, or just in view, as the tick-box says.
    if (panel.follow()) map.setView([at.lat, at.lon], FOLLOW_ZOOM, { animate: false });
    else if (map.getZoom() < 12) map.setView([at.lat, at.lon], 12);
    // For the compass and the sea: loaded in the background, shown once it lands.
    loadForcing(plan.reportMs).catch(() => {});
    // A person needs time to steer: past 45 s per second the helicopter crosses the whole
    // area a pattern would cover in a couple of seconds.
    if (panel.speedX() > 45) panel.setSpeed(15);
    ownTime([{ toS: plan.endS, label: 'your flight', cls: 'ph-search' }]);
    grabKeys();
    anim.state = 'armed';
    panel.setFlying('armed');
    render(0, true);
  }

  /* ---------------------------------------------------------------- the one clock */

  function ownTime(bands) {
    timeBar.take({ seek, togglePlay }, { endS: plan.endS, bands });
    if (!compass._map) compass.addTo(map);
  }

  /** Where the search can play to: the result if found, else the end of the window. */
  function stopAt() {
    if (mode === 'free') return flight.done ? flight.s : plan.endS;
    return result && result.found ? result.foundS : plan.endS;
  }

  /** Flying by hand and still airborne: time is made by flying, not replayed. */
  function live() {
    return mode === 'free' && !flight.done;
  }

  /** The time bar's read-out: T+ clock time, which the bar's tooltip explains. */
  function label(s) {
    const h = helicopterAt(plan, s);
    let what;
    if (anim.state === 'done' && s >= stopAt() - 1e-6) {
      const r = mode === 'free' ? flight.result() : result;
      what = r && r.found ? 'found' : (mode === 'free' ? 'flight over' : PHASES.done);
    } else if (mode === 'free') {
      what = anim.state === 'armed' ? 'press W A S D or an arrow key to take off' : 'you are flying';
    } else {
      what = PHASES[h.phase];
    }
    const pace = anim.state === 'playing' && !(mode === 'pattern' && s < plan.arriveS)
      ? ` · ${speedLabel(panel.speedX())}` : '';
    return `${formatElapsed(s)} · ${what}${pace}`;
  }

  /** The panel's read-out: the same, with the time said in words. */
  function phaseLine(s) {
    return label(s).replace(/^T\+\S+/, `${formatDuration(s)} since the call`);
  }

  function render(s, force = false) {
    const now = performance.now();
    layer.setTime(s);
    // A helicopter flown by hand crosses a screen in a few minutes: follow it close up,
    // or at least keep it clear of the panel on the left and the legends on the right.
    if (mode === 'free' && live()) {
      const p = flight.position();
      if (panel.follow()) map.setView([p.lat, p.lon], map.getZoom(), { animate: false });
      else {
        map.panInside([p.lat, p.lon], {
          paddingTopLeft: [400, 160], paddingBottomRight: [340, 300], animate: false,
        });
      }
    }
    // The compass and the sea read the forcing where the helicopter is now.
    const h = mode === 'free'
      ? (s >= flight.s ? flight.position() : flight.positionAt(s))
      : helicopterAt(plan, Math.min(s, plan.endS));
    const here = sampleHere(plan.reportMs + s * 1000, h.lat, h.lon);
    const current = here && here.current ? here.current : null;
    const wind = here && here.wind ? here.wind : null;
    compass.update({ heading: h.heading, current, wind });
    ocean.setConditions({ current, wind, rate: anim.state === 'playing' ? anim.rate : 0 });
    timeBar.update(s, label(s), anim.state === 'playing');
    if (force || now - anim.moment >= MOMENT_MS) {
      anim.moment = now;
      deps.setMoment(plan.reportMs + s * 1000);
    }
    panel.setPhase(phaseLine(s));
  }

  function play() {
    if (!plan) return;
    if (anim.state === 'done' || (!live() && anim.s >= stopAt() - 1e-6)) anim.s = 0;   // replay
    anim.state = 'playing';
    anim.last = null;
    panel.setFlying('playing');
    if (anim.raf) cancelAnimationFrame(anim.raf);
    anim.raf = requestAnimationFrame(tick);
  }

  function pause() {
    if (anim.state !== 'playing') return;
    anim.state = 'paused';
    if (anim.raf) cancelAnimationFrame(anim.raf);
    anim.raf = null;
    panel.setFlying('paused');
    render(anim.s, true);
  }

  function togglePlay() {
    if (!plan) return;
    if (anim.state === 'playing') pause();
    else play();
  }

  /** The slider, dragged. A flight still being flown cannot be rewound, only watched. */
  function seek(s) {
    if (!plan || live()) { timeBar.update(anim.s, label(anim.s), anim.state === 'playing'); return; }
    pause();
    anim.s = Math.min(Math.max(s, 0), stopAt());
    if (anim.state !== 'paused') { anim.state = 'paused'; panel.setFlying('paused'); }
    render(anim.s, true);
  }

  function tick(now) {
    if (anim.state !== 'playing' || !plan) return;
    if (anim.last === null) anim.last = now;
    const dt = Math.min((now - anim.last) / 1000, 0.25);
    anim.last = now;
    const speed = panel.speedX();

    if (live()) {
      anim.rate = speed;
      flight.advance(dt * speed, headingFromKeys(held));
      anim.s = flight.s;
    } else {
      // Launch and transit fast-forward to about eight seconds (three at the quick look).
      const rate = mode === 'pattern' && anim.s < plan.arriveS
        ? Math.max(300, plan.arriveS / (speed >= 600 ? 3 : 8)) : speed;
      anim.rate = rate;
      anim.s = Math.min(anim.s + dt * rate, stopAt());
    }

    if (mode === 'pattern' && !anim.zoomed && anim.s >= plan.arriveS) {
      anim.zoomed = true;
      const p = searchPath(plan);
      map.fitBounds([[Math.min(...p.lat), Math.min(...p.lon)], [Math.max(...p.lat), Math.max(...p.lon)]],
        { padding: [40, 40], maxZoom: 15 });
    }

    const over = mode === 'free' ? flight.done && anim.s >= flight.s - 1e-6 : anim.s >= stopAt() - 1e-6;
    if (now - anim.drawn >= FRAME_MS || over) {
      anim.drawn = now;
      render(anim.s, over);
    }
    if (over) { finish(); return; }
    anim.raf = requestAnimationFrame(tick);
  }

  /* ---------------------------------------------------------------- results */

  function finish() {
    anim.state = 'done';
    anim.raf = null;
    panel.setFlying('done');
    if (mode === 'free') releaseKeys();
    const lines = mode === 'free' ? freeLines() : patternLines();
    panel.setResult(lines.map((l) => `<p>${l}</p>`).join(''));
    timeBar.update(anim.s, label(anim.s), false);
    panel.setPhase(phaseLine(anim.s));
    const r = mode === 'free' ? flight.result() : result;
    setStatus(r && r.found ? 'Found. Drag the time bar to look back, or Reset.' : 'Done. Drag the time bar to look back, or Reset.');
  }

  function patternLines() {
    const lines = [];
    const name = PATTERNS[plan.patternKind].label;
    if (result.found) {
      lines.push(`<b class="sp-ok">Found</b> ${formatDuration(result.foundS)} after the call, `
        + `${formatDuration(result.foundS - plan.arriveS)} into the ${name}.`);
    } else {
      const pass = Number.isFinite(result.closestM) ? formatDistance(result.closestM) : 'none';
      lines.push(`<b class="sp-miss">Not found</b> in the 45-minute window. Closest pass ${pass}`
        + (result.closestS !== null ? `, ${formatDuration(result.closestS)} after the call.` : '.'));
    }
    if (datumErr !== null) {
      lines.push(`The <b>datum error</b> was ${formatDistance(datumErr)}: that far from the drift model's `
        + 'prediction to where the buoy really was when the helicopter arrived.');
    }
    lines.push(`First leg ${plan.firstBearingDeg.toFixed(0)}°, ${plan.bearingSource}.`);
    for (const note of plan.notes) lines.push(`Note: ${note}.`);
    return lines;
  }

  function freeLines() {
    const r = flight.result();
    const lines = [];
    const flown = flight.s - plan.arriveS;
    if (target) {
      if (r.found) {
        lines.push(`<b class="sp-ok">You found it</b>, ${formatDuration(r.foundS)} into your flight.`);
      } else {
        const pass = Number.isFinite(r.closestM) ? formatDistance(r.closestM) : 'none';
        lines.push(`<b class="sp-miss">Not found.</b> Your closest pass was ${pass}.`);
      }
    }
    lines.push(`You flew ${formatDistance(flight.lengthM)} in ${formatDuration(flown)}, sweeping about `
      + `${((flight.lengthM * SWEEP_WIDTH_M) / 1e6).toFixed(1)} km² if no strip overlapped.`);
    if (flown >= ON_SCENE_WINDOW_S - 1) lines.push('That was the whole 45-minute window.');
    return lines;
  }

  /* ---------------------------------------------------------------- tidy */

  function stop() {
    if (anim.raf) cancelAnimationFrame(anim.raf);
    anim.raf = null;
    anim.state = 'idle';
  }

  /** Clear the search and hand the time bar back to the site. Target and base stay. */
  function reset() {
    stop();
    releaseKeys();
    const owned = plan !== null;
    mode = null;
    plan = null;
    result = null;
    flight = null;
    datumErr = null;
    targetAt = null;
    anim.s = 0;
    anim.zoomed = false;
    layer.clear();
    layer.setBase(base);
    layer.setTarget(target ? target.lkp : null);
    panel.setFlying('idle');
    panel.setPhase('');
    panel.setResult('');
    panel.expandSetup();
    if (compass._map) map.removeControl(compass);
    ocean.setConditions({ rate: 0 });
    if (owned) timeBar.release();
  }

  function onReset() {
    reset();
    setStatus('Search reset.');
  }

  /** Whether the time bar's ▶ belongs to the search: whenever the Search view is open. */
  function claimsPlay() {
    return map.hasLayer(layer);
  }

  /** The time bar's ▶, in the Search view: the same as the primary button. */
  function playFromBar() {
    if (!plan && panel.currentTab() === 'search' && (!target || !base)) {
      setStatus(target ? 'Place the base, then ▶ flies the search.'
        : 'Choose who is missing and place the base, then ▶ flies the search.');
      return;
    }
    onPrimary(panel.choices());
  }

  return { layer, panel, consumeClick, claimsPlay, playFromBar };
}
