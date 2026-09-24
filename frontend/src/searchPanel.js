/**
 * searchPanel.js -- the controls for a doctrinal search (issue #64).
 *
 * Four things to choose, in the order the Coast Guard would: the target and when it was
 * reported, where the helicopter launches from, which pattern, and what kind of target
 * it is (which sets the leeway the datum is drifted with). Then Fly. The panel only
 * collects choices and shows state; searchView.js does the work.
 */

import L from 'leaflet';

import { PATTERNS } from './patterns.js';
import { TARGETS } from './searchRun.js';
import { SEARCH_COLOURS } from './searchLayer.js';

export const SPEEDS = [
  { x: 30, label: '30×' },
  { x: 45, label: '45× (window in a minute)' },
  { x: 120, label: '120×' },
  { x: 600, label: '600× (quick look)' },
];

export const SearchPanel = L.Control.extend({
  options: { position: 'topright' },

  /** @param {{onUseBuoy, onPlaceBase, onFly, onPause, onReset}} handlers */
  initialize(handlers) {
    this._h = handlers;
  },

  onAdd() {
    const root = L.DomUtil.create('div', 'search-panel');
    const patterns = Object.entries(PATTERNS)
      .map(([k, p]) => `<option value="${k}">${p.label}</option>`).join('');
    const targets = Object.entries(TARGETS)
      .map(([k, t]) => `<option value="${k}">${t.label}</option>`).join('');
    const speeds = SPEEDS.map((s) => `<option value="${s.x}"${s.x === 45 ? ' selected' : ''}>${s.label}</option>`).join('');
    root.innerHTML = `
      <div class="sp-head">
        <span class="sp-title">Search</span>
        <span class="sp-sub">fly a Coast Guard pattern</span>
        <button class="sp-toggle" type="button" aria-label="Collapse the search panel">&minus;</button>
      </div>
      <div class="sp-body">
        <div class="sp-row"><b>1</b><span class="sp-target">Pick a buoy in the list, move the clock to when it was "reported", then:</span></div>
        <button type="button" class="sp-use">Use the selected buoy at this time</button>
        <div class="sp-row"><b>2</b><span class="sp-base">Base: not placed</span></div>
        <button type="button" class="sp-place">Place the base (click the map)</button>
        <div class="sp-row"><b>3</b>
          <label>Pattern <select class="sp-pattern">${patterns}</select></label></div>
        <div class="sp-row"><b>4</b>
          <label>Target <select class="sp-kind">${targets}</select></label></div>
        <div class="sp-actions">
          <button type="button" class="sp-fly on">Fly</button>
          <button type="button" class="sp-pause" disabled>Pause</button>
          <button type="button" class="sp-reset">Reset</button>
          <select class="sp-speed" aria-label="Playback speed">${speeds}</select>
        </div>
        <div class="sp-phase" aria-live="polite"></div>
        <div class="sp-result" aria-live="polite"></div>
        <div class="sp-key">
          <span><i class="sp-sw" style="background:${SEARCH_COLOURS.helicopter}"></i>track, and the strip it sees (to scale, 185.2 m)</span>
          <span><i class="sp-dot" style="background:${SEARCH_COLOURS.marker}"></i>marker: drifts with the current</span>
          <span><i class="sp-dot" style="background:${SEARCH_COLOURS.target}"></i>the real buoy</span>
          <span><i class="sp-ring"></i>datum: where drift says it will be</span>
        </div>
      </div>`;
    L.DomEvent.disableClickPropagation(root);
    L.DomEvent.disableScrollPropagation(root);

    this._root = root;
    const q = (sel) => root.querySelector(sel);
    this._els = {
      target: q('.sp-target'), base: q('.sp-base'), place: q('.sp-place'), pattern: q('.sp-pattern'),
      kind: q('.sp-kind'), fly: q('.sp-fly'), pause: q('.sp-pause'), speed: q('.sp-speed'),
      phase: q('.sp-phase'), result: q('.sp-result'),
    };
    q('.sp-use').addEventListener('click', () => this._h.onUseBuoy());
    this._els.place.addEventListener('click', () => this._h.onPlaceBase());
    this._els.fly.addEventListener('click', () => this._h.onFly(this.choices()));
    this._els.pause.addEventListener('click', () => this._h.onPause());
    q('.sp-reset').addEventListener('click', () => this._h.onReset());
    q('.sp-toggle').addEventListener('click', () => {
      const collapsed = root.classList.toggle('collapsed');
      q('.sp-toggle').innerHTML = collapsed ? '+' : '&minus;';
    });
    return root;
  },

  /** What the user has chosen, for onFly. */
  choices() {
    return {
      patternKind: this._els.pattern.value,
      targetKind: this._els.kind.value,
      speedX: Number(this._els.speed.value),
    };
  },

  speedX() {
    return Number(this._els.speed.value);
  },

  setTarget(text) { this._els.target.textContent = text; },
  setBase(text) { this._els.base.textContent = text; },

  setPlacing(on) {
    this._els.place.classList.toggle('on', on);
    this._els.place.textContent = on ? 'Click the map to place the base…' : 'Place the base (click the map)';
  },

  setFlying(state) {
    // state: 'idle' | 'flying' | 'paused' | 'done'
    this._els.fly.disabled = state === 'flying' || state === 'paused';
    this._els.pause.disabled = !(state === 'flying' || state === 'paused');
    this._els.pause.textContent = state === 'paused' ? 'Resume' : 'Pause';
  },

  setPhase(text) { this._els.phase.textContent = text; },
  setResult(html) { this._els.result.innerHTML = html; },
});
