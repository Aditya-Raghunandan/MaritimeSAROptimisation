/**
 * searchPanel.js -- the controls for a search (issues #64, #68, #69).
 *
 * Written as the questions the Coast Guard answers, in the order it answers them, each
 * with one line saying what the choice actually changes -- because the first version's
 * "Target: person in water (2 % of wind)" left a reader asking whether it gave the
 * MARKER wind. It does not: it changes where the helicopter is sent.
 *
 *   1  Who is missing?                 a buoy from the list, at the clock's time
 *   2  Where does the helicopter start? a base, placed on the map
 *   3  How does the Coast Guard search? Expanding Square or Sector Search
 *   4  Where will it have drifted to?   the datum's drift: current, or current + wind
 *   then Fly, or instead spawn a helicopter and fly it yourself.
 *
 * The DOM is built ONCE and kept. Leaflet calls onAdd every time the view is switched
 * back to, and rebuilding here made the panel say "Base: not placed" over a search that
 * had a base. The panel only collects choices and shows state; searchView.js does the work.
 */

import L from 'leaflet';

import { PATTERNS } from './patterns.js';
import { TARGETS } from './searchRun.js';
import { SEARCH_COLOURS } from './searchLayer.js';

/** Playback, said as what it means: how much search time passes each real second. */
export const SPEEDS = [
  { x: 15, label: '15 s of search per second' },
  { x: 45, label: '45 s per second (the window in a minute)' },
  { x: 120, label: '2 min per second' },
  { x: 600, label: '10 min per second (quick look)' },
];

/** The speed's label, for the time bar. */
export function speedLabel(x) {
  const hit = SPEEDS.find((s) => s.x === x);
  return hit ? hit.label.replace(/ \(.*\)$/, '').replace(' of search', '') : `${x} s per second`;
}

const PATTERN_HELP = {
  expanding_square: 'a square spiral out from the marker: even coverage, growing outwards',
  sector_search: 'spokes through the marker: very dense near it, sparse at the edge',
};

export const SearchPanel = L.Control.extend({
  options: { position: 'topleft' },

  /** @param {{onUseBuoy, onChangeBuoy, onPlaceBase, onFly, onSpawn, onPlayPause, onReset}} handlers */
  initialize(handlers) {
    this._h = handlers;
    this._root = null;
  },

  onAdd(map) {
    if (!this._root) this._build();
    this._mapRef = map;
    this._onResize = () => this.fit();
    window.addEventListener('resize', this._onResize);
    // After Leaflet has placed it, so its top is known.
    setTimeout(() => this.fit(), 0);
    return this._root;
  },

  onRemove() {
    window.removeEventListener('resize', this._onResize);
  },

  _build() {
    const root = L.DomUtil.create('div', 'search-panel');
    const patterns = Object.entries(PATTERNS).map(([k, p], n) => `
      <label class="sp-choice"><input type="radio" name="sp-pattern" value="${k}"${n === 0 ? ' checked' : ''}>
        <span><b>${p.label}</b> <i>${PATTERN_HELP[k] ?? ''}</i></span></label>`).join('');
    const targets = Object.entries(TARGETS).map(([k, t], n) => `
      <label class="sp-choice"><input type="radio" name="sp-kind" value="${k}"${n === 0 ? ' checked' : ''}>
        <span>${t.label}</span></label>`).join('');
    const speeds = SPEEDS.map((s) => `<option value="${s.x}"${s.x === 45 ? ' selected' : ''}>${s.label}</option>`).join('');

    root.innerHTML = `
      <div class="sp-head">
        <span class="sp-title">Search</span>
        <span class="sp-sub">the Coast Guard's own method, against a real buoy</span>
        <button class="sp-toggle" type="button" aria-label="Collapse the search panel">&minus;</button>
      </div>
      <div class="sp-body">
        <section>
          <h4><b>1</b>Who is missing?</h4>
          <p class="sp-target">Pick a buoy in the Drifters list and move the clock to when it is reported missing.</p>
          <div class="sp-row-buttons">
            <button type="button" class="sp-use">Use the selected buoy at this time</button>
            <button type="button" class="sp-change" hidden>Choose another buoy</button>
          </div>
        </section>
        <section>
          <h4><b>2</b>Where does the helicopter start?</h4>
          <p class="sp-base">Base: not placed</p>
          <button type="button" class="sp-place">Place the base on the map</button>
        </section>
        <section>
          <h4><b>3</b>How does the Coast Guard search?</h4>
          ${patterns}
        </section>
        <section>
          <h4><b>4</b>Where will it have drifted to by the time they arrive?</h4>
          ${targets}
          <p class="sp-help">This sets the <b>datum</b>, where the helicopter flies to and drops its
            marker. The marker itself always drifts with the current alone, and the real buoy goes
            where it really went.</p>
        </section>
        <div class="sp-actions">
          <button type="button" class="sp-fly on">Fly the search</button>
          <button type="button" class="sp-pause" disabled>Pause</button>
          <button type="button" class="sp-reset">Reset</button>
        </div>
        <label class="sp-speed-row">Playback <select class="sp-speed" aria-label="Playback speed">${speeds}</select></label>
        <div class="sp-phase" aria-live="polite"></div>
        <div class="sp-result" aria-live="polite"></div>
        <section class="sp-yourself">
          <h4>Or fly it yourself</h4>
          <button type="button" class="sp-spawn">Spawn a helicopter on the map</button>
          <p class="sp-help">Click where it should appear, then steer with <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd>
            or the arrow keys; two keys fly a diagonal. It flies at 90 kt and sees 92.6 m either side.
            If a buoy is chosen, passing over it finds it.</p>
        </section>
        <div class="sp-key">
          <span><i class="sp-sw" style="background:${SEARCH_COLOURS.helicopter}"></i>track, and the strip it sees (to scale)</span>
          <span><i class="sp-ring"></i>datum: where drift predicts the buoy</span>
          <span><i class="sp-dot" style="background:${SEARCH_COLOURS.marker}"></i>marker: drifts with the current</span>
          <span><i class="sp-dot" style="background:${SEARCH_COLOURS.target}"></i>the real buoy</span>
        </div>
      </div>`;
    L.DomEvent.disableClickPropagation(root);
    L.DomEvent.disableScrollPropagation(root);

    this._root = root;
    const q = (sel) => root.querySelector(sel);
    this._body = q('.sp-body');
    this._els = {
      target: q('.sp-target'), use: q('.sp-use'), change: q('.sp-change'), base: q('.sp-base'),
      place: q('.sp-place'), fly: q('.sp-fly'), pause: q('.sp-pause'), speed: q('.sp-speed'),
      phase: q('.sp-phase'), result: q('.sp-result'), spawn: q('.sp-spawn'),
    };
    this._els.use.addEventListener('click', () => this._h.onUseBuoy());
    this._els.change.addEventListener('click', () => this._h.onChangeBuoy());
    this._els.place.addEventListener('click', () => this._h.onPlaceBase());
    this._els.fly.addEventListener('click', () => this._h.onFly(this.choices()));
    this._els.pause.addEventListener('click', () => this._h.onPlayPause());
    this._els.spawn.addEventListener('click', () => this._h.onSpawn());
    q('.sp-reset').addEventListener('click', () => this._h.onReset());
    q('.sp-toggle').addEventListener('click', () => {
      const collapsed = root.classList.toggle('collapsed');
      q('.sp-toggle').innerHTML = collapsed ? '+' : '&minus;';
    });
  },

  /** Keep the body inside the map: it scrolls rather than running under the time bar. */
  fit() {
    if (!this._root || !this._mapRef) return;
    const map = this._mapRef.getContainer().getBoundingClientRect();
    const top = this._body.getBoundingClientRect().top;
    // 40 px short of the bottom: the map's scale bar lives in that corner.
    this._body.style.maxHeight = `${Math.max(140, map.bottom - top - 40)}px`;
  },

  /** What the user has chosen, for onFly. */
  choices() {
    const pick = (name) => this._root.querySelector(`input[name="${name}"]:checked`).value;
    return { patternKind: pick('sp-pattern'), targetKind: pick('sp-kind') };
  },

  speedX() {
    return Number(this._els.speed.value);
  },

  setSpeed(x) {
    this._els.speed.value = String(x);
  },

  /** The target line, and whether one is chosen (which swaps the buttons). */
  setTarget(text, chosen) {
    this._els.target.textContent = text;
    this._els.use.hidden = Boolean(chosen);
    this._els.change.hidden = !chosen;
    this.fit();
  },

  setBase(text) { this._els.base.textContent = text; },

  /** Which placement a map click will make: 'base', 'spawn' or null. */
  setPlacing(mode) {
    this._els.place.classList.toggle('on', mode === 'base');
    this._els.place.textContent = mode === 'base' ? 'Click the map to place the base…' : 'Place the base on the map';
    this._els.spawn.classList.toggle('on', mode === 'spawn');
    this._els.spawn.textContent = mode === 'spawn' ? 'Click the map where it should appear…' : 'Spawn a helicopter on the map';
  },

  /** state: 'idle' | 'armed' | 'playing' | 'paused' | 'done' */
  setFlying(state) {
    this._els.pause.disabled = state === 'idle' || state === 'armed';
    this._els.pause.textContent = state === 'playing' ? 'Pause' : (state === 'done' ? 'Replay' : 'Play');
  },

  setPhase(text) { this._els.phase.textContent = text; },

  setResult(html) {
    this._els.result.innerHTML = html;
    this.fit();
    // The answer is what the search was for; bring it into view if the panel is scrolled.
    if (html) this._els.result.scrollIntoView({ block: 'nearest' });
  },
});
