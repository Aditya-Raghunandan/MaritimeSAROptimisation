/**
 * searchPanel.js -- the controls for a search (issues #64, #68, #69, #71).
 *
 * Written as the questions the Coast Guard answers, in the order it answers them, each
 * with a line saying what the choice actually changes -- because the first version's
 * "Target: person in water (2 % of wind)" left a reader asking whether it gave the
 * MARKER wind. It does not: it changes where the helicopter is sent.
 *
 *   1  Who is missing?                 a buoy from the list, at the clock's time
 *   2  Where does the helicopter start? a base, placed on the map
 *   3  How does the Coast Guard search? Expanding Square or Sector Search
 *   4  Where will it have drifted to?   the datum's drift: current, or current + wind
 *
 * IT HAS TO FIT (#71). With all four steps, a result, the fly-yourself controls and a key
 * open at once it was taller than a laptop's map, and it scrolled -- "the menus still
 * clip". So: two tabs, the Coast Guard search and flying it yourself, only one open; once
 * a search is flying the four steps fold into a one-line summary with a Change button;
 * and the longer help and the key fold away until asked for.
 *
 * ONE BUTTON (#73). There used to be three play buttons that meant different things:
 * "Fly the search" started one, the panel's Play did nothing until one ran, and the time
 * bar's ▶ played the site's hours. Now one primary button runs the whole life of a search
 * -- Fly the search, Pause, Resume, Replay -- and the time bar's ▶ does the same thing
 * while the Search view is open.
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


export const SearchPanel = L.Control.extend({
  options: { position: 'topleft' },

  /** @param {{onUseBuoy, onChangeBuoy, onPlaceBase, onPrimary, onSpawn, onEdit, onReset}} handlers */
  initialize(handlers) {
    this._h = handlers;
    this._root = null;
    this._tab = 'search';
    this._state = 'idle';
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
    // Side by side, with one line of help for the one chosen: two wrapped descriptions
    // were what pushed the set-up past the bottom of a 900 px screen (#71).
    const patterns = Object.entries(PATTERNS).map(([k, p], n) => `
      <label class="sp-pill" title="${p.label}: ${p.help}"><input type="radio" name="sp-pattern" value="${k}"${n === 0 ? ' checked' : ''}>
        <span>${p.short}</span></label>`).join('');
    const firstHelp = Object.values(PATTERNS)[0].help;
    const targets = Object.entries(TARGETS).map(([k, t], n) => `
      <label class="sp-choice"><input type="radio" name="sp-kind" value="${k}"${n === 0 ? ' checked' : ''}>
        <span>${t.label}</span></label>`).join('');
    const speeds = SPEEDS.map((s) => `<option value="${s.x}"${s.x === 45 ? ' selected' : ''}>${s.label}</option>`).join('');

    root.innerHTML = `
      <div class="sp-head">
        <span class="sp-title">Search</span>
        <span class="sp-sub">against a real buoy</span>
        <button class="sp-toggle" type="button" aria-label="Collapse the search panel">&minus;</button>
      </div>
      <div class="sp-body">
        <div class="sp-tabs" role="tablist">
          <button type="button" class="sp-tab on" data-tab="search" role="tab">Coast Guard search</button>
          <button type="button" class="sp-tab" data-tab="fly" role="tab">Fly it yourself</button>
        </div>

        <div class="sp-pane" data-pane="search">
          <div class="sp-summary" hidden>
            <dl class="sp-summary-text sp-facts"></dl>
            <button type="button" class="sp-edit">Change the set-up</button>
          </div>
          <div class="sp-setup">
            <section>
              <h4><b>1</b>Who is missing?</h4>
              <p class="sp-target"><span class="sp-target-text">Pick a buoy in the Drifters list and move the clock to when it is reported missing.</span>
                <button type="button" class="sp-link sp-change" hidden>change</button></p>
              <button type="button" class="sp-use">Use the selected buoy at this time</button>
            </section>
            <section>
              <h4><b>2</b>Where does the helicopter start?</h4>
              <p class="sp-base"><span class="sp-base-text">Base: not placed</span>
                <button type="button" class="sp-link sp-move" hidden>move</button></p>
              <button type="button" class="sp-place">Place the base on the map</button>
            </section>
            <section>
              <h4><b>3</b>How does the Coast Guard search?</h4>
              <div class="sp-pills">${patterns}</div>
              <p class="sp-help sp-pattern-help">${firstHelp}</p>
            </section>
            <section>
              <h4><b>4</b>Where will it be when they arrive?</h4>
              ${targets}
              <p class="sp-help sp-drogue" hidden></p>
              <details class="sp-more"><summary>What does this change?</summary>
                <p>The <b>datum</b>: where the helicopter flies to and drops its marker. The marker
                  itself always drifts with the current alone, and the real buoy goes where it really
                  went. A drifter buoy is built to follow the water, so for one, "with the current" is
                  the honest choice.</p></details>
            </section>
          </div>
        </div>

        <div class="sp-pane" data-pane="fly" hidden>
          <button type="button" class="sp-spawn">Spawn a helicopter on the map</button>
          <label class="sp-follow"><input type="checkbox" class="sp-follow-box" checked>
            Follow it close up, with the sea drawn</label>
          <p class="sp-help">Click where it should appear, then steer with <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd>
            or the arrow keys; two keys fly a diagonal. 90 kt, and it sees 92.6 m either side.
            If a buoy is chosen, flying over it finds it.</p>
        </div>

        <div class="sp-actions">
          <button type="button" class="sp-fly on">Fly the search</button>
          <button type="button" class="sp-reset">Reset</button>
          <select class="sp-speed" aria-label="Playback speed">${speeds}</select>
        </div>
        <div class="sp-phase" aria-live="polite"></div>
        <div class="sp-result" aria-live="polite"></div>
        <details class="sp-keywrap"><summary>Key</summary>
          <div class="sp-key">
            <span><i class="sp-sw" style="background:${SEARCH_COLOURS.helicopter}"></i>track, and the strip it sees (to scale)</span>
            <span><i class="sp-dash"></i>predicted drift, ending at the datum</span>
            <span><i class="sp-sw" style="background:${SEARCH_COLOURS.target}"></i>where the buoy really went</span>
            <span><i class="sp-dot" style="background:${SEARCH_COLOURS.marker}"></i>marker: drifts with the current</span>
            <span><i class="sp-wave"></i>close up: a drawn sea, drifting weed and the odd animal; the Close up key says what each is</span>
          </div>
        </details>
      </div>`;
    L.DomEvent.disableClickPropagation(root);
    L.DomEvent.disableScrollPropagation(root);

    this._root = root;
    const q = (sel) => root.querySelector(sel);
    this._body = q('.sp-body');
    this._els = {
      target: q('.sp-target-text'), use: q('.sp-use'), change: q('.sp-change'), base: q('.sp-base-text'),
      move: q('.sp-move'),
      place: q('.sp-place'), fly: q('.sp-fly'), speed: q('.sp-speed'), drogue: q('.sp-drogue'),
      follow: q('.sp-follow-box'),
      phase: q('.sp-phase'), result: q('.sp-result'), spawn: q('.sp-spawn'),
      summary: q('.sp-summary'), summaryText: q('.sp-summary-text'), setup: q('.sp-setup'),
    };
    this._els.use.addEventListener('click', () => this._h.onUseBuoy());
    this._els.change.addEventListener('click', () => this._h.onChangeBuoy());
    this._els.place.addEventListener('click', () => this._h.onPlaceBase());
    this._els.move.addEventListener('click', () => this._h.onPlaceBase());
    this._els.fly.addEventListener('click', () => this._h.onPrimary(this.choices()));
    this._els.spawn.addEventListener('click', () => this._h.onSpawn());
    // Changing the set-up means a new search: the loaded one is cleared first.
    q('.sp-edit').addEventListener('click', () => { this._h.onEdit(); this.expandSetup(); });
    q('.sp-reset').addEventListener('click', () => this._h.onReset());
    for (const tab of root.querySelectorAll('.sp-tab')) {
      tab.addEventListener('click', () => this.showTab(tab.dataset.tab));
    }
    for (const d of root.querySelectorAll('details')) d.addEventListener('toggle', () => this.fit());
    for (const r of root.querySelectorAll('input[name="sp-pattern"]')) {
      r.addEventListener('change', () => { q('.sp-pattern-help').textContent = PATTERNS[r.value].help; });
    }
    q('.sp-toggle').addEventListener('click', () => {
      const collapsed = root.classList.toggle('collapsed');
      q('.sp-toggle').innerHTML = collapsed ? '+' : '&minus;';
    });
  },

  /** Keep the body inside the map: if it ever must scroll, it scrolls rather than clips. */
  fit() {
    if (!this._root || !this._mapRef) return;
    const map = this._mapRef.getContainer().getBoundingClientRect();
    const top = this._body.getBoundingClientRect().top;
    // 44 px short of the bottom: the north arrow and the scale bar live in that corner.
    this._body.style.maxHeight = `${Math.max(140, map.bottom - top - 44)}px`;
  },

  /** 'search' or 'fly'. */
  showTab(name) {
    for (const tab of this._root.querySelectorAll('.sp-tab')) tab.classList.toggle('on', tab.dataset.tab === name);
    for (const pane of this._root.querySelectorAll('.sp-pane')) pane.hidden = pane.dataset.pane !== name;
    this._tab = name;
    this._syncPrimary();
    this.fit();
  },

  /** Which tab is open. A method, not a getter: L.Class.extend copies getters as values. */
  currentTab() {
    return this._tab;
  },

  /** Whether "Follow it close up" is ticked. */
  follow() {
    return this._els.follow.checked;
  },

  /** A line under step 4 about the chosen buoy's drogue, or nothing; `kind` colours its rule. */
  setDrogueHint(text, kind = null) {
    this._els.drogue.hidden = !text;
    this._els.drogue.textContent = text ?? '';
    if (kind) this._els.drogue.dataset.kind = kind;
    this.fit();
  },

  /** Fold the four steps into a short list of what was chosen, while a search flies. */
  collapseSetup(facts) {
    const dl = this._els.summaryText;
    dl.replaceChildren();
    for (const [k, v] of facts) {
      const dt = document.createElement('dt');
      dt.textContent = k;
      const dd = document.createElement('dd');
      dd.textContent = v;
      // A space between them, so the list still reads as text: "Buoy 3002...".
      dl.append(dt, ' ', dd, ' ');
    }
    this._els.summary.hidden = false;
    this._els.setup.hidden = true;
    this.fit();
  },

  expandSetup() {
    this._els.summary.hidden = true;
    this._els.setup.hidden = false;
    this.fit();
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

  /**
   * The target line. Once one is chosen the step shrinks to that line and a small
   * "change" link: a done step does not need a full-width button any more (#71).
   */
  setTarget(text, chosen) {
    this._els.target.textContent = text;
    this._els.use.hidden = Boolean(chosen);
    this._els.change.hidden = !chosen;
    this.fit();
  },

  /** The base line; placed, the step shrinks to it and a "move" link. */
  setBase(text, placed = false) {
    this._els.base.textContent = text;
    this._els.place.hidden = placed;
    this._els.move.hidden = !placed;
    this.fit();
  },

  /** Which placement a map click will make: 'base', 'spawn' or null. */
  setPlacing(mode) {
    this._els.place.classList.toggle('on', mode === 'base');
    this._els.place.textContent = mode === 'base' ? 'Click the map to place the base…' : 'Place the base on the map';
    this._els.move.textContent = mode === 'base' ? 'click the map…' : 'move';
    this._els.spawn.classList.toggle('on', mode === 'spawn');
    this._els.spawn.textContent = mode === 'spawn' ? 'Click the map where it should appear…' : 'Spawn a helicopter on the map';
  },

  /** state: 'idle' | 'armed' | 'playing' | 'paused' | 'done' -- the primary button follows it. */
  setFlying(state) {
    this._state = state;
    this._syncPrimary();
  },

  _syncPrimary() {
    const b = this._els.fly;
    const s = this._state;
    const labels = {
      idle: 'Fly the search', armed: 'Press a key to take off', playing: 'Pause', paused: 'Resume', done: 'Replay',
    };
    b.textContent = labels[s] ?? 'Fly the search';
    b.disabled = s === 'armed';
    // Idle on the fly-it-yourself tab there is nothing to start: Spawn does that.
    b.hidden = s === 'idle' && this._tab === 'fly';
  },

  setPhase(text) { this._els.phase.textContent = text; },

  setResult(html) {
    this._els.result.innerHTML = html;
    // A result can carry its own folded reasoning (#80); opening it must refit the panel.
    for (const d of this._els.result.querySelectorAll('details')) d.addEventListener('toggle', () => this.fit());
    this.fit();
    // The answer is what the search was for; bring it into view if the panel is scrolled.
    if (html) this._els.result.scrollIntoView({ block: 'nearest' });
  },
});
