/**
 * drifterPanel.js -- the drifter list, its search, and its key (issue #50).
 *
 * The list holds the same buoys as the map's dots: those seen in the current window,
 * sorted by first fix. The search is the one way out of that window -- it searches
 * EVERY buoy -- so a buoy named in the paper can be found whatever the slider shows.
 * Hovering a row lights its dot; clicking a row or a dot does the same thing.
 *
 * It sits top-right under the layer control: the left side belongs to the point
 * panel and the bottom-right to the legends.
 */

import L from 'leaflet';

import { DROGUE_COLOURS } from './drifterLayer.js';
import { day, lengthDays, searchBuoys } from './drifters.js';

/** More rows than this and the list is a wall; the search narrows it instead. */
const MAX_ROWS = 200;

export const DrifterPanel = L.Control.extend({
  options: { position: 'topright' },

  /**
   * @param {object[]} entries  the prepared index
   * @param {{onSelect: Function, onHover: Function}} handlers
   */
  initialize(entries, handlers) {
    this._entries = entries;
    this._handlers = handlers;
    this._visible = [];
    this._alive = 0;
    this._selected = null;
    this._query = '';
  },

  onAdd() {
    const root = L.DomUtil.create('div', 'drifter-panel');
    root.innerHTML = `
      <div class="dp-head">
        <span class="dp-title">Drifters</span>
        <span class="dp-sub">real buoys: the test data</span>
        <button class="dp-toggle" type="button" aria-label="Collapse the drifter list">&minus;</button>
      </div>
      <div class="dp-body">
        <input class="dp-search" type="search" aria-label="Search every buoy"
               placeholder="Search every buoy: ID, or a date like 2021-03">
        <div class="dp-status"></div>
        <ul class="dp-list"></ul>
        <div class="dp-key">
          <span><i class="dp-sw" style="background:${DROGUE_COLOURS.drogued}"></i>drogued</span>
          <span><i class="dp-sw" style="background:${DROGUE_COLOURS.undrogued}"></i>undrogued</span>
          <span><i class="dp-sw dp-dash"></i>&gt; 3 h from a real fix</span>
          <span><b class="dp-badge">sealed</b> look, but do not debug on it</span>
        </div>
      </div>`;
    L.DomEvent.disableClickPropagation(root);
    L.DomEvent.disableScrollPropagation(root);

    this._root = root;
    this._list = root.querySelector('.dp-list');
    this._status = root.querySelector('.dp-status');
    root.querySelector('.dp-search').addEventListener('input', (e) => {
      this._query = e.target.value;
      this._render();
    });
    root.querySelector('.dp-toggle').addEventListener('click', () => {
      this.setCollapsed(!root.classList.contains('collapsed'));
    });
    this._render();
    return root;
  },

  /** Fold the list to its header, or open it again. */
  setCollapsed(on) {
    if (!this._root) return;
    this._root.classList.toggle('collapsed', on);
    this._root.querySelector('.dp-toggle').innerHTML = on ? '+' : '&minus;';
  },

  /** The buoys seen in the current window, which the list shows when not searching. */
  setVisible(entries) {
    this._visible = entries;
    this._render();
  },

  /** How many buoys are in the water right now, for the status line. */
  setAlive(count) {
    this._alive = count;
    this._renderStatus();
  },

  /**
   * Mark the selected row. Classes only, never a rebuild: this runs inside the row's
   * own click, and a rebuild detached the clicked row mid-event, after which Leaflet
   * could not see the click had started inside the panel and treated it as a click
   * on the map ("Outside the data box.").
   */
  select(id) {
    this._selected = id;
    if (!this._list) return;
    for (const li of this._list.children) li.classList.toggle('selected', li.dataset.id === id);
  },

  highlight(id) {
    if (!this._list) return;
    for (const li of this._list.children) li.classList.toggle('hot', li.dataset.id === id);
  },

  _rows() {
    return this._query.trim() ? searchBuoys(this._entries, this._query) : this._visible;
  },

  _renderStatus() {
    if (!this._status) return;
    const rows = this._rows();
    let text;
    if (this._query.trim()) {
      text = `${rows.length} buoy${rows.length === 1 ? '' : 's'} match, across the whole archive`;
    } else if (!this._visible.length) {
      text = 'No buoys in this window. Widen the span, or search.';
    } else {
      const now = this._alive
        ? `${this._alive} in the water now`
        : 'none in the water at this moment';
      text = `${this._visible.length} buoy${this._visible.length === 1 ? '' : 's'} in this window · ${now}`;
    }
    this._status.textContent = text;
  },

  _render() {
    if (!this._list) return;
    this._renderStatus();
    this._list.innerHTML = '';
    for (const e of this._rows().slice(0, MAX_ROWS)) {
      const li = document.createElement('li');
      li.dataset.id = e.id;
      if (e.id === this._selected) li.classList.add('selected');
      li.innerHTML = `<i class="dp-tier dp-${e.tier}" title="${e.tier}"></i>`
        + `<span class="dp-id">${e.id}</span>`
        // First seen and how long, not two full dates: two dates pushed the sealed
        // badge off the end of the row.
        + `<span class="dp-dates" title="${day(e.startMs)} → ${day(e.endMs)}">`
        + `${day(e.startMs)} · ${lengthDays(e)} d</span>`
        + (e.sealed ? '<b class="dp-badge">sealed</b>' : '');
      li.addEventListener('click', (ev) => { ev.stopPropagation(); this._handlers.onSelect(e); });
      li.addEventListener('mouseenter', () => this._handlers.onHover(e.id));
      li.addEventListener('mouseleave', () => this._handlers.onHover(null));
      this._list.appendChild(li);
    }
  },
});
