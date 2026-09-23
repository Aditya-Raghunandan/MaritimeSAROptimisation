/**
 * drifterLayer.js -- the drifter test data on the map (issue #50).
 *
 * Three things are drawn, from three sources:
 *
 *   dots    every buoy seen in the current WINDOW, hollow, where it was first seen.
 *           From the index alone, so they cost nothing to show.
 *   live    every buoy in the water at the current MOMENT, solid, at its position
 *           now, coloured by its drogue state now. Needs that buoy's track file.
 *   trail   the SELECTED buoy's path so far, drawn behind it as the clock runs,
 *           coloured per step by drogue state and dashed where it is more than 3 h
 *           from a real GPS fix.
 *
 * The decisions -- which buoys, where, which colour -- are in drifters.js and
 * tested there. This file only draws them.
 */

import L from 'leaflet';

import {
  decodeTrack, describeBuoy, inWindow, positionAt, spanningNow, trailUpTo, undroguedAt,
} from './drifters.js';

/**
 * The drogue state's two colours. Neither is the wind's amber, the current's cyan or
 * the drift's green, so a drifter never reads as one of the forcing fields. Pink for
 * undrogued because it is the one the wind pushes, which is what makes it interesting.
 */
export const DROGUE_COLOURS = { drogued: '#dfe7ef', undrogued: '#ff5fa2' };

/** Buoys in the water at once whose tracks will be fetched for live markers. */
const LIVE_LIMIT = 40;

/**
 * Track files, each fetched once, the most recently used kept.
 *
 * A file is small (median 36 kB, largest 520 kB, measured 23 Sep) but there are 326
 * of them, so they are fetched when a buoy is selected or in the water, never all.
 */
export class TrackCache {
  constructor(base, { capacity = 60, fetchFn = (...args) => fetch(...args) } = {}) {
    this.base = base.replace(/\/$/, '');
    this.capacity = capacity;
    this._fetch = fetchFn;
    this._tracks = new Map();
    this._inflight = new Map();
  }

  /** The decoded track if it is already here, else undefined. Never fetches. */
  peek(id) {
    return this._tracks.get(id);
  }

  /** The decoded track, fetched if needed. Two calls for one buoy make one fetch. */
  async get(entry) {
    if (this._tracks.has(entry.id)) return this._tracks.get(entry.id);
    if (this._inflight.has(entry.id)) return this._inflight.get(entry.id);
    const job = (async () => {
      const res = await this._fetch(`${this.base}/${entry.file}`);
      if (!res.ok) throw new Error(`track ${entry.id}: HTTP ${res.status}`);
      const track = decodeTrack(await res.json());
      this._tracks.set(entry.id, track);
      while (this._tracks.size > this.capacity) {
        this._tracks.delete(this._tracks.keys().next().value);
      }
      return track;
    })().finally(() => this._inflight.delete(entry.id));
    this._inflight.set(entry.id, job);
    return job;
  }
}

function tooltipHtml(entry) {
  const d = describeBuoy(entry);
  return `<b>${d.title}</b><br>${d.dates}<br>${d.tier}`
    + (d.sealed ? `<br><span class="dp-badge">sealed</span> ${d.sealed.replace(/^sealed: /, '')}` : '');
}

export const DrifterLayer = L.Layer.extend({
  /**
   * @param {object[]} entries  the prepared index (drifters.js::prepareIndex)
   * @param {TrackCache} cache
   * @param {{onSelect?: Function, onHover?: Function}} options
   */
  initialize(entries, cache, options = {}) {
    this._entries = entries;
    this._byId = new Map(entries.map((e) => [e.id, e]));
    this._cache = cache;
    this._onSelect = options.onSelect ?? (() => {});
    this._onHover = options.onHover ?? (() => {});
    this._win = [-Infinity, Infinity];
    this._t = 0;
    this._selected = null;
    this._highlight = null;
    this._live = new Map();
    this._pending = new Set();
    this.aliveNow = 0;
  },

  onAdd(map) {
    this._map = map;
    this._dotGroup = L.layerGroup().addTo(map);
    this._trailGroup = L.layerGroup().addTo(map);
    this._liveGroup = L.layerGroup().addTo(map);
    this._renderDots();
    this._renderLive();
    this._renderTrail();
  },

  onRemove(map) {
    for (const g of [this._dotGroup, this._trailGroup, this._liveGroup]) map.removeLayer(g);
    this._live.clear();
    this._map = null;
  },

  /** The window the dots follow, in epoch ms, `end` exclusive. */
  setWindow(winStart, winEnd) {
    this._win = [winStart, winEnd];
    if (this._map) this._renderDots();
  },

  /** The moment the live markers and the trail follow, in epoch ms. */
  setTime(when) {
    this._t = when;
    if (this._map) {
      this._renderLive();
      this._renderTrail();
    }
  },

  /** Choose the buoy whose path is drawn. Fetches its track if needed. */
  select(id) {
    this._selected = id;
    const entry = this._byId.get(id);
    if (entry && !this._cache.peek(id)) {
      this._cache.get(entry).then(() => {
        if (this._map && this._selected === id) { this._renderTrail(); this._renderLive(); }
      }).catch(() => this.fire('trackerror', { id }));
    }
    if (this._map) { this._renderDots(); this._renderTrail(); this._renderLive(); }
  },

  /** Emphasise one buoy's dot, as the list is hovered. */
  highlight(id) {
    if (this._highlight === id) return;
    this._highlight = id;
    if (this._map) this._renderDots();
  },

  _renderDots() {
    this._dotGroup.clearLayers();
    for (const e of inWindow(this._entries, ...this._win)) {
      const picked = e.id === this._selected || e.id === this._highlight;
      const dot = L.circleMarker([e.lat0, e.lon0], {
        radius: picked ? 7 : 4.5,
        weight: picked ? 2.5 : 1.5,
        color: e.id === this._selected ? '#ffffff' : '#c3c2b7',
        opacity: 0.9,
        fill: true,
        fillOpacity: 0.05,
        // Sealed buoys get a broken ring: the track may be looked at, the engine
        // may not be compared against it before the frozen evaluation run (D025).
        dashArray: e.sealed ? '2 3' : null,
        className: `drifter-dot drifter-${e.id}`,
        // A click here picks the buoy; it must not also open the point panel.
        bubblingMouseEvents: false,
      });
      dot.bindTooltip(tooltipHtml(e), { direction: 'top', className: 'drifter-tip' });
      dot.on('click', () => this._onSelect(e));
      dot.on('mouseover', () => this._onHover(e.id));
      dot.on('mouseout', () => this._onHover(null));
      dot.addTo(this._dotGroup);
    }
  },

  _request(entry) {
    if (this._pending.has(entry.id)) return;
    this._pending.add(entry.id);
    this._cache.get(entry)
      .then(() => { if (this._map) this._renderLive(); })
      .catch(() => {})
      .finally(() => this._pending.delete(entry.id));
  },

  _renderLive() {
    const now = this._t;
    // The selected buoy first, so it is never the one the limit leaves out.
    const alive = spanningNow(this._entries, now)
      .sort((a, b) => (b.id === this._selected) - (a.id === this._selected))
      .slice(0, LIVE_LIMIT);
    const drawn = new Set();
    for (const e of alive) {
      const track = this._cache.peek(e.id);
      if (!track) { this._request(e); continue; }
      const pos = positionAt(track, now);
      if (!pos) continue;
      drawn.add(e.id);
      const colour = undroguedAt(track, now) ? DROGUE_COLOURS.undrogued : DROGUE_COLOURS.drogued;
      const style = {
        radius: e.id === this._selected ? 6.5 : 4.5,
        color: '#0b0b0b', weight: 1, fillColor: colour, fillOpacity: 1,
        className: 'drifter-live',
        bubblingMouseEvents: false,
      };
      let marker = this._live.get(e.id);
      if (!marker) {
        marker = L.circleMarker(pos, style).addTo(this._liveGroup);
        marker.bindTooltip(tooltipHtml(e), { direction: 'top', className: 'drifter-tip' });
        marker.on('click', () => this._onSelect(e));
        this._live.set(e.id, marker);
      } else {
        marker.setLatLng(pos);
        marker.setStyle(style);
      }
    }
    for (const [id, marker] of this._live) {
      if (!drawn.has(id)) { this._liveGroup.removeLayer(marker); this._live.delete(id); }
    }
    if (drawn.size !== this.aliveNow) {
      this.aliveNow = drawn.size;
      this.fire('alive', { count: drawn.size });
    }
  },

  _renderTrail() {
    this._trailGroup.clearLayers();
    const track = this._selected ? this._cache.peek(this._selected) : null;
    if (!track) return;
    for (const piece of trailUpTo(track, this._t)) {
      L.polyline(piece.points, {
        color: piece.undrogued ? DROGUE_COLOURS.undrogued : DROGUE_COLOURS.drogued,
        weight: 3,
        opacity: 0.95,
        dashArray: piece.far ? '5 6' : null,
        className: 'drifter-trail',
        interactive: false,
      }).addTo(this._trailGroup);
    }
  },
});
