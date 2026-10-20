/**
 * sea.js -- the game's sea: deep blue with small wave marks, under land from Esri's dark map.
 *
 * Esri's Dark Gray map paints water a flat grey (35, 34, 39) and land a lighter one (65-78),
 * measured on its tiles 14 Oct. The game draws its own sea underneath, and a filter in
 * game.html (#landOnly) turns the dark map's water transparent and its land a dark green, so
 * the sea shows through wherever there is water. Over open ocean Esri holds some close-up tiles
 * and not others; with its water see-through that patchwork no longer shows.
 *
 * The wave marks are fixed to the map, not the screen, and never move: in a game about drift,
 * water that seemed to flow would be read as a current.
 */

import L from 'leaflet';
import { esriTiles } from './tiles.js';

export const SEA = '#0a3358';
const WAVE = 'rgba(150, 215, 255, 0.13)';

/** A small 32-bit generator, so each tile's waves are the same every time it is drawn. */
function seeded(x, y, z) {
  let s = (Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791)) >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

export const SeaLayer = L.GridLayer.extend({
  options: { zIndex: 0, waves: 7 },

  createTile(c) {
    const tile = document.createElement('canvas');
    const size = this.getTileSize();
    tile.width = size.x;
    tile.height = size.y;
    const g = tile.getContext('2d');
    g.fillStyle = SEA;
    g.fillRect(0, 0, size.x, size.y);
    const rnd = seeded(c.x, c.y, c.z);
    g.strokeStyle = WAVE;
    g.lineWidth = 1.6;
    g.lineCap = 'round';
    for (let k = 0; k < this.options.waves; k += 1) {
      const w = 7 + 7 * rnd();
      const x = w + rnd() * (size.x - 2 * w);
      const y = 6 + rnd() * (size.y - 12);
      g.beginPath();
      g.moveTo(x - w, y);
      g.quadraticCurveTo(x - w / 2, y - 5, x, y);
      g.quadraticCurveTo(x + w / 2, y - 5, x + w, y);
      g.stroke();
    }
    return tile;
  },
});

/** The sea, then the land on it, on `map`. `options` go to the land layer (Esri's dark map). */
export function addSea(map, options = {}) {
  new SeaLayer().addTo(map);
  return esriTiles('Canvas/World_Dark_Gray_Base', { zIndex: 1, className: 'sea-land', ...options }).addTo(map);
}
