/**
 * tiles.js -- basemap tiles that never say "Map data not yet available" (#76).
 *
 * Esri's imagery and ocean services answer a zoom they hold no data for with a grey
 * placeholder tile, not an error, and over open water that is most of the close-up
 * zooms: measured 25 Sep, World_Imagery stops at zoom 13 over the Gulf Stream and
 * World_Ocean_Base at 10 in the Sargasso. Close up the Search view was a wall of
 * "Map data not yet available" squares.
 *
 * Asked with `blankTile=false`, the same services answer 404 instead, and a 404 can be
 * caught. The tile is then drawn from its parent one zoom out -- scaled up, and clipped
 * to its own square -- and from the grandparent if that is missing too. Where Esri does
 * hold the zoom, near a coast, the real tile loads and nothing changes.
 */

import L from 'leaflet';

import { MAX_FALLBACK, fallbackPlacement } from './tileFallback.js';

export const FallbackTileLayer = L.TileLayer.extend({
  options: { maxFallback: MAX_FALLBACK },

  createTile(coords, done) {
    const tile = L.TileLayer.prototype.createTile.call(this, coords, done);
    tile._fallback = { coords: { x: coords.x, y: coords.y, z: coords.z }, up: 0 };
    return tile;
  },

  _tileOnError(done, tile, e) {
    const f = tile._fallback;
    // A tile Leaflet has already thrown away (zoomed past) is left alone.
    if (f && tile.parentNode && f.up < this.options.maxFallback && f.coords.z - f.up > 0) {
      f.up += 1;
      const p = fallbackPlacement(f.coords, f.up, this.getTileSize().x);
      Object.assign(tile.style, {
        width: `${p.size}px`,
        height: `${p.size}px`,
        marginLeft: `${p.marginLeft}px`,
        marginTop: `${p.marginTop}px`,
        clipPath: `inset(${p.clip.map((v) => `${v}px`).join(' ')})`,
      });
      tile.src = this._urlAt(p.parent);
      return;
    }
    L.TileLayer.prototype._tileOnError.call(this, done, tile, e);
  },

  /** A tile's address at any zoom: Leaflet's own getTileUrl always uses the map's. */
  _urlAt(c) {
    const data = { r: L.Browser.retina ? '@2x' : '', s: this._getSubdomain(c), x: c.x, y: c.y, z: c.z };
    return L.Util.template(this._url, L.Util.extend(data, this.options));
  },
});

/** An Esri tile service that 404s where it has no data, so the gap can be filled. */
export function esriTiles(service, options) {
  return new FallbackTileLayer(
    `https://server.arcgisonline.com/ArcGIS/rest/services/${service}/MapServer/tile/{z}/{y}/{x}?blankTile=false`,
    options,
  );
}
