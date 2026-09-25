/**
 * tileFallback.js -- where a missing basemap tile is drawn from (#76). Pure, so it is
 * tested without a browser; tiles.js is the Leaflet layer that uses it.
 */

/** How many zooms out a missing tile may look: 13 serves up to 19 at its coarsest. */
export const MAX_FALLBACK = 6;

/**
 * How to draw tile `coords` from its ancestor `up` zooms out: the ancestor's address,
 * the size to draw it at, the offset that lines the right part of it up with this
 * tile's square, and the clip (CSS inset: top, right, bottom, left) that shows only it.
 */
export function fallbackPlacement(coords, up, tileSize = 256) {
  const scale = 2 ** up;
  const parent = { x: Math.floor(coords.x / scale), y: Math.floor(coords.y / scale), z: coords.z - up };
  const dx = coords.x - parent.x * scale;
  const dy = coords.y - parent.y * scale;
  return {
    parent,
    size: tileSize * scale,
    marginLeft: 0 - dx * tileSize,   // not -dx * size: that is -0 on the first column
    marginTop: 0 - dy * tileSize,
    clip: [dy, scale - dx - 1, scale - dy - 1, dx].map((n) => n * tileSize),
  };
}
