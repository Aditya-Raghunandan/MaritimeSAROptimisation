/**
 * domain.js -- everything outside the study box, pushed into the background.
 *
 * THE PROBLEM THIS SOLVES. The forcing covers 17-36 N, 82-63 W and nothing
 * else, so the painted field is a hard-edged rectangle sitting in the middle of
 * a world map. That reads as broken -- as though the data failed to load
 * outside the square -- when in fact the square IS the study domain and the
 * edge is the most honest thing on the screen.
 *
 * Two ways to fix an edge like that. Feathering it is the wrong one: it makes
 * the boundary look approximate when it is exact, and invites the eye to read
 * the fade as declining confidence rather than as an end of data. So instead
 * the rest of the world is dimmed and the boundary is drawn as a deliberate
 * line. The square stops looking like a hole in the map and starts looking
 * like the subject of it.
 *
 * HOW: one polygon with a hole in it. Leaflet takes an array of rings where the
 * first is the outer boundary and the rest are holes, so a world-sized ring
 * with the study box punched out gives a single filled shape covering
 * everything except the domain. That is cheaper and far more robust than
 * trying to clip or mask a tile layer, and it moves and zooms with the map for
 * free because it is just geometry.
 *
 * The outer ring is clamped to Leaflet's own latitude limits: Mercator sends
 * the poles to infinity, and a polygon reaching 90 N projects to a coordinate
 * the renderer will not accept.
 */

import L from 'leaflet';

/** Web Mercator gives up before the poles; so does Leaflet. */
const MAX_LAT = 85;
const MAX_LON = 180;

export function domainMask(bounds, opts = {}) {
  const sw = bounds.getSouthWest();
  const ne = bounds.getNorthEast();

  const world = [
    [MAX_LAT, -MAX_LON], [MAX_LAT, MAX_LON], [-MAX_LAT, MAX_LON], [-MAX_LAT, -MAX_LON],
  ];
  // The hole. Wound the opposite way round from the outer ring, which is what
  // the even-odd fill rule needs to cut rather than overlay.
  const hole = [
    [sw.lat, sw.lng], [sw.lat, ne.lng], [ne.lat, ne.lng], [ne.lat, sw.lng],
  ];

  const group = L.layerGroup();

  L.polygon([world, hole], {
    // Not interactive: this covers the whole map, and a click on the sea
    // outside the domain should still reach the map rather than hitting a
    // sheet of dimming glass.
    interactive: false,
    stroke: false,
    fillColor: opts.fillColor ?? '#05070a',
    fillOpacity: opts.fillOpacity ?? 0.55,
  }).addTo(group);

  // The boundary itself, drawn so the edge reads as a decision.
  L.rectangle(bounds, {
    interactive: false,
    color: opts.lineColor ?? '#c3c2b7',
    weight: 1,
    opacity: 0.45,
    dashArray: '6,5',
    fill: false,
  }).addTo(group);

  return group;
}

/**
 * A label for the boundary, so the edge says what it is rather than being
 * inferred. Placed at the north-west corner, outside the box, where it sits on
 * the dimmed area instead of over the data.
 */
export function domainLabel(bounds, text) {
  const ne = bounds.getNorthEast();
  const sw = bounds.getSouthWest();
  return L.marker([ne.lat, sw.lng], {
    interactive: false,
    icon: L.divIcon({
      className: '',
      html: `<span class="domain-label">${text}</span>`,
      iconSize: null,
      iconAnchor: [0, 16],
    }),
  });
}
