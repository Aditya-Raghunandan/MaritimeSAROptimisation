/**
 * labels.js -- map labels that never sit on each other (issue #86).
 *
 * The search labels its points -- real buoy, datum, marker, last known position, predicted
 * drift -- and each used a fixed offset. When the buoy, the marker and the path bunch up,
 * the chips landed on one another ("sometimes all the text ends up on each other").
 *
 * So each label is placed in turn, most important first, at the first of a ring of
 * positions round its point that overlaps no label already placed and no marker on the
 * map; failing that, a wider ring, out to four; failing that, a label unimportant enough to
 * spare steps aside for the frame, and any other goes wherever it overlaps least. A label keeps
 * the position it had last frame while that is still clear, so labels do not flicker
 * between spots as the points drift.
 *
 * Pure: boxes in, offsets out. searchLayer.js measures the chips and draws them.
 */

/** Where a w x h label may sit round its point, as the offset of its top-left corner. */
export function candidates(w, h) {
  const ring = (r) => [
    [r, -h / 2],                 // right
    [-r - w, -h / 2],            // left
    [-w / 2, -r - h],            // above
    [-w / 2, r],                 // below
    [r * 0.8, -r * 0.8 - h],     // upper right
    [r * 0.8, r * 0.8],          // lower right
    [-r * 0.8 - w, -r * 0.8 - h],// upper left
    [-r * 0.8 - w, r * 0.8],     // lower left
  ];
  return [...ring(11), ...ring(30), ...ring(55), ...ring(85)];
}

/** Overlap area of two boxes {x, y, w, h}, with a margin of `pad` px round each. */
export function overlap(a, b, pad = 2) {
  const x = Math.min(a.x + a.w + pad, b.x + b.w + pad) - Math.max(a.x - pad, b.x - pad);
  const y = Math.min(a.y + a.h + pad, b.y + b.h + pad) - Math.max(a.y - pad, b.y - pad);
  return x > 0 && y > 0 ? x * y : 0;
}

/**
 * Place labels. Each label is { id, x, y, w, h, priority } -- its point and its size, px --
 * lower priority numbers going first. `obstacles` are boxes to keep clear (the markers
 * themselves); `bounds` is the visible area, {x, y, w, h}; `previous` maps id to the
 * candidate index it used last time. A label of priority `spareFrom` or more that finds no
 * clear spot is hidden rather than laid over another. Returns id -> { dx, dy, index, hidden }.
 */
export function placeLabels(labels, obstacles = [], bounds = null, previous = {}, spareFrom = Infinity) {
  const placed = [...obstacles];
  const out = {};
  const ordered = [...labels].sort((a, b) => a.priority - b.priority);
  for (const l of ordered) {
    const offs = candidates(l.w, l.h);
    const order = offs.map((_, k) => k);
    const prev = previous[l.id];
    if (Number.isInteger(prev) && prev < offs.length) order.unshift(prev);
    let best = null;
    for (const k of order) {
      const [dx, dy] = offs[k];
      const box = { x: l.x + dx, y: l.y + dy, w: l.w, h: l.h };
      const clash = placed.reduce((sum, o) => sum + overlap(box, o), 0);
      let cost = clash;
      // Off screen counts against a spot, but less than sitting on another label.
      if (bounds) {
        const inside = overlap(box, bounds, 0);
        cost += (l.w * l.h - inside) * 0.5;
      }
      if (!best || cost < best.cost) best = { k, dx, dy, box, cost, clash };
      if (cost === 0) break;
    }
    if (best.clash > 0 && l.priority >= spareFrom) {
      out[l.id] = { dx: best.dx, dy: best.dy, index: best.k, hidden: true };
      continue;
    }
    placed.push(best.box);
    out[l.id] = { dx: best.dx, dy: best.dy, index: best.k, hidden: false };
  }
  return out;
}
