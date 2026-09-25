/**
 * cornerGuard.js -- the two right-hand corners never draw over each other (#76).
 *
 * Leaflet places each map corner on its own. The top-right stack -- the view presets,
 * the layer list, the drifter list -- grows down; the bottom-right one -- the compass
 * and the legends -- grows up; neither knows the other is there. On a shorter screen
 * they met: the drifter list lay over the legends, and the compass over the Surface
 * current key (screenshots, 25 Sep).
 *
 * So whenever either stack changes size the gap between them is measured, and if it is
 * short something gives way, in this order:
 *
 *   1. the drifter LIST gets shorter, down to four rows. It scrolls anyway, so a shorter
 *      list loses nothing but rows on screen;
 *   2. the legends fold to their titles, the top one first -- except one the viewer has
 *      opened or closed by hand, which is left as they set it.
 *
 * Every pass starts again from the natural layout, so what gives way depends only on
 * the space there is: a legend folded for room opens again when the room comes back.
 */

/** Space kept between the two stacks, px. */
const GAP_PX = 8;

/** Four rows of the drifter list. */
const MIN_LIST_PX = 100;

/** How far the upper stack runs into the lower, px; zero or less is clear. */
export function overlapPx(upperBottom, lowerTop, gap = GAP_PX) {
  return upperBottom + gap - lowerTop;
}

function setFolded(legend, folded) {
  legend.classList.toggle('collapsed', folded);
  const toggle = legend.querySelector('.legend-toggle');
  if (toggle) {
    toggle.textContent = folded ? '+' : '−';
    toggle.setAttribute('aria-label', folded ? 'Expand' : 'Collapse');
  }
}

/** Watch the map's right-hand corners and keep them apart. Returns a function that stops it. */
export function keepCornersApart(map) {
  const root = map.getContainer();
  const top = root.querySelector('.leaflet-top.leaflet-right');
  const bottom = root.querySelector('.leaflet-bottom.leaflet-right');
  if (!top || !bottom || typeof ResizeObserver === 'undefined') return () => {};

  const overlap = () => (top.children.length && bottom.children.length
    ? overlapPx(top.getBoundingClientRect().bottom, bottom.getBoundingClientRect().top) : 0);

  function balance() {
    // The layer list opens on hover and closes again; room is not rearranged for that.
    if (top.querySelector('.leaflet-control-layers-expanded')) return;
    const list = top.querySelector('.dp-list');
    if (list) list.style.maxHeight = '';
    for (const legend of bottom.querySelectorAll('.wind-legend[data-folded-for-room]')) {
      delete legend.dataset.foldedForRoom;
      setFolded(legend, false);
    }

    let over = overlap();
    if (over <= 0) return;
    if (list) {
      const want = Math.max(MIN_LIST_PX, list.clientHeight - over);
      if (want < list.clientHeight) {
        list.style.maxHeight = `${want}px`;
        over = overlap();
      }
    }
    for (const legend of bottom.querySelectorAll('.wind-legend')) {
      if (over <= 0) break;
      if (legend.classList.contains('collapsed') || legend.dataset.setByHand) continue;
      legend.dataset.foldedForRoom = '1';
      setFolded(legend, true);
      over = overlap();
    }
  }

  // A legend the viewer toggles is theirs from then on.
  const onClick = (e) => {
    const legend = e.target.closest && e.target.closest('.legend-toggle') && e.target.closest('.wind-legend');
    if (!legend) return;
    legend.dataset.setByHand = '1';
    delete legend.dataset.foldedForRoom;
  };
  bottom.addEventListener('click', onClick);

  const observer = new ResizeObserver(() => balance());
  for (const el of [root, top, bottom]) observer.observe(el);
  // Controls come and go with the views; a new one changes a corner's size, which the
  // observer sees, but one swapped for another of the same height would not be.
  const mo = new MutationObserver(() => balance());
  mo.observe(top, { childList: true });
  mo.observe(bottom, { childList: true });
  balance();

  return () => {
    observer.disconnect();
    mo.disconnect();
    bottom.removeEventListener('click', onClick);
  };
}
