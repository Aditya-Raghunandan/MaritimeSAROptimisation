/**
 * style.js -- how a field's marks are styled. Pure arithmetic, no imports.
 *
 * Split from quiver.js for the same reason geo.js is split from the map code:
 * it imports nothing, so it is testable without a DOM. Leaflet touches `window`
 * at module load, so anything importing it cannot be unit-tested in this
 * project's node test environment.
 *
 * COLOUR. One hue, light to dark, which is what a magnitude encoding takes --
 * the white-yellow-orange-red rainbow the previous library used is the thing to
 * avoid. Orange rather than the default blue because the default basemap is
 * Esri Ocean and a blue ramp on blue bathymetry is invisible; orange is both
 * the documented second sequential hue and the complement of the surface.
 * Anchored on the documented categorical orange `#eb6834`, and checked on the
 * gate that applies to a sequential ramp -- monotonic lightness (OKLab L
 * 0.933 -> 0.446, strictly decreasing) across a 14.9 deg hue spread.
 *
 * Speed is encoded TWICE, as colour and as arrow length. That redundancy is
 * deliberate: it survives colour-blindness, greyscale printing in the report,
 * and a busy basemap showing through.
 */

/** One hue, light -> dark. See the module docstring for the validation. */
export const SPEED_RAMP = ['#fde3d3', '#f9b98f', '#f28a54', '#eb6834', '#c04a1c', '#8c3410'];

/** Target on-screen spacing between arrows, px. Below this they overlap into mush. */
const TARGET_SPACING_PX = 26;

/**
 * Colour for a speed, as a step of the ramp.
 *
 * Stepped rather than continuously interpolated so the ramp reads as a small
 * number of distinguishable levels — a continuous gradient over thin marks on
 * a textured basemap is not readable, and cannot be keyed in a legend.
 */
export function speedColour(speed, maxSpeed, ramp = SPEED_RAMP) {
  if (!Number.isFinite(speed) || maxSpeed <= 0) return ramp[0];
  const t = Math.min(Math.max(speed / maxSpeed, 0), 1);
  return ramp[Math.min(ramp.length - 1, Math.floor(t * ramp.length))];
}

/**
 * How many grid cells to skip so arrows land about TARGET_SPACING_PX apart.
 *
 * Without this the field is unreadable at high zoom (arrows on top of each
 * other) and invisible at low zoom (one arrow per hundred pixels). Returns at
 * least 1.
 */
export function decimation(cellSpacingPx, target = TARGET_SPACING_PX) {
  if (!Number.isFinite(cellSpacingPx) || cellSpacingPx <= 0) return 1;
  return Math.max(1, Math.round(target / cellSpacingPx));
}


/**
 * Arrow length in pixels for a speed. Shared by the map and the key.
 *
 * SQUARE ROOT, NOT LINEAR, and not the linear-times-1.6 this started as.
 * That version multiplied by 1.6 to make light winds visible, which clamped
 * everything above ~62 % of the scale to the same length: with maxSpeed 15,
 * 12, 17 and 22 m/s all drew at 22 px. Length stopped encoding anything in
 * exactly the range that matters most, since leeway is a fraction of speed and
 * the strong-wind end is where it stops being negligible. It was visible in
 * the key, which is one of the things a key is for.
 *
 * sqrt keeps a light breeze long enough to see while leaving the top of the
 * range distinguishable, and it saturates only AT maxSpeed rather than well
 * below it. MIN_ARROW_PX keeps a near-calm cell as a visible mark instead of a
 * dot that reads as missing data.
 */
export function arrowLength(speed, maxSpeed, maxPx) {
  if (!Number.isFinite(speed) || !(maxSpeed > 0)) return MIN_ARROW_PX;
  const t = Math.min(Math.max(speed / maxSpeed, 0), 1);
  return Math.max(MIN_ARROW_PX, maxPx * Math.sqrt(t));
}

/** Shortest drawn arrow, px. Below this it reads as absent rather than calm. */
export const MIN_ARROW_PX = 4;

/** Longest drawn arrow, px. Beyond this neighbouring arrows cross. */
export const MAX_ARROW_PX = 22;
