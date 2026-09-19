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
/*
 * SATURATED ALL THE WAY DOWN, and that is the fix rather than a preference.
 *
 * Both ramps used to START NEAR WHITE -- #fde3d3 for wind, #d7f6ff for current
 * -- and most of the domain is slow most of the time, so most arrows on the
 * map were a pale grey smudge in both products and the wind/current split only
 * existed at speeds that are rare. Hue has to carry the distinction where the
 * data actually lives, which is the bottom third of the range.
 *
 * Still light-to-dark, so magnitude still reads as it did. What changed is
 * that the light end is now a saturated amber rather than an off-white.
 *
 * GENERATED, NOT PICKED. Both ramps are a constant OKLab hue with lightness
 * stepped evenly down and chroma held near the gamut edge -- amber at h = 62
 * deg, cyan at h = 233 deg, L from 0.88 to 0.50. A first attempt at this was
 * hand-picked hex and the style test caught it at 34 deg of hue spread against
 * its 30 deg gate: "one hue, light to dark, not a rainbow" is exactly the
 * property eyeballing a swatch cannot verify.
 */
export const SPEED_RAMP = ['#ffcd8d', '#ffaf62', '#ed9235', '#d27908', '#b0660c', '#8e5311'];

/**
 * The CURRENT's arrow ramp. One hue, light to dark, same rule as the wind's.
 *
 * Cyan against the current's magma raster, which is the same contrast argument
 * that put orange arrows on a viridis raster: the arrows must not be mistaken
 * for the field they sit on. It also puts the two products in opposite colour
 * families at the mark level -- warm arrows are wind, cool arrows are water --
 * which is the distinction that has to survive a projector at ten metres.
 */
export const CURRENT_RAMP = ['#9ae2ff', '#73c9f6', '#48b1e3', '#1e98cb', '#037fad', '#04668c'];

/** Target on-screen spacing between arrows, px. Below this they overlap into mush. */
// Sparser than it was. The arrows used to BE the field and had to cover it;
// now the painted raster carries the pattern and the particles carry the
// motion, so the arrows are an annotation over the top -- a readable value at
// a real cell centre every so often. At 26 px they crowded the streaks into
// mush.
// Raised with MAX_ARROW_PX below. These two move together or not at all: a
// longer arrow at the old spacing overlaps its neighbour, which is the mush
// this constant exists to prevent.
const TARGET_SPACING_PX = 48;

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
// Longer than it was. At 22 px the arrows read as texture rather than as
// values you could take a bearing off, which is what they are there for --
// the raster already carries the pattern and the particles carry the motion,
// so the arrow's one job is to be legible at a real cell centre.
export const MAX_ARROW_PX = 30;
