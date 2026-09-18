/**
 * colormap.js -- continuous colour for a scalar field. Pure, no imports.
 *
 * VIRIDIS, AND THE REASON IS NOT THAT IT LOOKS LIKE THE OTHER WEATHER SITES.
 *
 * Zoom Earth, nullschool and Windy all paint wind speed with a multi-hue ramp,
 * and it is genuinely better than a single hue for this job: a continuous field
 * over a whole ocean is read for STRUCTURE -- where the jet is, where the calm
 * is -- and more hues mean more distinguishable levels across a wide range.
 *
 * The usual objection is that rainbow ramps are bad, and it is a real one: a
 * naive HSV rainbow is non-monotonic in lightness, so it invents banding where
 * the data is smooth and hides differences where it is not. Viridis is the
 * answer to exactly that complaint -- multi-hue, but built so lightness rises
 * strictly from end to end. Checked, not assumed (OKLab L):
 *
 *     #440154 0.285 -> #31688e 0.497 -> #35b779 0.694 -> #fde725 0.918
 *
 * strictly increasing at every step, which is the gate that applies to a
 * sequential ramp. It also survives greyscale, which the report will need, and
 * it is safe for the common colour-vision deficiencies.
 *
 * The ARROWS stay on the one-hue orange ramp in style.js. Different jobs: the
 * raster answers "what is the pattern", the arrows answer "what is the value
 * here", and they are deliberately in different colour families so neither is
 * mistaken for the other.
 */

/** Viridis anchors, evenly spaced over [0, 1]. */
const VIRIDIS = [
  [68, 1, 84], [72, 40, 120], [62, 73, 137], [49, 104, 142], [38, 130, 142],
  [31, 158, 137], [53, 183, 121], [110, 206, 88], [181, 222, 43], [253, 231, 37],
];

/**
 * Colour for a normalised value in [0, 1], as [r, g, b].
 *
 * Linear interpolation between anchors. Values outside the range clamp rather
 * than wrap: a wrapped colour scale makes the strongest wind on the map look
 * like the calmest.
 */
export function viridis(t) {
  if (!Number.isFinite(t)) return null;          // NaN is absent, not zero
  const x = Math.min(Math.max(t, 0), 1) * (VIRIDIS.length - 1);
  const i = Math.min(Math.floor(x), VIRIDIS.length - 2);
  const f = x - i;
  const a = VIRIDIS[i];
  const b = VIRIDIS[i + 1];
  return [
    Math.round(a[0] + f * (b[0] - a[0])),
    Math.round(a[1] + f * (b[1] - a[1])),
    Math.round(a[2] + f * (b[2] - a[2])),
  ];
}

/** `viridis` as a CSS string, for legends and swatches. */
export function viridisCss(t) {
  const c = viridis(t);
  return c ? `rgb(${c[0]}, ${c[1]}, ${c[2]})` : 'transparent';
}

/**
 * Normalise a speed onto [0, 1] for the ramp.
 *
 * SQUARE ROOT, for the same reason the arrows use it. Wind speed over an ocean
 * is strongly skewed -- most of the box sits in the bottom third of the range
 * most of the time -- so a linear ramp spends most of its colours on winds that
 * almost never occur and renders the ordinary sea as one flat dark field. sqrt
 * pushes contrast into the range the data actually occupies.
 */
export function normaliseSpeed(speed, maxSpeed) {
  if (!Number.isFinite(speed) || !(maxSpeed > 0)) return NaN;
  return Math.sqrt(Math.min(Math.max(speed / maxSpeed, 0), 1));
}
