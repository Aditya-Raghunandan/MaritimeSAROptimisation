/**
 * beaufort.js -- what a wind speed in m/s actually means.
 *
 * Pure, no imports, testable without a DOM.
 *
 * WHY THIS IS HERE AND NOT DECORATION. "3.8 m/s" is a number; "Force 3, gentle
 * breeze, scattered whitecaps, about 0.6 m" is a sea a reader can picture. This
 * project's whole question is whether a person in the water can be found, and
 * the sea state is the bridge between the forcing data and that question:
 *
 *   - It is what a searcher actually sees out of the aircraft window.
 *   - It drives DETECTION, which is R4/R5's sweep width. A head in 0.3 m
 *     wavelets and the same head in 4 m swell are not the same search problem,
 *     and the 185 m sweep width this project quotes is a calm-water figure.
 *     That is a limitation the report has to state, and the map is where it
 *     first becomes visible.
 *
 * Thresholds are the WMO Beaufort scale at the standard 10 m reference height,
 * which is exactly what ERA5's `u10`/`v10` are, so no height correction is
 * needed. Wave heights are the WMO open-sea significant-height guides and are
 * indicative only -- real sea state depends on fetch and duration, not just on
 * the instantaneous wind, which is why they are labelled "about".
 */

/**
 * force      Beaufort number
 * max        upper bound of the band, m/s (exclusive); Infinity for force 12
 * name       the WMO name
 * sea        what the surface looks like
 * waveM      indicative significant wave height, metres, open sea
 */
export const BEAUFORT = [
  { force: 0,  max: 0.5,  name: 'Calm',           sea: 'sea like a mirror',                    waveM: 0 },
  { force: 1,  max: 1.6,  name: 'Light air',      sea: 'ripples, no foam crests',              waveM: 0.1 },
  { force: 2,  max: 3.4,  name: 'Light breeze',   sea: 'small wavelets, glassy crests',        waveM: 0.2 },
  { force: 3,  max: 5.5,  name: 'Gentle breeze',  sea: 'large wavelets, scattered whitecaps',  waveM: 0.6 },
  { force: 4,  max: 8.0,  name: 'Moderate breeze', sea: 'small waves, frequent whitecaps',     waveM: 1.0 },
  { force: 5,  max: 10.8, name: 'Fresh breeze',   sea: 'moderate waves, many whitecaps',       waveM: 2.0 },
  { force: 6,  max: 13.9, name: 'Strong breeze',  sea: 'large waves, foam crests everywhere',  waveM: 3.0 },
  { force: 7,  max: 17.2, name: 'Near gale',      sea: 'sea heaps up, foam blown in streaks',  waveM: 4.0 },
  { force: 8,  max: 20.8, name: 'Gale',           sea: 'moderately high waves, spindrift',     waveM: 5.5 },
  { force: 9,  max: 24.5, name: 'Strong gale',    sea: 'high waves, dense foam, spray',        waveM: 7.0 },
  { force: 10, max: 28.5, name: 'Storm',          sea: 'very high waves, sea white with foam', waveM: 9.0 },
  { force: 11, max: 32.7, name: 'Violent storm',  sea: 'exceptionally high waves',             waveM: 11.5 },
  { force: 12, max: Infinity, name: 'Hurricane force', sea: 'air filled with foam and spray',  waveM: 14.0 },
];

/** The band a wind speed falls in. Clamps rather than returning undefined. */
export function beaufort(speedMs) {
  if (!Number.isFinite(speedMs) || speedMs < 0) return BEAUFORT[0];
  return BEAUFORT.find((b) => speedMs < b.max) ?? BEAUFORT[BEAUFORT.length - 1];
}

/**
 * Whether a person in the water is realistically findable in this sea.
 *
 * Deliberately coarse -- three states, not a number -- because this project has
 * NOT measured a sea-state/detection curve and must not imply that it has. The
 * 185 m sweep width it quotes is a calm-water figure, so what this can honestly
 * say is "the quoted sweep width is optimistic here", not by how much.
 *
 * R4/R5 own the real detection model. Until one exists this is a caveat with a
 * colour, and it is labelled as one.
 */
export function detectionOutlook(speedMs) {
  const f = beaufort(speedMs).force;
  if (f <= 3) return { level: 'good', text: 'Sweep width near its calm-water value' };
  if (f <= 5) return { level: 'degraded', text: 'Whitecaps cut detection; sweep width optimistic' };
  return { level: 'poor', text: 'A head in the water is largely hidden by the sea' };
}

/** "Force 3 · gentle breeze" */
export function describe(speedMs) {
  const b = beaufort(speedMs);
  return `Force ${b.force} · ${b.name.toLowerCase()}`;
}
