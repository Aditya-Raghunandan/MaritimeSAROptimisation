/**
 * farBuoys.js -- the game's windows whose real buoy ends up far from its cloud, which the game
 * does not offer (Aditya, 15 Oct: a buoy just outside the cloud is fine, one far, far away
 * only confuses a visitor).
 *
 * Measured 15 Oct on the published scenarios/v4 bundles (`scripts/buoy_vs_cloud.py`): at the end
 * of each window, when the reveal shows the real buoy, its distance from the cloud's centre over
 * the radius holding 90 % of the cloud. 325 of the 330 windows are within 2.2 of that radius,
 * most under 1 (inside it). S10's buoy leaves its cloud from 30 min on: 2.5 to 6.0 radii away,
 * 17.5 km at 1 h with the nearest of 10,000 particles 12.7 km off. A window above FAR_RATIO is
 * left out; a buoy with no window left is not on the map. The paper's benchmark keeps all 55:
 * this is only what the game lets a visitor pick.
 *
 * Measured again 22 Oct on scenarios/v6 (the calibrated cloud, rvsc; `buoy_vs_cloud.py --version
 * v6 --noise rvsc`): 317 of 330 windows within 2.46 radii. Above 2.5: S10 from 30 min (3.1 to 7.3),
 * S12 at every arrival (2.83 to 2.86; v6's cloud moves 0.7 as far as the forecast and this buoy
 * kept pace with the forecast) and S02 at 2 and 3 h (2.51, 2.57).
 */

export const FAR_RATIO = 2.5;
export const FAR_WINDOWS = {
  S02: ['rvsc_2h', 'rvsc_3h'],
  S10: ['rvc_0.5h', 'rvc_0.75h', 'rvc_1h', 'rvc_2h', 'rvc_3h',
    'rvsc_0.5h', 'rvsc_0.75h', 'rvsc_1h', 'rvsc_2h', 'rvsc_3h'],
  S12: ['rvsc_0.25h', 'rvsc_0.5h', 'rvsc_0.75h', 'rvsc_1h', 'rvsc_2h', 'rvsc_3h'],
};

/** The scenario index with the far windows taken out, and any buoy left with none. */
export function playable(index, far = FAR_WINDOWS) {
  const scenarios = (index?.scenarios ?? [])
    .map((s) => ({ ...s, windows: (s.windows ?? []).filter((w) => !far[s.scenario]?.includes(w)) }))
    .filter((s) => s.windows.length > 0);
  return { ...index, scenarios };
}
