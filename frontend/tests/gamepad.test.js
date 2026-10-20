import { describe, expect, it } from 'vitest';
import { PadEvents, nearestThatWay, padHeading, padStick, randomName } from '../src/gamepad.js';
import { nickname } from '../src/leaderboard.js';

/** A standard-layout pad (what Chrome gave the stand's PS5 pad, 14 Oct) with these buttons held. */
function pad({ axes = [0, 0, 0, 0], held = [] } = {}) {
  return { connected: true, mapping: 'standard', axes, buttons: Array.from({ length: 18 }, (_, i) => ({ pressed: held.includes(i), value: held.includes(i) ? 1 : 0 })) };
}
const nav = (p) => ({ getGamepads: () => [null, p] });

describe('the controller as a heading', () => {
  it('reads either stick, up the stick being north', () => {
    expect(padHeading(pad({ axes: [0, -1, 0, 0] }))).toBeCloseTo(0);
    expect(padHeading(pad({ axes: [1, 0, 0, 0] }))).toBeCloseTo(90);
    expect(padHeading(pad({ axes: [0, 0, 0, 1] }))).toBeCloseTo(180);   // the right stick, pulled down
  });

  it('ignores a resting stick, and falls back to the d-pad, diagonals included', () => {
    expect(padHeading(pad({ axes: [0.1, -0.2, 0, 0] }))).toBeNull();
    expect(padStick(pad({ axes: [0.1, -0.2, 0, 0] }))).toEqual([0, 0]);
    expect(padHeading(pad({ held: [12, 15] }))).toBeCloseTo(45);        // up and right
    expect(padHeading(pad({ held: [0] }))).toBeNull();                  // ✕ alone points nowhere
  });
});

describe('the controller as menu presses', () => {
  it('fires once per press, and repeats a held direction like a key', () => {
    const got = [];
    let state = pad({ held: [0] });
    const ev = new PadEvents((n) => got.push(n), { getGamepads: () => [state] });
    ev.poll(0);
    ev.poll(16);                                    // still held: no second ✕
    state = pad({ held: [13] });
    ev.poll(100);                                   // down
    ev.poll(300);                                   // held, too soon to repeat
    ev.poll(500);                                   // repeats
    ev.poll(700);
    expect(got).toEqual(['cross', 'down', 'down', 'down']);
  });

  it('treats a stick pushed well over as the d-pad, but not a nudge', () => {
    const got = [];
    let state = pad({ axes: [0.4, 0, 0, 0] });
    const ev = new PadEvents((n) => got.push(n), { getGamepads: () => [state] });
    expect(ev.poll(0)).toBe(true);                  // touched, though not a menu move
    state = pad({ axes: [0.9, 0, 0, 0] });
    ev.poll(20);
    expect(got).toEqual(['right']);
    expect(new PadEvents(() => {}, nav(null)).poll(0)).toBe(false);
  });
});

describe('moving the highlight', () => {
  const box = (x, y, w = 100, h = 40) => ({ getBoundingClientRect: () => ({ left: x, top: y, width: w, height: h }) });
  it('goes to the nearest control that way, preferring one in line', () => {
    const start = box(0, 0);
    const right = box(200, 0);
    const below = box(0, 100);
    const farDiagonal = box(150, 90);
    const els = [start, right, below, farDiagonal];
    expect(nearestThatWay(start, els, 'right')).toBe(right);
    expect(nearestThatWay(start, els, 'down')).toBe(below);
    expect(nearestThatWay(start, els, 'up')).toBeNull();
  });
});

it('makes names the board keeps as they are', () => {
  for (let k = 0; k < 200; k += 1) {
    const name = randomName();
    expect(name.length).toBeLessThanOrEqual(16);
    expect(nickname(name)).toBe(name);
  }
});
