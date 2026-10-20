/**
 * gamepad.js -- a game controller for the open-day game: it steers the helicopter and works
 * the menus, so a visitor can play with nothing but the pad.
 *
 * Chrome shows a controller to a page only once a button has been pressed or a stick moved
 * with the page open. Asked of Chrome on the stand's laptop, 14 Oct, a PS5 pad over Bluetooth:
 * "DualSense Wireless Controller (STANDARD GAMEPAD Vendor: 054c Product: 0ce6)", mapping
 * "standard", 4 axes, 18 buttons, every one of them reporting. The standard layout, which an
 * Xbox pad shares (A for cross, B for circle):
 *   0 cross   1 circle   2 square   3 triangle   4 L1   5 R1   6 L2   7 R2
 *   8 create  9 options  10 L3      11 R3        12-15 d-pad up, down, left, right
 *   axes: 0, 1 the left stick (x right, y down); 2, 3 the right stick.
 */

import { bearing } from './kinematics.js';

export const BUTTONS = {
  cross: 0, circle: 1, square: 2, triangle: 3, l1: 4, r1: 5, options: 9, up: 12, down: 13, left: 14, right: 15,
};
export const DIRECTIONS = ['up', 'down', 'left', 'right'];
/** How far a stick must lean before it counts: a resting stick drifts a few per cent. */
export const STICK_DEAD = 0.35;
/** For the menus a stick must lean further, so a nudge does not skip two buttons. */
const STICK_MENU = 0.6;
/** A held direction repeats, as a held key does: after this long, then this often. */
const REPEAT_AFTER_MS = 380;
const REPEAT_EVERY_MS = 150;

/** The first controller Chrome is showing, or null. */
export function firstPad(nav = globalThis.navigator) {
  const pads = typeof nav?.getGamepads === 'function' ? nav.getGamepads() : [];
  for (const pad of pads ?? []) if (pad && pad.connected !== false) return pad;
  return null;
}

const pressed = (pad, i) => Boolean(pad.buttons[i]?.pressed);

/** A stick as [right, up], each -1 to 1, or [0, 0] inside the dead zone. The left stick, else the right. */
export function padStick(pad = firstPad()) {
  if (!pad) return [0, 0];
  for (const [ix, iy] of [[0, 1], [2, 3]]) {
    const x = pad.axes[ix] ?? 0;
    const y = pad.axes[iy] ?? 0;
    if (Math.hypot(x, y) > STICK_DEAD) return [x, -y];
  }
  return [0, 0];
}

/**
 * Where the controller points, as a compass heading (0 north, 90 east), or null: either stick,
 * else the d-pad (two arrows for a diagonal).
 */
export function padHeading(pad = firstPad()) {
  if (!pad) return null;
  const [x, y] = padStick(pad);
  if (x || y) return bearing(x, y);
  const e = (pressed(pad, BUTTONS.right) ? 1 : 0) - (pressed(pad, BUTTONS.left) ? 1 : 0);
  const n = (pressed(pad, BUTTONS.up) ? 1 : 0) - (pressed(pad, BUTTONS.down) ? 1 : 0);
  return e || n ? bearing(e, n) : null;
}

/** What the pad is doing now, by name: the four directions (d-pad or left stick) and the buttons. */
function stateOf(pad) {
  const x = pad.axes[0] ?? 0;
  const y = pad.axes[1] ?? 0;
  const s = {};
  for (const [name, i] of Object.entries(BUTTONS)) s[name] = pressed(pad, i);
  s.up ||= y < -STICK_MENU;
  s.down ||= y > STICK_MENU;
  s.left ||= x < -STICK_MENU;
  s.right ||= x > STICK_MENU;
  return s;
}

/**
 * The pad as events, for the menus: `onEvent(name)` once per press ('cross', 'circle', 'up' …),
 * a held direction repeating like a key. poll() once a frame; it returns whether the pad is
 * being touched at all, so the stand knows somebody is there.
 */
export class PadEvents {
  constructor(onEvent, nav = globalThis.navigator) {
    Object.assign(this, { onEvent, nav, prev: {}, repeatAt: {} });
  }

  poll(now = performance.now()) {
    const pad = firstPad(this.nav);
    if (!pad) return false;
    const s = stateOf(pad);
    for (const [name, on] of Object.entries(s)) {
      if (on && !this.prev[name]) {
        this.repeatAt[name] = now + REPEAT_AFTER_MS;
        this.onEvent(name);
      } else if (on && DIRECTIONS.includes(name) && now >= this.repeatAt[name]) {
        this.repeatAt[name] = now + REPEAT_EVERY_MS;
        this.onEvent(name);
      }
    }
    this.prev = s;
    const [sx, sy] = padStick(pad);
    return Boolean(sx || sy || pad.buttons.some((b) => b.pressed));
  }
}

/**
 * The control a direction moves to, from `from`, among `els`: the nearest one that way, a step
 * sideways costing twice a step along (so "down" from a button finds the one under it, not one
 * far down and across). Null when there is nothing that way.
 */
export function nearestThatWay(from, els, dir) {
  const r0 = from.getBoundingClientRect();
  const c0 = [r0.left + r0.width / 2, r0.top + r0.height / 2];
  let best = null;
  let bestCost = Infinity;
  for (const el of els) {
    if (el === from) continue;
    const r = el.getBoundingClientRect();
    const dx = r.left + r.width / 2 - c0[0];
    const dy = r.top + r.height / 2 - c0[1];
    const along = { right: dx, left: -dx, down: dy, up: -dy }[dir];
    const across = dir === 'left' || dir === 'right' ? Math.abs(dy) : Math.abs(dx);
    if (along <= 4) continue;
    const cost = along + 2 * across;
    if (cost < bestCost) {
      bestCost = cost;
      best = el;
    }
  }
  return best;
}

const NAMES = ['Captain Kelp', 'Salty Dog', 'Sea Otter', 'Storm Petrel', 'Coral Queen', 'Wave Rider',
  'Red Herring', 'Deep Diver', 'Flying Fish', 'Sea Biscuit', 'Gulf Streamer', 'Barnacle', 'Moby', 'Lighthouse'];

/** A name for the board that needs no keyboard: a sea creature and a number, at most 16 letters. */
export function randomName(rnd = Math.random) {
  return `${NAMES[Math.floor(rnd() * NAMES.length)]} ${10 + Math.floor(rnd() * 90)}`;
}
