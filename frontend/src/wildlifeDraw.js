/**
 * wildlifeDraw.js -- the close-up sea's animals, drawn from above (issue #79).
 *
 * Seen the way a helicopter crew sees them: a school of flying fish gliding on stiff
 * blue fins with their shadows racing beneath; a dolphin pod surfacing in turn, each
 * leaving a V of wake; a turtle paddling up for a breath; and, in winter, a humpback --
 * white flippers glowing under the water, a blow, a long dark back, the fluke raised as
 * it dives, and the smooth "footprint" it leaves on the sea. At night the wakes and
 * splashes glow faintly, as this water's plankton does.
 *
 * NOT TO SCALE, like the helicopter icon: a dolphin true to scale is one pixel at zoom 16.
 * Sizes grow gently with zoom instead. Each animal is pinned to the water where it
 * appeared and carried with it; its own swimming is measured in body lengths.
 *
 * `drawSighting(ctx, sighting, ageS, view)` draws one sighting `ageS` real seconds in.
 * The view supplies `anchor` (its screen point), `size(base)`, `night`, `sun`
 * ({ up, dx, dy }: how high, and the screen direction towards it) and `wind` (screen
 * direction the wind blows to). Randomness comes from the sighting's seed, so a
 * sighting looks the same every frame.
 */

import { seededRandom } from './closeUp.js';

const TAU = Math.PI * 2;

function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
function smooth(e0, e1, x) { const t = clamp01((x - e0) / (e1 - e0)); return t * t * (3 - 2 * t); }

/** 0 -> 1 over the first `fin` s, 1 -> 0 over the last `fout` s of a `dur` s visit. */
function presence(age, dur, fin = 0.9, fout = 1.4) {
  return smooth(0, fin, age) * (1 - smooth(dur - fout, dur, age));
}

/** Soft focus for anything under the water, where the canvas can blur. */
function underwater(ctx, px) {
  if ('filter' in ctx) ctx.filter = `blur(${px.toFixed(1)}px)`;
}
function dry(ctx) {
  if ('filter' in ctx) ctx.filter = 'none';
}

/** An expanding ring on the water, fading as it grows. */
function ring(ctx, x, y, r, alpha, colour) {
  if (alpha <= 0.01 || r <= 0) return;
  ctx.beginPath();
  ctx.ellipse(x, y, r, r * 0.92, 0, 0, TAU);
  ctx.strokeStyle = colour.replace('A', alpha.toFixed(3));
  ctx.lineWidth = Math.max(0.8, r * 0.08);
  ctx.stroke();
}

/** A burst of spray droplets. */
function spray(ctx, x, y, r, alpha, rand, n = 9, colour = 'rgba(240,250,255,A)') {
  if (alpha <= 0.01) return;
  ctx.fillStyle = colour.replace('A', alpha.toFixed(3));
  for (let k = 0; k < n; k += 1) {
    const a = rand() * TAU;
    const d = r * (0.3 + 0.7 * rand());
    ctx.beginPath();
    ctx.arc(x + Math.cos(a) * d, y + Math.sin(a) * d, Math.max(0.6, r * 0.09 * (0.5 + rand())), 0, TAU);
    ctx.fill();
  }
}

function glowColour(night) {
  return night > 0.5 ? 'rgba(110,255,215,A)' : 'rgba(235,248,255,A)';
}

/* ------------------------------------------------------------------ flying fish */

function fishBody(ctx, L, spread) {
  const w = L * 0.13;
  // Wings: the long pectoral fins held out stiff, blue and see-through.
  ctx.fillStyle = 'rgba(120,175,255,0.42)';
  for (const side of [-1, 1]) {
    ctx.beginPath();
    ctx.moveTo(side * w * 0.5, -L * 0.16);
    ctx.quadraticCurveTo(side * L * 0.46 * spread, -L * 0.12, side * L * 0.42 * spread, L * 0.16);
    ctx.quadraticCurveTo(side * L * 0.25 * spread, L * 0.08, side * w * 0.4, L * 0.04);
    ctx.closePath();
    ctx.fill();
  }
  ctx.strokeStyle = 'rgba(190,220,255,0.6)';
  ctx.lineWidth = 0.6;
  for (const side of [-1, 1]) {
    ctx.beginPath();
    ctx.moveTo(side * w * 0.5, -L * 0.14);
    ctx.lineTo(side * L * 0.44 * spread, L * 0.1);
    ctx.stroke();
  }
  // Body: dark blue back, silver sides.
  const g = ctx.createLinearGradient(-w, 0, w, 0);
  g.addColorStop(0, '#c9dbe8');
  g.addColorStop(0.5, '#1b4f86');
  g.addColorStop(1, '#c9dbe8');
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.moveTo(0, -L * 0.5);
  ctx.quadraticCurveTo(w, -L * 0.3, w * 0.7, L * 0.25);
  ctx.lineTo(w * 1.3, L * 0.5);            // forked tail
  ctx.lineTo(0, L * 0.34);
  ctx.lineTo(-w * 1.3, L * 0.5);
  ctx.lineTo(-w * 0.7, L * 0.25);
  ctx.quadraticCurveTo(-w, -L * 0.3, 0, -L * 0.5);
  ctx.fill();
  // Eye.
  ctx.fillStyle = '#0b1a2a';
  ctx.beginPath();
  ctx.arc(0, -L * 0.4, Math.max(0.5, L * 0.03), 0, TAU);
  ctx.fill();
}

function drawFlyingFish(ctx, s, age, v) {
  const rand = seededRandom(s.seed);
  const L = v.size(18);
  const h = (s.headingDeg * Math.PI) / 180;
  const fwd = [Math.sin(h), -Math.cos(h)];
  const side = [-fwd[1], fwd[0]];
  const glow = glowColour(v.night);
  for (let i = 0; i < s.count; i += 1) {
    const delay = rand() * 1.8;
    const glideS = 1.3 + rand() * 0.9;
    const hops = rand() < 0.4 ? 2 : 1;
    const turn = (rand() - 0.5) * 0.3;
    const lateral = (rand() - 0.5) * 7 * L;
    const back = (rand() - 0.5) * 5 * L;
    const speed = 7 * L;                       // px per second: about 15 m/s for a 30 cm fish
    const t = age - delay;
    const x0 = v.anchor[0] + side[0] * lateral + fwd[0] * back;
    const y0 = v.anchor[1] + side[1] * lateral + fwd[1] * back;
    const dir = [Math.sin(h + turn), -Math.cos(h + turn)];
    const flightS = glideS * hops + 0.3 * (hops - 1);
    // Splash where it left the water, and where each hop came down.
    const events = [0];
    for (let k = 1; k <= hops; k += 1) events.push(glideS * k + 0.3 * (k - 1));
    for (const e of events) {
      const since = t - e;
      if (since < 0 || since > 1.4) continue;
      const d = Math.min(t, e) * speed;
      const ex = x0 + dir[0] * Math.min(d, speed * flightS);
      const ey = y0 + dir[1] * Math.min(d, speed * flightS);
      ring(ctx, ex, ey, L * (0.2 + since * 0.55), 0.35 * (1 - since / 1.4), glow);
      if (since < 0.45) spray(ctx, ex, ey, L * (0.4 + since * 1.4), 0.8 * (1 - since / 0.45), seededRandom(s.seed + i * 31 + e * 7), 8, glow);
    }
    if (t < 0 || t > flightS) continue;
    // Which hop, and how high: a low arc above the water.
    let u = t;
    let hop = 0;
    while (hop < hops - 1 && u > glideS + 0.3) { u -= glideS + 0.3; hop += 1; }
    if (u > glideS) continue;                  // skimming between hops
    const lift = Math.sin(Math.PI * (u / glideS));
    const x = x0 + dir[0] * speed * t;
    const y = y0 + dir[1] * speed * t;
    // Its shadow on the sea, pushed away from the sun as it rises.
    if (v.sun.up > 0.05) {
      const off = lift * L * 1.4;
      ctx.save();
      ctx.translate(x - v.sun.dx * off, y - v.sun.dy * off);
      ctx.rotate(h + turn);
      underwater(ctx, 1.2);
      ctx.fillStyle = `rgba(0,15,30,${(0.35 * v.sun.up).toFixed(3)})`;
      ctx.beginPath();
      ctx.ellipse(0, 0, L * 0.5, L * 0.5, 0, 0, TAU);
      ctx.ellipse(0, 0, L * 0.12, L * 0.5, 0, 0, TAU);
      ctx.fill();
      dry(ctx);
      ctx.restore();
    }
    // The tail's zig-zag on the water as it taxis up to speed.
    if (hop === 0 && u < 0.35) {
      ctx.strokeStyle = glow.replace('A', (0.5 * (1 - u / 0.35)).toFixed(3));
      ctx.lineWidth = 0.9;
      ctx.beginPath();
      for (let k = 0; k <= 6; k += 1) {
        const back2 = (k / 6) * L * 2.2;
        const wig = (k % 2 ? 1 : -1) * L * 0.12;
        ctx.lineTo(x - dir[0] * back2 + dir[1] * wig, y - dir[1] * back2 - dir[0] * wig);
      }
      ctx.stroke();
    }
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(h + turn);
    const scale = 1 + lift * 0.25;
    ctx.scale(scale, scale);
    ctx.globalAlpha *= v.night > 0.5 ? 0.7 : 1;
    fishBody(ctx, L, 0.8 + 0.2 * Math.sin(age * 40 + i));
    ctx.restore();
  }
}

/* ------------------------------------------------------------------ dolphins */

function dolphinPath(ctx, L) {
  const w = L * 0.2;
  ctx.beginPath();
  ctx.moveTo(0, -L * 0.5);
  ctx.quadraticCurveTo(w * 0.3, -L * 0.47, w * 0.38, -L * 0.37);
  ctx.bezierCurveTo(w * 0.66, -L * 0.2, w * 0.56, L * 0.14, w * 0.16, L * 0.36);
  ctx.quadraticCurveTo(w * 0.7, L * 0.44, w * 1.05, L * 0.52);
  ctx.quadraticCurveTo(w * 0.3, L * 0.45, 0, L * 0.47);
  ctx.quadraticCurveTo(-w * 0.3, L * 0.45, -w * 1.05, L * 0.52);
  ctx.quadraticCurveTo(-w * 0.7, L * 0.44, -w * 0.16, L * 0.36);
  ctx.bezierCurveTo(-w * 0.56, L * 0.14, -w * 0.66, -L * 0.2, -w * 0.38, -L * 0.37);
  ctx.quadraticCurveTo(-w * 0.3, -L * 0.47, 0, -L * 0.5);
  ctx.closePath();
}

function dolphinFins(ctx, L) {
  const w = L * 0.2;
  for (const side of [-1, 1]) {
    ctx.beginPath();
    ctx.moveTo(side * w * 0.45, -L * 0.14);
    ctx.quadraticCurveTo(side * w * 1.35, -L * 0.04, side * w * 1.25, L * 0.06);
    ctx.quadraticCurveTo(side * w * 0.8, 0, side * w * 0.4, -L * 0.04);
    ctx.fill();
  }
}

function drawDolphins(ctx, s, age, v) {
  const rand = seededRandom(s.seed);
  const L = v.size(46);
  const h0 = (s.headingDeg * Math.PI) / 180;
  const glow = glowColour(v.night);
  const pod = [];
  for (let i = 0; i < s.count; i += 1) {
    pod.push({
      lateral: (i === 0 ? 0 : (i % 2 ? 1 : -1) * Math.ceil(i / 2) * 1.1 * L) + (rand() - 0.5) * 0.5 * L,
      back: Math.ceil(i / 2) * 0.9 * L + rand() * 0.6 * L,
      period: 2.6 + rand() * 0.8,
      phase: rand() * TAU,
      wig: rand() * TAU,
    });
  }
  const A = ctx.globalAlpha;
  const here = presence(age, s.durationS, 1.5, 2);
  for (const d of pod) {
    // Swim along the heading at about 1.7 body lengths a second, weaving gently.
    const along = age * 1.7 * L - d.back;
    const weave = Math.sin(age * 1.4 + d.wig) * 0.55 * L;
    const h = h0 + Math.cos(age * 1.4 + d.wig) * 0.18;
    const fx = Math.sin(h0);
    const fy = -Math.cos(h0);
    const x = v.anchor[0] + fx * along - fy * (d.lateral + weave);
    const y = v.anchor[1] + fy * along + fx * (d.lateral + weave);
    const rise = Math.sin((TAU * age) / d.period + d.phase);
    const up = smooth(0.2, 0.8, rise) * here;

    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(h);

    // The wake: a short V at the Kelvin half-angle, 19.5 deg, fading behind, strongest
    // just after it surfaces and gone when it is deep.
    const wakeA = (0.04 + 0.3 * up) * here * (v.night > 0.5 ? 1.5 : 1);
    if (wakeA > 0.03) {
      const k = Math.tan((19.5 * Math.PI) / 180);
      if (v.night > 0.5) { ctx.shadowColor = 'rgba(110,255,215,0.8)'; ctx.shadowBlur = 5; }
      ctx.lineWidth = Math.max(0.8, L * 0.045);
      for (const sgn of [-1, 1]) {
        const g = ctx.createLinearGradient(0, L * 0.42, 0, L * 1.35);
        g.addColorStop(0, glow.replace('A', wakeA.toFixed(3)));
        g.addColorStop(1, glow.replace('A', '0'));
        ctx.strokeStyle = g;
        ctx.beginPath();
        ctx.moveTo(sgn * L * 0.1, L * 0.42);
        ctx.quadraticCurveTo(sgn * (L * 0.1 + k * 0.4 * L), L * 0.85, sgn * (L * 0.1 + k * 0.95 * L), L * 1.35);
        ctx.stroke();
      }
      ctx.shadowBlur = 0;
    }

    // Under the water: a dark shape, softer and bluer the deeper it is.
    underwater(ctx, 0.9 + 1.3 * (1 - up));
    ctx.globalAlpha = A * (0.92 - 0.5 * up) * here;
    ctx.fillStyle = v.night > 0.5 ? '#04121a' : '#07202e';
    dolphinPath(ctx, L);
    ctx.fill();
    dolphinFins(ctx, L);
    dry(ctx);

    // At the surface: a grey back, a dorsal fin, and white water around it.
    if (up > 0.02) {
      ctx.globalAlpha = A * up;
      const g = ctx.createLinearGradient(-L * 0.12, 0, L * 0.12, 0);
      const dim = v.night > 0.5 ? 0.45 : 1;
      g.addColorStop(0, `rgba(${108 * dim},${122 * dim},${134 * dim},1)`);
      g.addColorStop(0.5, `rgba(${46 * dim},${58 * dim},${70 * dim},1)`);
      g.addColorStop(1, `rgba(${108 * dim},${122 * dim},${134 * dim},1)`);
      ctx.fillStyle = g;
      dolphinPath(ctx, L);
      ctx.fill();
      ctx.fillStyle = `rgba(${40 * dim},${52 * dim},${62 * dim},1)`;
      dolphinFins(ctx, L);
      // The wet shine along the back.
      ctx.strokeStyle = `rgba(255,255,255,${(0.35 * up * dim).toFixed(3)})`;
      ctx.lineWidth = Math.max(0.7, L * 0.025);
      ctx.beginPath();
      ctx.moveTo(0, -L * 0.36);
      ctx.quadraticCurveTo(L * 0.02, 0, 0, L * 0.3);
      ctx.stroke();
      // Dorsal fin, curved back.
      ctx.beginPath();
      ctx.moveTo(0, -L * 0.04);
      ctx.quadraticCurveTo(L * 0.035, L * 0.08, L * 0.01, L * 0.17);
      ctx.quadraticCurveTo(-L * 0.035, L * 0.08, 0, -L * 0.04);
      ctx.fillStyle = `rgba(${28 * dim},${36 * dim},${44 * dim},1)`;
      ctx.fill();
      // White water only where it breaks the surface: a bow wave at the head.
      ctx.strokeStyle = glow.replace('A', (0.6 * up).toFixed(3));
      ctx.lineWidth = Math.max(0.8, L * 0.03);
      ctx.beginPath();
      ctx.arc(0, -L * 0.3, L * 0.16, Math.PI * 1.15, Math.PI * 1.85);
      ctx.stroke();
      if (rise > 0.9) spray(ctx, 0, -L * 0.3, L * 0.12, (rise - 0.9) * 8 * here, seededRandom(s.seed + Math.floor(age * 2)), 6, glow);
    }
    ctx.restore();
  }
}

/* ------------------------------------------------------------------ turtle */

function turtleShell(ctx, L, dim) {
  const sl = L * 0.62;
  const sw = L * 0.5;
  const g = ctx.createRadialGradient(0, -sl * 0.1, sw * 0.1, 0, 0, sl * 0.55);
  g.addColorStop(0, `rgba(${150 * dim},${122 * dim},${64 * dim},1)`);
  g.addColorStop(0.7, `rgba(${104 * dim},${84 * dim},${40 * dim},1)`);
  g.addColorStop(1, `rgba(${70 * dim},${58 * dim},${30 * dim},1)`);
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.ellipse(0, 0, sw / 2, sl / 2, 0, 0, TAU);
  ctx.fill();
  // The plates: five down the middle, four each side.
  ctx.strokeStyle = `rgba(${226 * dim},${200 * dim},${130 * dim},0.8)`;
  ctx.lineWidth = Math.max(0.5, L * 0.018);
  for (let k = 0; k < 5; k += 1) {
    const cy = -sl * 0.34 + k * sl * 0.17;
    const r = sw * 0.14;
    ctx.beginPath();
    for (let j = 0; j < 6; j += 1) {
      const a = (j / 6) * TAU + Math.PI / 6;
      ctx.lineTo(Math.cos(a) * r, cy + Math.sin(a) * r * 0.85);
    }
    ctx.closePath();
    ctx.stroke();
  }
  for (const sgn of [-1, 1]) {
    for (let k = 0; k < 4; k += 1) {
      const y0 = -sl * 0.32 + k * sl * 0.2;
      ctx.beginPath();
      ctx.moveTo(sgn * sw * 0.14, y0);
      ctx.lineTo(sgn * sw * 0.4, y0 - sl * 0.02);
      ctx.lineTo(sgn * sw * 0.42, y0 + sl * 0.17);
      ctx.lineTo(sgn * sw * 0.14, y0 + sl * 0.17);
      ctx.stroke();
    }
  }
  ctx.strokeStyle = `rgba(${230 * dim},${210 * dim},${150 * dim},0.5)`;
  ctx.beginPath();
  ctx.ellipse(0, 0, sw / 2 - 0.5, sl / 2 - 0.5, 0, 0, TAU);
  ctx.stroke();
}

function turtleLimbs(ctx, L, stroke, dim) {
  ctx.fillStyle = `rgba(${120 * dim},${112 * dim},${80 * dim},1)`;
  // Front flippers: long, swept back and forth together.
  for (const sgn of [-1, 1]) {
    ctx.save();
    ctx.translate(sgn * L * 0.2, -L * 0.17);
    ctx.rotate(sgn * (0.9 + 0.55 * stroke));
    ctx.beginPath();
    ctx.ellipse(0, L * 0.2, L * 0.07, L * 0.24, 0, 0, TAU);
    ctx.fill();
    ctx.restore();
    ctx.save();
    ctx.translate(sgn * L * 0.16, L * 0.26);
    ctx.rotate(sgn * (2.4 - 0.25 * stroke));
    ctx.beginPath();
    ctx.ellipse(0, L * 0.06, L * 0.05, L * 0.1, 0, 0, TAU);
    ctx.fill();
    ctx.restore();
  }
  // Head.
  ctx.beginPath();
  ctx.ellipse(0, -L * 0.38, L * 0.09, L * 0.12, 0, 0, TAU);
  ctx.fill();
  ctx.fillStyle = '#10140c';
  for (const sgn of [-1, 1]) {
    ctx.beginPath();
    ctx.arc(sgn * L * 0.045, -L * 0.42, Math.max(0.5, L * 0.015), 0, TAU);
    ctx.fill();
  }
}

function drawTurtle(ctx, s, age, v) {
  const L = v.size(40);
  const h = (s.headingDeg * Math.PI) / 180;
  const here = presence(age, s.durationS, 1.8, 2.2);
  const along = age * 0.28 * L;
  const x = v.anchor[0] + Math.sin(h) * along;
  const y = v.anchor[1] - Math.cos(h) * along;
  // Up for a breath in the middle of the visit, then down again.
  const f = age / s.durationS;
  const up = smooth(0.25, 0.42, f) * (1 - smooth(0.66, 0.84, f)) * here;
  const stroke = Math.sin(age * 1.8);
  const dim = v.night > 0.5 ? 0.45 : 1;
  const glow = glowColour(v.night);
  const A = ctx.globalAlpha;
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(h);
  underwater(ctx, 3);
  ctx.globalAlpha = A * (0.55 - 0.4 * up) * here;
  ctx.fillStyle = v.night > 0.5 ? '#0b1f24' : '#1a4a4a';
  ctx.beginPath();
  ctx.ellipse(0, 0, L * 0.25, L * 0.31, 0, 0, TAU);
  ctx.fill();
  turtleLimbs(ctx, L, stroke, 0.3);
  dry(ctx);
  if (up > 0.02) {
    ctx.globalAlpha = A * up;
    turtleLimbs(ctx, L, stroke, dim);
    turtleShell(ctx, L, dim);
    // Breathing: rings spreading from the head.
    const b = (age * 0.8) % 1;
    ring(ctx, 0, -L * 0.4, L * (0.1 + b * 0.5), 0.5 * (1 - b) * up, glow);
  }
  ctx.restore();
}

/* ------------------------------------------------------------------ humpback */

function whaleBody(ctx, L) {
  const w = L * 0.12;
  ctx.beginPath();
  ctx.moveTo(0, -L * 0.5);
  ctx.bezierCurveTo(w * 0.9, -L * 0.48, w * 1.15, -L * 0.2, w * 1.0, L * 0.02);
  ctx.bezierCurveTo(w * 0.85, L * 0.2, w * 0.4, L * 0.34, w * 0.16, L * 0.4);
  ctx.lineTo(-w * 0.16, L * 0.4);
  ctx.bezierCurveTo(-w * 0.4, L * 0.34, -w * 0.85, L * 0.2, -w * 1.0, L * 0.02);
  ctx.bezierCurveTo(-w * 1.15, -L * 0.2, -w * 0.9, -L * 0.48, 0, -L * 0.5);
  ctx.closePath();
}

function whaleFlippers(ctx, L) {
  // A humpback's flippers are a third of its length, and white.
  for (const sgn of [-1, 1]) {
    ctx.beginPath();
    ctx.moveTo(sgn * L * 0.1, -L * 0.2);
    ctx.bezierCurveTo(sgn * L * 0.3, -L * 0.14, sgn * L * 0.4, -L * 0.02, sgn * L * 0.42, L * 0.08);
    ctx.quadraticCurveTo(sgn * L * 0.3, 0, sgn * L * 0.1, -L * 0.1);
    ctx.closePath();
    ctx.fill();
  }
}

function whaleFluke(ctx, L) {
  const w = L * 0.19;
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.bezierCurveTo(w * 0.4, -L * 0.02, w * 0.9, -L * 0.02, w * 1.05, L * 0.05);
  ctx.bezierCurveTo(w * 0.8, L * 0.05, w * 0.35, L * 0.06, 0, L * 0.09);
  ctx.bezierCurveTo(-w * 0.35, L * 0.06, -w * 0.8, L * 0.05, -w * 1.05, L * 0.05);
  ctx.bezierCurveTo(-w * 0.9, -L * 0.02, -w * 0.4, -L * 0.02, 0, 0);
  ctx.closePath();
}

function drawHumpback(ctx, s, age, v) {
  const L = v.size(170);
  const h = (s.headingDeg * Math.PI) / 180;
  const f = age / s.durationS;
  const here = presence(age, s.durationS, 1.5, 1.5);
  const along = age * 0.1 * L;
  const x = v.anchor[0] + Math.sin(h) * along;
  const y = v.anchor[1] - Math.cos(h) * along;
  const glow = glowColour(v.night);
  const dim = v.night > 0.5 ? 0.4 : 1;
  const surf = smooth(0.18, 0.28, f) * (1 - smooth(0.58, 0.7, f));
  const A = ctx.globalAlpha;
  const deep = 1 - surf;

  // The footprint: a round glassy slick where it went down, fading.
  const printAge = f - 0.8;
  if (printAge > 0) {
    const fx = v.anchor[0] + Math.sin(h) * s.durationS * 0.8 * 0.1 * L;
    const fy = v.anchor[1] - Math.cos(h) * s.durationS * 0.8 * 0.1 * L;
    const a = 0.3 * (1 - printAge / 0.2) * here;
    const g = ctx.createRadialGradient(fx, fy, 0, fx, fy, L * 0.32);
    g.addColorStop(0, `rgba(170,210,235,${(a * dim).toFixed(3)})`);
    g.addColorStop(1, 'rgba(170,210,235,0)');
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(fx, fy, L * 0.32, 0, TAU);
    ctx.fill();
    ring(ctx, fx, fy, L * (0.2 + printAge * 1.5), 0.3 * (1 - printAge / 0.2), glow);
  }

  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(h);

  // Deep: a great dark shape, and white flippers glowing turquoise through the water.
  underwater(ctx, 5);
  ctx.globalAlpha = A * 0.55 * deep * here * (1 - smooth(0.72, 0.92, f));
  ctx.fillStyle = v.night > 0.5 ? '#061821' : '#0b3346';
  whaleBody(ctx, L);
  ctx.fill();
  dry(ctx);
  // The white flippers, glowing turquoise through the water, whether or not the back is up.
  underwater(ctx, 2.5);
  ctx.globalAlpha = A * here * (0.5 + 0.3 * deep) * (1 - smooth(0.72, 0.92, f));
  ctx.fillStyle = v.night > 0.5 ? '#1f5a60' : '#5fd0d6';
  whaleFlippers(ctx, L);
  dry(ctx);

  // The blow: a column of mist that spreads and drifts away.
  const blowAge = (f - 0.2) * s.durationS;
  if (blowAge > 0 && blowAge < 3.2) {
    const k = blowAge / 3.2;
    const drift = k * L * 0.3;
    const bx = v.wind[0] * drift;
    const by = v.wind[1] * drift;
    const r = L * (0.06 + 0.2 * Math.sqrt(k));
    const g = ctx.createRadialGradient(bx, -L * 0.36 + by, 0, bx, -L * 0.36 + by, r);
    const a = 0.75 * (1 - k) * here * (v.night > 0.5 ? 0.35 : 1);
    g.addColorStop(0, `rgba(250,252,255,${a.toFixed(3)})`);
    g.addColorStop(1, 'rgba(250,252,255,0)');
    ctx.globalAlpha = A;
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(bx, -L * 0.36 + by, r, 0, TAU);
    ctx.fill();
  }

  // At the surface: the long dark back, the small hump of a fin, white water at its edges.
  if (surf > 0.02) {
    ctx.globalAlpha = A * surf * here;
    const g = ctx.createLinearGradient(-L * 0.12, 0, L * 0.12, 0);
    g.addColorStop(0, `rgba(${70 * dim},${82 * dim},${92 * dim},1)`);
    g.addColorStop(0.5, `rgba(${24 * dim},${30 * dim},${36 * dim},1)`);
    g.addColorStop(1, `rgba(${70 * dim},${82 * dim},${92 * dim},1)`);
    ctx.fillStyle = g;
    whaleBody(ctx, L);
    ctx.fill();
    // The knobs on its head: tubercles, which only humpbacks have.
    ctx.fillStyle = `rgba(${95 * dim},${105 * dim},${112 * dim},0.9)`;
    const bumps = seededRandom(s.seed);
    for (let k = 0; k < 10; k += 1) {
      ctx.beginPath();
      ctx.arc((bumps() - 0.5) * L * 0.1, -L * (0.3 + bumps() * 0.16), Math.max(0.6, L * 0.008), 0, TAU);
      ctx.fill();
    }
    ctx.fillStyle = `rgba(${18 * dim},${22 * dim},${26 * dim},1)`;
    ctx.beginPath();
    ctx.ellipse(0, L * 0.16, L * 0.025, L * 0.05, 0, 0, TAU);
    ctx.fill();
    // White water where the back breaks the surface: at the head, and along its flanks.
    ctx.strokeStyle = glow.replace('A', (0.55 * surf).toFixed(3));
    ctx.lineWidth = Math.max(1, L * 0.012);
    ctx.beginPath();
    ctx.arc(0, -L * 0.36, L * 0.13, Math.PI * 1.1, Math.PI * 1.9);
    ctx.stroke();
    ctx.strokeStyle = glow.replace('A', (0.3 * surf).toFixed(3));
    for (const sgn of [-1, 1]) {
      ctx.beginPath();
      ctx.moveTo(sgn * L * 0.13, -L * 0.22);
      ctx.quadraticCurveTo(sgn * L * 0.16, -L * 0.05, sgn * L * 0.12, L * 0.1);
      ctx.stroke();
    }
  }

  // The fluke raised as it dives: black above, white beneath, water pouring off it.
  const fluke = smooth(0.66, 0.74, f) * (1 - smooth(0.8, 0.88, f));
  if (fluke > 0.02) {
    ctx.globalAlpha = A * fluke * here;
    ctx.save();
    ctx.translate(0, L * 0.3);
    const lift = 1 + 0.35 * fluke;
    ctx.scale(lift, lift);
    if (v.sun.up > 0.05) {
      ctx.save();
      ctx.translate(-v.sun.dx * L * 0.1, -v.sun.dy * L * 0.1);
      underwater(ctx, 3);
      ctx.fillStyle = `rgba(0,12,24,${(0.35 * v.sun.up).toFixed(3)})`;
      whaleFluke(ctx, L);
      ctx.fill();
      dry(ctx);
      ctx.restore();
    }
    ctx.fillStyle = `rgba(${20 * dim},${24 * dim},${28 * dim},1)`;
    whaleFluke(ctx, L);
    ctx.fill();
    ctx.fillStyle = `rgba(${235 * dim},${238 * dim},${240 * dim},0.85)`;
    ctx.beginPath();
    ctx.ellipse(-L * 0.09, L * 0.045, L * 0.07, L * 0.02, 0.1, 0, TAU);
    ctx.ellipse(L * 0.09, L * 0.045, L * 0.07, L * 0.02, -0.1, 0, TAU);
    ctx.fill();
    spray(ctx, 0, L * 0.06, L * 0.2, 0.7 * fluke, seededRandom(s.seed + Math.floor(age * 3)), 14, glow);
    ctx.restore();
  }
  ctx.restore();
}

const DRAW = {
  flyingFish: drawFlyingFish,
  dolphins: drawDolphins,
  turtle: drawTurtle,
  humpback: drawHumpback,
};

/** Draw one sighting, `ageS` real seconds after it began. */
export function drawSighting(ctx, sighting, ageS, view) {
  const draw = DRAW[sighting.kind];
  if (!draw || ageS < 0 || ageS > sighting.durationS) return;
  ctx.save();
  ctx.globalAlpha = view.fade ?? 1;
  draw(ctx, sighting, ageS, view);
  dry(ctx);
  ctx.restore();
}

/* ------------------------------------------------------------------ a passing ship (#86) */

function hullPath(ctx, L, B) {
  ctx.beginPath();
  ctx.moveTo(0, -L / 2);
  ctx.bezierCurveTo(B * 0.45, -L * 0.43, B / 2, -L * 0.32, B / 2, -L * 0.2);
  ctx.lineTo(B / 2, L * 0.45);
  ctx.quadraticCurveTo(B / 2, L / 2, B * 0.36, L / 2);
  ctx.lineTo(-B * 0.36, L / 2);
  ctx.quadraticCurveTo(-B / 2, L / 2, -B / 2, L * 0.45);
  ctx.lineTo(-B / 2, -L * 0.2);
  ctx.bezierCurveTo(-B / 2, -L * 0.32, -B * 0.45, -L * 0.43, 0, -L / 2);
  ctx.closePath();
}

const BOXES = ['#b5462f', '#2f6db3', '#3d8c5a', '#c9a227', '#d6d6d2', '#7a3b8f', '#d9772b', '#3a8fa0'];

/**
 * A ship seen from above, bow up the screen before rotation: its wake -- a widening band
 * of churned water and the Kelvin V at 19.5 deg -- then its shadow, hull, deck, cargo and
 * bridge. At night the hull goes dark and the navigation lights show: red to port, green
 * to starboard, white on the masts. `view` gives its screen position and heading (radians,
 * 0 = up), its length L in px, the fade, the night and the sun's screen direction.
 */
export function drawShip(ctx, ship, ageS, view) {
  const { x, y, heading, L, fade, night, sun } = view;
  const B = L * 0.15;
  const rand = seededRandom(ship.seed);
  const dim = night > 0.5 ? 0.35 : 1;
  const glow = glowColour(night);
  ctx.save();
  ctx.globalAlpha = fade;
  ctx.translate(x, y);
  ctx.rotate(heading);

  // The wake: churned water, brightest at the stern, widening and breaking up behind it
  // over about three lengths -- soft-edged, not a beam.
  const w = ctx.createLinearGradient(0, L * 0.45, 0, L * 3.4);
  w.addColorStop(0, glow.replace('A', (0.34 * (night > 0.5 ? 0.8 : 1)).toFixed(3)));
  w.addColorStop(0.3, glow.replace('A', '0.1'));
  w.addColorStop(1, glow.replace('A', '0'));
  ctx.fillStyle = w;
  ctx.beginPath();
  ctx.moveTo(-B * 0.4, L * 0.46);
  ctx.lineTo(B * 0.4, L * 0.46);
  ctx.quadraticCurveTo(B * 0.9, L * 1.8, B * 1.1, L * 3.4);
  ctx.lineTo(-B * 1.1, L * 3.4);
  ctx.quadraticCurveTo(-B * 0.9, L * 1.8, -B * 0.4, L * 0.46);
  ctx.closePath();
  underwater(ctx, Math.max(2, B * 0.25));
  ctx.fill();
  dry(ctx);
  // Foam in it, in clumps that drift back and fade: what makes it read as water.
  ctx.fillStyle = glow.replace('A', '0.55');
  for (let k = 0; k < 70; k += 1) {
    const f = (rand() + ageS * 0.06) % 1;
    const along = L * (0.48 + 2.9 * f);
    const across = (rand() - 0.5) * B * (0.7 + 1.5 * f);
    ctx.globalAlpha = fade * 0.55 * (1 - f) * (1 - f);
    ctx.beginPath();
    ctx.ellipse(across, along, Math.max(0.7, B * (0.05 + 0.08 * rand())), Math.max(0.5, B * 0.035), rand() * 3, 0, TAU);
    ctx.fill();
  }
  ctx.globalAlpha = fade;
  // The Kelvin V from the bow shoulders.
  const k = Math.tan((19.5 * Math.PI) / 180);
  ctx.lineWidth = Math.max(0.7, L * 0.006);
  for (const sgn of [-1, 1]) {
    const g = ctx.createLinearGradient(0, -L * 0.35, 0, L * 2.4);
    g.addColorStop(0, glow.replace('A', '0.2'));
    g.addColorStop(1, glow.replace('A', '0'));
    ctx.strokeStyle = g;
    ctx.beginPath();
    ctx.moveTo(sgn * B * 0.3, -L * 0.38);
    ctx.lineTo(sgn * (B * 0.3 + k * L * 2.8), L * 2.4);
    ctx.stroke();
  }
  // Bow wave.
  ctx.strokeStyle = glow.replace('A', '0.55');
  ctx.lineWidth = Math.max(0.8, L * 0.012);
  ctx.beginPath();
  ctx.moveTo(-B * 0.55, -L * 0.3);
  ctx.quadraticCurveTo(0, -L * 0.58, B * 0.55, -L * 0.3);
  ctx.stroke();

  // Its shadow on the water, away from the sun.
  if (sun.up > 0.05) {
    ctx.save();
    ctx.translate(-sun.dx * L * 0.05, -sun.dy * L * 0.05);
    underwater(ctx, 2);
    ctx.fillStyle = `rgba(0,12,24,${(0.4 * sun.up).toFixed(3)})`;
    hullPath(ctx, L, B);
    ctx.fill();
    dry(ctx);
    ctx.restore();
  }

  // Hull and deck.
  ctx.fillStyle = `rgb(${Math.round(34 * dim)},${Math.round(40 * dim)},${Math.round(48 * dim)})`;
  hullPath(ctx, L, B);
  ctx.fill();
  ctx.save();
  ctx.scale(0.86, 0.95);
  ctx.fillStyle = ship.kind === 'tanker'
    ? `rgb(${Math.round(138 * dim)},${Math.round(48 * dim)},${Math.round(38 * dim)})`
    : `rgb(${Math.round(84 * dim)},${Math.round(96 * dim)},${Math.round(104 * dim)})`;
  hullPath(ctx, L, B);
  ctx.fill();
  ctx.restore();

  if (ship.kind === 'tanker') {
    // Pipes down the middle, and the manifold across it.
    ctx.strokeStyle = `rgba(${Math.round(220 * dim)},${Math.round(210 * dim)},${Math.round(190 * dim)},0.8)`;
    ctx.lineWidth = Math.max(0.6, B * 0.05);
    ctx.beginPath();
    ctx.moveTo(0, -L * 0.36);
    ctx.lineTo(0, L * 0.28);
    ctx.moveTo(-B * 0.36, -L * 0.02);
    ctx.lineTo(B * 0.36, -L * 0.02);
    ctx.stroke();
  } else {
    // Stacks of containers, bay after bay.
    const bayH = L * 0.043;
    for (let yb = -L * 0.36; yb < L * 0.26; yb += bayH * 1.12) {
      for (let s = -2; s <= 2; s += 1) {
        const c = BOXES[Math.floor(rand() * BOXES.length)];
        ctx.globalAlpha = fade * dim;
        ctx.fillStyle = c;
        ctx.fillRect(s * B * 0.17 - B * 0.08, yb, B * 0.16, bayH);
      }
    }
    ctx.globalAlpha = fade;
  }
  // The bridge, aft, white; a funnel behind it.
  ctx.fillStyle = `rgb(${Math.round(232 * dim)},${Math.round(232 * dim)},${Math.round(228 * dim)})`;
  ctx.fillRect(-B * 0.46, L * 0.29, B * 0.92, L * 0.09);
  ctx.fillStyle = `rgba(20,24,28,${0.8 * dim + 0.2})`;
  ctx.fillRect(-B * 0.4, L * 0.3, B * 0.8, L * 0.012);
  ctx.fillRect(-B * 0.12, L * 0.4, B * 0.24, L * 0.035);

  // Navigation lights at night: red to port, green to starboard, white on the masts.
  if (night > 0.5) {
    const light = (lx, ly, colour, r) => {
      ctx.save();
      ctx.shadowColor = colour;
      ctx.shadowBlur = r * 4;
      ctx.fillStyle = colour;
      ctx.beginPath();
      ctx.arc(lx, ly, r, 0, TAU);
      ctx.fill();
      ctx.restore();
    };
    const r = Math.max(1.3, L * 0.012);
    light(-B / 2, -L * 0.05, '#ff3b3b', r);
    light(B / 2, -L * 0.05, '#3bff7a', r);
    light(0, -L * 0.3, '#fff6d8', r);
    light(0, L * 0.33, '#fff6d8', r);
  }
  ctx.restore();
}
