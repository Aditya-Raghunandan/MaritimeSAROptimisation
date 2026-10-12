/**
 * handler.js -- one request in, one email out. The same code runs behind server.js (a laptop
 * at the stand, or any Node host) and api/send.js (a Vercel function).
 *
 * The game posts its result as JSON; this checks every field, builds the email and sends it
 * through the SMTP account in the environment, the same variables KnockKnock.prof used
 * (SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, EMAIL_FROM). Nothing is stored: the address
 * is used for this one message and is never logged.
 */

import nodemailer from 'nodemailer';
import { buildResultEmail } from './email.js';

const EMAIL_RE = /^[^\s@<>()[\]\\,;:"]{1,64}@[A-Za-z0-9.-]{1,190}\.[A-Za-z]{2,24}$/;
const MAX_IMAGE_CHARS = 900_000;           // a 640 px JPEG is ~80 kB; this is generous
const IMAGE_RE = /^data:image\/(png|jpeg);base64,([A-Za-z0-9+/=]+)$/;

/** Per-address-of-the-sender limits: a few emails a minute, and a ceiling for the day. */
export class Limits {
  constructor({ perIp = 5, windowMs = 10 * 60_000, perDay = 400 } = {}) {
    Object.assign(this, { perIp, windowMs, perDay, seen: new Map(), day: '', count: 0 });
  }

  allow(ip, now = Date.now()) {
    const day = new Date(now).toISOString().slice(0, 10);
    if (day !== this.day) Object.assign(this, { day, count: 0, seen: new Map() });
    if (this.count >= this.perDay) return false;
    const recent = (this.seen.get(ip) ?? []).filter((t) => now - t < this.windowMs);
    if (recent.length >= this.perIp) return false;
    recent.push(now);
    this.seen.set(ip, recent);
    this.count += 1;
    return true;
  }
}

const num = (x, lo, hi) => typeof x === 'number' && Number.isFinite(x) && x >= lo && x <= hi;

function searcher(s, name) {
  if (!s || !num(s.pos, 0, 1)) throw new Error(`${name}.pos must be a share between 0 and 1`);
  return {
    pos: s.pos,
    found: Boolean(s.found),
    foundS: num(s.foundS, 0, 2700) ? s.foundS : 0,
    missedKm: num(s.missedKm, 0, 1000) ? s.missedKm : NaN,
  };
}

/** The game's JSON, checked field by field, as the result buildResultEmail takes. */
export function validate(body) {
  if (!body || typeof body !== 'object') throw new Error('expected a JSON object');
  const email = String(body.email ?? '').trim();
  if (!EMAIL_RE.test(email)) throw new Error('that does not look like an email address');
  if (!num(body.points, 0, 100_000) || !num(body.areaKm2, 0.01, 100_000)) throw new Error('points and areaKm2 are required');
  const you = searcher(body.you, 'you');
  const cg = searcher(body.cg, 'cg');
  const ai = { ...searcher(body.ai, 'ai'), name: body.ai?.name === 'Telemachus' ? 'Telemachus' : 'the AI' };
  let image = null;
  if (body.image != null) {
    const m = typeof body.image === 'string' && body.image.length <= MAX_IMAGE_CHARS && IMAGE_RE.exec(body.image);
    if (!m) throw new Error('image must be a PNG or JPEG data URL under 900 kB');
    image = { type: m[1], base64: m[2] };
  }
  const bench = Array.isArray(body.bench)
    ? body.bench.slice(0, 7).filter((b) => b && typeof b.label === 'string' && num(b.pos, 0, 1))
      .map((b) => ({ label: b.label.slice(0, 40), pos: b.pos }))
    : [];
  return {
    email,
    name: String(body.name ?? 'Player').slice(0, 16),
    points: Math.round(body.points),
    areaKm2: body.areaKm2,
    you,
    cg,
    ai,
    beatAi: you.pos > ai.pos,
    beatCg: you.pos > cg.pos,
    bench,
    image,
  };
}

/** The SMTP transport, as KnockKnock.prof built it. */
export function transportFrom(env) {
  if (!env.SMTP_HOST || !env.SMTP_USER || !env.SMTP_PASS) return null;
  const port = Number(env.SMTP_PORT) || 587;
  return nodemailer.createTransport({
    host: env.SMTP_HOST,
    port,
    secure: port === 465,
    family: 4,
    auth: { user: env.SMTP_USER, pass: env.SMTP_PASS },
  });
}

/**
 * Handle one POST body. Returns {status, json}. `send` is the transport's sendMail (injected
 * so the tests never reach the network).
 */
export async function handleSend(body, { env = process.env, send, limits, ip = '?' } = {}) {
  let r;
  try {
    r = validate(body);
  } catch (err) {
    return { status: 400, json: { ok: false, error: err.message } };
  }
  if (limits && !limits.allow(ip)) {
    return { status: 429, json: { ok: false, error: 'too many emails from here; try again in a few minutes' } };
  }
  if (!send) return { status: 503, json: { ok: false, error: 'email is not set up on this server' } };
  const msg = buildResultEmail(r, { gameUrl: env.GAME_URL ?? '', voteUrl: env.VOTE_URL ?? '' },
    r.image ? 'flight' : null);
  try {
    await send({
      from: env.EMAIL_FROM || env.SMTP_USER,
      to: r.email,
      subject: msg.subject,
      html: msg.html,
      text: msg.text,
      attachments: r.image ? [{
        filename: `your-flight.${r.image.type === 'png' ? 'png' : 'jpg'}`,
        content: Buffer.from(r.image.base64, 'base64'),
        contentType: `image/${r.image.type}`,
        cid: 'flight',
      }] : [],
    });
  } catch (err) {
    console.error('send failed:', err.code ?? '', err.message);   // never the address
    return { status: 502, json: { ok: false, error: 'the email server did not accept it' } };
  }
  return { status: 200, json: { ok: true } };
}

/** CORS: the game's own site, plus anything listed in ALLOWED_ORIGINS (comma separated). */
export function corsHeaders(origin, env = process.env) {
  const allowed = (env.ALLOWED_ORIGINS ?? 'https://aditya-raghunandan.github.io,http://localhost:5173,http://localhost:4173')
    .split(',').map((s) => s.trim()).filter(Boolean);
  const ok = origin && allowed.includes(origin);
  return {
    'Access-Control-Allow-Origin': ok ? origin : allowed[0],
    'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    Vary: 'Origin',
  };
}
