/**
 * email.js -- the email a player gets after "Survive the Triangle": did they beat Telemachus,
 * the path they flew (an inline picture the game draws), how they did against the Coast
 * Guard's pattern and our ML model on the same sea, and a link to vote for us.
 *
 * Pure: a result in, {subject, html, text} out. Everything the player typed (their name) is
 * escaped; the numbers are checked by validate() before they get here.
 */

const SWEEP_M = 185.2;              // what the helicopter sees across its track (D027)
const TRACK_M = 90 * (1852 / 3600) * 45 * 60;   // 45 minutes at 90 kt: 125 km
/** What one helicopter can see in its 45 minutes, km²: a 185 m strip, 125 km long (23 km²). */
const SEEN_KM2 = (SWEEP_M * TRACK_M) / 1e6;

/**
 * The size of the search in plain numbers (Aditya, 15 Oct: football pitches, then 125 km of
 * flying against a cloud of a few km², was confusing). Where they could be, what one helicopter
 * can see, and what that means: a share of the sea, or how many times over it could be covered.
 */
export function sizeWords(areaKm2) {
  const area = Math.max(1, Math.round(areaKm2));
  const seen = Math.round(SEEN_KM2);
  if (SEEN_KM2 < areaKm2) {
    return { area, seen, so: `So at best you could search about ${Math.round((100 * SEEN_KM2) / areaKm2)} % of it. Where you fly matters.` };
  }
  const times = Math.round(SEEN_KM2 / areaKm2);
  return {
    area, seen,
    so: `Enough to cover it ${times <= 1 ? 'about once' : `about ${times} times over`}, if you fly neat lanes: every turn costs time.`,
  };
}

export const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}[c]));

const pct = (x) => `${(100 * x).toFixed(1)} %`;
const clock = (s) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, '0')}`;

/** How the computer's searcher is named in a row: the trained model, or greedy, which does not learn. */
const tag = (name) => (name === 'Telemachus' ? 'Telemachus (ML model)' : 'The greedy search');

const ordinal = (n) => {
  const v = n % 100;
  const s = ['th', 'st', 'nd', 'rd'];
  return `${n}${s[(v - 20) % 10] || s[v] || s[0]}`;
};
const BOARD_MODE = { map: 'with the probability map', currents: 'with currents only' };

/** Where the player stands on the board, in words (plain text, not escaped), or null with no board. */
export function standingWords(board) {
  if (!board) return null;
  if (board.rank === 1) {
    return board.of === 1 ? 'You are the first name on the board.'
      : `You are top of the board right now, ahead of ${board.of - 1} other player${board.of > 2 ? 's' : ''}.`;
  }
  const leader = board.lines.find((l) => l.rank === 1);
  const you = board.lines.find((l) => l.you);
  const head = `You are ${ordinal(board.rank)} of ${board.of} on the board right now`;
  if (!leader || !you) return `${head}.`;
  const gap = leader.score - you.score;
  return gap > 0 ? `${head}, ${gap.toLocaleString('en-GB')} points behind ${leader.name}.` : `${head}, level with ${leader.name}.`;
}

/** The board's lines as table rows, the player's highlighted, with a gap row before theirs if it is further down. */
function boardRows(board) {
  return board.lines.map((l, i) => {
    const gap = i > 0 && l.rank > board.lines[i - 1].rank + 1
      ? '<tr><td colspan="3" style="padding:2px 10px;color:#7f93a8;">…</td></tr>' : '';
    return `${gap}<tr style="${l.you ? 'background:#16314d;' : ''}">
      <td style="padding:7px 10px;color:#9fb3c8;width:34px;">${l.rank}.</td>
      <td style="padding:7px 0;color:#ffffff;${l.you ? 'font-weight:700;' : ''}">${escapeHtml(l.name)}${l.you ? ' (you)' : ''}</td>
      <td align="right" style="padding:7px 10px;color:#ffc24a;font-weight:700;">${l.score.toLocaleString('en-GB')}</td></tr>`;
  }).join('');
}

/** The headline and the line under it, by who the player beat. */
export function verdictOf(r) {
  const ai = r.ai.name;
  // Found them but did not search the most (15 Oct): a find is cheered, the prize explained.
  if (r.you?.found && !(r.beatAi && r.beatCg)) {
    const ahead = [!r.beatCg && 'the Coast Guard’s pattern', !r.beatAi && ai].filter(Boolean).join(' and ');
    return {
      subject: '\u{1F6DF} You found them!',
      head: 'You found them!',
      line: `You found the person in the water. ${ahead.charAt(0).toUpperCase()}${ahead.slice(1)} still searched `
        + 'more of the likely sea, and over many rescues the bigger search finds more people. Beat both to win.',
    };
  }
  if (r.beatAi && r.beatCg) {
    return {
      subject: `\u{1F3C6} You beat ${ai} AND the Coast Guard`,
      head: `You beat ${ai} and the Coast Guard.`,
      line: 'Humans 1, machine 0. You out-searched the model and the textbook pattern on the same sea.',
    };
  }
  if (r.beatAi) {
    return {
      subject: ai === 'Telemachus' ? '\u{1F389} You beat Telemachus, our ML model' : `\u{1F389} You beat ${ai}`,
      head: `You beat ${ai}.`,
      line: 'Humans 1, machine 0. The Coast Guard’s pattern still edged you out: the textbook is hard to beat.',
    };
  }
  if (r.beatCg) {
    return {
      subject: `\u{1F916} You beat the Coast Guard, but ${ai} beat you`,
      head: `So close. ${ai} beat you.`,
      line: 'You out-searched the Coast Guard’s pattern, but the machine searched more of the sea. Rematch?',
    };
  }
  return {
    subject: `\u{1F916} ${ai} won this round. Rematch?`,
    head: `${ai} won this round.`,
    line: 'The machine searched more of the sea this time. It never gets tired, but it can be beaten.',
  };
}

function personWords(s) {
  if (s.found) return s.foundS < 30 ? 'found them on arrival' : `found them ${clock(s.foundS)} into the search`;
  if (Number.isFinite(s.missedKm)) return `missed them by ${s.missedKm.toFixed(1)} km`;
  return 'did not find them';
}

function bar(label, s, colour, max, bold) {
  const w = Math.max(2, Math.round((100 * s.pos) / max));
  return `
  <tr><td style="padding:10px 0 2px;font-size:15px;color:#e8eef5;${bold ? 'font-weight:700;' : ''}">${label}
    <span style="float:right;font-weight:700;color:#ffffff;">${pct(s.pos)}</span></td></tr>
  <tr><td style="padding:0;"><div style="background:#16314d;border-radius:6px;height:14px;">
    <div style="width:${w}%;background:${colour};border-radius:6px;height:14px;"></div></div></td></tr>
  <tr><td style="padding:3px 0 0;font-size:13px;color:#9fb3c8;">${s.found ? '✅' : '❌'} ${personWords(s)}</td></tr>`;
}

/**
 * The email for one game. `r` is a validated result; `links` has gameUrl and, once we have
 * it, voteUrl; `imageCid` names the inline picture of the flight, or null for none.
 */
export function buildResultEmail(r, { gameUrl, voteUrl = '' } = {}, imageCid = 'flight') {
  const v = verdictOf(r);
  const standing = standingWords(r.board);
  const subject = r.board?.rank === 1 && r.board.of > 1 ? `${v.subject} · top of the board!` : v.subject;
  const name = escapeHtml(r.name || 'Player');
  const ai = escapeHtml(r.ai.name);
  const max = Math.max(r.you.pos, r.cg.pos, r.ai.pos, 0.01);
  const size = sizeWords(r.areaKm2);
  const bench = (r.bench ?? []).slice(0, 4);
  const field = r.field ?? [];
  const button = (href, label, colour) => `<a href="${escapeHtml(href)}" style="display:inline-block;background:${colour};`
    + 'color:#ffffff;text-decoration:none;font-weight:700;font-size:16px;padding:13px 26px;border-radius:10px;'
    + `margin:6px 6px 0 0;">${label}</a>`;

  const html = `<!doctype html><html><body style="margin:0;padding:0;background:#020914;">
<div style="display:none;max-height:0;overflow:hidden;">You searched ${pct(r.you.pos)} of the sea. ${ai} searched ${pct(r.ai.pos)}.</div>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#020914;">
<tr><td align="center" style="padding:24px 12px;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:580px;background:#071a2c;border-radius:16px;overflow:hidden;font-family:'Segoe UI',Arial,sans-serif;">
  <tr><td style="background:#03101c;padding:22px 28px;border-bottom:3px solid #ff5a1f;">
    <div style="color:#ff5a1f;font-size:13px;font-weight:800;letter-spacing:3px;">SURVIVE THE BERMUDA TRIANGLE</div>
    <div style="color:#9fb3c8;font-size:13px;margin-top:4px;">A search-and-rescue game from our Wits final-year project</div>
  </td></tr>
  <tr><td style="padding:26px 28px 6px;">
    <div style="color:#9fb3c8;font-size:15px;">Hi ${name},</div>
    <div style="color:#ffffff;font-size:28px;font-weight:800;line-height:1.2;margin:8px 0 8px;">${escapeHtml(v.head)}</div>
    <div style="color:#d3dde8;font-size:16px;line-height:1.5;">${escapeHtml(v.line)}</div>
  </td></tr>
  ${imageCid ? `<tr><td style="padding:18px 28px 4px;">
    <img src="cid:${imageCid}" width="524" alt="Your flight over the cloud of places they could be"
      style="display:block;width:100%;max-width:524px;height:auto;border-radius:12px;border:1px solid #1d4a73;" />
    <div style="color:#9fb3c8;font-size:12px;line-height:1.5;margin-top:6px;">The glow is the 10,000 places they could
      have drifted to. White: your flight. Blue: the Coast Guard's Expanding Square. Pink: ${ai}. Red dashes: where
      they really drifted, from a real satellite-tracked buoy.</div>
  </td></tr>` : ''}
  <tr><td style="padding:18px 28px 4px;">
    <div style="color:#ffffff;font-size:18px;font-weight:700;">Same sea, same 45 minutes</div>
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
      ${bar(`You (${name})`, r.you, '#ffffff', max, true)}
      ${bar('The Coast Guard’s pattern', r.cg, '#3987e5', max, false)}
      ${bar(tag(ai), r.ai, '#ff6fb5', max, false)}
    </table>
  </td></tr>
  ${field.length > 3 ? `<tr><td style="padding:16px 28px 4px;">
    <div style="color:#ffffff;font-size:16px;font-weight:700;">Everyone who searched your sea</div>
    <div style="color:#9fb3c8;font-size:12px;margin:2px 0 6px;">The same cloud, flown by every searcher we tested: the Coast Guard's patterns, our ML model, greedy and random. Best first.</div>
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="font-size:14px;">
      ${field.map((s, i) => `<tr style="${s.you ? 'background:#16314d;' : ''}">
        <td style="padding:4px 8px;color:#9fb3c8;width:24px;">${i + 1}</td>
        <td style="padding:4px 8px;color:#ffffff;${s.you ? 'font-weight:800;' : ''}">${escapeHtml(s.label)}</td>
        <td style="padding:4px 8px;color:#ffffff;font-weight:700;text-align:right;">${pct(s.pos)}</td>
        <td style="padding:4px 8px;text-align:right;">${s.found ? '✅' : '❌'}</td></tr>`).join('')}
    </table>
  </td></tr>` : ''}
  <tr><td style="padding:18px 28px 4px;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#0b2238;border-radius:12px;">
      <tr><td style="padding:16px 18px;">
        <div style="color:#ffc24a;font-size:30px;font-weight:800;">${r.points.toLocaleString('en-GB')} points</div>
        <table role="presentation" cellpadding="0" cellspacing="0" style="margin-top:8px;color:#d3dde8;font-size:14px;line-height:1.5;">
          <tr><td style="padding:2px 16px 2px 0;">Where they could be</td>
            <td style="padding:2px 0;color:#ffffff;font-weight:700;text-align:right;">${size.area} km²</td></tr>
          <tr><td style="padding:2px 16px 2px 0;">What your helicopter can see in 45 minutes (a 185 m strip)</td>
            <td style="padding:2px 0;color:#ffffff;font-weight:700;text-align:right;">${size.seen} km²</td></tr>
        </table>
        <div style="color:#d3dde8;font-size:14px;line-height:1.55;margin-top:6px;">${size.so}</div>
      </td></tr>
    </table>
  </td></tr>
  ${r.board ? `<tr><td style="padding:18px 28px 4px;">
    <div style="color:#ffffff;font-size:18px;font-weight:700;">\u{1F3C6} The leaderboard <span style="color:#9fb3c8;font-size:14px;font-weight:400;">(${BOARD_MODE[r.board.mode]})</span></div>
    <div style="color:#d3dde8;font-size:15px;line-height:1.5;margin:4px 0 8px;">${escapeHtml(standing)}</div>
    ${r.board.lines.length ? `<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#0b2238;border-radius:12px;font-size:15px;">
      ${boardRows(r.board)}
    </table>` : ''}
  </td></tr>` : ''}
  <tr><td style="padding:16px 28px 0;color:#d3dde8;font-size:14px;line-height:1.55;">
    <b style="color:#ffffff;">What "searched" means.</b> It is the share of the 10,000 possible positions your
    strip passed over. If our drift model is right, a search that covers 60 % finds the person about 6 times in 10.
  </td></tr>
  ${bench.length ? `<tr><td style="padding:14px 28px 0;color:#d3dde8;font-size:14px;line-height:1.55;">
    <b style="color:#ffffff;">Across all 55 real buoys we tested</b>${r.arrival ? ` (helicopter arriving ${escapeHtml(r.arrival)} after the call)` : ''}:
    ${bench.map((b) => `${escapeHtml(b.label)} ${pct(b.pos)}`).join(' · ')}.
  </td></tr>` : ''}
  <tr><td style="padding:22px 28px 8px;">
    ${voteUrl ? button(voteUrl, '\u{1F5F3}️ Vote for us', '#ff5a1f') : ''}
    ${gameUrl ? button(gameUrl, 'Play again', '#3987e5') : ''}
  </td></tr>
  <tr><td style="padding:18px 28px 24px;color:#7f93a8;font-size:12px;line-height:1.6;border-top:1px solid #16314d;">
    Made by Aditya Raghunandan and Suné Toerien for ELEN4012, our final-year investigation project at the
    School of Electrical and Information Engineering, University of the Witwatersrand. The ocean is real:
    HYCOM currents, ERA5 wind and NOAA's satellite-tracked drifter buoys, 2019 to 2023.<br />
    We used your email address once, to send you this, and kept nothing.
  </td></tr>
</table>
</td></tr></table></body></html>`;

  const text = [
    `Hi ${r.name || 'Player'},`, '', v.head, v.line, '',
    `You: searched ${pct(r.you.pos)}, ${personWords(r.you)}`,
    `The Coast Guard's pattern: searched ${pct(r.cg.pos)}, ${personWords(r.cg)}`,
    `${tag(r.ai.name)}: searched ${pct(r.ai.pos)}, ${personWords(r.ai)}`, '',
    ...(field.length > 3 ? ['Everyone who searched your sea, best first:',
      ...field.map((s, i) => `${i + 1}. ${s.label}  ${pct(s.pos)}${s.found ? '  found them' : ''}`), ''] : []),
    `${r.points} points.`,
    `Where they could be: ${size.area} km2. What your helicopter can see in 45 minutes (a 185 m strip): ${size.seen} km2.`,
    size.so,
    ...(standing ? ['', `The leaderboard (${BOARD_MODE[r.board.mode]}): ${standing}`,
      ...r.board.lines.map((l) => `${l.rank}. ${l.name}${l.you ? ' (you)' : ''}  ${l.score}`)] : []),
    voteUrl ? `Vote for us: ${voteUrl}` : '', gameUrl ? `Play again: ${gameUrl}` : '', '',
    'Aditya Raghunandan and Suné Toerien, ELEN4012, Wits EIE. We used your email address once and kept nothing.',
  ].filter((l, i, a) => l !== '' || a[i - 1] !== '').join('\n');

  return { subject, html, text };
}
