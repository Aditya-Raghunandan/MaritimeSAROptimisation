/**
 * email.js -- the email a player gets after "Survive the Triangle": did they beat Telemachus,
 * the path they flew (an inline picture the game draws), how they did against the Coast
 * Guard's pattern and our ML model on the same sea, and a link to vote for us.
 *
 * Pure: a result in, {subject, html, text} out. Everything the player typed (their name) is
 * escaped; the numbers are checked by validate() before they get here.
 */

const PITCH_M2 = 105 * 68;          // a FIFA football pitch
const SWEEP_M = 185.2;              // what the helicopter sees across its track (D027)

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
  if (s.found) return `found them ${clock(s.foundS)} into the search`;
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
  const pitches = Math.round((r.areaKm2 * 1e6) / PITCH_M2);
  const strips = Math.round((r.areaKm2 * 1e6) / (SWEEP_M * 1000));
  const bench = (r.bench ?? []).slice(0, 4);
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
  <tr><td style="padding:18px 28px 4px;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#0b2238;border-radius:12px;">
      <tr><td style="padding:16px 18px;">
        <div style="color:#ffc24a;font-size:30px;font-weight:800;">${r.points.toLocaleString('en-GB')} points</div>
        <div style="color:#d3dde8;font-size:14px;line-height:1.55;margin-top:4px;">
          The sea you had to search was <b>${Math.round(r.areaKm2)} km²</b>: about
          <b>${pitches.toLocaleString('en-GB')} football pitches</b>. Your helicopter could see a strip
          <b>185 m</b> wide, so covering all of it once would take about ${strips.toLocaleString('en-GB')} km of flying.
          You had 45 minutes, which is 125 km. That is why where you fly matters.
        </div>
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
    `${r.points} points. The sea you had to search was ${Math.round(r.areaKm2)} km2, about ${pitches} football pitches.`,
    ...(standing ? ['', `The leaderboard (${BOARD_MODE[r.board.mode]}): ${standing}`,
      ...r.board.lines.map((l) => `${l.rank}. ${l.name}${l.you ? ' (you)' : ''}  ${l.score}`)] : []),
    voteUrl ? `Vote for us: ${voteUrl}` : '', gameUrl ? `Play again: ${gameUrl}` : '', '',
    'Aditya Raghunandan and Suné Toerien, ELEN4012, Wits EIE. We used your email address once and kept nothing.',
  ].filter((l, i, a) => l !== '' || a[i - 1] !== '').join('\n');

  return { subject, html, text };
}
