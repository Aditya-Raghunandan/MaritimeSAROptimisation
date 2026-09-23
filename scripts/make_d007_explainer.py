"""make_d007_explainer.py: the D007 hand-off document, written to be read by a person.

Writes `D007 - the probability map explained.{html,pdf}` into the vault's `figures/`,
beside `D007 - slicing the probability map.html`, which is the same kind of artefact: prose
about a decision, aimed at the other half of the project rather than at a marker.

WHY A SCRIPT AND NOT A HAND-WRITTEN FILE. Every number in it is computed here, in the same
file as the prose that quotes it. A document whose figures are pasted in drifts from its own
argument the first time a parameter changes; this one cannot, because changing the parameter
changes the sentence. The same rule as `make_figures.py` emitting `figure_numbers.json`.

The diagrams are matplotlib rendered to inline SVG rather than PNG, so they stay sharp when
the HTML is printed to PDF. Each figure gets its own `svg.hashsalt` because matplotlib names
its clip-path ids from that salt, and two inline SVGs sharing an id silently corrupt each
other's clipping -- one figure's mask applied to another's axes.

Run:
    python scripts/make_d007_explainer.py
    python scripts/make_d007_explainer.py --out some/dir --no-pdf
"""

from __future__ import annotations

import argparse
import io
import math
import re
import subprocess
from datetime import date
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

VAULT_FIGURES = Path(
    r"C:\Users\aragh\OneDrive - University of Witwatersrand\InvestigationMaritime\figures"
)
STEM = "D007 - the probability map explained"

# ---------------------------------------------------------------------------
# The numbers. Everything the prose asserts is computed here, once.
# ---------------------------------------------------------------------------

W_M = 185.0                 # sweep width, person in the water -- STILL UNCITED
DRIFT_MS = 1.8              # Gulf Stream typical
BOX_KM = 100.0              # the on-scene box
CELL_M = 250.0              # physics-map cell
ONSCENE_MIN = 45.0          # R7a
PHI0 = 26.5                 # box-centre latitude
M_PER_DEG_LAT = 111_320.0


def m_per_deg_lon(lat_deg: float) -> float:
    return M_PER_DEG_LAT * math.cos(math.radians(lat_deg))


NUM = {
    "drift_per_min_m": DRIFT_MS * 60,
    "onscene_drift_km": DRIFT_MS * ONSCENE_MIN * 60 / 1000,
    "travel_72h_km": DRIFT_MS * 72 * 3600 / 1000,
    "dlat_deg": CELL_M / M_PER_DEG_LAT,
    "dlon_deg": CELL_M / m_per_deg_lon(PHI0),
    "agent_cell_km_64": BOX_KM / 64,
    "agent_cell_km_32": BOX_KM / 32,
    "swept_frac_64": W_M / (BOX_KM * 1000 / 64),
    "blob_15min_km": DRIFT_MS * 15 * 60 / 1000,
}
NUM["dlon_over_dlat"] = NUM["dlon_deg"] / NUM["dlat_deg"]
NUM["unswept_pct_64"] = (1 - NUM["swept_frac_64"]) * 100
NUM["blob_15min_cells"] = NUM["blob_15min_km"] * 1000 / CELL_M


def cosphi_error_pct(from_lat: float, to_lat: float) -> float:
    """How wrong an eastward speed becomes if cos(phi) is frozen at `from_lat`."""
    return abs(m_per_deg_lon(from_lat) / m_per_deg_lon(to_lat) - 1) * 100


def cell_area_variation_pct(box_km: float, centre: float = PHI0) -> float:
    half = (box_km / 111.32) / 2
    return abs(1 - math.cos(math.radians(centre + half))
               / math.cos(math.radians(centre - half))) * 100


def map_mb(box_km: float, cell_m: float, slices: int) -> float:
    n = box_km * 1000 / cell_m
    return n * n * slices * 4 / 1e6


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

INK = "#0D2430"
INK3 = "#64808D"
TEAL = "#0E7C87"
AMBER = "#A8650C"
MASS = "#A81E48"
RULE = "#C6D2D8"


def _style(ax, *, box=True):
    ax.tick_params(colors=INK3, labelsize=7, length=3)
    for s in ax.spines.values():
        s.set_color(RULE if box else "none")
    ax.set_facecolor("none")


def _svg(fig, salt: str) -> str:
    """Render to an inline <svg>, with ids salted so two figures cannot collide."""
    matplotlib.rcParams["svg.hashsalt"] = salt
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    svg = buf.getvalue()
    svg = svg[svg.index("<svg"):]                       # drop the XML/doctype preamble
    svg = re.sub(r'\s(width|height)="[\d.]+pt"', "", svg, count=2)
    return svg.replace("<svg ", '<svg style="width:100%;height:auto" ', 1)


def _density(ax, x, y, lo, hi, cell_km):
    """Draw a binned density as ONE raster image, not one path per bin.

    `hist2d` emits a QuadMesh, and in SVG that is a separate element for every
    non-empty cell -- 400x400 bins turned this document into 41 MB. `imshow` of a
    precomputed histogram embeds a single small image instead, which is also what a
    density map actually is.
    """
    bins = np.arange(lo, hi + cell_km / 2, cell_km)
    h, _, _ = np.histogram2d(x, y, bins=[bins, bins])
    h = np.ma.masked_where(h == 0, h)
    ax.imshow(h.T, origin="lower", extent=[lo, hi, lo, hi], cmap="OrRd",
              interpolation="nearest", aspect="equal")


def cloud(n: int, seed: int = 7):
    """An elongated, slightly curved drift cloud in a 100 km box."""
    rng = np.random.default_rng(seed)
    t = rng.beta(2.2, 2.2, n)
    along = 18 + 46 * t
    across = rng.normal(0, 2.4 + 3.4 * t, n)
    ang = np.radians(38 - 14 * t)
    x = 26 + along * np.cos(ang) - across * np.sin(ang)
    y = 24 + along * np.sin(ang) + across * np.cos(ang)
    return x, y


def fig_three_grids() -> str:
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1))
    x, y = cloud(160_000)

    ax = axes[0]
    g = np.arange(4, 100, 8.0)
    X, Y = np.meshgrid(g, g)
    ax.quiver(X, Y, np.ones_like(X), 0.55 + 0.25 * np.sin(Y / 26), color=AMBER,
              scale=26, width=0.005, headwidth=3.4)
    ax.set_title("1  Ocean data\n~8 km (HYCOM, fixed)", fontsize=8.5, color=INK, pad=7)

    ax = axes[1]
    _density(ax, x, y, 0, 100, CELL_M / 1000)
    ax.set_title(f"2  Physics map\n{CELL_M:.0f} m cells", fontsize=8.5, color=INK, pad=7)

    ax = axes[2]
    _density(ax, x, y, 0, 100, 100 / 32)
    for k in np.linspace(0, 100, 33):
        ax.axhline(k, color=RULE, lw=0.35, zorder=3)
        ax.axvline(k, color=RULE, lw=0.35, zorder=3)
    ax.set_title(f"3  Agent's view\n32x32, ~{NUM['agent_cell_km_32']:.1f} km cells",
                 fontsize=8.5, color=INK, pad=7)

    for ax in axes:
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        ax.set_aspect("equal"); ax.set_xticks([0, 50, 100]); ax.set_yticks([0, 50, 100])
        ax.set_xlabel("km", fontsize=7, color=INK3)
        _style(ax)
    axes[0].set_ylabel("km", fontsize=7, color=INK3)
    return _svg(fig, "grids")


def fig_speckle() -> str:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    for ax, n, label, colour in ((axes[0], 100_000, "static-like noise", MASS),
                                 (axes[1], 1_000_000, "smooth, same physics", TEAL)):
        x, y = cloud(n)
        _density(ax, x, y, 30, 70, CELL_M / 1000)
        ax.set_title(f"$10^{int(math.log10(n))}$ particles in {CELL_M:.0f} m cells",
                     fontsize=8.5, color=INK, pad=7)
        ax.text(0.04, 0.05, label, transform=ax.transAxes, fontsize=7.5,
                color=colour, fontweight="bold")
        ax.set_xlim(30, 70); ax.set_ylim(30, 70); ax.set_aspect("equal")
        ax.set_xticks([30, 50, 70]); ax.set_yticks([30, 50, 70])
        ax.set_xlabel("km", fontsize=7, color=INK3)
        _style(ax)
    return _svg(fig, "speckle")


def fig_cosphi() -> str:
    """The new one: freezing cos(phi) corrupts eastward drift."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    lats = np.linspace(17, 36, 300)
    for frm, c in ((17, MASS), (26.5, AMBER), (33, TEAL)):
        ax.plot(lats, [cosphi_error_pct(frm, t) for t in lats], color=c, lw=2,
                label=f"cos $\\varphi$ frozen at {frm}\u00b0 N")
    ax.axhline(5, color=INK3, lw=0.9, ls=(0, (3, 3)))
    # Parked on the right, where no curve runs: at 17 N all three lines are
    # climbing through the left half and the label was sitting on the teal one.
    ax.text(35.6, 5.4, "5 % error", fontsize=7, color=INK3, ha="right")
    ax.set_xlabel("latitude the particle reaches (\u00b0N)", fontsize=8, color=INK3)
    ax.set_ylabel("eastward speed error (%)", fontsize=8, color=INK3)
    ax.legend(fontsize=7, frameon=False, labelcolor=INK)
    ax.set_xlim(17, 36); ax.set_ylim(0, 20)
    _style(ax)

    ax = axes[1]
    labels = ["sphericity\n(arc vs chord)", "cell shape\n(cos $\\varphi$ in the box)",
              "integrator\n(cos $\\varphi$ frozen)"]
    vals = [0.004, cell_area_variation_pct(BOX_KM), cosphi_error_pct(17, 36)]
    bars = ax.barh(labels, vals, color=[INK3, AMBER, MASS], height=0.55)
    for b, v in zip(bars, vals):
        ax.text(v * 1.12, b.get_y() + b.get_height() / 2,
                f"{v:.3g} %", va="center", fontsize=8, color=INK)
    ax.set_xscale("log"); ax.set_xlim(0.002, 60)
    ax.set_xlabel("error (%, log scale)", fontsize=8, color=INK3)
    ax.tick_params(axis="y", labelsize=7.5)
    _style(ax)
    return _svg(fig, "cosphi")


def fig_box_placement() -> str:
    fig, ax = plt.subplots(figsize=(9.0, 2.9))
    t = np.linspace(0, 1, 300)
    px = 10 + 430 * t
    py = 40 + 46 * np.sin(np.pi * t * 0.92)
    ax.plot(px, py, color=INK, lw=1.8, zorder=2)
    ax.plot([10], [40], "o", color=INK, ms=6, zorder=3)
    ax.text(10, 30, "Last known\nposition", fontsize=7.5, color=INK, ha="center")

    ax.add_patch(plt.Rectangle((-40, -10), 100, 100, fill=False, ec=MASS,
                               lw=1.5, ls=(0, (5, 3))))
    ax.text(10, 96, "same box placed at the START  \u2717\n(the cloud leaves it within hours)",
            fontsize=7.5, color=MASS, ha="center")

    rng = np.random.default_rng(3)
    cx, cy = 400, py[-1] + 4
    # 700, not 2500: every marker is its own SVG path, and this is a schematic --
    # the cloud only has to read as a cloud.
    ax.scatter(rng.normal(cx, 11, 700), rng.normal(cy, 7, 700), s=1.1,
               color="#E4632D", alpha=0.55, zorder=4)
    ax.add_patch(plt.Rectangle((cx - 50, cy - 50), 100, 100, fill=False, ec=TEAL, lw=1.8))
    ax.text(cx, cy + 56, "100 km box, fixed,\nplaced at ON-SCENE time  \u2713",
            fontsize=7.5, color=TEAL, ha="center", fontweight="bold")
    ax.text(cx + 58, cy, f"helicopter searches\nonly here, for {ONSCENE_MIN:.0f} min\n"
                         f"(drift \u2248 {NUM['onscene_drift_km']:.0f} km)",
            fontsize=7.5, color=INK3, va="center")

    ax.annotate("", xy=(10, -22), xytext=(440, -22),
                arrowprops=dict(arrowstyle="<->", color=INK3, lw=1))
    ax.text(225, -33, f"up to ~{NUM['travel_72h_km']:.0f} km of travel in 72 h "
                      f"({DRIFT_MS} m/s)", fontsize=7.5, color=INK3, ha="center")
    ax.set_xlim(-60, 520); ax.set_ylim(-45, 120); ax.axis("off")
    return _svg(fig, "boxplace")


def fig_coverage() -> str:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 2.9))

    ax = axes[0]
    w = NUM["agent_cell_km_64"] * 1000
    ax.add_patch(plt.Rectangle((0, 0), w, w, fc="#FBE3D6", ec=INK, lw=1.2))
    ax.add_patch(plt.Rectangle((w / 2 - W_M / 2, 0), W_M, w, fc=TEAL, ec="none"))
    ax.text(w / 2, w * 1.06, f"{W_M:.0f} m swept", fontsize=7.5, color=TEAL,
            ha="center", fontweight="bold")
    ax.text(w * 0.24, w / 2, f"{NUM['unswept_pct_64']:.0f} %\nstill\nunswept",
            fontsize=8, color=AMBER, ha="center", va="center")
    ax.set_title(f"One agent cell (64x64 over {BOX_KM:.0f} km)\n"
                 f"= {NUM['agent_cell_km_64']*1000:.0f} m wide",
                 fontsize=8.5, color=INK, pad=7)
    ax.text(w / 2, -w * 0.13, "the coarse cell still looks 'hot'", fontsize=7,
            color=INK3, ha="center")
    ax.set_xlim(-w * 0.1, w * 1.1); ax.set_ylim(-w * 0.2, w * 1.2)
    ax.set_aspect("equal"); ax.axis("off")

    ax = axes[1]
    ax.add_patch(plt.Rectangle((0, 0), 185, 900, fc="none", ec=TEAL, lw=1.4,
                               ls=(0, (4, 3))))
    ax.text(92, 940, "swept strip\n(earth-fixed\nmask)", fontsize=7, color=TEAL, ha="center")
    ax.add_patch(plt.Rectangle((1620, 0), 185, 900, fc=TEAL, ec="none", alpha=0.85))
    ax.text(1712, 940, "where that\nwater is\n15 min later", fontsize=7, color=INK, ha="center")
    ax.annotate("", xy=(1600, 450), xytext=(210, 450),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax.text(900, 500, f"current {DRIFT_MS} m/s\n\u2192 {NUM['blob_15min_km']:.1f} km",
            fontsize=7.5, color=INK, ha="center")
    ax.set_title("The swept water drifts away", fontsize=8.5, color=INK, pad=7)
    ax.text(900, -190, "a mask pinned to the map is wrong within ~2 min",
            fontsize=7, color=MASS, ha="center")
    ax.set_xlim(-150, 1980); ax.set_ylim(-300, 1120)
    ax.set_aspect("equal"); ax.axis("off")
    return _svg(fig, "coverage")


def fig_timeaxis() -> str:
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 3.8),
                            gridspec_kw={"height_ratios": [1, 1.45], "hspace": 0.75})

    ax = axes[0]
    for k in range(19):
        ax.add_patch(plt.Rectangle((k, 0), 0.92, 1, fc="#E8EEF1", ec=RULE, lw=0.6))
    for k in range(19, 30):
        ax.add_patch(plt.Rectangle((k, 0), 0.92, 1, fc=TEAL, ec="none", alpha=0.75))
    ax.text(9, 1.5, "coarse: 15-min snapshots (drift only, nobody searching)",
            fontsize=7.5, color=INK3, ha="center")
    ax.text(24, 1.5, f"fine: 60 s, {ONSCENE_MIN:.0f} min on-scene",
            fontsize=7.5, color=TEAL, ha="center", fontweight="bold")
    ax.text(0, -0.75, "LKP", fontsize=7.5, color=INK)
    ax.text(19, -1.35, "arrival \u2014 a scenario input\n(transit + daylight), not the agent's choice",
            fontsize=7, color=INK3, ha="center")
    ax.text(30.4, 0.5, "return", fontsize=7.5, color=INK3, va="center")
    ax.set_xlim(-1, 33); ax.set_ylim(-1.8, 2.2); ax.axis("off")

    ax = axes[1]
    g = np.linspace(-1.5, 3.5, 700)
    gauss = lambda m, s: np.exp(-0.5 * ((g - m) / s) ** 2)
    ax.plot(g, gauss(0, 0.42), color=INK3, lw=1, ls=(0, (2, 2)))
    ax.plot(g, gauss(1.62, 0.42), color=INK3, lw=1, ls=(0, (2, 2)))
    ax.plot(g, gauss(0.81, 0.42), color=TEAL, lw=2.1,
            label="true map at 7.5 min (blob moved)")
    ax.plot(g, 0.5 * (gauss(0, 0.42) + gauss(1.62, 0.42)), color=MASS, lw=2.1,
            label="blend of the 0 & 15 min maps (two half-ghosts)")
    ax.legend(fontsize=7, frameon=False, labelcolor=INK, loc="upper right")
    ax.set_xlabel("km along the current", fontsize=8, color=INK3)
    ax.set_yticks([]); ax.set_xlim(-1.5, 3.5)
    _style(ax)
    return _svg(fig, "timeaxis")


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------

CSS = """
:root{--ink:#0D2430;--ink-2:#34515F;--ink-3:#64808D;--paper:#ECF0F2;--paper-2:#F7F9FA;
--rule:#C6D2D8;--rule-soft:#DDE5E9;--teal:#0E7C87;--amber:#A8650C;--mass:#A81E48;
--tint-teal:#E4F1F2;--tint-warn:#FBEDE2;--warn-ink:#8A4A0B}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);font-family:"IBM Plex Serif",Georgia,serif;
font-size:15.5px;line-height:1.6;margin:0;padding:0 20px 70px}
.wrap{max-width:820px;margin:0 auto}
header.mast{padding:44px 0 26px;border-bottom:2px solid var(--ink);margin-bottom:30px}
h1{font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:35px;line-height:1.1;
margin:0 0 12px;letter-spacing:-.4px}
.standfirst{font-size:16px;color:var(--teal);margin:0;line-height:1.5}
h2{font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:21px;margin:40px 0 12px;
padding-top:14px;border-top:1px solid var(--rule);letter-spacing:-.2px}
h3{font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:16px;margin:26px 0 8px;
color:var(--teal)}
h3.warn{color:var(--amber)}
p{margin:0 0 13px}
.short{background:var(--paper-2);border-left:4px solid var(--teal);padding:16px 20px;
margin:0 0 26px;font-size:15px}
.prop{background:var(--tint-teal);border-left:4px solid var(--teal);padding:14px 18px;margin:16px 0}
.note{background:var(--tint-warn);border-left:4px solid var(--amber);padding:14px 18px;
margin:16px 0;color:var(--warn-ink)}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:13.5px;
font-family:"IBM Plex Sans",system-ui,sans-serif}
th{text-align:left;font-weight:600;border-bottom:2px solid var(--ink);padding:7px 9px;
background:var(--paper-2)}
td{border-bottom:1px solid var(--rule-soft);padding:7px 9px;vertical-align:top}
tr:last-child td{border-bottom:1px solid var(--rule)}
figure{margin:22px 0}
figcaption{font-size:12.5px;color:var(--ink-3);margin-top:8px;line-height:1.45;
font-family:"IBM Plex Sans",system-ui,sans-serif}
code{font-family:"IBM Plex Mono",monospace;font-size:.88em;background:var(--paper-2);
padding:1px 4px;border-radius:3px}
strong{font-weight:600}
ul{margin:0 0 13px;padding-left:22px}li{margin-bottom:5px}
.foot{margin-top:44px;padding-top:14px;border-top:1px solid var(--rule);
font-size:11.5px;color:var(--ink-3);font-family:"IBM Plex Sans",system-ui,sans-serif;
display:flex;justify-content:space-between}
@page{size:A4;margin:16mm 14mm}
@media print{body{background:#fff;padding:0}
h2,h3{break-after:avoid}
p{orphans:2;widows:2}
figure,.prop,.note,.short{break-inside:avoid}
figcaption{break-before:avoid}
/* Tables MAY split -- keeping them whole was leaving half-empty pages. The
   header repeats on the continuation so a split table is still readable. */
table{break-inside:auto}
thead{display:table-header-group}
tr{break-inside:avoid}}
"""


def document(figs):
    n = NUM
    try:
        today = date.today().strftime("%-d %b %Y")
    except ValueError:                       # Windows strftime has no %-d
        today = date.today().strftime("%d %b %Y").lstrip("0")
    run = ("ELEN4012A · D007 probability map interface · "
           "Aditya → Suné · " + today)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>The probability map, in plain English</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&amp;family=IBM+Plex+Sans:wght@400;500;600;700&amp;family=IBM+Plex+Serif:ital,wght@0,400;0,500;1,400&amp;display=swap">
<style>{CSS}</style></head><body><div class="wrap">

<header class="mast">
<h1>The probability map, in plain English</h1>
<p class="standfirst">What I'm proposing for D007, the four problems I found in it, and what I
need your yes on. Aditya &rarr; Sun&eacute; &middot; {today} &middot; about a 12-minute read</p>
</header>

<div class="short"><strong>The short version.</strong> We've been using the word <em>grid</em> for
three different things, and that is why issue&nbsp;#7 kept feeling impossible. Pull them apart and
most of D007 becomes easy. Think of it as a <strong>road map</strong> (the ocean data), a
<strong>high-resolution photo</strong> taken with it (the physics map) and a <strong>thumbnail</strong>
of that photo on your phone (what the agent sees). Three objects, three cell sizes, three
different reasons for each size.
<br><br><em>Second edition.</em> Everything from the first still stands. What's new is
<strong>&sect;3, Problem&nbsp;4</strong> &mdash; a problem in the integrator rather than the map, which
would have corrupted the physics by up to {cosphi_error_pct(17, 36):.0f}&nbsp;% while looking
entirely plausible &mdash; plus measured timings in place of my earlier estimate, and a list of the
numbers we are still quoting without a citation.</div>

<h2>1 &middot; The three grids</h2>
<figure>{figs['grids']}
<figcaption>Same {BOX_KM:.0f}&nbsp;km box, same drift cloud, drawn three ways. Panel&nbsp;3 is what
PPO receives. Illustrative data, not a real scenario.</figcaption></figure>

<table><thead><tr><th>Grid</th><th>Cell size</th><th>What sets it</th><th>Owner</th></tr></thead>
<tbody>
<tr><td>1 &middot; Ocean data</td><td>0.08&deg; lon &times; 0.04&deg; lat &asymp; 8 &times; 4.5 km</td>
<td>HYCOM. We can't change it &mdash; and note it is <em>not</em> 1/12&deg; and <em>not</em> square,
which is what issue&nbsp;#7 assumes.</td><td>&mdash;</td></tr>
<tr><td>2 &middot; Physics map</td><td>{CELL_M:.0f}&ndash;500 m</td>
<td>Sweep width W &asymp; {W_M:.0f} m. Cells much bigger than W can't tell 'just searched' from
'a kilometre to the side'.</td><td>Aditya (engine)</td></tr>
<tr><td>3 &middot; Agent's view</td><td>32&times;32 to 64&times;64 cells over the box</td>
<td>What a CNN policy in PPO can learn from (your draft answer).</td><td>Sun&eacute; (env)</td></tr>
</tbody></table>

<p>Because they are three objects and not one, the cell size of each is a <strong>parameter</strong>
in the code, not a constant. We do not know the best agent grid and should not pretend to &mdash;
it is something the training runs can answer for us.</p>

<h2>2 &middot; Three things I've already settled on my side</h2>

<h3>A &middot; The speckle problem &rarr; more particles, not blurring</h3>
<p>With 10<sup>5</sup> particles spread over {CELL_M:.0f}&nbsp;m cells, each cell only gets a
handful, so the map looks like TV static. The easy fix is to smooth it (kernel density
estimation). I'm <strong>not</strong> doing that, because smoothing spreads the cloud wider than
the physics did, and our validation score R2c (the fraction of real drifter positions that fall
inside the 90&nbsp;% contour) would then partly be scoring my smoothing, not the drift model.
Wider contour, better score, less truth.</p>
<figure>{figs['speckle']}
<figcaption>Same cloud, same {CELL_M:.0f}&nbsp;m cells. The only difference is particle count. The
noise per cell falls roughly as 1/&radic;(particles per cell).</figcaption></figure>
<p><strong>Measured, not estimated.</strong> Last time I wrote &ldquo;about 5 hours of cluster
time&rdquo; from a guess. I have since benchmarked a stand-in integration step in NumPy:</p>
<table><thead><tr><th>N</th><th>ms per step</th><th>per 48&nbsp;h scenario</th>
<th>all 35 on one core</th></tr></thead><tbody>
<tr><td>10<sup>4</sup></td><td>0.8</td><td>2.2 s</td><td>1.3 min</td></tr>
<tr><td>10<sup>5</sup></td><td>12.7</td><td>36.6 s</td><td>21 min</td></tr>
<tr><td><strong>10<sup>6</sup></strong></td><td>175</td><td><strong>504 s</strong></td>
<td><strong>294 min</strong></td></tr>
</tbody></table>
<p>So the guess was right, and now it is a number. Since D009 already runs a Slurm array over
scenarios, 35 at once is about <strong>8 minutes of wall clock</strong>. D006's claim that the
Monte Carlo is not our compute problem survives the tenfold increase.</p>

<h3>B &middot; Square cells without a map projection</h3>
<p>A degree of longitude shrinks as you go north, so equal-degree cells aren't square. Instead of
a projection, each scenario uses plain lat/lon with the two sides sized separately:
&Delta;lat = {CELL_M:.0f}&nbsp;m &divide; {M_PER_DEG_LAT/1000:.2f}&nbsp;km = {n['dlat_deg']:.6f}&deg;,
and &Delta;lon = &Delta;lat &divide; cos(&phi;<sub>0</sub>) = {n['dlon_deg']:.6f}&deg;, a ratio of
<strong>{n['dlon_over_dlat']:.3f}</strong>. Across a {BOX_KM:.0f}&nbsp;km box the residual error is
{cell_area_variation_pct(BOX_KM):.2f}&nbsp;% &mdash; about 1&nbsp;m on a {CELL_M:.0f}&nbsp;m cell
&mdash; so it's small enough to ignore. It also keeps everything readable by xarray and KML.</p>
<p>That ratio is also the precise reason issue&nbsp;#7's API cannot work as written: a single
<code>resolution</code> argument cannot express two different step sizes, and all four of its
acceptance criteria would pass against a square grid that does not exist.</p>

<h3>C &middot; Order of work</h3>
<p>I write the D007 spec, you check it, <em>then</em> we both code to it, using one shared loader
and one validator that both halves import. I'd rather lose two days now than retrain agents after
a change to the interface.</p>

<h2>3 &middot; Four problems I found in my own plan</h2>
<p>Two are real design problems; the third turned out not to apply to us; the fourth is new since
the first edition and is the worst of them.</p>

<h3>Problem 1: The box can't be fixed at the start</h3>
<p>The target can travel <strong>~{n['travel_72h_km']:.0f} km in 72&nbsp;h</strong>
({DRIFT_MS}&nbsp;m/s). A {BOX_KM:.0f}&nbsp;km box pinned at the last known position is empty within
hours. <strong>But the fine map only has to exist while someone is searching</strong>, and that's
the {ONSCENE_MIN:.0f}-minute on-scene window. During those {ONSCENE_MIN:.0f} minutes the cloud
drifts about {n['onscene_drift_km']:.0f}&nbsp;km.</p>
<figure>{figs['boxplace']}
<figcaption>Anchor the box when the helicopter arrives, not when the person goes
missing.</figcaption></figure>
<p>The alternative &mdash; one box big enough to hold the whole run &mdash; is not affordable. Sized
by travel rather than by spread it is about 495&nbsp;km at 48&nbsp;h, and the study domain itself is
roughly 2100 &times; 1900&nbsp;km:</p>
<table><thead><tr><th>Box</th><th>Cells at 500 m</th><th>Per scenario</th><th>All 35</th>
</tr></thead><tbody>
<tr><td>{BOX_KM:.0f} km (on-scene)</td><td>200 &times; 200</td>
<td>{map_mb(100,500,141):.0f} MB</td><td>0.8 GB</td></tr>
<tr><td>400 km (48 h of travel)</td><td>800 &times; 800</td>
<td>{map_mb(400,500,141):.0f} MB</td><td>12.6 GB</td></tr>
<tr><td>2100 km (the whole domain)</td><td>4200 &times; 4200</td>
<td><strong>{map_mb(2100,500,141)/1000:.1f} GB</strong></td><td>348 GB</td></tr>
</tbody></table>
<div class="prop"><strong>Proposal.</strong> One fixed box per scenario, <strong>placed on the
cloud at arrival time</strong>. Use the <strong>same box size for every scenario</strong>, so one
agent cell always means the same number of km. Check that each scenario's cloud fits inside it,
and log any mass outside as <code>lost_mass</code> (already specified in D016). The box edge acts
as a wall for the agent. Before arrival we keep particles only, plus coarse snapshots for the site
and the plots.</div>

<h3>Problem 2: How the agent 'remembers' where it has searched</h3>
<p>Two separate problems. <strong>(a)</strong> At 64&times;64 over {BOX_KM:.0f}&nbsp;km, one agent
cell is {n['agent_cell_km_64']*1000:.0f}&nbsp;m wide. One pass sweeps {W_M:.0f}&nbsp;m of it, so
<strong>{n['unswept_pct_64']:.0f}&nbsp;%</strong> of the cell still looks hot and the agent may keep
circling it. <strong>(b)</strong> Subtler: if searched water is recorded as a mask pinned to the
map, the mask goes stale almost immediately, because the water moves
{n['drift_per_min_m']:.0f}&nbsp;m <em>per minute</em> &mdash; most of a sweep width. A moving grid
doesn't solve (b) either. <strong>The coverage has to move with the water.</strong></p>
<figure>{figs['coverage']}
<figcaption>Left: a coarse cell hides the track. Right: the swept water itself drifts
off.</figcaption></figure>
<div class="prop"><strong>Proposal: do the sweep on the particles, not on a grid.</strong> This is
what R4c already says: <em>detection is modelled when the searcher passes within the sweep width
of a particle</em>. Each step, any particle within W/2 of the track has its weight cut (to 0, or
&times;(1 &minus; POD)). Every map is then just a picture of the weight that's left. Three things
come free:
<ul><li>the reward (weight removed this step) is exact, however coarse the agent's view is;</li>
<li>coverage drifts with the water automatically;</li>
<li>the baseline and PPO get scored by the same code.</li></ul>
For the agent's view, your two-channel idea is the right one: channel&nbsp;A is the whole box,
coarse (32&times;32), and channel&nbsp;B a fine window around the helicopter (e.g. 32&times;32 at
{CELL_M:.0f}&nbsp;m, about 8&nbsp;km across) so it can see its own tracks. Sizes are your call.
<br><br><strong>What it changes in the interface:</strong> the engine hands over particle positions
for the on-scene window, not just pictures &mdash; about 370&nbsp;MB per scenario, 13&nbsp;GB for
all 35, which is fine on the cluster. If sweeping is slow inside PPO, the env can use a
10<sup>5</sup> subsample. <strong>Your per-timestep normalisation stays</strong>: the map the agent
sees sums to 1 and means <em>where the target is, given we haven't found it yet</em>. The reward is
counted on the raw weight <em>before</em> renormalising, or rewards aren't comparable.</div>

<h3>Problem 3: The time axis. This one mostly doesn't apply</h3>
<p>The worry was that if the agent chooses <em>when</em> it arrives, you can't precompute the fine
window. In our project it doesn't choose. R7a fixes the transit (MH-60T, 300&nbsp;nmi out) and D012
places the {ONSCENE_MIN:.0f}-minute window in daylight for each scenario. <strong>Arrival time is a
scenario input</strong>, known before the episode, so coarse-then-fine works. The fine 60&nbsp;s
steps are just the engine's own &Delta;t (D002).</p>
<figure>{figs['timeaxis']}
<figcaption>Top: the time axis. Bottom: why we must never fill gaps by blending two maps. In
15 min the blob moves ~{n['blob_15min_km']:.1f}&nbsp;km
({n['blob_15min_cells']:.0f} cells at {CELL_M:.0f}&nbsp;m), and a blend gives two ghosts instead of
one moved blob.</figcaption></figure>
<div class="note"><strong>One thing to write down anyway.</strong> If we ever randomise the arrival
time to improve generalisation, <strong>don't</strong> make the in-between maps by linearly
blending the 15-minute ones. Re-run the {ONSCENE_MIN:.0f} minutes from the stored particle
positions instead. It's cheap, and Problem&nbsp;2 already means we keep them.</div>

<h3 class="warn">Problem 4: NEW &mdash; the one that would have quietly broken the physics</h3>
<p>This one isn't about the map at all. It's in the integrator, and I only found it by asking what
&ldquo;curvature&rdquo; actually means here. It turns out to mean three different things, and only
one of them matters:</p>
<figure>{figs['cosphi']}
<figcaption>Left: how wrong an eastward speed becomes if cos&nbsp;&phi; is computed once and reused
as the particle travels north. Right: the three effects people mean by &ldquo;curvature&rdquo;, on a
log scale.</figcaption></figure>
<p>To move a particle we convert its eastward speed in m/s into degrees of longitude per second,
which means dividing by {M_PER_DEG_LAT/1000:.2f}&nbsp;km &times; cos&nbsp;&phi;. <strong>If
cos&nbsp;&phi; is evaluated once &mdash; at the datum, say &mdash; and then reused while the
particle drifts north, every eastward step is wrong.</strong></p>
<table><thead><tr><th>cos&nbsp;&phi; frozen at</th><th>particle reaches</th>
<th>eastward speed wrong by</th></tr></thead><tbody>
<tr><td>26.5&deg; N</td><td>30&deg; N</td><td>{cosphi_error_pct(26.5,30):.1f} %</td></tr>
<tr><td>26.5&deg; N</td><td>36&deg; N</td><td><strong>{cosphi_error_pct(26.5,36):.1f} %</strong></td></tr>
<tr><td>17&deg; N</td><td>36&deg; N</td><td><strong>{cosphi_error_pct(17,36):.1f} %</strong></td></tr>
</tbody></table>
<p>A particle crossing the Stream's full latitude range would be carried up to
{cosphi_error_pct(17,36):.0f}&nbsp;% too far east or west &mdash; <strong>and it would look exactly
like physics.</strong> No error, no warning, a plausible trajectory in the wrong place. It is the
same shape as the bug we just fixed in the visualiser, where a field was painted 44&nbsp;km north of
where it belonged because latitude was assumed linear on a Mercator screen.</p>
<div class="prop"><strong>Proposal.</strong> cos&nbsp;&phi; is recomputed <strong>per particle,
every step</strong>, and a test pins it: a particle pushed due east at constant speed from
17&nbsp;N and from 35&nbsp;N must cover <em>different</em> longitude and the <em>same</em> distance
in metres. The other two effects we ignore, and say so: sphericity is 0.004&nbsp;% over the box, far
below every other error in the system.</div>

<h2>4 &middot; Numbers we are still quoting without a source</h2>
<p>Worth saying plainly, because they are load-bearing and the report will be marked on them.
W sets the cell size, the track spacing (R4b) and the detection model (R4c) &mdash; so an uncited W
is three uncited results.</p>
<table><thead><tr><th>Quantity</th><th>What we use</th><th>Status</th></tr></thead><tbody>
<tr><td>Sweep width, person in the water</td><td>{W_M:.0f} m (0.1 nmi)</td>
<td><strong>No citation yet.</strong> Needs the IAMSAR table with its assumed conditions and sea
state, and a decision on whether detection inside W is deterministic or has a lateral-range
curve</td></tr>
<tr><td>On-scene window</td><td>{ONSCENE_MIN:.0f} min</td><td>R7a, needs the MH-60T fuel source</td></tr>
<tr><td>Outbound radius</td><td>300 nmi</td><td>R7a, same</td></tr>
<tr><td>Search speed</td><td>&mdash;</td>
<td><strong>Recorded nowhere.</strong> Without it we cannot turn the window into track miles, and so
cannot state what fraction of the box one aircraft can actually sweep</td></tr>
<tr><td>Gulf Stream speed</td><td>1&ndash;2.5 m/s</td><td>measured from HYCOM &mdash; ours</td></tr>
</tbody></table>
<div class="note"><strong>Also unresolved, and it belongs in the limitations register.</strong>
Beaching is detected from HYCOM's land mask at ~8&nbsp;km, against a probability grid at
{CELL_M:.0f}&nbsp;m. D016 already requires a proper coastline cross-check &ldquo;before any result
depends on it&rdquo;, and we haven't done it.</div>

<h2>5 &middot; What I need from you</h2>
<table><thead><tr><th>#</th><th>Question</th><th>My suggestion</th></tr></thead><tbody>
<tr><td>1</td><td>Fixed box, same size every scenario, anchored at arrival time?</td><td>Yes</td></tr>
<tr><td>2</td><td>Detection done on particles (R4c), so the engine delivers particle positions for
the on-scene window?</td><td>Yes</td></tr>
<tr><td>3</td><td>Agent observation = coarse whole box + fine window around the helicopter.
Sizes?</td><td>Your call</td></tr>
<tr><td>4</td><td>Final choice between 32&times;32 and 64&times;64 for channel A?</td>
<td>Your call &mdash; and let's leave it a parameter so the training runs can decide</td></tr>
<tr><td>5</td><td>Arrival time fixed per scenario for now (not randomised)?</td><td>Yes, for now</td></tr>
<tr><td>6</td><td><strong>New:</strong> do you agree W must be pinned with a citation before either
of us codes against it?</td><td>Yes</td></tr>
</tbody></table>
<p>Please reply in <code>handoff/to-Aditya.md</code>. Once we agree, I'll freeze D007 with these
answers, and the shared loader and validator follow the same day.</p>

<p style="margin-top:26px;font-size:13.5px;color:var(--ink-3)"><em>Further work, noted not built:
two aircraft searching together. It turns this from one path into a coordinated multi-agent problem
&mdash; a different RL formulation &mdash; but the reward, newly swept mass, generalises to it
unchanged, which is the interesting part.</em></p>

<div class="foot"><span>{run}</span><span>generated by <code>scripts/make_d007_explainer.py</code></span></div>
</div></body></html>"""


def to_pdf(html_path: Path, pdf_path: Path) -> None:
    """Print the document via the frontend's Playwright.

    Shelling out rather than importing: Playwright here is the Node install added for
    the browser tests, and a second Python copy would mean another browser download and
    another version to keep in step. `cwd` is `frontend/` because that is where
    node_modules is, and Node resolves an ESM import from the script's own directory.
    """
    repo = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        ["node", "tools/html-to-pdf.mjs", str(html_path), str(pdf_path)],
        cwd=repo / "frontend", capture_output=True, text=True, shell=False,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise SystemExit("html-to-pdf.mjs failed -- is `npm ci` done in frontend/?")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=str(VAULT_FIGURES), help="where the document goes")
    p.add_argument("--no-pdf", action="store_true", help="HTML only")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("rendering figures...")
    figs = {
        "grids": fig_three_grids(),
        "speckle": fig_speckle(),
        "boxplace": fig_box_placement(),
        "coverage": fig_coverage(),
        "timeaxis": fig_timeaxis(),
        "cosphi": fig_cosphi(),
    }

    html_path = out / (STEM + ".html")
    html_path.write_text(document(figs), encoding="utf-8")
    print("  %s  %.0f kB" % (html_path.name, html_path.stat().st_size / 1024))

    if not args.no_pdf:
        pdf_path = out / (STEM + ".pdf")
        to_pdf(html_path, pdf_path)
        print("  %s  %.0f kB" % (pdf_path.name, pdf_path.stat().st_size / 1024))

    print("\nkey numbers: dlon/dlat=%.3f  cos-phi 17->36N error=%.1f%%  unswept at 64x64=%.0f%%"
          % (NUM["dlon_over_dlat"], cosphi_error_pct(17, 36), NUM["unswept_pct_64"]))


if __name__ == "__main__":
    main()
