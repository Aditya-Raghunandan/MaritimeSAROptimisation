"""performance_breakdown.py: each run's test result against the Expanding Square, by arrival time and place.

Works for any run folder, v1 (sar.rl), v2 (sar.rl_v2) or v3 (sar.rl_v3), once its wrap-up has
written test/flights.csv: one row per test window and searcher, both flown on the same windows.
For every run, arrival time and place, and for all places together, it gives the model's mean
POS, the Expanding Square's, the model's mean lead on the same windows, the share of windows
the model won, and how many windows that is.

Writes two files: <out>.csv, and <out>.xlsx with each arrival time in its own colour (a CSV
cannot hold colour). The .xlsx is written with the standard library alone, so no extra package
is needed on the cluster.

    PYTHONPATH=src python scripts/performance_breakdown.py \
        --runs /home/26p67/data/derived/rl/runs/run_baseline \
               /home/26p67/data/derived/rl/runs_v2/run_baseline_v2 \
        --out /home/26p67/data/derived/rl/performance_breakdown
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd

LOCATIONS_FILE = Path(__file__).with_name("rl_locations.txt")
SQUARE = "expanding-square"
ALL_PLACES = "all places"
# One fill per arrival time, light enough to read black text on; any other arrival time stays white.
ARRIVAL_FILL = {30: "DDEBF7", 60: "E2EFDA", 120: "FFF2CC", 180: "FCE4D6"}
COLUMNS = ["run", "version", "arrival_min", "place", "windows", "ppo_pos", "square_pos", "lead", "ppo_wins"]


def places(path=LOCATIONS_FILE) -> dict[str, tuple[float, float]]:
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            lat, lon, name = line.strip().split(";")
            out[name] = (float(lat), float(lon))
    return out


def place_of(lat: float, lon: float, named: dict[str, tuple[float, float]]) -> str:
    """The named place nearest a window's LKP; windows names keep it to 0.01 degree."""
    return min(named, key=lambda n: (named[n][0] - lat) ** 2 + (named[n][1] - lon) ** 2)


def breakdown(run: Path, named) -> pd.DataFrame:
    flights = pd.read_csv(run / "test" / "flights.csv")
    config = json.loads((run / "config.json").read_text())
    flights["place"] = [place_of(a, b, named) for a, b in zip(flights["lat"], flights["lon"])]
    wide = flights.pivot_table(index=["window", "arrival_min", "place"], columns="searcher",
                               values="pos").reset_index()
    if "ppo" not in wide or SQUARE not in wide:
        raise SystemExit(f"{run}/test/flights.csv needs both ppo and {SQUARE} flights")
    wide["lead"] = wide["ppo"] - wide[SQUARE]
    rows = []
    for (minutes, place), g in pd.concat([wide, wide.assign(place=ALL_PLACES)]).groupby(["arrival_min", "place"]):
        rows.append({"run": run.name, "version": config.get("version", "v1"), "arrival_min": int(minutes),
                     "place": place, "windows": len(g), "ppo_pos": round(g["ppo"].mean(), 4),
                     "square_pos": round(g[SQUARE].mean(), 4), "lead": round(g["lead"].mean(), 4),
                     "ppo_wins": round(float((g["lead"] > 0).mean()), 3)})
    table = pd.DataFrame(rows)
    table["_all"] = table["place"] != ALL_PLACES
    return table.sort_values(["arrival_min", "_all", "place"]).drop(columns="_all")


def _cell(ref: str, value, style: int) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}" s="{style}"><v>{value}</v></c>'
    return f'<c r="{ref}" s="{style}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def write_xlsx(table: pd.DataFrame, path: Path) -> None:
    """A one-sheet workbook: bold header, frozen; each row filled by its arrival time."""
    fills = list(ARRIVAL_FILL.values())
    letters = [chr(ord("A") + i) for i in range(len(COLUMNS))]
    rows = ['<row r="1">' + "".join(_cell(f"{c}1", h, 1) for c, h in zip(letters, COLUMNS)) + "</row>"]
    for i, rec in enumerate(table[COLUMNS].itertuples(index=False), start=2):
        minutes, bold = rec.arrival_min, rec.place == ALL_PLACES   # all-places rows also bold
        if minutes in ARRIVAL_FILL:
            style = 2 + list(ARRIVAL_FILL).index(minutes) + (len(fills) if bold else 0)
        else:
            style = 1 if bold else 0
        rows.append(f'<row r="{i}">' + "".join(_cell(f"{c}{i}", v, style) for c, v in zip(letters, rec)) + "</row>")
    sheet = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
             '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
             '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" state="frozen"/>'
             '</sheetView></sheetViews><cols><col min="1" max="1" width="24" customWidth="1"/>'
             '<col min="4" max="4" width="30" customWidth="1"/></cols>'
             f'<sheetData>{"".join(rows)}</sheetData></worksheet>')
    fill_xml = ('<fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill>'
                + "".join(f'<fill><patternFill patternType="solid"><fgColor rgb="FF{c}"/></patternFill></fill>'
                          for c in fills))
    # cellXfs: 0 plain, 1 bold header, 2.. one per arrival fill, then the same fills in bold.
    xf = '<xf numFmtId="0" fontId="{}" fillId="{}" borderId="0" xfId="0" applyFont="1" applyFill="1"/>'
    xfs = [xf.format(0, 0), xf.format(1, 0)]
    xfs += [xf.format(0, 2 + i) for i in range(len(fills))] + [xf.format(1, 2 + i) for i in range(len(fills))]
    styles = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
              '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
              '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>'
              '<font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
              f'<fills count="{2 + len(fills)}">{fill_xml}</fills>'
              '<borders count="1"><border/></borders>'
              '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
              f'<cellXfs count="{len(xfs)}">{"".join(xfs)}</cellXfs>'
              '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="xml" ContentType="application/xml"/>'
                   '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                   '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                   '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
                   '</Types>')
        z.writestr("_rels/.rels",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
                   '</Relationships>')
        z.writestr("xl/workbook.xml",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                   'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                   '<sheets><sheet name="by arrival and place" sheetId="1" r:id="rId1"/></sheets></workbook>')
        z.writestr("xl/_rels/workbook.xml.rels",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
                   '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
                   '</Relationships>')
        z.writestr("xl/worksheets/sheet1.xml", sheet)
        z.writestr("xl/styles.xml", styles)


def _cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--runs", nargs="+", required=True, help="run folders, each with test/flights.csv")
    p.add_argument("--out", required=True, help="path without extension; writes <out>.csv and <out>.xlsx")
    p.add_argument("--locations-file", default=str(LOCATIONS_FILE), help="lat;lon;name lines")
    args = p.parse_args(argv)

    named = places(args.locations_file)
    table = pd.concat([breakdown(Path(r), named) for r in args.runs], ignore_index=True)[COLUMNS]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out.with_suffix(".csv"), index=False)
    write_xlsx(table, out.with_suffix(".xlsx"))
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(table.to_string(index=False))
    print("wrote", out.with_suffix(".csv"), "and", out.with_suffix(".xlsx"))


if __name__ == "__main__":
    _cli()
