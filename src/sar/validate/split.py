"""
split.py -- the sealed drifter validation set (issue #51).

Writes <data>/derived/validation_split.csv: one row per validation unit, saying
whether it is `dev`, `sealed` or `holdout-2023`.

    python -m sar.validate.split --data C:/maritime-data
    python -m sar.validate.split --data C:/maritime-data --seed 20260923 --sealed-fraction 0.25

WHY IT EXISTS. D018: "Reserve a subset. There is no excuse for reporting on
tracks that were used while debugging." Every drifter track anyone looks at
while tuning the engine stops being an independent test of it. So the tracks
the paper reports on are chosen HERE, before the first engine run, and the
hindcast scorer refuses them unless told otherwise.

THREE KINDS OF UNIT
-------------------
  dev            hourly 2019 - Oct 2022; debug on these freely
  sealed         hourly 2019 - Oct 2022; reported, never debugged on
  holdout-2023   every 6-hourly unit, Nov 2022 - 2023: a year never developed on,
                 reported separately, the strongest test there is

WHAT A UNIT IS. A contiguous in-box run of 48 h or more (duration, not fix
count: 48 hourly fixes span 47 h, and a 47 h track cannot verify a 48 h
forecast), and of ONE drogue tier. A run that loses its drogue part-way is cut
at the loss, because D018's test compares drogued with undrogued and a mixed
run is neither. Measured 23 Sep: 38 runs are mixed, and cutting them adds 56
pure pieces of 48 h or more.

WHY GROUPS, NOT BUOYS. Buoys drift together. On 23 Sep, 52 buoy pairs came
within 10 km of each other at the same hour, one cluster of 14 almost certainly
deployed together. 10 km is about one HYCOM cell diagonal (sqrt(8^2 + 4.5^2) =
9.2 km), so two buoys that close are forced by the same current values:
debugging on one is debugging on the other. So buoys that ever shared water are
grouped, and a group goes wholly to dev or wholly to sealed. At 25 km the pairs
chain into a group of 68, which would make the split too lumpy to stratify.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from sar.fetch.drifters import output_paths
from sar.utils.data_io import combine_products, load_drifters

MIN_HOURS = 48.0
SHARED_WATER_KM = 10.0
SEALED_FRACTION = 0.25
# The date the split was first drawn. Changing it redraws the split, which is
# only honest before the first engine run: the seed and the sealed list are
# recorded in the vault decision for exactly that reason.
DEFAULT_SEED = 20260923

# The study window each product was fetched for. The file names follow from these
# through `sar.fetch.drifters.output_paths`, so there is one source for them.
HOURLY_WINDOW = ("2019-01-01", "2024-01-01")
SIX_HOURLY_WINDOW = ("2022-11-01", "2024-01-01")

EARTH_RADIUS_KM = 6371.0
SPLITS = ("dev", "sealed", "holdout-2023")


def label_pieces(df: pd.DataFrame) -> pd.Series:
    """A unit key for every fix: its segment, cut wherever the drogue tier changes.

    The key is `ID@start`, the buoy and the first fix of the piece. It depends only
    on the buoy's own record, not on row order or on which other buoys were
    loaded, so the same piece gets the same key after a re-pull.
    """
    # By POSITION, not index label: a table joined from two products can repeat
    # labels, and aligning on them would scramble the keys.
    order = np.lexsort((df["time"].to_numpy(), df["ID"].astype(str).to_numpy()))
    d = df.iloc[order]
    seg, und = d["segment_id"].to_numpy(), d["undrogued"].to_numpy()
    cut = np.r_[True, (seg[1:] != seg[:-1]) | (und[1:] != und[:-1])]
    piece = np.cumsum(cut)
    start = d["time"].groupby(piece).transform("min").to_numpy()
    keys = d["ID"].astype(str).to_numpy() + "@" + pd.DatetimeIndex(start).strftime("%Y-%m-%dT%H:%M")
    out = np.empty(len(df), dtype=object)
    out[order] = keys
    return pd.Series(out, index=df.index)


def validation_units(df: pd.DataFrame, min_hours: float = MIN_HOURS) -> pd.DataFrame:
    """One row per validation unit: a single-tier in-box run of at least `min_hours`."""
    # Sorted by time within each buoy, so `first` is the earliest fix, not the
    # first row that happened to arrive.
    d = df.assign(unit=label_pieces(df).to_numpy()).sort_values(["ID", "time"], kind="stable")
    g = d.groupby("unit", sort=True)
    units = pd.DataFrame({
        "ID": g["ID"].first(),
        "product": g["product"].first(),
        "start": g["time"].min(),
        "end": g["time"].max(),
        "fixes": g.size(),
        "lat0": g["lat"].first(),
        "lon0": g["lon"].first(),
        "uncertain": g["tier_uncertain"].any(),
        "undrogued": g["undrogued"].all(),
    })
    units["hours"] = (units["end"] - units["start"]).dt.total_seconds() / 3600.0
    units["tier"] = np.where(units["uncertain"], "uncertain",
                             np.where(units["undrogued"], "undrogued", "drogued"))
    units = units[units["hours"] >= min_hours].drop(columns=["uncertain", "undrogued"])
    return units.reset_index()


def _pairwise_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Great-circle distance between every pair of positions, in km."""
    la, lo = np.radians(lat), np.radians(lon)
    dla = la[:, None] - la[None, :]
    dlo = lo[:, None] - lo[None, :]
    a = np.sin(dla / 2) ** 2 + np.cos(la[:, None]) * np.cos(la[None, :]) * np.sin(dlo / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def shared_water_pairs(df: pd.DataFrame, km: float = SHARED_WATER_KM) -> set[tuple[str, str]]:
    """Every pair of buoys that were within `km` of each other at the same timestamp.

    Pairs come back ordered, `(a, b)` with `a < b`, so the set holds each once.
    """
    pairs: set[tuple[str, str]] = set()
    for _, at in df.groupby("time"):
        if at["ID"].nunique() < 2:
            continue
        ids = at["ID"].astype(str).to_numpy()
        close = _pairwise_km(at["lat"].to_numpy(float), at["lon"].to_numpy(float)) < km
        i, j = np.nonzero(np.triu(close, 1))
        for a, b in zip(ids[i], ids[j]):
            if a != b:
                pairs.add((a, b) if a < b else (b, a))
    return pairs


def buoy_groups(ids, pairs: set[tuple[str, str]]) -> dict[str, int]:
    """Buoys joined by any chain of shared water, as `ID -> group`.

    A buoy that never came near another is a group of one. Groups are numbered in
    the order of their smallest member ID, so the numbering is deterministic.
    """
    names = sorted({str(i) for i in ids} | {b for p in pairs for b in p})
    index = {name: k for k, name in enumerate(names)}
    rows = [index[a] for a, _ in pairs]
    cols = [index[b] for _, b in pairs]
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(names), len(names)))
    _, labels = connected_components(graph, directed=False)
    first_seen: dict[int, int] = {}
    for label in labels:                      # names are sorted, so this is by smallest ID
        first_seen.setdefault(label, len(first_seen))
    return {name: first_seen[labels[k]] for name, k in index.items()}


def assign_split(units: pd.DataFrame, groups: dict[str, int], *, seed: int = DEFAULT_SEED,
                 sealed_fraction: float = SEALED_FRACTION) -> pd.Series:
    """`dev`, `sealed` or `holdout-2023` for every unit.

    Every 6-hourly unit is `holdout-2023`. The hourly units are split BY GROUP: the
    groups are shuffled with `seed`, then taken in turn, and a group goes to
    `sealed` while its main tier still has fewer than `sealed_fraction` of that
    tier's units sealed. That keeps drogued and undrogued both near the target
    rather than letting the larger tier fill the sealed set.
    """
    if not 0.0 < sealed_fraction < 1.0:
        raise ValueError(f"sealed_fraction must be between 0 and 1, got {sealed_fraction}")
    split = pd.Series("holdout-2023", index=units.index, dtype=object)
    hourly = units[units["product"] == "hourly"]
    group_of = hourly["ID"].astype(str).map(groups)
    if group_of.isna().any():
        missing = hourly.loc[group_of.isna(), "ID"].unique()[:3]
        raise ValueError(f"units for buoys with no group, e.g. {list(missing)}")

    total = hourly["tier"].value_counts()
    sealed = pd.Series(0, index=total.index)
    rng = np.random.default_rng(seed)
    for g in rng.permutation(np.sort(group_of.unique())):
        members = hourly.index[group_of == g]
        tiers = hourly.loc[members, "tier"].value_counts()
        main = sorted(tiers.index, key=lambda t: (-tiers[t], t))[0]
        to_sealed = sealed[main] < sealed_fraction * total[main]
        split.loc[members] = "sealed" if to_sealed else "dev"
        if to_sealed:
            sealed = sealed.add(tiers, fill_value=0)
    return split


def check_split(units: pd.DataFrame, groups: dict[str, int], *, min_sealed: int = 10) -> None:
    """Refuse a split that would not be an honest test. Raises ValueError, says why.

    - no shared-water group, and so no buoy, has hourly units on both sides;
    - the sealed set holds at least `min_sealed` units (R6d) and both tiers;
    - every 6-hourly unit is the 2023 holdout, and no hourly unit is.
    """
    bad = set(units["split"]) - set(SPLITS)
    if bad:
        raise ValueError(f"unknown split label(s) {sorted(bad)}")
    hourly = units[units["product"] == "hourly"]
    sides = hourly.groupby(hourly["ID"].astype(str).map(groups))["split"].nunique()
    if (sides > 1).any():
        raise ValueError(f"{int((sides > 1).sum())} shared-water group(s) straddle dev and sealed")
    if (hourly["split"] == "holdout-2023").any():
        raise ValueError("an hourly unit is labelled holdout-2023")
    six = units[units["product"] == "6-hourly"]
    if (six["split"] != "holdout-2023").any():
        raise ValueError("a 6-hourly unit is not in the 2023 holdout")

    sealed = hourly[hourly["split"] == "sealed"]
    if len(sealed) < min_sealed:
        raise ValueError(f"only {len(sealed)} sealed units; R6d needs at least {min_sealed}")
    for tier in ("drogued", "undrogued"):
        if not (sealed["tier"] == tier).any():
            raise ValueError(f"no {tier} unit is sealed; D018's comparison needs both tiers")


def summarise(units: pd.DataFrame) -> pd.DataFrame:
    """Units per split and tier, with the number of distinct buoys behind each split."""
    table = pd.crosstab(units["split"], units["tier"], margins=True, margins_name="all")
    table["buoys"] = units.groupby("split")["ID"].nunique().reindex(table.index)
    table.loc["all", "buoys"] = units["ID"].nunique()
    return table


def load_study_drifters(data: str | Path) -> pd.DataFrame:
    """Both GDP products for the study window, with their buoy tables, combined."""
    h_tracks, h_buoys = output_paths(data, "hourly", *HOURLY_WINDOW)
    s_tracks, s_buoys = output_paths(data, "6-hourly", *SIX_HOURLY_WINDOW)
    hourly = load_drifters(h_tracks, product="hourly", buoys=h_buoys)
    six = load_drifters(s_tracks, product="6-hourly", buoys=s_buoys)
    return combine_products(hourly, six)


def build(df: pd.DataFrame, *, seed: int = DEFAULT_SEED, sealed_fraction: float = SEALED_FRACTION,
          min_sealed: int = 10) -> tuple[pd.DataFrame, dict[str, int]]:
    """Units, shared-water groups and the split, checked. The whole pipeline."""
    units = validation_units(df)
    in_units = df.assign(unit=label_pieces(df).to_numpy())
    in_units = in_units[in_units["unit"].isin(units["unit"])]
    groups = buoy_groups(units["ID"], shared_water_pairs(in_units))
    units["group"] = units["ID"].astype(str).map(groups)
    units["split"] = assign_split(units, groups, seed=seed, sealed_fraction=sealed_fraction)
    check_split(units, groups, min_sealed=min_sealed)
    return units, groups


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--data", required=True, help="archive root holding raw/, e.g. C:/maritime-data")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--sealed-fraction", type=float, default=SEALED_FRACTION)
    p.add_argument("--out", help="CSV to write (default <data>/derived/validation_split.csv)")
    args = p.parse_args()

    units, groups = build(load_study_drifters(args.data), seed=args.seed,
                          sealed_fraction=args.sealed_fraction)
    out = Path(args.out) if args.out else Path(args.data) / "derived" / "validation_split.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    units.to_csv(out, index=False)

    sizes = pd.Series(list(groups.values())).value_counts()
    print(f"seed {args.seed}, sealed fraction {args.sealed_fraction}")
    print(f"shared-water groups: {sizes.size} ({int((sizes > 1).sum())} with more than one buoy, "
          f"largest {int(sizes.max())})")
    print(summarise(units).to_string())
    sealed = units[units["split"] == "sealed"]
    print(f"\nsealed buoys ({sealed['ID'].nunique()}): {', '.join(sorted(sealed['ID'].unique()))}")
    print(f"\nwrote {out} ({len(units)} units)")


if __name__ == "__main__":
    main()
