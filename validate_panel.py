#!/usr/bin/env python
"""Cohort-1 episode dry run against the real Bloomberg panel.

STANDALONE. Run it in the directory that holds the panel CSV; it imports
nothing from this repository, because the CSV cannot enter this repo (IP).

    python episode_dryrun.py --csv data.csv --out golden/
    python episode_dryrun.py --csv data.csv --emit-fomc

What it prints: per event and parameter cell, the raw trigger count, the
episode count after min_separation collapsing, the concentration statistics
and the episode DATE LIST. Dates and counts only -- never a level, a mean or
a standard deviation in native units.

Acceptance: the eff_n column must reproduce section 4 of
panel_validation_synthesis.md exactly. It computes the same quantities the
same way -- on the raw CSV, dropna, no PIT grid, no publication lag -- so any
difference between the two is a bug in one of them, not a tolerance to close.
`tests/scripts/test_episode_dryrun.py` pins that the trigger arithmetic here
is identical to `bpm.events.triggers`; nothing pins the eff_n numbers
themselves, because doing that needs the panel this script exists to keep
out of the repo.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

WINDOW = 252
MIN_SEPARATION = 5

# Tenor label ("2Y", ...) -> Bloomberg ticker. The Treasury names strip the
# trailing "Y" that TENORS below carries so both dicts can be built off one
# label list; the swap names use bare tenor numbers because Bloomberg's OIS
# curve tickers do not carry a "Y" suffix at all (USSO10, not USSO10Y).
TENOR_LABELS = ["2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
TREASURY = {t: "USGG%sYR Index" % t[:-1] for t in TENOR_LABELS}
SWAP = {t: "USSO%s Curncy" % t[:-1] for t in TENOR_LABELS}


def rolling_z(series: pd.Series, window: int) -> pd.Series:
    """Trailing z-score excluding the current observation.

    Mirrors bpm.events.triggers.rolling_z exactly, on purpose:
    tests/scripts/test_episode_dryrun.py asserts the two agree value for
    value on a shared fixture. If they ever diverge, one of them is wrong
    and the golden episode lists this script produces are not trustworthy
    evidence for the in-repo service.
    """
    mean = series.rolling(window, min_periods=window).mean().shift(1)
    sigma = series.rolling(window, min_periods=window).std().shift(1)
    return (series - mean) / sigma.replace(0.0, np.nan)


def z_move(series: pd.Series, n: int, window: int) -> pd.Series:
    """The sharp-move statistic: the z-score of the n-period change."""
    return rolling_z(series.diff(n), window)


def z_level(series: pd.Series, window: int) -> pd.Series:
    """The dislocation statistic: the z-score of the level itself."""
    return rolling_z(series, window)


def collapse(dates, min_separation_days: int) -> List[List[pd.Timestamp]]:
    """Mirrors bpm.events.clustering.collapse: chain from the cluster's LAST
    date, not its first, so a persistent dislocation collapses to one
    episode instead of one per day it stays extreme."""
    clusters: List[List[pd.Timestamp]] = []
    for date in sorted(pd.DatetimeIndex(dates).normalize().unique()):
        date = pd.Timestamp(date)
        if clusters and (date - clusters[-1][-1]).days <= min_separation_days:
            clusters[-1].append(date)
        else:
            clusters.append([date])
    return clusters


def load_panel(path: str, date_column: str = "date") -> pd.DataFrame:
    """Read the wide Bloomberg CSV into a sorted, numeric, date-indexed frame."""
    frame = pd.read_csv(path)
    columns = {str(c).strip().lower(): c for c in frame.columns}
    key = columns.get(date_column.lower(), frame.columns[0])
    frame[key] = pd.to_datetime(frame[key], errors="coerce")
    frame = frame.dropna(subset=[key]).set_index(key).sort_index()
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame.apply(pd.to_numeric, errors="coerce")


def column(frame: pd.DataFrame, name: str) -> Optional[pd.Series]:
    """Case/whitespace-insensitive column lookup, dropping NaN rows.

    The desk's CSV export is not guaranteed to spell a header identically to
    the contract's ticker string, so this matches loosely rather than
    letting a stray space silently drop an entire event from the report.
    """
    for candidate in frame.columns:
        if candidate.strip().lower() == name.strip().lower():
            return frame[candidate].dropna()
    return None


def cohort_masks(frame: pd.DataFrame, window: int = WINDOW) -> Dict[str, pd.Series]:
    """Every cohort-1 event at every registered parameter cell, keyed by a
    human-readable label ("<event_id> k=2.0 n=1", ...).

    Deliberately duplicates contract/events.yaml's compositions rather than
    reading them: this function has to run with nothing but pandas and
    numpy, in a directory this repo's code never sees. The label format is
    also read back by tests/events/test_golden_episodes.py to recover the
    event id and parameter cell from a golden file's own keys.

    fomc_decision is not here: it is a calendar event with no trigger
    arithmetic of its own, handled by emit_fomc below.
    """
    masks: Dict[str, pd.Series] = {}

    two, ten = column(frame, "USGG2YR Index"), column(frame, "USGG10YR Index")
    if two is not None and ten is not None:
        curve = pd.concat([two, ten], axis=1).dropna()
        curve.columns = ["two", "ten"]
        slope = curve["ten"] - curve["two"]
        level = (curve["ten"] + curve["two"]) / 2.0
        for n in (1, 5):
            slope_z = z_move(slope, n, window)
            change = level.diff(n)
            for k in (2.0, 2.5, 3.0):
                steep, flat = slope_z > k, slope_z < -k
                masks["bull_steepening n=%d k=%.1f" % (n, k)] = steep & (change < 0)
                masks["bear_steepening n=%d k=%.1f" % (n, k)] = steep & (change > 0)
                masks["bull_flattening n=%d k=%.1f" % (n, k)] = flat & (change < 0)
                masks["bear_flattening n=%d k=%.1f" % (n, k)] = flat & (change > 0)

    move = column(frame, "MOVE Index")
    if move is not None:
        for n in (1, 5):
            for k in (2.0, 2.5, 3.0):
                masks["move_spike n=%d k=%.1f" % (n, k)] = z_move(move, n, window) > k

    for label in TENOR_LABELS:
        treasury, swap = column(frame, TREASURY[label]), column(frame, SWAP[label])
        if treasury is None or swap is None:
            continue
        spread = ((treasury - swap) * 100.0).dropna()
        if len(spread) < window + 50:
            continue
        z = z_level(spread, window)
        for k in (2.0, 2.5, 3.0):
            masks["asw_dislocation tenor=%s k=%.1f" % (label.lower(), k)] = z.abs() > k

    repo, iorb = column(frame, "GCFRTSY Index"), column(frame, "IRRBIOER Index")
    if repo is not None and iorb is not None:
        spread = ((repo - iorb) * 100.0).dropna()
        z = z_level(spread, window)
        for k in (2.0, 2.5, 3.0):
            masks["repo_funding_squeeze k=%.1f" % k] = z > k

    cesi = column(frame, "CESIUSD Index")
    if cesi is not None:
        for k in (2.0, 2.5):
            masks["macro_surprise k=%.1f" % k] = z_move(cesi, 5, window).abs() > k

    return masks


def report(masks: Dict[str, pd.Series], min_separation: int) -> None:
    """Print raw trigger count, episode count and concentration shares.

    Percentages, not levels: max_cluster_share and max_year_share are
    unit-free by construction, which is what keeps this table inside the
    dates-and-counts-only constraint even though it summarizes every cell.
    """
    print("%-40s %6s %6s %9s %8s" % ("event [params]", "n", "eff_n", "maxclust", "maxyear"))
    for label, mask in masks.items():
        hits = mask[mask.fillna(False)].index
        if len(hits) == 0:
            print("%-40s %6s" % (label, "none"))
            continue
        clusters = collapse(hits, min_separation)
        firsts = [c[0] for c in clusters]
        years = pd.Series([d.year for d in firsts]).value_counts().sort_index()
        print(
            "%-40s %6d %6d %8.0f%% %7.0f%%"
            % (
                label,
                len(hits),
                len(clusters),
                100 * max(len(c) for c in clusters) / float(len(hits)),
                100 * years.max() / float(len(clusters)),
            )
        )
        print("      years: " + " ".join("%d:%d" % (y, c) for y, c in years.items()))


def emit_golden(masks: Dict[str, pd.Series], min_separation: int, window: int) -> str:
    """The episode date lists, as YAML, for tests/events/golden/.

    Every field is a date, a count, or a fixed metadata string -- never a
    level, mean or standard deviation of the underlying series. That is the
    standing IP constraint on this script and it is pinned by
    tests/scripts/test_episode_dryrun.py.
    """
    lines = []
    for label, mask in masks.items():
        hits = mask[mask.fillna(False)].index
        clusters = collapse(hits, min_separation)
        lines.append("%s:" % label)
        lines.append("  source: episode_dryrun.py on the desk panel")
        lines.append("  window_days: %d" % window)
        lines.append("  min_separation_days: %d" % min_separation)
        lines.append("  n_triggers: %d" % len(hits))
        lines.append("  effective_n: %d" % len(clusters))
        lines.append("  dates:")
        for cluster in clusters:
            lines.append('    - "%s"' % cluster[0].date())
    return "\n".join(lines) + "\n"


def emit_fomc(frame: pd.DataFrame) -> str:
    """A DRAFT of contract/fomc_dates.yaml's date list, from FDTR changes.

    Every hold meeting is missing by construction: a hold moves nothing, so
    it leaves no trace in FDTR's own first difference. `bpm.events.fomc.
    check_dates` will say so once this draft is pasted in; finding the hold
    dates from federalreserve.gov and adding them is the human half of the
    ratification this script cannot do on its own.
    """
    fdtr = column(frame, "FDTR Index")
    if fdtr is None:
        return "# FDTR Index is not in this panel; no draft can be made.\n"
    changed = fdtr.diff().fillna(0.0) != 0.0
    lines = ["# DRAFT: FDTR change dates only. Every hold meeting is missing.", "dates:"]
    lines.extend('  - "%s"' % d.date() for d in fdtr.index[changed])
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--min-separation", type=int, default=MIN_SEPARATION)
    parser.add_argument("--out", default=None, help="write golden_episodes.yaml here")
    parser.add_argument("--emit-fomc", action="store_true")
    args = parser.parse_args(argv)

    frame = load_panel(args.csv, args.date_column)
    print(
        "panel: %d rows, %d columns, %s to %s"
        % (frame.shape[0], frame.shape[1], frame.index.min().date(), frame.index.max().date())
    )

    if args.emit_fomc:
        print(emit_fomc(frame))
        return 0

    masks = cohort_masks(frame, args.window)
    report(masks, args.min_separation)
    if args.out:
        # Create the directory rather than dying on FileNotFoundError after
        # the whole panel has already been scored. This runs on the user's
        # own machine against a panel this repo never sees, so the last step
        # is the worst possible place to lose a completed run to a typo.
        os.makedirs(args.out, exist_ok=True)
        path = "%s/golden_episodes.yaml" % args.out.rstrip("/")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(emit_golden(masks, args.min_separation, args.window))
        print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
