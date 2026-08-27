#!/usr/bin/env python
"""Validate the proposed event cohort against the real Bloomberg panel.

Run this in the directory that holds the CSV. It depends only on pandas and
numpy -- nothing from this repository -- so it is safe to copy anywhere.

    python validate_panel.py path/to/panel.csv > synthesis.txt

WHAT IT ANSWERS
---------------
1. Panel inventory: which of the tickers we care about are present, over what
   span, at what frequency, and with what shape. This resolves the questions a
   start-date table cannot: is LUMSMD daily or monthly, are the PD* series
   levels or changes, is anything a step function or a pegged feed.
2. The ASW oracle: does USSFCT{n} really equal -(USGG{n}YR - USS0{n}) * 100?
   That single check pins the sign convention AND the unit convention for
   every asset-swap measure in the contract, over the ~5-year overlap.
3. Fat-tail measurement: the analytic power estimates assumed normality.
   This measures the true exceedance rate at each k, which is what actually
   determines episode counts.
4. Episode counts for the proposed cohort, with min_separation collapsing,
   year histograms and concentration shares -- the power pre-check, run a
   priori, before we commit to an event registry.

IP SAFETY
---------
Output is deliberately restricted to counts, dates, unit-free ratios
(kurtosis, exceedance rates, correlations, z-scores) and validation
residuals. No raw level, mean, or standard deviation in native units is ever
printed, so the synthesis can be pasted back without disclosing the panel.
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Tickers we need, grouped by the question each group answers.
# --------------------------------------------------------------------------

TREASURY = {
    "2Y": "USGG2YR", "3Y": "USGG3YR", "5Y": "USGG5YR", "7Y": "USGG7YR",
    "10Y": "USGG10YR", "20Y": "USGG20YR", "30Y": "USGG30YR",  # 30Y expected absent
}
SWAP = {
    "2Y": "USS02", "3Y": "USS03", "5Y": "USS05", "7Y": "USS07",
    "10Y": "USS010", "20Y": "USS020", "30Y": "USS030",
}
SWAP_SPREAD = {  # Bloomberg's own swap spread, our independent oracle
    "2Y": "USSFCT02", "3Y": "USSFCT03", "5Y": "USSFCT05", "7Y": "USSFCT07",
    "10Y": "USSFCT10", "20Y": "USSFCT20", "30Y": "USSFCT30",
}

# Cohort 1 drivers plus the cohort 2 candidates whose frequency or meaning is
# unresolved. Inventorying the cohort 2 names now costs one extra run of the
# same code and is what lets us finalise the second cohort without a re-run.
OTHER = [
    "USGG12M", "USS01",                 # front end
    "MOVE", "VIX", "NDX", "DXY",        # vol / flight-to-quality
    "GCFRTSY", "IRRBIOER", "SOFRRATE", "US0003M",   # funding
    "LUMSMD",                            # convexity  (frequency UNCONFIRMED)
    "KIMWTP10",                          # term premium (revised -- see notes)
    "CESIUSD",                           # macro surprise proxy
    "FARBAST",                           # balance sheet
    "GDBR10", "GBTPGR10",                # cross-market / peripheral
    "PDPPTOTG", "PDPPTOTF", "PDPCPCCS", "PDPPPCCS", "PDPPCC11", "PDPPCC3-6",
    "FDTR", "FEDL01",                    # expected absent; confirms the request
]

SUFFIXES = (" INDEX", " CURNCY", " COMDTY", " EQUITY", " GOVT")


# --------------------------------------------------------------------------
# Column resolution. The CSV's headers may or may not carry the Bloomberg
# suffix, may differ in case, and may have collapsed whitespace.
# --------------------------------------------------------------------------

def _norm(name: object) -> str:
    return re.sub(r"\s+", " ", str(name)).strip().upper()


def _strip_suffix(name: str) -> str:
    for suffix in SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)].strip()
    return name


def build_column_index(columns: Sequence[object]) -> Dict[str, object]:
    """Map a bare ticker (USGG10YR) to whatever the CSV actually calls it."""
    index: Dict[str, object] = {}
    for col in columns:
        normalized = _norm(col)
        for key in {normalized, _strip_suffix(normalized)}:
            index.setdefault(key, col)
    return index


def resolve(index: Dict[str, object], ticker: str) -> Optional[object]:
    return index.get(_norm(ticker)) or index.get(_strip_suffix(_norm(ticker)))


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_panel(path: str, date_column: Optional[str],
               dayfirst: bool = False) -> pd.DataFrame:
    frame = pd.read_csv(path)

    if date_column is None:
        # Take the first column that parses as dates for most of its rows.
        for candidate in frame.columns:
            parsed = pd.to_datetime(frame[candidate], errors="coerce", dayfirst=dayfirst)
            if parsed.notna().mean() > 0.9:
                date_column = candidate
                break
    if date_column is None:
        raise SystemExit(
            "Could not identify a date column. Re-run with --date-column NAME.\n"
            "Columns seen: " + ", ".join(str(c) for c in frame.columns[:12])
        )

    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce", dayfirst=dayfirst)
    frame = frame.dropna(subset=[date_column]).set_index(date_column).sort_index()
    frame.index.name = "date"

    # Everything else must be numeric; Bloomberg pulls often carry '#N/A' strings.
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


# --------------------------------------------------------------------------
# Section 1 -- inventory
# --------------------------------------------------------------------------

def infer_frequency(series: pd.Series) -> str:
    observed = series.dropna()
    if len(observed) < 3:
        return "insufficient"
    gaps = pd.Series(observed.index).diff().dt.days.dropna()
    if gaps.empty:
        return "insufficient"
    median_gap = float(gaps.median())
    if median_gap <= 1.5:
        return "daily"
    if median_gap <= 4.5:
        return "business-daily"
    if median_gap <= 10:
        return "weekly"
    if median_gap <= 20:
        return "biweekly"
    if median_gap <= 45:
        return "monthly"
    if median_gap <= 120:
        return "quarterly"
    return "sparser-than-quarterly"


def describe(name: str, series: Optional[pd.Series]) -> Dict[str, object]:
    if series is None:
        return {"ticker": name, "present": False}

    observed = series.dropna()
    if observed.empty:
        return {"ticker": name, "present": True, "n_obs": 0}

    changed = observed.diff().dropna()
    # A step function (a policy rate) barely ever changes; a pegged or stale
    # feed holds its last value for long runs. Both break a rolling-sigma
    # trigger, and both are invisible in a start-date table.
    flat_share = float((changed == 0).mean()) if len(changed) else float("nan")
    longest_flat = 0
    if len(changed):
        run = 0
        for value in (changed == 0).to_numpy():
            run = run + 1 if value else 0
            longest_flat = max(longest_flat, run)

    return {
        "ticker": name,
        "present": True,
        "n_obs": int(len(observed)),
        "start": observed.index.min().date(),
        "end": observed.index.max().date(),
        "freq": infer_frequency(observed),
        "n_unique": int(observed.nunique()),
        "flat_share": flat_share,
        "longest_flat_run": int(longest_flat),
        # Sign profile identifies a level (yields: all positive) versus a change
        # or a spread (mixed sign). This is how we tell whether the PD* series
        # are positions or changes in positions.
        "pct_negative": float((observed < 0).mean()),
    }


def section_inventory(frame: pd.DataFrame, index: Dict[str, object]) -> None:
    print("=" * 78)
    print("SECTION 1 -- PANEL INVENTORY")
    print("=" * 78)
    print("%-12s %-4s %7s %-11s %-11s %-15s %6s %6s %5s"
          % ("ticker", "have", "n_obs", "start", "end", "freq", "flat%", "maxflat", "neg%"))

    wanted: List[str] = (
        list(TREASURY.values()) + list(SWAP.values()) + list(SWAP_SPREAD.values()) + OTHER
    )
    missing: List[str] = []
    for ticker in wanted:
        column = resolve(index, ticker)
        info = describe(ticker, frame[column] if column is not None else None)
        if not info.get("present"):
            missing.append(ticker)
            print("%-12s %-4s" % (ticker, "NO"))
            continue
        if not info.get("n_obs"):
            # The column exists but is entirely empty. Worth distinguishing
            # from absent: it usually means the vendor returned the ticker
            # with no data rather than the request being wrong.
            missing.append(ticker + " (column present, all-NaN)")
            print("%-12s %-4s %7d  -- column present but empty --" % (ticker, "yes", 0))
            continue
        print("%-12s %-4s %7d %-11s %-11s %-15s %5.1f%% %6d %4.0f%%" % (
            ticker, "yes", info["n_obs"], info["start"], info["end"], info["freq"],
            100 * info["flat_share"], info["longest_flat_run"], 100 * info["pct_negative"],
        ))

    print("\nMISSING (%d): %s" % (len(missing), ", ".join(missing) if missing else "none"))
    print("\nTotal columns in CSV: %d   Date span: %s -> %s"
          % (frame.shape[1], frame.index.min().date(), frame.index.max().date()))

    # Name the columns we never asked about, so the registry work has a list.
    known = {_norm(t) for t in wanted} | {_strip_suffix(_norm(t)) for t in wanted}
    unclassified = [str(c) for c in frame.columns
                    if _norm(c) not in known and _strip_suffix(_norm(c)) not in known]
    print("Columns not in our target list (%d):" % len(unclassified))
    for i in range(0, len(unclassified), 4):
        print("   " + " | ".join(unclassified[i:i + 4]))


# --------------------------------------------------------------------------
# Section 2 -- the ASW oracle
# --------------------------------------------------------------------------

def section_asw_oracle(frame: pd.DataFrame, index: Dict[str, object]) -> None:
    print("\n" + "=" * 78)
    print("SECTION 2 -- ASW SIGN AND UNIT ORACLE")
    print("=" * 78)
    print("Testing   USSFCT{n}  ==  -(USGG{n}YR - USS0{n}) * 100   [bp]")
    print("A near-zero residual confirms BOTH the asset-swap sign convention and")
    print("that USSFCT is in bp while the yield legs are in percent.\n")
    print("%-5s %6s %-11s %-11s %10s %10s %10s"
          % ("tenor", "n_days", "overlap_from", "overlap_to", "corr", "mae_bp", "med_bias_bp"))

    for tenor in ["2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]:
        treasury_col = resolve(index, TREASURY[tenor])
        swap_col = resolve(index, SWAP[tenor])
        spread_col = resolve(index, SWAP_SPREAD[tenor])
        if treasury_col is None or swap_col is None or spread_col is None:
            have = [n for n, c in (("UST", treasury_col), ("SWAP", swap_col),
                                   ("SPRD", spread_col)) if c is not None]
            print("%-5s  skipped -- have only: %s" % (tenor, ", ".join(have) or "nothing"))
            continue

        computed = (frame[treasury_col] - frame[swap_col]) * 100.0
        reported = frame[spread_col]
        # Compare against the NEGATED computed value: the asset-swap convention
        # (treasury - swap) is the opposite sign to Bloomberg's swap spread.
        pair = pd.concat([-computed, reported], axis=1).dropna()
        pair.columns = ["computed_negated", "reported"]
        if len(pair) < 30:
            print("%-5s  overlap too short (%d rows)" % (tenor, len(pair)))
            continue

        difference = pair["reported"] - pair["computed_negated"]
        print("%-5s %6d %-11s %-11s %10.4f %10.2f %10.2f" % (
            tenor, len(pair), pair.index.min().date(), pair.index.max().date(),
            float(pair["computed_negated"].corr(pair["reported"])),
            float(difference.abs().mean()), float(difference.median()),
        ))

    print("\nReading it: corr ~ +1.0 with a small mae confirms the convention.")
    print("corr ~ -1.0 means the sign is inverted from what we assumed.")
    print("A large mae with corr ~ +1.0 means a unit or basis mismatch, not a sign error.")


# --------------------------------------------------------------------------
# Section 3 -- trigger primitives (the same four Plan 2 will implement)
# --------------------------------------------------------------------------

def rolling_z(series: pd.Series, window: int) -> pd.Series:
    """Trailing z-score whose mean and sigma EXCLUDE the current observation.

    The shift(1) matters. Including t in its own estimation window shrinks the
    z-score of exactly the extreme point the trigger exists to find, and biases
    every episode count downward. It is also the causal choice: at time t we
    are scoring today's value against what we knew yesterday.
    """
    mean = series.rolling(window, min_periods=window).mean().shift(1)
    sigma = series.rolling(window, min_periods=window).std().shift(1)
    return (series - mean) / sigma.replace(0.0, np.nan)


def z_move(series: pd.Series, n: int, window: int) -> pd.Series:
    """z-score of the n-period change -- the 'sharp move' primitive."""
    return rolling_z(series.diff(n), window)


def z_level(series: pd.Series, window: int) -> pd.Series:
    """z-score of the level against its own trailing mean -- 'dislocation'."""
    return rolling_z(series, window)


def collapse(dates: pd.DatetimeIndex, min_separation_days: int) -> List[List[pd.Timestamp]]:
    """Group trigger dates into episodes: anything within min_separation is one.

    This is the anti-double-counting rule from spec section 7, and n_clusters
    (the length of the returned list) is the definition of effective_n.
    """
    clusters: List[List[pd.Timestamp]] = []
    for date in sorted(dates):
        if clusters and (date - clusters[-1][-1]).days <= min_separation_days:
            clusters[-1].append(date)
        else:
            clusters.append([date])
    return clusters


def report_episodes(label: str, mask: pd.Series, min_separation_days: int,
                    show_years: bool = True) -> Optional[Dict[str, object]]:
    hits = mask[mask.fillna(False)].index
    if len(hits) == 0:
        print("%-42s  no triggers" % label)
        return None

    clusters = collapse(pd.DatetimeIndex(hits), min_separation_days)
    first_dates = [c[0] for c in clusters]
    years = pd.Series([d.year for d in first_dates])
    year_counts = years.value_counts().sort_index()

    max_cluster_share = max(len(c) for c in clusters) / float(len(hits))
    max_year_share = float(year_counts.max()) / float(len(clusters))

    print("%-42s  n=%4d  eff_n=%4d  maxclust=%4.0f%%  maxyear=%4.0f%%  %s..%s"
          % (label, len(hits), len(clusters), 100 * max_cluster_share,
             100 * max_year_share, first_dates[0].date(), first_dates[-1].date()))
    if show_years and len(clusters) > 0:
        compact = " ".join("%d:%d" % (y, c) for y, c in year_counts.items())
        print("      years: %s" % compact)
    return {"n": len(hits), "eff_n": len(clusters),
            "max_cluster_share": max_cluster_share, "max_year_share": max_year_share}


# --------------------------------------------------------------------------
# Section 3a -- fat tails
# --------------------------------------------------------------------------

def section_fat_tails(frame: pd.DataFrame, index: Dict[str, object],
                      window: int, horizons: Sequence[int]) -> None:
    print("\n" + "=" * 78)
    print("SECTION 3 -- FAT TAILS: TRUE ONE-SIDED EXCEEDANCE RATE vs NORMAL")
    print("=" * 78)
    print("Normal one-sided reference:  k=2 -> 2.275%   k=2.5 -> 0.621%   k=3 -> 0.135%")
    print("The ratio column is what multiplies every analytic episode estimate.\n")
    print("%-12s %2s %8s %8s %8s %8s %7s"
          % ("series", "n", "kurtosis", "k>2", "k>2.5", "k>3", "x-norm@3"))

    drivers = ["USGG2YR", "USGG10YR", "MOVE", "VIX", "GCFRTSY", "LUMSMD",
               "KIMWTP10", "CESIUSD", "PDPPTOTG"]
    for ticker in drivers:
        column = resolve(index, ticker)
        if column is None:
            continue
        for n in horizons:
            z = z_move(frame[column].dropna(), n, window).dropna()
            if len(z) < window:
                continue
            rates = [float((z > k).mean()) for k in (2.0, 2.5, 3.0)]
            ratio = rates[2] / 0.00135 if rates[2] > 0 else 0.0
            print("%-12s %2d %8.2f %7.3f%% %7.3f%% %7.3f%% %6.1fx" % (
                ticker, n, float(z.kurtosis()),
                100 * rates[0], 100 * rates[1], 100 * rates[2], ratio,
            ))


# --------------------------------------------------------------------------
# Section 4 -- the proposed cohort
# --------------------------------------------------------------------------

def section_cohort(frame: pd.DataFrame, index: Dict[str, object],
                   window: int, min_separation_days: int) -> None:
    print("\n" + "=" * 78)
    print("SECTION 4 -- PROPOSED COHORT 1: EPISODE COUNTS")
    print("=" * 78)
    print("eff_n = n_clusters after %d-day min_separation collapsing." % min_separation_days)
    print("maxclust = largest single episode's share of raw triggers.")
    print("maxyear  = busiest calendar year's share of episodes.")
    print("")
    print("NOTE on reading n vs eff_n. For a MOVE event (a sharp change) n is")
    print("roughly the episode count already. For a LEVEL event (a dislocation)")
    print("the series stays above threshold for as long as the dislocation")
    print("lasts, so n counts DAYS-IN-STATE while eff_n counts ENTRIES-INTO-")
    print("STATE. eff_n is the number that matters in both cases; a large")
    print("n/eff_n ratio on a level event is persistence, not double-counting.\n")

    two_year = resolve(index, "USGG2YR")
    ten_year = resolve(index, "USGG10YR")

    # -- Events 1-4: bull/bear steepening and flattening -------------------
    if two_year is not None and ten_year is not None:
        print("-- Events 1-4: curve moves, split by direction AND driver -------------")
        curve = frame[[two_year, ten_year]].dropna()
        slope = curve[ten_year] - curve[two_year]          # 2s10s, percent
        level = (curve[ten_year] + curve[two_year]) / 2.0  # average yield

        for n in (1, 5):
            slope_z = z_move(slope, n, window)
            level_change = level.diff(n)
            for k in (2.0, 2.5, 3.0):
                steepening = slope_z > k
                flattening = slope_z < -k
                # bull = yields fell, bear = yields rose. The level condition
                # only PARTITIONS the population; it is not itself a k-sigma
                # test, so the two halves sum back to the unsplit event.
                for name, mask in (
                    ("bull_steepening", steepening & (level_change < 0)),
                    ("bear_steepening", steepening & (level_change > 0)),
                    ("bull_flattening", flattening & (level_change < 0)),
                    ("bear_flattening", flattening & (level_change > 0)),
                ):
                    report_episodes("%s n=%d k=%.1f" % (name, n, k), mask,
                                    min_separation_days, show_years=False)
            print("")
    else:
        print("-- Events 1-4 SKIPPED: need USGG2YR and USGG10YR\n")

    # -- Event 5: move_spike ----------------------------------------------
    move = resolve(index, "MOVE")
    if move is not None:
        print("-- Event 5: move_spike ------------------------------------------------")
        series = frame[move].dropna()
        for n in (1, 5):
            for k in (2.0, 2.5, 3.0):
                report_episodes("move_spike n=%d k=%.1f" % (n, k),
                                z_move(series, n, window) > k,
                                min_separation_days, show_years=(k == 2.5 and n == 5))
        print("")
    else:
        print("-- Event 5 SKIPPED: MOVE absent\n")

    # -- Event 6: asw_dislocation, per tenor -------------------------------
    print("-- Event 6: asw_dislocation (per tenor) -------------------------------")
    for tenor in ["2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]:
        treasury_col = resolve(index, TREASURY[tenor])
        swap_col = resolve(index, SWAP[tenor])
        if treasury_col is None or swap_col is None:
            print("asw_%-4s  SKIPPED (missing leg)" % tenor.lower())
            continue
        asw = ((frame[treasury_col] - frame[swap_col]) * 100.0).dropna()  # bp
        if len(asw) < window + 50:
            print("asw_%-4s  span too short for a %d-day window (%d obs)"
                  % (tenor.lower(), window, len(asw)))
            continue
        z = z_level(asw, window)
        for k in (2.0, 2.5, 3.0):
            report_episodes("asw_dislocation %s k=%.1f" % (tenor, k),
                            z.abs() > k, min_separation_days,
                            show_years=(k == 2.5))
    print("")

    # -- Event 7: repo_funding_squeeze -------------------------------------
    repo = resolve(index, "GCFRTSY")
    iorb = resolve(index, "IRRBIOER")
    if repo is not None and iorb is not None:
        print("-- Event 7: repo_funding_squeeze --------------------------------------")
        spread = (frame[repo] - frame[iorb]).dropna() * 100.0  # bp
        z = z_level(spread, window)
        for k in (2.0, 2.5, 3.0):
            report_episodes("repo_funding_squeeze k=%.1f" % k, z > k,
                            min_separation_days, show_years=(k == 2.5))
        print("")
    else:
        print("-- Event 7 SKIPPED: need GCFRTSY and IRRBIOER\n")

    # Event 8, fomc_decision, is a calendar event with no data dependency:
    # its episode count is simply the number of FOMC decision dates in the
    # span (~8/year). Nothing here can validate or invalidate it.
    print("-- Event 8 (fomc_decision): calendar event, no data dependency. ------")
    print("   Episode count is the FOMC date list itself; nothing to validate.")


# --------------------------------------------------------------------------
# Section 5 -- cohort 2 candidates
# --------------------------------------------------------------------------

def section_cohort2(frame: pd.DataFrame, index: Dict[str, object],
                    window: int, min_separation_days: int) -> None:
    print("\n" + "=" * 78)
    print("SECTION 5 -- COHORT 2 CANDIDATES (exploratory)")
    print("=" * 78)
    print("Counted now so the second cohort can be finalised without a re-run.")
    print("Frequency matters here: an n-day trigger on a weekly series is")
    print("meaningless, so read these against the freq column in Section 1.\n")

    single = [
        ("convexity_hedging_shock", "LUMSMD", "move"),
        ("term_premium_shock", "KIMWTP10", "move"),
        ("macro_surprise (CESI)", "CESIUSD", "move"),
        ("balance_sheet_pace_shift", "FARBAST", "move"),
        ("dealer_inventory_extreme", "PDPPTOTG", "level"),
        ("front_end_repricing", "USGG12M", "move"),
    ]
    for label, ticker, kind in single:
        column = resolve(index, ticker)
        if column is None:
            print("%-28s SKIPPED (%s absent)" % (label, ticker))
            continue
        series = frame[column].dropna()
        if len(series) < window + 50:
            print("%-28s too short (%d obs)" % (label, len(series)))
            continue
        z = z_move(series, 5, window) if kind == "move" else z_level(series, window)
        for k in (2.0, 2.5):
            report_episodes("%s k=%.1f" % (label, k), z.abs() > k,
                            min_separation_days, show_years=False)

    # Spread events need both legs on the same grid.
    pairs = [
        ("cross_market_divergence", "USGG10YR", "GDBR10"),
        ("peripheral_stress", "GBTPGR10", "GDBR10"),
    ]
    for label, left, right in pairs:
        left_col, right_col = resolve(index, left), resolve(index, right)
        if left_col is None or right_col is None:
            print("%-28s SKIPPED (missing leg)" % label)
            continue
        spread = (frame[left_col] - frame[right_col]).dropna()
        for k in (2.0, 2.5):
            report_episodes("%s k=%.1f" % (label, k),
                            z_move(spread, 5, window).abs() > k,
                            min_separation_days, show_years=False)

    # Composites exercise the conditional_and primitive.
    nasdaq, ten_year, dollar = resolve(index, "NDX"), resolve(index, "USGG10YR"), resolve(index, "DXY")
    if all(c is not None for c in (nasdaq, ten_year, dollar)):
        joint = frame[[nasdaq, ten_year, dollar]].dropna()
        equity_down = z_move(joint[nasdaq], 5, window)
        yields_down = z_move(joint[ten_year], 5, window)
        dollar_up = joint[dollar].diff(5) > 0
        for k in (1.5, 2.0):
            # Strict: all three conditions, both z-tests at k.
            report_episodes("flight_to_quality STRICT k=%.1f" % k,
                            (equity_down < -k) & (yields_down < -k) & dollar_up,
                            min_separation_days, show_years=False)
            # Two-of-three: a triple AND of k-sigma tests is a very small
            # target, and on the dry run it produced a handful of episodes.
            # Reporting both tells us whether the strict form is viable on
            # real (correlated) data or whether the event needs the looser rule.
            two_of_three = (((equity_down < -k).astype(int)
                             + (yields_down < -k).astype(int)
                             + dollar_up.astype(int)) >= 2)
            report_episodes("flight_to_quality 2OF3  k=%.1f" % k, two_of_three,
                            min_separation_days, show_years=False)
    else:
        print("%-28s SKIPPED (missing leg)" % "flight_to_quality")

    move, vix = resolve(index, "MOVE"), resolve(index, "VIX")
    if move is not None and vix is not None:
        joint = frame[[move, vix]].dropna()
        # The difference of two z-scores is NOT itself a z-score: if the two
        # are roughly independent its variance is ~2, so thresholding it at k
        # is really a test at k/sqrt(2). Re-standardise the difference against
        # its own trailing distribution before applying k, or the event fires
        # far too often (the dry run gave eff_n in the hundreds).
        divergence = z_move(joint[move], 5, window) - z_move(joint[vix], 5, window)
        divergence_z = z_level(divergence.dropna(), window)
        for k in (2.0, 2.5):
            report_episodes("rates_equity_vol_divergence k=%.1f" % k,
                            divergence_z.abs() > k, min_separation_days, show_years=False)


# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="path to the wide Bloomberg panel CSV")
    parser.add_argument("--date-column", default=None,
                        help="name of the date column (auto-detected if omitted)")
    parser.add_argument("--window", type=int, default=252,
                        help="trailing window for rolling mean/sigma (default 252)")
    parser.add_argument("--min-separation", type=int, default=5,
                        help="episodes closer than this many days collapse (default 5)")
    parser.add_argument("--dayfirst", action="store_true",
                        help="parse dates as DD/MM/YYYY rather than MM/DD/YYYY")
    parser.add_argument("--skip-cohort2", action="store_true")
    args = parser.parse_args(argv)

    frame = load_panel(args.csv, args.date_column, args.dayfirst)
    index = build_column_index(frame.columns)

    print("PANEL VALIDATION SYNTHESIS")
    print("csv=%s  rows=%d  cols=%d  window=%d  min_separation=%d"
          % (args.csv, frame.shape[0], frame.shape[1], args.window, args.min_separation))
    print("pandas=%s  numpy=%s" % (pd.__version__, np.__version__))

    section_inventory(frame, index)
    section_asw_oracle(frame, index)
    section_fat_tails(frame, index, args.window, horizons=(1, 5))
    section_cohort(frame, index, args.window, args.min_separation)
    if not args.skip_cohort2:
        section_cohort2(frame, index, args.window, args.min_separation)

    print("\n" + "=" * 78)
    print("END OF SYNTHESIS -- this output contains no raw levels and is safe to share.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
