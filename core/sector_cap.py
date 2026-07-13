"""
Sector cap — FRESH DESIGN, not a port.

swing-finder-v2 has no "max N per sector" logic anywhere (only an opt-in sector-
momentum inclusion filter, unrelated). This implements a straightforward post-ranking
cap: walk the ranked list (whatever order the caller sorted it in) and keep at most
`cap` tickers per sector. Fully generic — doesn't hardcode a ranking column, just
assumes the caller already sorted ranked_df by their priority order.

Validate this against the actual "SwingFinder Screening Parameters (v2 - Tuned)"
Google Sheet definition before relying on it at full scale — this is a reasonable
interpretation of "sector cap: 3", not a verified spec.
"""

from __future__ import annotations

import pandas as pd


def apply_sector_cap(ranked_df: pd.DataFrame, cap: int = 3, sector_col: str = "Sector") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    ranked_df must already be sorted by the caller's priority order (descending).

    Returns (kept_df, excluded_df). excluded_df carries an "ExclusionReason"
    column ("sector_cap") so the reason a ticker was dropped stays visible in
    the final output rather than silently disappearing.
    """
    if ranked_df.empty:
        return ranked_df, ranked_df.assign(ExclusionReason=pd.Series(dtype=str))

    sector_counts: dict[str, int] = {}
    keep_mask = []

    for sector in ranked_df[sector_col]:
        sector = sector or "Unknown"
        count = sector_counts.get(sector, 0)
        if count < cap:
            sector_counts[sector] = count + 1
            keep_mask.append(True)
        else:
            keep_mask.append(False)

    keep_mask = pd.Series(keep_mask, index=ranked_df.index)

    kept_df = ranked_df[keep_mask].copy()
    excluded_df = ranked_df[~keep_mask].copy()
    excluded_df["ExclusionReason"] = "sector_cap"

    return kept_df, excluded_df
