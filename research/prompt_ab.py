"""
RESEARCH ONLY — compare two Decision Agent system prompts on an IDENTICAL saved payload.

Input: a payload written by `python pipeline.py --save-decision-input PATH` (the exact user
prompt the Decision Agent would receive). Each prompt ranks that same payload; the outputs
are audited against rules checkable from the payload itself:

  coverage        every candidate is ranked or explicitly excluded; none invented
  order           ranking inversions vs the intended priority order
                  (support_status, then SetupType: trend_continuation > ema_band_pullback >
                  reversion_bounce > null) — v1 never had this rule; measured for contrast
  still_falling   any still_falling ranked above a confirmed/forming name
  sizing          any sizing language (share counts, "size", "sizing", "position size")
  summary_names   tickers named in overall_recommendation (the sector cap runs after it)
  recent_age      "recent" catalysts whose catalyst_date is > 7 days old (pipeline would
                  downgrade these to "none" — counts how often the model breaks the rule)
  flags           v2: flags outside the allowed judgment set; v1: nothing (it set all flags)

The model's sampling can't be fixed (claude-sonnet-5 rejects temperature), so a single run
per prompt is one sample — use --runs N to repeat.

Usage:
  python -m research.prompt_ab --input decision_input.json [--runs 1]
    old prompt: research/prompts/decision_prompt_v1.txt ; new prompt: the live SYSTEM_PROMPT
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

from agents.decision_agent import SYSTEM_PROMPT, DecisionAgent
from config.settings import load_settings

HERE = Path(__file__).resolve().parent
V1_PATH = HERE / "prompts" / "decision_prompt_v1.txt"
OUT = HERE / "prompt_ab.md"
RAW_OUT = HERE / "prompt_ab_raw.json"

SUPPORT_ORDER = {"confirmed": 0, "forming": 1, "still_falling": 2}
SETUP_ORDER = {"trend_continuation": 0, "ema_band_pullback": 1, "reversion_bounce": 2, None: 3}
SIZING_RE = re.compile(r"\b(siz(e|ed|ing)|shares?\b.*\$|position value|risk_amount|position_shares)",
                       re.IGNORECASE)
ALLOWED_V2_FLAGS = {"CatalystFaded", "EarningsCatalyst"}


def audit(out: dict, payload: dict, as_of: pd.Timestamp, label: str) -> dict:
    if "error" in out:
        return {"label": label, "error": out.get("error"), "truncated": out.get("truncated")}
    shortlist = {r["Ticker"]: r for r in payload["shortlist"]}
    picks = sorted(out.get("ranked_picks") or [], key=lambda p: p.get("rank", 1e9))
    excluded = {e.get("ticker") for e in (out.get("excluded") or []) if isinstance(e, dict)}
    ranked = [p.get("ticker") for p in picks]

    missing = sorted(set(shortlist) - set(ranked) - excluded)
    invented = sorted(set(ranked) - set(shortlist))

    def key(p):
        setup = shortlist.get(p.get("ticker"), {}).get("SetupType")
        return (SUPPORT_ORDER.get(p.get("support_status"), 3), SETUP_ORDER.get(setup, 3))

    inversions, sf_violations = [], []
    for i, a in enumerate(picks):
        for b in picks[i + 1:]:
            if key(a) > key(b):
                inversions.append(f"{a['ticker']}>{b['ticker']}")
            if a.get("support_status") == "still_falling" and b.get("support_status") != "still_falling":
                sf_violations.append(f"{a['ticker']}>{b['ticker']}")

    sizing = [p["ticker"] for p in picks
              if SIZING_RE.search(" ".join(str(p.get(k, "")) for k in ("rationale", "bear_case", "research_highlight")))]
    summary = out.get("overall_recommendation", "") or ""
    summary_sizing = bool(SIZING_RE.search(summary))
    summary_names = sorted(t for t in shortlist if re.search(rf"\b{re.escape(t)}\b", summary))

    stale_recent = []
    for p in picks:
        if p.get("catalyst_status") == "recent":
            d = pd.to_datetime(p.get("catalyst_date"), errors="coerce")
            if pd.isna(d) or (as_of - d.normalize()).days > 7:
                stale_recent.append(f"{p['ticker']}({p.get('catalyst_date')})")

    bad_flags = sorted({f for p in picks for f in (p.get("flags") or [])} - ALLOWED_V2_FLAGS)
    return {
        "label": label, "ranked": len(picks), "excluded": sorted(excluded),
        "missing": missing, "invented": invented,
        "order_inversions": len(inversions), "order_inversion_examples": inversions[:8],
        "still_falling_violations": len(sf_violations),
        "sizing_picks": sizing, "summary_sizing": summary_sizing, "summary_names": summary_names,
        "stale_recent": stale_recent, "flags_outside_judgment_set": bad_flags,
        "catalyst_counts": pd.Series([p.get("catalyst_status") for p in picks]).value_counts().to_dict(),
        "support_counts": pd.Series([p.get("support_status") for p in picks]).value_counts().to_dict(),
        "order": ranked,
    }


def run_prompt(settings, prompt: str, user_prompt: str, n: int) -> tuple[dict, float]:
    t0 = time.time()
    out = DecisionAgent(settings, system_prompt=prompt).synthesize_from_prompt(user_prompt, n)
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--runs", type=int, default=1)
    a = ap.parse_args()

    settings = load_settings()
    user_prompt = Path(a.input).read_text(encoding="utf-8")
    payload = json.loads(user_prompt)
    n = len(payload["shortlist"])
    as_of = pd.Timestamp.now().normalize()
    prompts = {"v1 (old)": V1_PATH.read_text(encoding="utf-8"), "v2 (new)": SYSTEM_PROMPT}
    setup_of = {r["Ticker"]: r.get("SetupType") for r in payload["shortlist"]}

    results, raw = [], {}
    for run in range(1, a.runs + 1):
        for label, prompt in prompts.items():
            out, secs = run_prompt(settings, prompt, user_prompt, n)
            tag = f"{label} run{run}"
            raw[tag] = out
            r = audit(out, payload, as_of, tag)
            r["seconds"] = round(secs)
            results.append(r)
            print(f"[ab] {tag}: {json.dumps({k: v for k, v in r.items() if k != 'order'})}", file=sys.stderr)

    L = ["# Decision Agent prompt A/B (RESEARCH)", "",
         f"- payload: {n} candidates, as of {as_of.date()}; runs per prompt: {a.runs}",
         f"- v1 = research/prompts/decision_prompt_v1.txt "
         f"({len(prompts['v1 (old)'].split())} words); v2 = live SYSTEM_PROMPT "
         f"({len(prompts['v2 (new)'].split())} words)",
         "- one sample per run: claude-sonnet-5 sampling can't be pinned", "",
         "| metric | " + " | ".join(r["label"] for r in results) + " |",
         "|---|" + "---|" * len(results)]
    rows = [("error", "error"), ("ranked", "ranked"), ("excluded", "excluded"), ("missing", "missing"),
            ("invented", "invented"), ("order inversions (support→setup)", "order_inversions"),
            ("still_falling above others", "still_falling_violations"),
            ("sizing talk (picks)", "sizing_picks"), ("sizing talk (summary)", "summary_sizing"),
            ("tickers named in summary", "summary_names"), ("'recent' older than 7d", "stale_recent"),
            ("flags outside judgment set", "flags_outside_judgment_set"),
            ("catalyst_status counts", "catalyst_counts"), ("support_status counts", "support_counts"),
            ("seconds", "seconds")]
    for name, k in rows:
        vals = []
        for r in results:
            v = r.get(k, "")
            vals.append(str(len(v)) + (f" {v[:6]}" if v else "") if isinstance(v, list) else str(v))
        L.append(f"| {name} | " + " | ".join(vals) + " |")
    L += ["", "## Top 20 side by side (first run)", "",
          "| # | v1 (old) | v2 (new) |", "|---|---|---|"]
    o1 = results[0].get("order", []) if len(results) > 0 else []
    o2 = results[1].get("order", []) if len(results) > 1 else []
    for i in range(20):
        c1 = f"{o1[i]} ({setup_of.get(o1[i])})" if i < len(o1) else ""
        c2 = f"{o2[i]} ({setup_of.get(o2[i])})" if i < len(o2) else ""
        L.append(f"| {i + 1} | {c1} | {c2} |")
    L += ["", "Inversion examples (first run): "
          + "; ".join(f"{r['label']}: {r.get('order_inversion_examples')}" for r in results[:2])]

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    RAW_OUT.write_text(json.dumps(raw, indent=1, default=str), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
