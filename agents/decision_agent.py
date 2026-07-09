"""
Decision Agent — Anthropic API.

Pure synthesis/judgment layer on top of deterministic numbers: takes the
SmartScore shortlist + FMP research + Webull position context and produces
a final ranked, explained shortlist with flags. Never recomputes SmartScore,
sector cap, or any other numeric filter — those are already-decided facts by
the time they reach this agent.
"""

from __future__ import annotations

import json
import re
from typing import Optional

import pandas as pd
from anthropic import Anthropic

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_code_fence(text: str) -> str:
    """Claude sometimes wraps JSON responses in a ```json ... ``` fence despite
    being asked for raw JSON — strip it before parsing rather than fighting the
    model with ever-more-emphatic prompt wording."""
    return _CODE_FENCE_RE.sub("", text).strip()

MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """You are the final synthesis step of a swing-trading screening pipeline.
You receive tickers that have ALREADY passed deterministic scoring (SmartScore), sector-cap
filtering, and an earnings-buffer filter, each with a pre-computed trade plan (Entry/Stop/
Target/RRRatio from core/trade_plan.py — swing-low/EMA-anchored stop, Fibonacci-extension
target refined against real support/resistance). SmartScore has ALREADY been adjusted (bonus/
penalty, do not reapply) for two things: where price sits relative to its own volume profile
(PriceVsPOCPct/VolumeProfileFlag — rewarded if at/below the point of control, penalized if
stretched above it, especially "extended_above_value_area") and the ML ensemble's 5-day
directional call (MLEdgePct/MLEdgeFlag — rewarded if positive, penalized if negative or
"ml_edge_unavailable"). Your job is ONLY to:

1. Rank the provided tickers by overall attractiveness, using the given SmartScore, trade
   plan, and research context as inputs to your judgment.
2. Explain each ranking in 1-2 sentences referencing concrete factors already provided
   (do not invent facts not present in the input, and never recompute Stop/Target/RRRatio,
   PriceVsPOCPct, or MLEdgePct yourself — pass them through as given).
3. Flag risks: sector concentration relative to EXISTING Webull positions (not just the
   day's shortlist), earnings-date conflicts, whether the VIX gate is open or closed, and
   ALWAYS flag if WeakRR is true (R:R fell short of the minimum after support/resistance
   refinement), StopSanityFlag is true (R:R >= 15:1 more often means an unusually tight
   stop than an unusually good target — say so explicitly, don't just repeat the number),
   VolumeProfileFlag is "extended_above_value_area" (price has run past where 70% of recent
   volume actually traded — thin support underneath), or MLEdgeFlag is "negative_ml_edge"
   (a clean technical setup the model itself doesn't confirm — say so explicitly, this is
   exactly the kind of case that looks good on SmartScore alone but may not be worth trading).
4. If the VIX gate is closed (market_gate_open=false), your top-level recommendation must
   bias toward "monitor only, no new entries" regardless of individual SmartScores.

Do NOT recompute or second-guess the SmartScore, sector cap, earnings buffer, or trade
plan numbers — treat them as given. Respond with ONLY a JSON object matching this shape:
{
  "market_gate_open": bool,
  "overall_recommendation": str,
  "ranked_picks": [
    {"ticker": str, "rank": int, "smartscore": number, "entry": number, "stop": number,
     "target": number, "rr_ratio": number, "rationale": str, "flags": [str, ...]}
  ]
}"""


class DecisionAgent:
    def __init__(self, settings):
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for DecisionAgent. Add it to your .env.")
        self.settings = settings
        self.client = Anthropic(api_key=settings.anthropic_api_key)

    def _build_user_prompt(
        self,
        smartscore_shortlist: pd.DataFrame,
        research_data: pd.DataFrame,
        portfolio_context: dict,
        market_gate_open: bool,
    ) -> str:
        shortlist_records = json.loads(research_data.to_json(orient="records")) if not research_data.empty else []
        payload = {
            "market_gate_open": market_gate_open,
            "shortlist": shortlist_records,
            "existing_positions": portfolio_context.get("positions", []),
            "account_balance": portfolio_context.get("balance", {}),
            "existing_sector_exposure": portfolio_context.get("sector_exposure", {}),
        }
        return json.dumps(payload, default=str, indent=2)

    def synthesize(
        self,
        smartscore_shortlist: pd.DataFrame,
        research_data: pd.DataFrame,
        portfolio_context: dict,
        market_gate_open: bool,
    ) -> dict:
        user_prompt = self._build_user_prompt(smartscore_shortlist, research_data, portfolio_context, market_gate_open)

        # Scaled to shortlist size — a fixed 4096 truncated mid-response on an 11-ticker
        # shortlist in testing (~630 tokens/ticker of rationale+flags). 800/ticker + 1500
        # overhead leaves real headroom; capped at 16000 as a sanity ceiling.
        num_tickers = len(smartscore_shortlist)
        max_tokens = min(16000, max(4096, 800 * num_tickers + 1500))

        response = self.client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        text = "".join(block.text for block in response.content if block.type == "text")
        try:
            return json.loads(_strip_code_fence(text))
        except json.JSONDecodeError:
            return {
                "error": "Failed to parse Claude's response as JSON",
                "truncated": response.stop_reason == "max_tokens",
                "max_tokens_used": max_tokens,
                "raw_response": text,
            }
