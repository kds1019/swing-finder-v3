"""
Central configuration for the SwingFinder agent pipeline.

Credentials are loaded from environment / .env but NOT validated here — each
agent checks only the keys it actually needs in its own __init__, so e.g.
running the Market Data Agent alone doesn't require FMP_API_KEY to be set.

Screening parameters below are the canonical values from the "SwingFinder
Screening Parameters (v2 - Tuned)" sheet (2026-07-06). They intentionally
supersede both swing-finder-v2's config.py defaults and its live scanner.py
hardcoded values, which disagree with each other and with this sheet.
"""

from dataclasses import dataclass, field
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # --- Credentials (may be empty string if not yet configured) ---
    alpaca_api_key: str = field(default_factory=lambda: os.environ.get("ALPACA_API_KEY", ""))
    alpaca_secret_key: str = field(default_factory=lambda: os.environ.get("ALPACA_SECRET_KEY", ""))
    fmp_api_key: str = field(default_factory=lambda: os.environ.get("FMP_API_KEY", ""))
    webull_app_key: str = field(default_factory=lambda: os.environ.get("WEBULL_APP_KEY", ""))
    webull_app_secret: str = field(default_factory=lambda: os.environ.get("WEBULL_APP_SECRET", ""))
    webull_region_id: str = field(default_factory=lambda: os.environ.get("WEBULL_REGION_ID", "us"))
    webull_environment: str = field(default_factory=lambda: os.environ.get("WEBULL_ENVIRONMENT", "prod"))
    webull_token_dir: str = field(default_factory=lambda: os.environ.get("WEBULL_TOKEN_DIR", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))

    # Tickers the user already holds long-term (not swing positions) and never wants
    # re-picked or flagged by this pipeline — excluded right after the screener, before
    # sector cap/research, so they don't consume a sector-cap slot or an FMP call.
    # HELP and CYBN are the same company under two ticker symbols (renamed) — both listed
    # since the universe CSV may carry either depending on when it was last exported.
    excluded_tickers: tuple[str, ...] = ("HELP", "CYBN")

    # --- Canonical screening parameters (from the tuned sheet) ---
    # price_min/price_max/min_volume now directly gate live universe construction
    # (core.universe.build_universe's FMP company-screener call), not just the
    # technical screener downstream — universe membership and these settings are
    # the same source of truth, they can no longer silently drift apart.
    vix_gate_ceiling: float = 20.0
    # Raised from the sheet's 2.0 to 3.0 on 2026-07-07: professional swing-trading
    # convention favors >=3:1 given swing trades' longer holds and lower win rates
    # (2:1 only needs a 40% win rate to break even; 3:1 gives more cushion against
    # losing streaks). See core/trade_plan.py for where this is actually enforced.
    min_risk_reward: float = 3.0
    price_min: float = 10.0
    price_max: float = 150.0
    min_volume: int = 500_000
    # Dollar-volume floor (Price * Volume), applied client-side in core.universe alongside
    # the share-count floor above. Share count alone is the wrong unit for liquidity — a
    # $12 stock at 500k shares turns over ~$6M/day, a $130 stock at 500k turns over ~$65M.
    # $10M is the low end of the range the practitioner write-ups converge on for a small
    # account; raise toward $20-30M if fill slippage becomes the constraint. See
    # docs/strategy.md.
    min_dollar_volume: float = 10_000_000.0
    # Market-cap floor in $millions, applied client-side in core.universe. Cuts micro/small
    # caps whose "pullback" is disproportionately the start of a dilution spiral or a
    # news-driven collapse that gaps straight through a stop — the deep-pullback entry is
    # already buying weakness, so the company needs enough size that the weakness reads as a
    # correction, not an existential problem. $1.5B keeps most of the small/mid universe
    # (where the deep-pullback-with-upside trade lives) without drifting into the
    # low-volatility mega-cap zone. Tickers with a missing/zero market cap fail this and are
    # dropped. Candidate for data-driven refinement — see docs/strategy.md.
    market_cap_min_musd: float = 1_500.0
    sector_cap: int = 3
    # earnings_buffer_exclude_days: absolute floor — earnings the same day or next day is
    # always hard-excluded (see pipeline.py::apply_earnings_buffer), no override possible,
    # since a stop can't protect against an overnight gap through a print that close.
    # earnings_buffer_soft_days/hard_days: no longer exclusion cutoffs on their own past the
    # floor above — tickers reporting within earnings_buffer_hard_days (but beyond the floor)
    # are tagged "earnings_imminent" (<= soft_days) or "earnings_upcoming" and passed to the
    # Decision Agent, which decides per-ticker whether the fundamentals genuinely support an
    # earnings-catalyst pick rather than mechanically dropping every one of them.
    earnings_buffer_exclude_days: int = 1
    earnings_buffer_soft_days: int = 7
    earnings_buffer_hard_days: int = 14
    smartscore_baseline: int = 50
    # Max % of account net liquidation value to risk on a single trade (position size =
    # this dollar amount / abs(entry - stop)). Used by DecisionAgent to size each ranked pick.
    # Raised from 1.0 to 4.0 on 2026-07-13 per explicit user instruction (wanted 3-5%,
    # picked the midpoint) after seeing the 1%-based sizing round several picks down to
    # 2-4 shares on a small account.
    risk_per_trade_pct: float = 4.0

    # --- Data pull parameters ---
    # Bumped from 60 to 300 (2026-07-13): core.pullback_reversal's EMA200 uptrend check
    # looks EMA200_TREND_LOOKBACK_DAYS=126 bars back, and needs real EMA200 warmup on top
    # of that — 60 days was the old core.smartscore's own minimum (`len(df) < 60`), left
    # over from before the screener replacement, and silently caused every ticker in the
    # first two live runs of the new pipeline to fail with reason="insufficient_data",
    # not because the pattern itself is rare. 300 matches the WARMUP_BARS convention
    # research/walk_forward_backtest.py already uses for the same underlying reason.
    bars_lookback_days: int = 300
    # Scaled down proportionally from 85 (calibrated for 60-day lookback, per
    # agents/market_data_agent.py's module docstring: "~85-87 symbols x 60-day lookback
    # per call; Alpaca's 1MB response cap is the real constraint") — 300/60 = 5x more data
    # per symbol now, so batch_size must shrink roughly 5x (85/5=17) to avoid exceeding
    # that same response cap. Not verified against Alpaca's actual cap at this new size;
    # revisit if fetch_universe_bars starts erroring or silently truncating batches.
    alpaca_batch_size: int = 17


def load_settings() -> Settings:
    return Settings()
