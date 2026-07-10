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

    # --- Universe ---
    universe_csv_path: str = field(default_factory=lambda: os.environ.get("UNIVERSE_CSV_PATH", "data/universe.csv"))

    # --- Canonical screening parameters (from the tuned sheet) ---
    vix_gate_ceiling: float = 20.0
    # Raised from the sheet's 2.0 to 3.0 on 2026-07-07: professional swing-trading
    # convention favors >=3:1 given swing trades' longer holds and lower win rates
    # (2:1 only needs a 40% win rate to break even; 3:1 gives more cushion against
    # losing streaks). See core/trade_plan.py for where this is actually enforced.
    min_risk_reward: float = 3.0
    price_min: float = 10.0
    price_max: float = 150.0
    min_volume: int = 500_000
    sector_cap: int = 3
    earnings_buffer_soft_days: int = 7   # scanner: soft exclude/flag
    earnings_buffer_hard_days: int = 14  # base scanner: hard exclude
    smartscore_baseline: int = 50
    # Max % of account net liquidation value to risk on a single trade (position size =
    # this dollar amount / abs(entry - stop)) — standard conservative retail swing-trading
    # convention. Used by DecisionAgent to size each ranked pick.
    risk_per_trade_pct: float = 1.0

    # --- Data pull parameters ---
    bars_lookback_days: int = 60
    alpaca_batch_size: int = 85


def load_settings() -> Settings:
    return Settings()
