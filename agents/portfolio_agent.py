"""
Portfolio Agent — Webull OpenAPI SDK (direct, not via MCP).

Uses the `webull` package (PyPI `webull-openapi-python-sdk`) directly, the same
one the already-configured `webull-openapi-mcp` CLI wraps — reusing the exact
same env vars and, critically, the same already-authenticated token directory
(`WEBULL_TOKEN_DIR`), so this project never re-triggers the 2FA flow that was
just fixed. The bootstrap sequence below (ApiClient -> set_token_dir ->
TradeClient) mirrors webull_openapi_mcp/sdk_client.py's WebullSDKClient exactly.

place_order() defaults to dry_run=True. Flipping to live execution is an
explicit, separate decision — never a config default (see handoff doc's open
questions).
"""

from __future__ import annotations

import uuid
from typing import Optional

import pandas as pd

from webull.core.client import ApiClient
from webull.core.exception.exceptions import ClientException, ServerException
from webull.trade.trade_client import TradeClient


class WebullAuthError(RuntimeError):
    """Raised when the Webull token is missing or not yet verified (2FA pending)."""


class PortfolioAgent:
    def __init__(self, settings):
        if not settings.webull_app_key or not settings.webull_app_secret:
            raise RuntimeError(
                "WEBULL_APP_KEY / WEBULL_APP_SECRET are required for PortfolioAgent. "
                "Add them to your .env (reuse the same values as webull-openapi-mcp)."
            )
        self.settings = settings

        api_client = ApiClient(
            settings.webull_app_key,
            settings.webull_app_secret,
            settings.webull_region_id.lower(),
        )
        if settings.webull_token_dir:
            api_client.set_token_dir(settings.webull_token_dir)

        try:
            self.trade = TradeClient(api_client)
        except ServerException as e:
            if e.error_code == "NO_AVAILABLE_DEVICE":
                raise WebullAuthError(
                    "No device registered for 2FA. Log into the Webull mobile app with "
                    "this account, then run: webull-openapi-mcp auth"
                ) from e
            raise
        except ClientException as e:
            if "ERROR_INIT_TOKEN" in str(e) or "ERROR_CHECK_TOKEN" in str(e):
                raise WebullAuthError(
                    "Webull token missing or not yet verified (2FA pending). Run: "
                    "webull-openapi-mcp auth — then approve the push in the Webull app."
                ) from e
            raise

        self._account_id: Optional[str] = None

    @staticmethod
    def _extract(response):
        """The SDK returns raw requests.Response objects from most calls (not a
        typed wrapper with a `.data` attribute) — parse the JSON body directly."""
        if response is None:
            return None
        if hasattr(response, "json") and callable(response.json):
            try:
                return response.json()
            except Exception:
                return getattr(response, "content", response)
        return response

    def _default_account_id(self) -> str:
        if self._account_id is None:
            accounts = self.get_account_list()
            if not accounts:
                raise RuntimeError("No Webull accounts returned for this API key.")
            self._account_id = str(accounts[0]["account_id"])
        return self._account_id

    def get_account_list(self) -> list[dict]:
        response = self.trade.account_v2.get_account_list()
        data = self._extract(response)
        if isinstance(data, dict):
            return data.get("data") or data.get("accounts") or []
        return data or []

    def get_account_balance(self, account_id: Optional[str] = None) -> dict:
        account_id = account_id or self._default_account_id()
        response = self.trade.account_v2.get_account_balance(account_id)
        data = self._extract(response)
        return data.get("data", data) if isinstance(data, dict) else data

    def get_positions(self, account_id: Optional[str] = None) -> pd.DataFrame:
        account_id = account_id or self._default_account_id()
        response = self.trade.account_v2.get_account_position(account_id)
        data = self._extract(response)
        if isinstance(data, dict):
            data = data.get("data", data)
        positions = data if isinstance(data, list) else (data.get("positions", []) if isinstance(data, dict) else [])
        return pd.DataFrame(positions)

    def get_open_orders(self, account_id: Optional[str] = None) -> pd.DataFrame:
        account_id = account_id or self._default_account_id()
        response = self.trade.order_v3.get_order_open(account_id=account_id)
        data = self._extract(response)
        if isinstance(data, dict):
            data = data.get("data", data)
        orders = data if isinstance(data, list) else (data.get("orders", []) if isinstance(data, dict) else [])
        return pd.DataFrame(orders)

    def check_sector_exposure(self, positions_df: pd.DataFrame, sector_lookup: dict[str, str]) -> dict[str, int]:
        """Current sector counts among held positions — feeds sector-cap-aware
        decisions that account for existing holdings, not just the day's shortlist."""
        if positions_df.empty or "symbol" not in positions_df.columns:
            return {}
        counts: dict[str, int] = {}
        for symbol in positions_df["symbol"]:
            sector = sector_lookup.get(symbol, "Unknown")
            counts[sector] = counts.get(sector, 0) + 1
        return counts

    def place_order(
        self,
        ticker: str,
        qty: float,
        side: str,
        order_type: str = "MARKET",
        time_in_force: str = "DAY",
        limit_price: Optional[float] = None,
        account_id: Optional[str] = None,
        dry_run: bool = True,
    ) -> dict:
        """
        side: "BUY" or "SELL". order_type: "MARKET" or "LIMIT" (limit_price required).
        dry_run=True (default) returns the order payload that WOULD be sent,
        without calling the SDK. Live execution requires an explicit dry_run=False.
        """
        account_id = account_id or self._default_account_id()
        coid = str(uuid.uuid4())

        order: dict = {
            "combo_type": "NORMAL",
            "client_order_id": coid,
            "instrument_type": "EQUITY",
            "market": "US",
            "symbol": ticker,
            "order_type": order_type,
            "entrust_type": "QTY",
            "support_trading_session": "CORE",
            "time_in_force": time_in_force,
            "side": side,
            "quantity": str(qty),
        }
        if order_type == "LIMIT":
            if limit_price is None:
                raise ValueError("limit_price is required for LIMIT orders")
            order["limit_price"] = str(limit_price)

        if dry_run:
            return {"dry_run": True, "account_id": account_id, "order": order}

        response = self.trade.order_v3.place_order(account_id=account_id, new_orders=[order])
        data = self._extract(response)
        return {"dry_run": False, "account_id": account_id, "order": order, "response": data}
