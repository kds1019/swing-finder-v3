# Source

Vendored from [alpacahq/alpaca-skills](https://github.com/alpacahq/alpaca-skills),
`skills/trading-api/backtest/`, commit `9044692bab2497c801f206528fbcb133c554ac89`
(2026-06-29). Licensed under Apache License 2.0 (see `LICENSE` in this directory).

Not modified from upstream. Re-sync by re-copying `SKILL.md` and `reference.md`
from the source repo when Alpaca updates the skill.

Requires the `alpaca` CLI (`go install github.com/alpacahq/cli/cmd/alpaca@latest`)
and `ALPACA_API_KEY`/`ALPACA_SECRET_KEY` — the same credentials this project's
`agents/market_data_agent.py` already uses for market data.
