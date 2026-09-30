# Decision Agent prompt A/B (RESEARCH)

- payload: 30 candidates, as of 2026-09-30; runs per prompt: 1
- v1 = research/prompts/decision_prompt_v1.txt (4475 words); v2 = live SYSTEM_PROMPT (1790 words)
- one sample per run: claude-sonnet-5 sampling can't be pinned

| metric | v1 (old) run1 | v2 (new) run1 |
|---|---|---|
| error |  |  |
| ranked | 27 | 28 |
| excluded | 0 | 2 ['BEAM', 'WERN'] |
| missing | 3 ['ALGT', 'SYRE', 'WERN'] | 0 |
| invented | 0 | 0 |
| order inversions (support→setup) | 67 | 0 |
| still_falling above others | 0 | 0 |
| sizing talk (picks) | 3 ['FLYW', 'PECO', 'PTCT'] | 1 ['FLEX'] |
| sizing talk (summary) | False | False |
| tickers named in summary | 7 ['ALGT', 'ASTH', 'CSCO', 'FLEX', 'KMT', 'SYRE'] | 0 |
| 'recent' older than 7d | 0 | 0 |
| flags outside judgment set | 6 ['AboveVolumePOC', 'HeavilyShorted', 'InsiderBuying', 'InsiderSelling', 'SectorOverlap', 'ShortsAdding'] | 0 |
| catalyst_status counts | {'none': 24, 'recent': 3} | {'none': 27, 'recent': 1} |
| support_status counts | {'confirmed': 20, 'forming': 7} | {'confirmed': 28} |
| seconds | 542 | 438 |

## Top 20 side by side (first run)

| # | v1 (old) | v2 (new) |
|---|---|---|
| 1 | FLEX (trend_continuation) | FLEX (trend_continuation) |
| 2 | KMT (reversion_bounce) | CSCO (trend_continuation) |
| 3 | CSCO (trend_continuation) | EBAY (trend_continuation) |
| 4 | ASTH (trend_continuation) | WTTR (trend_continuation) |
| 5 | EBAY (trend_continuation) | TKR (trend_continuation) |
| 6 | PVH (reversion_bounce) | CGNX (trend_continuation) |
| 7 | CGNX (trend_continuation) | ASTH (trend_continuation) |
| 8 | KIM (reversion_bounce) | RCUS (trend_continuation) |
| 9 | SBUX (reversion_bounce) | WLY (trend_continuation) |
| 10 | TKR (trend_continuation) | RLJ (trend_continuation) |
| 11 | WLY (trend_continuation) | FLYW (trend_continuation) |
| 12 | WTTR (trend_continuation) | SRRK (trend_continuation) |
| 13 | SNDX (reversion_bounce) | SYRE (trend_continuation) |
| 14 | IDA (reversion_bounce) | KMT (reversion_bounce) |
| 15 | ON (reversion_bounce) | PVH (reversion_bounce) |
| 16 | RCUS (trend_continuation) | KIM (reversion_bounce) |
| 17 | SRRK (trend_continuation) | ON (reversion_bounce) |
| 18 | FLYW (trend_continuation) | SBUX (reversion_bounce) |
| 19 | PECO (reversion_bounce) | IDA (reversion_bounce) |
| 20 | COGT (reversion_bounce) | PTCT (reversion_bounce) |

Inversion examples (first run): v1 (old) run1: ['KMT>CSCO', 'KMT>ASTH', 'KMT>EBAY', 'KMT>CGNX', 'KMT>TKR', 'KMT>WLY', 'KMT>SRRK', 'KMT>FLYW']; v2 (new) run1: []
