# Decision Agent prompt A/B (RESEARCH)

- payload: 30 candidates, as of 2026-09-30; runs per prompt: 2
- v1 = research/prompts/decision_prompt_v1.txt (4475 words); v2 = live SYSTEM_PROMPT (1879 words)
- one sample per run: claude-sonnet-5 sampling can't be pinned

| metric | v1 (old) run1 | v2 (new) run1 | v1 (old) run2 | v2 (new) run2 |
|---|---|---|---|---|
| error |  |  |  |  |
| ranked | 29 | 29 | 25 | 29 |
| excluded | 0 | 1 ['SRRK'] | 0 | 1 ['WERN'] |
| missing | 1 ['SYRE'] | 0 | 5 ['ALGT', 'BEAM', 'GVA', 'PTCT', 'WERN'] | 0 |
| invented | 0 | 0 | 0 | 0 |
| order inversions (support→setup) | 73 | 0 | 77 | 0 |
| still_falling above others | 0 | 0 | 0 | 0 |
| support overrides vs KnifeRiskTier | 3 ['SWX:stabilising->forming', 'WERN:stabilising->forming', 'GEO:stabilising->forming'] | 11 ['CSCO:stabilising->forming', 'CGNX:stabilising->forming', 'GEO:stabilising->forming', 'WTTR:stabilising->forming', 'IDA:stabilising->forming', 'AEP:stabilising->forming'] | 2 ['GEO:stabilising->forming', 'SWX:stabilising->forming'] | 10 ['WTTR:stabilising->forming', 'GEO:stabilising->forming', 'CSCO:stabilising->forming', 'CGNX:stabilising->forming', 'SRRK:stabilising->forming', 'GVA:stabilising->forming'] |
|   ...not explained in rationale | 0 | 0 | 0 | 0 |
| sizing talk (picks) | 0 | 3 ['RHP', 'WLY', 'PTCT'] | 0 | 3 ['RHP', 'SYRE', 'CGNX'] |
| sizing talk (summary) | True | False | False | False |
| tickers named in summary | 16 ['ALGT', 'ASTH', 'CGNX', 'CSCO', 'EBAY', 'FLYW'] | 8 ['CSCO', 'EBAY', 'KIM', 'KMT', 'PVH', 'RHP'] | 13 ['ALGT', 'BEAM', 'CGNX', 'CSCO', 'EBAY', 'GEO'] | 0 |
| 'recent' older than 7d | 0 | 0 | 0 | 0 |
| flags outside judgment set | 7 ['AboveVolumePOC', 'HeavilyShorted', 'InsiderBuying', 'InsiderSelling', 'SectorOverlap', 'ShortsAdding'] | 0 | 6 ['AboveVolumePOC', 'HeavilyShorted', 'InsiderBuying', 'InsiderSelling', 'SectorOverlap', 'ShortsAdding'] | 0 |
| catalyst_status counts | {'none': 26, 'recent': 3} | {'none': 28, 'recent': 1} | {'none': 22, 'recent': 3} | {'none': 28, 'recent': 1} |
| support_status counts | {'confirmed': 26, 'forming': 3} | {'confirmed': 18, 'forming': 11} | {'confirmed': 23, 'forming': 2} | {'confirmed': 19, 'forming': 9, 'still_falling': 1} |
| seconds | 594 | 512 | 508 | 535 |

## Top 20 side by side (first run)

| # | v1 (old) | v2 (new) |
|---|---|---|
| 1 | ASTH (trend_continuation) | EBAY (trend_continuation) |
| 2 | EBAY (trend_continuation) | TKR (trend_continuation) |
| 3 | CGNX (trend_continuation) | RHP (trend_continuation) |
| 4 | TKR (trend_continuation) | ASTH (trend_continuation) |
| 5 | PVH (reversion_bounce) | WLY (trend_continuation) |
| 6 | ON (reversion_bounce) | RCUS (trend_continuation) |
| 7 | RHP (trend_continuation) | FLYW (trend_continuation) |
| 8 | RCUS (trend_continuation) | RLJ (trend_continuation) |
| 9 | KMT (reversion_bounce) | SYRE (trend_continuation) |
| 10 | IDA (reversion_bounce) | PVH (reversion_bounce) |
| 11 | CSCO (trend_continuation) | KIM (reversion_bounce) |
| 12 | WTTR (trend_continuation) | KMT (reversion_bounce) |
| 13 | KIM (reversion_bounce) | SBUX (reversion_bounce) |
| 14 | SBUX (reversion_bounce) | ON (reversion_bounce) |
| 15 | PECO (reversion_bounce) | PTCT (reversion_bounce) |
| 16 | AEP (reversion_bounce) | SNDX (reversion_bounce) |
| 17 | SNDX (reversion_bounce) | PECO (reversion_bounce) |
| 18 | PNW (reversion_bounce) | BEAM (reversion_bounce) |
| 19 | SRRK (trend_continuation) | CSCO (trend_continuation) |
| 20 | GVA (reversion_bounce) | CGNX (trend_continuation) |

Inversion examples (first run): v1 (old) run1: ['PVH>RHP', 'PVH>RCUS', 'PVH>CSCO', 'PVH>WTTR', 'PVH>SRRK', 'PVH>FLYW', 'PVH>RLJ', 'PVH>WLY']; v2 (new) run1: []
