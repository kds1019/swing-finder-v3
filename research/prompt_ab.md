# Decision Agent prompt A/B (RESEARCH)

- payload: 30 candidates, as of 2026-09-30; runs per prompt: 2
- v1 = research/prompts/decision_prompt_v1.txt (4475 words); v2 = live SYSTEM_PROMPT (1858 words)
- one sample per run: claude-sonnet-5 sampling can't be pinned

| metric | v1 (old) run1 | v2 (new) run1 | v1 (old) run2 | v2 (new) run2 |
|---|---|---|---|---|
| error |  |  |  |  |
| ranked | 28 | 29 | 30 | 29 |
| excluded | 0 | 1 ['BEAM'] | 0 | 1 ['BEAM'] |
| missing | 2 ['ALGT', 'BEAM'] | 0 | 0 | 0 |
| invented | 0 | 0 | 0 | 0 |
| order inversions (support→setup) | 68 | 0 | 62 | 0 |
| still_falling above others | 0 | 0 | 0 | 0 |
| support overrides vs KnifeRiskTier | 8 ['CVS:stabilising->forming', 'GEO:stabilising->forming', 'QSR:stabilising->forming', 'L:stabilising->forming', 'PNW:stabilising->forming', 'WERN:stabilising->forming'] | 8 ['GEO:stabilising->forming', 'CVS:stabilising->forming', 'PTCT:stabilising->forming', 'QSR:stabilising->forming', 'PNW:stabilising->forming', 'L:stabilising->forming'] | 8 ['GEO:stabilising->forming', 'CVS:stabilising->forming', 'QSR:stabilising->forming', 'PTCT:stabilising->forming', 'SWX:stabilising->forming', 'PNW:stabilising->forming'] | 8 ['GEO:stabilising->forming', 'CVS:stabilising->forming', 'QSR:stabilising->forming', 'PTCT:stabilising->forming', 'WERN:stabilising->forming', 'SWX:stabilising->forming'] |
|   ...not explained in rationale | 0 | 1 ['PTCT'] | 3 ['CVS', 'QSR', 'SWX'] | 0 |
| deviations from SupportCheck (v1 never saw it) | 0 | 0 | 0 | 0 |
| sizing talk (picks) | 0 | 0 | 0 | 0 |
| sizing talk (summary) | False | False | False | False |
| tickers named in summary | 14 ['ALGT', 'BEAM', 'CGNX', 'CSCO', 'CVS', 'GEO'] | 0 | 7 ['CSCO', 'GEO', 'KIM', 'KMT', 'PVH', 'TKR'] | 0 |
| 'recent' older than 7d | 0 | 0 | 0 | 0 |
| flags outside judgment set | 6 ['AboveVolumePOC', 'HeavilyShorted', 'InsiderBuying', 'InsiderSelling', 'SectorOverlap', 'ShortsAdding'] | 0 | 7 ['AboveVolumePOC', 'HeavilyShorted', 'InsiderBuying', 'InsiderSelling', 'SectorOverlap', 'ShortsAdding'] | 0 |
| catalyst_status counts | {'none': 24, 'recent': 4} | {'none': 28, 'recent': 1} | {'none': 26, 'recent': 4} | {'none': 28, 'recent': 1} |
| support_status counts | {'confirmed': 20, 'forming': 8} | {'confirmed': 21, 'forming': 8} | {'confirmed': 22, 'forming': 8} | {'confirmed': 21, 'forming': 8} |
| seconds | 432 | 427 | 520 | 450 |

## Top 20 side by side (first run)

| # | v1 (old) | v2 (new) |
|---|---|---|
| 1 | CSCO (trend_continuation) | CSCO (trend_continuation) |
| 2 | CGNX (trend_continuation) | WTTR (trend_continuation) |
| 3 | TKR (trend_continuation) | RCUS (trend_continuation) |
| 4 | PVH (reversion_bounce) | CGNX (trend_continuation) |
| 5 | KMT (reversion_bounce) | ASTH (trend_continuation) |
| 6 | ATRO (reversion_bounce) | TKR (trend_continuation) |
| 7 | IDA (reversion_bounce) | FLYW (trend_continuation) |
| 8 | KIM (reversion_bounce) | RLJ (trend_continuation) |
| 9 | WTTR (trend_continuation) | SRRK (trend_continuation) |
| 10 | ASTH (trend_continuation) | PVH (reversion_bounce) |
| 11 | RCUS (trend_continuation) | KMT (reversion_bounce) |
| 12 | SBUX (reversion_bounce) | ATRO (reversion_bounce) |
| 13 | PECO (reversion_bounce) | KIM (reversion_bounce) |
| 14 | CVS (reversion_bounce) | SBUX (reversion_bounce) |
| 15 | SNDX (reversion_bounce) | IDA (reversion_bounce) |
| 16 | GEO (trend_continuation) | PECO (reversion_bounce) |
| 17 | SRRK (trend_continuation) | AMKR (reversion_bounce) |
| 18 | QSR (reversion_bounce) | COGT (reversion_bounce) |
| 19 | COGT (reversion_bounce) | SNDX (reversion_bounce) |
| 20 | L (reversion_bounce) | GVA (reversion_bounce) |

Inversion examples (first run): v1 (old) run1: ['PVH>WTTR', 'PVH>ASTH', 'PVH>RCUS', 'PVH>SRRK', 'PVH>RLJ', 'PVH>FLYW', 'KMT>WTTR', 'KMT>ASTH']; v2 (new) run1: []
