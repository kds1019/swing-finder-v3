# Candidate-pool funnel diagnostic — SetupType-blind ordering + pre-research sector cap (RESEARCH)

Pure price-data diagnostic (zero FMP/Anthropic calls) replaying the funnel that runs BEFORE the Decision Agent ever sees a candidate: KnifeRiskTier-then-depth ordering, an 8/sector pre-research cap, then a 30-candidate pool size cut. Uses today's live-gate signal set as the historical stand-in for each day's raw screener matches, tagged retroactively with TrendState/SetupType. 'final list' below uses the SAME funnel ordering as a structural proxy for the Decision Agent's ranking (no real LLM call, no point-in-time research data) — it measures the MECHANICAL effect of this design, not real final-output quality.
- 1231 distinct screener-match days; 376 of them (31%) had at least one raw trend_continuation match
- setup_type counts across all signals: {'reversion_bounce': 16254, nan: 13210, 'trend_continuation': 481}

## Question 1 — does SetupType-blind ordering actually lose trend_continuation candidates before research?

On days with >=1 raw trend_continuation match, % of those days it survived to reach the research pool / the (proxy) final list, by ordering (holding the pre-research cap ON, today's actual setting):

| ordering | trades reaches research | reaches final list |
|---|---|---|
| current (tier, depth) | 86% | 64% |
| setup-priority (setup, tier, depth) | 100% | 100% |

## Question 2 — does the pre-research sector cap matter, and does removing it cost anything?

### Raw sector concentration (before any cap — how skewed are the day's matches?)

| stat | value |
|---|---|
| mean max-sector-share | 30% |
| median max-sector-share | 28% |
| p90 max-sector-share | 45% |
| days with max-sector-share > 50% | 5% |

### Effect of removing the pre-research cap (current ordering held fixed)

| | research pool size | research pool # sectors | final list size |
|---|---|---|---|
| with pre-research cap | 20.4 avg | 7.4 avg | 15.2 avg |
| no pre-research cap | 20.6 avg | 7.3 avg | 15.1 avg |

### Same comparison, restricted to the most sector-concentrated quartile of days (306 days, max-sector-share > 35%)

| | research pool size | research pool # sectors | final list size |
|---|---|---|---|
| with pre-research cap | 13.7 avg | 5.5 avg | 10.3 avg |
| no pre-research cap | 14.1 avg | 5.5 avg | 10.1 avg |

## All four combinations — trend_continuation survival to final list

| ordering | pre-research cap | reaches research | reaches final list |
|---|---|---|---|
| current (tier, depth) | with pre-research cap | 86% | 64% |
| current (tier, depth) | no pre-research cap | 87% | 63% |
| setup-priority (setup, tier, depth) | with pre-research cap | 100% | 100% |
| setup-priority (setup, tier, depth) | no pre-research cap | 100% | 100% |

_Signal set is today's live gate only (G1-G4 + current-trend slope gate + weak-RR dropped) — matches what agents/market_data_agent.py would find, not the full raw universe before any live gate. Sector labels are today's live universe applied retroactively (same limitation as every other backtest in this repo)._