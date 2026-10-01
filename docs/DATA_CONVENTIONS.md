# Data conventions

- Account CSV: `external_id,timestamp,type,symbol,quantity,price,amount,fee,currency,fx_cad,account,account_type,affiliated`. Blank optional fields use defaults. IDs deduplicate identical records/reject conflicts. Ledger cash and holdings reconcile per account. Affiliated records inform tax without adding owned assets.
- Market CSV: `timestamp,symbol,open,high,low,close,volume`. Daily timestamps use UTC session-open; intraday timestamps use bar-open. Missing/unaligned bars and invalid NYSE sessions are rejected. Add explicit actions through `/api/v1/market/import`'s dataset JSON. Yahoo history is reconstructed into historical units; Alpaca uses IEX raw bars/actions. Coverage/publication limits apply.
- Cash CSV: `external_id,timestamp,amount,category,currency`. Income positive, spending negative, one currency.
- News CSV: `external_id,timestamp,available_at,symbols,text,source,url,fixture`. Quote comma-separated symbols. Availability timestamps prevent future posts entering simulation.

Defaults: long-only, fractional shares, USD 100,000, zero commission, 5 bps slippage, 1% participation.
Next-bar execution, conservative limits and cash/volume checks restrict partial fills. Splits adjust holdings/orders; dividends credit at the supplied event timestamp.
Use payment-date events when dividend payment timing matters. Unsupported corporate actions require normalized imports.

Chronological 60/20/20 session partitions; walk-forward uses fixed rules and rolling past-only history, without test-set tuning.
Benchmark total-return curves do not include simulated benchmark costs.
Annualized return is arithmetic; VaR/ES use daily returns; drawdown uses the full curve.
Factor exposure is explicitly one benchmark factor. Optimizer failures/caps are validated; CVaR uses daily 95% tail loss and Black–Litterman requires annual return views.

Paper state/cursor/orders/fills commit together. Docker polls complete provider bars every minute; native mode uses Poll.
Forward warmup contains only completed bars. Staleness uses exchange sessions/trading minutes, including holidays and publication grace. Failed provider polls flag the account stale.
Backtests record execution dates and a calculated comparison with their own same-period benchmark. Holdings are share quantities; cash is USD. Compare strategies across matching evaluation dates.
Baseline sentiment is a lexicon. DistilBERT, forecasts and RL are separate from Qwen.
Impact evaluation purges overlapping training outcomes. Forecasts use chronological holdout evaluation, recurrence detection and residual-bootstrap intervals.
Plaid is Sandbox-only. PPO/DDPG save seeds/checkpoints and compare held-out performance with simple rules; no outperformance claim.

Tax pools CAD ACB across supplied own taxable accounts and needs transaction-date FX.
Registered/affiliated activity informs partial replacement checks. Missing records/FX and unfinished windows stay provisional.
Only completed, unambiguous own-taxable replacements receive automatic ACB additions.
Overlapping/affiliated ACB allocation is flagged for review. Harvester marks are CAD; this does not produce a tax filing.

Public Bluesky cashtags: `python scripts/social_stream.py --owner demo --symbols AAPL MSFT SPY --max-posts 100`.
The collector saves its cursor, reconnects with backoff and handles deletion events. Posts remain labelled social/unverified.
