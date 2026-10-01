# Implementation status

Branch: `codex/complete-quantara`. Baseline: remote `origin/main` commit `2f28d508cbdff1882f26c1ec3c00b8a72076b5a0`, including the eight commits missing locally.
Alternate frontend chat work was reviewed. Broken combined-repository gitlinks supplied no recoverable source.
Original prototypes are preserved under `legacy/` and excluded from the supported runtime.

All seven milestones now have executable implementations. The earlier 15–25% estimate was informal; feature and verification evidence is more useful than a new percentage.

| Area | Implementation |
|---|---|
| Integration | Next.js/FastAPI, typed contracts, safe Markdown, team authentication, ownership isolation, versioned migrations, CI, native setup and Compose |
| Portfolio/data | Account cash/holdings/ledger, settings/watchlists, validated CSV/dedup, daily/1m/5m calendars, actions/provenance, Yahoo and optional Alpaca. Live Yahoo history import verified |
| Analytics | Equal weight, maximum Sharpe, minimum volatility, risk parity, Black–Litterman and CVaR; constraints/infeasibility, daily VaR/ES, drawdown, benchmark-factor exposure and shocks |
| Simulation | Shared next-bar accounting, five templates plus news/RL strategies, costs/partial fills/cash checks/limits/actions, chronological splits, walk-forward evaluation, benchmarks and trade export |
| Paper | Persisted lifecycle/orders/fills/balances, replay and forward polling, atomic cursor updates, duplicate/retry protection, completed bars, calendar-aware stale flags and failure/restart recovery |
| Signals/cash flow | Timestamped news/public social/CSV/replay, local DistilBERT and dedup, purged chronological news-impact evaluation; baseline/Prophet/LSTM, recurring payments, intervals, alerts and optional Plaid Sandbox |
| Canadian tax | Pooled CAD ACB, transaction-date FX, taxable/registered/affiliated distinctions, partial superficial-loss windows, provisional flags, harvesting proposals and CSV export |
| RL | Shared Gymnasium accounting, actual PPO/DDPG training, seeds/checkpoints, chronological held-out comparisons, executable saved policies |
| Qwen/documents | Installed local Qwen3.5-4B, registered validated tools, bounded calls/context, progress, persisted conversations, document/page citations, local embeddings/FAISS with revision checks and lexical retrieval |
| Operations | Local jobs and shared Celery execution, PostgreSQL locked updates, SSE/polling, job cancellation/retry attempt fencing, Docker/CI configuration and automated demo/evaluation scripts |

See [verification results](VERIFICATION.md) for measured tests, local-model results and remaining environment acceptance.

Remaining external acceptance: Docker Desktop's WSL startup failure prevents a complete Compose run here. Redis/Celery transport, live Alpaca/Plaid credentials, and a long-running live paper/social-feed session still need validation in the intended environment. PostgreSQL itself was verified using an isolated native cluster. Remote CI has not been run by this session.

Research limits are explicit: synthetic fixtures are labelled; sparse bars are rejected; action publication can lag. Factor exposure uses one benchmark factor. Benchmark curves exclude simulated benchmark costs. Complex overlapping/affiliated ACB allocation is flagged for manual review. Short model runs demonstrate the workflows and do not establish investment performance.

Local/free-first team research uses virtual orders. Public onboarding, real brokerage execution and general-purpose tax filing remain outside scope.
