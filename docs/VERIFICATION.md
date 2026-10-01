# Verification record — 2026-10-01

The desktop preview runs at http://127.0.0.1:3100 with the API at http://127.0.0.1:8200.
Demo login: `demo / [removed public demo credential]`. Data, model weights, checkpoints and full evaluation artifacts are kept in ignored `runtime/`.

The subsequent [simulation and training accuracy audit](RESEARCH_ACCURACY.md) fixes execution/evaluation issues and records expanded tests and three-seed model results. Earlier policy checkpoints require retraining for observation version 2; earlier simulator reports should be rerun. The model figures below describe the earlier release demonstration.

## Automated checks

- Final core suite: **30 passed, 4 optional-model/PostgreSQL tests skipped**. PostgreSQL migrations and concurrent locked updates also passed against an isolated native cluster. Checks cover numerical/accounting constraints, actions, missing bars/calendars, no-lookahead execution, partial fills, replay/restart/duplicates, ownership, CSV, migrations, tool/citation/outage guards and retry fencing.
- Installed local-model suite: **5 passed**. Actual DistilBERT classification, local embeddings/FAISS retrieval, Prophet/LSTM forecasts, Gymnasium accounting and PPO/DDPG training/checkpoint/held-out evaluation were exercised.
- Ruff, API contract generation and frontend TypeScript checks pass. Production Next.js build and desktop/mobile browser workflows pass, including persisted watchlists.
- Compose configuration parses. Docker Desktop cannot start its Linux engine on this machine; the full container/Redis/Celery stack remains unverified here. The native PostgreSQL verification does not replace that check.
- A real Yahoo daily history import returned 63 aligned AAPL/MSFT/SPY bars for January 2024 using reconstructed raw historical units.

## Local Qwen

Ollama **0.35.0**, `qwen3.5:4b`, Q4_K_M, 8,192-token context, cloud disabled. Calculations and simulations remain usable during model outages; tests confirm local chat makes no hosted inference calls.

Final isolated evaluation, saved as `runtime/qwen-evaluation-acceptance.json`:

| Case | Seconds | Outcome |
|---|---:|---|
| Risk | 20.41 | Correct tool, valid arguments, recorded VaR/ES values and valid source identifiers |
| Equal-weight optimization | 12.58 | Correct tool, valid arguments and cited calculated metrics |
| Canadian tax | 16.30 | Correct tool, CAD ACB values, explicit record-completeness and missing-mark explanations |
| Document question | 17.84 | Correct CAD 12,345 fixture answer and exact supplied document/page citation |

These were warm requests on this machine, not a broad quality benchmark. The numerical examples were reviewed against their saved Python reports. Source-identifier checks do not prove every prose claim. Earlier evaluations exposed skipped tools, invented citations, optional-argument confusion and prompt truncation; guards and explicit result metadata were added in response. The full-demo explanation also misstated volatility and compared different evaluation periods; explicit period metadata and a basic volatility invariant guard were added. These guards are partial, and explanations still need review against the recorded evidence.

Hardware: approximately **16 GB system RAM**, **GTX 1660 SUPER with 6 GB VRAM**. Ollama reports **3.27 GB** loaded model/VRAM allocation; the GPU's total observed usage during evaluation was about **5,196 MiB**, including runtime buffers and other GPU use. This is not a measured host peak-RSS figure. The prompt cache is capped at 256 MB. No larger model is recommended from this measurement.

## Model evaluation scope

The full demonstration includes real local forecasts and short seeded policy training. On the synthetic seed-42 fixture, an earlier 128-step PPO run returned **0.14%**, DDPG **1.49%**, and buy-and-hold **1.81%** on the untouched 51-observation test partition. Both policies underperformed that comparator. These small training runs establish executable workflows, not trained-model quality.

One synthetic cash-flow example produced held-out MAE **156.27 CAD** for LSTM and **154.96 CAD** for Prophet, against a **191.21 CAD** baseline. These single-fixture results do not establish general forecasting accuracy. Intervals are labelled empirical held-out-residual research uncertainty.

Run `scripts/demo.py --include-models` to reproduce the complete workflow, and `scripts/benchmark_ai.py` for a fresh recorded Qwen evaluation. Their `--url` option selects the running API.
The full release demonstration is saved at `runtime/demo-release.json`, including every engine result, local model evaluation and Qwen explanation. For stored-result explanations without inline model citations, the application attaches the available records separately for verification; that does not certify claim-level citation accuracy.

## Remaining environment acceptance

Validate full Compose startup and Redis/Celery transport in a working Docker environment; run CI after publishing the branch. Exercise Alpaca and Plaid Sandbox using supplied credentials, then soak-test live paper polling, reconnects and social-feed ingestion. No live brokerage orders or tax filings are implemented.
