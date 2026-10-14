# Quantara

[![Quantara checks](https://github.com/achitaan/Quantara/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/achitaan/Quantara/actions/workflows/ci.yml)

Quantara is a local research workspace for portfolio accounting, strategy simulation, and explanations grounded in saved results and documents. Import transactions and market history, test an idea, inspect the trades and assumptions, then ask the assistant about the evidence.

I am building it to learn quantitative finance, machine learning, and full-stack engineering. I want to understand how a result was produced: which data a strategy could see, when an order could fill, and whether a trained model holds up against a simple baseline. That has shaped both the research engine and the interface.

![Current portfolio overview with recorded holdings and a historical simulation](docs/assets/current-dashboard.png)

_The Current interface with a synthetic test portfolio and a fixture-based simulation. The balances, holdings, and results are demonstration data, not a real account or evidence of trading performance._

## What you can do

The eleven sections cover a workflow from importing records to reviewing results:

- Record accounts, transactions, cash, and fractional holdings in Portfolio. Import daily, one-minute, or five-minute history in Market data, with exchange-calendar checks, corporate actions, and provenance. Yahoo history and optional Alpaca feeds are supported.
- Define rules in Strategies and run them in Simulator. Inspect next-bar execution, market and limit orders, fees, partial fills, chronological partitions, benchmarks, and exported trades. Walk-forward evaluation uses a fixed policy without retraining per fold.
- Run virtual accounts in Paper trading through historical replay or forward bar polling. Orders and fills persist, with duplicate protection, restart recovery, and stale-feed indicators. Native development uses the Poll action; the container scheduler polls completed provider bars.
- Research allocations and risk, including maximum Sharpe, minimum volatility, risk parity, Black-Litterman, CVaR, VaR, expected shortfall, drawdown, benchmark-factor exposure, and stress scenarios.
- Replay timestamped news and social records in Signals, compare cash-flow forecasts on chronological holdouts, or explore Canadian adjusted cost base and provisional superficial-loss cases in Tax research.
- Train seeded PPO/DDPG policies in Models and compare held-out results with buy-and-hold, SMA, momentum, and a benchmark. Learning curves and checkpoint provenance stay with the run; training rewards and held-out returns appear separately.
- Ask Assistant about saved evidence using local Qwen and validated analytical tools. It supports streamed responses, saved conversations, document excerpts, retrieval, and page citations. Optional model dependencies enable neural training, sentiment, forecasting, and document embeddings.

See [implementation status](docs/IMPLEMENTATION.md) for the detailed feature scope and [data conventions](docs/DATA_CONVENTIONS.md) before preparing an import.

## The Current interface

Current brings all eleven sections into horizontal navigation. Its aqua fabric header, petrol typography, coral accents, and locally bundled DM Sans assets carry through light and dark themes. The overview shows the active portfolio's recorded cash and currency, fractional holdings, latest simulation, watchlist, and activity. Holdings, Research, Simulation, Watchlist, and Activity tabs provide different views of those records.

The fabric header stays in place while the holdings section scrolls over it; reduced-motion settings use normal page scrolling. Responsive simulation and training charts support series toggles, drawdown views, and pointer or keyboard inspection. The browser remembers the theme even when account synchronization fails. Signing out avoids an extra protected-data refresh, and streaming polling avoids appending the same completed reply twice.

The overview is a transaction ledger. It does not supply live prices, holdings market values, daily changes, or market-value allocations. Portfolio risk research uses equal weights over recorded symbols, rather than weights inferred from share counts.

Recent cleanup gives API requests and file uploads the same response handling, so plain-text, HTML, or empty failure responses produce readable HTTP errors. Manual Refresh makes one read pass and preserves recorded results. Errors while loading an authenticated workspace are visible and can be retried.

## Run locally

Use Python 3.12 and Node.js 22, the versions configured in [CI](.github/workflows/ci.yml). Installing dependencies needs internet access. The core app and synthetic demo work without Ollama, optional neural-model packages, or a market-data provider.

On Windows, run these commands from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe scripts\setup_local.py
.\.venv\Scripts\python.exe -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000
```

Leave the API running. Open a second terminal at the repository root:

```powershell
cd frontend
npm ci
$env:BACKEND_URL = "http://127.0.0.1:8000"
npm run dev -- --port 3100
```

Open [http://127.0.0.1:3100](http://127.0.0.1:3100). Sign in with `TEAM_USERNAME` and `TEAM_PASSWORD` from the root `.env` file. The generated username defaults to `demo`. Copy the password value without its surrounding quotes; the browser does not prefill it. Setup saves credentials without printing them. Keep this ignored file private.

Select **Load research demo** in Overview, then inspect the holdings or open Simulator. The demo creates labeled synthetic records in the signed-in account. API documentation is at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

On later runs, start the API and frontend with the same server commands and reuse your credentials. If setup reports `Configuration exists`, continue with the existing root `.env`. See [local setup](docs/LOCAL_SETUP.md) for Linux/macOS commands, login recovery, local AI, optional models, Docker, and team configuration.

## How it fits together

| Path                | Responsibility                                                                                        |
| ------------------- | ----------------------------------------------------------------------------------------------------- |
| `frontend/`         | Next.js, React, and TypeScript interface, interactive charts, typed API client, and browser workflows |
| `backend/quantara/` | FastAPI services for accounting, analytics, simulation, retrieval, and model research                 |
| `backend/tests/`    | Numerical, accounting, API, ownership, recovery, and model-evaluation checks                          |
| `scripts/`          | Setup, demonstrations, audits, generated API contracts, and benchmarks                                |
| `docs/`             | Data conventions, setup guidance, implementation status, and recorded verification                    |
| `legacy/`           | Preserved prototypes, excluded from the supported runtime                                             |

One accounting and execution engine serves historical backtests, paper replay, and the Gymnasium environment. Cash, holdings, fees, splits, dividends, and rewards therefore follow the same rules. Training records include seeds, requested and actual steps, costs, data fingerprints, package versions, and checkpoint checksums.

Native development uses SQLite and local executors with one API process. The container configuration uses PostgreSQL and Redis/Celery. Jobs have saved IDs, polling/SSE progress, retry, and cancellation. Databases, downloaded weights, and checkpoints live under the ignored `runtime/` directory in native development.

The assistant keeps numerical calculations in Python, validates analytical tool arguments, and retrieves document passages with citation checks. The generated [OpenAPI contract](docs/openapi.json) and frontend API types are checked together in CI to catch drift between the API and its client.

## Research limits

- All orders are virtual. Real brokerage execution and public onboarding are outside the current scope.
- Evaluation separates training, validation, and test history and enforces next-bar fills. Short synthetic runs demonstrate the mechanics; they do not establish predictive accuracy or trading outperformance. Saved comparisons include cases where a neural model underperforms a simple baseline.
- Fills use bar-based assumptions for costs and volume participation. Order-book queues, market impact, settlement delays, and dividend withholding are not modeled. Supplied dividend event dates determine credit timing. The [research audit](docs/RESEARCH_ACCURACY.md) documents the execution rules and leakage checks.
- Canadian tax outputs are research calculations. Missing affiliated records, unfinished future windows, and complex overlapping transactions remain provisional or need review. The tool does not produce a tax filing.
- Seeded results are checked on the tested runtime. Hardware and library versions can affect reproducibility. Older checkpoints need retraining to record learning curves that were not saved originally.
- Assistant explanations still need comparison with the cited evidence. Valid citation identifiers alone do not establish that every statement is correct.
- Live-provider behavior, prolonged forward-feed recovery, and the full container stack need acceptance checks in their intended environment. Plaid support is Sandbox-only.

## Verify a change

From the repository root, with native dependencies installed:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check backend\quantara backend\tests scripts
.\.venv\Scripts\python.exe scripts\export_openapi.py
cd frontend
npm run types
npm run typecheck
npm run build
npm run test:e2e
```

Browser tests start an isolated API and production frontend on ports 8101/3101, which must be free. Build the frontend first. Windows uses installed Edge by default; Linux needs Playwright's Chromium installed. The [verification guide](docs/LOCAL_SETUP.md#verification-and-walkthroughs) covers browser selection, optional model checks, PostgreSQL integration, and demo scripts.

The [verification history](docs/VERIFICATION.md), [simulation and training audit](docs/RESEARCH_ACCURACY.md), and [visualization and assistant report](docs/VISUALIZATION_AND_ASSISTANT.md) record earlier runs and their limits. Their counts and timings describe those runs, not a fresh result for every revision. The CI badge above links to the current workflow status.
