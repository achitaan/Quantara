# Quantara

Local-first team research: portfolio accounting, six optimizers, risk analysis, historical simulation, virtual paper trading, signals, forecasts, Canadian loss research and grounded Qwen explanations.

**Default AI: Ollama / Qwen3.5-4B. No automatic paid fallback.** Calculations work when the model is offline.

## Native setup

Python 3.12 and Node 22, from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
Copy-Item backend\.env.example .env
.\.venv\Scripts\python.exe -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000
```

Second terminal:

```powershell
cd frontend
npm ci
$env:BACKEND_URL = "http://127.0.0.1:8000"
npm run dev -- --port 3100
```

Open **http://127.0.0.1:3100**. API docs: **http://127.0.0.1:8000/docs**.
Demo login: **demo / quantara-local-demo**. Click **Load research demo** for labelled synthetic history.
Port 3100 avoids another app using 3000 on this machine.
Linux/macOS: use `python3.12` and `.venv/bin/python`. Native SQLite/local jobs support one API process; use PostgreSQL/Celery for multiple processes. Data persists under `runtime/`.

## Local models

On Windows, the workspace scripts install the pinned official Ollama release, verify its published SHA-256 checksum and run a local-only service:

```powershell
.\scripts\setup_ollama.ps1
.\scripts\serve_ollama.ps1
# In another terminal:
.\runtime\ollama\0.35.0\ollama.exe pull qwen3.5:4b
```

Alternatively install [Ollama](https://ollama.com/download) and set `OLLAMA_NO_CLOUD=1` on its actual service before `ollama serve`.
The workspace service uses an 8,192-token context, one parallel request and a bounded prompt cache.
Defaults: `LLM_PROVIDER=ollama`, `LLM_MODEL=qwen3.5:4b`, `OLLAMA_URL=http://127.0.0.1:11434`.
Cloud model tags and nonlocal Ollama hosts are rejected. See [local-only configuration](https://docs.ollama.com/faq) and [tool calling](https://docs.ollama.com/capabilities/tool-calling).

Optional DistilBERT, embeddings/FAISS, Prophet, LSTM and PPO/DDPG:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r backend\requirements-models.txt
$env:QUANTARA_MODEL_TESTS = "1"
$env:HF_HOME = Join-Path (Get-Location) "runtime/huggingface"
.\.venv\Scripts\python.exe scripts\cache_embedding_model.py
.\.venv\Scripts\python.exe -m pytest backend\tests\test_optional_models.py backend\tests\test_lightweight_models.py
```

Embedding weights are downloaded explicitly with the command above. Document inference loads cached weights offline on CPU and reuses the encoder. Other optional weights download on first use; inference/training runs locally. The financial classifier is [navu013/finbert-sentiment-distilbert](https://huggingface.co/navu013/finbert-sentiment-distilbert).
Its published benchmark is not a Quantara performance claim.
Embeddings default to `sentence-transformers/all-MiniLM-L6-v2`. Import sources and rebuild their index in Assistant.
Indexes record model/revision/dimensions/document hash and reject incompatible data. Legacy paid-embedding indexes are never loaded. Lexical search works before indexing.

Run `python scripts/benchmark_ai.py` after model setup for real latency, residency and tool/citation evaluation.
Review its saved explanations against calculated results for numerical agreement. No larger model is recommended before measurement.

Simulator charts support portfolio value, drawdown, series toggles and keyboard/pointer inspection. Models shows live training rewards, completed episode returns and held-out curves for the policy, buy-and-hold, SMA, momentum and the benchmark. Older checkpoints lack recorded learning curves and must be retrained to populate those charts. Training rewards never substitute for test performance.

Assistant streams clearly labeled reply previews while checking the final citations. Document search can be toggled; completed messages list passages, pages, retrieval mode and elapsed time. Greetings and document-only questions avoid unrelated records/tool schemas. Local inference keeps Qwen loaded for 30 minutes after a request (`LLM_KEEP_ALIVE`) and defaults to 600 output tokens (`LLM_MAX_TOKENS`); cold model loading can still delay the first reply.
Hosted inference requires explicit `LLM_PROVIDER=hosted`, HTTPS `HOSTED_LLM_URL`, `HOSTED_LLM_KEY` and `LLM_MODEL`.

## Docker / team use

```sh
docker compose up --build
# Optional Qwen:
docker compose --profile ai up --build
docker compose exec ollama ollama pull qwen3.5:4b
```

Compose starts PostgreSQL, Redis, API, Celery worker/scheduler and frontend bound to localhost.
Migrations run under a PostgreSQL advisory lock. Ollama 0.35.0 disables cloud.
Set `INSTALL_MODELS=true` before rebuilding for the optional research runtime. Named volumes preserve data/models.

For team use, set `DEMO_MODE=false`, `TEAM_USERNAME`, `TEAM_PASSWORD` and a nondefault `POSTGRES_PASSWORD`.
Add users with `python scripts/team_user.py`. Passwords are hashed; sessions use HttpOnly cookies, origins are checked, results are per user.
For HTTPS configure `COOKIE_SECURE=true` and `ALLOWED_ORIGINS`.

## Demo / validation

With the API running: `python scripts/demo.py`.
It records **import → risk → optimize → backtest → paper trading → forecast → signals → tax proposals → Qwen explanation**.
An unavailable model is recorded as an outage.
Add `--include-models` to exercise Prophet/LSTM, short seeded PPO/DDPG training, saved checkpoints and held-out policy backtests.
Use `--url http://127.0.0.1:8200` for the currently running desktop preview backend.

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

Browser tests use dedicated ports 3101/8101 and Edge on Windows; Linux CI installs Chromium.
Set `BROWSER_CHANNEL` for another installed browser. `TEST_POSTGRES_URL` enables PostgreSQL integration tests.
Optional model tests need the environment switch above. CI includes core/PostgreSQL/browser/model jobs.
Contracts: `docs/openapi.json` and `frontend/lib/api-types.ts`. Pins: `backend/requirements.txt`, `requirements-models.txt` and `frontend/package-lock.json`.
`requirements-lock.txt` records the installed core transitive versions.

See [data conventions](docs/DATA_CONVENTIONS.md), [implementation status](docs/IMPLEMENTATION.md) and [measured verification](docs/VERIFICATION.md).
Earlier experiments remain in `legacy/` and are excluded from the supported runtime.
Real brokerage execution, public onboarding and general tax filing remain outside scope.
