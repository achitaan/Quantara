# Local setup and recovery

Start with the [README quickstart](../README.md#run-locally) for the core app. These commands assume the repository root unless a block changes directory. The supported CI versions are Python 3.12 and Node.js 22. Native examples use PowerShell on Windows; Docker commands require Docker Compose and a running Docker engine.

## Native development

The root `.env` contains the configured login and database settings. `scripts/setup_local.py` generates private credentials from [the example configuration](../backend/.env.example). The default username is `demo`; copy the password value without surrounding quotes when signing in. Setup reports that credentials were saved without printing them.

On subsequent runs, start the API from the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000
```

Start the frontend in a second terminal at the repository root:

```powershell
cd frontend
$env:BACKEND_URL = "http://127.0.0.1:8000"
npm run dev -- --port 3100
```

Open [http://127.0.0.1:3100](http://127.0.0.1:3100). Stop each server with Ctrl+C. Local databases, downloaded weights, and checkpoints live under the ignored `runtime/` directory. Native SQLite and local jobs support one API process; use PostgreSQL and Celery when running multiple processes.

### Linux and macOS

Use these Python commands from the repository root for the first run:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements.txt
.venv/bin/python scripts/setup_local.py
.venv/bin/python -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000
```

In a second terminal at the repository root:

```sh
cd frontend
npm ci
BACKEND_URL=http://127.0.0.1:8000 npm run dev -- --port 3100
```

For later Python commands in this guide, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python` and use forward slashes in file paths.

### Recover an existing login

`setup_local.py` refuses to replace a configured login and reports `Configuration exists`. Continue with the existing root `.env` credentials. To intentionally replace the native login password, stop the API and run this from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\setup_local.py --rotate-login
```

Restart the API and sign in with the new password from `.env`. Rotation revokes existing sessions. It does not change the password of an already initialized PostgreSQL role; that requires a separate database password change. Keep `.env` private and do not delete `runtime/` to recover a login, because it contains saved research data.

## Local Qwen through Ollama

The default is Ollama with `qwen3.5:4b`. Calculations and simulations remain available when the language model is offline. Local mode rejects cloud model tags and nonlocal Ollama hosts and has no automatic hosted fallback.

From the repository root on Windows:

```powershell
.\scripts\setup_ollama.ps1
.\scripts\serve_ollama.ps1
```

Leave the service running and pull the model in another terminal:

```powershell
.\runtime\ollama\0.35.0\ollama.exe pull qwen3.5:4b
```

The setup script downloads the pinned official Ollama release and checks its published SHA-256 before extracting it. The service script binds to localhost and disables cloud features. Alternatively, use an installed Ollama service with `OLLAMA_NO_CLOUD=1`, then run `ollama serve` and `ollama pull qwen3.5:4b`.

| Root `.env` setting | Default                                  | Purpose                                   |
| ------------------- | ---------------------------------------- | ----------------------------------------- |
| `LLM_PROVIDER`      | `ollama`                                 | Inference provider                        |
| `LLM_MODEL`         | `qwen3.5:4b`                             | Model tag                                 |
| `OLLAMA_URL`        | `http://127.0.0.1:11434`                 | Backend connection to Ollama              |
| `LLM_MAX_TOKENS`    | `600`                                    | Maximum generated response length         |
| `LLM_KEEP_ALIVE`    | `30m`                                    | Requested model residency after inference |
| `EMBEDDING_MODEL`   | `sentence-transformers/all-MiniLM-L6-v2` | Document encoder                          |

The bundled service uses an 8,192-token context and one parallel inference request. Latency depends on hardware, prompt size, and whether the model is loaded. The [assistant report](VISUALIZATION_AND_ASSISTANT.md) records measurements from a specific local machine; they are not performance guarantees.

Hosted inference is an explicit configuration choice: `LLM_PROVIDER=hosted`, an HTTPS `HOSTED_LLM_URL`, `HOSTED_LLM_KEY`, and a configured model. Python services still perform numerical calculations.

## Optional models and document embeddings

PPO/DDPG, DistilBERT, Prophet, LSTM, and local embedding/FAISS retrieval need the optional dependencies. From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r backend\requirements-models.txt
$env:HF_HOME = Join-Path (Get-Location) "runtime/huggingface"
.\.venv\Scripts\python.exe scripts\cache_embedding_model.py
```

Cache the embedding weights once with internet access. The script reports the encoder name and dimension, then prompts you to rebuild the index. Retrieval loads the cached encoder offline on CPU. Other optional models may download weights on first use. Import documents in Assistant and build the local index. Text search works before indexing; stale or incompatible indexes prompt a rebuild.

## Containers and team configuration

Generate the root `.env` with `scripts/setup_local.py` before starting Docker Compose. Reuse existing credentials if setup is complete. Check [compose.yaml](../compose.yaml) when changing service configuration.

```sh
docker compose up --build
```

Compose defines PostgreSQL, Redis, the API, Celery worker and scheduler, and the frontend at [http://127.0.0.1:3000](http://127.0.0.1:3000). Named volumes preserve database and runtime data. Stop the stack with `docker compose down`; avoid `--volumes` unless you intend to delete that data.

To include Ollama, use the AI profile, then pull the model from another terminal:

```sh
docker compose --profile ai up --build
docker compose exec ollama ollama pull qwen3.5:4b
```

Set `INSTALL_MODELS=true` in the root `.env` before rebuilding to include optional model dependencies. Initialize embedding weights in the shared runtime volume before indexing:

```sh
docker compose exec api python -c "from quantara.config import settings; from sentence_transformers import SentenceTransformer; SentenceTransformer(settings.embedding_model, device='cpu')"
```

For a team environment, configure `DEMO_MODE=false`, `TEAM_USERNAME`, `TEAM_PASSWORD`, and a private `POSTGRES_PASSWORD`. The native command below creates an additional login interactively in the configured database. It requires a password of at least 12 characters and refuses to replace an existing user.

```powershell
.\.venv\Scripts\python.exe scripts\team_user.py
```

Passwords are hashed, sessions use HttpOnly cookies, and API requests enforce ownership and origin checks. For HTTPS deployment, set `COOKIE_SECURE=true` and the intended `ALLOWED_ORIGINS` in the API environment. The supplied Compose file fixes allowed origins to local port 3000 and does not forward `COOKIE_SECURE` from `.env`; update its API environment mapping for deployment.

CI validates Compose configuration. Full Compose/Redis/Celery startup remains an acceptance check for the intended environment; native checks do not establish container deployment readiness.

## Verification and walkthroughs

The [README](../README.md#verify-a-change) lists the core checks from [CI](../.github/workflows/ci.yml). Browser tests start their own API and production frontend on ports 8101/3101 using an isolated database and test credential. Those ports must be free; build the frontend first.

Windows uses installed Edge by default. Set `BROWSER_CHANNEL` to use another installed Chromium-based browser. Linux CI installs Chromium with `npx playwright install --with-deps chromium`; a local Linux run also needs that browser installed. On Linux/macOS, run Playwright from a terminal with the virtual environment's Python on `PATH`, or set `PYTHON` to its absolute path. See [the Playwright configuration](../frontend/playwright.config.ts) for server startup and browser defaults.

The test API deliberately points Ollama at an unavailable port to check model-outage usability. Real Qwen behavior is measured separately. To check optional models after installing dependencies and caching weights, start a new terminal at the repository root:

```powershell
$env:QUANTARA_MODEL_TESTS = "1"
$env:HF_HOME = Join-Path (Get-Location) "runtime/huggingface"
.\.venv\Scripts\python.exe -m pytest backend\tests\test_optional_models.py backend\tests\test_lightweight_models.py backend\tests\test_research_accuracy.py
cd frontend
npm run test:e2e
```

`QUANTARA_MODEL_TESTS=1` also enables the policy-training browser workflow. `TEST_POSTGRES_URL` enables the PostgreSQL integration check. CI runs core, PostgreSQL, and browser checks separately from the optional model job, which caches embeddings before offline retrieval tests.

For an application walkthrough, leave the native API running and use another terminal at the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\demo.py --url http://127.0.0.1:8000
# Add forecasts, seeded training, and checkpoint backtests:
.\.venv\Scripts\python.exe scripts\demo.py --url http://127.0.0.1:8000 --include-models
# With Qwen running and its configured model installed:
.\.venv\Scripts\python.exe scripts\benchmark_ai.py --url http://127.0.0.1:8000
```

These scripts read the configured login from `.env` and create synthetic demo records in that account. The benchmark records tools, citations, latency, and residency under `runtime/`. Use a research account if you want to keep those records separate from your own imports.

The [verification history](VERIFICATION.md), [simulation and training audit](RESEARCH_ACCURACY.md), and [visualization and assistant report](VISUALIZATION_AND_ASSISTANT.md) retain the evidence and scope of previous runs. Their counts and timings describe those runs, not a fresh result for every revision.
