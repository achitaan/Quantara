import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import io
import csv
import json
from time import monotonic

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import httpx
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from . import accounting, market, simulation
from .config import Settings
from .db import Store, now
from .jobs import Jobs
from .schemas import (
    Backtest,
    Cashflow,
    Chat,
    Dataset,
    DatasetView,
    DocumentInput,
    JobView,
    Login,
    MarketFetch,
    MarketInfo,
    NewsFetch,
    NewsImpact,
    NewsImport,
    Optimize,
    PaperControl,
    PaperCreate,
    PaperStep,
    Portfolio,
    PortfolioView,
    RiskRequest,
    SettingsInput,
    Strategy,
    StrategyView,
    TaxRequest,
    TransactionsInput,
    TrainRequest,
    CashTransaction,
    PlaidExchange,
)
from .service import Services, payload
from .serialization import clean


def create_app(settings=None, store=None):
    settings = settings or Settings()
    settings.runtime.mkdir(parents=True, exist_ok=True)
    store = store or Store(settings.database_url)
    services = Services(store, settings)

    jobs = Jobs(store, services.execute, settings.task_backend, settings.redis_url)

    @asynccontextmanager
    async def lifespan(_app):
        store.seed_user(settings.demo_mode)
        for user in store.list("user", "system"):
            if settings.task_backend == "local":
                jobs.recover(user["username"])
        yield
        jobs.close()

    app = FastAPI(title="Quantara research API", version="1.0.0", lifespan=lifespan)
    app.state.store, app.state.services, app.state.jobs = store, services, jobs
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type", "Authorization"],
    )

    @app.exception_handler(ValueError)
    async def value_error(_request, exc):
        return Response(
            json.dumps({"detail": str(exc)}),
            status_code=422,
            media_type="application/json",
        )

    @app.exception_handler(KeyError)
    async def missing(_request, _exc):
        return Response(
            '{"detail":"Record not found"}',
            status_code=404,
            media_type="application/json",
        )

    @app.exception_handler(ValidationError)
    async def validation(_request, exc):
        return Response(
            exc.json(include_url=False, include_context=False),
            status_code=422,
            media_type="application/json",
        )

    def owner(request: Request):
        origin = request.headers.get("origin")
        if request.method != "GET" and origin and origin not in settings.origins:
            raise HTTPException(403, "Origin not permitted")
        token = request.cookies.get("quantara_session")
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
        try:
            return store.authenticate(token)
        except KeyError:
            raise HTTPException(401, "Sign in to Quantara") from None

    limits = defaultdict(list)

    @app.post("/api/v1/auth/login")
    def login(body: Login, request: Request, response: Response):
        origin = request.headers.get("origin")
        if origin and origin not in settings.origins:
            raise HTTPException(403, "Origin not permitted")
        key = request.client.host if request.client else "unknown"
        limits[key] = [t for t in limits[key] if monotonic() - t < 60]
        if len(limits[key]) >= 10:
            raise HTTPException(429, "Too many login attempts; try again in a minute")
        limits[key].append(monotonic())
        try:
            token = store.login(body.username, body.password)
        except ValueError:
            raise HTTPException(401, "Invalid credentials") from None
        response.set_cookie(
            "quantara_session",
            token,
            httponly=True,
            secure=settings.cookie_secure,
            samesite="strict",
            max_age=86400,
            path="/",
        )
        return {"username": body.username}

    @app.get("/api/v1/auth/me")
    def me(user=Depends(owner)):
        return {"username": user, "demo": settings.demo_mode}

    @app.post("/api/v1/auth/logout")
    def logout(request: Request, response: Response, user=Depends(owner)):
        token = request.cookies.get("quantara_session")
        if token:
            store.delete(
                "session", "session:" + sha256(token.encode()).hexdigest(), "system"
            )
        response.delete_cookie("quantara_session", path="/")
        return {"signed_out": True}

    @app.get("/health")
    @app.get("/api/v1/health")
    def health():
        ollama = False
        models = []
        if settings.llm_provider == "ollama":
            try:
                response = httpx.get(
                    settings.ollama_url + "/api/tags", timeout=1, trust_env=False
                )
                response.raise_for_status()
                models = [m["name"] for m in response.json().get("models", [])]
                ollama = settings.llm_model in models
            except (httpx.HTTPError, ValueError, KeyError):
                pass
        return {
            "status": "ok",
            "database": "ready",
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "model_ready": ollama,
            "installed_models": models,
            "hosted_fallback": False,
        }

    @app.get("/api/v1/portfolios", response_model=list[PortfolioView])
    def portfolios(user=Depends(owner)):
        return [
            r | {"balance": accounting.ledger(Portfolio.model_validate(payload(r)))}
            for r in store.list("portfolio", user)
        ]

    @app.post("/api/v1/portfolios", response_model=PortfolioView)
    def portfolio(body: Portfolio, user=Depends(owner)):
        balance = accounting.ledger(body)
        return store.create("portfolio", user, body.model_dump(mode="json")) | {
            "balance": balance
        }

    @app.post(
        "/api/v1/portfolios/{identifier}/transactions", response_model=PortfolioView
    )
    def transactions(identifier: str, body: TransactionsInput, user=Depends(owner)):
        with store.edit("portfolio", identifier, user) as saved:
            value, _ = accounting.import_transactions(
                saved, [t.model_dump(mode="json") for t in body.transactions]
            )
            saved.update(value)
        value = store.get("portfolio", identifier, user)
        return value | {
            "balance": accounting.ledger(Portfolio.model_validate(payload(value)))
        }

    @app.post("/api/v1/portfolios/{identifier}/csv")
    async def transaction_csv(
        identifier: str, file: UploadFile = File(...), user=Depends(owner)
    ):
        data = await file.read(5_000_001)
        if len(data) > 5_000_000:
            raise HTTPException(413, "CSV limit is 5 MB")
        rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
        for row in rows:
            for key in list(row):
                if row[key] == "":
                    del row[key]
        body = TransactionsInput.model_validate({"transactions": rows})
        return await run_in_threadpool(transactions, identifier, body, user)

    @app.get("/api/v1/market", response_model=list[MarketInfo])
    def datasets(user=Depends(owner)):
        result = []
        for record in store.list("dataset", user):
            d = Dataset.model_validate(payload(record))
            result.append(
                {
                    "id": record["id"],
                    **{
                        k: getattr(d, k)
                        for k in ("name", "interval", "source", "coverage", "fixture")
                    },
                    "symbols": sorted({b.symbol for b in d.bars}),
                    "start": d.bars[0].timestamp.isoformat(),
                    "end": d.bars[-1].timestamp.isoformat(),
                    "bars_count": len(d.bars),
                    "version": market.version(d),
                }
            )
        return result

    @app.post("/api/v1/market/import", response_model=MarketInfo)
    def dataset_import(body: Dataset, user=Depends(owner)):
        market.validate_sessions(body)
        market.aligned_prices(body)
        saved = store.create("dataset", user, body.model_dump(mode="json"))
        return next(d for d in datasets(user) if d["id"] == saved["id"])

    @app.get("/api/v1/market/{identifier}", response_model=DatasetView)
    def dataset_get(identifier: str, user=Depends(owner)):
        return store.get("dataset", identifier, user)

    @app.post("/api/v1/market/csv", response_model=MarketInfo)
    async def market_csv(
        file: UploadFile = File(...), options: str = Form(...), user=Depends(owner)
    ):
        data = await file.read(20_000_001)
        if len(data) > 20_000_000:
            raise HTTPException(413, "CSV limit is 20 MB")
        metadata = json.loads(options)
        bars = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
        body = Dataset.model_validate({**metadata, "bars": bars})
        return await run_in_threadpool(dataset_import, body, user)

    @app.post("/api/v1/market/fetch", response_model=JobView)
    def market_fetch(body: MarketFetch, user=Depends(owner)):
        return jobs.submit("market", body.model_dump(mode="json"), user)

    @app.get("/api/v1/strategies", response_model=list[StrategyView])
    def strategies(user=Depends(owner)):
        return store.list("strategy", user)

    @app.post("/api/v1/strategies", response_model=StrategyView)
    def strategy(body: Strategy, user=Depends(owner)):
        return store.create("strategy", user, body.model_dump(mode="json"))

    @app.put("/api/v1/strategies/{identifier}", response_model=StrategyView)
    def update_strategy(identifier: str, body: Strategy, user=Depends(owner)):
        # Running accounts pin definitions; updates require a new strategy.
        if any(
            a["config"]["strategy_id"] == identifier for a in store.list("paper", user)
        ):
            raise ValueError(
                "This strategy is pinned by a paper account. Create a new version."
            )
        with store.edit("strategy", identifier, user) as saved:
            saved.update(body.model_dump(mode="json"))
        return store.get("strategy", identifier, user)

    def job_route(path, operation, model):
        def endpoint(body: model, user=Depends(owner)):
            return jobs.submit(operation, body.model_dump(mode="json"), user)

        app.post("/api/v1/" + path, response_model=JobView)(endpoint)

    for path, op, schema in [
        ("risk", "risk", RiskRequest),
        ("portfolios/optimize", "optimize", Optimize),
        ("backtests", "backtest", Backtest),
        ("cashflow", "cashflow", Cashflow),
        ("tax/research", "tax", TaxRequest),
        ("signals/import", "news", NewsImport),
        ("signals/evaluate", "news_impact", NewsImpact),
        ("signals/fetch", "news_fetch", NewsFetch),
        ("models/train", "train", TrainRequest),
        ("conversations/chat", "chat", Chat),
    ]:
        job_route(path, op, schema)

    @app.get("/api/v1/backtests")
    def backtests(user=Depends(owner)):
        return clean(store.list("backtest", user))

    @app.get("/api/v1/backtests/{identifier}")
    def backtest_get(identifier: str, user=Depends(owner)):
        return clean(store.get("backtest", identifier, user))

    @app.get("/api/v1/backtests/{identifier}/trades.csv")
    def trades_csv(identifier: str, user=Depends(owner)):
        rows = store.get("backtest", identifier, user)["state"]["fills"]
        stream = io.StringIO()
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "id",
                "order_id",
                "timestamp",
                "symbol",
                "side",
                "quantity",
                "price",
                "commission",
                "slippage_cost",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
        return Response(
            stream.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=trades.csv"},
        )

    @app.get("/api/v1/paper-accounts")
    def paper_accounts(user=Depends(owner)):
        return store.list("paper", user)

    @app.post("/api/v1/paper-accounts")
    def paper_create(body: PaperCreate, user=Depends(owner)):
        d = services.dataset(body.config.dataset_id, user)
        s = services.strategy(body.config.strategy_id, user)
        market.aligned_prices(d, s.symbols)
        market.validate_sessions(d)
        if body.mode == "replay":
            simulation.execution_groups(d, body.config)
        elif body.config.start or body.config.end or body.config.evaluation != "full":
            raise ValueError(
                "Forward accounts use arriving bars; date ranges and test partitions apply to replay only"
            )
        if body.mode == "forward" and d.fixture:
            raise ValueError("Forward paper accounts require a real provider dataset")
        if body.mode == "forward" and "adjusted" in d.source:
            raise ValueError(
                "Forward execution requires raw bars and explicit corporate actions"
            )
        state = simulation.initial_state(body.config.capital)
        if body.mode == "forward":
            if len(
                market.completed_bars(d.bars, d.interval, datetime.now(timezone.utc))
            ) != len(d.bars):
                raise ValueError(
                    "Forward warmup must contain only completed market bars"
                )
            ts, bars = simulation.groups(d)[-1]
            simulation.advance(
                state,
                ts,
                bars,
                market.aligned_prices(d, s.symbols),
                s,
                body.config,
                d.actions,
                store.list("news", user),
                services.policy(s, user),
            )
        return store.create(
            "paper",
            user,
            {
                "name": body.name,
                "config": body.config.model_dump(mode="json"),
                "mode": body.mode,
                "feed": body.feed,
                "state": state,
                "status": "created",
                "stale": False,
                "created_at": now(),
                "last_feed_timestamp": state["last_timestamp"],
            },
        )

    @app.post("/api/v1/paper-accounts/{identifier}/control")
    def control(identifier: str, body: PaperControl, user=Depends(owner)):
        transitions = {
            "start": ("created", "running"),
            "pause": ("running", "paused"),
            "resume": ("paused", "running"),
            "stop": (None, "stopped"),
        }
        before, after = transitions[body.action]
        with store.edit("paper", identifier, user) as saved:
            if saved["status"] == "stopped" or (before and saved["status"] != before):
                raise ValueError("Invalid paper lifecycle transition")
            saved["status"] = after
            if after == "stopped":
                for order in saved["state"]["orders"]:
                    if order["status"] in ("pending", "partial"):
                        order["status"] = "cancelled"
        return store.get("paper", identifier, user)

    @app.post("/api/v1/paper-accounts/{identifier}/step", response_model=JobView)
    def step(identifier: str, body: PaperStep, user=Depends(owner)):
        account = store.get("paper", identifier, user)
        if account["mode"] != "replay":
            raise ValueError("Use poll for forward accounts")
        return jobs.submit(
            "paper_step", {"paper_id": identifier, "bars": body.bars}, user
        )

    @app.post("/api/v1/paper-accounts/{identifier}/poll", response_model=JobView)
    def poll(identifier: str, user=Depends(owner)):
        store.get("paper", identifier, user)
        return jobs.submit("paper_poll", {"paper_id": identifier}, user)

    @app.get("/api/v1/reports")
    def reports(user=Depends(owner)):
        return clean(store.list("report", user))

    @app.get("/api/v1/cashflow/transactions")
    def cashbook(user=Depends(owner)):
        try:
            return store.get("cashbook", "cashbook:" + user, user)
        except KeyError:
            return {"transactions": []}

    def save_cashbook(transactions, user):
        old = cashbook(user)["transactions"]
        records = {t["external_id"]: t for t in old}
        for raw in transactions:
            t = CashTransaction.model_validate(raw).model_dump(mode="json")
            if t["external_id"] in records and records[t["external_id"]] != t:
                raise ValueError(
                    "Cash transaction ID conflicts with an existing record"
                )
            records[t["external_id"]] = t
        value = {"transactions": sorted(records.values(), key=lambda t: t["timestamp"])}
        try:
            with store.edit("cashbook", "cashbook:" + user, user) as saved:
                saved.update(value)
        except KeyError:
            store.create("cashbook", user, value, "cashbook:" + user)
        return value

    @app.post("/api/v1/cashflow/csv")
    async def cash_csv(file: UploadFile = File(...), user=Depends(owner)):
        data = await file.read(5_000_001)
        if len(data) > 5_000_000:
            raise HTTPException(413, "CSV limit is 5 MB")
        tx = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
        return await run_in_threadpool(save_cashbook, tx, user)

    @app.post("/api/v1/cashflow/demo")
    def cash_demo(user=Depends(owner)):
        if not settings.demo_mode:
            raise HTTPException(403, "Demo fixtures are disabled")
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        tx = [
            {
                "external_id": "fixture-cash-" + str(i),
                "timestamp": (start + timedelta(days=i)).isoformat(),
                "amount": 3000 if i % 30 == 0 else -40,
                "category": "salary" if i % 30 == 0 else "spending",
                "currency": "CAD",
            }
            for i in range(180)
        ]
        return save_cashbook(tx, user)

    @app.post("/api/v1/cashflow/plaid/link-token")
    def plaid_link(user=Depends(owner)):
        from .plaid import link_token

        return {"link_token": link_token(user), "environment": "sandbox"}

    @app.post("/api/v1/cashflow/plaid/exchange")
    def plaid_exchange(body: PlaidExchange, user=Depends(owner)):
        from .plaid import exchange

        return exchange(body.public_token, store, user)

    @app.post("/api/v1/cashflow/plaid/{identifier}/sync")
    def plaid_sync(identifier: str, user=Depends(owner)):
        from .plaid import sync

        return sync(identifier, store, user)

    @app.get("/api/v1/reports/{identifier}/tax.csv")
    def tax_csv(identifier: str, user=Depends(owner)):
        result = store.get("report", identifier, user)["result"]
        if "dispositions" not in result:
            raise ValueError("This report is not a tax research result")
        stream = io.StringIO()
        columns = [
            "external_id",
            "timestamp",
            "symbol",
            "quantity",
            "proceeds_cad",
            "allocated_acb_cad",
            "gain_cad",
            "allowable_loss_cad",
            "denied_loss_cad",
            "status",
        ]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in result["dispositions"]:
            check = row.get("superficial_loss", {})
            writer.writerow(
                {k: row.get(k) for k in columns[:8]}
                | {
                    "denied_loss_cad": check.get("denied_loss_cad", 0),
                    "status": check.get("status", "final"),
                }
            )
        return Response(
            stream.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=tax-research.csv"},
        )

    @app.post("/api/v1/signals/csv", response_model=JobView)
    async def signals_csv(
        file: UploadFile = File(...),
        options: str = Form('{"method":"baseline"}'),
        user=Depends(owner),
    ):
        data = await file.read(10_000_001)
        if len(data) > 10_000_000:
            raise HTTPException(413, "CSV limit is 10 MB")
        items = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
        for item in items:
            item["symbols"] = item["symbols"].split(",")
        request = NewsImport.model_validate({**json.loads(options), "items": items})
        return jobs.submit("news", request.model_dump(mode="json"), user)

    @app.get("/api/v1/signals")
    def signals(user=Depends(owner)):
        return store.list("news", user)

    @app.get("/api/v1/models")
    def models(user=Depends(owner)):
        return store.list("model", user)

    @app.get("/api/v1/conversations")
    def conversations(user=Depends(owner)):
        return store.list("conversation", user)

    @app.get("/api/v1/conversations/{identifier}")
    def conversation(identifier: str, user=Depends(owner)):
        return store.get("conversation", identifier, user)

    @app.get("/api/v1/documents")
    def documents(user=Depends(owner)):
        return store.list("document", user)

    @app.post("/api/v1/documents")
    def document(body: DocumentInput, user=Depends(owner)):
        return store.create("document", user, body.model_dump(mode="json"))

    @app.post("/api/v1/documents/pdf")
    async def document_pdf(file: UploadFile = File(...), user=Depends(owner)):
        from pypdf import PdfReader

        data = await file.read(10_000_001)
        if len(data) > 10_000_000:
            raise HTTPException(413, "PDF limit is 10 MB")
        reader = PdfReader(io.BytesIO(data))
        if len(reader.pages) > 300:
            raise ValueError("PDF exceeds 300-page limit")
        imported = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                imported.append(
                    document(
                        DocumentInput(
                            name=file.filename or "Research PDF", text=text, page=i + 1
                        ),
                        user,
                    )
                )
        if not imported:
            raise ValueError("No extractable text. OCR the document before importing.")
        return {"pages": imported}

    @app.post("/api/v1/documents/reindex", response_model=JobView)
    def reindex(user=Depends(owner)):
        return jobs.submit("reindex", {}, user)

    @app.get("/api/v1/settings")
    def preferences(user=Depends(owner)):
        try:
            return store.get("settings", "settings:" + user, user)
        except KeyError:
            return {"watchlist": [], "theme": "dark"}

    @app.put("/api/v1/settings")
    def set_preferences(body: SettingsInput, user=Depends(owner)):
        ident = "settings:" + user
        try:
            with store.edit("settings", ident, user) as saved:
                saved.update(body.model_dump())
        except KeyError:
            store.create("settings", user, body.model_dump(), ident)
        return preferences(user)

    @app.get("/api/v1/jobs", response_model=list[JobView])
    def job_list(include_results: bool = True, user=Depends(owner)):
        records = store.list("job", user)
        if not include_results:
            return [dict(job, result=None, details=job.get("details", {}) if job["status"] in ("running", "queued") else {}) for job in records]
        return records

    @app.get("/api/v1/jobs/{identifier}", response_model=JobView)
    def job_get(identifier: str, user=Depends(owner)):
        return clean(store.get("job", identifier, user))

    @app.post("/api/v1/jobs/{identifier}/cancel", response_model=JobView)
    def cancel(identifier: str, user=Depends(owner)):
        with store.edit("job", identifier, user) as saved:
            if saved["status"] not in ("running", "queued"):
                raise ValueError("Only pending jobs can be cancelled")
            saved.update(status="cancelled", message="Cancelled", updated_at=now())
        return store.get("job", identifier, user)

    @app.post("/api/v1/jobs/{identifier}/retry", response_model=JobView)
    def retry(identifier: str, user=Depends(owner)):
        return jobs.retry(identifier, user)

    @app.get("/api/v1/jobs/{identifier}/events")
    async def events(identifier: str, include_results: bool = True, user=Depends(owner)):
        store.get("job", identifier, user)

        async def stream():
            last = None
            while True:
                job = clean(await run_in_threadpool(store.get, "job", identifier, user))
                if not include_results:
                    job["result"] = None
                    if job["status"] not in ("queued", "running"):
                        job["details"] = {}
                encoded = json.dumps(job)
                if encoded != last:
                    yield "data: " + encoded + "\n\n"
                    last = encoded
                if job["status"] in ("complete", "failed", "cancelled"):
                    break
                await asyncio.sleep(0.5)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @app.post("/api/v1/demo")
    def demo(user=Depends(owner)):
        if not settings.demo_mode:
            raise HTTPException(403, "Demo fixtures are disabled")
        existing = [
            d for d in store.list("dataset", user) if d["source"] == "fixture:seed42:v1"
        ]
        dataset = (
            existing[0]
            if existing
            else store.create("dataset", user, market.fixture().model_dump(mode="json"))
        )
        saved_portfolios = store.list("portfolio", user)
        if not saved_portfolios:
            body = Portfolio.model_validate(
                {
                    "name": "Research portfolio",
                    "transactions": [
                        {
                            "external_id": "demo-deposit",
                            "timestamp": "2024-01-02T14:30:00Z",
                            "type": "deposit",
                            "amount": 100000,
                        },
                        {
                            "external_id": "demo-aapl",
                            "timestamp": "2024-01-03T14:30:00Z",
                            "type": "buy",
                            "symbol": "AAPL",
                            "quantity": 100,
                            "price": 180,
                            "fx_cad": 1.34,
                        },
                        {
                            "external_id": "demo-msft",
                            "timestamp": "2024-01-03T14:30:00Z",
                            "type": "buy",
                            "symbol": "MSFT",
                            "quantity": 50,
                            "price": 370,
                            "fx_cad": 1.34,
                        },
                    ],
                }
            )
            p = portfolio(body, user)
        else:
            p = saved_portfolios[0]
        saved_strategies = store.list("strategy", user)
        s = (
            saved_strategies[0]
            if saved_strategies
            else strategy(
                Strategy(name="Balanced buy and hold", symbols=["AAPL", "MSFT"]), user
            )
        )
        return {
            "dataset_id": dataset["id"],
            "portfolio_id": p["id"],
            "strategy_id": s["id"],
            "fixture": True,
        }

    # Compatibility adapters use the same authenticated, local-provider path.
    @app.post("/chat", response_model=JobView)
    def legacy_chat(body: Chat, user=Depends(owner)):
        return jobs.submit("chat", body.model_dump(mode="json"), user)

    @app.get("/")
    def root():
        return {
            "application": "Quantara",
            "docs": "/docs",
            "frontend": "http://localhost:3000",
        }

    return app
