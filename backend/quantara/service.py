from datetime import datetime, timedelta, timezone

from . import analytics, forecast, market, news, rl, simulation, tax
from .llm import LocalAgent
from .jobs import current_job, resource_id
from .retrieval import Retrieval
from .serialization import clean
from .db import now
from .schemas import (
    Backtest,
    Cashflow,
    Chat,
    Dataset,
    MarketFetch,
    NewsImpact,
    NewsImport,
    Optimize,
    RiskRequest,
    Strategy,
    TaxRequest,
    TrainRequest,
)


def payload(record):
    return {k: v for k, v in record.items() if k != "id"}


class Services:
    def __init__(self, store, settings):
        self.store, self.settings = store, settings
        self.retrieval = Retrieval(settings.runtime, settings.embedding_model)
        self.agent = LocalAgent(settings, store, self.retrieval, self.tool)

    def dataset(self, identifier, owner):
        return Dataset.model_validate(
            payload(self.store.get("dataset", identifier, owner))
        )

    def strategy(self, identifier, owner):
        return Strategy.model_validate(
            payload(self.store.get("strategy", identifier, owner))
        )

    def policy(self, strategy, owner):
        if strategy.type != "rl":
            return None
        record = self.store.get("model", strategy.model_id, owner)
        if record["symbols"] != strategy.symbols:
            raise ValueError("Strategy symbols differ from the trained policy")
        return rl.trained_policy(record, self.settings.runtime)

    def tool(self, name, request, owner):
        operations = {
            "analyze_risk": "risk",
            "optimize_portfolio": "optimize",
            "backtest_strategy": "backtest",
            "forecast_cashflow": "cashflow",
            "research_tax": "tax",
        }
        return self.run(operations[name], request.model_dump(mode="json"), owner)

    def execute(self, operation, arguments, owner, progress=lambda *_: None):
        """Identical persisted, JSON-safe results for local jobs and Celery."""
        is_report = operation in ("risk", "optimize", "cashflow", "tax", "news_impact")
        identifier = resource_id("report", {"operation": operation, **arguments})
        if is_report and identifier:
            try:
                report = self.store.get("report", identifier, owner)
                return {"report_id": report["id"], **report["result"]}
            except KeyError:
                pass
        result = clean(self.run(operation, arguments, owner, progress))
        if is_report:
            report = self.store.create(
                "report",
                owner,
                {"name": operation, "result": result, "created_at": now()},
                identifier,
            )
            return {"report_id": report["id"], **result}
        return result

    def run(self, operation, arguments, owner, progress=lambda *_: None):
        if operation == "market":
            data = market.fetch_market(MarketFetch.model_validate(arguments))
            return self.store.create("dataset", owner, data.model_dump(mode="json"))
        if operation == "optimize":
            req = Optimize.model_validate(arguments)
            dataset = self.dataset(req.dataset_id, owner)
            frame = market.aligned_prices(dataset, list(req.holdings) or None)
            weights = analytics.optimize(
                frame,
                req.method,
                req.max_weight,
                req.rf,
                req.views,
                req.confidence,
                req.risk_aversion,
                req.allow_short,
            )
            return {
                "weights": weights,
                "method": req.method,
                "dataset_version": market.version(dataset),
                "risk": analytics.risk_report(dataset, weights),
            }
        if operation == "risk":
            req = RiskRequest.model_validate(arguments)
            return analytics.risk_report(
                self.dataset(req.dataset_id, owner),
                req.weights,
                req.benchmark,
                req.confidence,
                req.shocks,
            )
        if operation == "backtest":
            ident = resource_id("backtest", arguments)
            if ident:
                try:
                    return self.store.get("backtest", ident, owner)
                except KeyError:
                    pass
            req = Backtest.model_validate(arguments)
            strategy = self.strategy(req.strategy_id, owner)
            result = simulation.run(
                self.dataset(req.dataset_id, owner),
                strategy,
                req,
                progress,
                self.store.list("news", owner),
                self.policy(strategy, owner),
            )
            if strategy.type == "rl":
                model = self.store.get("model", strategy.model_id, owner)
                out_of_sample = result["period"]["start"][:10] > model["validation_end"]
                changed_costs = [
                    name
                    for name in (
                        "capital",
                        "commission",
                        "slippage_bps",
                        "spread_bps",
                        "participation",
                    )
                    if getattr(req, name) != model["costs"][name]
                ]
                result["model_evaluation"] = {
                    "model_id": strategy.model_id,
                    "checkpoint_sha256": model["checkpoint_sha256"],
                    "training_dataset_version": model["dataset_version"],
                    "training_end": model["train_end"],
                    "validation_end": model["validation_end"],
                    "out_of_sample_dates": out_of_sample,
                    "cost_settings_changed": changed_costs,
                    "scope": "Dates follow validation"
                    if out_of_sample
                    else "Includes training/validation dates; this is not a held-out performance estimate",
                }
            return self.store.create(
                "backtest",
                owner,
                {"name": strategy.name, "config": arguments, **result},
                ident,
            )
        if operation == "paper_step":
            identifier = arguments["paper_id"]
            # Serialize the complete account update. Cursor, fills and cash commit together.
            with self.store.edit("paper", identifier, owner) as account:
                applied = account.setdefault("applied_jobs", [])
                if current_job.get() in applied:
                    return {"id": identifier, **account}
                dataset = self.dataset(account["config"]["dataset_id"], owner)
                strategy = self.strategy(account["config"]["strategy_id"], owner)
                value = simulation.replay_step(
                    account,
                    dataset,
                    strategy,
                    min(1000, arguments.get("bars", 1)),
                    self.store.list("news", owner),
                    self.policy(strategy, owner),
                )
                account.update(value)
                if current_job.get():
                    account["applied_jobs"] = applied + [current_job.get()]
            return self.store.get("paper", identifier, owner)
        if operation == "paper_poll":
            account = self.store.get("paper", arguments["paper_id"], owner)
            if account["status"] != "running" or account["mode"] != "forward":
                raise ValueError("Only running forward accounts can poll live bars")
            strategy = self.strategy(account["config"]["strategy_id"], owner)
            old = self.dataset(account["config"]["dataset_id"], owner)
            now = datetime.now(timezone.utc)
            start = (
                datetime.fromisoformat(account["state"]["last_timestamp"])
                if account["state"]["last_timestamp"]
                else now - timedelta(days=7)
            )
            start -= timedelta(
                days=7
            )  # Overlap for gap recovery and at least three provider sessions.
            request = MarketFetch(
                symbols=list(set(strategy.symbols + [account["config"]["benchmark"]])),
                start=start,
                end=now - timedelta(seconds=60),
                interval=old.interval,
                provider=account["feed"],
            )
            try:
                incoming = market.fetch_market(request)
            except Exception:
                with self.store.edit("paper", arguments["paper_id"], owner) as saved:
                    saved.update(stale=True, polled_at=now.isoformat())
                raise
            # Only complete sessions/bars. Historical adjustments must not rewrite execution history.
            bars = market.completed_bars(incoming.bars, old.interval, now)
            merged = {(b.timestamp, b.symbol): b for b in old.bars}
            for b in bars:
                merged.setdefault((b.timestamp, b.symbol), b)
            data = old.model_copy(
                update={
                    "bars": sorted(
                        merged.values(), key=lambda b: (b.timestamp, b.symbol)
                    ),
                    "actions": old.actions
                    + [a for a in incoming.actions if a not in old.actions],
                }
            )
            market.validate_sessions(data)
            market.aligned_prices(data, strategy.symbols)
            with self.store.edit(
                "dataset", account["config"]["dataset_id"], owner
            ) as saved:
                saved.update(data.model_dump(mode="json"))
            result = self.run(
                "paper_step",
                {"paper_id": arguments["paper_id"], "bars": 1000},
                owner,
                progress,
            )
            last = result["last_feed_timestamp"]
            # Calendar-aware stale check, with a small publication grace.
            stale = market.feed_stale(old.interval, last, now)
            with self.store.edit("paper", arguments["paper_id"], owner) as saved:
                saved.update(stale=stale, polled_at=now.isoformat())
            return self.store.get("paper", arguments["paper_id"], owner)
        if operation == "cashflow":
            return forecast.forecast(Cashflow.model_validate(arguments), progress)
        if operation == "tax":
            req = TaxRequest.model_validate(arguments)
            return tax.research(
                payload(self.store.get("portfolio", req.portfolio_id, owner)),
                req.as_of,
                req.affiliated_records_complete,
                req.marks_cad,
            )
        if operation == "news":
            req = NewsImport.model_validate(arguments)
            return {"items": news.ingest(req.items, req.method, self.store, owner)}
        if operation == "news_impact":
            req = NewsImpact.model_validate(arguments)
            return news.impact(
                self.dataset(req.dataset_id, owner),
                self.store.list("news", owner),
                req.horizon_bars,
            )
        if operation == "news_fetch":
            from .schemas import NewsFetch

            req = NewsFetch.model_validate(arguments)
            items = news.alpaca_news(
                req.symbols, req.start.isoformat(), req.end.isoformat()
            )
            return {"items": news.ingest(items, req.method, self.store, owner)}
        if operation == "train":
            ident = resource_id("model", arguments)
            if ident:
                try:
                    return self.store.get("model", ident, owner)
                except KeyError:
                    pass
            req = TrainRequest.model_validate(arguments)
            result = rl.train(
                req,
                self.dataset(req.dataset_id, owner),
                self.settings.runtime,
                progress,
            )
            return self.store.create("model", owner, result, ident)
        if operation == "reindex":
            return self.retrieval.rebuild(owner, self.store.list("document", owner))
        if operation == "chat":
            return self.agent.chat(owner, Chat.model_validate(arguments), progress)
        raise ValueError("Unknown job operation")
