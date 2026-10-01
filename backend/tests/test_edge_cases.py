from dataclasses import replace
from datetime import datetime, timezone
from copy import deepcopy

import pytest
import httpx

from quantara import market, news, simulation, tax
from quantara.config import Settings
from quantara.db import Store
from quantara.llm import LocalAgent
from quantara.retrieval import Retrieval
from quantara.schemas import Backtest, Chat, NewsItem, Strategy


def test_migration_preserves_v1_records_and_orders_new_records(tmp_path):
    from sqlalchemy import create_engine, text

    url = "sqlite:///" + str(tmp_path / "old.db")
    engine = create_engine(url)
    with engine.begin() as connection:
        connection.execute(
            text("CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY)")
        )
        connection.execute(text("INSERT INTO schema_migrations VALUES (1)"))
        connection.execute(
            text(
                "CREATE TABLE records (id VARCHAR(80) PRIMARY KEY, kind VARCHAR(40) NOT NULL, owner VARCHAR(100) NOT NULL, payload JSON NOT NULL, version INTEGER NOT NULL)"
            )
        )
        connection.execute(
            text("INSERT INTO records VALUES ('original','note','team',:payload,1)"),
            {"payload": '{"text":"preserved"}'},
        )
    engine.dispose()
    store = Store(url)
    assert store.get("note", "original", "team")["text"] == "preserved"
    store.create("note", "team", {"text": "first"}, "z-first")
    store.create("note", "team", {"text": "second"}, "a-second")
    with store.edit("note", "z-first", "team") as record:
        record["text"] = "edited first"
    assert [r["id"] for r in store.list("note", "team")] == [
        "original",
        "z-first",
        "a-second",
    ]
    with store.engine.connect() as connection:
        assert list(
            connection.scalars(
                text("SELECT version FROM schema_migrations ORDER BY version")
            )
        ) == [1, 2]


def test_explicit_tool_request_is_not_silently_skipped(tmp_path, monkeypatch):
    store = Store("sqlite:///" + str(tmp_path / "ai-tools.db"))
    agent = LocalAgent(
        replace(Settings(), runtime=tmp_path),
        store,
        Retrieval(tmp_path, "test"),
        lambda *_: {},
    )
    requests = []

    def no_tool(messages):
        requests.append(messages[-1]["content"])
        return {"content": "A previous calculation exists."}

    monkeypatch.setattr(agent, "request", no_tool)
    result = agent.chat("team", Chat(message="Use research_tax for portfolio abc"))
    assert len(requests) == 4
    assert result["mode"] == "limited"
    assert "did not execute" in result["warning"]


def test_oversized_local_context_is_rejected_before_inference(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Oversized context should not reach a model request")

    monkeypatch.setattr(httpx, "post", forbidden)
    agent = LocalAgent(
        replace(Settings(), runtime=tmp_path),
        Store("sqlite://"),
        Retrieval(tmp_path, "test"),
        lambda *_: {},
    )
    with pytest.raises(ValueError, match="context budget"):
        agent.request([{"role": "system", "content": "x" * 18000}])


def test_impossible_volatility_and_invented_dataset_citation_are_flagged(
    tmp_path, monkeypatch
):
    agent = LocalAgent(
        replace(Settings(), runtime=tmp_path),
        Store("sqlite://"),
        Retrieval(tmp_path, "test"),
        lambda *_: {},
    )
    monkeypatch.setattr(
        agent,
        "request",
        lambda _: {"content": "The strategy has negative volatility [dataset:ID:pN]."},
    )
    result = agent.chat("team", Chat(message="Explain the results"))
    assert result["mode"] == "limited"
    assert "metric check" in result["content"]
    assert result["citation_validation"]["invalid"] == ["dataset:ID:pN"]


def test_missing_sessions_and_minutes():
    d = market.fixture()
    missing = d.model_copy(update={"bars": d.bars[:9] + d.bars[12:]})
    with pytest.raises(ValueError, match="Missing"):
        market.validate_sessions(missing)
    d = market.fixture("5m")
    assert market.validate_sessions(d) is d
    missing = d.model_copy(update={"bars": d.bars[:9] + d.bars[12:]})
    with pytest.raises(ValueError, match="Missing"):
        market.validate_sessions(missing)


def test_completed_bars_and_calendar_feed_age():
    def utc(value):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    daily = market.fixture().bars[:9]
    assert len(market.completed_bars(daily, "1d", utc("2024-01-03T20:59:00Z"))) == 3
    assert len(market.completed_bars(daily, "1d", utc("2024-01-03T21:00:00Z"))) == 6
    intraday = market.fixture("5m").bars[:6]
    assert len(market.completed_bars(intraday, "5m", utc("2024-01-02T14:35:00Z"))) == 3
    friday = "2024-01-12T14:30:00+00:00"
    assert not market.feed_stale("1d", friday, utc("2024-01-15T20:00:00Z"))
    assert not market.feed_stale("1d", friday, utc("2024-01-16T21:10:00Z"))
    assert market.feed_stale("1d", friday, utc("2024-01-16T21:16:00Z"))
    last = "2024-01-12T20:55:00+00:00"
    assert not market.feed_stale("5m", last, utc("2024-01-16T14:40:00Z"))
    assert market.feed_stale("5m", last, utc("2024-01-16T15:00:00Z"))


def test_daily_and_intraday_flat_market_agree():
    daily = market.fixture()
    daily = daily.model_copy(
        update={
            "bars": [
                b.model_copy(
                    update={
                        "open": 100,
                        "high": 100,
                        "low": 100,
                        "close": 100,
                        "volume": 100000,
                    }
                )
                for b in daily.bars[:18]
            ]
        }
    )
    intraday = market.fixture("5m")
    intraday = intraday.model_copy(
        update={
            "bars": [
                b.model_copy(
                    update={
                        "open": 100,
                        "high": 100,
                        "low": 100,
                        "close": 100,
                        "volume": 100000,
                    }
                )
                for b in intraday.bars
            ]
        }
    )
    s = Strategy(name="Hold", symbols=["AAPL"])
    c = Backtest(dataset_id="test", strategy_id="hold", slippage_bps=0)
    a, b = simulation.run(daily, s, c), simulation.run(intraday, s, c)
    assert a["state"]["holdings"] == b["state"]["holdings"] == {"AAPL": 1000}
    assert a["metrics"]["total_return"] == b["metrics"]["total_return"] == 0


def test_news_duplicates_do_not_erase_different_dates(tmp_path):
    store = Store("sqlite:///" + str(tmp_path / "news.db"))
    item = NewsItem(
        external_id="first",
        timestamp="2024-01-02T14:30:00Z",
        available_at="2024-01-02T14:30:00Z",
        symbols=["AAPL"],
        text="Strong profit growth",
        source="fixture",
    )
    following = item.model_copy(
        update={
            "external_id": "next",
            "timestamp": datetime(2024, 1, 3, 14, 30, tzinfo=timezone.utc),
            "available_at": datetime(2024, 1, 3, 14, 30, tzinfo=timezone.utc),
        }
    )
    news.ingest([item, item, following], "baseline", store, "team")
    assert len(store.list("news", "team")) == 2


def test_news_training_coefficients_ignore_test_prices():
    d = market.fixture()
    items = [
        {
            "symbols": ["AAPL"],
            "available_at": b.timestamp.isoformat(),
            "score": (0.5 if i % 2 else -0.5),
        }
        for i, b in enumerate([b for b in d.bars if b.symbol == "AAPL"][::10])
    ]
    result = news.impact(d, items, 5)
    boundary = datetime.fromisoformat(result["test_start"])
    altered = d.model_copy(
        update={
            "bars": [
                b.model_copy(
                    update={
                        "open": b.open * 2,
                        "high": b.high * 2,
                        "low": b.low * 2,
                        "close": b.close * 2,
                    }
                )
                if b.timestamp >= boundary
                else b
                for b in d.bars
            ]
        }
    )
    changed = news.impact(altered, items, 5)
    assert result["coefficients"] == changed["coefficients"]


def test_registered_replacement_and_missing_fx():
    records = {
        "name": "Tax",
        "transactions": [
            {
                "external_id": "buy",
                "timestamp": "2024-01-02T14:30:00Z",
                "symbol": "AAPL",
                "type": "buy",
                "quantity": 100,
                "price": 10,
                "currency": "CAD",
            },
            {
                "external_id": "sell",
                "timestamp": "2024-04-02T14:30:00Z",
                "symbol": "AAPL",
                "type": "sell",
                "quantity": 100,
                "price": 8,
                "currency": "CAD",
            },
            {
                "external_id": "replace",
                "timestamp": "2024-04-10T14:30:00Z",
                "symbol": "AAPL",
                "type": "buy",
                "quantity": 100,
                "price": 8,
                "currency": "CAD",
                "account": "rrsp",
                "account_type": "registered",
            },
        ],
    }
    result = tax.research(records, datetime(2024, 6, 1, tzinfo=timezone.utc), True)
    assert result["dispositions"][0]["superficial_loss"]["denied_loss_cad"] == 200
    assert result["pools"]["AAPL"]["acb_cad"] == 0
    missing = deepcopy(records)
    missing["transactions"][0]["currency"] = "USD"
    result = tax.research(missing, datetime(2024, 6, 1, tzinfo=timezone.utc), True)
    assert result["incomplete_records"] and result["provisional"]


def test_model_malformed_and_bound_loop(tmp_path, monkeypatch):
    store = Store("sqlite:///" + str(tmp_path / "ai.db"))
    settings = replace(Settings(), runtime=tmp_path)
    agent = LocalAgent(
        settings, store, Retrieval(tmp_path, settings.embedding_model), lambda *_: {}
    )
    monkeypatch.setattr(
        agent, "request", lambda _: {"tool_calls": [{"function": None}]}
    )
    assert agent.chat("team", Chat(message="Test malformed"))["warning"]
    monkeypatch.setattr(
        agent,
        "request",
        lambda _: {"tool_calls": [{"function": {"name": "unknown", "arguments": {}}}]},
    )
    result = agent.chat("team", Chat(message="Test bounds"))
    assert len(result["tools"]) == 4
    assert "loop limit" in result["warning"]
