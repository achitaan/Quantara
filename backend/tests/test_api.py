from dataclasses import replace
import json
import time
from datetime import datetime, timezone

from fastapi.testclient import TestClient
import httpx
import pytest

from quantara.api import create_app
from quantara.config import Settings
from quantara.db import Store
from quantara.llm import LocalAgent
from quantara.retrieval import Retrieval
from quantara.schemas import Chat
from quantara.jobs import current_job


@pytest.fixture
def client(tmp_path):
    settings = replace(
        Settings(),
        runtime=tmp_path,
        database_url="sqlite:///" + str(tmp_path / "test.db"),
    )
    app = create_app(settings)
    with TestClient(app) as c:
        assert (
            c.post(
                "/api/v1/auth/login",
                json={"username": "demo", "password": "quantara-local-demo"},
            ).status_code
            == 200
        )
        yield c


def wait(client, response):
    assert response.status_code == 200, response.text
    identifier = response.json()["id"]
    for _ in range(200):
        value = client.get("/api/v1/jobs/" + identifier).json()
        if value["status"] in ("complete", "failed", "cancelled"):
            assert value["status"] == "complete", value
            return value["result"]
        time.sleep(0.03)
    pytest.fail("Job timeout")


def test_end_to_end_research(client, monkeypatch):
    def offline(*_args, **_kwargs):
        raise httpx.ConnectError("No Ollama in test")

    monkeypatch.setattr(httpx, "post", offline)
    monkeypatch.setattr(httpx, "stream", offline)
    d = client.post("/api/v1/demo").json()
    assert client.get("/api/v1/market").json()[0]["fixture"]
    risk = wait(
        client,
        client.post(
            "/api/v1/risk",
            json={"dataset_id": d["dataset_id"], "weights": {"AAPL": 0.5, "MSFT": 0.5}},
        ),
    )
    assert risk["expected_shortfall"] >= risk["historical_var"]
    opt = wait(
        client,
        client.post(
            "/api/v1/portfolios/optimize",
            json={
                "dataset_id": d["dataset_id"],
                "method": "risk_parity",
                "holdings": {"AAPL": 0.5, "MSFT": 0.5},
            },
        ),
    )
    assert sum(opt["weights"].values()) == pytest.approx(1)
    bt = wait(
        client,
        client.post(
            "/api/v1/backtests",
            json={"dataset_id": d["dataset_id"], "strategy_id": d["strategy_id"]},
        ),
    )
    assert len(bt["state"]["fills"]) > 0
    assert (
        "timestamp" in client.get("/api/v1/backtests/" + bt["id"] + "/trades.csv").text
    )
    paper = client.post(
        "/api/v1/paper-accounts",
        json={
            "name": "Replay",
            "config": {"dataset_id": d["dataset_id"], "strategy_id": d["strategy_id"]},
        },
    ).json()
    assert (
        client.post(
            "/api/v1/paper-accounts/" + paper["id"] + "/control",
            json={"action": "start"},
        ).status_code
        == 200
    )
    stepped = wait(
        client,
        client.post(
            "/api/v1/paper-accounts/" + paper["id"] + "/step", json={"bars": 20}
        ),
    )
    assert stepped["state"]["steps"] == 20
    assert (
        stepped["state"]["fills"]
        == bt["state"]["fills"][: len(stepped["state"]["fills"])]
    )
    tx = client.post("/api/v1/cashflow/demo").json()["transactions"]
    flow = wait(client, client.post("/api/v1/cashflow", json={"transactions": tx}))
    assert len(flow["forecast"]) == 30
    tax = wait(
        client,
        client.post(
            "/api/v1/tax/research",
            json={
                "portfolio_id": d["portfolio_id"],
                "as_of": "2025-01-01T00:00:00Z",
                "marks_cad": {"AAPL": 150, "MSFT": 450},
            },
        ),
    )
    assert tax["harvesting_proposals"] and tax["provisional"]
    chat = wait(
        client,
        client.post(
            "/api/v1/conversations/chat",
            json={"message": "Explain my results", "dataset_id": d["dataset_id"]},
        ),
    )
    assert chat["warning"] and chat["provider"] == "ollama"
    assert len(client.get("/api/v1/conversations").json()) == 1


def test_auth_ownership_and_csrf(client):
    d = client.post("/api/v1/demo").json()
    client.app.state.store.team_user("other", "long-password")
    client.cookies.clear()
    assert client.get("/api/v1/market").status_code == 401
    assert (
        client.post(
            "/api/v1/auth/login",
            json={"username": "other", "password": "long-password"},
        ).status_code
        == 200
    )
    assert client.get("/api/v1/market").json() == []
    assert client.get("/api/v1/market/" + d["dataset_id"]).status_code == 404
    assert (
        client.post(
            "/api/v1/demo", headers={"origin": "https://untrusted.example"}
        ).status_code
        == 403
    )


def test_csv_validation_is_atomic(client):
    d = client.post("/api/v1/demo").json()
    path = "/api/v1/portfolios/" + d["portfolio_id"] + "/csv"
    before = client.get("/api/v1/portfolios").json()
    csv = "external_id,timestamp,type,symbol,quantity,price\nbad,2024-05-01T14:30:00Z,sell,AAPL,100000,200\n"
    assert (
        client.post(path, files={"file": ("trades.csv", csv, "text/csv")}).status_code
        == 422
    )
    assert client.get("/api/v1/portfolios").json() == before


def test_paper_retry_after_commit_does_not_execute_twice(client):
    d = client.post("/api/v1/demo").json()
    paper = client.post(
        "/api/v1/paper-accounts",
        json={
            "name": "Retry test",
            "config": {
                "dataset_id": d["dataset_id"],
                "strategy_id": d["strategy_id"],
            },
        },
    ).json()
    client.post(
        "/api/v1/paper-accounts/" + paper["id"] + "/control", json={"action": "start"}
    )
    jobs = client.app.state.jobs
    original = jobs.runner

    def interrupted(operation, arguments, owner, progress):
        original(operation, arguments, owner, progress)
        raise ValueError("Simulated worker failure after account commit")

    jobs.runner = interrupted
    job = client.post(
        "/api/v1/paper-accounts/" + paper["id"] + "/step", json={"bars": 20}
    ).json()
    for _ in range(200):
        if client.get("/api/v1/jobs/" + job["id"]).json()["status"] == "failed":
            break
        time.sleep(0.03)
    else:
        pytest.fail("Failure was not recorded")
    before = client.app.state.store.get("paper", paper["id"], "demo")
    jobs.runner = original
    retried = wait(client, client.post("/api/v1/jobs/" + job["id"] + "/retry"))
    assert retried["state"] == before["state"]
    assert retried["state"]["steps"] == 20
    assert retried["applied_jobs"].count(job["id"]) == 1
    events = client.get("/api/v1/jobs/" + job["id"] + "/events")
    assert events.headers["content-type"].startswith("text/event-stream")
    assert '"status": "complete"' in events.text


def test_worker_result_persistence_and_retry(client):
    d = client.post("/api/v1/demo").json()
    services = client.app.state.services
    arguments = {"dataset_id": d["dataset_id"], "weights": {"AAPL": 1}}
    token = current_job.set("worker-test")
    try:
        first = services.execute("risk", arguments, "demo")
        repeated = services.execute("risk", arguments, "demo")
    finally:
        current_job.reset(token)
    assert first == repeated
    saved = client.app.state.store.get("report", first["report_id"], "demo")
    assert saved["result"]["historical_var"] == first["historical_var"]
    assert len(client.app.state.store.list("report", "demo")) == 1


def test_persisted_watchlist_and_settings(client):
    saved = client.put(
        "/api/v1/settings", json={"watchlist": ["AAPL", "MSFT"], "theme": "light"}
    )
    assert saved.status_code == 200
    assert client.get("/api/v1/settings").json()["watchlist"] == ["AAPL", "MSFT"]
    assert (
        client.put("/api/v1/settings", json={"watchlist": ["bad ticker!"]}).status_code
        == 422
    )


def test_forward_arrivals_are_complete_immutable_and_restart_safe(client, monkeypatch):
    from quantara import market, service

    fixture = market.fixture()
    warmup = fixture.model_copy(
        update={"bars": fixture.bars[:9], "fixture": False, "source": "test:raw"}
    )
    dataset = client.post(
        "/api/v1/market/import", json=warmup.model_dump(mode="json")
    ).json()
    strategy = client.post(
        "/api/v1/strategies", json={"name": "Forward rule", "symbols": ["AAPL", "MSFT"]}
    ).json()
    account = client.post(
        "/api/v1/paper-accounts",
        json={
            "name": "Forward arrival test",
            "mode": "forward",
            "feed": "alpaca",
            "config": {"dataset_id": dataset["id"], "strategy_id": strategy["id"]},
        },
    ).json()
    path = "/api/v1/paper-accounts/" + account["id"]
    client.post(path + "/control", json={"action": "start"})

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2024, 1, 9, 15, tzinfo=timezone.utc)

    monkeypatch.setattr(service, "datetime", Clock)
    incoming = fixture.model_copy(update={"bars": fixture.bars[:21]})
    monkeypatch.setattr(market, "fetch_market", lambda _: incoming)
    first = wait(client, client.post(path + "/poll"))
    assert first["last_feed_timestamp"].startswith("2024-01-08")
    assert first["state"]["steps"] == 3 and first["state"]["fills"]
    assert not first["stale"]
    assert client.post(path + "/control", json={"action": "pause"}).status_code == 200
    assert client.post(path + "/control", json={"action": "resume"}).status_code == 200
    second = wait(client, client.post(path + "/poll"))
    assert second["state"] == first["state"]

    def unavailable(_):
        raise ValueError("Test feed disconnected")

    monkeypatch.setattr(market, "fetch_market", unavailable)
    failed = client.post(path + "/poll").json()
    for _ in range(200):
        if client.get("/api/v1/jobs/" + failed["id"]).json()["status"] == "failed":
            break
        time.sleep(0.01)
    assert client.app.state.store.get("paper", account["id"], "demo")["stale"]


def test_agent_resolves_short_citations_to_owned_results(client, monkeypatch):
    import json

    d = client.post("/api/v1/demo").json()
    doc = client.app.state.store.create(
        "document",
        "demo",
        {
            "name": "Risk notes",
            "text": "Risk limits protect cash reserves.",
            "page": 4,
        },
    )

    def reply(messages):
        if messages[-1]["role"] != "tool":
            return {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "analyze_risk",
                            "arguments": {
                                "dataset_id": d["dataset_id"],
                                "weights": {"AAPL": 1},
                            },
                        }
                    }
                ],
            }
        calculated = json.loads(messages[-1]["content"])
        return {
            "role": "assistant",
            "content": "Daily VaR: "
            + str(calculated["result"]["historical_var"])
            + " ["
            + calculated["citation"]
            + "]. Notes [doc:D1:p4].",
        }

    monkeypatch.setattr(client.app.state.services.agent, "request", reply)
    result = wait(
        client,
        client.post("/api/v1/conversations/chat", json={"message": "Risk cash limits"}),
    )
    assert result["mode"] == "explanation"
    assert "[doc:" + doc["id"] + ":p4]" in result["content"]
    assert "[report:" + result["tools"][0]["report_id"] + "]" in result["content"]
    assert "report:R1" not in result["content"] and "doc:D1" not in result["content"]


def test_agent_tool_validation_citations_and_no_hosted_calls(tmp_path, monkeypatch):
    store = Store("sqlite:///" + str(tmp_path / "llm.db"))
    settings = replace(Settings(), runtime=tmp_path)
    requests = []

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def raise_for_status(self):
            pass

        def iter_lines(self):
            yield json.dumps({**self.json(), "done": True})

        def json(self):
            return (
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "analyze_risk",
                                    "arguments": {
                                        "dataset_id": "missing",
                                        "weights": {"AAPL": 1},
                                        "evil": True,
                                    },
                                }
                            }
                        ],
                    }
                }
                if len(requests) == 1
                else {
                    "message": {
                        "role": "assistant",
                        "content": "[doc:invented:p1] Tool returned an error.",
                    }
                }
            )

    def post(url, **kwargs):
        assert url == "http://127.0.0.1:11434/api/chat"
        requests.append(kwargs["json"])
        return Reply()

    monkeypatch.setattr(httpx, "stream", lambda method, url, **kwargs: post(url, **kwargs))
    called = []
    agent = LocalAgent(
        settings,
        store,
        Retrieval(tmp_path, settings.embedding_model),
        lambda *_: called.append(True),
    )
    result = agent.chat("team", Chat(message="Analyze risk"))
    assert not called
    assert result["tools"][0]["error"]
    assert "[doc:invented:p1]" not in result["content"]
    assert all(r["think"] is False for r in requests)


def test_lexical_retrieval_and_incompatible_index(tmp_path):
    r = Retrieval(tmp_path, "model-new")
    docs = [
        {
            "id": "abc",
            "name": "Cash study",
            "page": 4,
            "text": "Cash flow can be seasonal.",
        }
    ]
    result = r.search("user", docs, "cash")
    assert result["sources"][0]["citation"] == "doc:abc:p4"
    directory = r.directory("user")
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text('{"schema_version":0,"model":"model-old"}')
    # The metadata must be checked before attempting optional model imports.
    with pytest.raises((ValueError, ImportError)):
        r.search("user", docs, "cash")
