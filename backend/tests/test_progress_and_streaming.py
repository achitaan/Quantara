from dataclasses import replace
from copy import deepcopy
import json

import httpx
import pytest

from quantara.config import Settings
from quantara.db import Store
from quantara.jobs import Cancelled
from quantara.jobs import Jobs
from quantara.llm import LocalAgent, stream_observer
from quantara.retrieval import Retrieval
from quantara.schemas import Backtest, Chat, Strategy
from quantara import market, simulation


class Progress:
    def __init__(self):
        self.updates, self.snapshots = [], []

    def __call__(self, fraction, message):
        self.updates.append((fraction, message))

    def snapshot(self, details):
        self.snapshots.append(deepcopy(details))


def mocked_stream(monkeypatch, chunks, requests):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def raise_for_status(self):
            pass

        def iter_lines(self):
            for chunk in chunks:
                yield json.dumps(chunk)

    def stream(method, url, **kwargs):
        assert method == "POST" and url == "http://127.0.0.1:11434/api/chat"
        assert kwargs["trust_env"] is False
        requests.append(kwargs["json"])
        return Response()

    monkeypatch.setattr(httpx, "stream", stream)


def agent(tmp_path):
    store = Store("sqlite://")
    return LocalAgent(replace(Settings(), runtime=tmp_path), store, Retrieval(tmp_path, "test"), lambda *_: {})


def test_greeting_streams_without_research_context_or_tool_schema(tmp_path, monkeypatch):
    instance = agent(tmp_path)
    instance.store.create("document", "team", {"name": "private source", "text": "sensitive-marker " * 100, "page": 4})
    requests, progress = [], Progress()
    mocked_stream(monkeypatch, [{"message": {"content": "Hello!"}, "done": False}, {"message": {"content": " Ask about your simulation."}, "done": True, "load_duration": 1000000, "prompt_eval_count": 59, "eval_count": 12}], requests)
    result = instance.chat("team", Chat(message="Hi"), progress)
    assert result["content"] == "Hello! Ask about your simulation."
    assert not requests[0]["tools"]
    assert "sensitive-marker" not in json.dumps(requests)
    assert requests[0]["stream"] and requests[0]["think"] is False
    assert requests[0]["keep_alive"] == "30m"
    assert any(p.get("draft") == "Hello!" for p in progress.snapshots)
    assert progress.snapshots[-1]["phase"] == "validating" and not progress.snapshots[-1]["draft"]
    assert result["timing"]["requests"][0]["load_seconds"] == .001
    assert stream_observer.get() is None


def test_streamed_document_citation_is_resolved_and_unknown_citation_removed(tmp_path, monkeypatch):
    instance = agent(tmp_path)
    doc = instance.store.create("document", "team", {"name": "Rent note", "text": "Monthly rent is CAD 100.", "page": 4})
    requests, progress = [], Progress()
    mocked_stream(monkeypatch, [{"message": {"content": "Rent is CAD 100 [doc:D1:p4]. [doc:invented:p8]"}, "done": True}], requests)
    result = instance.chat("team", Chat(message="According to the document, what is monthly rent?"), progress)
    assert not requests[0]["tools"]
    assert result["retrieval_mode"] == "lexical"
    assert "[doc:" + doc["id"] + ":p4]" in result["content"]
    assert "[doc:invented:p8]" not in result["content"]
    assert result["warning"] and result["sources"][0]["page"] == 4


def test_incomplete_stream_and_cancellation_never_save_preview_as_final(tmp_path, monkeypatch):
    instance = agent(tmp_path)
    mocked_stream(monkeypatch, [{"message": {"content": "Unfinished"}, "done": False}], [])
    result = instance.chat("team", Chat(message="Hi"), Progress())
    assert result["mode"] == "limited" and "stream ended" in result["warning"]
    assert "Unfinished" not in result["content"]
    assert stream_observer.get() is None

    class Stop(Progress):
        def snapshot(self, details):
            if details.get("draft"):
                raise Cancelled()

    mocked_stream(monkeypatch, [{"message": {"content": "Cancel me"}, "done": False}], [])
    before = len(instance.store.list("conversation", "team"))
    with pytest.raises(Cancelled):
        instance.chat("team", Chat(message="Hi"), Stop())
    assert stream_observer.get() is None
    assert len(instance.store.list("conversation", "team")) == before + 1
    assert not instance.store.list("conversation", "team")[-1]["messages"]


def test_live_replay_snapshots_are_actual_prefixes_not_future_valuations():
    data, progress = market.fixture(), Progress()
    result = simulation.run(data, Strategy(name="hold", type="buy_hold", symbols=["AAPL", "MSFT"]), Backtest(dataset_id="fixture", strategy_id="hold"), progress=progress)
    saved = {p["timestamp"]: p["equity"] for p in result["state"]["equity"]}
    assert len(progress.snapshots) > 2
    for snapshot in progress.snapshots:
        assert len(snapshot["equity"]) <= 200
        assert snapshot["bars_executed"] <= snapshot["total_bars"]
        for point in snapshot["equity"]:
            assert point["equity"] == saved[point["timestamp"]]
        assert snapshot["equity"][-1] == result["state"]["equity"][snapshot["bars_executed"]]


def test_chat_is_not_queued_behind_two_research_workers(tmp_path):
    from threading import Barrier, Event
    from time import monotonic, sleep

    store, started, release = Store("sqlite:///" + str(tmp_path / "workers.db")), Barrier(3), Event()

    def runner(operation, arguments, owner, progress):
        if operation != "chat":
            started.wait(timeout=5)
            release.wait(timeout=5)
        progress.snapshot({"draft": "Actual worker snapshot"})
        return {"ok": True}

    jobs = Jobs(store, runner)
    try:
        jobs.submit("train", {}, "team")
        jobs.submit("backtest", {}, "team")
        started.wait(timeout=5)
        chat = jobs.submit("chat", {}, "team")
        deadline = monotonic() + 3
        while monotonic() < deadline and store.get("job", chat["id"], "team")["status"] != "complete":
            sleep(.01)
        saved = store.get("job", chat["id"], "team")
        assert saved["status"] == "complete"
        assert saved["details"]["draft"] == "Actual worker snapshot"
        assert not release.is_set()
    finally:
        release.set()
        jobs.close()
