"""Run with QUANTARA_MODEL_TESTS=1 after installing requirements-models.txt."""

from datetime import datetime, timedelta, timezone
import os

import pytest

from quantara import forecast, market, news, rl
from quantara.schemas import Backtest, Cashflow, TrainRequest
from quantara.retrieval import Retrieval

pytestmark = pytest.mark.skipif(
    os.getenv("QUANTARA_MODEL_TESTS") != "1",
    reason="Optional model runtime not selected",
)


def test_gym_shared_accounting_and_both_rl_algorithms(tmp_path):
    from stable_baselines3.common.env_checker import check_env

    d = market.fixture()
    env = rl.make_environment(
        d, ["AAPL", "MSFT"], Backtest(dataset_id="fixture", strategy_id="hold")
    )
    check_env(env)
    for algorithm in ("PPO", "DDPG"):
        request = TrainRequest(
            dataset_id="fixture",
            symbols=["AAPL", "MSFT"],
            algorithm=algorithm,
            timesteps=128,
        )
        result = rl.train(request, d, tmp_path)
        assert (tmp_path / "models" / (result["checkpoint"] + ".zip")).exists()
        assert result["evaluation"]["policy"]["observations"] > 0
        assert set(result["evaluation"]["comparisons"]) == {
            "buy_hold",
            "sma",
            "momentum",
        }
        assert result["train_end"] < result["test_start"]


def test_forecast_models():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    tx = [
        {
            "external_id": str(i),
            "timestamp": start + timedelta(days=i),
            "amount": 100 if i % 7 == 0 else -10,
            "category": "weekly",
            "currency": "CAD",
        }
        for i in range(140)
    ]
    for method in ("prophet", "lstm"):
        result = forecast.forecast(Cashflow(transactions=tx, method=method, days=7))
        assert len(result["forecast"]) == 7
        assert result["evaluation"]["train_end"] < result["evaluation"]["test_start"]


def test_local_embeddings_and_financial_distilbert(tmp_path):
    r = Retrieval(tmp_path, "sentence-transformers/all-MiniLM-L6-v2")
    docs = [
        {
            "id": "cash",
            "name": "Cash",
            "page": 2,
            "text": "Recurring monthly rent reduces cash reserves.",
        },
        {
            "id": "risk",
            "name": "Risk",
            "page": 1,
            "text": "Portfolio volatility measures return variability.",
        },
    ]
    meta = r.rebuild("team", docs)
    assert meta["dimension"] > 0 and meta["revision"]
    result = r.search("team", docs, "monthly rent and cash balance")
    assert result["sources"][0]["document_id"] == "cash"
    sentiment = news.classify(
        "The company reported strong profit growth.", "distilbert"
    )
    assert sentiment["model"] == news.DISTILBERT_MODEL
    assert -1 <= sentiment["score"] <= 1
