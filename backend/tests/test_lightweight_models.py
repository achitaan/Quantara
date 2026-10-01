"""Exercise models that do not need Torch or downloaded neural weights."""

from datetime import datetime, timedelta, timezone
from importlib.util import find_spec

import numpy as np
import pytest

from quantara import forecast, market, rl
from quantara.schemas import Backtest, Cashflow


@pytest.mark.skipif(find_spec("gymnasium") is None, reason="Gymnasium is optional")
def test_gym_environment_accounting_and_seed():
    from gymnasium.utils.env_checker import check_env

    data = market.fixture()
    env = rl.make_environment(
        data,
        ["AAPL", "MSFT"],
        Backtest(
            dataset_id="fixture",
            strategy_id="hold",
            commission=1,
            slippage_bps=5,
        ),
    )
    check_env(env, skip_render_check=True)
    first, _ = env.reset(seed=42)
    second, _ = env.reset(seed=42)
    np.testing.assert_array_equal(first, second)
    observed, reward, ended, _, _ = env.step(np.array([0.5, 0.5], dtype=np.float32))
    assert env.observation_space.contains(observed)
    assert np.isfinite(reward) and not ended
    assert env.state["cash"] >= 0 and env.state["fills"]
    prices = env.frame.iloc[env.cursor]
    equity = env.state["cash"] + sum(
        q * prices[s] for s, q in env.state["holdings"].items()
    )
    assert env.state["equity"][-1]["equity"] == pytest.approx(equity)
    assert sum(f["commission"] for f in env.state["fills"]) > 0


@pytest.mark.skipif(find_spec("prophet") is None, reason="Prophet is optional")
def test_prophet_chronological_forecast():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    transactions = [
        {
            "external_id": str(i),
            "timestamp": start + timedelta(days=i),
            "amount": 100 if i % 7 == 0 else -10,
            "category": "weekly",
            "currency": "CAD",
        }
        for i in range(140)
    ]
    result = forecast.forecast(
        Cashflow(transactions=transactions, method="prophet", days=7)
    )
    assert result["model"]["engine"] == "Prophet"
    assert result["evaluation"]["train_end"] < result["evaluation"]["test_start"]
    assert np.isfinite(result["evaluation"]["mae"])
    assert len(result["forecast"]) == 7
    assert all(
        np.isfinite(r["balance"]) and r["lower"] <= r["upper"]
        for r in result["forecast"]
    )
