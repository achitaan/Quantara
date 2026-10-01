"""Adversarial checks and an independent ledger oracle for research results."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
import time

import numpy as np
import pandas as pd
import pytest

from quantara import analytics, forecast, market, rl, simulation
from quantara.schemas import (
    Backtest,
    Cashflow,
    CorporateAction,
    Dataset,
    Strategy,
    TrainRequest,
)


def action_history():
    sessions = market.calendar().sessions_in_range("2024-01-02", "2024-01-18")
    closes = [100, 102, 98, 104, 52, 51, 49, 54, 50, 55, 60, 58]
    rows = []
    for i, (session, close) in enumerate(zip(sessions, closes)):
        opening = closes[max(0, i - 1)] / (2 if i == 4 else 1)
        for symbol in ("AAPL", "SPY"):
            rows.append(
                dict(
                    timestamp=market.calendar().session_open(session),
                    symbol=symbol,
                    open=opening,
                    close=close,
                    high=max(opening, close) * 1.02,
                    low=min(opening, close) * 0.98,
                    volume=20,
                )
            )
    ts = market.calendar().session_open(sessions[4])
    return Dataset(
        name="Hand-verifiable actions",
        source="fixture:accuracy",
        coverage="12 sessions",
        fixture=True,
        bars=rows,
        actions=[
            CorporateAction(timestamp=ts, symbol="AAPL", type="dividend", amount=1),
            CorporateAction(timestamp=ts, symbol="AAPL", type="split", amount=2),
        ],
    )


def reconcile_independently(dataset, result, capital, participation):
    """Reconstruct cash and shares from actions/fills without any execution helpers."""
    cash, holdings = capital, {}
    previous = None
    opening_orders = {o["id"]: o for o in result["state"]["orders"]}
    seen = set()
    for row in result["state"]["equity"][1:]:
        ts = pd.Timestamp(row["timestamp"])
        for action in dataset.actions:
            if action.timestamp <= ts and (
                previous is None or action.timestamp > previous
            ):
                if action.type == "split":
                    holdings[action.symbol] = (
                        holdings.get(action.symbol, 0) * action.amount
                    )
                else:
                    cash += holdings.get(action.symbol, 0) * action.amount
        bars = {b.symbol: b for b in dataset.bars if b.timestamp == ts}
        capacity = {s: b.volume * participation for s, b in bars.items()}
        for fill in result["state"]["fills"]:
            if pd.Timestamp(fill["timestamp"]) != ts:
                continue
            assert fill["id"] not in seen
            seen.add(fill["id"])
            assert ts > pd.Timestamp(opening_orders[fill["order_id"]]["submitted_at"])
            symbol, q = fill["symbol"], fill["quantity"]
            capacity[symbol] -= q
            assert capacity[symbol] >= -1e-8
            sign = 1 if fill["side"] == "buy" else -1
            cash -= sign * q * fill["price"] + fill["commission"]
            holdings[symbol] = holdings.get(symbol, 0) + sign * q
            assert cash >= -1e-7 and holdings[symbol] >= -1e-8
        marked = cash + sum(q * bars[s].close for s, q in holdings.items())
        assert row["cash"] == pytest.approx(cash, abs=1e-7)
        assert row["equity"] == pytest.approx(marked, abs=1e-7)
        previous = ts
    assert result["state"]["cash"] == pytest.approx(cash)
    assert result["state"]["holdings"] == pytest.approx(holdings)
    assert result["metrics"]["total_return"] == pytest.approx(marked / capital - 1)


@pytest.mark.parametrize(
    "template", ["buy_hold", "sma", "rsi", "momentum", "rebalance"]
)
@pytest.mark.parametrize("limit", [None, 50])
def test_fills_actions_and_costs_reconcile_with_independent_ledger(template, limit):
    d = action_history()
    s = Strategy(
        name="Oracle",
        type=template,
        symbols=["AAPL"],
        fast=2,
        slow=3,
        rebalance_every=2,
        limit_offset_bps=limit,
    )
    c = Backtest(
        dataset_id="d",
        strategy_id="s",
        capital=1000,
        commission=0.25,
        slippage_bps=10,
        spread_bps=20,
        participation=0.1,
    )
    result = simulation.run(d, s, c)
    reconcile_independently(d, result, c.capital, c.participation)


@pytest.mark.parametrize("evaluation", ["full", "test", "walk_forward"])
def test_replay_ranges_partitions_and_serialized_restarts_match_backtest(evaluation):
    d = market.fixture()
    s = Strategy(name="Same engine", type="sma", symbols=["AAPL", "MSFT"])
    sessions = sorted({b.timestamp for b in d.bars})
    c = Backtest(
        dataset_id="d",
        strategy_id="s",
        evaluation=evaluation,
        start=sessions[25],
        end=sessions[140],
        commission=0.5,
    )
    expected = simulation.run(d, s, c)
    account = dict(
        status="running",
        mode="replay",
        config=c.model_dump(mode="json"),
        state=simulation.initial_state(c.capital),
    )
    for batch in [1, 3, 17, 1000]:
        account = simulation.replay_step(json.loads(json.dumps(account)), d, s, batch)
    assert account["replay_complete"]
    assert account["state"] == expected["state"]
    assert simulation.replay_step(account, d, s, 5)["state"] == account["state"]


@pytest.mark.parametrize(
    "template", ["buy_hold", "sma", "rsi", "momentum", "rebalance"]
)
def test_future_price_changes_cannot_rewrite_earlier_orders(template):
    d = market.fixture()
    boundary = sorted({b.timestamp for b in d.bars})[100]
    altered = d.model_copy(
        update={
            "bars": [
                b.model_copy(
                    update={
                        k: getattr(b, k) * 1.8 for k in ("open", "high", "low", "close")
                    }
                )
                if b.timestamp >= boundary
                else b
                for b in d.bars
            ]
        }
    )
    s = Strategy(name="Causality", type=template, symbols=["AAPL", "MSFT"])
    c = Backtest(dataset_id="d", strategy_id="s")
    a, b = simulation.run(d, s, c), simulation.run(altered, s, c)

    def prefix(result):
        return [
            f
            for f in result["state"]["fills"]
            if pd.Timestamp(f["timestamp"]) < boundary
        ]

    assert prefix(a) == prefix(b)
    assert [
        r for r in a["state"]["equity"] if pd.Timestamp(r["timestamp"]) < boundary
    ] == [r for r in b["state"]["equity"] if pd.Timestamp(r["timestamp"]) < boundary]


def test_walk_forward_fold_returns_include_every_boundary():
    d = market.fixture()
    c = Backtest(dataset_id="d", strategy_id="s", evaluation="walk_forward")
    result = simulation.run(d, Strategy(name="Hold", symbols=["AAPL"]), c)
    linked = (
        np.prod(
            [1 + f["metrics"]["total_return"] for f in result["walk_forward"]["folds"]]
        )
        - 1
    )
    assert linked == pytest.approx(result["metrics"]["total_return"])


def test_numerical_risk_matches_hand_calculated_returns():
    a = np.array([-0.1, 0.1, -0.05, 0.2, -0.02])
    b = np.array([-0.02, 0.03, -0.1, 0.1, 0.04])
    base = market.fixture()
    stamps = sorted({bar.timestamp for bar in base.bars})[:6]
    rows = []
    for symbol, returns in [("AAPL", a), ("MSFT", b), ("SPY", a)]:
        closes = 100 * np.r_[1, np.cumprod(1 + returns)]
        for ts, close in zip(stamps, closes):
            rows.append(
                dict(
                    timestamp=ts,
                    symbol=symbol,
                    open=close,
                    close=close,
                    high=close,
                    low=close,
                    volume=1000,
                )
            )
    d = Dataset(
        name="Known risk", source="fixture", coverage="Known returns", bars=rows
    )
    result = analytics.risk_report(d, {"AAPL": 0.25, "MSFT": 0.75}, confidence=0.8)
    expected = 0.25 * a + 0.75 * b
    cutoff = np.quantile(-expected, 0.8)
    curve = np.r_[1, np.cumprod(1 + expected)]
    assert result["historical_var"] == pytest.approx(cutoff)
    assert result["expected_shortfall"] == pytest.approx(
        (-expected)[-expected >= cutoff].mean()
    )
    assert result["total_return"] == pytest.approx(curve[-1] - 1)
    assert result["volatility"] == pytest.approx(expected.std(ddof=1) * np.sqrt(252))
    assert result["max_drawdown"] == pytest.approx(
        -(curve / np.maximum.accumulate(curve) - 1).min()
    )


def test_optimizers_match_closed_form_uncorrelated_portfolio():
    returns = np.tile([[-0.1, -0.2], [-0.1, 0.2], [0.1, -0.2], [0.1, 0.2]], (10, 1))
    prices = pd.DataFrame(
        100 * np.vstack([np.ones(2), np.cumprod(1 + returns, axis=0)]),
        index=pd.date_range("2024-01-01", periods=41, tz="UTC"),
        columns=["A", "B"],
    )
    minimum = analytics.optimize(prices, "min_volatility")
    parity = analytics.optimize(prices, "risk_parity")
    assert minimum["A"] == pytest.approx(0.8, abs=1e-6)
    assert parity["A"] == pytest.approx(2 / 3, abs=1e-6)


@pytest.mark.parametrize("interval", ["1m", "5m"])
def test_intraday_accounting_costs_match_daily_on_a_flat_market(interval):
    def flat(d):
        days = sorted({b.timestamp.date() for b in d.bars})[:3]
        return d.model_copy(
            update={
                "bars": [
                    b.model_copy(
                        update=dict(
                            open=100, high=100, low=100, close=100, volume=1000000
                        )
                    )
                    for b in d.bars
                    if b.timestamp.date() in days
                ]
            }
        )

    s = Strategy(name="Flat hold", symbols=["AAPL"])
    c = Backtest(
        dataset_id="d",
        strategy_id="s",
        capital=1000,
        commission=2,
        slippage_bps=5,
        spread_bps=10,
    )
    daily = simulation.run(flat(market.fixture()), s, c)
    intraday = simulation.run(flat(market.fixture(interval)), s, c)
    price = 100.1
    expected_shares = 998 / price
    assert daily["state"]["holdings"]["AAPL"] == pytest.approx(expected_shares)
    assert intraday["state"]["holdings"]["AAPL"] == pytest.approx(expected_shares)
    assert daily["metrics"]["total_return"] == pytest.approx(
        intraday["metrics"]["total_return"]
    )


def test_duplicate_actions_and_late_actions_fail_without_mutating_state():
    d = action_history()
    with pytest.raises(ValueError, match="corporate actions"):
        Dataset.model_validate(d.model_dump() | {"actions": d.actions + [d.actions[0]]})
    ts, bars = simulation.groups(d)[0]
    c = Backtest(dataset_id="d", strategy_id="s")
    state = simulation.initial_state(1000)
    simulation.execute(state, bars, ts, c)
    before = deepcopy(state)
    late = CorporateAction(timestamp=ts, symbol="AAPL", type="dividend", amount=1)
    at, next_bars = simulation.groups(d)[1]
    with pytest.raises(ValueError, match="Late corporate action"):
        simulation.execute(state, next_bars, at, c, [late])
    assert state == before


def test_incomplete_benchmark_and_missing_sessions_are_rejected():
    d = market.fixture()
    incomplete = d.model_copy(update={"bars": d.bars[:5] + d.bars[6:]})
    with pytest.raises(ValueError, match="missing or unaligned"):
        simulation.run(
            incomplete,
            Strategy(name="Hold", symbols=["AAPL"]),
            Backtest(dataset_id="d", strategy_id="s"),
        )
    missing = d.model_copy(update={"bars": d.bars[:3] + d.bars[6:]})
    with pytest.raises(ValueError, match="Missing"):
        simulation.run(
            missing,
            Strategy(name="Hold", symbols=["AAPL"]),
            Backtest(dataset_id="d", strategy_id="s"),
        )


def test_sentiment_uses_latest_available_article_despite_import_order():
    frame = market.aligned_prices(market.fixture(), ["AAPL"]).iloc[:5]
    news = [
        dict(symbols=["AAPL"], available_at=frame.index[-1].isoformat(), score=-0.8),
        dict(symbols=["AAPL"], available_at=frame.index[-2].isoformat(), score=0.9),
        dict(
            symbols=["AAPL"],
            available_at=(frame.index[-1] + timedelta(days=1)).isoformat(),
            score=1,
        ),
    ]
    weights = simulation.targets(
        frame,
        Strategy(name="News", type="sentiment", symbols=["AAPL"]),
        simulation.initial_state(1000),
        news,
    )
    assert weights == {"AAPL": 0}


def test_rl_observation_uses_raw_marks_and_rewards_reconcile_after_actions():
    pytest.importorskip("gymnasium")
    d = action_history()
    c = Backtest(
        dataset_id="d",
        strategy_id="s",
        capital=1000,
        commission=0.5,
        participation=1,
        slippage_bps=0,
    )
    env = rl.make_environment(d, ["AAPL"], c)
    env.reset(seed=42)
    rewards, ended = [], False
    while not ended:
        obs, reward, ended, _, _ = env.step(np.array([0.8]))
        rewards.append(reward)
        value = env.state["equity"][-1]["equity"]
        assert obs[-1] == pytest.approx(env.state["cash"] / value, abs=1e-6)
        assert obs[-2] == pytest.approx(
            env.state["holdings"].get("AAPL", 0) * env.state["marks"]["AAPL"] / value,
            abs=1e-6,
        )
        assert obs[-1] + obs[-2] == pytest.approx(1, abs=1e-6)
    assert np.exp(sum(rewards)) == pytest.approx(
        env.state["equity"][-1]["equity"] / c.capital
    )
    with pytest.raises(ValueError, match="reset"):
        env.step(np.array([0.8]))


@pytest.mark.parametrize("action", [[np.nan], [np.inf], [1, 2], [[0.5]]])
def test_invalid_policy_actions_rejected(action):
    with pytest.raises(ValueError, match="finite value"):
        rl.allocation(action, ["AAPL"])


def test_obsolete_policies_and_untrained_ddpg_requests_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="retrain"):
        rl.trained_policy({"observation_version": 1}, tmp_path)
    with pytest.raises(ValueError, match="missing"):
        rl.trained_policy({"observation_version": 2, "checkpoint": "absent"}, tmp_path)
    with pytest.raises(ValueError, match="warmup"):
        TrainRequest(dataset_id="d", symbols=["AAPL"], algorithm="DDPG", timesteps=100)
    with pytest.raises(ValueError, match="unique"):
        TrainRequest(dataset_id="d", symbols=["AAPL", "AAPL"])


def cash_history():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [
        dict(
            external_id=str(i),
            timestamp=start + timedelta(days=i),
            amount=100 if i % 7 == 0 else -10,
            category="weekly",
            currency="CAD",
        )
        for i in range(140)
    ]


@pytest.mark.parametrize("method", ["baseline", "prophet", "lstm"])
def test_forecast_holdout_prices_do_not_leak_into_training(monkeypatch, method):
    if method == "lstm" and os.getenv("QUANTARA_MODEL_TESTS") != "1":
        pytest.skip("Neural runtime not selected")
    if method == "prophet":
        pytest.importorskip("prophet")
    captured = []
    original = forecast.predict

    def observe(train, days, selected):
        predicted, meta = original(train, days, selected)
        captured.append(predicted.to_numpy())
        return predicted, meta

    monkeypatch.setattr(forecast, "predict", observe)
    tx = cash_history()
    forecast.forecast(Cashflow(transactions=tx, method=method, days=7))
    altered = [r | {"amount": 10000} if i >= 112 else r for i, r in enumerate(tx)]
    forecast.forecast(Cashflow(transactions=altered, method=method, days=7))
    np.testing.assert_allclose(captured[0], captured[2], rtol=0, atol=1e-8)


@pytest.mark.skipif(
    os.getenv("QUANTARA_MODEL_TESTS") != "1", reason="Neural runtime not selected"
)
@pytest.mark.parametrize("algorithm", ["PPO", "DDPG"])
def test_real_training_reproducibility_future_isolation_and_checkpoint_integrity(
    tmp_path, algorithm
):
    d = market.fixture()
    sessions = sorted({b.timestamp for b in d.bars})
    train_boundary = sessions[int(len(sessions) * 0.6)]
    altered = d.model_copy(
        update={
            "bars": [
                b.model_copy(
                    update={
                        k: getattr(b, k) * 1.5 for k in ("open", "high", "low", "close")
                    }
                )
                if b.timestamp >= train_boundary
                else b
                for b in d.bars
            ]
        }
    )
    request = TrainRequest(
        dataset_id="d",
        symbols=["AAPL", "MSFT"],
        algorithm=algorithm,
        timesteps=129,
        seed=71,
        commission=0.5,
        spread_bps=10,
    )
    a, b = rl.train(request, d, tmp_path), rl.train(request, altered, tmp_path)
    assert a["parameter_sha256"] == b["parameter_sha256"]
    assert a["training_data_version"] == b["training_data_version"]
    assert a["training_updates"] > 0
    assert a["timesteps"] >= request.timesteps
    assert a["requested_timesteps"] == request.timesteps
    assert a["costs"]["commission"] == 0.5 and a["costs"]["spread_bps"] == 10
    assert a["train_end"] < a["validation_end"] < a["test_start"]
    assert a["training_history"] == b["training_history"]
    assert a["training_history"][-1]["step"] == a["timesteps"]
    assert sum(p["window_steps"] for p in a["training_history"]) == a["timesteps"]
    for name, curve in a["evaluation"]["curves"].items():
        assert 2 <= len(curve) <= 1000
        assert curve[0]["equity"] == request.capital
        assert curve[1]["timestamp"][:10] >= a["test_start"]
        assert curve[-1]["timestamp"] == a["evaluation"]["period"]["end"]
    assert a["evaluation"]["curves"]["policy"][-1]["equity"] == pytest.approx(request.capital * (1 + a["evaluation"]["policy"]["total_return"]))
    path = tmp_path / "models" / (a["checkpoint"] + ".zip")
    path.write_bytes(path.read_bytes() + b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        rl.trained_policy(a, tmp_path)


@pytest.mark.skipif(
    os.getenv("QUANTARA_MODEL_TESTS") != "1", reason="Neural runtime not selected"
)
def test_training_job_and_policy_backtest_evaluation_scope(tmp_path):
    from dataclasses import replace
    from fastapi.testclient import TestClient
    from quantara.api import create_app
    from quantara.config import Settings

    app = create_app(
        replace(
            Settings(),
            runtime=tmp_path,
            database_url="sqlite:///" + str(tmp_path / "audit.db"),
        )
    )
    with TestClient(app) as client:
        client.post(
            "/api/v1/auth/login",
            json={"username": "demo", "password": "quantara-local-demo"},
        ).raise_for_status()
        demo = client.post("/api/v1/demo").json()

        def wait(path, arguments):
            submitted = client.post("/api/v1/" + path, json=arguments)
            submitted.raise_for_status()
            for _ in range(1000):
                job = client.get("/api/v1/jobs/" + submitted.json()["id"]).json()
                if job["status"] in ("complete", "failed"):
                    assert job["status"] == "complete", job
                    return job["result"]
                time.sleep(0.02)
            pytest.fail("Training job timed out")

        model = wait(
            "models/train",
            dict(
                dataset_id=demo["dataset_id"],
                symbols=["AAPL", "MSFT"],
                timesteps=129,
                capital=5000,
                commission=0.5,
                spread_bps=10,
            ),
        )
        strategy = client.post(
            "/api/v1/strategies",
            json=dict(
                name="Audited policy",
                type="rl",
                symbols=model["symbols"],
                model_id=model["id"],
            ),
        ).json()
        config = dict(
            dataset_id=demo["dataset_id"],
            strategy_id=strategy["id"],
            capital=5000,
            commission=0.5,
            spread_bps=10,
        )
        full = wait("backtests", config)
        heldout = wait("backtests", config | {"evaluation": "test"})
        changed_costs = wait(
            "backtests", config | {"evaluation": "test", "commission": 1}
        )
        assert not full["model_evaluation"]["out_of_sample_dates"]
        assert heldout["model_evaluation"]["out_of_sample_dates"]
        assert heldout["model_evaluation"]["cost_settings_changed"] == []
        assert changed_costs["model_evaluation"]["cost_settings_changed"] == [
            "commission"
        ]
        assert heldout["metrics"] == model["evaluation"]["policy"]


@pytest.mark.skipif(
    os.getenv("QUANTARA_MODEL_TESTS") != "1", reason="Neural runtime not selected"
)
def test_concurrent_neural_jobs_preserve_seeded_results(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    data = market.fixture()
    requests = [
        TrainRequest(
            dataset_id="d",
            symbols=["AAPL", "MSFT"],
            algorithm=algorithm,
            timesteps=129,
            seed=seed,
        )
        for algorithm, seed in [("PPO", 61), ("DDPG", 62)]
    ]
    sequential = [rl.train(request, data, tmp_path) for request in requests]
    cash = Cashflow(transactions=cash_history(), method="lstm", days=7)
    expected_forecast = forecast.forecast(cash)
    with ThreadPoolExecutor(max_workers=3) as pool:
        parallel = [
            pool.submit(rl.train, request, data, tmp_path) for request in requests
        ]
        parallel_forecast = pool.submit(forecast.forecast, cash)
        actual = [future.result() for future in parallel]
        assert parallel_forecast.result() == expected_forecast
    assert [m["parameter_sha256"] for m in sequential] == [
        m["parameter_sha256"] for m in actual
    ]
