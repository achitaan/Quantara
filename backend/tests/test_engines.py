from datetime import datetime, timezone

import numpy as np
import pytest

from quantara import accounting, analytics, forecast, market, simulation, tax
from quantara.schemas import (
    Backtest,
    Cashflow,
    CorporateAction,
    Dataset,
    Portfolio,
    Strategy,
)


@pytest.fixture(scope="module")
def dataset():
    return market.fixture()


def test_all_optimizers(dataset):
    frame = market.aligned_prices(dataset, ["AAPL", "MSFT"])
    for method in (
        "equal_weight",
        "mean_variance",
        "min_volatility",
        "risk_parity",
        "black_litterman",
        "cvar",
    ):
        weights = analytics.optimize(
            frame, method, views={"AAPL": 0.08} if method == "black_litterman" else None
        )
        assert sum(weights.values()) == pytest.approx(1)
        assert all(0 <= w <= 1 for w in weights.values())
    with pytest.raises(ValueError, match="Infeasible"):
        analytics.optimize(frame, max_weight=0.4)


def test_ledger_reconciliation_and_import():
    p = Portfolio.model_validate(
        {
            "name": "Ledger",
            "transactions": [
                {
                    "external_id": "1",
                    "type": "deposit",
                    "timestamp": "2024-01-02T14:30:00Z",
                    "amount": 1000,
                },
                {
                    "external_id": "2",
                    "type": "buy",
                    "timestamp": "2024-01-03T14:30:00Z",
                    "symbol": "AAPL",
                    "quantity": 5,
                    "price": 100,
                    "fee": 2,
                },
                {
                    "external_id": "3",
                    "type": "split",
                    "timestamp": "2024-01-04T14:30:00Z",
                    "symbol": "AAPL",
                    "quantity": 2,
                },
                {
                    "external_id": "4",
                    "type": "dividend",
                    "timestamp": "2024-01-05T14:30:00Z",
                    "symbol": "AAPL",
                    "amount": 10,
                },
                {
                    "external_id": "5",
                    "type": "sell",
                    "timestamp": "2024-01-08T14:30:00Z",
                    "symbol": "AAPL",
                    "quantity": 2,
                    "price": 60,
                    "fee": 1,
                },
            ],
        }
    )
    result = accounting.ledger(p)
    assert result["cash"] == 627
    assert result["holdings"] == {"AAPL": 8}
    same, added = accounting.import_transactions(
        p.model_dump(mode="json"), [p.transactions[1].model_dump(mode="json")]
    )
    assert added == 0 and len(same["transactions"]) == 5
    with pytest.raises(ValueError, match="different"):
        accounting.import_transactions(
            same,
            [
                p.transactions[1]
                .model_copy(update={"price": 101})
                .model_dump(mode="json")
            ],
        )


def test_no_lookahead_costs_and_restart(dataset):
    strategy = Strategy(name="Hold", symbols=["AAPL"])
    config = Backtest(
        dataset_id="fixture",
        strategy_id="hold",
        capital=100000,
        commission=1,
        participation=0.00001,
    )
    result = simulation.run(dataset, strategy, config)
    fills = result["state"]["fills"]
    assert fills[0]["timestamp"] > result["state"]["orders"][0]["submitted_at"]
    assert fills[0]["quantity"] <= 10
    assert len(fills) > 1
    assert all(r["cash"] >= 0 for r in result["state"]["equity"])
    prefix = dataset.model_copy(update={"bars": dataset.bars[:90]})
    shorter = simulation.run(prefix, strategy, config)
    assert fills[: len(shorter["state"]["fills"])] == shorter["state"]["fills"]
    account = {
        "config": config.model_dump(mode="json"),
        "status": "running",
        "state": simulation.initial_state(100000),
    }
    stepped = simulation.replay_step(account, dataset, strategy, 20)
    restarted = simulation.replay_step(stepped, dataset, strategy, 20)
    assert restarted["state"]["fills"] == fills[: len(restarted["state"]["fills"])]
    state = restarted["state"]
    ts, bars = simulation.groups(dataset)[39]
    before = len(state["fills"])
    assert not simulation.execute(state, bars, ts, config)
    assert len(state["fills"]) == before


def test_actions_and_conservative_limit(dataset):
    bars = [
        b.model_copy(
            update={"open": 100, "high": 101, "low": 99, "close": 100, "volume": 100000}
        )
        for b in dataset.bars[:12]
        if b.symbol == "AAPL"
    ]
    ts = bars[2].timestamp
    bars[2] = bars[2].model_copy(
        update={"open": 50, "high": 51, "low": 49, "close": 50}
    )
    bars[3] = bars[3].model_copy(
        update={"open": 50, "high": 51, "low": 49, "close": 50}
    )
    d = Dataset(
        name="Actions",
        source="fixture",
        coverage="test",
        bars=bars,
        actions=[
            CorporateAction(timestamp=ts, symbol="AAPL", type="split", amount=2),
            CorporateAction(timestamp=ts, symbol="AAPL", type="dividend", amount=1),
        ],
    )
    state = simulation.initial_state(1000)
    s = Strategy(name="Hold", symbols=["AAPL"])
    c = Backtest(dataset_id="x", strategy_id="s", slippage_bps=0, benchmark="AAPL")
    frame = market.aligned_prices(d)
    for at, group in simulation.groups(d):
        simulation.advance(state, at, group, frame.loc[:at], s, c, d.actions)
    assert state["holdings"]["AAPL"] == pytest.approx(20)
    assert state["cash"] == pytest.approx(20)
    state = simulation.initial_state(1000)
    s = s.model_copy(update={"limit_offset_bps": 100})
    for at, group in simulation.groups(d)[:2]:
        simulation.advance(state, at, group, frame.loc[:at], s, c)
    assert (
        state["fills"] == []
    )  # Touch at 99 does not conservatively fill a buy limit at 99.


def test_tax_cad_pool_fx_and_partial_superficial():
    p = {
        "name": "Tax",
        "transactions": [
            {
                "external_id": "a",
                "timestamp": "2024-01-02T14:30:00Z",
                "type": "buy",
                "symbol": "AAPL",
                "quantity": 100,
                "price": 10,
                "fx_cad": 1.2,
            },
            {
                "external_id": "b",
                "timestamp": "2024-04-02T14:30:00Z",
                "type": "sell",
                "symbol": "AAPL",
                "quantity": 100,
                "price": 8,
                "fx_cad": 1.3,
            },
            {
                "external_id": "c",
                "timestamp": "2024-04-10T14:30:00Z",
                "type": "buy",
                "symbol": "AAPL",
                "quantity": 25,
                "price": 8,
                "fx_cad": 1.3,
            },
        ],
    }
    result = tax.research(p, datetime(2024, 6, 1, tzinfo=timezone.utc), True)
    d = result["dispositions"][0]
    assert d["gain_cad"] == -160
    assert d["superficial_loss"]["denied_loss_cad"] == 40
    assert d["allowable_loss_cad"] == 120
    assert result["pools"]["AAPL"]["acb_cad"] == pytest.approx(300)
    pending = tax.research(p, datetime(2024, 4, 20, tzinfo=timezone.utc), True)
    assert pending["provisional"]


def test_risk_and_forecast(dataset):
    risk = analytics.risk_report(dataset, {"AAPL": 0.5, "MSFT": 0.5})
    assert risk["expected_shortfall"] >= risk["historical_var"] >= 0
    assert risk["stress_loss"] == 0.2
    rows = [
        {
            "external_id": str(i),
            "timestamp": f"2024-{1 + i // 28:02d}-{1 + i % 28:02d}T00:00:00Z",
            "amount": 2000 if i % 28 == 0 else -20,
            "category": "salary" if i % 28 == 0 else "daily",
        }
        for i in range(112)
    ]
    result = forecast.forecast(Cashflow(transactions=rows))
    assert len(result["forecast"]) == 30
    assert result["evaluation"]["train_end"] < result["evaluation"]["test_start"]
    assert np.isfinite(result["evaluation"]["mae"])
    assert all(r["lower"] <= r["upper"] for r in result["forecast"])
