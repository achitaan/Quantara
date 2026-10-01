"""One deterministic, serializable execution engine for replay, paper and RL."""

from copy import deepcopy
from hashlib import sha256
import math

import numpy as np
import pandas as pd

from .analytics import optimize, performance
from .market import aligned_prices, return_frame, version


def initial_state(capital):
    return {
        "cash": capital,
        "holdings": {},
        "orders": [],
        "fills": [],
        "events": [],
        "equity": [],
        "last_timestamp": None,
        "steps": 0,
        "actions": [],
        "target": {},
    }


def groups(dataset):
    grouped = {}
    for bar in dataset.bars:
        grouped.setdefault(bar.timestamp, {})[bar.symbol] = bar
    return sorted(grouped.items())


def equity(state, bars):
    return state["cash"] + sum(q * bars[s].close for s, q in state["holdings"].items())


def submit_targets(state, bars, weights, strategy, timestamp):
    if (
        any(w < 0 or not math.isfinite(w) for w in weights.values())
        or sum(weights.values()) > 1 + 1e-8
    ):
        raise ValueError("Targets must be finite, nonnegative and sum to at most one")
    # Replace unfilled rebalancing orders, preserving their audit trail.
    for order in state["orders"]:
        if order["status"] in ("pending", "partial"):
            order["status"] = "cancelled"
    value = equity(state, bars)
    for symbol in strategy.symbols:
        change = weights.get(symbol, 0) * value / bars[symbol].close - state[
            "holdings"
        ].get(symbol, 0)
        if abs(change) * bars[symbol].close < 1:
            continue
        side = "buy" if change > 0 else "sell"
        limit = None
        if strategy.limit_offset_bps is not None:
            limit = bars[symbol].close * (
                1 + (1 if side == "sell" else -1) * strategy.limit_offset_bps / 10000
            )
        key = f"{timestamp.isoformat()}:{symbol}:{side}"
        state["orders"].append(
            {
                "id": sha256(key.encode()).hexdigest()[:24],
                "symbol": symbol,
                "side": side,
                "quantity": abs(change),
                "remaining": abs(change),
                "limit": limit,
                "submitted_at": timestamp.isoformat(),
                "status": "pending",
            }
        )
    state["target"] = weights


def execute(state, bars, timestamp, config, actions=()):
    stamp = timestamp.isoformat()
    if state["last_timestamp"] and timestamp <= pd.Timestamp(state["last_timestamp"]):
        return False
    # Actions are applied exactly once before this bar's orders. Stable keys survive restarts.
    for action in sorted(actions, key=lambda a: (a.timestamp, a.type != "split")):
        key = f"{action.timestamp.isoformat()}:{action.symbol}:{action.type}:{action.amount}"
        if action.timestamp > timestamp or key in state["actions"]:
            continue
        symbol = action.symbol
        if action.type == "split":
            state["holdings"][symbol] = state["holdings"].get(symbol, 0) * action.amount
            for order in state["orders"]:
                if order["symbol"] == symbol and order["status"] in (
                    "pending",
                    "partial",
                ):
                    order["remaining"] *= action.amount
                    order["quantity"] *= action.amount
                    if order["limit"]:
                        order["limit"] /= action.amount
        else:
            state["cash"] += state["holdings"].get(symbol, 0) * action.amount
        state["actions"].append(key)
        state["events"].append(
            {
                "timestamp": stamp,
                "type": action.type,
                "symbol": symbol,
                "amount": action.amount,
            }
        )
    available = {s: b.volume * config.participation for s, b in bars.items()}
    # Sell first so a rebalance may fund purchases without borrowing.
    for order in sorted(state["orders"], key=lambda o: o["side"] == "buy"):
        if (
            order["status"] not in ("pending", "partial")
            or pd.Timestamp(order["submitted_at"]) >= timestamp
        ):
            continue
        symbol, side = order["symbol"], order["side"]
        b = bars[symbol]
        sign = 1 if side == "buy" else -1
        cost = (config.slippage_bps + config.spread_bps / 2) / 10000
        price = b.open * (1 + sign * cost)
        limit = order["limit"]
        if limit is not None:
            # A touch is insufficient. A bar must trade through the limit after costs.
            if (side == "buy" and b.low * (1 + cost) >= limit) or (
                side == "sell" and b.high * (1 - cost) <= limit
            ):
                continue
            price = min(price, limit) if side == "buy" else max(price, limit)
        quantity = min(order["remaining"], available[symbol])
        if side == "buy":
            quantity = min(
                quantity, max(0, (state["cash"] - config.commission) / price)
            )
        else:
            quantity = min(quantity, state["holdings"].get(symbol, 0))
            if quantity * price <= config.commission:
                quantity = 0
        if quantity <= 1e-10:
            continue
        state["cash"] -= sign * quantity * price + config.commission
        state["cash"] = (
            max(0, state["cash"]) if state["cash"] > -1e-7 else state["cash"]
        )
        state["holdings"][symbol] = state["holdings"].get(symbol, 0) + sign * quantity
        available[symbol] -= quantity
        order["remaining"] -= quantity
        order["status"] = "filled" if order["remaining"] < 1e-8 else "partial"
        state["fills"].append(
            {
                "id": f"{order['id']}:{stamp}",
                "order_id": order["id"],
                "timestamp": stamp,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "commission": config.commission,
                "slippage_cost": abs(price - b.open) * quantity,
            }
        )
    if state["cash"] < -1e-6 or any(q < -1e-6 for q in state["holdings"].values()):
        raise ValueError("Execution violated cash or long-only constraints")
    state["last_timestamp"], state["steps"] = stamp, state["steps"] + 1
    state["equity"].append(
        {"timestamp": stamp, "equity": equity(state, bars), "cash": state["cash"]}
    )
    return True


def targets(history, strategy, state, news=(), policy=None):
    n = len(strategy.symbols)
    if strategy.type == "rl":
        if policy is None:
            raise ValueError("A trained local policy is required")
        return policy(history, state)
    if strategy.type == "buy_hold":
        return {s: 1 / n for s in strategy.symbols} if state["steps"] == 1 else None
    if strategy.type == "rebalance":
        if (state["steps"] - 1) % strategy.rebalance_every:
            return None
        if (
            strategy.method != "equal_weight"
            and len(history.resample("1D").last().dropna()) < 3
        ):
            return None
        return optimize(history, method=strategy.method)
    active = []
    for symbol in strategy.symbols:
        series = history[symbol]
        if strategy.type == "sentiment":
            latest = [
                x
                for x in news
                if symbol in x["symbols"]
                and pd.Timestamp(x["available_at"]) <= history.index[-1]
            ]
            if latest and latest[-1]["score"] >= strategy.sentiment_threshold:
                active.append(symbol)
            continue
        if len(series) < strategy.slow:
            continue
        if strategy.type == "sma":
            include = (
                series.iloc[-strategy.fast :].mean()
                > series.iloc[-strategy.slow :].mean()
            )
        elif strategy.type == "momentum":
            include = series.iloc[-1] > series.iloc[-strategy.slow]
        elif strategy.type == "rsi":
            delta = series.diff().iloc[-strategy.fast :]
            up, down = delta.clip(lower=0).mean(), -delta.clip(upper=0).mean()
            rsi = (
                50
                if up == down == 0
                else 100
                if down == 0
                else 100 - 100 / (1 + up / down)
            )
            include = rsi <= strategy.rsi_buy or (
                state["holdings"].get(symbol, 0) > 0 and rsi < strategy.rsi_sell
            )
        else:
            raise ValueError("Unsupported strategy")
        if include:
            active.append(symbol)
    weights = {s: 1 / n if s in active else 0 for s in strategy.symbols}
    return None if weights == state["target"] else weights


def advance(
    state, timestamp, bars, history, strategy, config, actions=(), news=(), policy=None
):
    if not execute(state, bars, timestamp, config, actions):
        return state
    weights = targets(
        history.loc[:timestamp, strategy.symbols], strategy, state, news, policy
    )
    if weights is not None:
        submit_targets(state, bars, weights, strategy, timestamp)
    return state


def run(dataset, strategy, config, progress=lambda *_: None, news=(), policy=None):
    prices = aligned_prices(dataset, strategy.symbols)
    if config.benchmark not in return_frame(dataset):
        raise ValueError("Benchmark history is missing from the dataset")
    grouped = [
        (ts, bars)
        for ts, bars in groups(dataset)
        if (config.start is None or ts >= config.start)
        and (config.end is None or ts <= config.end)
    ]
    if len(grouped) < 3:
        raise ValueError("At least three aligned execution bars are required")
    dates = sorted({ts.date() for ts, _ in grouped})
    if len(dates) < 3:
        raise ValueError(
            "At least three sessions are required for chronological partitions"
        )
    train_end = max(1, int(len(dates) * config.train_fraction))
    val_end = max(
        train_end + 1,
        int(len(dates) * (config.train_fraction + config.validation_fraction)),
    )
    val_end = min(val_end, len(dates) - 1)
    test_start = dates[val_end]
    if config.evaluation in ("test", "walk_forward"):
        grouped = [(ts, bars) for ts, bars in grouped if ts.date() >= test_start]
    state = initial_state(config.capital)
    # A prior instant preserves the initial capital and execution costs in total return.
    baseline_ts = pd.Timestamp(grouped[0][0]) - pd.Timedelta(days=1)
    state["equity"].append(
        {
            "timestamp": baseline_ts.isoformat(),
            "equity": config.capital,
            "cash": config.capital,
        }
    )
    for i, (ts, bars) in enumerate(grouped):
        advance(
            state,
            ts,
            bars,
            prices.loc[:ts],
            strategy,
            config,
            dataset.actions,
            news,
            policy,
        )
        if i % max(1, len(grouped) // 100) == 0:
            progress((i + 1) / len(grouped), f"Executed {i + 1}/{len(grouped)} bars")
    curve = pd.Series(
        [v["equity"] for v in state["equity"]],
        index=pd.to_datetime([v["timestamp"] for v in state["equity"]], utc=True),
    )
    benchmark = return_frame(dataset)[config.benchmark].reindex(
        [ts for ts, _ in grouped]
    )
    bench = config.capital * benchmark / benchmark.iloc[0]
    bench.loc[baseline_ts] = config.capital
    bench = bench.sort_index()
    result = {
        "state": state,
        "metrics": performance(curve),
        "period": {
            "start": grouped[0][0].isoformat(),
            "end": grouped[-1][0].isoformat(),
        },
        "benchmark_metrics": performance(bench),
        "benchmark_curve": [
            {"timestamp": ts.isoformat(), "equity": float(v)} for ts, v in bench.items()
        ],
        "dataset_version": version(dataset),
        "fixture": dataset.fixture,
        "source": dataset.source,
        "partitions": {
            "train_end": str(dates[train_end - 1]),
            "validation_end": str(dates[val_end - 1]),
            "test_start": str(test_start),
        },
        "execution": "Signals at bar close; orders eligible only on a later bar. Long-only, fractional shares.",
        "evaluation": config.evaluation,
    }
    result["benchmark_comparison"] = {
        "strategy_return": result["metrics"]["total_return"],
        "benchmark_return": result["benchmark_metrics"]["total_return"],
        "outperformed": result["metrics"]["total_return"]
        > result["benchmark_metrics"]["total_return"],
        "benchmark": config.benchmark,
        "scope": "Same evaluation dates. Benchmark total-return curve excludes simulated trading costs. Compare with other strategies only when dates match.",
    }
    if config.evaluation == "walk_forward":
        folds = []
        for chunk in np.array_split(
            np.array(sorted({ts.date() for ts, _ in grouped}), dtype=object), 3
        ):
            if not len(chunk):
                continue
            sub = curve[[ts.date() in chunk for ts in curve.index]]
            if len(sub):
                folds.append(
                    {
                        "start": str(chunk[0]),
                        "end": str(chunk[-1]),
                        "metrics": performance(sub),
                    }
                )
        result["walk_forward"] = {
            "folds": folds,
            "policy": "Fixed rules; rolling past-only windows, no test-set tuning",
        }
    return result


def replay_step(account, dataset, strategy, batch=1, news=(), policy=None):
    from .schemas import Backtest

    value = deepcopy(account)
    if value["status"] != "running":
        raise ValueError("Paper account is not running")
    config = Backtest.model_validate(value["config"])
    frame = aligned_prices(dataset, strategy.symbols)
    remaining = [
        (ts, bars)
        for ts, bars in groups(dataset)
        if not value["state"]["last_timestamp"]
        or ts > pd.Timestamp(value["state"]["last_timestamp"])
    ]
    for ts, bars in remaining[:batch]:
        advance(
            value["state"],
            ts,
            bars,
            frame.loc[:ts],
            strategy,
            config,
            dataset.actions,
            news,
            policy,
        )
    value["last_feed_timestamp"] = value["state"]["last_timestamp"]
    value["replay_complete"] = len(remaining) <= batch
    return value
