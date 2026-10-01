from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import os

import exchange_calendars as xcals
import httpx
import numpy as np
import pandas as pd

from .schemas import Bar, CorporateAction, Dataset, MarketFetch


def calendar():
    return xcals.get_calendar("XNYS")


def completed_bars(bars, interval, at):
    cal = calendar()
    return [
        bar
        for bar in bars
        if (
            cal.session_close(bar.timestamp.date().isoformat()).to_pydatetime() <= at
            if interval == "1d"
            else bar.timestamp + timedelta(minutes=1 if interval == "1m" else 5) <= at
        )
    ]


def feed_stale(interval, last_timestamp, at):
    if not last_timestamp:
        return True
    cal = calendar()
    reference = pd.Timestamp(at).floor("min")
    last = pd.Timestamp(last_timestamp)
    if interval == "1d":
        session = cal.date_to_session(
            reference.date().isoformat(), direction="previous"
        )
        if reference < cal.session_close(session) + pd.Timedelta(minutes=15):
            session = cal.previous_session(session)
        return last < cal.session_open(session)
    # Trading minutes, rather than overnight/weekend wall time, determine feed age.
    start = max(last.floor("min"), reference - pd.Timedelta(days=4))
    return len(cal.minutes_in_range(start, reference)) > 20


def validate_sessions(dataset: Dataset):
    cal = calendar()
    step = {"1m": 1, "5m": 5}.get(dataset.interval)
    for bar in dataset.bars:
        ts = pd.Timestamp(bar.timestamp)
        if dataset.interval == "1d":
            if not cal.is_session(ts.date().isoformat()):
                raise ValueError(f"{ts.date()} is not a US market session")
            if ts != cal.session_open(ts.date().isoformat()):
                raise ValueError("Daily timestamps must use the session open in UTC")
        else:
            if not cal.is_trading_minute(ts):
                raise ValueError(f"{ts} is outside regular market hours")
            session = cal.minute_to_session(ts)
            minutes = int((ts - cal.session_open(session)).total_seconds() / 60)
            if minutes % step or ts + pd.Timedelta(minutes=step) > cal.session_close(
                session
            ):
                raise ValueError(
                    "Intraday bar is misaligned or extends beyond the session"
                )
    observed = sorted({pd.Timestamp(b.timestamp) for b in dataset.bars})
    sessions = cal.sessions_in_range(observed[0].date(), observed[-1].date())
    if dataset.interval == "1d":
        expected = [cal.session_open(s) for s in sessions]
    else:
        expected = [
            ts
            for session in sessions
            for ts in pd.date_range(
                cal.session_open(session),
                cal.session_close(session),
                freq=f"{step}min",
                inclusive="left",
            )
            if observed[0] <= ts <= observed[-1]
        ]
    missing = sorted(set(expected) - set(observed))
    if missing:
        raise ValueError(
            f"Missing {len(missing)} market bars inside coverage; first missing {missing[0]}"
        )
    return dataset


def version(dataset):
    return sha256(
        json.dumps(dataset.model_dump(mode="json"), sort_keys=True).encode()
    ).hexdigest()


def raw_frame(dataset):
    rows = [b.model_dump() for b in dataset.bars]
    return (
        pd.DataFrame(rows)
        .pivot(index="timestamp", columns="symbol", values="close")
        .sort_index()
    )


def return_frame(dataset):
    """Point-in-time total-return prices: actions alter returns only when effective."""
    raw = raw_frame(dataset)
    curve = raw.copy()
    for symbol in raw.columns:
        previous = None
        value = None
        last_ts = None
        for ts, price in raw[symbol].items():
            if pd.isna(price):
                curve.loc[ts, symbol] = np.nan
                continue
            if previous is None:
                value = float(price)
            else:
                ratio, dividend = 1.0, 0.0
                for action in sorted(
                    dataset.actions, key=lambda a: (a.timestamp, a.type != "split")
                ):
                    if action.symbol == symbol and last_ts < action.timestamp <= ts:
                        if action.type == "split":
                            ratio *= action.amount
                        else:
                            dividend += action.amount * ratio
                value *= (float(price) * ratio + dividend) / previous
            curve.loc[ts, symbol] = value
            previous, last_ts = float(price), ts
    return curve


def aligned_prices(dataset, symbols=None):
    frame = return_frame(dataset)
    symbols = symbols or list(frame.columns)
    missing = set(symbols) - set(frame.columns)
    if missing:
        raise ValueError("Missing symbols: " + ", ".join(sorted(missing)))
    frame = frame[symbols]
    if frame.isna().any().any():
        raise ValueError("Symbols have missing or unaligned bars; import aligned data")
    if len(frame) < 3:
        raise ValueError("At least three aligned bars are required")
    return frame


def daily_prices(frame):
    return frame.resample("1D").last().dropna(how="all")


def fixture(interval="1d"):
    cal = calendar()
    rng = np.random.default_rng(42)
    days = cal.sessions_in_range("2024-01-02", "2024-12-31")
    if interval != "1d":
        days = days[:6]
    prices = {"AAPL": 180.0, "MSFT": 370.0, "SPY": 470.0}
    bars = []
    for day in days:
        timestamps = (
            [cal.session_open(day)]
            if interval == "1d"
            else list(
                pd.date_range(
                    cal.session_open(day),
                    cal.session_close(day),
                    freq=f"{1 if interval == '1m' else 5}min",
                    inclusive="left",
                )
            )
        )
        for ts in timestamps:
            common = rng.normal(0.0002, 0.008 if interval == "1d" else 0.0005)
            for symbol in prices:
                opening = prices[symbol]
                close = opening * np.exp(
                    common + rng.normal(0, 0.005 if interval == "1d" else 0.0004)
                )
                bars.append(
                    Bar(
                        timestamp=ts.to_pydatetime(),
                        symbol=symbol,
                        open=opening,
                        high=max(opening, close) * 1.001,
                        low=min(opening, close) * 0.999,
                        close=close,
                        volume=1000000 if interval == "1d" else 10000,
                    )
                )
                prices[symbol] = close
    return Dataset(
        name=f"Seeded synthetic research fixture ({interval})",
        interval=interval,
        source="fixture:seed42:v1",
        coverage="Synthetic US session bars, AAPL/MSFT/SPY; not market history",
        fixture=True,
        bars=bars,
    )


def fetch_market(req: MarketFetch):
    if req.provider == "alpaca":
        return fetch_alpaca(req)
    import yfinance as yf

    if req.interval != "1d" and req.start < datetime.now(req.start.tzinfo) - timedelta(
        days=60
    ):
        raise ValueError(
            "Yahoo intraday history is limited; import an archive or use Alpaca"
        )
    if req.start.tzinfo is None or req.end.tzinfo is None:
        raise ValueError("Date boundaries require timezones")
    # Yahoo OHLC is split-adjusted even with auto_adjust=False. Gather all subsequent
    # splits to reconstruct contemporaneous price/share units, then filter the requested range.
    history_end = max(req.end, datetime.now(timezone.utc) + timedelta(days=1))
    prices = yf.download(
        req.symbols,
        start=req.start,
        end=history_end if req.interval == "1d" else req.end,
        interval=req.interval,
        auto_adjust=False,
        actions=True,
        group_by="ticker",
        progress=False,
        threads=False,
        ignore_tz=False,
    )
    if prices is None or prices.empty:
        raise ValueError("Provider returned no data for this range")
    bars, actions = [], []
    cal = calendar()
    for symbol in req.symbols:
        frame = prices[symbol] if isinstance(prices.columns, pd.MultiIndex) else prices
        action_history = (
            frame
            if req.interval == "1d"
            else yf.Ticker(symbol).history(
                start=req.start,
                end=history_end,
                interval="1d",
                auto_adjust=False,
                actions=True,
            )
        )
        if action_history is None or action_history.empty:
            raise ValueError(f"Cannot verify corporate-action history for {symbol}")
        splits = [
            (pd.Timestamp(ts).date(), float(row["Stock Splits"]))
            for ts, row in action_history.iterrows()
            if float(row.get("Stock Splits", 0)) > 0
        ]
        if frame.empty or frame["Close"].isna().any():
            raise ValueError(f"Provider returned missing bars for {symbol}")
        for ts, row in frame.iterrows():
            ts = pd.Timestamp(ts)
            if req.interval == "1d":
                ts = cal.session_open(ts.date().isoformat())
            elif ts.tzinfo is None:
                ts = ts.tz_localize("America/New_York").tz_convert("UTC")
            if not req.start <= ts.to_pydatetime() < req.end:
                continue
            factor = float(
                np.prod([ratio for date, ratio in splits if date > ts.date()])
            )
            bars.append(
                Bar(
                    timestamp=ts.to_pydatetime(),
                    symbol=symbol,
                    open=float(row["Open"]) * factor,
                    high=float(row["High"]) * factor,
                    low=float(row["Low"]) * factor,
                    close=float(row["Close"]) * factor,
                    volume=float(row["Volume"]) / factor,
                )
            )
            for field, kind in (("Stock Splits", "split"), ("Dividends", "dividend")):
                if float(row.get(field, 0)) > 0:
                    actions.append(
                        CorporateAction(
                            timestamp=ts.to_pydatetime(),
                            symbol=symbol,
                            type=kind,
                            amount=float(row[field])
                            * (factor if kind == "dividend" else 1),
                        )
                    )
        if req.interval != "1d":
            # Intraday feed omits actions. Apply daily action events at session open.
            for at, row in action_history.iterrows():
                ts = cal.session_open(
                    pd.Timestamp(at).date().isoformat()
                ).to_pydatetime()
                if req.start <= ts < req.end:
                    factor = float(
                        np.prod([ratio for date, ratio in splits if date > ts.date()])
                    )
                    for field, kind in (
                        ("Stock Splits", "split"),
                        ("Dividends", "dividend"),
                    ):
                        value = float(row.get(field, 0))
                        if value > 0:
                            action = CorporateAction(
                                timestamp=ts,
                                symbol=symbol,
                                type=kind,
                                amount=value * (factor if kind == "dividend" else 1),
                            )
                            if action not in actions:
                                actions.append(action)
    bars = completed_bars(bars, req.interval, datetime.now(timezone.utc))
    if not bars:
        raise ValueError("Provider returned no completed bars for this range")
    dataset = Dataset(
        name=f"Yahoo {','.join(req.symbols)}",
        interval=req.interval,
        source="yahoo:reconstructed-raw",
        coverage="Yahoo split-adjusted history reconstructed into historical units; research feed coverage limits apply",
        bars=bars,
        actions=actions,
    )
    validate_sessions(dataset)
    aligned_prices(dataset, req.symbols)
    return dataset


def fetch_alpaca(req):
    key, secret = os.getenv("ALPACA_KEY"), os.getenv("ALPACA_SECRET")
    if not key or not secret:
        raise ValueError("Configure ALPACA_KEY and ALPACA_SECRET for the IEX feed")
    rows = []
    params = {
        "symbols": ",".join(req.symbols),
        "start": req.start.isoformat(),
        "end": req.end.isoformat(),
        "timeframe": {"1d": "1Day", "1m": "1Min", "5m": "5Min"}[req.interval],
        "feed": "iex",
        "adjustment": "raw",
        "limit": 10000,
    }
    with httpx.Client(timeout=30) as client:
        for _ in range(20):
            response = client.get(
                "https://data.alpaca.markets/v2/stocks/bars",
                params=params,
                headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            )
            response.raise_for_status()
            result = response.json()
            for symbol, bars in result.get("bars", {}).items():
                for b in bars:
                    ts = pd.Timestamp(b["t"])
                    if req.interval == "1d":
                        ts = calendar().session_open(ts.date().isoformat())
                    elif not calendar().is_trading_minute(ts):
                        continue
                    rows.append(
                        Bar(
                            timestamp=ts.to_pydatetime(),
                            symbol=symbol,
                            open=b["o"],
                            high=b["h"],
                            low=b["l"],
                            close=b["c"],
                            volume=b["v"],
                        )
                    )
            token = result.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
        else:
            raise ValueError(
                "Requested history exceeds import limit; choose a smaller range"
            )
    rows = completed_bars(rows, req.interval, datetime.now(timezone.utc))
    if not rows:
        raise ValueError("Provider returned no completed bars for this range")
    actions = alpaca_actions(req)
    dataset = Dataset(
        name="Alpaca IEX research bars",
        source="alpaca:iex:raw",
        coverage="IEX only; raw bars, explicit split/dividend events. Provider action publication may lag.",
        interval=req.interval,
        bars=rows,
        actions=actions,
    )
    validate_sessions(dataset)
    aligned_prices(dataset, req.symbols)
    return dataset


def alpaca_actions(req):
    params = {
        "symbols": ",".join(req.symbols),
        "start": req.start.date().isoformat(),
        "end": req.end.date().isoformat(),
        "limit": 1000,
        "data_quality": "complete",
    }
    actions = []
    headers = {
        "APCA-API-KEY-ID": os.getenv("ALPACA_KEY"),
        "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET"),
    }
    with httpx.Client(timeout=30) as client:
        for _ in range(20):
            response = client.get(
                "https://data.alpaca.markets/v1/corporate-actions",
                params=params,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
            for kind, records in data.get("corporate_actions", {}).items():
                if records and kind not in (
                    "forward_splits",
                    "reverse_splits",
                    "cash_dividends",
                ):
                    raise ValueError(
                        f"Unsupported corporate action {kind}; import an explicitly normalized dataset"
                    )
                for action in records:
                    ex_date = action["ex_date"]
                    ts = calendar().session_open(ex_date).to_pydatetime()
                    value = (
                        action["rate"]
                        if kind == "cash_dividends"
                        else action["new_rate"] / action["old_rate"]
                    )
                    actions.append(
                        CorporateAction(
                            timestamp=ts,
                            symbol=action["symbol"],
                            type="dividend" if kind == "cash_dividends" else "split",
                            amount=value,
                        )
                    )
            token = data.get("next_page_token")
            if not token:
                return actions
            params["page_token"] = token
    raise ValueError("Corporate-action archive exceeds import limit")
