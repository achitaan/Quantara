# for weight conversion, normalization, and other utility functions.
from typing import Dict, Iterable
import numpy as np
import pandas as pd
import yfinance as yf

def fetch_prices(tickers: Iterable[str], period: str = "1y") -> pd.DataFrame:
    tickers = list(tickers)
    if not tickers:
        return pd.DataFrame()
    px = yf.download(tickers, period=period, progress=False)["Adj Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    return px.dropna()

def holdings_to_current_weights(holdings: Dict[str, float], latest_prices: pd.Series) -> Dict[str, float]:
    values = {t: float(holdings.get(t, 0.0)) * float(latest_prices.get(t, 0.0)) for t in latest_prices.index}
    total = sum(values.values())
    if total <= 0:
        n = len(latest_prices.index)
        return {t: 1.0/n for t in latest_prices.index} if n else {}
    return {t: v/total for t, v in values.items()}

def metrics(weights: Dict[str, float], prices: pd.DataFrame, rf: float = 0.02) -> Dict[str, float]:
    if prices.empty or not weights:
        return {"expected_return": 0.0, "volatility": 0.0, "sharpe": 0.0}
    rets = prices.pct_change().dropna()
    mu = rets.mean() * 252.0
    Sigma = rets.cov() * 252.0
    # align order
    tickers = list(prices.columns)
    w = np.array([weights.get(t, 0.0) for t in tickers])
    ret = float(w @ mu[tickers].values)
    vol = float(np.sqrt(max(w @ Sigma.loc[tickers, tickers].values @ w, 1e-18)))
    sharpe = (ret - rf) / vol if vol > 0 else 0.0
    return {"expected_return": ret, "volatility": vol, "sharpe": sharpe}
