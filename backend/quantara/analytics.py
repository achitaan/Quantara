import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from scipy.stats import norm

from .market import aligned_prices, daily_prices


def optimize(
    prices,
    method="equal_weight",
    max_weight=1.0,
    rf=0.02,
    views=None,
    confidence=0.5,
    risk_aversion=2.5,
    allow_short=False,
):
    if allow_short:
        raise ValueError(
            "This release supports long-only allocations; short selling requires a borrow/margin model"
        )
    if prices.isna().any().any() or len(prices) < 3 or (prices <= 0).any().any():
        raise ValueError("Need at least three complete positive price observations")
    symbols = list(prices.columns)
    n = len(symbols)
    if not n or not 0 < max_weight <= 1 or n * max_weight < 1 - 1e-9:
        raise ValueError(
            "Infeasible allocation cap: number of assets × max_weight must be at least 1"
        )
    daily = daily_prices(prices).pct_change().dropna()
    if len(daily) < 2:
        raise ValueError(
            "At least three market sessions are required for annualized optimization"
        )
    mu = daily.mean().to_numpy() * 252
    sigma = daily.cov().to_numpy() * 252
    # Shrink covariance toward the diagonal to stabilize small research samples.
    sigma = 0.9 * sigma + 0.1 * np.diag(np.diag(sigma)) + np.eye(n) * 1e-10
    initial = np.ones(n) / n
    bounds = [(0, max_weight)] * n
    constraint = {"type": "eq", "fun": lambda w: w.sum() - 1}
    if method == "equal_weight":
        weights = initial
    elif method == "cvar":
        losses = -daily.to_numpy()
        count = len(losses)
        objective = np.r_[np.zeros(n), 1.0, np.ones(count) / (0.05 * count)]
        aub = np.c_[losses, -np.ones(count), -np.eye(count)]
        result = linprog(
            objective,
            A_ub=aub,
            b_ub=np.zeros(count),
            A_eq=np.array([np.r_[np.ones(n), np.zeros(count + 1)]]),
            b_eq=[1],
            bounds=bounds + [(None, None)] + [(0, None)] * count,
            method="highs",
        )
        if not result.success:
            raise ValueError("CVaR solver failed: " + result.message)
        weights = result.x[:n]
    else:
        if method == "black_litterman":
            if not views or set(views) - set(symbols):
                raise ValueError(
                    "Black-Litterman requires views for valid portfolio symbols"
                )
            p = np.array([[float(s == symbol) for s in symbols] for symbol in views])
            q = np.array(list(views.values()))
            tau = 0.05
            prior = risk_aversion * sigma @ initial
            omega = np.diag(
                np.diag(p @ (tau * sigma) @ p.T) * (1 - confidence) / confidence
            )
            mu = np.linalg.solve(
                np.linalg.inv(tau * sigma) + p.T @ np.linalg.solve(omega, p),
                np.linalg.solve(tau * sigma, prior) + p.T @ np.linalg.solve(omega, q),
            )

            def objective(w):
                return risk_aversion * (w @ sigma @ w) / 2 - w @ mu
        elif method == "mean_variance":

            def objective(w):
                return -(w @ mu - rf) / np.sqrt(max(w @ sigma @ w, 1e-12))
        elif method == "min_volatility":

            def objective(w):
                return w @ sigma @ w
        elif method == "risk_parity":

            def objective(w):
                contributions = w * (sigma @ w)
                normalized = contributions / max(contributions.sum(), 1e-12)
                return np.sum((normalized - 1 / n) ** 2)
        else:
            raise ValueError("Unknown optimization method")
        result = minimize(
            objective,
            initial,
            bounds=bounds,
            constraints=constraint,
            method="SLSQP",
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        if not result.success:
            raise ValueError("Allocation solver failed: " + result.message)
        weights = result.x
    if (
        abs(weights.sum() - 1) > 1e-5
        or weights.min() < -1e-6
        or weights.max() > max_weight + 1e-6
    ):
        raise ValueError("Solver returned an invalid allocation")
    return dict(zip(symbols, map(float, weights)))


def performance(equity, rf=0.02):
    daily = pd.Series(equity, dtype=float).resample("1D").last().dropna()
    returns = daily.pct_change().dropna()
    total = float(daily.iloc[-1] / daily.iloc[0] - 1) if len(daily) else 0
    vol = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else 0.0
    annual = float(returns.mean() * 252) if len(returns) else 0.0
    full = pd.Series(equity, dtype=float)
    drawdown = full / full.cummax() - 1
    return {
        "total_return": total,
        "annualized_return": annual,
        "volatility": vol,
        "sharpe": float((annual - rf) / vol) if vol > 1e-12 else None,
        "max_drawdown": float(-drawdown.min()) if len(drawdown) else 0,
        "observations": len(returns),
    }


def risk_report(dataset, weights, benchmark="SPY", confidence=0.95, shocks=None):
    if not weights or abs(sum(weights.values()) - 1) > 1e-6:
        raise ValueError("Portfolio weights must sum to one")
    prices = daily_prices(aligned_prices(dataset, list(weights)))
    returns = prices.pct_change().dropna()
    if len(returns) < 2:
        raise ValueError("Insufficient daily history for risk metrics")
    w = np.array([weights[s] for s in prices.columns])
    rp = returns.to_numpy() @ w
    losses = -rp
    cutoff = float(np.quantile(losses, confidence))
    tail = losses[losses >= cutoff]
    equity = pd.Series(np.r_[1, np.cumprod(1 + rp)], index=prices.index)
    report = performance(equity)
    report.update(
        {
            "historical_var": max(0.0, cutoff),
            "expected_shortfall": max(0.0, float(tail.mean())),
            "parametric_var": max(
                0.0, float(-rp.mean() + norm.ppf(confidence) * rp.std(ddof=1))
            ),
            "confidence": confidence,
            "metric_definitions": {
                "historical_var": "Daily historical loss quantile at the supplied confidence; not a guaranteed maximum loss",
                "expected_shortfall": "Mean daily loss in the tail at or above historical VaR; not a lower bound on every tail loss",
                "benchmark_beta": "Return sensitivity to one benchmark factor; not a volatility ratio or causal attribution",
            },
            "worst_day": float(rp.min()),
            "best_day": float(rp.max()),
            "correlation": returns.corr().to_dict(),
            "stress_loss": float(
                -sum(weights[s] * (shocks or {}).get(s, -0.2) for s in weights)
            ),
        }
    )
    all_prices = daily_prices(aligned_prices(dataset))
    if benchmark in all_prices:
        market = all_prices[benchmark].pct_change().reindex(returns.index).dropna()
        port = pd.Series(rp, index=returns.index).reindex(market.index)
        variance = market.var()
        report["benchmark_beta"] = (
            float(port.cov(market) / variance) if variance > 1e-12 else None
        )
        report["factor_exposures"] = {
            s: float(returns[s].cov(market) / variance) if variance > 1e-12 else None
            for s in returns.columns
        }
        report["factor_definition"] = (
            "Single market factor, benchmark return; not Fama-French attribution"
        )
    return report
