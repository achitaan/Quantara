import os
from typing import Dict
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from ..base_optimizer import BaseOptimizer

class MeanVarianceOptimizer(BaseOptimizer):
    def __init__(self, rf: float | None = None, max_weight: float | None = None, allow_short: bool = False):
        self.rf = float(rf) if rf is not None else float(os.getenv("RISK_FREE_RATE", "0.02"))
        self.max_weight = float(max_weight) if max_weight is not None else float(os.getenv("MAX_WEIGHT", "1.0"))
        self.allow_short = allow_short

    def optimize(self, prices: pd.DataFrame, holdings: Dict[str, float]) -> Dict[str, float]:
        # returns & annualized moments
        rets = prices.pct_change().dropna() # turn prices to daily simple returns
        mu = rets.mean().values * 252.0 # compute expected returns and covariance matrix
        Sigma = rets.cov().values * 252.0 # annualize

        n = len(mu)
        if n == 0:
            return {}

        # max-sharpe using SLSQP
        def neg_sharpe(w: np.ndarray) -> float:
            ret = float(w @ mu)
            vol = float(np.sqrt(max(w @ Sigma @ w, 1e-18)))
            return - (ret - self.rf) / vol

        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
        if self.allow_short:
            bounds = [(-self.max_weight, self.max_weight)] * n
        else:
            bounds = [(0.0, self.max_weight)] * n

        w0 = np.ones(n) / n
        res = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 500})
        w = res.x if res.success else w0
        # normalize and clip tiny negatives from numeric noise
        w = np.array(w, dtype=float)
        w[w < 1e-10] = 0.0
        w = w / w.sum() if w.sum() != 0 else np.ones(n)/n

        return {t: float(w[i]) for i, t in enumerate(prices.columns)}
