# Same weight for every company, more balanced portfolio
from typing import Dict
import pandas as pd
from ..base_optimizer import BaseOptimizer

class EqualWeightOptimizer(BaseOptimizer):
    def optimize(self, prices: pd.DataFrame, holdings: Dict[str, float]) -> Dict[str, float]:
        # Use the tickers present in the price DataFrame
        tickers = list(prices.columns)
        n = len(tickers)
        w = 1.0 / n if n else 0.0 # equal weight
        return {t: w for t in tickers}
