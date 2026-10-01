# chooses the strategy
from .base_optimizer import BaseOptimizer
from .strategies.equal_weight import EqualWeightOptimizer
from .strategies.mean_variance import MeanVarianceOptimizer

def get_optimizer(method: str, **kwargs) -> BaseOptimizer:
    m = (method or "").lower()
    if m in ("equal_weight", "equal-weight", "ew"):
        return EqualWeightOptimizer()
    if m in ("mean_variance", "mean-variance", "mvo", "max_sharpe", "max-sharpe"):
        return MeanVarianceOptimizer(
            rf=kwargs.get("rf"),
            max_weight=kwargs.get("max_weight"),
            allow_short=kwargs.get("allow_short", False),
        )
    raise ValueError(f"Unsupported optimization method: {method}")