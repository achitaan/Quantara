#shared interface
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict

class BaseOptimizer(ABC):
    @abstractmethod
    def optimize(self, prices: pd.DataFrame, holdings: Dict[str, float]) -> Dict[str, float]:
        """
        Takes historical price data and user holdings.
        Returns optimized portfolio weights.
        """
        pass