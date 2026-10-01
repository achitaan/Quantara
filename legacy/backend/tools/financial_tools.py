"""Financial calculation tools for portfolio analysis and risk assessment."""

import math
import chainlit as cl
import numpy as np
from typing import List, Dict, Any

calculate_portfolio_metrics_def = {
    "name": "calculate_portfolio_metrics",
    "description": "Calculates key portfolio metrics including expected return, volatility, and Sharpe ratio.",
    "parameters": {
        "type": "object",
        "properties": {
            "weights": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Portfolio weights for each asset (should sum to 1.0)",
            },
            "expected_returns": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Expected annual returns for each asset (as decimals, e.g., 0.08 for 8%)",
            },
            "volatilities": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Annual volatilities for each asset (as decimals, e.g., 0.15 for 15%)",
            },
            "correlation_matrix": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "number"}
                },
                "description": "Correlation matrix between assets",
            },
            "risk_free_rate": {
                "type": "number",
                "description": "Risk-free rate (as decimal, e.g., 0.02 for 2%)",
                "default": 0.02,
            },
        },
        "required": ["weights", "expected_returns", "volatilities", "correlation_matrix"],
    },
}

async def calculate_portfolio_metrics_handler(
    weights: List[float], 
    expected_returns: List[float], 
    volatilities: List[float], 
    correlation_matrix: List[List[float]], 
    risk_free_rate: float = 0.02
):
    """
    Calculates portfolio metrics using Modern Portfolio Theory.
    """
    try:
        weights = np.array(weights)
        expected_returns = np.array(expected_returns)
        volatilities = np.array(volatilities)
        correlation_matrix = np.array(correlation_matrix)
        
        # Validation
        if not np.isclose(weights.sum(), 1.0, atol=0.01):
            return {"error": "Portfolio weights must sum to 1.0"}
        
        if len(weights) != len(expected_returns) != len(volatilities):
            return {"error": "All input arrays must have the same length"}
        
        # Portfolio expected return
        portfolio_return = np.dot(weights, expected_returns)
        
        # Covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        # Portfolio variance and volatility
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        return {
            "portfolio_expected_return": round(portfolio_return * 100, 2),  # Convert to percentage
            "portfolio_volatility": round(portfolio_volatility * 100, 2),  # Convert to percentage
            "sharpe_ratio": round(sharpe_ratio, 3),
            "risk_free_rate": round(risk_free_rate * 100, 2),
            "excess_return": round((portfolio_return - risk_free_rate) * 100, 2),
        }
    
    except Exception as e:
        return {"error": str(e)}

calculate_portfolio_metrics = (calculate_portfolio_metrics_def, calculate_portfolio_metrics_handler)

calculate_var_def = {
    "name": "calculate_var",
    "description": "Calculates Value at Risk (VaR) for a portfolio using the parametric method.",
    "parameters": {
        "type": "object",
        "properties": {
            "portfolio_value": {
                "type": "number",
                "description": "Current portfolio value in dollars",
            },
            "expected_return": {
                "type": "number",
                "description": "Expected annual return (as decimal, e.g., 0.08 for 8%)",
            },
            "volatility": {
                "type": "number",
                "description": "Annual volatility (as decimal, e.g., 0.15 for 15%)",
            },
            "confidence_level": {
                "type": "number",
                "description": "Confidence level (as decimal, e.g., 0.95 for 95%)",
                "default": 0.95,
            },
            "time_horizon": {
                "type": "number",
                "description": "Time horizon in days",
                "default": 1,
            },
        },
        "required": ["portfolio_value", "expected_return", "volatility"],
    },
}

async def calculate_var_handler(
    portfolio_value: float,
    expected_return: float,
    volatility: float,
    confidence_level: float = 0.95,
    time_horizon: int = 1
):
    """
    Calculates Value at Risk using the parametric method.
    """
    try:
        from scipy.stats import norm
        
        # Convert annual metrics to the specified time horizon
        daily_return = expected_return / 252  # 252 trading days per year
        daily_volatility = volatility / math.sqrt(252)
        
        horizon_return = daily_return * time_horizon
        horizon_volatility = daily_volatility * math.sqrt(time_horizon)
        
        # Calculate VaR
        z_score = norm.ppf(1 - confidence_level)
        var_return = horizon_return + z_score * horizon_volatility
        var_dollar = portfolio_value * abs(var_return)
        
        return {
            "portfolio_value": portfolio_value,
            "confidence_level": round(confidence_level * 100, 1),
            "time_horizon_days": time_horizon,
            "var_dollar": round(var_dollar, 2),
            "var_percentage": round(abs(var_return) * 100, 2),
            "interpretation": f"There is a {round((1-confidence_level)*100, 1)}% chance that the portfolio will lose more than ${round(var_dollar, 2)} over {time_horizon} day(s)."
        }
    
    except Exception as e:
        return {"error": str(e)}

calculate_var = (calculate_var_def, calculate_var_handler)

calculate_beta_def = {
    "name": "calculate_beta",
    "description": "Calculates the beta coefficient of an asset relative to a benchmark.",
    "parameters": {
        "type": "object",
        "properties": {
            "asset_returns": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Historical returns of the asset",
            },
            "benchmark_returns": {
                "type": "array", 
                "items": {"type": "number"},
                "description": "Historical returns of the benchmark (e.g., market index)",
            },
        },
        "required": ["asset_returns", "benchmark_returns"],
    },
}

async def calculate_beta_handler(asset_returns: List[float], benchmark_returns: List[float]):
    """
    Calculates beta coefficient using linear regression.
    """
    try:
        asset_returns = np.array(asset_returns)
        benchmark_returns = np.array(benchmark_returns)
        
        if len(asset_returns) != len(benchmark_returns):
            return {"error": "Asset and benchmark return arrays must have the same length"}
        
        if len(asset_returns) < 2:
            return {"error": "Need at least 2 data points to calculate beta"}
        
        # Calculate covariance and variance
        covariance = np.cov(asset_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns, ddof=1)
        
        if benchmark_variance == 0:
            return {"error": "Benchmark variance is zero, cannot calculate beta"}
        
        # Calculate beta
        beta = covariance / benchmark_variance
        
        # Calculate correlation for additional insight
        correlation = np.corrcoef(asset_returns, benchmark_returns)[0, 1]
        
        # Interpret beta
        if beta > 1:
            interpretation = "Asset is more volatile than the benchmark"
        elif beta < 1 and beta > 0:
            interpretation = "Asset is less volatile than the benchmark"
        elif beta < 0:
            interpretation = "Asset moves inversely to the benchmark"
        else:
            interpretation = "Asset has no correlation with the benchmark"
        
        return {
            "beta": round(beta, 3),
            "correlation": round(correlation, 3),
            "interpretation": interpretation,
            "data_points": len(asset_returns)
        }
    
    except Exception as e:
        return {"error": str(e)}

calculate_beta = (calculate_beta_def, calculate_beta_handler)

# Export tools
financial_tools = [calculate_portfolio_metrics, calculate_var, calculate_beta]
