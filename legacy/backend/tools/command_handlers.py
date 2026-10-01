"""Command handlers for direct tool execution via Chainlit interface."""

import chainlit as cl
import json
from tools import get_tool_handler

@cl.action_callback("calculate_portfolio")
async def handle_portfolio_calculation():
    """Handle portfolio calculation with user input."""
    # Example portfolio data
    weights = [0.6, 0.4]  # 60% stocks, 40% bonds
    expected_returns = [0.10, 0.04]  # 10% stocks, 4% bonds
    volatilities = [0.15, 0.05]  # 15% stocks, 5% bonds
    correlation_matrix = [[1.0, 0.3], [0.3, 1.0]]  # Low correlation
    
    handler = get_tool_handler("calculate_portfolio_metrics")
    result = await handler(
        weights=weights,
        expected_returns=expected_returns,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
        risk_free_rate=0.02
    )
    
    if "error" in result:
        await cl.Message(f"❌ Error: {result['error']}").send()
    else:
        message = f"""📊 **Portfolio Analysis Results**

**Portfolio Composition:**
- 60% Stocks (10% expected return, 15% volatility)
- 40% Bonds (4% expected return, 5% volatility)

**Key Metrics:**
- Expected Return: {result['portfolio_expected_return']}%
- Volatility (Risk): {result['portfolio_volatility']}%
- Sharpe Ratio: {result['sharpe_ratio']}
- Excess Return: {result['excess_return']}%

**Interpretation:**
The Sharpe ratio of {result['sharpe_ratio']} indicates the risk-adjusted return per unit of risk."""

        await cl.Message(message).send()

@cl.action_callback("search_documents") 
async def handle_document_search():
    """Handle document search example."""
    handler = get_tool_handler("search_documents")
    result = await handler(
        query="What are the key principles of risk management?",
        num_results=3
    )
    
    if "error" in result:
        await cl.Message(f"❌ Error: {result['error']}").send()
    else:
        message = f"""🔍 **Document Search Results**

**Query:** {result['query']}
**Found:** {result['num_results']} relevant documents

"""
        for i, doc in enumerate(result['results']):
            message += f"""**Result {i+1}:**
- **Source:** {doc['source']}
- **Page:** {doc['page']}
- **Content:** {doc['content']}

---
"""
        
        await cl.Message(message).send()

# Add action buttons to the interface
portfolio_action = cl.Action(
    name="calculate_portfolio",
    value="calculate",
    description="📊 Calculate Portfolio Metrics",
    label="Portfolio Analysis"
)

search_action = cl.Action(
    name="search_documents", 
    value="search",
    description="🔍 Search Knowledge Base",
    label="Document Search"
)

DEMO_ACTIONS = [portfolio_action, search_action]
