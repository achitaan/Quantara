"""Master tool registry for the Quantara application."""

from .stock_tools import tools as stock_tools
from .rag_tools import rag_tools
from .financial_tools import financial_tools
from .regulatory_tools import regulatory_tools

# Combine all tools
ALL_TOOLS = stock_tools + rag_tools + financial_tools + regulatory_tools

# Organize tools by category for easy access
TOOL_CATEGORIES = {
    "stock": stock_tools,
    "rag": rag_tools,
    "financial": financial_tools,
    "regulatory": regulatory_tools,
}

# Tool definitions only (for LLM function calling)
TOOL_DEFINITIONS = [tool[0] for tool in ALL_TOOLS]

# Tool handlers mapping (name -> handler function)
TOOL_HANDLERS = {tool[0]["name"]: tool[1] for tool in ALL_TOOLS}

def get_tool_handler(tool_name: str):
    """Get the handler function for a specific tool."""
    return TOOL_HANDLERS.get(tool_name)

def get_tools_by_category(category: str):
    """Get all tools in a specific category."""
    return TOOL_CATEGORIES.get(category, [])

def get_tool_names():
    """Get list of all available tool names."""
    return list(TOOL_HANDLERS.keys())

def get_tool_definitions():
    """Get all tool definitions for LLM function calling."""
    return TOOL_DEFINITIONS

# Tool descriptions for user reference
TOOL_DESCRIPTIONS = {
    "Stock Tools": {
        "query_stock_price": "Get current stock prices and historical data",
        "draw_plotly_chart": "Create interactive charts and visualizations",
    },
    "RAG Tools": {
        "search_documents": "Search through the knowledge base for relevant information",
        "answer_question": "Get comprehensive answers using the RAG system",
    },
    "Financial Tools": {
        "calculate_portfolio_metrics": "Calculate portfolio returns, volatility, and Sharpe ratio",
        "calculate_var": "Calculate Value at Risk for portfolios",
        "calculate_beta": "Calculate beta coefficient relative to a benchmark",
    },
    "Regulatory Tools": {
        "regulatory_lookup": "Search regulatory documents and compliance requirements",
        "risk_assessment": "Perform structured risk assessments",
        "compliance_checklist": "Generate compliance checklists for regulations",
    },
}

def get_tool_info():
    """Get formatted information about all available tools."""
    info = []
    for category, tools_dict in TOOL_DESCRIPTIONS.items():
        info.append(f"\n**{category}:**")
        for tool_name, description in tools_dict.items():
            info.append(f"  • `{tool_name}`: {description}")
    
    return "\n".join(info)
