import chainlit as cl
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()  # ← makes OPENAI_API_KEY available for both LLM & embeddings

# ── LangChain / LangGraph imports ────────────────────────────────────────────
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessageChunk
from langchain_core.runnables.config import RunnableConfig
from langchain.chains import ConversationalRetrievalChain

# ── Tool layer imports ───────────────────────────────────────────────────────
from tools import (
    get_tool_definitions, 
    get_tool_handler, 
    get_tool_info,
    TOOL_HANDLERS
)
import json



# ── Build the RAG retriever once ─────────────────────────────────────────────
INDEX_DIR = (
    Path(__file__).resolve().parent / "vector_store" / "faiss"
)  # backend/vector_store/faiss
_vectordb = FAISS.load_local(
    str(INDEX_DIR),
    OpenAIEmbeddings(),
     allow_dangerous_deserialization=True,
)

_retriever = _vectordb.as_retriever(search_kwargs={"k": 6})

# ── LangGraph workflow ───────────────────────────────────────────────────────
workflow = StateGraph(state_schema=MessagesState)
llm = ChatOpenAI(model="gpt-4", temperature=0, streaming=True)

async def handle_tool_call(tool_name: str, tool_args: dict):
    """Handle tool calls and return results."""
    try:
        handler = get_tool_handler(tool_name)
        if handler:
            result = await handler(**tool_args)
            return result
        else:
            return {"error": f"Tool '{tool_name}' not found"}
    except Exception as e:
        return {"error": f"Tool execution failed: {str(e)}"}

def call_rag_with_tools(state: MessagesState):
    """Enhanced RAG with tool calling capability."""
    last_user_msg = state["messages"][-1]  # HumanMessage
    user_content = last_user_msg.content.lower()
    
    # Simple pattern matching for tool usage
    if any(word in user_content for word in ["calculate portfolio", "portfolio metrics", "sharpe ratio"]):
        content = "🔧 To use portfolio calculation tools, please provide:\n" \
                 "- Portfolio weights (must sum to 1.0)\n" \
                 "- Expected returns for each asset\n" \
                 "- Volatilities for each asset\n" \
                 "- Correlation matrix\n\n" \
                 "Example: I'll calculate metrics for a portfolio with 60% stocks (10% return, 15% volatility) and 40% bonds (4% return, 5% volatility)"
    
    elif any(word in user_content for word in ["calculate var", "value at risk"]):
        content = "🔧 To calculate Value at Risk (VaR), please provide:\n" \
                 "- Portfolio value in dollars\n" \
                 "- Expected annual return (as percentage)\n" \
                 "- Annual volatility (as percentage)\n" \
                 "- Confidence level (e.g., 95%)\n" \
                 "- Time horizon in days\n\n" \
                 "Example: Calculate VaR for a $1,000,000 portfolio with 8% expected return and 15% volatility"
    
    elif any(word in user_content for word in ["stock price", "ticker", "financial data"]):
        content = "🔧 To get stock data, please specify:\n" \
                 "- Stock symbol (e.g., AAPL, MSFT)\n" \
                 "- Time period (e.g., 1d, 1mo, 1y)\n\n" \
                 "Example: Get Apple stock price for the last month"
    
    elif any(word in user_content for word in ["regulatory", "basel", "compliance"]):
        # Use RAG for regulatory queries but mention regulatory tools
        chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(model="gpt-4", temperature=0, streaming=True),
            _retriever,
            return_source_documents=True,
            verbose=False,
        )
        
        response = chain.invoke(
            {"question": last_user_msg.content, "chat_history": []}
        )
        
        answer = response["answer"]
        sources = response["source_documents"]
        unique_sources = {Path(doc.metadata.get('source', 'unknown')).name for doc in sources}
        
        content = answer + "\n\n**Sources**\n"
        for source in unique_sources:
            content += f"- {source}\n"
        
        content += "\n\n💡 **Tip**: I also have specialized regulatory tools for compliance checklists and risk assessments!"
    
    else:
        # Use regular RAG chain
        chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(model="gpt-4", temperature=0, streaming=True),
            _retriever,
            return_source_documents=True,
            verbose=False,
        )
        
        response = chain.invoke(
            {"question": last_user_msg.content, "chat_history": []}
        )
        
        # Build a markdown reply with unique citations
        answer = response["answer"]
        sources = response["source_documents"]
        unique_sources = {Path(doc.metadata.get('source', 'unknown')).name for doc in sources}
        
        content = answer + "\n\n**Sources**\n"
        for source in unique_sources:
            content += f"- {source}\n"
    
    # Return back to the graph as an AI message
    return {"messages": AIMessageChunk(content=content)}

workflow.add_edge(START, "rag_node")
workflow.add_node("rag_node", call_rag_with_tools)

# Memory (optional but kept from your original example)
memory          = MemorySaver()
langgraph_app   = workflow.compile(checkpointer=memory)

# ── Chainlit auth (unchanged) ────────────────────────────────────────────────
@cl.password_auth_callback
def auth_callback(username, password):
    if (username, password) == ("admin", "admin"):
        return cl.User(identifier="admin", metadata={"role": "admin"})
    return None

from rag.qa_chain import make_chain

@cl.on_chat_start
async def on_chat_start():
    chain = make_chain(k=6)
    cl.user_session.set("chain", chain)
    
    welcome_message = """🚀 **Quantara-AI ready!** 

I can help you with:
• **Financial Analysis**: Risk calculations, portfolio metrics, beta analysis
• **Document Search**: Search through regulatory docs, 10-K filings, research papers
• **Stock Data**: Real-time stock prices and charts
• **Regulatory Compliance**: Basel Framework, compliance checklists, risk assessments

**Available Tools:**
""" + get_tool_info() + """

Ask me anything about finance, risk management, or regulatory compliance!"""
    
    await cl.Message(welcome_message).send()


@cl.on_message
async def main(message: cl.Message):
    placeholder = cl.Message(content="")
    await placeholder.send()

    cfg: RunnableConfig = {"configurable": {"thread_id": cl.context.session.thread_id}}

    # Stream LangGraph output back to the UI
    for chunk, _ in langgraph_app.stream(
        {"messages": [HumanMessage(content=message.content)]},
        cfg,
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessageChunk):
            placeholder.content += chunk.content  # type: ignore
            await placeholder.update()
