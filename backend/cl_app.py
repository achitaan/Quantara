import chainlit as cl
from dotenv import load_dotenv
from pathlib import Path
from typing import TypedDict, Optional

load_dotenv()  # ← makes OPENAI_API_KEY available for both LLM & embeddings

# ── LangChain / LangGraph imports ────────────────────────────────────────────
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessageChunk, SystemMessage, AIMessage
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

# ── Custom State for Chain-of-Thought ───────────────────────────────────────
class CoTState(TypedDict):
    messages: list
    thinking: Optional[str]
    show_thinking: Optional[bool]



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
workflow = StateGraph(state_schema=CoTState)
llm = ChatOpenAI(model="gpt-4", temperature=0, streaming=True)

def thinking_node(state: CoTState):
    """Generate Chain-of-Thought reasoning before answering."""
    last_user_msg = state["messages"][-1]
    
    # Create thinking prompt
    thinking_prompt = f"""
    You are Quantara-AI. Before answering the following question, think through your approach step-by-step.

    Question: {last_user_msg.content}

    Provide your thinking process in this format:
    **Thinking:**
    - What type of question is this?
    - What information do I need to gather?
    - Which sources or tools might help?
    - How should I structure my analysis?
    - What are the key considerations?

    Only provide the thinking process, not the final answer yet.
    """
    
    response = llm.invoke([HumanMessage(content=thinking_prompt)])
    
    # Store thinking in state
    state["thinking"] = response.content
    return state

def call_rag_with_cot(state: CoTState):
    """Enhanced RAG with Chain-of-Thought integration."""
    last_user_msg = state["messages"][-1]
    user_content = last_user_msg.content.lower()
    thinking = state.get("thinking", "")
    
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
        # Use RAG for regulatory queries
        chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(model="gpt-4", temperature=0, streaming=True),
            _retriever,
            return_source_documents=True,
            verbose=False,
        )
        
        # Include thinking in the prompt for better context
        enhanced_prompt = f"""
        Previous thinking: {thinking}
        
        Now answer this question: {last_user_msg.content}
        
        Follow the Quantara style guide with thinking process visible.
        """
        
        response = chain.invoke(
            {"question": enhanced_prompt, "chat_history": []}
        )
        
        answer = response["answer"]
        sources = response["source_documents"]
        unique_sources = {Path(doc.metadata.get('source', 'unknown')).name for doc in sources}
        
        content = answer + "\n\n**Sources**\n"
        for source in unique_sources:
            content += f"- {source}\n"
        
        content += "\n\n💡 **Tip**: I also have specialized regulatory tools for compliance checklists and risk assessments!"
    
    else:
        # Use regular RAG chain with thinking context
        chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(model="gpt-4", temperature=0, streaming=True),
            _retriever,
            return_source_documents=True,
            verbose=False,
        )
        
        # Include thinking context in the query
        enhanced_prompt = f"""
        My thinking process: {thinking}
        
        Question: {last_user_msg.content}
        
        Now provide the final answer following the Quantara style guide.
        """
        
        response = chain.invoke(
            {"question": enhanced_prompt, "chat_history": []}
        )
        
        answer = response["answer"]
        sources = response["source_documents"]
        unique_sources = {Path(doc.metadata.get('source', 'unknown')).name for doc in sources}
        
        content = answer + "\n\n**Sources**\n"
        for source in unique_sources:
            content += f"- {source}\n"
    
    # Store both thinking and final answer in state
    state["final_answer"] = content
    
    # Also add the AI response to the messages for conversation history
    state["messages"].append(AIMessage(content=content))
    
    return state

workflow.add_edge(START, "thinking_node")
workflow.add_edge("thinking_node", "rag_node")
workflow.add_node("thinking_node", thinking_node)
workflow.add_node("rag_node", call_rag_with_cot)

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
    
    # Add settings for Chain-of-Thought display
    await cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="show_cot",
                label="🧠 Chain-of-Thought Reasoning",
                values=["off", "on"],
                initial_index=0,
            ),
        ]
    ).send()
    
    welcome_message = """🚀 **Quantara-AI ready!** 

I can help you with:
• **Financial Analysis**: Risk calculations, portfolio metrics, beta analysis
• **Document Search**: Search through regulatory docs, 10-K filings, research papers
• **Stock Data**: Real-time stock prices and charts
• **Regulatory Compliance**: Basel Framework, compliance checklists, risk assessments

**🧠 Chain-of-Thought Feature:**
Use the settings above to enable reasoning visibility for complex questions!

**Available Tools:**
""" + get_tool_info() + """

Ask me anything about finance, risk management, or regulatory compliance!"""
    
    await cl.Message(welcome_message).send()


@cl.on_settings_update
async def on_settings_update(settings):
    """Handle settings updates for Chain-of-Thought display."""
    cl.user_session.set("settings", settings)

@cl.on_message
async def main(message: cl.Message):
    # Get user settings
    settings = cl.user_session.get("settings", {})
    show_cot = settings.get("show_cot", "off") == "on"
    
    # Create placeholder for final response
    placeholder = cl.Message(content="")
    await placeholder.send()

    cfg: RunnableConfig = {"configurable": {"thread_id": cl.context.session.thread_id}}

    # Initialize state with settings
    initial_state = {
        "messages": [HumanMessage(content=message.content)],
        "show_thinking": show_cot,
        "thinking": None
    }

    # Stream LangGraph output
    final_state = None
    for chunk in langgraph_app.stream(
        initial_state,
        cfg,
        stream_mode="values",
    ):
        final_state = chunk
    
    # Extract thinking and final answer from final state
    thinking = final_state.get("thinking", "") if final_state else ""
    messages = final_state.get("messages", []) if final_state else []
    
    # First try to get the final answer from the state
    final_answer = final_state.get("final_answer", "") if final_state else ""
    
    # If no final_answer in state, look for the last AI message
    if not final_answer and messages:
        for msg in reversed(messages):
            # Look specifically for AI messages (not HumanMessage)
            if hasattr(msg, 'content') and isinstance(msg.content, str) and msg.content.strip():
                # Check if this is not a HumanMessage (user message)
                if not isinstance(msg, HumanMessage):
                    final_answer = msg.content
                    break
    
    # Fallback if still no answer
    if not final_answer:
        final_answer = "No response generated."
    
    # Prepare elements for display
    elements = []
    
    # Add thinking as an accordion if enabled and available
    if show_cot and thinking:
        elements.append(
            cl.Accordion(
                content=thinking,
                title="🧠 Chain-of-Thought Reasoning",
                open=False  # Collapsed by default
            )
        )
    
    # Update the placeholder with final content and elements
    placeholder.content = final_answer
    placeholder.elements = elements
    await placeholder.update()
