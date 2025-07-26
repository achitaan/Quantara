import chainlit as cl
from dotenv import load_dotenv
from pathlib import Path
from typing import TypedDict, Optional
import re
from dataclasses import dataclass

load_dotenv()  # ← makes OPENAI_API_KEY available for both LLM & embeddings

# ── Configuration Classes ─────────────────────────────────────────────────
@dataclass
class LLMConfig:
    model: str = "gpt-4"
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    streaming: bool = True
    timeout: int = 30

@dataclass
class ReflectionConfig:
    enabled: bool = True
    min_score_threshold: float = 7.0
    max_iterations: int = 2
    use_cheaper_model: bool = True

@dataclass
class RetrievalConfig:
    mode: str = "hybrid"  # basic, hybrid, rerank, compressed
    k: int = 6
    enable_compression: bool = False
    rerank_top_k: int = 3

@dataclass
class UIConfig:
    thinking_display_time: float = 2.0  # seconds to display thinking before answer
    auto_open_reflection: bool = True
    show_thinking_before_answer: bool = True

@dataclass
class QuantaraConfig:
    llm: LLMConfig = LLMConfig()
    reflection: ReflectionConfig = ReflectionConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    ui: UIConfig = UIConfig()
    debug_mode: bool = False

# Global configuration
config = QuantaraConfig()

# Thread-safe chain storage to avoid serialization issues
import threading
chain_storage = threading.local()

# ── LangChain / LangGraph imports ────────────────────────────────────────────
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import START, MessagesState, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessageChunk, SystemMessage, AIMessage
from langchain_core.runnables.config import RunnableConfig
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ── Tool layer imports ───────────────────────────────────────────────────────
from tools import (
    get_tool_definitions, 
    get_tool_handler, 
    get_tool_info,
    TOOL_HANDLERS
)
import json

# ── Utility Functions ────────────────────────────────────────────────────────
import time
import asyncio
from functools import wraps

def monitor_performance(operation_name: str):
    """Decorator to monitor operation performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            if config.debug_mode:
                print(f"{operation_name} took {end_time - start_time:.2f} seconds")
            
            return result
        return wrapper
    return decorator

def analyze_query_intent(user_content: str) -> dict:
    """Analyze user intent to determine best response strategy."""
    intent_patterns = {
        "calculation": ["calculate", "compute", "what is", "value", "metrics"],
        "analysis": ["analyze", "compare", "evaluate", "assess", "explain"],
        "research": ["search", "find", "lookup", "information about"],
        "regulatory": ["basel", "compliance", "regulation", "requirement"],
        "tool_demo": ["show me", "example", "demo", "how to use"]
    }
    
    user_lower = user_content.lower()
    scores = {}
    
    for intent, keywords in intent_patterns.items():
        score = sum(1 for keyword in keywords if keyword in user_lower)
        if score > 0:
            scores[intent] = score
    
    primary_intent = max(scores.items(), key=lambda x: x[1])[0] if scores else "general"
    
    return {
        "primary_intent": primary_intent,
        "confidence": max(scores.values()) if scores else 0,
        "all_scores": scores
    }

def calculate_response_quality(response: str, user_question: str) -> dict:
    """Calculate objective quality metrics for responses."""
    metrics = {
        "length_score": min(len(response) / 500, 1.0),  # Normalize to 500 chars
        "structure_score": 0.0,
        "source_score": 0.0,
        "completeness_score": 0.0
    }
    
    # Structure scoring
    structure_indicators = ["**", "###", "-", "1.", "2.", "3."]
    metrics["structure_score"] = min(
        sum(1 for indicator in structure_indicators if indicator in response) / 4, 1.0
    )
    
    # Source scoring
    if "**Sources**" in response or "**Source" in response:
        metrics["source_score"] = 1.0
    
    # Completeness scoring (basic keyword matching)
    question_keywords = set(user_question.lower().split())
    response_keywords = set(response.lower().split())
    overlap = len(question_keywords.intersection(response_keywords))
    metrics["completeness_score"] = min(overlap / max(len(question_keywords), 1), 1.0)
    
    overall_score = sum(metrics.values()) / len(metrics) * 10
    
    return {
        **metrics,
        "overall_score": overall_score
    }

# ── Custom State for Chain-of-Thought with Self-Reflection ─────────────────
class CoTState(TypedDict):
    messages: list
    thinking: Optional[str]
    show_thinking: Optional[bool]
    initial_answer: Optional[str]
    reflection: Optional[str]
    final_answer: Optional[str]
    reflection_score: Optional[float]
    improvement_needed: Optional[bool]
    iteration_count: Optional[int]
    show_reflection: Optional[bool]
    # Remove chain from state - we'll get it from session instead



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

# Use different models for different tasks to optimize costs
thinking_llm = ChatOpenAI(
    model=config.llm.model if not config.reflection.use_cheaper_model else "gpt-4o-mini", 
    temperature=0.3, 
    streaming=True,
    timeout=config.llm.timeout
)
main_llm = ChatOpenAI(
    model=config.llm.model, 
    temperature=config.llm.temperature, 
    streaming=config.llm.streaming,
    timeout=config.llm.timeout
)
reflection_llm = ChatOpenAI(
    model="gpt-4o-mini" if config.reflection.use_cheaper_model else config.llm.model, 
    temperature=0.1, 
    streaming=True,
    timeout=config.llm.timeout
)

async def thinking_node(state: CoTState):
    """Generate Chain-of-Thought reasoning before answering."""
    last_user_msg = state["messages"][-1]
    
    # More focused thinking prompt to optimize costs
    thinking_prompt = f"""
    Analyze this financial question efficiently:
    Question: {last_user_msg.content}

    Provide concise thinking:
    1. Question type: [analysis/calculation/research/regulatory]
    2. Required data: [list key information needed]
    3. Approach: [methodology in 1-2 sentences]
    4. Key considerations: [main factors to address]
    """
    
    try:
        response = thinking_llm.invoke([HumanMessage(content=thinking_prompt)])
        state["thinking"] = response.content
    except Exception as e:
        print(f"Error in thinking_node: {e}")
        state["thinking"] = f"Error generating thinking process: {str(e)}"
    
    return state

@monitor_performance("RAG with CoT")
def call_rag_with_cot(state: CoTState):
    """Enhanced RAG with Chain-of-Thought integration and error handling."""
    try:
        last_user_msg = state["messages"][-1]
        user_content = last_user_msg.content
        thinking = state.get("thinking", "")
        
        # Analyze intent for better routing
        intent_analysis = analyze_query_intent(user_content)
        user_content_lower = user_content.lower()
        
        # Get chain from thread-local storage or session
        chain = getattr(chain_storage, 'chain', None)
        if not chain:
            try:
                chain = cl.user_session.get("chain")
                if chain:
                    chain_storage.chain = chain
            except:
                pass  # Session might not be available in this context
        
        if not chain:
            content = "❌ Error: RAG chain not initialized. Please refresh the page."
            state["initial_answer"] = content
            state["messages"].append(AIMessage(content=content))
            return state
        
        # Intent-based routing with improved prompts
        if intent_analysis["primary_intent"] == "calculation" or any(word in user_content_lower for word in ["calculate portfolio", "portfolio metrics", "sharpe ratio"]):
            content = "🔧 To use portfolio calculation tools, please provide:\n" \
                     "- Portfolio weights (must sum to 1.0)\n" \
                     "- Expected returns for each asset\n" \
                     "- Volatilities for each asset\n" \
                     "- Correlation matrix\n\n" \
                     "Example: I'll calculate metrics for a portfolio with 60% stocks (10% return, 15% volatility) and 40% bonds (4% return, 5% volatility)"
        
        elif any(word in user_content_lower for word in ["calculate var", "value at risk"]):
            content = "🔧 To calculate Value at Risk (VaR), please provide:\n" \
                     "- Portfolio value in dollars\n" \
                     "- Expected annual return (as percentage)\n" \
                     "- Annual volatility (as percentage)\n" \
                     "- Confidence level (e.g., 95%)\n" \
                     "- Time horizon in days\n\n" \
                     "Example: Calculate VaR for a $1,000,000 portfolio with 8% expected return and 15% volatility"
        
        elif any(word in user_content_lower for word in ["stock price", "ticker", "financial data"]):
            content = "🔧 To get stock data, please specify:\n" \
                     "- Stock symbol (e.g., AAPL, MSFT)\n" \
                     "- Time period (e.g., 1d, 1mo, 1y)\n\n" \
                     "Example: Get Apple stock price for the last month"
        
        elif intent_analysis["primary_intent"] == "regulatory" or any(word in user_content_lower for word in ["regulatory", "basel", "compliance"]):
            # Enhanced prompt for regulatory queries
            enhanced_prompt = f"""
            Context from thinking process: {thinking}
            
            Question: {user_content}
            
            Please provide a comprehensive answer about this regulatory/compliance topic. Focus on:
            1. Key regulatory requirements and their rationale
            2. Practical implementation guidance with specific steps
            3. Risk considerations and mitigation strategies
            4. Best practices from industry experience
            5. Recent updates or changes if applicable
            
            Use clear formatting with headers and bullet points. Include specific examples where helpful.
            """
            
            try:
                response = chain.invoke(
                    {
                        "input": enhanced_prompt, 
                        "chat_history": []
                    },
                    config={"timeout": 30}
                )
                
                answer = response.get("answer", "No answer generated")
                sources = response.get("context", [])  # New chain returns 'context' instead of 'source_documents'
                
                # Extract unique source names
                unique_sources = set()
                if sources:
                    for doc in sources:
                        if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                            unique_sources.add(Path(doc.metadata['source']).name)
                
                content = answer + "\n\n**Sources**\n"
                for source in unique_sources:
                    content += f"- {source}\n"
                
                content += "\n\n💡 **Tip**: I also have specialized regulatory tools for compliance checklists and risk assessments!"
                
            except Exception as e:
                print(f"RAG chain error for regulatory query: {e}")
                content = f"❌ I encountered an error while searching regulatory information: {str(e)}\n\nPlease try rephrasing your question."
        
        else:
            # Enhanced prompt for general queries
            enhanced_prompt = f"""
            My analysis approach: {thinking}
            
            Question: {user_content}
            
            Please provide a comprehensive, well-structured answer following these guidelines:
            1. Start with a clear executive summary
            2. Provide detailed analysis with supporting evidence from the sources
            3. Include practical implications and actionable insights
            4. Use proper formatting with headers, bullet points, and numbered lists
            5. Provide specific examples or calculations where relevant
            6. Address potential risks or considerations
            7. Always cite your sources accurately
            
            Structure your response to be both comprehensive and accessible to financial professionals.
            """
            
            try:
                response = chain.invoke(
                    {
                        "input": enhanced_prompt, 
                        "chat_history": []
                    },
                    config={"timeout": 30}
                )
                
                answer = response.get("answer", "No answer generated")
                sources = response.get("context", [])  # New chain returns 'context' instead of 'source_documents'
                
                # Extract unique source names
                unique_sources = set()
                if sources:
                    for doc in sources:
                        if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                            unique_sources.add(Path(doc.metadata['source']).name)
                
                content = answer + "\n\n**Sources**\n"
                for source in unique_sources:
                    content += f"- {source}\n"
                    
            except Exception as e:
                print(f"RAG chain error for general query: {e}")
                content = f"❌ I encountered an error while searching for information: {str(e)}\n\nPlease try rephrasing your question."
        
    except Exception as e:
        print(f"Error in call_rag_with_cot: {e}")
        content = f"❌ An unexpected error occurred: {str(e)}"
    
    # Store initial answer in state for reflection
    state["initial_answer"] = content
    
    # Also add the AI response to the messages for conversation history
    state["messages"].append(AIMessage(content=content))
    
    return state

def reflection_node(state: CoTState):
    """Reflect on the initial answer and identify improvements."""
    print("DEBUG: reflection_node called")
    
    initial_answer = state.get("initial_answer", "")
    # Fix: Get user question from the right place
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    user_question = user_messages[-1].content if user_messages else ""
    
    print(f"DEBUG: initial_answer length = {len(initial_answer)}")
    
    reflection_prompt = f"""
    You are an expert financial advisor reviewing your own response. 
    
    Original Question: {user_question}
    
    Your Initial Answer: {initial_answer}
    
    Please critically evaluate this response on:
    1. **Accuracy**: Are all facts and calculations correct?
    2. **Completeness**: Does it address all aspects of the question?
    3. **Clarity**: Is the explanation clear and well-structured?
    4. **Sources**: Are the sources relevant and sufficient?
    5. **Actionability**: Does it provide practical, actionable advice?
    
    Rate the response 1-10 and explain what could be improved.
    
    Format your reflection as:
    **Score**: X/10
    **Strengths**: 
    - [List what works well]
    **Areas for Improvement**:
    - [List specific improvements needed]
    **Recommendation**: [IMPROVE/ACCEPT]
    """
    
    try:
        reflection_response = reflection_llm.invoke([HumanMessage(content=reflection_prompt)])
        
        # Parse the reflection to determine if improvement is needed
        reflection_text = reflection_response.content
        score_match = re.search(r'\*\*Score\*\*:\s*(\d+)', reflection_text)
        score = float(score_match.group(1)) if score_match else 5.0
        
        improvement_needed = "IMPROVE" in reflection_text or score < config.reflection.min_score_threshold
        
        print(f"DEBUG: reflection score = {score}, improvement_needed = {improvement_needed}")
        
        state["reflection"] = reflection_text
        state["reflection_score"] = score
        state["improvement_needed"] = improvement_needed
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        
    except Exception as e:
        print(f"Error in reflection_node: {e}")
        state["reflection"] = f"Error during reflection: {str(e)}"
        state["reflection_score"] = 5.0
        state["improvement_needed"] = False
        state["iteration_count"] = state.get("iteration_count", 0) + 1
    
    return state

def improvement_node(state: CoTState):
    """Generate an improved response based on reflection feedback."""
    # Fix: Get user question from the right place
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    user_question = user_messages[-1].content if user_messages else ""
    
    initial_answer = state.get("initial_answer", "")
    reflection = state.get("reflection", "")
    thinking = state.get("thinking", "")
    
    improvement_prompt = f"""
    Based on the reflection feedback, provide an improved response.
    
    Original Question: {user_question}
    
    Initial Thinking: {thinking}
    
    Initial Answer: {initial_answer}
    
    Reflection Feedback: {reflection}
    
    Now provide an improved, comprehensive response that addresses the identified weaknesses while maintaining the strengths. Use the same format as before with thinking, structured answer, and sources.
    """
    
    try:
        # Use the chain from thread-local storage or session
        chain = getattr(chain_storage, 'chain', None)
        if not chain:
            try:
                chain = cl.user_session.get("chain")
                if chain:
                    chain_storage.chain = chain
            except:
                pass  # Session might not be available in this context
        
        if not chain:
            # Fallback error handling
            state["final_answer"] = "❌ Error: Cannot improve response - RAG chain not available."
            return state
        
        response = chain.invoke(
            {
                "input": improvement_prompt, 
                "chat_history": []
            },
            config={"timeout": 30}
        )
        
        answer = response.get("answer", "No improved answer generated")
        sources = response.get("context", [])  # New chain returns 'context' instead of 'source_documents'
        
        # Extract unique source names
        unique_sources = set()
        if sources:
            for doc in sources:
                if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                    unique_sources.add(Path(doc.metadata['source']).name)
        
        improved_content = answer + "\n\n**Sources**\n"
        for source in unique_sources:
            improved_content += f"- {source}\n"
        
        state["final_answer"] = improved_content
        
    except Exception as e:
        print(f"Error in improvement_node: {e}")
        state["final_answer"] = f"❌ Error during improvement: {str(e)}\n\n**Fallback:** Using initial answer.\n\n{initial_answer}"
    
    return state

def finalize_response(state: CoTState):
    """Finalize the response, choosing between initial and improved answers."""
    final_answer = state.get("final_answer") or state.get("initial_answer", "")
    reflection = state.get("reflection", "")
    score = state.get("reflection_score", 0)
    iteration_count = state.get("iteration_count", 0)
    show_reflection = state.get("show_reflection", False)
    
    # Add reflection metadata to the response if reflection was performed and user wants to see it
    if reflection and iteration_count > 0 and show_reflection:
        final_answer += f"\n\n---\n**🔍 Self-Reflection Summary:**\n"
        final_answer += f"- Quality Score: {score}/10\n"
        final_answer += f"- Iterations: {iteration_count}\n"
        if score >= 8:
            final_answer += f"- Status: ✅ High-quality response\n"
        elif score >= 6:
            final_answer += f"- Status: ⚠️ Adequate response\n"
        else:
            final_answer += f"- Status: 🔄 Response improved through reflection\n"
    
    # Update the final message in state
    if state["messages"] and isinstance(state["messages"][-1], AIMessage):
        state["messages"][-1] = AIMessage(content=final_answer)
    else:
        state["messages"].append(AIMessage(content=final_answer))
    
    return state

def should_reflect(state: CoTState) -> str:
    """Decide whether to reflect on the answer."""
    # Fix: Access the correct message for user content
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if not user_messages:
        return "finalize"
    
    user_content = user_messages[-1].content.lower()
    
    print(f"DEBUG should_reflect: user_content = '{user_content[:50]}...'")
    
    # Trigger reflection for complex financial questions
    complex_keywords = [
        "analyze", "compare", "evaluate", "assess", "strategy", 
        "portfolio", "risk", "regulatory", "compliance", "calculate",
        "framework", "implementation", "factors", "consider"
    ]
    
    is_complex = any(keyword in user_content for keyword in complex_keywords)
    show_reflection = state.get("show_reflection", False)
    
    print(f"DEBUG should_reflect: is_complex = {is_complex}, show_reflection = {show_reflection}")
    
    if is_complex and show_reflection and config.reflection.enabled:
        print("DEBUG should_reflect: returning 'reflect'")
        return "reflect"
    else:
        print("DEBUG should_reflect: returning 'finalize'")
        return "finalize"

def should_improve(state: CoTState) -> str:
    """Decide whether to improve the answer based on reflection."""
    improvement_needed = state.get("improvement_needed", False)
    iteration_count = state.get("iteration_count", 0)
    
    # Use configuration for maximum iterations
    if improvement_needed and iteration_count < config.reflection.max_iterations:
        return "improve"
    else:
        return "finalize"

# Build the workflow with reflection
workflow.add_edge(START, "thinking_node")
workflow.add_edge("thinking_node", "rag_node")
workflow.add_node("thinking_node", thinking_node)
workflow.add_node("rag_node", call_rag_with_cot)
workflow.add_node("reflection_node", reflection_node)
workflow.add_node("improvement_node", improvement_node)
workflow.add_node("finalize_node", finalize_response)

# Add conditional edges for reflection workflow
workflow.add_conditional_edges(
    "rag_node",
    should_reflect,
    {
        "reflect": "reflection_node",
        "finalize": "finalize_node"
    }
)
workflow.add_conditional_edges(
    "reflection_node", 
    should_improve,
    {
        "improve": "improvement_node",
        "finalize": "finalize_node"
    }
)
workflow.add_edge("improvement_node", "finalize_node")
workflow.add_edge("finalize_node", END)

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
    """Initialize the enhanced chain when a chat session starts."""
    # Create enhanced chain with default settings
    chain = make_chain(k=config.retrieval.k, retrieval_mode=config.retrieval.mode)
    
    # Store in both session and thread-local storage
    cl.user_session.set("chain", chain)
    chain_storage.chain = chain
    
    # Create enhanced settings UI
    await create_settings_ui()
    
    welcome_message = """🚀 **Quantara-AI ready with Enhanced RAG!** 

I can help you with:
• **Financial Analysis**: Risk calculations, portfolio metrics, beta analysis
• **Document Search**: Search through regulatory docs, 10-K filings, research papers
• **Stock Data**: Real-time stock prices and charts
• **Regulatory Compliance**: Basel Framework, compliance checklists, risk assessments

**🧠 Advanced Features:**
• **Chain-of-Thought**: See my thinking process *before* the answer appears - watch me analyze the question step-by-step, then see the thinking get replaced by the final answer
• **Self-Reflection**: I evaluate and improve my own responses for better quality
• **Hybrid Search**: Combines dense vectors and keyword matching for better retrieval
• **Reranking**: Advanced relevance scoring for more accurate results

**💡 New Thinking Display:**
When Chain-of-Thought is enabled, you'll see my analysis process appear first (styled in a code block), followed by a brief pause, then it gets replaced with the comprehensive answer. You can adjust the timing in settings!

Use the settings above to customize these features!

**Available Tools:**
""" + get_tool_info() + """

Ask me anything about finance, risk management, or regulatory compliance!"""
    
    await cl.Message(welcome_message).send()

async def create_settings_ui():
    """Create enhanced settings UI."""
    settings = [
        cl.input_widget.Switch(
            id="show_cot",
            label="🧠 Chain-of-Thought Reasoning",
            initial=False
        ),
        cl.input_widget.Switch(
            id="show_reflection",
            label="🔍 Self-Reflection Process",
            initial=False
        ),
        cl.input_widget.Select(
            id="retrieval_mode",
            label="🔍 Retrieval Mode",
            values=["basic", "hybrid", "rerank", "compressed"],
            initial_index=1  # hybrid
        ),
        cl.input_widget.Slider(
            id="rag_k",
            label="📊 Documents to Retrieve",
            initial=config.retrieval.k,
            min=3,
            max=12,
            step=1
        )
    ]
    
    await cl.ChatSettings(settings).send()


@cl.on_settings_update
async def on_settings_update(settings):
    """Handle settings updates and recreate chain if needed."""
    print(f"Settings updated: {settings}")
    cl.user_session.set("settings", settings)
    
    # Check if retrieval settings changed
    current_mode = config.retrieval.mode
    current_k = config.retrieval.k
    
    new_mode = settings.get("retrieval_mode", current_mode)
    new_k = settings.get("rag_k", current_k)
    
    # Update UI config for thinking display time
    thinking_time = settings.get("thinking_time", config.ui.thinking_display_time)
    config.ui.thinking_display_time = thinking_time
    
    # Recreate chain if retrieval settings changed
    if new_mode != current_mode or new_k != current_k:
        try:
            chain = make_chain(k=new_k, retrieval_mode=new_mode)
            chain_storage.chain = chain
            cl.user_session.set("chain", chain)
            
            await cl.Message(
                content=f"✅ **Settings Updated:**\n- Retrieval mode: `{new_mode}`\n- Documents to retrieve: `{new_k}`\n- Chain-of-Thought: `{'on' if settings.get('show_cot') else 'off'}`\n- Self-Reflection: `{'on' if settings.get('show_reflection') else 'off'}`\n- Thinking display time: `{thinking_time}s`"
            ).send()
        except Exception as e:
            await cl.Message(
                content=f"⚠️ Error updating retrieval settings: {str(e)}\nUsing previous configuration."
            ).send()
    else:
        # Just update UI settings without recreating chain
        await cl.Message(
            content=f"✅ **UI Settings Updated:**\n- Chain-of-Thought: `{'on' if settings.get('show_cot') else 'off'}`\n- Self-Reflection: `{'on' if settings.get('show_reflection') else 'off'}`\n- Thinking display time: `{thinking_time}s`"
        ).send()

@cl.on_message
async def main(message: cl.Message):
    """Process user messages with Chain-of-Thought reasoning and RAG."""
    # Get user settings
    settings = cl.user_session.get("settings", {})
    show_cot = settings.get("show_cot", False)
    show_reflection = settings.get("show_reflection", False)
    
    # Store chain in thread-local storage to avoid serialization issues
    chain = cl.user_session.get("chain")
    if chain:
        chain_storage.chain = chain
    
    # Create a single message that will transform from thinking to answer
    msg = cl.Message(content="🤔 Processing your request...")
    await msg.send()
    
    try:
        # Phase 1: Show thinking process if enabled
        if show_cot:
            await show_thinking_process(msg, message.content)
        
        # Phase 2: Generate and show the answer
        answer = await generate_answer(message.content, show_reflection)
        msg.content = answer["content"]
        
        # Phase 3: Add reflection if enabled and available
        elements = []
        if show_reflection and answer.get("reflection"):
            reflection_accordion = cl.Accordion(
                content=answer["reflection"],
                title="🔍 Self-Reflection Analysis",
                open=config.ui.auto_open_reflection
            )
            elements.append(reflection_accordion)
            msg.elements = elements
        
        await msg.update()
            
    except Exception as e:
        print(f"Error in main message handler: {e}")
        error_content = f"❌ An error occurred while processing your request: {str(e)}"
        msg.content = error_content
        await msg.update()

async def show_thinking_process(msg: cl.Message, user_input: str):
    """Show real-time thinking process."""
    thinking_prompt = f"""
    Analyze this financial question efficiently:
    Question: {user_input}

    Provide concise thinking:
    1. Question type: [analysis/calculation/research/regulatory]
    2. Required data: [list key information needed]
    3. Approach: [methodology in 1-2 sentences]
    4. Key considerations: [main factors to address]
    """
    
    # Initialize thinking display
    msg.content = """**🧠 Thinking Process:**

```markdown
Analyzing question...
```

⏳ *Processing...*"""
    await msg.update()
    
    thinking_content = ""
    try:
        # Stream thinking in real-time
        async for chunk in thinking_llm.astream([HumanMessage(content=thinking_prompt)]):
            if hasattr(chunk, 'content') and chunk.content:
                thinking_content += chunk.content
                
                # Update display with current thinking
                formatted_thinking = f"""**🧠 Thinking Process:**

```markdown
{thinking_content}
```

⏳ *Generating answer based on this analysis...*"""
                
                msg.content = formatted_thinking
                await msg.update()
                
                # Small delay to make streaming visible
                await asyncio.sleep(0.1)
                
    except Exception as e:
        print(f"Error in thinking stream: {e}")
        msg.content = f"""**🧠 Thinking Process:**

```markdown
Error generating thinking process: {str(e)}
```

⏳ *Proceeding to answer generation...*"""
        await msg.update()

async def generate_answer(user_input: str, show_reflection: bool = False):
    """Generate the main answer using the RAG chain."""
    try:
        # Get chain from thread-local storage
        chain = getattr(chain_storage, 'chain', None)
        if not chain:
            chain = cl.user_session.get("chain")
            if chain:
                chain_storage.chain = chain
        
        if not chain:
            return {
                "content": "❌ Error: RAG chain not initialized. Please refresh the page.",
                "reflection": None
            }
        
        # Generate answer using RAG
        response = chain.invoke(
            {"input": user_input, "chat_history": []},
            config={"timeout": 30}
        )
        
        answer = response.get("answer", "No answer generated")
        sources = response.get("context", [])
        
        # Extract unique source names
        unique_sources = set()
        if sources:
            for doc in sources:
                if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                    unique_sources.add(Path(doc.metadata['source']).name)
        
        content = answer + "\n\n**Sources**\n"
        for source in unique_sources:
            content += f"- {source}\n"
        
        # Generate reflection if enabled
        reflection_content = None
        if show_reflection:
            reflection_content = await generate_reflection(user_input, content)
        
        return {
            "content": content,
            "reflection": reflection_content
        }
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        return {
            "content": f"❌ An error occurred while generating the answer: {str(e)}",
            "reflection": None
        }

async def generate_reflection(user_question: str, answer: str):
    """Generate reflection on the answer quality."""
    reflection_prompt = f"""
    You are an expert financial advisor reviewing your own response. 
    
    Original Question: {user_question}
    
    Your Answer: {answer}
    
    Please critically evaluate this response on:
    1. **Accuracy**: Are all facts and calculations correct?
    2. **Completeness**: Does it address all aspects of the question?
    3. **Clarity**: Is the explanation clear and well-structured?
    4. **Sources**: Are the sources relevant and sufficient?
    5. **Actionability**: Does it provide practical, actionable advice?
    
    Rate the response 1-10 and explain what could be improved.
    
    Format your reflection as:
    **Score**: X/10
    **Strengths**: 
    - [List what works well]
    **Areas for Improvement**:
    - [List specific improvements needed]
    """
    
    try:
        reflection_response = await reflection_llm.ainvoke([HumanMessage(content=reflection_prompt)])
        return reflection_response.content
    except Exception as e:
        print(f"Error generating reflection: {e}")
        return f"Error during reflection: {str(e)}"
