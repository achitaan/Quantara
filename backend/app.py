from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import asyncio
import concurrent.futures
from functools import wraps
import time
import os
from pathlib import Path
from typing import Dict, Literal, Optional
from pydantic import BaseModel, Field

from optimization.optimizer_factory import get_optimizer
from optimization.utils import fetch_prices, holdings_to_current_weights, metrics
# Load environment variables at startup
load_dotenv()

class OptimizeRequest(BaseModel):
    holdings: Dict[str, float]                         # {"AAPL": 10, "MSFT": 5}
    method: Literal["equal_weight","mean_variance","mean-variance","mvo","max_sharpe","max-sharpe"]
    rf: Optional[float] = None
    max_weight: Optional[float] = Field(default=None, ge=0, le=1)
    allow_short: bool = False
    horizon: Optional[str] = None                      # e.g., "6mo", "1y"

class OptimizeResponse(BaseModel):
    optimized_weights: Dict[str, float]
    current_weights: Dict[str, float]
    diff: Dict[str, float]
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]

app = FastAPI(
    title="Quantara AI API", 
    description="Enhanced Financial AI assistant with RAG, Chain-of-Thought reasoning, and work display features",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8081", "http://localhost:80", "http://localhost:3000", "http://localhost:3001", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for blocking operations
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def run_in_thread(func):
    """Decorator to run blocking functions in thread pool."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(thread_pool, func, *args, **kwargs)
    return wrapper

# Import enhanced agent functions
from agno_app import get_agent_response

# Enhanced imports for thinking and reflection
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    
    # Initialize specialized models
    thinking_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    reflection_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False
    print("Enhanced features not available - using basic mode only")

# Thread-safe wrapper for agent response
@run_in_thread
def get_agent_response_sync(message: str, use_rag: bool = True) -> str:
    """Thread-safe wrapper for agent response."""
    return get_agent_response(message, use_rag)

# Enhanced utility functions
def analyze_query_intent(user_content: str) -> dict:
    """Analyze user intent to determine best response strategy."""
    intent_patterns = {
        "calculation": ["calculate", "compute", "what is", "value", "metrics"],
        "analysis": ["analyze", "compare", "evaluate", "assess", "explain"],
        "research": ["search", "find", "lookup", "information about"],
        "regulatory": ["basel", "compliance", "regulation", "requirement"],
        "portfolio": ["portfolio", "diversification", "allocation", "risk management"]
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
    """Calculate comprehensive quality metrics for responses."""
    metrics = {
        "length_score": min(len(response) / 500, 1.0),
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
    if "**Sources**" in response or "**📚 Sources" in response:
        metrics["source_score"] = 1.0
    
    # Completeness scoring
    question_keywords = set(user_question.lower().split())
    response_keywords = set(response.lower().split())
    overlap = len(question_keywords.intersection(response_keywords))
    metrics["completeness_score"] = min(overlap / max(len(question_keywords), 1), 1.0)
    
    # Calculate overall score
    overall_score = sum([
        metrics["length_score"] * 1.0,
        metrics["structure_score"] * 2.0,
        metrics["source_score"] * 1.5,
        metrics["completeness_score"] * 2.0,
    ]) / 6.5 * 10
    
    return {
        **metrics,
        "overall_score": overall_score
    }

# Thread-safe enhanced functions
@run_in_thread
def generate_thinking_sync(user_input: str) -> str:
    """Generate structured thinking process for a question."""
    if not ENHANCED_FEATURES_AVAILABLE:
        return "Enhanced thinking features not available"
    
    thinking_prompt = f"""
    Analyze this financial question systematically:
    Question: {user_input}

    Provide structured thinking:
    1. **Question Category**: [analysis/calculation/research/regulatory/strategy]
    2. **Core Concepts**: [key financial concepts involved]
    3. **Information Needed**: [data, documents, or context required]
    4. **Analytical Approach**: [methodology and framework]
    5. **Key Considerations**: [important factors and constraints]
    6. **Expected Output**: [format and depth of response needed]
    """
    
    try:
        response = thinking_llm.invoke([HumanMessage(content=thinking_prompt)])
        return response.content
    except Exception as e:
        return f"Error generating thinking process: {str(e)}"

@run_in_thread
def generate_reflection_sync(content: str, user_question: str) -> str:
    """Generate reflection on answer quality."""
    if not ENHANCED_FEATURES_AVAILABLE:
        return "Enhanced reflection features not available"
    
    reflection_prompt = f"""
    You are an expert financial advisor reviewing a response. 
    
    Original Question: {user_question}
    
    Response to Review: {content}
    
    Please critically evaluate this response on:
    1. **Accuracy**: Are all facts and calculations correct?
    2. **Completeness**: Does it address all aspects of the question?
    3. **Clarity**: Is the explanation clear and well-structured?
    4. **Sources**: Are the sources relevant and sufficient?
    5. **Actionability**: Does it provide practical, actionable advice?
    
    Rate the response 1-10 and explain what works well and what could be improved.
    
    Format your reflection as:
    **Score**: X/10
    **Strengths**: 
    - [List what works well]
    **Areas for Improvement**:
    - [List specific improvements needed]
    """
    
    try:
        reflection_response = reflection_llm.invoke([HumanMessage(content=reflection_prompt)])
        return reflection_response.content
    except Exception as e:
        return f"Error during reflection: {str(e)}"


@app.get("/health")
async def health_check():
    """Enhanced health check with system status."""
    # Check environment variables
    env_status = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY") is not None
    }
    
    # Check RAG components
    rag_status = "unknown"
    try:
        vector_store_path = Path(__file__).resolve().parent / "vector_store" / "faiss"
        if vector_store_path.exists():
            rag_status = "available"
        else:
            rag_status = "unavailable"
    except Exception as e:
        rag_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "version": "2.0.0",
        "features": {
            "rag": rag_status,
            "thinking": "available" if ENHANCED_FEATURES_AVAILABLE else "unavailable",
            "reflection": "available" if ENHANCED_FEATURES_AVAILABLE else "unavailable",
            "quality_metrics": "available",
            "thread_safe": "enabled"
        },
        "environment": env_status
    }


@app.post("/chat")
async def chat_endpoint(request: Request):
    """Enhanced chat endpoint with thinking and reflection capabilities."""
    try:
        data = await request.json()
        message = data.get("message", "")
        
        if not message:
            return JSONResponse(status_code=400, content={"error": "Message is required"})

        # Extract options
        use_rag = bool(data.get("use_rag", True))
        show_thinking = bool(data.get("show_thinking", False))
        show_reflection = bool(data.get("show_reflection", False))
        enhanced_mode = bool(data.get("enhanced_mode", True))

        start_time = time.time()
        
        if enhanced_mode and ENHANCED_FEATURES_AVAILABLE:
            # Enhanced mode with all features
            thinking = None
            if show_thinking:
                thinking = await generate_thinking_sync(message)
            
            # Get the main response
            response_text = await get_agent_response_sync(message, use_rag)
            
            # Calculate quality metrics
            intent_analysis = analyze_query_intent(message)
            quality_metrics = calculate_response_quality(response_text, message)
            
            # Generate reflection if requested
            reflection = None
            if show_reflection:
                reflection = await generate_reflection_sync(response_text, message)
            
            # Build enhanced response
            result = {
                "content": response_text,
                "thinking": thinking,
                "reflection": reflection,
                "quality_score": quality_metrics["overall_score"],
                "quality_metrics": quality_metrics,
                "intent_analysis": intent_analysis,
                "processing_time": round(time.time() - start_time, 2),
                "enhanced_mode": True
            }
            
            # Extract sources if available
            sources = []
            if "**📚 Sources" in response_text or "**Sources**" in response_text:
                # Simple source extraction
                lines = response_text.split('\n')
                for line in lines:
                    if line.strip().startswith('-') and '.pdf' in line:
                        source_name = line.strip().replace('-', '').strip()
                        sources.append(source_name)
            
            result["sources"] = sources
            return result
            
        else:
            # Simple mode - just basic response
            response_text = await get_agent_response_sync(message, use_rag)
            
            # Ensure we return a proper response even in simple mode
            if not response_text or response_text.strip() == "":
                response_text = "I apologize, but I was unable to generate a response. Please try rephrasing your question."
            
            return {
                "response": response_text,  # Use 'response' field for compatibility
                "content": response_text,   # Also include 'content' field for enhanced compatibility
                "processing_time": round(time.time() - start_time, 2),
                "enhanced_mode": False
            }

    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"error": f"Agent error: {str(e)}"}
        )


@app.post("/thinking")
async def thinking_endpoint(request: Request):
    """Get just the thinking process for a question."""
    try:
        data = await request.json()
        message = data.get("message", "")
        
        if not message:
            return JSONResponse(status_code=400, content={"error": "Message is required"})

        if not ENHANCED_FEATURES_AVAILABLE:
            return JSONResponse(status_code=503, content={"error": "Enhanced features not available"})

        # Get thinking process only
        thinking = await generate_thinking_sync(message)
        intent_analysis = analyze_query_intent(message)
        
        return {
            "thinking": thinking,
            "intent_analysis": intent_analysis,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"error": f"Thinking generation error: {str(e)}"}
        )


@app.post("/analyze")
async def analyze_endpoint(request: Request):
    """Analyze a query without generating a full response."""
    try:
        data = await request.json()
        message = data.get("message", "")
        
        if not message:
            return JSONResponse(status_code=400, content={"error": "Message is required"})

        # Get intent analysis
        intent_analysis = analyze_query_intent(message)
        
        return {
            "message": message,
            "intent_analysis": intent_analysis,
            "suggested_approach": {
                "use_rag": intent_analysis.get("primary_intent") in ["research", "regulatory", "analysis"],
                "show_thinking": intent_analysis.get("primary_intent") in ["calculation", "analysis"],
                "show_reflection": intent_analysis.get("confidence", 0) < 2
            }
        }

    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"error": f"Analysis error: {str(e)}"}
        )


@app.get("/")
async def root():
    return {
        "message": "Quantara AI Agent with Enhanced Agno Framework", 
        "version": "2.0.0",
        "features": [
            "Chain-of-Thought reasoning" if ENHANCED_FEATURES_AVAILABLE else "Basic reasoning",
            "RAG integration with FAISS",
            "Response quality metrics",
            "Self-reflection capabilities" if ENHANCED_FEATURES_AVAILABLE else "Basic responses",
            "Intent analysis",
            "Thread-safe processing"
        ],
        "endpoints": {
            "/health": "System health check",
            "/chat": "Main chat with full features",
            "/thinking": "Get thinking process only",
            "/analyze": "Analyze query intent"
        },
        "enhanced_features": ENHANCED_FEATURES_AVAILABLE
    }

@app.post("/optimize-portfolio", response_model=OptimizeResponse)
async def optimize_portfolio(req: OptimizeRequest):
    period = req.horizon or os.getenv("DATA_HORIZON", "1y")
    prices = fetch_prices(req.holdings.keys(), period=period)
    if prices.empty:
        raise HTTPException(status_code=400, detail="No price data found for given holdings/period")

    latest = prices.iloc[-1]
    curr_w = holdings_to_current_weights(req.holdings, latest)

    optimizer = get_optimizer(req.method, rf=req.rf, max_weight=req.max_weight, allow_short=req.allow_short)
    opt_w = optimizer.optimize(prices, req.holdings)

    rf = req.rf if req.rf is not None else float(os.getenv("RISK_FREE_RATE", "0.02"))
    before = metrics(curr_w, prices, rf=rf)
    after  = metrics(opt_w, prices, rf=rf)

    # diff and rounding
    tickers = list(prices.columns)
    diff = {t: round(opt_w.get(t, 0.0) - curr_w.get(t, 0.0), 6) for t in tickers}
    rnd = lambda d: {k: round(float(v), 6) for k, v in d.items()}

    return OptimizeResponse(
        optimized_weights=rnd(opt_w),
        current_weights=rnd(curr_w),
        diff=diff,
        metrics_before={k: round(v, 6) for k, v in before.items()},
        metrics_after={k: round(v, 6) for k, v in after.items()},
    )