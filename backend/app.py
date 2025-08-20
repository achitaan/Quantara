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
from typing import Dict, Any

# Load environment variables at startup
load_dotenv()

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

def clean_response_text(response: str) -> str:
    """Clean up response text by removing thinking sections and answer headers."""
    # Handle the response by looking for structured sections
    
    # First, try to find if there's an "Answer:" section and extract everything after it
    if "**Answer:**" in response:
        # Split on **Answer:** and take everything after
        parts = response.split("**Answer:**", 1)
        if len(parts) > 1:
            response = parts[1].strip()
    elif "Answer:" in response:
        # Split on Answer: and take everything after
        parts = response.split("Answer:", 1)
        if len(parts) > 1:
            response = parts[1].strip()
    
    # Now remove any thinking sections that might be at the beginning
    lines = response.split('\n')
    cleaned_lines = []
    skip_thinking = False
    found_content = False
    
    for line in lines:
        line_stripped = line.strip()
        line_lower = line_stripped.lower()
        
        # Check if this is a thinking section header
        if ('thinking:' in line_lower or '**thinking:**' in line_lower) and not found_content:
            skip_thinking = True
            continue
        
        # Check if we've reached actual content (starts with ## or # headers typically)
        if line_stripped.startswith('#') or line_stripped.startswith('**') or (line_stripped and not skip_thinking):
            skip_thinking = False
            found_content = True
        
        # Skip lines that are part of the thinking section
        if skip_thinking and not found_content:
            continue
        
        # Keep the line if it's actual content
        cleaned_lines.append(line)
    
    # Join the cleaned lines and remove extra whitespace
    cleaned_text = '\n'.join(cleaned_lines).strip()
    
    # Remove any remaining headers at the beginning
    while True:
        old_text = cleaned_text
        cleaned_text = cleaned_text.lstrip()
        
        if cleaned_text.lower().startswith('answer:'):
            cleaned_text = cleaned_text[7:].strip()
        elif cleaned_text.startswith('**Answer:**'):
            cleaned_text = cleaned_text[11:].strip()
        elif cleaned_text.lower().startswith('thinking:'):
            # Find the next section
            lines = cleaned_text.split('\n')
            start_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('#') or (line.strip().startswith('**') and 'thinking' not in line.lower()):
                    start_idx = i
                    break
            cleaned_text = '\n'.join(lines[start_idx:]).strip()
        
        if cleaned_text == old_text:
            break
    
    return cleaned_text

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
    """Generate concise thinking process for a question."""
    if not ENHANCED_FEATURES_AVAILABLE:
        return "Enhanced thinking features not available"
    
    thinking_prompt = f"""
    You are an AI assistant analyzing this question: {user_input}

    Provide a brief, concise thinking process in bullet points (similar to ChatGPT's reasoning display):
    - What type of question this is (analysis/calculation/research/etc.)
    - Key concepts or data I need to consider
    - My approach to answering this
    - Any important assumptions or limitations

    Keep it lightweight and focused - about 3-5 bullet points total. Don't include the actual answer, just the reasoning process.
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


from fastapi.responses import StreamingResponse
import json

@app.post("/thinking-stream")
async def thinking_stream_endpoint(request: Request):
    """Stream thinking process in real-time."""
    try:
        data = await request.json()
        message = data.get("message", "")
        
        if not message:
            return JSONResponse(status_code=400, content={"error": "Message is required"})

        if not ENHANCED_FEATURES_AVAILABLE:
            return JSONResponse(status_code=503, content={"error": "Enhanced features not available"})

        async def generate_thinking_stream():
            thinking_prompt = f"""
            Analyze this financial question step by step:
            Question: {message}

            Provide detailed thinking process:
            1. **Question Analysis**: What type of question is this and what are the key components?
            2. **Required Information**: What data, documents, or knowledge do I need?
            3. **Methodology**: What approach will I use to answer this comprehensively?
            4. **Key Considerations**: What important factors should I keep in mind?
            5. **Expected Outcome**: What kind of response would be most helpful?
            
            Be thorough but concise - aim for 4-6 detailed points.
            """
            
            try:
                async for chunk in thinking_llm.astream([HumanMessage(content=thinking_prompt)]):
                    if hasattr(chunk, 'content') and chunk.content:
                        yield f"data: {json.dumps({'type': 'thinking', 'content': chunk.content})}\n\n"
                
                yield f"data: {json.dumps({'type': 'complete'})}\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        return StreamingResponse(
            generate_thinking_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Thinking stream error: {str(e)}"})

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
            
            # Clean up the response text - remove thinking and answer headers
            cleaned_response = clean_response_text(response_text)
            
            # Calculate quality metrics
            intent_analysis = analyze_query_intent(message)
            quality_metrics = calculate_response_quality(cleaned_response, message)
            
            # Generate reflection if requested
            reflection = None
            if show_reflection:
                reflection = await generate_reflection_sync(cleaned_response, message)
            
            # Build enhanced response
            result = {
                "content": cleaned_response,
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
            
            # Clean up the response text even in simple mode
            cleaned_response = clean_response_text(response_text)
            
            # Ensure we return a proper response even in simple mode
            if not cleaned_response or cleaned_response.strip() == "":
                cleaned_response = "I apologize, but I was unable to generate a response. Please try rephrasing your question."
            
            return {
                "response": cleaned_response,  # Use 'response' field for compatibility
                "content": cleaned_response,   # Also include 'content' field for enhanced compatibility
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
