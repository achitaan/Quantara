from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import asyncio
import concurrent.futures
from functools import wraps

# Load environment variables at startup
load_dotenv()

app = FastAPI(title="Quantara AI API - Simple", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for blocking operations
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

def run_in_thread(func):
    """Decorator to run blocking functions in thread pool."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(thread_pool, func, *args, **kwargs)
    return wrapper

# Import agent functions
from agno_app import get_agent_response

# Wrap the blocking agent call
@run_in_thread
def get_agent_response_sync(message: str, use_rag: bool = True) -> str:
    """Thread-safe wrapper for agent response."""
    return get_agent_response(message, use_rag)

@app.get("/health")
async def health_check():
    """Simple health check."""
    try:
        # Check if vector store exists
        from pathlib import Path
        vector_store_path = Path(__file__).resolve().parent / "vector_store" / "faiss"
        rag_status = "available" if vector_store_path.exists() else "unavailable"
        
        return {
            "status": "healthy",
            "version": "1.0.0",
            "rag": rag_status,
            "environment": {
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY") is not None
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.post("/chat")
async def chat_endpoint(request: Request):
    """Simple chat endpoint with thread-safe agent calls."""
    try:
        data = await request.json()
        message = data.get("message", "")
        
        if not message:
            return JSONResponse(status_code=400, content={"error": "Message is required"})

        use_rag = bool(data.get("use_rag", True))
        
        # Call agent in thread pool to avoid blocking event loop
        response_text = await get_agent_response_sync(message, use_rag)
        
        return {"response": response_text}

    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"error": f"Agent error: {str(e)}"}
        )

@app.get("/")
async def root():
    return {
        "message": "Quantara AI Agent - Robust Version", 
        "status": "running",
        "features": ["RAG integration", "Thread-safe processing", "Financial analysis"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
