from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# Load environment variables at startup
load_dotenv()

app = FastAPI(title="Quantara AI API - Working", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import agent function
from agno_app import get_agent_response

@app.get("/health")
def health_check():
    """Simple health check."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY") is not None
        }
    }

@app.post("/chat")
def chat_endpoint(request: dict):
    """Simple chat endpoint that actually works."""
    try:
        message = request.get("message", "")
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")

        use_rag = bool(request.get("use_rag", True))
        
        # Call agent directly (synchronous)
        response_text = get_agent_response(message, use_rag)
        
        # Ensure we have a response
        if not response_text or response_text.strip() == "":
            response_text = "I apologize, but I was unable to generate a response. Please try rephrasing your question."
        
        return {
            "response": response_text,
            "status": "success"
        }

    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"error": f"Agent error: {str(e)}", "status": "error"}
        )

@app.get("/")
def root():
    return {"message": "Quantara AI Agent - Simple Working Version", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
