"""
Ultra simple server test
"""
import sys
import traceback
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Create a minimal test app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    use_rag: bool = True

@app.post("/chat")
def simple_chat(request: ChatRequest):
    try:
        print(f"Received: {request.message}")
        return {"response": f"Echo: {request.message}"}
    except Exception as e:
        print(f"Error in chat: {e}")
        return {"response": f"Error: {str(e)}"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    try:
        print("Starting ultra simple server...")
        print(f"Python version: {sys.version}")
        print(f"Working directory: {sys.path[0]}")
        
        # Try to start server with debug info
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000,
            log_level="debug",
            access_log=True,
            reload=False
        )
    except Exception as e:
        print(f"Server error: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")
