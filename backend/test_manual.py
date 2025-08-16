"""
Manual server test to identify the exact issue
"""
import asyncio
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import hypercorn.asyncio
from hypercorn.config import Config

print("Testing with Hypercorn instead of Uvicorn...")

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

async def main():
    config = Config()
    config.bind = ["0.0.0.0:8000"]
    
    print(f"Python version: {sys.version}")
    print("Starting Hypercorn server...")
    
    try:
        await hypercorn.asyncio.serve(app, config)
    except Exception as e:
        print(f"Hypercorn error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped by user")
    except Exception as e:
        print(f"Server error: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
