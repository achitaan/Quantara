"""
Working HTTP server with Agno integration
"""
import sys
import threading
import time
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Set up environment
from dotenv import load_dotenv
load_dotenv()

# Import our enhanced agent
from agno_app import get_agent_response_enhanced, get_agent_response
import asyncio

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "message": "Agno server running"}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/chat':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                message = data.get('message', '')
                use_rag = data.get('use_rag', True)
                include_thinking = data.get('include_thinking', False)
                include_reflection = data.get('include_reflection', False)
                
                print(f"Received message: {message}")
                print(f"RAG: {use_rag}, Thinking: {include_thinking}, Reflection: {include_reflection}")
                
                # Get response from enhanced agent
                if include_thinking or include_reflection:
                    # Create an event loop for async function
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(
                            get_agent_response_enhanced(
                                message, 
                                use_rag=use_rag,
                                show_thinking=include_thinking,
                                show_reflection=include_reflection
                            )
                        )
                        response = {
                            "response": result.get("response", ""),
                            "thinking": result.get("thinking", "") if include_thinking else None,
                            "reflection": result.get("reflection", "") if include_reflection else None,
                            "quality_score": result.get("quality_score", None),
                            "intent": result.get("intent", None)
                        }
                    finally:
                        loop.close()
                else:
                    agent_response = get_agent_response(message, use_rag=use_rag)
                    response = {"response": agent_response}
                
                print(f"Response length: {len(response.get('response', ''))}")
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                
            except Exception as e:
                print(f"Error processing request: {e}")
                import traceback
                traceback.print_exc()
                
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                error_response = {
                    "response": f"Error processing request: {str(e)}",
                    "error": True
                }
                self.wfile.write(json.dumps(error_response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def log_message(self, format, *args):
        print(f"[{self.address_string()}] {format % args}")

def run_server():
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f"🚀 Agno Enhanced Server running on http://0.0.0.0:8000")
    print(f"Python version: {sys.version}")
    print("✅ RAG components initialized")
    print("✅ Enhanced features available (thinking, reflection, quality metrics)")
    print("Press Ctrl+C to stop")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        httpd.shutdown()

if __name__ == "__main__":
    run_server()
