#!/usr/bin/env python3
"""Test script to verify system prompt loading and chain functionality"""

import sys
from pathlib import Path

# Add the backend directory to the path
backend_dir = Path(__file__).parent
sys.path.append(str(backend_dir))

from rag.qa_chain import make_chain, SYSTEM

def test_system_prompt():
    """Test if system prompt loads correctly"""
    print("=== System Prompt Test ===")
    print(f"System prompt length: {len(SYSTEM)} characters")
    print("First 200 characters:")
    print(SYSTEM[:200])
    print("\n" + "="*50 + "\n")
    
    # Check for key phrases that should prevent refusal
    key_phrases = [
        "financial expert",
        "banking regulation", 
        "ALLOWED and ENCOURAGED",
        "Never refuse to answer"
    ]
    
    for phrase in key_phrases:
        if phrase in SYSTEM:
            print(f"✅ Found: '{phrase}'")
        else:
            print(f"❌ Missing: '{phrase}'")
    
    print("\n" + "="*50 + "\n")

def test_chain():
    """Test if chain works with a simple query"""
    print("=== Chain Test ===")
    try:
        chain = make_chain(k=3)
        print("✅ Chain created successfully")
        
        # Test with a simple regulatory question
        response = chain.invoke({
            "question": "What are the main pillars of Basel III?",
            "chat_history": []
        })
        
        answer = response.get("answer", "No answer")
        print(f"Answer length: {len(answer)} characters")
        print("First 200 characters:")
        print(answer[:200])
        
        if "sorry" in answer.lower() and "can't assist" in answer.lower():
            print("❌ Chain is refusing to answer")
        else:
            print("✅ Chain provided a substantive answer")
            
    except Exception as e:
        print(f"❌ Error testing chain: {e}")
    
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    test_system_prompt()
    test_chain()
