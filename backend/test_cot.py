"""
Test script to verify Chain-of-Thought implementation works correctly.
"""

from rag.qa_chain import make_chain
from tools.rag_tools import make_retriever

def test_thinking_process():
    """Test the thinking process separately."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    test_question = "What are Apple's main risk factors according to their 2024 10-K filing?"
    
    thinking_prompt = f"""
    You are Quantara-AI. Before answering the following question, think through your approach step-by-step.

    Question: {test_question}

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
    print("=== THINKING PROCESS ===")
    print(response.content)
    print("\n" + "="*50 + "\n")
    
    # Now test with RAG
    chain = make_chain()
    
    enhanced_prompt = f"""
    My thinking process: {response.content}
    
    Question: {test_question}
    
    Now provide the final answer following the Quantara style guide.
    """
    
    rag_response = chain.invoke({
        "question": enhanced_prompt,
        "chat_history": []
    })
    
    print("=== FINAL ANSWER ===")
    print(rag_response["answer"])
    
    print("\n=== SOURCES ===")
    for i, doc in enumerate(rag_response.get("source_documents", [])):
        print(f"Source {i+1}: {doc.metadata.get('source', 'Unknown')}")

if __name__ == "__main__":
    test_thinking_process()
