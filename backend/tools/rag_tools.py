"""RAG-based tools for document search and knowledge retrieval."""

import chainlit as cl
from rag.qa_chain import make_chain, make_retriever

# Initialize the RAG chain and retriever
rag_chain = make_chain()
retriever = make_retriever()

search_documents_def = {
    "name": "search_documents",
    "description": "Searches through the Quantara knowledge base for relevant information on financial topics, regulations, and research.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query or question to find relevant information in the documents",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of document chunks to retrieve (default: 6)",
                "default": 6,
            },
        },
        "required": ["query"],
    },
}

async def search_documents_handler(query: str, num_results: int = 6):
    """
    Searches through the document knowledge base and returns relevant information.
    """
    try:
        # Get relevant documents
        docs = retriever.get_relevant_documents(query)[:num_results]
        
        if not docs:
            return {"error": "No relevant documents found for the query."}
        
        results = []
        for i, doc in enumerate(docs):
            results.append({
                "chunk_id": i + 1,
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
            })
        
        return {
            "query": query,
            "num_results": len(results),
            "results": results
        }
    
    except Exception as e:
        return {"error": str(e)}

search_documents = (search_documents_def, search_documents_handler)

answer_question_def = {
    "name": "answer_question",
    "description": "Uses the RAG system to provide a comprehensive answer to a financial question based on the knowledge base.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The financial question to answer using the knowledge base",
            },
            "chat_history": {
                "type": "array",
                "description": "Previous conversation history for context",
                "items": {"type": "string"},
                "default": [],
            },
        },
        "required": ["question"],
    },
}

async def answer_question_handler(question: str, chat_history: list = None):
    """
    Provides a comprehensive answer using the RAG system.
    """
    try:
        if chat_history is None:
            chat_history = []
        
        # Use the RAG chain to get an answer
        result = rag_chain.invoke({
            "question": question, 
            "chat_history": chat_history
        })
        
        # Format the response with sources
        sources = []
        for doc in result.get("source_documents", []):
            source_info = {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            sources.append(source_info)
        
        return {
            "question": question,
            "answer": result["answer"],
            "sources": sources,
            "num_sources": len(sources)
        }
    
    except Exception as e:
        return {"error": str(e)}

answer_question = (answer_question_def, answer_question_handler)

# Export tools
rag_tools = [search_documents, answer_question]
