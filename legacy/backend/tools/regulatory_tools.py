"""Regulatory and compliance tools for financial analysis."""

import chainlit as cl
from datetime import datetime, timedelta
from typing import List, Dict, Any

regulatory_lookup_def = {
    "name": "regulatory_lookup",
    "description": "Searches for regulatory information and compliance requirements from the Basel Framework and other regulatory documents.",
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The regulatory topic to search for (e.g., 'capital requirements', 'operational risk', 'liquidity risk')",
            },
            "regulation_type": {
                "type": "string",
                "description": "Type of regulation (e.g., 'basel', 'sec', 'general')",
                "default": "general",
            },
        },
        "required": ["topic"],
    },
}

async def regulatory_lookup_handler(topic: str, regulation_type: str = "general"):
    """
    Searches regulatory documents for specific compliance information.
    """
    try:
        # Import here to avoid circular imports
        from rag.qa_chain import make_retriever
        
        retriever = make_retriever()
        
        # Create a regulatory-focused query
        if regulation_type.lower() == "basel":
            query = f"Basel Framework {topic} requirements regulations compliance"
        elif regulation_type.lower() == "sec":
            query = f"SEC regulations {topic} compliance requirements"
        else:
            query = f"regulatory requirements {topic} compliance framework"
        
        # Get relevant documents
        docs = retriever.get_relevant_documents(query)[:5]
        
        if not docs:
            return {"error": f"No regulatory information found for topic: {topic}"}
        
        regulatory_info = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            
            # Prioritize Basel Framework and regulatory documents
            relevance_score = 1.0
            if "basel" in source.lower():
                relevance_score = 1.5
            elif any(reg in source.lower() for reg in ["sec", "regulatory", "compliance"]):
                relevance_score = 1.2
            
            regulatory_info.append({
                "section_id": i + 1,
                "content": doc.page_content[:800] + "..." if len(doc.page_content) > 800 else doc.page_content,
                "source": source,
                "page": doc.metadata.get("page", "N/A"),
                "relevance_score": relevance_score,
            })
        
        # Sort by relevance score
        regulatory_info.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return {
            "topic": topic,
            "regulation_type": regulation_type,
            "found_sections": len(regulatory_info),
            "regulatory_information": regulatory_info
        }
    
    except Exception as e:
        return {"error": str(e)}

regulatory_lookup = (regulatory_lookup_def, regulatory_lookup_handler)

risk_assessment_def = {
    "name": "risk_assessment",
    "description": "Provides a structured risk assessment based on regulatory frameworks and best practices.",
    "parameters": {
        "type": "object",
        "properties": {
            "risk_type": {
                "type": "string",
                "description": "Type of risk to assess (e.g., 'credit', 'operational', 'market', 'liquidity')",
            },
            "institution_type": {
                "type": "string",
                "description": "Type of financial institution (e.g., 'bank', 'investment firm', 'insurance')",
                "default": "bank",
            },
            "risk_factors": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific risk factors to consider",
                "default": [],
            },
        },
        "required": ["risk_type"],
    },
}

async def risk_assessment_handler(
    risk_type: str, 
    institution_type: str = "bank", 
    risk_factors: List[str] = None
):
    """
    Provides a structured risk assessment framework.
    """
    try:
        if risk_factors is None:
            risk_factors = []
        
        # Import here to avoid circular imports
        from rag.qa_chain import make_chain
        
        chain = make_chain()
        
        # Create a comprehensive risk assessment query
        factors_text = ", ".join(risk_factors) if risk_factors else "general factors"
        
        query = f"""
        Provide a comprehensive risk assessment framework for {risk_type} risk in a {institution_type}. 
        Consider the following specific factors: {factors_text}.
        Include:
        1. Key risk indicators and metrics
        2. Regulatory requirements and compliance considerations
        3. Risk mitigation strategies
        4. Monitoring and reporting requirements
        5. Industry best practices
        """
        
        # Get assessment from RAG system
        result = chain.invoke({
            "question": query,
            "chat_history": []
        })
        
        # Structure the response
        assessment = {
            "risk_type": risk_type,
            "institution_type": institution_type,
            "risk_factors_considered": risk_factors,
            "assessment": result["answer"],
            "regulatory_sources": [],
            "timestamp": datetime.now().isoformat(),
        }
        
        # Extract source information
        for doc in result.get("source_documents", []):
            source_info = {
                "document": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
            }
            assessment["regulatory_sources"].append(source_info)
        
        return assessment
    
    except Exception as e:
        return {"error": str(e)}

risk_assessment = (risk_assessment_def, risk_assessment_handler)

compliance_checklist_def = {
    "name": "compliance_checklist",
    "description": "Generates a compliance checklist for specific regulatory requirements.",
    "parameters": {
        "type": "object",
        "properties": {
            "regulation": {
                "type": "string",
                "description": "The regulation or framework (e.g., 'Basel III', 'CCAR', 'Solvency II')",
            },
            "focus_area": {
                "type": "string",
                "description": "Specific area of focus (e.g., 'capital adequacy', 'stress testing', 'governance')",
            },
            "institution_size": {
                "type": "string",
                "description": "Size of institution (e.g., 'large', 'medium', 'small')",
                "default": "medium",
            },
        },
        "required": ["regulation", "focus_area"],
    },
}

async def compliance_checklist_handler(
    regulation: str, 
    focus_area: str, 
    institution_size: str = "medium"
):
    """
    Generates a detailed compliance checklist.
    """
    try:
        # Import here to avoid circular imports
        from rag.qa_chain import make_chain
        
        chain = make_chain()
        
        query = f"""
        Create a detailed compliance checklist for {regulation} requirements in the area of {focus_area} 
        for a {institution_size} financial institution. Include:
        1. Specific regulatory requirements
        2. Documentation needed
        3. Processes and procedures
        4. Reporting obligations
        5. Timeline considerations
        6. Key compliance milestones
        """
        
        result = chain.invoke({
            "question": query,
            "chat_history": []
        })
        
        checklist = {
            "regulation": regulation,
            "focus_area": focus_area,
            "institution_size": institution_size,
            "checklist_items": result["answer"],
            "generated_date": datetime.now().strftime("%Y-%m-%d"),
            "sources": []
        }
        
        # Add source documents
        for doc in result.get("source_documents", []):
            checklist["sources"].append({
                "document": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A")
            })
        
        return checklist
    
    except Exception as e:
        return {"error": str(e)}

compliance_checklist = (compliance_checklist_def, compliance_checklist_handler)

# Export tools
regulatory_tools = [regulatory_lookup, risk_assessment, compliance_checklist]
