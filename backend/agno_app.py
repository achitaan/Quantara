"""
Quantara AI Agent using Agno Framework
Enhanced with Chain-of-Thought reasoning, RAG integration, and work display features.
"""

from dotenv import load_dotenv
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import time
from datetime import datetime

# Agno imports
from agno.agent import Agent
from agno.models.openai.chat import OpenAIChat

# LangChain imports for enhanced functionality
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage

# Try to import FAISS, handle import error gracefully
try:
    from langchain_community.vectorstores import FAISS
    FAISS_AVAILABLE = True
except ImportError:
    print("Warning: langchain_community not available. RAG functionality may be limited.")
    FAISS_AVAILABLE = False

# Load environment variables
load_dotenv()

# Import the existing RAG chain functionality
try:
    from rag.qa_chain import make_chain
    RAG_AVAILABLE = True
except ImportError:
    print("Warning: RAG chain not available. Some functionality may be limited.")
    RAG_AVAILABLE = False

# Initialize specialized models for different tasks
thinking_llm = ChatOpenAI(
    model="gpt-4o-mini",  # Use cheaper model for thinking
    temperature=0.3,
    streaming=True
)

main_llm = ChatOpenAI(
    model="gpt-4",  # Use full model for main responses
    temperature=0.0,
    streaming=True
)

reflection_llm = ChatOpenAI(
    model="gpt-4o-mini",  # Use cheaper model for reflection
    temperature=0.1,
    streaming=True
)

# Initialize FAISS vector store for RAG
INDEX_DIR = Path(__file__).resolve().parent / "vector_store" / "faiss"
try:
    if FAISS_AVAILABLE:
        _vectordb = FAISS.load_local(
            str(INDEX_DIR),
            OpenAIEmbeddings(),
            allow_dangerous_deserialization=True,
        )
        _retriever = _vectordb.as_retriever(search_kwargs={"k": 6})
        print("✅ RAG components initialized successfully")
    else:
        _retriever = None
        print("⚠️ FAISS not available, RAG functionality disabled")
except Exception as e:
    print(f"⚠️ Error initializing RAG components: {e}")
    _retriever = None

# Tools are optional; existing tools in this repo are Chainlit-specific and async,
# so we skip attaching them to Agno for now to avoid compatibility issues.
TOOLS_AVAILABLE = False


class QuantaraAgent(Agent):
    """
    Enhanced Quantara Financial AI Agent using Agno framework.
    Provides financial analysis, document search, and regulatory compliance assistance.
    """

    def __init__(self):
        # Load environment variables explicitly
        load_dotenv()
        
        # Get API key with validation
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize with OpenAI chat model
        model = OpenAIChat(
            id=os.getenv("OPENAI_MODEL", "gpt-4o"),
            api_key=api_key,
            temperature=0.0,
        )

        # Agent instructions
        instructions = """
        You are Quantara-AI, an advanced financial analysis and regulatory compliance assistant.
        
        **Core Capabilities:**
        • Financial Analysis: Risk calculations, portfolio metrics, beta analysis
        • Document Search: Search through regulatory docs, 10-K filings, research papers  
        • Stock Data: Real-time stock prices and charts
        • Regulatory Compliance: Basel Framework, compliance checklists, risk assessments
        
        **CRITICAL: Response Format Rules:**
        - Do NOT include any "Thinking:" or "**Thinking:**" sections in your response
        - Do NOT include any "Answer:" or "**Answer:**" headers
        - Start directly with your structured analysis
        - Your response should contain ONLY the final analysis, not your reasoning process
        
        **Response Guidelines:**
        1. Provide structured, professional responses
        2. Use clear headers and formatting
        3. Include relevant calculations and metrics
        4. Cite sources when available
        5. Offer actionable recommendations
        6. Highlight risks and considerations
        
        **Response Structure:**
        For financial queries, start directly with:
        ## Executive Summary
        ## Detailed Analysis  
        ## Key Metrics & Calculations
        ## Risk Assessment
        ## Actionable Recommendations
        
        For regulatory queries, start directly with:
        ## Regulatory Overview
        ## Key Requirements & Standards
        ## Implementation Framework
        ## Compliance Considerations
        ## Industry Best Practices
        
        Always maintain a professional tone and provide comprehensive, accurate information.
        """

        # Initialize tools (none for now to keep compatibility)
        tools = []

        super().__init__(
            model=model,
            instructions=instructions,
            tools=tools,
            markdown=True,
            debug_mode=os.getenv("DEBUG_MODE", "false").lower() == "true",
        )

        # Initialize RAG chain lazily to avoid blocking import/startup
        self.rag_chain = None
        self.retrieval_mode = os.getenv("RETRIEVAL_MODE", "hybrid")

        # Configuration
        self.config = {
            "show_thinking": os.getenv("SHOW_THINKING", "false").lower() == "true",
            "show_reflection": os.getenv("SHOW_REFLECTION", "false").lower() == "true",
            "debug_mode": os.getenv("DEBUG_MODE", "false").lower() == "true",
        }

    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze user query to determine intent and response structure."""
        query_lower = query.lower()

        # Define intent patterns
        intent_patterns = {
            "calculation": ["calculate", "compute", "what is", "value", "metrics", "ratio"],
            "analysis": ["analyze", "compare", "evaluate", "assess", "explain", "breakdown"],
            "research": ["search", "find", "lookup", "information about", "tell me about"],
            "regulatory": ["basel", "compliance", "regulation", "requirement", "framework"],
            "portfolio": ["portfolio", "diversification", "allocation", "risk management"],
            "tool_demo": ["show me", "example", "demo", "how to use"]
        }
        
        user_lower = query.lower()
        scores = {}
        
        for intent, keywords in intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in user_lower)
            if score > 0:
                scores[intent] = score
        
        primary_intent = max(scores.items(), key=lambda x: x[1])[0] if scores else "general"
        
        return {
            "primary_intent": primary_intent,
            "confidence": max(scores.values()) if scores else 0,
            "all_scores": scores,
            "requires_calculation": any(keyword in query_lower for keyword in intent_patterns["calculation"]),
            "requires_data": "stock" in query_lower or "price" in query_lower,
            "requires_regulatory": any(keyword in query_lower for keyword in intent_patterns["regulatory"]),
        }

    async def generate_thinking(self, user_input: str) -> str:
        """Generate structured thinking process for a question."""
        thinking_prompt = f"""
        Analyze this financial question systematically:
        Question: {user_input}

        Provide structured thinking:
        1. **Question Category**: [analysis/calculation/research/regulatory/strategy]
        2. **Core Concepts**: [key financial concepts involved]
        3. **Information Needed**: [data, documents, or context required]
        4. **Analytical Approach**: [methodology and framework]
        5. **Key Considerations**: [important factors and constraints]
        6. **Expected Output**: [format and depth of response needed]
        """
        
        try:
            response = await thinking_llm.ainvoke([HumanMessage(content=thinking_prompt)])
            return response.content
        except Exception as e:
            print(f"Error in thinking generation: {e}")
            return f"Error generating thinking process: {str(e)}"

    def calculate_response_quality(self, response: str, user_question: str) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics for responses."""
        metrics = {
            "length_score": min(len(response) / 500, 1.0),  # Normalize to 500 chars
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

    async def generate_reflection(self, content: str, user_question: str) -> str:
        """Generate reflection on answer quality."""
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
            reflection_response = await reflection_llm.ainvoke([HumanMessage(content=reflection_prompt)])
            return reflection_response.content
        except Exception as e:
            print(f"Error generating reflection: {e}")
            return f"Error during reflection: {str(e)}"

    def enhance_query_with_context(self, query: str, intent: Dict[str, Any]) -> str:
        """Enhance user query with context and structure for better responses."""
        
        if intent["primary_intent"] == "regulatory":
            enhanced_query = f"""
            **Regulatory Query Analysis**
            Question: {query}
            Intent: {intent['primary_intent']} (confidence: {intent['confidence']})
            
            Please provide a structured regulatory analysis following this format:
            
            ## Regulatory Overview
            [Brief context and scope]
            
            ## Key Requirements & Standards  
            [Specific regulatory requirements with citations]
            
            ## Implementation Framework
            [Step-by-step implementation approach]
            
            ## Compliance Considerations
            [Risk factors and mitigation strategies]
            
            ## Industry Best Practices
            [Proven approaches and recommendations]
            
            Use professional formatting with clear headers and actionable insights.
            """
        else:
            enhanced_query = f"""
            **Financial Analysis Request**
            Query Intent: {intent['primary_intent']} (confidence: {intent['confidence']})
            Question: {query}
            
            Please provide a comprehensive analysis following this format:
            
            ## Executive Summary
            [Key takeaways and conclusions]
            
            ## Detailed Analysis
            [In-depth examination with data and evidence]
            
            ## Key Metrics & Calculations
            [Relevant financial ratios, calculations, or quantitative analysis]
            
            ## Risk Assessment
            [Potential risks and considerations]
            
            ## Actionable Recommendations
            [Specific, implementable advice]
            
            Structure your response professionally with clear formatting and cite sources when available.
            """

        return enhanced_query

    async def run_with_rag_and_thinking(self, message: str, show_thinking: bool = False, show_reflection: bool = False) -> Dict[str, Any]:
        """Enhanced run method with thinking, RAG, and reflection capabilities."""
        start_time = time.time()
        
        # Analyze query intent
        intent = self.analyze_query_intent(message)
        
        # Generate thinking process if requested
        thinking = None
        if show_thinking:
            thinking = await self.generate_thinking(message)
        
        # Lazy init the RAG chain on first use
        if self.rag_chain is None and RAG_AVAILABLE:
            try:
                print("Initializing RAG chain (lazy)...")
                self.rag_chain = make_chain(k=6, retrieval_mode=self.retrieval_mode)
                print("✅ RAG chain initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize RAG chain: {e}")
                self.rag_chain = None

        # Generate the main response
        try:
            if self.rag_chain and _retriever:
                # Use RAG for document-based responses
                enhanced_query = self.enhance_query_with_context(message, intent)
                
                # Run RAG chain synchronously (it's not async)
                rag_response = self.rag_chain.invoke({
                    "input": enhanced_query,
                    "chat_history": [],
                })
                
                answer = rag_response.get("answer", "")
                sources = rag_response.get("context", [])
                
                # Format response with sources
                formatted_response = answer
                
                # Add source citations if available
                if sources:
                    unique_sources = set()
                    for doc in sources:
                        if hasattr(doc, "metadata") and "source" in doc.metadata:
                            unique_sources.add(Path(doc.metadata["source"]).name)

                    if unique_sources:
                        formatted_response += "\n\n" + "─" * 50 + "\n**📚 Sources & References**\n"
                        for i, source in enumerate(sorted(unique_sources), 1):
                            formatted_response += f"{i}. {source}\n"
                
                source_list = list(unique_sources) if sources else []
            else:
                # Fallback to direct agent response
                enhanced_query = self.enhance_query_with_context(message, intent)
                agent_response = super().run(enhanced_query)
                formatted_response = agent_response.content if hasattr(agent_response, 'content') else str(agent_response)
                source_list = []
            
        except Exception as e:
            print(f"Error in response generation: {e}")
            formatted_response = f"❌ An error occurred while generating the response: {str(e)}"
            source_list = []
        
        # Calculate response quality
        quality_metrics = self.calculate_response_quality(formatted_response, message)
        
        # Generate reflection if requested
        reflection = None
        if show_reflection:
            reflection = await self.generate_reflection(formatted_response, message)
        
        # Calculate total processing time
        processing_time = time.time() - start_time
        
        # Add debug info if enabled
        if self.config["debug_mode"]:
            debug_info = f"\n\n*Debug Info: Processed in {processing_time:.2f}s | Intent: {intent['primary_intent']} | Quality Score: {quality_metrics['overall_score']:.1f}/10*"
            formatted_response += debug_info
        
        return {
            "content": formatted_response,
            "thinking": thinking,
            "reflection": reflection,
            "quality_score": quality_metrics["overall_score"],
            "quality_metrics": quality_metrics,
            "sources": source_list,
            "intent_analysis": intent,
            "processing_time": processing_time
        }

    def run_with_rag(self, message: str) -> str:
        """Legacy method - kept for backward compatibility."""
        # Create new event loop for async call
        import asyncio
        
        try:
            # Check if there's already an event loop running
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, we need to create a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._run_with_rag_sync(message))
                    result = future.result()
            else:
                # No loop running, we can use asyncio.run
                result = asyncio.run(self._run_with_rag_sync(message))
        except:
            # Fallback to simple sync method
            result = self._run_with_rag_fallback(message)
        
        return result["content"] if isinstance(result, dict) else str(result)
    
    async def _run_with_rag_sync(self, message: str) -> Dict[str, Any]:
        """Helper async method for sync wrapper."""
        return await self.run_with_rag_and_thinking(message, show_thinking=False, show_reflection=False)
    
    def _run_with_rag_fallback(self, message: str) -> Dict[str, Any]:
        """Fallback sync method for RAG."""
        try:
            # Lazy init the RAG chain on first use
            if self.rag_chain is None and RAG_AVAILABLE:
                try:
                    print("Initializing RAG chain (lazy)...")
                    self.rag_chain = make_chain(k=6, retrieval_mode=self.retrieval_mode)
                    print("✅ RAG chain initialized successfully")
                except Exception as e:
                    print(f"Warning: Could not initialize RAG chain: {e}")
                    self.rag_chain = None

            if not self.rag_chain:
                response = self.run(message)
                return {"content": response.content if hasattr(response, "content") else str(response)}

            # Use RAG chain for document-based responses
            intent = self.analyze_query_intent(message)
            enhanced_query = self.enhance_query_with_context(message, intent)
            
            rag_response = self.rag_chain.invoke({
                "input": enhanced_query,
                "chat_history": [],
            })

            answer = rag_response.get("answer", "")
            sources = rag_response.get("context", [])

            # Format response with sources
            formatted_response = answer

            # Add source citations if available
            if sources:
                unique_sources = set()
                for doc in sources:
                    if hasattr(doc, "metadata") and "source" in doc.metadata:
                        unique_sources.add(Path(doc.metadata["source"]).name)

                if unique_sources:
                    formatted_response += "\n\n" + "─" * 50 + "\n**📚 Sources & References**\n"
                    for i, source in enumerate(sorted(unique_sources), 1):
                        formatted_response += f"{i}. {source}\n"

            return {"content": formatted_response}

        except Exception as e:
            print(f"Error in RAG processing: {e}")
            # Fallback to regular agent response
            response = self.run(message)
            return {"content": response.content if hasattr(response, "content") else str(response)}

    def run(self, message: str, **kwargs) -> Any:
        """Override run method to add custom processing."""
        try:
            # Analyze intent for better responses
            intent = self.analyze_query_intent(message)
            enhanced_message = self.enhance_query_with_context(message, intent)

            # Add timestamp and session info
            session_info = f"\n\n*Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"

            # Run the agent
            response = super().run(enhanced_message, **kwargs)

            # Add session info if debug mode is enabled
            if self.config["debug_mode"] and hasattr(response, "content"):
                response.content += session_info

            return response

        except Exception as e:
            print(f"Error in agent run: {e}")
            # Return error response in expected format
            class ErrorResponse:
                def __init__(self, content):
                    self.content = content

            return ErrorResponse(f"❌ An error occurred while processing your request: {str(e)}")


# Create the global agent instance (lazy initialization)
quantara_agent = None

def _ensure_agent():
    """Ensure the agent is initialized."""
    global quantara_agent
    if quantara_agent is None:
        quantara_agent = QuantaraAgent()
    return quantara_agent


# Enhanced function to get agent response with thinking capabilities
async def get_agent_response_enhanced(
    message: str, 
    use_rag: bool = True, 
    show_thinking: bool = False, 
    show_reflection: bool = False
) -> Dict[str, Any]:
    """Get enhanced response from Quantara agent with thinking and reflection."""
    try:
        agent = _ensure_agent()
        
        if use_rag:
            result = await agent.run_with_rag_and_thinking(
                message, 
                show_thinking=show_thinking, 
                show_reflection=show_reflection
            )
        else:
            # Direct agent call without RAG
            intent = agent.analyze_query_intent(message)
            enhanced_query = agent.enhance_query_with_context(message, intent)
            
            thinking = None
            if show_thinking:
                thinking = await agent.generate_thinking(message)
            
            response = agent.run(enhanced_query)
            content = response.content if hasattr(response, "content") else str(response)
            
            quality_metrics = agent.calculate_response_quality(content, message)
            
            reflection = None
            if show_reflection:
                reflection = await agent.generate_reflection(content, message)
            
            result = {
                "content": content,
                "thinking": thinking,
                "reflection": reflection,
                "quality_score": quality_metrics["overall_score"],
                "quality_metrics": quality_metrics,
                "sources": [],
                "intent_analysis": intent,
                "processing_time": 0.0
            }
        
        return result
        
    except Exception as e:
        return {
            "content": f"❌ Error: {str(e)}",
            "thinking": None,
            "reflection": None,
            "quality_score": 0.0,
            "quality_metrics": {},
            "sources": [],
            "intent_analysis": {},
            "processing_time": 0.0
        }


# Legacy function for backward compatibility
def get_agent_response(message: str, use_rag: bool = True) -> str:
    """Get response from Quantara agent (legacy interface)."""
    try:
        agent = _ensure_agent()
        if use_rag:
            response = agent._run_with_rag_fallback(message)
            return response["content"] if isinstance(response, dict) else str(response)
        else:
            response = agent.run(message)
            return response.content if hasattr(response, "content") else str(response)
        
    except Exception as e:
        print(f"Error in get_agent_response: {e}")
        return f"❌ Error: {str(e)}"


if __name__ == "__main__":
    # Test the enhanced agent
    import asyncio
    
    async def test_agent():
        print("Testing Enhanced Quantara Agent...")
        test_message = "What are the key risk factors I should consider when analyzing a tech stock portfolio?"
        
        # Test with thinking and reflection
        result = await get_agent_response_enhanced(
            test_message, 
            use_rag=True, 
            show_thinking=True, 
            show_reflection=True
        )
        
        print("\n=== THINKING ===")
        print(result.get("thinking", "No thinking generated"))
        
        print("\n=== RESPONSE ===")
        print(result.get("content", "No content generated"))
        
        print("\n=== REFLECTION ===")
        print(result.get("reflection", "No reflection generated"))
        
        print(f"\n=== METRICS ===")
        print(f"Quality Score: {result.get('quality_score', 0):.1f}/10")
        print(f"Processing Time: {result.get('processing_time', 0):.2f}s")
        print(f"Intent: {result.get('intent_analysis', {}).get('primary_intent', 'unknown')}")
    
    asyncio.run(test_agent())
