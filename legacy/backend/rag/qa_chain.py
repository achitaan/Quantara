"""Factory that returns a Conversational RAG chain using the custom Quantara prompt."""

from pathlib import Path
from dotenv import load_dotenv
import pickle

# Load environment variables
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever

# Try to import BM25 - it might not be available in all environments
try:
    from langchain_community.retrievers import BM25Retriever
    BM25_AVAILABLE = True
except ImportError:
    print("BM25Retriever not available - using dense retrieval only")
    BM25_AVAILABLE = False

# Try to import reranker - it might not be available in all environments
try:
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    print("CrossEncoderReranker not available - skipping reranking")
    RERANKER_AVAILABLE = False

# ── Load system prompt ─────────────────────────────────────────────────────────
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system_quantara.md"
SYSTEM = PROMPT_PATH.read_text(encoding='utf-8')

# Contextualize question prompt for history-aware retrieval
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Answer question prompt with custom system prompt
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "Question: {input}\n\nContext:\n{context}"),
    ]
)

# ── Vector store / retriever ──────────────────────────────────────────────────
INDEX_DIR = Path(__file__).resolve().parent.parent / "vector_store" / "faiss"


def make_retriever(k: int = 6):
    """Create basic MMR retriever."""
    db = FAISS.load_local(
        str(INDEX_DIR),
        OpenAIEmbeddings(),
        allow_dangerous_deserialization=True,
    )
    return db.as_retriever(search_type="mmr", search_kwargs={"k": k})

def make_hybrid_retriever(k: int = 6):
    """Create hybrid retriever combining FAISS (dense) and BM25 (sparse)."""
    if not BM25_AVAILABLE:
        print("BM25 not available, falling back to dense retrieval")
        return make_retriever(k)
    
    # Dense retriever (existing)
    db = FAISS.load_local(
        str(INDEX_DIR),
        OpenAIEmbeddings(),
        allow_dangerous_deserialization=True,
    )
    dense_retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": k})
    
    # BM25 sparse retriever
    bm25_path = INDEX_DIR.parent / "bm25_retriever.pkl"
    
    try:
        if bm25_path.exists():
            with open(bm25_path, 'rb') as f:
                bm25_retriever = pickle.load(f)
                bm25_retriever.k = k
        else:
            # Create BM25 from documents if not exists
            docs = list(db.docstore._dict.values())
            texts = [doc.page_content for doc in docs]
            bm25_retriever = BM25Retriever.from_texts(texts)
            bm25_retriever.k = k
            
            # Save for future use
            with open(bm25_path, 'wb') as f:
                pickle.dump(bm25_retriever, f)
        
        # Combine retrievers (70% dense, 30% sparse)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )
        
        return ensemble_retriever
        
    except Exception as e:
        print(f"Error creating hybrid retriever: {e}")
        return dense_retriever

def make_reranked_retriever(k: int = 6, top_k_rerank: int = 3):
    """Add reranking to improve retrieval quality."""
    if not RERANKER_AVAILABLE:
        print("Reranker not available, using hybrid retrieval")
        return make_hybrid_retriever(k)
    
    try:
        base_retriever = make_hybrid_retriever(k * 2)  # Get more docs first
        
        # Initialize cross-encoder for reranking
        model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        compressor = CrossEncoderReranker(model=model, top_k=top_k_rerank)
        
        # Wrap with compression
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        return compression_retriever
        
    except Exception as e:
        print(f"Error creating reranked retriever: {e}")
        return make_hybrid_retriever(k)

def make_compressed_retriever(k: int = 6):
    """Add context compression to reduce tokens and improve relevance."""
    try:
        base_retriever = make_reranked_retriever(k)
        
        # Use cheaper model for compression
        compressor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        compressor = LLMChainExtractor.from_llm(compressor_llm)
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        return compression_retriever
        
    except Exception as e:
        print(f"Error creating compressed retriever: {e}")
        return make_reranked_retriever(k)


# ── Factory ───────────────────────────────────────────────────────────────────
def make_chain(k: int = 6, retrieval_mode: str = "hybrid"):
    """
    Enhanced chain factory with multiple retrieval modes.
    
    Args:
        k: Number of documents to retrieve
        retrieval_mode: 'basic', 'hybrid', 'rerank', 'compressed'
    """
    llm = ChatOpenAI(model="gpt-4", temperature=0, streaming=True)
    
    # Select retriever based on mode
    print(f"Creating chain with retrieval mode: {retrieval_mode}")
    
    if retrieval_mode == "hybrid":
        retriever = make_hybrid_retriever(k)
    elif retrieval_mode == "rerank":
        retriever = make_reranked_retriever(k)
    elif retrieval_mode == "compressed":
        retriever = make_compressed_retriever(k)
    else:  # basic
        retriever = make_retriever(k)
    
    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # Enhanced QA prompt with CoT
    enhanced_qa_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""{SYSTEM}

When answering, follow this process:
1. First, analyze the question and context carefully
2. Think through the key points step by step
3. Provide a comprehensive, well-structured answer
4. Include relevant examples where helpful
5. Always cite your sources

Remember to be precise with financial terminology and calculations."""),
        MessagesPlaceholder("chat_history"),
        ("human", """Question: {input}

Context:
{context}

Please provide a thorough analysis following the process outlined above."""),
    ])
    
    # Create document chain with enhanced prompt
    question_answer_chain = create_stuff_documents_chain(llm, enhanced_qa_prompt)
    
    # Create retrieval chain
    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return chain
