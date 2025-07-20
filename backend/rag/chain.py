# backend/rag/chain.py
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

INDEX_PATH: Path = Path(__file__).resolve().parent.parent / "vector_store" / "faiss"

def make_rag_chain() -> ConversationalRetrievalChain:
    """Return a streaming Conversational RAG chain ready for Chainlit."""
    db        = FAISS.load_local(str(INDEX_PATH), OpenAIEmbeddings())
    retriever = db.as_retriever(search_kwargs={"k": 6})

    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
        streaming=True,
    )

    return ConversationalRetrievalChain.from_llm(
        llm,
        retriever,
        return_source_documents=True,
        verbose=True,
    )
