"""
Run once to build the FAISS index used by your RAG retriever.

Steps
-----
1. Load every .txt / .md / .pdf inside backend/data/
2. Split docs into 1 kB chunks with 128-token overlap
3. Embed each chunk with OpenAI embeddings
4. Save vector index to backend/vector_store/faiss/
"""

from pathlib import Path
from dotenv import load_dotenv

# ─── Load env vars (OPENAI_API_KEY, etc.) ───────────────────────────────────────
load_dotenv()

# ─── LangChain imports (v0.2+ namespaces) ──────────────────────────────────────
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ─── Paths relative to this file (cross-platform) ──────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent  
DATA_DIR  = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "vector_store" / "faiss" 

# ------------------------------------------------------------------------------
def load_documents() -> list:
    """Load .txt/.md and .pdf documents from DATA_DIR (recursive)."""
    docs = []

    # Text / Markdown
    docs += DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.[tm][dx]",        # *.txt, *.md
        loader_cls=TextLoader,
        show_progress=True,
    ).load()

    # PDFs
    docs += DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    ).load()

    return docs


def main() -> None:
    print("→ Loading documents")
    documents = load_documents()
    print(f"  Loaded {len(documents)} docs. Splitting …")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=128,
    )
    chunks = splitter.split_documents(documents)
    print(f"  {len(chunks)} chunks. Embedding …")

    embeddings = OpenAIEmbeddings()  # uses OPENAI_API_KEY from .env
    vectordb = FAISS.from_documents(chunks, embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectordb.save_local(str(INDEX_DIR))
    print(f"✓ Saved index to {INDEX_DIR.resolve()}")


if __name__ == "__main__":
    main()
