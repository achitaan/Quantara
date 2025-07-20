"""Factory that returns a Conversational RAG chain using the custom Quantara prompt."""

from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate

# ── Load system prompt ─────────────────────────────────────────────────────────
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system_quantara.md"
SYSTEM = PROMPT_PATH.read_text(encoding='utf-8')

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        ("human", "Question: {question}"),
        ("human", "Context:\n{context}"),
    ]
)

# ── Vector store / retriever ──────────────────────────────────────────────────
INDEX_DIR = Path(__file__).resolve().parent.parent / "vector_store" / "faiss"


def make_retriever(k: int = 6):
    db = FAISS.load_local(
        str(INDEX_DIR),
        OpenAIEmbeddings(),
        allow_dangerous_deserialization=True,
    )
    return db.as_retriever(search_type="mmr", search_kwargs={"k": k})


# ── Factory ───────────────────────────────────────────────────────────────────
def make_chain(k: int = 6):
    llm = ChatOpenAI(model="gpt-4", temperature=0, streaming=True)
    retriever = make_retriever(k)

    return ConversationalRetrievalChain.from_llm(
        llm,
        retriever,
        return_source_documents=True,
        verbose=False,
    )
