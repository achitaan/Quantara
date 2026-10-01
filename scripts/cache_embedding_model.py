"""Explicit, one-time weight download; ordinary RAG inference stays offline."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from quantara.config import settings
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    model = SentenceTransformer(settings.embedding_model, device="cpu")
    print(f"Cached {settings.embedding_model}; dimension={model.get_sentence_embedding_dimension()}")
    print("Rebuild the local index in Assistant after importing documents.")
