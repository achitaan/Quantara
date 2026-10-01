from hashlib import sha256
import json
from pathlib import Path
import re
from uuid import uuid4
from functools import lru_cache

import numpy as np

from .model_runtime import serialized_training


@lru_cache(maxsize=2)
@serialized_training
def make_encoder(model):
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(model, local_files_only=True, device="cpu")
    except OSError as exc:
        raise ValueError("Local embedding weights are unavailable. Download the configured embedding model once; text search works before indexing.") from exc


def chunks(documents):
    result = []
    for document in documents:
        words = document["text"].split()
        for offset in range(0, len(words), 180):
            result.append(
                {
                    "document_id": document["id"],
                    "name": document["name"],
                    "page": document.get("page", 1),
                    "text": " ".join(words[offset : offset + 220]),
                    "citation": f"doc:{document['id']}:p{document.get('page', 1)}",
                }
            )
    return result


class Retrieval:
    def __init__(self, runtime, model):
        self.runtime, self.model = Path(runtime), model

    def directory(self, owner):
        return self.runtime / "indexes" / sha256(owner.encode()).hexdigest()

    def rebuild(self, owner, documents):
        import faiss

        parts = chunks(documents)
        if not parts:
            raise ValueError("No documents to embed")
        encoder = make_encoder(self.model)
        vectors = encoder.encode(
            [p["text"] for p in parts], normalize_embeddings=True
        ).astype(np.float32)
        revision = getattr(encoder[0].auto_model.config, "_commit_hash", None)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        directory = self.directory(owner)
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": 1,
            "model": self.model,
            "revision": revision,
            "dimension": vectors.shape[1],
            "documents_hash": sha256(
                json.dumps(parts, sort_keys=True).encode()
            ).hexdigest(),
            "chunks": parts,
            "index_file": "index-" + uuid4().hex + ".faiss",
        }
        # Publish one pointer to an immutable generation. Readers never see a mixed pair.
        faiss.write_index(index, str(directory / metadata["index_file"]))
        temporary = directory / ("metadata-" + uuid4().hex + ".tmp")
        temporary.write_text(json.dumps(metadata), encoding="utf-8")
        temporary.replace(directory / "metadata.json")
        return {k: v for k, v in metadata.items() if k != "chunks"}

    def search(self, owner, documents, query, limit=4):
        parts = chunks(documents)
        if not parts:
            return {"mode": "empty", "sources": []}
        directory = self.directory(owner)
        if (directory / "metadata.json").exists():
            metadata = json.loads(
                (directory / "metadata.json").read_text(encoding="utf-8")
            )
            if metadata["model"] != self.model or metadata["schema_version"] != 1:
                raise ValueError(
                    "Embedding index is incompatible. Rebuild it using the configured model."
                )
            digest = sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()
            if digest != metadata["documents_hash"]:
                raise ValueError(
                    "Documents changed. Rebuild the local embedding index."
                )
            import faiss

            encoder = make_encoder(self.model)
            if (
                getattr(encoder[0].auto_model.config, "_commit_hash", None)
                != metadata["revision"]
            ):
                raise ValueError("Embedding model revision changed; rebuild the index")
            vector = encoder.encode([query], normalize_embeddings=True).astype(
                np.float32
            )
            filename = metadata.get("index_file", "index.faiss")
            if not re.fullmatch(r"index(?:-[a-f0-9]{32})?\.faiss", filename):
                raise ValueError("Invalid embedding index filename")
            index = faiss.read_index(str(directory / filename))
            if index.d != vector.shape[1] or index.ntotal != len(parts):
                raise ValueError("Invalid embedding index dimensions")
            distances, indices = index.search(vector, min(limit, len(parts)))
            return {
                "mode": "local_embeddings",
                "sources": [
                    parts[i] | {"score": float(score)}
                    for i, score in zip(indices[0], distances[0])
                    if i >= 0
                ],
            }
        tokens = set(re.findall(r"\w+", query.lower()))
        ranked = sorted(
            parts,
            key=lambda p: len(tokens & set(re.findall(r"\w+", p["text"].lower()))),
            reverse=True,
        )
        return {
            "mode": "lexical",
            "sources": [
                p
                for p in ranked[:limit]
                if tokens & set(re.findall(r"\w+", p["text"].lower()))
            ],
        }
