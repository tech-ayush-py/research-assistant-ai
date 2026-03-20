"""
ChromaDB vector store for research paper RAG.
Handles ingestion, embedding, and retrieval of academic papers.
"""
import hashlib
import logging
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config.settings import CHROMA_DIR, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_embedder: SentenceTransformer | None = None
_client: chromadb.PersistentClient | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _embedder


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def get_collection(name: str = "research_papers"):
    client = _get_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_papers(papers: List[Dict[str, Any]], collection_name: str = "research_papers") -> int:
    """
    Embed and store papers in ChromaDB.
    Each paper dict: {title, abstract, authors, year, url, source, paper_id}
    Returns number of new papers added.
    """
    collection = get_collection(collection_name)
    embedder = _get_embedder()

    texts, ids, metadatas = [], [], []
    for p in papers:
        doc_text = f"{p.get('title', '')}. {p.get('abstract', '')}"
        doc_id = hashlib.md5(doc_text.encode()).hexdigest()

        # skip duplicates
        existing = collection.get(ids=[doc_id])
        if existing["ids"]:
            continue

        texts.append(doc_text)
        ids.append(doc_id)
        metadatas.append({
            "title":                     p.get("title", ""),
            "authors":                   ", ".join(p.get("authors", [])),
            "year":                      str(p.get("year", "")),
            "url":                       p.get("url", ""),
            "source":                    p.get("source", ""),
            "paper_id":                  p.get("paper_id", ""),
            "citation_count":            str(p.get("citation_count", 0) or 0),
            "influential_citation_count": str(p.get("influential_citation_count", 0) or 0),
            "journal":                   p.get("journal", ""),
        })

    if not texts:
        return 0

    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
    collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)
    logger.info("Ingested %d new papers into collection '%s'", len(texts), collection_name)
    return len(texts)


def similarity_search(
    query: str,
    n_results: int = 10,
    collection_name: str = "research_papers",
) -> List[Dict[str, Any]]:
    """Return top-k similar papers with metadata and distance score."""
    collection = get_collection(collection_name)
    embedder = _get_embedder()

    q_embed = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=q_embed,
        n_results=min(n_results, collection.count() or 1),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({**meta, "snippet": doc[:300], "similarity": round(1 - dist, 4)})
    return output


def reset_collection(collection_name: str = "research_papers") -> None:
    """
    Delete and recreate the collection, clearing all previously ingested papers.
    Called at the start of each pipeline run to prevent papers from a previous
    topic (e.g. swarm robotics) from contaminating results for a new topic
    (e.g. women in espionage history).
    """
    client = _get_client()
    try:
        client.delete_collection(name=collection_name)
        logger.info("Cleared collection '%s' for fresh pipeline run.", collection_name)
    except Exception:
        pass  # collection may not exist yet on first run
    client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def collection_stats(collection_name: str = "research_papers") -> Dict[str, Any]:
    col = get_collection(collection_name)
    return {"collection": collection_name, "total_papers": col.count()}
