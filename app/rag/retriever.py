"""Vector store indexing and hybrid retrieval pipeline using Qdrant and lightweight in-memory BM25."""

import math
import re
import threading
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import settings
from app.models.schemas import RetrievedChunk, RetrievalResult, SemanticChunk
from app.rag.embeddings import EmbeddingService, embedding_service


def generate_chunk_uuid(chunk_id: str) -> str:
    """Derive a deterministic UUID5 from a chunk identifier.

    Guarantees idempotency: the same chunk_id always yields the exact same UUID.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


class PurePythonBM25:
    """Lightweight in-memory Okapi BM25 index over semantic chunks."""

    def __init__(self, chunks: List[SemanticChunk], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len: Dict[str, int] = {}
        self.term_freqs: Dict[str, Counter] = {}
        self.doc_freqs: Counter = Counter()
        self.num_docs = len(chunks)
        total_len = 0

        for c in chunks:
            aliases_str = " ".join(c.metadata.get("aliases", [])) if isinstance(c.metadata, dict) else ""
            text = f"{c.name} {aliases_str} {c.vulnerability_id} {c.section} {c.content}"
            tokens = re.findall(r"[a-zA-Z0-9_\-]+", text.lower())
            doc_len = len(tokens)
            total_len += doc_len
            self.doc_len[c.chunk_id] = doc_len
            tf = Counter(tokens)
            self.term_freqs[c.chunk_id] = tf
            for t in tf:
                self.doc_freqs[t] += 1

        self.avg_doc_len = total_len / self.num_docs if self.num_docs else 1.0
        self.idf: Dict[str, float] = {}
        for t, df in self.doc_freqs.items():
            self.idf[t] = math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> Dict[str, float]:
        """Compute BM25 score for a query against all indexed documents."""
        tokens = re.findall(r"[a-zA-Z0-9_\-]+", query.lower())
        scores: Dict[str, float] = defaultdict(float)
        for t in tokens:
            if t not in self.idf:
                continue
            idf_val = self.idf[t]
            for doc_id, tf_map in self.term_freqs.items():
                if t in tf_map:
                    tf = tf_map[t]
                    num = tf * (self.k1 + 1.0)
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_len[doc_id] / self.avg_doc_len))
                    scores[doc_id] += idf_val * (num / denom)
        return scores


class QdrantVectorStore:
    """Manages local Qdrant collection, indexing, and vector similarity retrieval.

    Enforces singleton instance reuse per filesystem path to prevent SQLite/portalocker
    lock collisions when local embedded Qdrant is accessed across multiple components.
    """

    _instances: Dict[str, "QdrantVectorStore"] = {}
    _lock = threading.Lock()

    def __new__(
        cls,
        collection_name: Optional[str] = None,
        storage_path: Optional[Union[str, Path]] = None,
        client: Optional[QdrantClient] = None,
        embedder: Optional[EmbeddingService] = None,
        enable_hybrid: Optional[bool] = None,
        hybrid_weight: Optional[float] = None,
        bm25: Optional[PurePythonBM25] = None,
    ) -> "QdrantVectorStore":
        path_str = str(storage_path or settings.QDRANT_STORAGE_PATH)
        # For :memory: or when an explicit custom client is passed, do not cache
        if path_str == ":memory:" or client is not None:
            instance = super().__new__(cls)
            instance._init_lock = threading.Lock()
            instance._initialized = False
            return instance

        norm_path = str(Path(path_str).resolve())
        with cls._lock:
            if norm_path in cls._instances:
                return cls._instances[norm_path]
            instance = super().__new__(cls)
            instance._init_lock = threading.Lock()
            instance._initialized = False
            cls._instances[norm_path] = instance
            return instance

    def __init__(
        self,
        collection_name: Optional[str] = None,
        storage_path: Optional[Union[str, Path]] = None,
        client: Optional[QdrantClient] = None,
        embedder: Optional[EmbeddingService] = None,
        enable_hybrid: Optional[bool] = None,
        hybrid_weight: Optional[float] = None,
        bm25: Optional[PurePythonBM25] = None,
    ):
        init_lock = getattr(self, "_init_lock", None)
        if init_lock is None:
            init_lock = threading.Lock()
            self._init_lock = init_lock

        with init_lock:
            if getattr(self, "_initialized", False):
                return
            self._storage_path_str = str(storage_path or settings.QDRANT_STORAGE_PATH)
            self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
            self.embedder = embedder or embedding_service
            self._external_client = client is not None
            self.enable_hybrid = settings.ENABLE_HYBRID_RETRIEVAL if enable_hybrid is None else enable_hybrid
            self.hybrid_weight = settings.HYBRID_BM25_WEIGHT if hybrid_weight is None else hybrid_weight
            self._bm25 = bm25

            try:
                if client is not None:
                    self.client = client
                else:
                    path_str = self._storage_path_str
                    if path_str != ":memory:":
                        Path(path_str).mkdir(parents=True, exist_ok=True)
                    self.client = QdrantClient(path=path_str)

                self._ensure_collection_exists()
                self._initialized = True
            except Exception:
                if hasattr(self, "_storage_path_str") and self._storage_path_str != ":memory:":
                    norm_path = str(Path(self._storage_path_str).resolve())
                    with self._lock:
                        self._instances.pop(norm_path, None)
                raise

    @property
    def bm25(self) -> PurePythonBM25:
        """Lazy load or build the in-memory BM25 index once from Master KB chunks."""
        if self._bm25 is None:
            init_lock = getattr(self, "_init_lock", None)
            if init_lock is not None:
                with init_lock:
                    if self._bm25 is None:
                        self._build_bm25()
            else:
                if self._bm25 is None:
                    self._build_bm25()
        return self._bm25

    def _build_bm25(self) -> None:
        from app.rag.chunking import chunk_vulnerabilities
        from app.rag.ingestion import ingest_master_kb

        records, _ = ingest_master_kb(settings.MASTER_KB_PATH)
        chunks, _ = chunk_vulnerabilities(records)
        self._bm25 = PurePythonBM25(chunks)

    def _ensure_collection_exists(self) -> None:
        """Create the collection if it does not already exist."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedder.dimension,
                    distance=Distance.COSINE,
                ),
            )

    def close(self) -> None:
        """Close the underlying client connection if owned."""
        init_lock = getattr(self, "_init_lock", None)
        if init_lock is not None:
            with init_lock:
                self._perform_close()
        else:
            self._perform_close()

    def _perform_close(self) -> None:
        if hasattr(self, "_storage_path_str") and self._storage_path_str != ":memory:":
            norm_path = str(Path(self._storage_path_str).resolve())
            with self._lock:
                self._instances.pop(norm_path, None)
        if hasattr(self, "client") and hasattr(self.client, "close") and not self._external_client:
            self.client.close()
        self._initialized = False

    def count_points(self) -> int:
        """Return the current number of points in the collection."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count or 0

    def index_chunks(
        self,
        chunks: List[SemanticChunk],
        batch_size: int = 32,
        recreate: bool = False,
        show_progress: bool = False,
    ) -> Dict[str, Any]:
        """Deterministically index or upsert chunks into the Qdrant collection.

        Args:
            chunks: List of SemanticChunk objects to index.
            batch_size: Embedding and upsert batch size.
            recreate: If True, drops and recreates the collection before indexing.
            show_progress: If True, displays embedding progress bar.

        Returns:
            Dictionary with indexing summary statistics.
        """
        if recreate and self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
            self._ensure_collection_exists()

        if not chunks:
            return {
                "indexed_chunks": 0,
                "total_points": self.count_points(),
                "collection_name": self.collection_name,
            }

        total_chunks = len(chunks)
        indexed_count = 0

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            contents = [c.content for c in batch]
            vectors = self.embedder.embed_texts(
                contents,
                batch_size=len(batch),
                show_progress=show_progress,
            )

            points: List[PointStruct] = []
            for chunk, vector in zip(batch, vectors):
                point_id = generate_chunk_uuid(chunk.chunk_id)
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "vulnerability_id": chunk.vulnerability_id,
                    "name": chunk.name,
                    "category": chunk.category,
                    "section": chunk.section,
                    "severity_level": chunk.severity_level,
                    "cwe": chunk.cwe,
                    "owasp": chunk.owasp,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                }
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            indexed_count += len(batch)

        # Update cached BM25 index if new chunks were indexed
        if self._bm25 is not None or self.enable_hybrid:
            self._bm25 = PurePythonBM25(chunks)

        return {
            "indexed_chunks": indexed_count,
            "total_points": self.count_points(),
            "collection_name": self.collection_name,
        }

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_vulnerability_id: Optional[str] = None,
        filter_section: Optional[str] = None,
    ) -> RetrievalResult:
        """Search the vector store for the most relevant semantic chunks.

        When enable_hybrid is True, combines dense semantic search with lexical
        BM25 score fusion using the configured hybrid weight.

        Args:
            query: The user query string. Must not be empty.
            top_k: Number of chunks to retrieve (default: settings.TOP_K_RETRIEVAL).
            score_threshold: Optional minimum score threshold.
            filter_vulnerability_id: Optional exact match filter for vulnerability_id.
            filter_section: Optional exact match filter for section.

        Returns:
            RetrievalResult containing matches with scores, latency decomposition, and metadata.

        Raises:
            ValueError: if query is empty or whitespace-only.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace-only.")

        k = top_k or settings.TOP_K_RETRIEVAL

        # 1. Embed query
        t0 = time.perf_counter()
        query_vector = self.embedder.embed_text(query.strip())
        t1 = time.perf_counter()
        embed_ms = (t1 - t0) * 1000.0

        # Build Qdrant query filter if requested
        query_filter: Optional[Filter] = None
        conditions = []
        if filter_vulnerability_id:
            conditions.append(
                FieldCondition(
                    key="vulnerability_id",
                    match=MatchValue(value=filter_vulnerability_id),
                )
            )
        if filter_section:
            conditions.append(
                FieldCondition(
                    key="section",
                    match=MatchValue(value=filter_section),
                )
            )
        if conditions:
            query_filter = Filter(must=conditions)

        # 2. Vector search & Hybrid scoring
        use_hybrid = self.enable_hybrid and (self.hybrid_weight > 0.0)
        candidate_limit = max(k * 5, 25) if use_hybrid else k

        search_response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=candidate_limit,
            query_filter=query_filter,
        )

        matches: List[RetrievedChunk] = []

        if use_hybrid and search_response.points:
            bm25_scores = self.bm25.score(query.strip())
            max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0

            scored_candidates = []
            for pt in search_response.points:
                payload = pt.payload or {}
                cid = payload.get("chunk_id", "")
                dense_score = float(pt.score)
                lexical_score = (bm25_scores.get(cid, 0.0) / max_bm25) if max_bm25 > 0.0 else 0.0
                fused_score = (1.0 - self.hybrid_weight) * dense_score + self.hybrid_weight * lexical_score

                if score_threshold is not None and fused_score < score_threshold:
                    continue

                chunk_obj = RetrievedChunk(
                    chunk_id=payload.get("chunk_id", ""),
                    vulnerability_id=payload.get("vulnerability_id", ""),
                    name=payload.get("name", ""),
                    category=payload.get("category", ""),
                    section=payload.get("section", ""),
                    severity_level=payload.get("severity_level", ""),
                    content=payload.get("content", ""),
                    score=round(fused_score, 4),
                    cwe=payload.get("cwe", []),
                    owasp=payload.get("owasp", []),
                    metadata=payload.get("metadata", {}),
                )
                scored_candidates.append(chunk_obj)

            scored_candidates.sort(key=lambda x: x.score, reverse=True)
            matches = scored_candidates[:k]
        else:
            for pt in search_response.points:
                payload = pt.payload or {}
                score_val = float(pt.score)
                if score_threshold is not None and score_val < score_threshold:
                    continue
                matches.append(
                    RetrievedChunk(
                        chunk_id=payload.get("chunk_id", ""),
                        vulnerability_id=payload.get("vulnerability_id", ""),
                        name=payload.get("name", ""),
                        category=payload.get("category", ""),
                        section=payload.get("section", ""),
                        severity_level=payload.get("severity_level", ""),
                        content=payload.get("content", ""),
                        score=round(score_val, 4),
                        cwe=payload.get("cwe", []),
                        owasp=payload.get("owasp", []),
                        metadata=payload.get("metadata", {}),
                    )
                )

        t2 = time.perf_counter()
        search_ms = (t2 - t1) * 1000.0
        total_ms = (t2 - t0) * 1000.0

        return RetrievalResult(
            query=query.strip(),
            top_k=k,
            latency_ms=round(total_ms, 2),
            embed_latency_ms=round(embed_ms, 2),
            search_latency_ms=round(search_ms, 2),
            matches=matches,
        )


# ---------------------------------------------------------------------------
# Process-Wide Shared Vector Store Lifecycle
# ---------------------------------------------------------------------------


def get_shared_vector_store(
    collection_name: Optional[str] = None,
    storage_path: Optional[Union[str, Path]] = None,
    client: Optional[QdrantClient] = None,
    embedder: Optional[EmbeddingService] = None,
    enable_hybrid: Optional[bool] = None,
    hybrid_weight: Optional[float] = None,
    bm25: Optional[PurePythonBM25] = None,
) -> QdrantVectorStore:
    """Obtain or initialize the process-wide shared QdrantVectorStore instance.

    Guarantees that local embedded Qdrant (which uses an exclusive filesystem lock)
    is only instantiated once per process across health checks and the chat pipeline.
    """
    return QdrantVectorStore(
        collection_name=collection_name,
        storage_path=storage_path,
        client=client,
        embedder=embedder,
        enable_hybrid=enable_hybrid,
        hybrid_weight=hybrid_weight,
        bm25=bm25,
    )


def set_shared_vector_store(store: Optional[QdrantVectorStore]) -> None:
    """Inject or reset the process-wide shared QdrantVectorStore instance (useful for testing)."""
    norm_path = str(Path(str(settings.QDRANT_STORAGE_PATH)).resolve())
    with QdrantVectorStore._lock:
        if store is None:
            QdrantVectorStore._instances.pop(norm_path, None)
        else:
            QdrantVectorStore._instances[norm_path] = store


def close_shared_vector_store() -> None:
    """Close and reset the process-wide shared QdrantVectorStore instance on application shutdown."""
    norm_path = str(Path(str(settings.QDRANT_STORAGE_PATH)).resolve())
    with QdrantVectorStore._lock:
        inst = QdrantVectorStore._instances.pop(norm_path, None)
    if inst is not None:
        inst.close()


