"""Automated test suite for Phase 3: Embeddings, Vector Indexing, Hybrid Retrieval, and Evaluation."""

import threading
import uuid
from unittest.mock import patch
import pytest

from app.config import settings
from app.rag.chunking import chunk_vulnerabilities
from app.rag.embeddings import EmbeddingService, embedding_service
from app.rag.evaluator import RetrievalEvaluator
from app.rag.ingestion import compute_file_sha256, ingest_master_kb
from app.rag.retriever import PurePythonBM25, QdrantVectorStore, generate_chunk_uuid


@pytest.fixture(scope="session")
def loaded_chunks():
    """Load and chunk Master KB once for session tests."""
    records, _ = ingest_master_kb(settings.MASTER_KB_PATH)
    chunks, _ = chunk_vulnerabilities(records)
    return chunks


@pytest.fixture(scope="session")
def vector_store():
    """Shared vector store connected to the indexed Qdrant database."""
    store = QdrantVectorStore()
    yield store
    store.close()


def test_kb_checksum_baseline_preserved():
    """Verify Master KB SHA-256 matches the pristine baseline."""
    current_hash = compute_file_sha256(settings.MASTER_KB_PATH)
    assert current_hash == settings.BASELINE_KB_SHA256, "Master KB hash differs from baseline!"


def test_deterministic_chunk_uuid_generation():
    """Verify that chunk UUID generation is deterministic and collision-free."""
    id1 = generate_chunk_uuid("sql-injection_overview_and_mechanism")
    id2 = generate_chunk_uuid("sql-injection_overview_and_mechanism")
    id3 = generate_chunk_uuid("sql-injection_prevention_and_remediation")

    # Idempotent: same ID gives same UUID
    assert id1 == id2
    # Distinct: different IDs give different UUIDs
    assert id1 != id3
    # Valid UUID format
    assert uuid.UUID(id1).version == 5


def test_embedding_service_dimension_and_output():
    """Verify embedding service outputs 384-dimensional normalized vectors."""
    vector = embedding_service.embed_text("SQL injection vulnerability")
    assert isinstance(vector, list)
    assert len(vector) == settings.EMBEDDING_DIMENSION
    assert all(isinstance(v, float) for v in vector)


def test_embedding_service_rejects_empty_text():
    """Verify embed_text raises ValueError on empty or whitespace strings."""
    with pytest.raises(ValueError, match="Cannot embed an empty"):
        embedding_service.embed_text("")

    with pytest.raises(ValueError, match="Cannot embed an empty"):
        embedding_service.embed_text("   \n\t  ")


def test_qdrant_total_indexed_point_count(vector_store):
    """Verify exactly 588 substantive chunks are present in the Qdrant index."""
    points_count = vector_store.count_points()
    assert points_count == 588, f"Expected 588 indexed points, found {points_count}"


def test_indexing_idempotency(loaded_chunks):
    """Verify re-indexing existing chunks into Qdrant does not create duplicate points."""
    memory_store = QdrantVectorStore(
        collection_name="test_idempotency",
        storage_path=":memory:",
        enable_hybrid=False,
    )
    try:
        sample_batch = loaded_chunks[:10]

        # First indexing pass
        res1 = memory_store.index_chunks(sample_batch, batch_size=10)
        assert res1["indexed_chunks"] == 10
        assert memory_store.count_points() == 10

        # Second indexing pass with the same chunks
        res2 = memory_store.index_chunks(sample_batch, batch_size=10)
        assert res2["indexed_chunks"] == 10
        # Point count MUST remain 10 (no duplicates)
        assert memory_store.count_points() == 10
    finally:
        memory_store.close()


def test_pure_python_bm25_indexing_and_scoring(loaded_chunks):
    """Verify pure-Python BM25 index builds properly and produces correct lexical scores."""
    bm25 = PurePythonBM25(loaded_chunks)
    assert bm25.num_docs == len(loaded_chunks)
    assert len(bm25.idf) > 0

    scores = bm25.score("SQL Injection")
    assert len(scores) > 0
    # Top score should be a sql-injection chunk
    top_chunk_id = max(scores.items(), key=lambda x: x[1])[0]
    assert "sql-injection" in top_chunk_id


def test_hybrid_ranking_deterministic_behavior(vector_store):
    """Verify hybrid retrieval produces strictly deterministic rankings and scores."""
    res1 = vector_store.retrieve("Server-Side Request Forgery cloud metadata", top_k=5)
    res2 = vector_store.retrieve("Server-Side Request Forgery cloud metadata", top_k=5)

    assert len(res1.matches) == len(res2.matches)
    for m1, m2 in zip(res1.matches, res2.matches):
        assert m1.chunk_id == m2.chunk_id
        assert m1.score == m2.score


def test_hybrid_retrieval_decomposed_latency(vector_store):
    """Verify decomposed latency metrics are populated and plausible."""
    res = vector_store.retrieve("Cross-site request forgery prevention", top_k=3)
    assert res.embed_latency_ms > 0.0
    assert res.search_latency_ms > 0.0
    assert res.latency_ms >= res.embed_latency_ms
    assert res.latency_ms < 200.0


def test_retrieval_metadata_fidelity(vector_store):
    """Verify retrieved chunks contain complete, accurate source metadata."""
    res = vector_store.retrieve("XML external entity parsing injection", top_k=3)
    assert len(res.matches) > 0
    top = res.matches[0]

    assert top.chunk_id
    assert top.vulnerability_id
    assert top.name
    assert top.category
    assert top.section
    assert top.severity_level
    assert top.content
    assert isinstance(top.score, float)
    assert 0.0 <= top.score <= 1.0


def test_retrieval_top_k_ordering(vector_store):
    """Verify retrieved chunks are sorted in descending order of similarity score."""
    res = vector_store.retrieve("cross site scripting prevention", top_k=5)
    assert len(res.matches) == 5

    scores = [m.score for m in res.matches]
    assert scores == sorted(scores, reverse=True), "Matches are not sorted descending by score!"


def test_retrieval_rejects_empty_query(vector_store):
    """Verify retrieve raises ValueError when provided empty or whitespace query."""
    with pytest.raises(ValueError, match="Query string cannot be empty"):
        vector_store.retrieve("")

    with pytest.raises(ValueError, match="Query string cannot be empty"):
        vector_store.retrieve("    \t\n  ")


def test_retrieval_vulnerability_id_filter(vector_store):
    """Verify exact filtering by vulnerability_id returns only matching chunks."""
    res = vector_store.retrieve(
        "security vulnerabilities",
        top_k=5,
        filter_vulnerability_id="sql-injection",
    )
    assert len(res.matches) > 0
    for match in res.matches:
        assert match.vulnerability_id == "sql-injection"


def test_retrieval_section_filter(vector_store):
    """Verify exact filtering by section returns only matching chunks."""
    res = vector_store.retrieve(
        "how to mitigate risks",
        top_k=5,
        filter_section="prevention_and_remediation",
    )
    assert len(res.matches) > 0
    for match in res.matches:
        assert match.section == "prevention_and_remediation"


def test_bilingual_exploratory_benchmark_thresholds(vector_store):
    """Verify the exploratory evaluation benchmark achieves >= 90% record recall with hybrid search."""
    evaluator = RetrievalEvaluator(vector_store=vector_store)
    report = evaluator.run_benchmark(top_k=5)

    # 1. Total queries evaluated
    assert report.overall.total_queries == 24
    assert report.by_language["en"].total_queries == 12
    assert report.by_language["ar"].total_queries == 12

    # 2. Recall@5 thresholds (hybrid achieves 91.67% overall, 100% EN, 83.33% AR)
    assert report.by_language["en"].record_recall_at_k == 100.0
    assert report.by_language["ar"].record_recall_at_k >= 80.0
    assert report.overall.record_recall_at_k >= 90.0
    assert report.overall.chunk_recall_at_k >= 70.0

    # 3. Latency threshold (sub-60ms)
    assert report.overall.mean_latency_ms < 60.0, f"Mean latency too high: {report.overall.mean_latency_ms} ms"


def test_bilingual_heldout_generalization_benchmark(vector_store):
    """Verify the held-out generalization benchmark achieves 100% record recall."""
    heldout_path = settings.DATA_DIR / "evaluation" / "retrieval_heldout_dataset.json"
    evaluator = RetrievalEvaluator(vector_store=vector_store)
    report = evaluator.run_benchmark(dataset_path=heldout_path, top_k=5)

    assert report.overall.total_queries == 12
    assert report.overall.record_recall_at_k == 100.0
    assert report.overall.chunk_recall_at_k >= 80.0
    assert report.overall.mean_latency_ms < 60.0


def test_concurrent_initialization_deterministic_race_prevention(tmp_path):
    """Verify concurrent initialization of the same path never exposes an uninitialized client.

    Uses an isolated temporary storage directory. Instruments initialization using threading.Event
    to pause Thread-1 inside __init__ before client creation, verifies Thread-2 blocks and cannot
    complete or access an uninitialized client, releases Thread-1, and confirms both threads receive
    the exact same initialized instance without errors or lock collisions.
    """
    from pathlib import Path
    temp_dir = tmp_path / "concurrent_det_qdrant"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path_str = str(temp_dir)

    t1_paused = threading.Event()
    release_t1 = threading.Event()

    orig_mkdir = Path.mkdir

    def instrumented_mkdir(self, *args, **kwargs):
        orig_mkdir(self, *args, **kwargs)
        if "concurrent_det_qdrant" in str(self) and threading.current_thread().name == "Worker-1":
            t1_paused.set()
            release_t1.wait(timeout=5.0)

    inst1 = None
    inst2 = None
    t2_error = None
    t2_completed = threading.Event()

    def worker_1():
        nonlocal inst1
        inst1 = QdrantVectorStore(storage_path=path_str, enable_hybrid=False)

    def worker_2():
        nonlocal inst2, t2_error
        try:
            inst2 = QdrantVectorStore(storage_path=path_str, enable_hybrid=False)
            assert inst2.client is not None
            assert hasattr(inst2.client, "get_collection")
        except Exception as e:
            t2_error = e
        finally:
            t2_completed.set()

    t1 = threading.Thread(target=worker_1, name="Worker-1")
    t2 = threading.Thread(target=worker_2, name="Worker-2")

    try:
        with patch.object(Path, "mkdir", instrumented_mkdir):
            t1.start()
            assert t1_paused.wait(timeout=5.0), "Worker-1 failed to pause inside initialization hook"

            # Launch Worker-2 while Worker-1 is paused
            t2.start()

            # Deterministic check: Worker-2 MUST be blocked waiting on Worker-1's init_lock
            completed_early = t2_completed.wait(timeout=0.3)
            assert not completed_early, "Worker-2 should be blocked waiting for Worker-1 to finish initialization!"

            # Release Worker-1 to complete initialization
            release_t1.set()
            t1.join(timeout=5.0)
            t2.join(timeout=5.0)

        assert t2_error is None, f"Worker-2 raised an unexpected error: {t2_error}"
        assert inst1 is not None, "Worker-1 did not instantiate vector store"
        assert inst2 is not None, "Worker-2 did not instantiate vector store"
        assert inst1 is inst2, "Worker-1 and Worker-2 did not receive the identical multiton instance!"
        assert inst1.client is not None
        assert inst1.count_points() == 0
    finally:
        if inst1 is not None:
            inst1.close()


def test_simultaneous_multi_threaded_barrier_initialization(tmp_path):
    """Verify 10 threads hitting the vector store simultaneously all receive the same instance safely."""
    temp_dir = tmp_path / "concurrent_barrier_qdrant"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path_str = str(temp_dir)

    num_threads = 10
    barrier = threading.Barrier(num_threads)
    results = [None] * num_threads
    errors = [None] * num_threads

    def worker(idx):
        try:
            barrier.wait(timeout=5.0)
            store = QdrantVectorStore(storage_path=path_str, enable_hybrid=False)
            assert store.client is not None
            _ = store.count_points()
            results[idx] = store
        except Exception as e:
            errors[idx] = e

    threads = [threading.Thread(target=worker, args=(i,), name=f"BarrierWorker-{i}") for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    try:
        assert all(e is None for e in errors), f"One or more threads encountered errors: {errors}"
        assert all(r is not None for r in results), "One or more threads returned None"

        first_store = results[0]
        for r in results:
            assert r is first_store, "Threads received different instances instead of the shared multiton!"
        assert first_store.count_points() == 0
    finally:
        if results[0] is not None:
            results[0].close()


def test_post_test_master_kb_checksum_integrity():
    """Verify Master KB on disk remains 100% byte-for-byte identical to baseline."""
    post_hash = compute_file_sha256(settings.MASTER_KB_PATH)
    assert post_hash == settings.BASELINE_KB_SHA256, (
        f"Master KB on disk was modified! Baseline: {settings.BASELINE_KB_SHA256}, Actual: {post_hash}"
    )
