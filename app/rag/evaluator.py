"""Retrieval evaluation module measuring bilingual Recall@5 and latency metrics."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.config import settings
from app.rag.retriever import QdrantVectorStore


class QueryEvalResult(BaseModel):
    """Detailed evaluation result for an individual test question."""
    question_id: str
    language: str
    query: str
    expected_vulnerability_id: str
    expected_chunk_id: str
    top_k: int
    latency_ms: float
    embed_latency_ms: float = 0.0
    search_latency_ms: float = 0.0
    record_hit_at_k: bool
    chunk_hit_at_k: bool
    retrieved_vulnerability_ids: List[str] = Field(default_factory=list)
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    top_score: float = 0.0


class GroupMetrics(BaseModel):
    """Aggregated evaluation metrics for a group of queries."""
    total_queries: int
    record_recall_at_k: float
    chunk_recall_at_k: float
    mean_latency_ms: float
    mean_embed_latency_ms: float = 0.0
    mean_search_latency_ms: float = 0.0
    min_latency_ms: float
    max_latency_ms: float


class BenchmarkReport(BaseModel):
    """Complete evaluation benchmark report."""
    dataset_path: str
    top_k: int
    overall: GroupMetrics
    by_language: Dict[str, GroupMetrics]
    query_results: List[QueryEvalResult]


def compute_group_metrics(results: List[QueryEvalResult]) -> GroupMetrics:
    """Calculate Recall@k and latency metrics for a subset of evaluation results."""
    if not results:
        return GroupMetrics(
            total_queries=0,
            record_recall_at_k=0.0,
            chunk_recall_at_k=0.0,
            mean_latency_ms=0.0,
            mean_embed_latency_ms=0.0,
            mean_search_latency_ms=0.0,
            min_latency_ms=0.0,
            max_latency_ms=0.0,
        )

    n = len(results)
    record_hits = sum(1 for r in results if r.record_hit_at_k)
    chunk_hits = sum(1 for r in results if r.chunk_hit_at_k)
    latencies = [r.latency_ms for r in results]
    embed_latencies = [r.embed_latency_ms for r in results]
    search_latencies = [r.search_latency_ms for r in results]

    return GroupMetrics(
        total_queries=n,
        record_recall_at_k=round((record_hits / n) * 100.0, 2),
        chunk_recall_at_k=round((chunk_hits / n) * 100.0, 2),
        mean_latency_ms=round(sum(latencies) / n, 2),
        mean_embed_latency_ms=round(sum(embed_latencies) / n, 2),
        mean_search_latency_ms=round(sum(search_latencies) / n, 2),
        min_latency_ms=round(min(latencies), 2),
        max_latency_ms=round(max(latencies), 2),
    )


class RetrievalEvaluator:
    """Runs retrieval evaluation benchmarks over test datasets."""

    def __init__(self, vector_store: Optional[QdrantVectorStore] = None):
        self.vector_store = vector_store or QdrantVectorStore()
        # Warm up the embedder model so first-query model loading does not skew latency metrics
        _ = self.vector_store.embedder.embed_text("warmup")

    def run_benchmark(
        self,
        dataset_path: Optional[Path] = None,
        top_k: int = 5,
    ) -> BenchmarkReport:
        """Run benchmark over the test dataset."""
        path = dataset_path or (settings.DATA_DIR / "evaluation" / "retrieval_eval_dataset.json")
        if not path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            cases = json.load(f)

        results: List[QueryEvalResult] = []

        for case in cases:
            q_id = case["question_id"]
            lang = case["language"]
            query = case["query"]
            exp_vuln_id = case["expected_vulnerability_id"]
            exp_chunk_id = case["expected_chunk_id"]

            retrieval = self.vector_store.retrieve(query, top_k=top_k)

            retrieved_v_ids = [m.vulnerability_id for m in retrieval.matches]
            retrieved_c_ids = [m.chunk_id for m in retrieval.matches]
            top_score = retrieval.matches[0].score if retrieval.matches else 0.0

            record_hit = exp_vuln_id in retrieved_v_ids
            chunk_hit = exp_chunk_id in retrieved_c_ids

            results.append(
                QueryEvalResult(
                    question_id=q_id,
                    language=lang,
                    query=query,
                    expected_vulnerability_id=exp_vuln_id,
                    expected_chunk_id=exp_chunk_id,
                    top_k=top_k,
                    latency_ms=retrieval.latency_ms,
                    embed_latency_ms=retrieval.embed_latency_ms,
                    search_latency_ms=retrieval.search_latency_ms,
                    record_hit_at_k=record_hit,
                    chunk_hit_at_k=chunk_hit,
                    retrieved_vulnerability_ids=retrieved_v_ids,
                    retrieved_chunk_ids=retrieved_c_ids,
                    top_score=round(top_score, 4),
                )
            )

        # Separate results by language
        languages = sorted(list({r.language for r in results}))
        by_language = {}
        for lang in languages:
            lang_results = [r for r in results if r.language == lang]
            by_language[lang] = compute_group_metrics(lang_results)

        overall = compute_group_metrics(results)

        return BenchmarkReport(
            dataset_path=str(path),
            top_k=top_k,
            overall=overall,
            by_language=by_language,
            query_results=results,
        )
