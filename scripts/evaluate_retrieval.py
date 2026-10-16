"""CLI script to run retrieval evaluation benchmark and output metrics."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.rag.evaluator import RetrievalEvaluator
from app.rag.ingestion import compute_file_sha256
from app.rag.retriever import QdrantVectorStore


def main():
    parser = argparse.ArgumentParser(description="Run retrieval benchmark evaluation.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to evaluation JSON dataset.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks to evaluate (default: 5).",
    )
    args = parser.parse_args()

    print("=" * 95)
    print("AI Cybersecurity Assistant — Phase 3 Retrieval Benchmark (Hybrid BM25 Integrated)")
    print("=" * 95)

    kb_path = settings.MASTER_KB_PATH
    initial_hash = compute_file_sha256(kb_path)
    print(f"Master KB SHA-256:       {initial_hash}")
    print(f"Hybrid Retrieval Active: {settings.ENABLE_HYBRID_RETRIEVAL}")
    print(f"Hybrid BM25 Weight:      {settings.HYBRID_BM25_WEIGHT}")
    if initial_hash != settings.BASELINE_KB_SHA256:
        print(f"ERROR: Checksum mismatch against baseline ({settings.BASELINE_KB_SHA256})!")
        sys.exit(1)

    ds_path = Path(args.dataset) if args.dataset else None
    store = QdrantVectorStore()
    try:
        evaluator = RetrievalEvaluator(vector_store=store)
        report = evaluator.run_benchmark(dataset_path=ds_path, top_k=args.top_k)
    finally:
        store.close()

    print("\n" + "-" * 95)
    print(f"PER-QUERY EVALUATION RESULTS (Top-k = {args.top_k})")
    print("-" * 95)
    header = f"{'ID':<14} {'Lang':<5} {'Expected Record':<24} {'Rec Hit':<9} {'Chk Hit':<9} {'Embed(ms)':<10} {'Search(ms)':<11} {'Total(ms)':<10} {'Top Score':<9}"
    print(header)
    print("-" * 95)

    failed_queries = []
    for q in report.query_results:
        rec_hit_str = "YES" if q.record_hit_at_k else "NO"
        chk_hit_str = "YES" if q.chunk_hit_at_k else "NO"
        if not q.record_hit_at_k or not q.chunk_hit_at_k:
            failed_queries.append(q)
        print(
            f"{q.question_id:<14} {q.language.upper():<5} {q.expected_vulnerability_id:<24} "
            f"{rec_hit_str:<9} {chk_hit_str:<9} {q.embed_latency_ms:<10.2f} {q.search_latency_ms:<11.2f} {q.latency_ms:<10.2f} {q.top_score:<9.4f}"
        )

    print("\n" + "=" * 95)
    print("AGGREGATED BENCHMARK METRICS SUMMARY")
    print("=" * 95)
    print(f"{'Group':<15} {'Queries':<10} {'Record Recall@5':<18} {'Chunk Recall@5':<16} {'Embed (ms)':<12} {'Search (ms)':<13} {'Total (ms)':<12}")
    print("-" * 95)

    # Breakdown by language
    for lang, metrics in report.by_language.items():
        lang_label = f"Language ({lang.upper()})"
        print(
            f"{lang_label:<15} {metrics.total_queries:<10} {metrics.record_recall_at_k:>14.2f}% "
            f"{metrics.chunk_recall_at_k:>13.2f}% {metrics.mean_embed_latency_ms:>10.2f} ms "
            f"{metrics.mean_search_latency_ms:>11.2f} ms {metrics.mean_latency_ms:>10.2f} ms"
        )

    print("-" * 95)
    # Overall
    print(
        f"{'OVERALL':<15} {report.overall.total_queries:<10} {report.overall.record_recall_at_k:>14.2f}% "
        f"{report.overall.chunk_recall_at_k:>13.2f}% {report.overall.mean_embed_latency_ms:>10.2f} ms "
        f"{report.overall.mean_search_latency_ms:>11.2f} ms {report.overall.mean_latency_ms:>10.2f} ms"
    )
    print("=" * 95)

    if failed_queries:
        print("\n" + "-" * 95)
        print("FAILED QUERY DETAILS (Top-5 Retrieved vs Expected)")
        print("-" * 95)
        for f in failed_queries:
            status = []
            if not f.record_hit_at_k:
                status.append("RECORD MISS")
            if not f.chunk_hit_at_k:
                status.append("CHUNK MISS")
            print(f"[{', '.join(status)}] {f.question_id} ({f.language.upper()}): '{f.query}'")
            print(f"   Expected: Record = '{f.expected_vulnerability_id}', Chunk = '{f.expected_chunk_id}'")
            print(f"   Top-5 VIDs:   {f.retrieved_vulnerability_ids}")
            print(f"   Top-5 Chunks: {f.retrieved_chunk_ids}\n")

    # Post-run hash verification
    final_hash = compute_file_sha256(kb_path)
    print("-" * 95)
    print(f"Post-execution SHA-256: {final_hash}")
    if initial_hash == final_hash:
        print("INTEGRITY CONFIRMED: Master KB on disk remains 100% UNCHANGED byte-for-byte.")
    else:
        print("CRITICAL ERROR: File was modified on disk!")
        sys.exit(1)


if __name__ == "__main__":
    main()
