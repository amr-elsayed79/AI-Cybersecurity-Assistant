"""Controlled experiments comparing baseline dense retrieval against lightweight improvements.

Evaluates:
- Baseline (Pure Dense)
- Experiment 1 (Query Normalization)
- Experiment 2 (Lightweight Pure-Python BM25 Hybrid)
- Experiment 3 (Query Normalization + BM25 Hybrid)

Evaluates on both Exploratory Dataset (24 queries) and Held-Out Dataset (12 queries).
"""

import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.rag.chunking import chunk_vulnerabilities
from app.rag.embeddings import embedding_service
from app.rag.ingestion import compute_file_sha256, ingest_master_kb
from app.rag.retriever import QdrantVectorStore


def normalize_query_text(text: str) -> str:
    """Normalize query punctuation, diacritics, and isolate acronyms."""
    # Isolate acronyms in parentheses: e.g. (SSTI) -> SSTI
    text = re.sub(r"\(([A-Za-z0-9_\-]+)\)", r" \1 ", text)
    # Remove Arabic tashkeel (diacritics)
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    # Normalize Arabic alefs
    text = re.sub(r"[إأآٱ]", "ا", text)
    # Normalize teh marbuta
    text = re.sub(r"ة", "ه", text)
    # Normalize alef maqsura
    text = re.sub(r"ى", "ي", text)
    # Clean redundant whitespace
    return re.sub(r"\s+", " ", text).strip()


class PurePythonBM25:
    """Lightweight in-memory Okapi BM25 index over semantic chunks."""

    def __init__(self, chunks: list, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len: Dict[str, int] = {}
        self.term_freqs: Dict[str, Counter] = {}
        self.doc_freqs: Counter = Counter()
        self.num_docs = len(chunks)
        total_len = 0

        for c in chunks:
            # Field weighting: name, aliases, section, vulnerability_id and content
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


def evaluate_dataset(
    dataset_path: Path,
    store: QdrantVectorStore,
    bm25: Optional[PurePythonBM25] = None,
    use_normalization: bool = False,
    hybrid_weight: float = 0.0,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Run evaluation and compute decomposed latency and recall metrics."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    results = []
    embed_latencies = []
    search_latencies = []
    total_latencies = []

    for case in cases:
        raw_query = case["query"]
        lang = case["language"]
        exp_vid = case["expected_vulnerability_id"]
        exp_cid = case["expected_chunk_id"]

        query = normalize_query_text(raw_query) if use_normalization else raw_query

        # 1. Embed query
        t0 = time.perf_counter()
        query_vector = store.embedder.embed_text(query)
        t1 = time.perf_counter()
        embed_ms = (t1 - t0) * 1000.0

        # 2. Vector search (retrieve top 30 candidates if hybrid, else top_k)
        candidate_k = 30 if (bm25 is not None and hybrid_weight > 0.0) else top_k
        search_res = store.client.query_points(
            collection_name=store.collection_name,
            query=query_vector,
            limit=candidate_k,
        )
        t2 = time.perf_counter()
        search_ms = (t2 - t1) * 1000.0

        # 3. Hybrid fusion if enabled
        if bm25 is not None and hybrid_weight > 0.0:
            bm25_scores = bm25.score(query)
            max_b = max(bm25_scores.values()) if bm25_scores else 1.0
            fused_candidates = []
            for pt in search_res.points:
                payload = pt.payload or {}
                cid = payload.get("chunk_id", "")
                dense_score = float(pt.score)
                lexical_score = (bm25_scores.get(cid, 0.0) / max_b) if max_b > 0 else 0.0
                fused_score = (1.0 - hybrid_weight) * dense_score + hybrid_weight * lexical_score
                fused_candidates.append((fused_score, payload))
            fused_candidates.sort(key=lambda x: x[0], reverse=True)
            top_matches = fused_candidates[:top_k]
            top_vids = [m[1].get("vulnerability_id", "") for m in top_matches]
            top_cids = [m[1].get("chunk_id", "") for m in top_matches]
        else:
            top_matches = search_res.points[:top_k]
            top_vids = [(pt.payload or {}).get("vulnerability_id", "") for pt in top_matches]
            top_cids = [(pt.payload or {}).get("chunk_id", "") for pt in top_matches]

        t3 = time.perf_counter()
        ranking_ms = (t3 - t2) * 1000.0
        total_ms = (t3 - t0) * 1000.0

        rec_hit = exp_vid in top_vids
        chk_hit = exp_cid in top_cids

        embed_latencies.append(embed_ms)
        search_latencies.append(search_ms + ranking_ms)
        total_latencies.append(total_ms)

        results.append({
            "question_id": case["question_id"],
            "language": lang,
            "expected_vid": exp_vid,
            "expected_cid": exp_cid,
            "rec_hit": rec_hit,
            "chk_hit": chk_hit,
            "top_vids": top_vids,
            "top_cids": top_cids,
        })

    def calc_group(sub):
        n = len(sub)
        if not n:
            return {"count": 0, "rec_recall": 0.0, "chk_recall": 0.0}
        r_hits = sum(1 for x in sub if x["rec_hit"])
        c_hits = sum(1 for x in sub if x["chk_hit"])
        return {
            "count": n,
            "rec_recall": round((r_hits / n) * 100.0, 2),
            "chk_recall": round((c_hits / n) * 100.0, 2),
        }

    return {
        "overall": calc_group(results),
        "en": calc_group([x for x in results if x["language"] == "en"]),
        "ar": calc_group([x for x in results if x["language"] == "ar"]),
        "latency": {
            "mean_embed_ms": round(sum(embed_latencies) / len(embed_latencies), 2),
            "mean_search_ms": round(sum(search_latencies) / len(search_latencies), 2),
            "mean_total_ms": round(sum(total_latencies) / len(total_latencies), 2),
        },
        "details": results,
    }


def main():
    print("=" * 80)
    print("AI Cybersecurity Assistant — Controlled Retrieval Improvement Experiments")
    print("=" * 80)

    kb_path = settings.MASTER_KB_PATH
    initial_hash = compute_file_sha256(kb_path)
    print(f"Master KB Baseline Checksum: {initial_hash}")

    records, _ = ingest_master_kb(kb_path)
    chunks, _ = chunk_vulnerabilities(records)
    bm25 = PurePythonBM25(chunks)

    store = QdrantVectorStore()
    _ = store.embedder.embed_text("warmup")

    exploratory_path = settings.DATA_DIR / "evaluation" / "retrieval_eval_dataset.json"
    heldout_path = settings.DATA_DIR / "evaluation" / "retrieval_heldout_dataset.json"

    configs = [
        ("Baseline (Pure Dense)", False, 0.0),
        ("Exp 1 (Query Normalization)", True, 0.0),
        ("Exp 2 (BM25 Hybrid w=0.15)", False, 0.15),
        ("Exp 3 (Normalization + BM25)", True, 0.15),
    ]

    all_exp_results = {}

    for name, norm_flag, h_weight in configs:
        res_exp = evaluate_dataset(
            exploratory_path,
            store=store,
            bm25=bm25,
            use_normalization=norm_flag,
            hybrid_weight=h_weight,
        )
        res_held = evaluate_dataset(
            heldout_path,
            store=store,
            bm25=bm25,
            use_normalization=norm_flag,
            hybrid_weight=h_weight,
        )
        all_exp_results[name] = {
            "exploratory": res_exp,
            "heldout": res_held,
        }

    store.close()

    # Print Summary Tables
    print("\n" + "=" * 80)
    print("RESULTS: EXPLORATORY DATASET (24 Queries: 12 EN, 12 AR)")
    print("=" * 80)
    header = f"{'Configuration':<30} {'Overall Rec@5':<15} {'AR Rec@5':<12} {'EN Rec@5':<12} {'Chunk Rec@5':<14} {'Embed(ms)':<10} {'Search(ms)':<11} {'Total(ms)':<10}"
    print(header)
    print("-" * 105)
    for name in configs:
        c_name = name[0]
        r = all_exp_results[c_name]["exploratory"]
        print(
            f"{c_name:<30} "
            f"{r['overall']['rec_recall']:>11.1f}% "
            f"{r['ar']['rec_recall']:>10.1f}% "
            f"{r['en']['rec_recall']:>10.1f}% "
            f"{r['overall']['chk_recall']:>12.1f}% "
            f"{r['latency']['mean_embed_ms']:>8.2f} "
            f"{r['latency']['mean_search_ms']:>9.2f} "
            f"{r['latency']['mean_total_ms']:>8.2f}"
        )

    print("\n" + "=" * 80)
    print("RESULTS: HELD-OUT GENERALIZATION DATASET (12 Queries: 6 EN, 6 AR)")
    print("=" * 80)
    print(header)
    print("-" * 105)
    for name in configs:
        c_name = name[0]
        r = all_exp_results[c_name]["heldout"]
        print(
            f"{c_name:<30} "
            f"{r['overall']['rec_recall']:>11.1f}% "
            f"{r['ar']['rec_recall']:>10.1f}% "
            f"{r['en']['rec_recall']:>10.1f}% "
            f"{r['overall']['chk_recall']:>12.1f}% "
            f"{r['latency']['mean_embed_ms']:>8.2f} "
            f"{r['latency']['mean_search_ms']:>9.2f} "
            f"{r['latency']['mean_total_ms']:>8.2f}"
        )

    # Post-check SHA-256
    post_hash = compute_file_sha256(kb_path)
    print("\n" + "-" * 80)
    print(f"Post-run Master KB SHA-256: {post_hash}")
    assert initial_hash == post_hash, "Master KB was modified!"
    print("INTEGRITY CONFIRMED: Master KB remains 100% untouched.")


if __name__ == "__main__":
    main()
