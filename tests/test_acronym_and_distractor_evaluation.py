"""Offline automated evaluation suite for acronym, boundary/distractor, and out-of-scope queries.

Evaluates:
- Acronym queries (short keywords, e.g. XSS, SSRF, IDOR, BOLA, XXE, SQLi, NoSQLi, SSTI, LFI, ReDoS)
- Boundary queries (single-concept vs. explicit comparison, e.g. SQLi vs NoSQLi, Path Traversal vs LFI)
- Out-of-scope queries (measuring the dense embedding noise floor)

All tests run 100% offline with zero live LLM calls, zero network access, and zero modification
to production code or the Master KB.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List
import pytest

# Ensure sentence-transformers / huggingface runs strictly offline
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from app.config import settings
from app.rag.ingestion import compute_file_sha256
from app.rag.retriever import QdrantVectorStore


DATASET_PATH = settings.DATA_DIR / "evaluation" / "acronym_and_distractor_dataset.json"


@pytest.fixture(scope="session")
def vector_store():
    """Shared vector store connected to the indexed Qdrant database."""
    store = QdrantVectorStore()
    yield store
    store.close()


def test_master_kb_checksum_intact():
    """Verify Master KB SHA-256 matches the pristine baseline before evaluation."""
    current_hash = compute_file_sha256(settings.MASTER_KB_PATH)
    assert current_hash == settings.BASELINE_KB_SHA256, "Master KB hash differs from baseline!"


def test_dataset_schema_and_integrity():
    """Validate dataset structure, schema types, query uniqueness, and category counts."""
    assert DATASET_PATH.exists(), f"Evaluation dataset not found at: {DATASET_PATH}"

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list), "Dataset must be a JSON array"
    assert len(data) == 20, f"Expected exactly 20 test cases, found {len(data)}"

    seen_ids = set()
    single_concept_count = 0
    comparison_count = 0
    out_of_scope_count = 0

    for case in data:
        qid = case.get("question_id")
        assert qid, "Each case must have a non-empty question_id"
        assert qid not in seen_ids, f"Duplicate question_id detected: {qid}"
        seen_ids.add(qid)

        qtype = case.get("query_type")
        assert qtype in {"single_concept", "comparison", "out_of_scope"}, (
            f"Invalid query_type '{qtype}' in case {qid}"
        )

        query = case.get("query")
        assert isinstance(query, str) and query.strip(), f"Empty query in case {qid}"

        expected_ids = case.get("expected_vulnerability_ids")
        assert isinstance(expected_ids, list), f"expected_vulnerability_ids must be a list in {qid}"

        rationale = case.get("rationale")
        assert isinstance(rationale, str) and rationale.strip(), f"Missing rationale in {qid}"

        if qtype == "single_concept":
            single_concept_count += 1
            assert len(expected_ids) == 1, (
                f"single_concept query {qid} must have exactly 1 expected target, found {len(expected_ids)}"
            )
        elif qtype == "comparison":
            comparison_count += 1
            assert len(expected_ids) >= 2, (
                f"comparison query {qid} must have at least 2 expected targets, found {len(expected_ids)}"
            )
        elif qtype == "out_of_scope":
            out_of_scope_count += 1
            assert len(expected_ids) == 0, (
                f"out_of_scope query {qid} must have empty expected_vulnerability_ids, found {expected_ids}"
            )

    # 10 acronyms (single_concept) + 3 boundary single_concept = 13 single_concept
    # 2 boundary comparison = 2 comparison
    # 5 out_of_scope = 5 out_of_scope
    assert single_concept_count == 13, f"Expected 13 single_concept queries, found {single_concept_count}"
    assert comparison_count == 2, f"Expected 2 comparison queries, found {comparison_count}"
    assert out_of_scope_count == 5, f"Expected 5 out_of_scope queries, found {out_of_scope_count}"


def test_expected_target_ids_exist_in_master_kb():
    """Verify every target vulnerability ID is a canonical record in the Master KB."""
    with open(settings.MASTER_KB_PATH, "r", encoding="utf-8") as f:
        kb_data = json.load(f)

    records = kb_data.get("records", [])
    assert len(records) == 74, f"Expected 74 canonical records in Master KB, found {len(records)}"

    canonical_ids = {r["id"] for r in records}

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    for case in dataset:
        for target_id in case["expected_vulnerability_ids"]:
            assert target_id in canonical_ids, (
                f"Target ID '{target_id}' in query {case['question_id']} does not exist in Master KB!"
            )


def test_offline_retrieval_evaluation_suite(vector_store, capsys):
    """Execute offline retrieval evaluation and compute baseline metrics across all categories."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Separate test subsets
    acronym_cases = [c for c in dataset if c["question_id"].startswith("eval-acronym")]
    boundary_cases = [c for c in dataset if c["question_id"].startswith("eval-boundary")]
    oos_cases = [c for c in dataset if c["query_type"] == "out_of_scope"]

    # Metrics accumulators
    acronym_p1_hits = 0
    acronym_recall5_hits = 0
    acronym_distractor_intrusions = 0

    boundary_p1_hits = 0
    boundary_single_distractor_intrusions = 0
    boundary_comparison_full_recalls = 0

    all_latencies = []
    oos_scores = []

    print("\n" + "=" * 90)
    print("OFFLINE RETRIEVAL EVALUATION REPORT: ACRONYM, BOUNDARY, AND OUT-OF-SCOPE SUITE")
    print("=" * 90)

    # 1. Acronym Queries Evaluation
    print("\n--- [PART 1: 10 ACRONYM QUERIES (single_concept)] ---")
    for case in acronym_cases:
        qid = case["question_id"]
        q = case["query"]
        target = case["expected_vulnerability_ids"][0]

        res = vector_store.retrieve(q, top_k=5)
        all_latencies.append(res.latency_ms)

        retrieved_ids = [m.vulnerability_id for m in res.matches]
        top1_id = retrieved_ids[0] if retrieved_ids else None
        top1_hit = (top1_id == target)
        recall5_hit = (target in retrieved_ids)

        if top1_hit:
            acronym_p1_hits += 1
        if recall5_hit:
            acronym_recall5_hits += 1

        # Distractor intrusion in top 4
        top4_ids = retrieved_ids[:4]
        intrusions = sum(1 for vid in top4_ids if vid != target)
        acronym_distractor_intrusions += intrusions

        top_details = ", ".join([f"{m.vulnerability_id} ({m.score:.4f})" for m in res.matches[:3]])
        print(f"[{qid}] '{q}' -> Target: {target} | P@1: {top1_hit} | R@5: {recall5_hit} | Top-3: {top_details}")

    # 2. Boundary Queries Evaluation
    print("\n--- [PART 2: 5 BOUNDARY & COMPARISON QUERIES] ---")
    for case in boundary_cases:
        qid = case["question_id"]
        q = case["query"]
        qtype = case["query_type"]
        targets = case["expected_vulnerability_ids"]

        res = vector_store.retrieve(q, top_k=5)
        all_latencies.append(res.latency_ms)

        retrieved_ids = [m.vulnerability_id for m in res.matches]

        if qtype == "single_concept":
            target = targets[0]
            top1_hit = (retrieved_ids[0] == target) if retrieved_ids else False
            recall5_hit = (target in retrieved_ids)
            if top1_hit:
                boundary_p1_hits += 1
            top4_ids = retrieved_ids[:4]
            intrusions = sum(1 for vid in top4_ids if vid != target)
            boundary_single_distractor_intrusions += intrusions
            print(f"[{qid}] (Single) '{q}' -> Target: {target} | P@1: {top1_hit} | R@5: {recall5_hit} | Top-1: {retrieved_ids[0]}")
        else:  # comparison
            retrieved_set = set(retrieved_ids)
            all_targets_present = all(t in retrieved_set for t in targets)
            if all_targets_present:
                boundary_comparison_full_recalls += 1
            print(f"[{qid}] (Comparison) '{q}' -> Targets: {targets} | Both in top-5: {all_targets_present} | Retrieved: {retrieved_ids[:4]}")

    # 3. Out-of-Scope Queries Evaluation
    print("\n--- [PART 3: 5 OUT-OF-SCOPE QUERIES (NOISE FLOOR MEASUREMENT)] ---")
    for case in oos_cases:
        qid = case["question_id"]
        q = case["query"]

        res = vector_store.retrieve(q, top_k=5)
        all_latencies.append(res.latency_ms)

        top_match = res.matches[0] if res.matches else None
        top_score = top_match.score if top_match else 0.0
        top_vid = top_match.vulnerability_id if top_match else "None"
        oos_scores.append(top_score)

        print(f"[{qid}] '{q[:50]}...' -> Noise Peak: {top_vid} (score: {top_score:.4f})")

    # Aggregate Metrics Calculation
    num_acronyms = len(acronym_cases)
    acronym_p1_pct = (acronym_p1_hits / num_acronyms) * 100.0
    acronym_r5_pct = (acronym_recall5_hits / num_acronyms) * 100.0
    acronym_intrusion_rate = (acronym_distractor_intrusions / (num_acronyms * 4)) * 100.0

    boundary_single_cases = [c for c in boundary_cases if c["query_type"] == "single_concept"]
    boundary_comparison_cases = [c for c in boundary_cases if c["query_type"] == "comparison"]

    boundary_p1_pct = (boundary_p1_hits / len(boundary_single_cases)) * 100.0 if boundary_single_cases else 0.0
    boundary_comparison_pct = (boundary_comparison_full_recalls / len(boundary_comparison_cases)) * 100.0 if boundary_comparison_cases else 0.0

    mean_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0.0
    sorted_latencies = sorted(all_latencies)
    median_latency = sorted_latencies[len(sorted_latencies) // 2] if sorted_latencies else 0.0

    print("\n" + "=" * 90)
    print("SUMMARY METRICS TABLE:")
    print(f"Acronym Queries (N=10)       : Precision@1 = {acronym_p1_pct:.1f}%, Recall@5 = {acronym_r5_pct:.1f}%")
    print(f"Acronym Distractor Intrusion : {acronym_intrusion_rate:.1f}% of top-4 slots occupied by distractors")
    print(f"Boundary Single-Concept (N=3): Precision@1 = {boundary_p1_pct:.1f}%")
    print(f"Boundary Comparison (N=2)    : Multi-Target Recall@5 = {boundary_comparison_pct:.1f}%")
    print(f"Out-of-Scope Noise Range     : Min = {min(oos_scores):.4f}, Mean = {sum(oos_scores)/len(oos_scores):.4f}, Max = {max(oos_scores):.4f}")
    print(f"Latency (N=20)               : Mean = {mean_latency:.2f} ms, Median = {median_latency:.2f} ms")
    print("=" * 90 + "\n")

    # Non-arbitrary integrity checks: All 20 queries must execute and return valid results
    assert len(all_latencies) == 20
    assert len(oos_scores) == 5
    assert acronym_p1_hits >= 0
