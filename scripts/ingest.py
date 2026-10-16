"""Command-line script to execute Knowledge Base validation, in-memory normalization, and semantic chunking."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.rag.chunking import chunk_vulnerabilities
from app.rag.ingestion import compute_file_sha256, ingest_master_kb


def main():
    print("=" * 70)
    print("AI Cybersecurity Assistant — Phase 2 Ingestion & Validation Pipeline")
    print("=" * 70)

    kb_path = settings.MASTER_KB_PATH
    print(f"Target Knowledge Base: {kb_path}")

    # 1. Pre-execution hash check
    initial_hash = compute_file_sha256(kb_path)
    print(f"Pre-execution SHA-256:  {initial_hash}")

    if initial_hash != settings.BASELINE_KB_SHA256:
        print(f"ERROR: Checksum mismatch against baseline ({settings.BASELINE_KB_SHA256})!")
        sys.exit(1)

    # 2. Ingest and normalize in-memory
    print("\n[Step 1] Ingesting and validating Master KB...")
    records, ingest_stats = ingest_master_kb(kb_path)
    print(f"  - Total canonical records loaded: {ingest_stats['total_records']}")
    print(f"  - In-memory normalized records:   {ingest_stats['normalized_count']}")
    print(f"    Affected IDs: {', '.join(ingest_stats['normalized_ids'])}")
    print(f"  - Preserved unaltered records:    {ingest_stats['unmodified_count']}")

    # 3. Semantic chunking
    print("\n[Step 2] Executing semantic chunking...")
    chunks, chunk_stats = chunk_vulnerabilities(records)
    print(f"  - Total semantic chunks created:  {chunk_stats['total_chunks']}")
    print(f"  - Average chunks per record:     {chunk_stats['average_chunks_per_record']}")
    print("  - Per-section chunk distribution:")
    for section, count in chunk_stats["per_section_counts"].items():
        print(f"      * {section:<30}: {count} chunks")

    # 4. Post-execution hash check
    final_hash = compute_file_sha256(kb_path)
    print("\n[Step 3] Post-execution verification:")
    print(f"  - Post-execution SHA-256: {final_hash}")
    if initial_hash == final_hash:
        print("  - INTEGRITY CONFIRMED: Master KB on disk is 100% UNCHANGED byte-for-byte.")
    else:
        print("  - CRITICAL ERROR: File was modified on disk!")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("Phase 2 Pipeline Completed Successfully (Dry-Run / In-Memory Only)")
    print("=" * 70)


if __name__ == "__main__":
    main()
