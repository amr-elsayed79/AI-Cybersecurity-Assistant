"""CLI script to index Master Knowledge Base chunks into local Qdrant vector store."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.rag.chunking import chunk_vulnerabilities
from app.rag.ingestion import compute_file_sha256, ingest_master_kb
from app.rag.retriever import QdrantVectorStore


def main():
    parser = argparse.ArgumentParser(description="Index Master KB chunks into local Qdrant vector store.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection before indexing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding and upserting (default: 32).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("AI Cybersecurity Assistant — Phase 3 Vector Indexing")
    print("=" * 70)

    kb_path = settings.MASTER_KB_PATH
    print(f"Master KB Path:      {kb_path}")
    print(f"Qdrant Storage Path: {settings.QDRANT_STORAGE_PATH}")
    print(f"Collection Name:     {settings.QDRANT_COLLECTION_NAME}")
    print(f"Embedding Model:     {settings.EMBEDDING_MODEL_NAME}")
    print(f"Embedding Dimension: {settings.EMBEDDING_DIMENSION}")

    # 1. Pre-execution hash check
    initial_hash = compute_file_sha256(kb_path)
    print(f"\n[Step 1] Pre-execution SHA-256: {initial_hash}")
    if initial_hash != settings.BASELINE_KB_SHA256:
        print(f"ERROR: Checksum mismatch against baseline ({settings.BASELINE_KB_SHA256})!")
        sys.exit(1)

    # 2. Ingest and chunk
    print("\n[Step 2] Ingesting and chunking Master KB...")
    records, _ = ingest_master_kb(kb_path)
    chunks, chunk_stats = chunk_vulnerabilities(records)
    print(f"  - Canonical records loaded: {len(records)}")
    print(f"  - Total chunks generated:   {chunk_stats['total_chunks']}")

    # 3. Vector store indexing
    print("\n[Step 3] Indexing chunks into local Qdrant...")
    store = QdrantVectorStore()
    try:
        initial_points = store.count_points()
        print(f"  - Points in collection before indexing: {initial_points}")

        index_stats = store.index_chunks(
            chunks=chunks,
            batch_size=args.batch_size,
            recreate=args.recreate,
            show_progress=True,
        )

        final_points = store.count_points()
        print(f"  - Successfully processed: {index_stats['indexed_chunks']} chunks")
        print(f"  - Points in collection after indexing:  {final_points}")
    finally:
        store.close()

    # 4. Post-execution hash verification
    final_hash = compute_file_sha256(kb_path)
    print("\n[Step 4] Post-execution verification:")
    print(f"  - Post-execution SHA-256: {final_hash}")
    if initial_hash == final_hash:
        print("  - INTEGRITY CONFIRMED: Master KB on disk remains 100% UNCHANGED byte-for-byte.")
    else:
        print("  - CRITICAL ERROR: File was modified on disk!")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("Phase 3 Vector Indexing Completed Successfully")
    print("=" * 70)


if __name__ == "__main__":
    main()
